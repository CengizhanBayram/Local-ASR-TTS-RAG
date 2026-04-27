/**
 * Voice AI RAG — Frontend
 * VAD silence detection, SSE streaming, inline citations, session memory
 */

// ============================================================
// Configuration
// ============================================================
const CONFIG = {
    API_BASE:  'http://localhost:8000/api',
    WS_BASE:   'ws://localhost:8000/api',
    SAMPLE_RATE: 16000,
    AUDIO_CHUNK_SIZE: 4096,
    RECONNECT_DELAY: 1000,
    MAX_RECONNECT_ATTEMPTS: 5,
    SESSION_KEY: 'voice_ai_session_id',
    // VAD — overridden by server on WS connect
    VAD_SILENCE_THRESHOLD:   0.008,
    VAD_SILENCE_DURATION_MS: 1500,
};

// ============================================================
// State
// ============================================================
const state = {
    chatMode:  'rag',
    ws: null, clientId: null, isConnected: false, reconnectAttempts: 0,
    sessionId: localStorage.getItem(CONFIG.SESSION_KEY) || null,
    turnCount: 0,
    isRecording: false,
    audioContext: null, mediaStream: null, processor: null,
    playbackContext: null, audioQueue: [], isPlaying: false, fullAudioBuffer: null,
    currentState: 'idle',
    documents: [],
    currentUserMsgId: null,
    currentAssistantMsgId: null,
    // SSE
    sseStreamId: null,
    // Timing
    queryStartTime: null,
};

// ============================================================
// DOM
// ============================================================
const elements = {
    navItems:    document.querySelectorAll('.nav-item'),
    views:       document.querySelectorAll('.view'),
    modeRag:     document.getElementById('mode-rag'),
    modeFree:    document.getElementById('mode-free'),
    modeIndicator: document.getElementById('mode-indicator'),
    messages:    document.getElementById('messages'),
    voiceBtn:    document.getElementById('voice-btn'),
    voiceStatus: document.getElementById('voice-status'),
    textInput:   document.getElementById('text-input'),
    sendBtn:     document.getElementById('send-btn'),
    clearChat:   document.getElementById('clear-chat'),
    uploadArea:  document.getElementById('upload-area'),
    fileInput:   document.getElementById('file-input'),
    uploadProgress: document.getElementById('upload-progress'),
    progressText:   document.getElementById('progress-text'),
    progressFill:   document.getElementById('progress-fill'),
    listContent:    document.getElementById('list-content'),
    clearAllBtn:    document.getElementById('clear-all-btn'),
    docCount:    document.getElementById('doc-count'),
    turnCount:   document.getElementById('turn-count'),
    apiStatus:   document.getElementById('api-status'),
    sessionInfo: document.getElementById('session-info'),
    toastContainer: document.getElementById('toast-container'),
};

// ============================================================
// Init
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    updateSessionDisplay();
    initNavigation();
    initModeToggle();
    initVoiceButton();
    initTextInput();
    initFileUpload();
    initDocumentActions();
    initChatActions();
    connectWebSocket();
    loadDocuments();
});

// ============================================================
// Session
// ============================================================
function saveSession(sessionId) {
    state.sessionId = sessionId;
    localStorage.setItem(CONFIG.SESSION_KEY, sessionId);
    updateSessionDisplay();
}
function updateSessionDisplay() {
    if (!elements.sessionInfo) return;
    if (state.sessionId) {
        const short = state.sessionId.slice(0, 8) + '…';
        elements.sessionInfo.textContent = `Session: ${short}`;
        elements.sessionInfo.title = `Session ID: ${state.sessionId}`;
    } else {
        elements.sessionInfo.textContent = 'New session';
    }
}
function updateTurnCount(count) {
    state.turnCount = count;
    if (elements.turnCount) elements.turnCount.textContent = `${count} Turns`;
}

// ============================================================
// Chat Actions
// ============================================================
function initChatActions() {
    elements.clearChat?.addEventListener('click', () => {
        if (confirm('Clear chat and reset session?')) clearChat();
    });
}
async function clearChat() {
    if (state.sessionId) {
        try { await fetch(`${CONFIG.API_BASE}/chat/sessions/${state.sessionId}`, { method: 'DELETE' }); }
        catch (_) {}
    }
    state.sessionId = null;
    localStorage.removeItem(CONFIG.SESSION_KEY);
    updateSessionDisplay();
    updateTurnCount(0);
    const welcome = elements.messages.querySelector('.message.system');
    elements.messages.innerHTML = '';
    if (welcome) elements.messages.appendChild(welcome);
    showToast('success', 'Cleared', 'Chat and session reset');
}

// ============================================================
// Mode Toggle
// ============================================================
function initModeToggle() {
    elements.modeRag?.addEventListener('click',  () => setMode('rag'));
    elements.modeFree?.addEventListener('click', () => setMode('free'));
}
function setMode(mode) {
    state.chatMode = mode;
    elements.modeRag?.classList.toggle('active',  mode === 'rag');
    elements.modeFree?.classList.toggle('active', mode === 'free');
    if (elements.modeIndicator) {
        if (mode === 'rag') {
            elements.modeIndicator.className = 'mode-badge';
            elements.modeIndicator.innerHTML = '<i class="fas fa-book-open"></i><span>RAG Mode</span>';
        } else {
            elements.modeIndicator.className = 'mode-badge free';
            elements.modeIndicator.innerHTML = '<i class="fas fa-comments"></i><span>Free Mode</span>';
        }
    }
    showToast('info', 'Mode Changed', mode === 'rag' ? 'RAG Mode (Document-based)' : 'Free Mode (General Chat)');
}

// ============================================================
// WebSocket
// ============================================================
function connectWebSocket() {
    state.clientId = generateId();
    const wsUrl = `${CONFIG.WS_BASE}/ws/realtime/${state.clientId}`;
    updateStatus('connecting', 'Connecting...');
    try {
        state.ws = new WebSocket(wsUrl);
        state.ws.onopen    = handleWsOpen;
        state.ws.onclose   = handleWsClose;
        state.ws.onerror   = handleWsError;
        state.ws.onmessage = handleWsMessage;
    } catch (err) {
        console.error('WS creation failed:', err);
        updateStatus('offline', 'Connection error');
    }
}
function handleWsOpen()  { state.isConnected = true; state.reconnectAttempts = 0; updateStatus('online', 'Connected ⚡'); }
function handleWsClose(e) {
    state.isConnected = false;
    updateStatus('offline', 'Disconnected');
    if (state.reconnectAttempts < CONFIG.MAX_RECONNECT_ATTEMPTS) {
        state.reconnectAttempts++;
        setTimeout(connectWebSocket, CONFIG.RECONNECT_DELAY * state.reconnectAttempts);
    }
}
function handleWsError(e) { console.error('WS error:', e); updateStatus('offline', 'Error'); }

function handleWsMessage(event) {
    try {
        const { type, data } = JSON.parse(event.data);
        switch (type) {
            case 'connected':
                // Apply server-side VAD config
                if (data.vad_silence_threshold)   CONFIG.VAD_SILENCE_THRESHOLD   = data.vad_silence_threshold;
                if (data.vad_silence_duration_ms) CONFIG.VAD_SILENCE_DURATION_MS = data.vad_silence_duration_ms;
                break;
            case 'session':      saveSession(data.session_id); break;
            case 'state':        handleStateChange(data.state); break;
            case 'listening_started': break;
            case 'transcription': handleTranscription(data.text, data.is_final); break;
            case 'user_message':  finalizeUserMessage(data.text); break;
            case 'answer_token':  handleAnswerToken(data.text); break;
            case 'answer':        handleAnswer(data.text, data.sources || [], data.total_ms); break;
            case 'audio_chunk':   handleAudioChunk(data.data); break;
            case 'audio_complete': handleAudioComplete(data.full_audio); break;
            case 'error':         handleError(data.message); break;
            case 'pong': break;
            case 'canceled':      handleCanceled(); break;
        }
    } catch (e) { console.error('WS parse error:', e); }
}

function sendWsMessage(type, data = {}) {
    if (state.ws?.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify({ type, mode: state.chatMode, session_id: state.sessionId, ...data }));
        return true;
    }
    return false;
}

// ============================================================
// State Handling
// ============================================================
function handleStateChange(newState) {
    state.currentState = newState;
    const cfg = {
        idle:       { text: 'Click mic to speak 🎙️', cls: '' },
        listening:  { text: 'Listening... click again to stop 🎤', cls: 'listening' },
        processing: { text: 'Thinking... 🧠', cls: 'processing' },
        speaking:   { text: 'Speaking... 🔊', cls: 'speaking' },
    };
    const c = cfg[newState] || cfg.idle;
    elements.voiceStatus.textContent = c.text;
    elements.voiceStatus.className = `voice-status ${c.cls}`;
    elements.voiceBtn.classList.toggle('recording', newState === 'listening');

    if (newState === 'idle' && state.currentUserMsgId) {
        const msgEl = document.getElementById(state.currentUserMsgId);
        if (msgEl?.querySelector('.text p')?.textContent.replace('|', '').trim() === '')
            removeMessage(state.currentUserMsgId);
        state.currentUserMsgId = null;
    }
}
function updateStatus(type, text) {
    elements.apiStatus.className = `stat-item status ${type}`;
    elements.apiStatus.querySelector('span').textContent = text;
}

// ============================================================
// VAD Voice Manager
// ============================================================
const VoiceManager = {
    isRecording: false,
    audioContext: null,
    mediaStream: null,
    processor: null,
    isToggling: false,
    _silenceStart: null,
    _hadSpeech: false,

    init() {
        elements.voiceBtn.addEventListener('click', (e) => {
            e.preventDefault(); e.stopPropagation();
            getPlaybackContext();
            this.handleToggle();
        });
    },
    handleToggle() {
        if (this.isToggling) return;
        this.isToggling = true;
        setTimeout(() => { this.isToggling = false; }, 300);
        this.isRecording ? this.stop() : this.start();
    },
    stop() {
        this.isRecording = false;
        this._silenceStart = null;
        this._hadSpeech = false;
        state.queryStartTime = Date.now();
        elements.voiceBtn.classList.remove('recording');
        updateStatus('processing', 'Processing...');
        handleStateChange('processing');
        if (this.processor)   { this.processor.disconnect(); this.processor = null; }
        if (this.mediaStream) { this.mediaStream.getTracks().forEach(t => t.stop()); this.mediaStream = null; }
        if (this.audioContext){ this.audioContext.close().catch(() => {}); this.audioContext = null; }
        sendWsMessage('stop');
        state.isRecording = false;
    },
    async start() {
        if (this.isRecording) { this.stop(); return; }
        if (state.currentState === 'processing' || state.currentState === 'speaking') {
            if (state.isPlaying) {
                if (state.playbackContext) { state.playbackContext.close(); state.playbackContext = null; }
                state.isPlaying = false;
                sendWsMessage('cancel');
            }
        }
        this.isRecording = true;
        this._silenceStart = null;
        this._hadSpeech = false;
        elements.voiceBtn.classList.add('recording');
        updateStatus('listening', 'Listening...');
        state.isRecording = true;
        if (state.currentUserMsgId) { removeMessage(state.currentUserMsgId); state.currentUserMsgId = null; }
        try {
            if (this.audioContext) { await this.audioContext.close().catch(() => {}); this.audioContext = null; }
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: { channelCount: 1, sampleRate: CONFIG.SAMPLE_RATE,
                         echoCancellation: true, noiseSuppression: true, autoGainControl: true }
            });
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: CONFIG.SAMPLE_RATE });
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            this.processor = this.audioContext.createScriptProcessor(CONFIG.AUDIO_CHUNK_SIZE, 1, 1);

            this.processor.onaudioprocess = (e) => {
                if (!this.isRecording) return;
                const pcm = e.inputBuffer.getChannelData(0);

                // ── VAD ──────────────────────────────────────────────────────
                const rms = computeRMS(pcm);
                if (rms > CONFIG.VAD_SILENCE_THRESHOLD) {
                    this._hadSpeech = true;
                    this._silenceStart = null;
                } else if (this._hadSpeech) {
                    if (!this._silenceStart) this._silenceStart = Date.now();
                    if (Date.now() - this._silenceStart >= CONFIG.VAD_SILENCE_DURATION_MS) {
                        this.stop();
                        return;
                    }
                }

                sendWsMessage('audio', { data: arrayBufferToBase64(float32ToInt16(pcm).buffer) });
            };

            source.connect(this.processor);
            this.processor.connect(this.audioContext.destination);
            sendWsMessage('start');
            handleStateChange('listening');
            showToast('success', 'Recording Started', 'Start speaking...');
        } catch (err) {
            console.error('VoiceManager error:', err);
            showToast('error', 'Microphone Error', 'Access denied or error occurred');
            this.stop();
            handleStateChange('idle');
        }
    },
};

function computeRMS(samples) {
    let sum = 0;
    for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
    return Math.sqrt(sum / samples.length);
}

function initVoiceButton() { VoiceManager.init(); }
function handleVoiceToggle() { VoiceManager.handleToggle(); }
function startRecording()    { VoiceManager.start(); }
function stopRecording()     { VoiceManager.stop(); }

// ============================================================
// Transcription Handling
// ============================================================
function handleTranscription(text, isFinal) {
    if (!state.currentUserMsgId) {
        if (!text && !isFinal) return;
        state.currentUserMsgId = addMessage('user', text || '', !isFinal);
    }
    const msgEl = document.getElementById(state.currentUserMsgId);
    if (!msgEl) return;
    const textEl = msgEl.querySelector('.text p');
    if (!textEl) return;
    if (isFinal) {
        textEl.textContent = capitalize(text);
        textEl.classList.remove('transcribing');
        state.currentUserMsgId = null;
    } else {
        textEl.innerHTML = `<span class="transcribing">${escapeHtml(text)}</span><span class="cursor">|</span>`;
    }
    scrollToBottom();
}
function finalizeUserMessage(text) {
    if (state.currentUserMsgId) {
        const el = document.getElementById(state.currentUserMsgId)?.querySelector('.text p');
        if (el) { el.textContent = text; el.classList.remove('transcribing'); }
    }
    state.currentUserMsgId = null;
    state.currentAssistantMsgId = addStreamingMessage();
}

// ============================================================
// WS Answer Streaming
// ============================================================
function handleAnswerToken(token) {
    if (!state.currentAssistantMsgId) {
        state.currentAssistantMsgId = addStreamingMessage();
    }
    appendStreamToken(state.currentAssistantMsgId, token);
}
function handleAnswer(text, sources, totalMs) {
    const responseTimeSec = totalMs != null
        ? totalMs / 1000
        : (state.queryStartTime ? (Date.now() - state.queryStartTime) / 1000 : null);
    state.queryStartTime = null;
    if (state.currentAssistantMsgId) {
        finalizeStreamingMessage(state.currentAssistantMsgId, text, sources, null, null, responseTimeSec);
    } else {
        state.currentAssistantMsgId = addAssistantMessage(text, sources, true, null, null, null, responseTimeSec);
    }
    state.audioQueue = [];
    state.fullAudioBuffer = null;
    updateTurnCount(state.turnCount + 1);
}

// ============================================================
// Audio
// ============================================================
let globalPlaybackContext = null;
function getPlaybackContext() {
    if (!globalPlaybackContext) globalPlaybackContext = new (window.AudioContext || window.webkitAudioContext)();
    if (globalPlaybackContext.state === 'suspended') globalPlaybackContext.resume().catch(() => {});
    return globalPlaybackContext;
}
function handleAudioChunk(base64Data) {
    const ctx = getPlaybackContext();
    state.playbackContext = ctx;
    if (!state.isPlaying || !state.nextAudioTime || state.nextAudioTime < ctx.currentTime) {
        state.nextAudioTime = ctx.currentTime;
        state.audioChain = Promise.resolve();
        state.isPlaying = true;
    }
    state.audioChain = state.audioChain.then(async () => {
        try {
            const buf = await ctx.decodeAudioData(base64ToArrayBuffer(base64Data));
            const src = ctx.createBufferSource();
            src.buffer = buf;
            src.connect(ctx.destination);
            const t = Math.max(ctx.currentTime, state.nextAudioTime);
            src.start(t);
            state.nextAudioTime = t + buf.duration;
        } catch (_) {}
    });
}
function handleAudioComplete(fullAudioBase64) {
    state.fullAudioBuffer = fullAudioBase64;
    const msgEl = document.getElementById(state.currentAssistantMsgId);
    if (msgEl) {
        const audioDiv = msgEl.querySelector('.message-audio');
        if (audioDiv) {
            audioDiv.classList.remove('streaming');
            audioDiv.dataset.audio = fullAudioBase64;
            const btn = audioDiv.querySelector('.play-btn');
            if (btn) {
                btn.classList.remove('playing');
                btn.innerHTML = '<i class="fas fa-play"></i>';
                btn.onclick = () => playStoredAudio(fullAudioBase64, btn);
            }
        }
    }
    state.currentAssistantMsgId = null;
}
function handleError(message) {
    console.error('Server error:', message);
    showToast('error', 'Error', message);
    if (state.isRecording) stopRecording();
    if (state.currentUserMsgId) {
        const el = document.getElementById(state.currentUserMsgId)?.querySelector('.text p');
        if (!el?.textContent.replace('|', '').trim()) removeMessage(state.currentUserMsgId);
        state.currentUserMsgId = null;
    }
    if (state.currentAssistantMsgId) { removeMessage(state.currentAssistantMsgId); state.currentAssistantMsgId = null; }
    handleStateChange('idle');
}
function handleCanceled() { handleStateChange('idle'); showToast('info', 'Cancelled', 'Operation cancelled'); }
function playStoredAudio(base64Audio, btn) {
    const wave = btn.nextElementSibling;
    const icon = btn.querySelector('i');
    if (state.isPlaying) {
        if (state.playbackContext) { state.playbackContext.close(); state.playbackContext = null; }
        state.isPlaying = false; icon.className = 'fas fa-play'; wave?.classList.remove('playing'); return;
    }
    const audio = new Audio(`data:audio/wav;base64,${base64Audio}`);
    audio.onplay  = () => { state.isPlaying = true;  icon.className = 'fas fa-pause'; wave?.classList.add('playing'); };
    audio.onended = () => { state.isPlaying = false; icon.className = 'fas fa-play';  wave?.classList.remove('playing'); };
    audio.onerror = () => { showToast('error', 'Audio Error', 'Playback failed'); icon.className = 'fas fa-play'; };
    audio.play();
}

// ============================================================
// Navigation
// ============================================================
function initNavigation() {
    elements.navItems.forEach(item => item.addEventListener('click', (e) => {
        e.preventDefault(); switchView(item.dataset.view);
    }));
}
function switchView(viewName) {
    elements.navItems.forEach(i => i.classList.toggle('active', i.dataset.view === viewName));
    elements.views.forEach(v => v.classList.toggle('active', v.id === `${viewName}-view`));
}

// ============================================================
// Text Input → SSE stream
// ============================================================
function initTextInput() {
    elements.sendBtn.addEventListener('click', sendTextMessage);
    elements.textInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendTextMessage(); });
}

async function sendTextMessage() {
    const text = elements.textInput.value.trim();
    if (!text) return;
    elements.textInput.value = '';
    state.queryStartTime = Date.now();

    addMessage('user', text);
    const streamMsgId = addStreamingMessage();
    state.sseStreamId = streamMsgId;

    const payload = { query: text, session_id: state.sessionId, include_audio: false, mode: state.chatMode };

    try {
        const response = await fetch(`${CONFIG.API_BASE}/chat/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Query failed');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';
        let sources = [];
        let rewrittenQuery = null;
        let metrics = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });

            const lines = buf.split('\n');
            buf = lines.pop(); // keep incomplete line

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const json = line.slice(6).trim();
                if (!json) continue;
                let evt;
                try { evt = JSON.parse(json); } catch (_) { continue; }

                if (evt.type === 'token') {
                    appendStreamToken(streamMsgId, evt.text);
                } else if (evt.type === 'sources') {
                    sources = evt.sources || [];
                } else if (evt.type === 'rewrite') {
                    rewrittenQuery = evt.query;
                } else if (evt.type === 'done') {
                    if (evt.session_id) saveSession(evt.session_id);
                    if (evt.turn)       updateTurnCount(evt.turn);
                    metrics = evt.metrics;
                    state.queryStartTime = null;
                    // Finalize the streaming message with full content
                    const msgEl = document.getElementById(streamMsgId);
                    const fullText = msgEl?.querySelector('.stream-text')?.textContent || '';
                    finalizeStreamingMessage(streamMsgId, fullText, sources, metrics, rewrittenQuery);
                    state.sseStreamId = null;

                    // Request TTS separately if needed
                    if (fullText) {
                        _requestTtsForMessage(streamMsgId, fullText);
                    }
                } else if (evt.type === 'error') {
                    throw new Error(evt.message);
                }
            }
        }
    } catch (err) {
        removeMessage(streamMsgId);
        state.sseStreamId = null;
        showToast('error', 'Error', err.message);
        addMessage('assistant', 'An error occurred. Please try again.');
    }
}

async function _requestTtsForMessage(msgId, text) {
    try {
        const response = await fetch(`${CONFIG.API_BASE}/chat/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: text.slice(0, 100),  // just to get TTS for the answer
                session_id: state.sessionId,
                include_audio: true,
                mode: 'free',  // no RAG — we already have the answer
            }),
        });
        // Note: This is a workaround. In production, add a dedicated /tts endpoint.
        // For now we skip this to avoid double LLM call.
    } catch (_) {}
}

// ============================================================
// Message Rendering
// ============================================================
function addMessage(type, content, isTranscribing = false) {
    const id = `msg-${Date.now()}-${Math.random().toString(36).slice(2,6)}`;
    const div = document.createElement('div');
    div.id = id; div.className = `message ${type}`;
    div.innerHTML = `
        <div class="message-content">
            <div class="text">
                <p class="${isTranscribing ? 'transcribing' : ''}">${
                    escapeHtml(content) || (isTranscribing ? '<span class="cursor">|</span>' : '')
                }</p>
            </div>
        </div>`;
    elements.messages.appendChild(div);
    scrollToBottom();
    return id;
}

function addStreamingMessage() {
    const id = `msg-${Date.now()}-${Math.random().toString(36).slice(2,6)}`;
    const div = document.createElement('div');
    div.id = id; div.className = 'message assistant streaming';
    div.innerHTML = `
        <div class="message-content">
            <div class="text">
                <p class="stream-text"></p>
                <span class="stream-cursor">▍</span>
            </div>
        </div>`;
    elements.messages.appendChild(div);
    scrollToBottom();
    return id;
}

function appendStreamToken(msgId, token) {
    const msgEl = document.getElementById(msgId);
    if (!msgEl) return;
    const textEl = msgEl.querySelector('.stream-text');
    if (textEl) { textEl.textContent += token; scrollToBottom(); }
}

function finalizeStreamingMessage(msgId, text, sources = [], metrics = null, rewrittenQuery = null, responseTimeSec = null) {
    const msgEl = document.getElementById(msgId);
    if (!msgEl) return;
    msgEl.classList.remove('streaming');
    const textContainer = msgEl.querySelector('.text');
    if (!textContainer) return;
    const timeSec = responseTimeSec ?? (metrics?.total_ms != null ? metrics.total_ms / 1000 : null);
    textContainer.innerHTML = buildAssistantHTML(text, sources, metrics, rewrittenQuery, false, null, timeSec);
    scrollToBottom();
}

function addAssistantMessage(text, sources = [], isStreaming = false, audioBase64 = null, metrics = null, rewrittenQuery = null, responseTimeSec = null) {
    const id = `msg-${Date.now()}-${Math.random().toString(36).slice(2,6)}`;
    const div = document.createElement('div');
    div.id = id; div.className = 'message assistant';
    div.innerHTML = `<div class="message-content"><div class="text">${
        buildAssistantHTML(text, sources, metrics, rewrittenQuery, isStreaming, audioBase64, responseTimeSec)
    }</div></div>`;
    elements.messages.appendChild(div);
    scrollToBottom();

    if (audioBase64 && !isStreaming) {
        setTimeout(() => div.querySelector('.play-btn')?.click(), 300);
    }
    return id;
}

function buildAssistantHTML(text, sources, metrics, rewrittenQuery, isStreaming, audioBase64, responseTimeSec = null) {
    // Response time badge
    let timeBadge = '';
    if (responseTimeSec != null) {
        const label = responseTimeSec < 10 ? responseTimeSec.toFixed(1) + 's' : Math.round(responseTimeSec) + 's';
        timeBadge = `<div class="response-time-badge">⚡ ${label}</div>`;
    }

    // Sources
    let sourcesHtml = '';
    if (sources?.length > 0) {
        const items = sources.map(s => {
            const pct = Math.round((s.score ?? 0) * 100);
            const cls = pct >= 70 ? 'high' : pct >= 45 ? 'mid' : 'low';
            return `<span class="source-item">
                <i class="fas fa-file-lines"></i>${escapeHtml(s.filename)}
                <span class="score-badge ${cls}">${pct}%</span>
            </span>`;
        }).join('');
        sourcesHtml = `<div class="sources"><div class="sources-title">📚 Sources</div>${items}</div>`;
    }

    // Rewrite badge
    let rewriteHtml = rewrittenQuery
        ? `<div class="rewrite-badge"><i class="fas fa-wand-magic-sparkles"></i> Query rewritten: "<em>${escapeHtml(rewrittenQuery)}</em>"</div>`
        : '';

    // Metrics
    let metricsHtml = '';
    if (metrics) {
        const parts = [];
        if (metrics.stt_ms      != null) parts.push(`<span class="metric-item"><i class="fas fa-microphone"></i>STT <span class="val">${metrics.stt_ms.toFixed(0)}ms</span></span>`);
        if (metrics.rewrite_ms  != null) parts.push(`<span class="metric-item"><i class="fas fa-pen"></i>Rewrite <span class="val">${metrics.rewrite_ms.toFixed(0)}ms</span></span>`);
        if (metrics.retrieval_ms!= null) parts.push(`<span class="metric-item"><i class="fas fa-magnifying-glass"></i>RAG <span class="val">${metrics.retrieval_ms.toFixed(0)}ms</span></span>`);
        if (metrics.llm_ms      != null) parts.push(`<span class="metric-item"><i class="fas fa-brain"></i>LLM <span class="val">${metrics.llm_ms.toFixed(0)}ms</span></span>`);
        if (metrics.tts_ms      != null) parts.push(`<span class="metric-item"><i class="fas fa-volume-high"></i>TTS <span class="val">${metrics.tts_ms.toFixed(0)}ms</span></span>`);
        if (metrics.docs_after_threshold != null) parts.push(`<span class="metric-item"><i class="fas fa-filter"></i>Docs <span class="val">${metrics.docs_after_threshold}/${metrics.docs_retrieved ?? '?'}</span></span>`);
        parts.push(`<span class="metric-item"><i class="fas fa-clock"></i>Total <span class="val">${(metrics.total_ms || 0).toFixed(0)}ms</span></span>`);
        metricsHtml = `<div class="metrics-panel">${parts.join('')}</div>`;
    }

    // Audio
    let audioHtml = '';
    if (isStreaming) {
        audioHtml = `<div class="message-audio streaming">
            <button class="play-btn"><i class="fas fa-volume-up"></i></button>
            <div class="audio-wave playing"><span></span><span></span><span></span><span></span><span></span></div>
        </div>`;
    } else if (audioBase64) {
        audioHtml = `<div class="message-audio" data-audio="${audioBase64}">
            <button class="play-btn" onclick="playStoredAudio('${audioBase64}', this)"><i class="fas fa-play"></i></button>
            <div class="audio-wave"><span></span><span></span><span></span><span></span><span></span></div>
        </div>`;
    }

    return `${timeBadge}<p>${formatText(text)}</p>${audioHtml}${rewriteHtml}${sourcesHtml}${metricsHtml}`;
}

function addLoadingMessage() {
    const id = `loading-${Date.now()}`;
    const div = document.createElement('div');
    div.id = id; div.className = 'message assistant loading';
    div.innerHTML = `<div class="message-content"><div class="loading-dots"><span></span><span></span><span></span></div></div>`;
    elements.messages.appendChild(div);
    scrollToBottom();
    return id;
}

function removeMessage(id) { document.getElementById(id)?.remove(); }
function scrollToBottom()  { elements.messages.scrollTop = elements.messages.scrollHeight; }

// ── Text formatting with inline citations ────────────────────────────────────
function formatText(text) {
    if (!text) return '';
    return escapeHtml(text)
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\[(\d+)\]/g, '<sup class="citation-ref" title="Source $1">[$1]</sup>');
}
function escapeHtml(text) {
    if (!text) return '';
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}
function capitalize(text) { return text ? text.charAt(0).toUpperCase() + text.slice(1) : ''; }

// ============================================================
// File Upload
// ============================================================
function initFileUpload() {
    elements.uploadArea.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', (e) => uploadFiles(Array.from(e.target.files)));
    elements.uploadArea.addEventListener('dragover',  (e) => { e.preventDefault(); elements.uploadArea.classList.add('dragover'); });
    elements.uploadArea.addEventListener('dragleave', ()  => elements.uploadArea.classList.remove('dragover'));
    elements.uploadArea.addEventListener('drop',      (e) => { e.preventDefault(); elements.uploadArea.classList.remove('dragover'); uploadFiles(Array.from(e.dataTransfer.files)); });
}
async function uploadFiles(files) { for (const f of files) await uploadFile(f); loadDocuments(); }
async function uploadFile(file) {
    if (!['pdf','docx','txt','md'].includes(file.name.split('.').pop().toLowerCase())) {
        showToast('error', 'Unsupported Format', `${file.name.split('.').pop()} not supported`); return;
    }
    elements.uploadProgress.classList.remove('hidden');
    elements.progressText.textContent = `Uploading: ${file.name}`;
    elements.progressFill.style.width = '0%';
    try {
        const formData = new FormData();
        formData.append('file', file);
        let progress = 0;
        const interval = setInterval(() => { progress = Math.min(progress + 15, 90); elements.progressFill.style.width = `${progress}%`; }, 100);
        const response = await fetch(`${CONFIG.API_BASE}/documents/upload`, { method: 'POST', body: formData });
        clearInterval(interval);
        elements.progressFill.style.width = '100%';
        if (!response.ok) throw new Error('Upload failed');
        const data = await response.json();
        showToast('success', 'Success', `${file.name} uploaded (${data.chunk_count} chunks)`);
    } catch (err) {
        showToast('error', 'Error', err.message);
    } finally {
        setTimeout(() => elements.uploadProgress.classList.add('hidden'), 800);
    }
}

// ============================================================
// Document Management
// ============================================================
async function loadDocuments() {
    try {
        const data = await (await fetch(`${CONFIG.API_BASE}/documents`)).json();
        state.documents = data.documents;
        renderDocuments();
        elements.docCount.textContent = `${state.documents.length} Docs`;
    } catch (_) {}
}
function renderDocuments() {
    if (!state.documents.length) {
        elements.listContent.innerHTML = `<div class="empty-state"><i class="fas fa-folder-open"></i><p>No documents uploaded yet</p></div>`;
        return;
    }
    const iconMap = { pdf: 'file-pdf', docx: 'file-word', txt: 'file-lines', md: 'file-code' };
    elements.listContent.innerHTML = state.documents.map(doc => `
        <div class="document-item">
            <div class="doc-icon ${doc.file_type}"><i class="fas fa-${iconMap[doc.file_type] || 'file'}"></i></div>
            <div class="doc-info">
                <div class="doc-name">${escapeHtml(doc.filename)}</div>
                <div class="doc-meta">
                    <span><i class="fas fa-weight-hanging"></i> ${formatSize(doc.file_size)}</span>
                    <span><i class="fas fa-puzzle-piece"></i> ${doc.chunk_count} chunks</span>
                </div>
            </div>
            <button class="doc-delete" onclick="deleteDocument('${doc.id}')"><i class="fas fa-trash-can"></i></button>
        </div>
    `).join('');
}
function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024*1024) return (bytes/1024).toFixed(1) + ' KB';
    return (bytes/(1024*1024)).toFixed(1) + ' MB';
}
function initDocumentActions() {
    elements.clearAllBtn.addEventListener('click', async () => {
        if (!confirm('Delete all documents?')) return;
        await fetch(`${CONFIG.API_BASE}/documents`, { method: 'DELETE' });
        loadDocuments();
        showToast('success', 'Done', 'All documents deleted');
    });
}
async function deleteDocument(id) {
    if (!confirm('Delete this document?')) return;
    await fetch(`${CONFIG.API_BASE}/documents/${id}`, { method: 'DELETE' });
    loadDocuments();
    showToast('success', 'Done', 'Document deleted');
}

// ============================================================
// Utilities
// ============================================================
function generateId()  { return 'client_' + Math.random().toString(36).substring(2, 12); }
function float32ToInt16(f32) {
    const i16 = new Int16Array(f32.length);
    for (let i = 0; i < f32.length; i++) { const s = Math.max(-1, Math.min(1, f32[i])); i16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF; }
    return i16;
}
function arrayBufferToBase64(buf) {
    const bytes = new Uint8Array(buf); let b = '';
    for (let i = 0; i < bytes.byteLength; i++) b += String.fromCharCode(bytes[i]);
    return btoa(b);
}
function base64ToArrayBuffer(b64) {
    const bin = atob(b64); const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes.buffer;
}
function showToast(type, title, message) {
    const icons = { success: 'fa-circle-check', error: 'fa-circle-xmark', warning: 'fa-triangle-exclamation', info: 'fa-circle-info' };
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<i class="fas ${icons[type]}"></i>
        <div class="toast-content"><div class="toast-title">${escapeHtml(title)}</div><div class="toast-message">${escapeHtml(message)}</div></div>
        <button class="toast-close" onclick="this.parentElement.remove()"><i class="fas fa-xmark"></i></button>`;
    elements.toastContainer.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '0'; toast.style.transform = 'translateX(100%)'; setTimeout(() => toast.remove(), 300); }, 4000);
}

// Heartbeat
setInterval(() => { if (state.isConnected) sendWsMessage('ping'); }, 30000);

// Particles
function createParticles() {
    const container = document.getElementById('particles');
    if (!container) return;
    const colors = ['#8b5cf6','#06b6d4','#10b981','#f472b6'];
    for (let i = 0; i < 30; i++) {
        const p = document.createElement('div');
        p.className = 'particle';
        p.style.left = `${Math.random()*100}%`;
        p.style.background = colors[Math.floor(Math.random()*colors.length)];
        p.style.animationDuration  = `${15+Math.random()*20}s`;
        p.style.animationDelay    = `${Math.random()*20}s`;
        p.style.width = p.style.height = `${2+Math.random()*4}px`;
        container.appendChild(p);
    }
}
createParticles();
document.addEventListener('mousemove', (e) => {
    document.body.style.setProperty('--mouse-x', e.clientX / window.innerWidth);
    document.body.style.setProperty('--mouse-y', e.clientY / window.innerHeight);
});

window.playStoredAudio = playStoredAudio;
window.deleteDocument  = deleteDocument;
