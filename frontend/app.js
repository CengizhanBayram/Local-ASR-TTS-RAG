/**
 * Voice AI RAG - Frontend
 * Conversation memory, score badge, pipeline metrics destekli
 */

// ============================================
// Configuration
// ============================================
const CONFIG = {
    API_BASE: 'http://localhost:8000/api',
    WS_BASE: 'ws://localhost:8000/api',
    SAMPLE_RATE: 16000,
    AUDIO_CHUNK_SIZE: 4096,
    RECONNECT_DELAY: 1000,
    MAX_RECONNECT_ATTEMPTS: 5,
    SESSION_KEY: 'voice_ai_session_id'
};

// ============================================
// Application State
// ============================================
const state = {
    chatMode: 'rag',

    // WebSocket
    ws: null,
    clientId: null,
    isConnected: false,
    reconnectAttempts: 0,

    // Session / Conversation
    sessionId: localStorage.getItem(CONFIG.SESSION_KEY) || null,
    turnCount: 0,

    // Audio Recording
    isRecording: false,
    audioContext: null,
    mediaStream: null,
    processor: null,

    // Audio Playback
    playbackContext: null,
    audioQueue: [],
    isPlaying: false,
    fullAudioBuffer: null,

    // UI State
    currentState: 'idle',
    documents: [],

    currentUserMsgId: null,
    currentAssistantMsgId: null
};

// ============================================
// DOM Elements
// ============================================
const elements = {
    navItems: document.querySelectorAll('.nav-item'),
    views: document.querySelectorAll('.view'),

    modeRag: document.getElementById('mode-rag'),
    modeFree: document.getElementById('mode-free'),
    modeIndicator: document.getElementById('mode-indicator'),

    messages: document.getElementById('messages'),
    voiceBtn: document.getElementById('voice-btn'),
    voiceStatus: document.getElementById('voice-status'),
    textInput: document.getElementById('text-input'),
    sendBtn: document.getElementById('send-btn'),
    clearChat: document.getElementById('clear-chat'),

    uploadArea: document.getElementById('upload-area'),
    fileInput: document.getElementById('file-input'),
    uploadProgress: document.getElementById('upload-progress'),
    progressText: document.getElementById('progress-text'),
    progressFill: document.getElementById('progress-fill'),
    listContent: document.getElementById('list-content'),
    clearAllBtn: document.getElementById('clear-all-btn'),

    docCount: document.getElementById('doc-count'),
    turnCount: document.getElementById('turn-count'),
    apiStatus: document.getElementById('api-status'),
    sessionInfo: document.getElementById('session-info'),

    toastContainer: document.getElementById('toast-container')
};

// ============================================
// Initialization
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    console.log('Voice AI starting...');
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

// ============================================
// Session Management
// ============================================
function saveSession(sessionId) {
    state.sessionId = sessionId;
    localStorage.setItem(CONFIG.SESSION_KEY, sessionId);
    updateSessionDisplay();
}

function updateSessionDisplay() {
    if (elements.sessionInfo) {
        if (state.sessionId) {
            const short = state.sessionId.slice(0, 8) + '…';
            elements.sessionInfo.textContent = `Session: ${short}`;
            elements.sessionInfo.title = `Session ID: ${state.sessionId}`;
        } else {
            elements.sessionInfo.textContent = 'Yeni oturum';
        }
    }
}

function updateTurnCount(count) {
    state.turnCount = count;
    if (elements.turnCount) {
        elements.turnCount.textContent = `${count} Tur`;
    }
}

// ============================================
// Chat Actions
// ============================================
function initChatActions() {
    if (elements.clearChat) {
        elements.clearChat.addEventListener('click', () => {
            if (confirm('Sohbeti temizlemek istiyor musunuz? Oturum geçmişi de sıfırlanacak.')) {
                clearChat();
            }
        });
    }
}

async function clearChat() {
    // Session'ı sil
    if (state.sessionId) {
        try {
            await fetch(`${CONFIG.API_BASE}/chat/sessions/${state.sessionId}`, { method: 'DELETE' });
        } catch (e) { /* ignore */ }
    }
    state.sessionId = null;
    localStorage.removeItem(CONFIG.SESSION_KEY);
    updateSessionDisplay();
    updateTurnCount(0);

    const welcomeMsg = elements.messages.querySelector('.message.system');
    elements.messages.innerHTML = '';
    if (welcomeMsg) elements.messages.appendChild(welcomeMsg);
    showToast('success', 'Temizlendi', 'Sohbet ve oturum sıfırlandı');
}

// ============================================
// Mode Toggle
// ============================================
function initModeToggle() {
    if (elements.modeRag) elements.modeRag.addEventListener('click', () => setMode('rag'));
    if (elements.modeFree) elements.modeFree.addEventListener('click', () => setMode('free'));
}

function setMode(mode) {
    state.chatMode = mode;
    if (elements.modeRag) elements.modeRag.classList.toggle('active', mode === 'rag');
    if (elements.modeFree) elements.modeFree.classList.toggle('active', mode === 'free');

    if (elements.modeIndicator) {
        if (mode === 'rag') {
            elements.modeIndicator.className = 'mode-badge';
            elements.modeIndicator.innerHTML = '<i class="fas fa-book-open"></i><span>RAG Modu</span>';
        } else {
            elements.modeIndicator.className = 'mode-badge free';
            elements.modeIndicator.innerHTML = '<i class="fas fa-comments"></i><span>Serbest Mod</span>';
        }
    }

    const modeText = mode === 'rag' ? 'RAG Modu (Belge Tabanlı)' : 'Serbest Mod (Genel Sohbet)';
    showToast('info', 'Mod Değiştirildi', modeText);
}

// ============================================
// WebSocket Connection
// ============================================
function connectWebSocket() {
    state.clientId = generateId();
    const wsUrl = `${CONFIG.WS_BASE}/ws/realtime/${state.clientId}`;

    console.log('Connecting to:', wsUrl);
    updateStatus('connecting', 'Bağlanıyor...');

    try {
        state.ws = new WebSocket(wsUrl);
        state.ws.onopen = handleWsOpen;
        state.ws.onclose = handleWsClose;
        state.ws.onerror = handleWsError;
        state.ws.onmessage = handleWsMessage;
    } catch (error) {
        console.error('WebSocket creation failed:', error);
        updateStatus('offline', 'Bağlantı hatası');
    }
}

function handleWsOpen() {
    console.log('WebSocket connected');
    state.isConnected = true;
    state.reconnectAttempts = 0;
    updateStatus('online', 'Bağlı ⚡');
}

function handleWsClose(event) {
    console.log('WebSocket closed:', event.code);
    state.isConnected = false;
    updateStatus('offline', 'Bağlantı kesildi');

    if (state.reconnectAttempts < CONFIG.MAX_RECONNECT_ATTEMPTS) {
        state.reconnectAttempts++;
        const delay = CONFIG.RECONNECT_DELAY * state.reconnectAttempts;
        setTimeout(connectWebSocket, delay);
    }
}

function handleWsError(error) {
    console.error('WebSocket error:', error);
    updateStatus('offline', 'Hata');
}

function handleWsMessage(event) {
    try {
        const message = JSON.parse(event.data);
        const { type, data } = message;

        switch (type) {
            case 'connected':
                console.log('Connected as:', data.client_id);
                break;
            case 'session':
                // WS pipeline'dan session ID gel
                saveSession(data.session_id);
                break;
            case 'state':
                handleStateChange(data.state);
                break;
            case 'listening_started':
                break;
            case 'transcription':
                handleTranscription(data.text, data.is_final);
                break;
            case 'user_message':
                finalizeUserMessage(data.text);
                break;
            case 'answer':
                handleAnswer(data.text, data.sources || []);
                break;
            case 'audio_chunk':
                handleAudioChunk(data.data);
                break;
            case 'audio_complete':
                handleAudioComplete(data.full_audio);
                break;
            case 'error':
                handleError(data.message);
                break;
            case 'pong':
                break;
            case 'canceled':
                handleCanceled();
                break;
        }
    } catch (error) {
        console.error('Message parse error:', error);
    }
}

function sendWsMessage(type, data = {}) {
    if (state.ws?.readyState === WebSocket.OPEN) {
        const message = {
            type,
            mode: state.chatMode,
            session_id: state.sessionId,
            ...data
        };
        state.ws.send(JSON.stringify(message));
        return true;
    }
    console.warn('WebSocket not connected');
    return false;
}

// ============================================
// State Handling
// ============================================
function handleStateChange(newState) {
    state.currentState = newState;

    const statusConfig = {
        'idle': { text: 'Konuşmak için mikrofona tıklayın 🎙️', class: '' },
        'listening': { text: 'Dinliyorum... Durdurmak için tekrar tıklayın 🎤', class: 'listening' },
        'processing': { text: 'Düşünüyorum... 🧠', class: 'processing' },
        'speaking': { text: 'Konuşuyorum... 🔊', class: 'speaking' }
    };

    const config = statusConfig[newState] || statusConfig.idle;
    elements.voiceStatus.textContent = config.text;
    elements.voiceStatus.className = `voice-status ${config.class}`;
    elements.voiceBtn.classList.toggle('recording', newState === 'listening');

    if (newState === 'idle' && state.currentUserMsgId) {
        const msgEl = document.getElementById(state.currentUserMsgId);
        if (msgEl) {
            const textEl = msgEl.querySelector('.text p');
            if (textEl && !textEl.textContent.replace('|', '').trim()) {
                removeMessage(state.currentUserMsgId);
                state.currentUserMsgId = null;
            }
        }
    }
}

function updateStatus(type, text) {
    elements.apiStatus.className = `stat-item status ${type}`;
    elements.apiStatus.querySelector('span').textContent = text;
}

// ============================================
// Voice Recording
// ============================================
const VoiceManager = {
    isRecording: false,
    audioContext: null,
    mediaStream: null,
    processor: null,
    isToggling: false,

    init() {
        elements.voiceBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            getPlaybackContext();
            this.handleToggle();
        });
    },

    handleToggle() {
        if (this.isToggling) return;
        this.isToggling = true;
        setTimeout(() => { this.isToggling = false; }, 300);

        if (this.isRecording) {
            this.stop();
        } else {
            this.start();
        }
    },

    stop() {
        this.isRecording = false;
        elements.voiceBtn.classList.remove('recording');
        updateStatus('processing', 'İşleniyor...');
        handleStateChange('processing');

        if (this.processor) { this.processor.disconnect(); this.processor = null; }
        if (this.mediaStream) { this.mediaStream.getTracks().forEach(t => t.stop()); this.mediaStream = null; }
        if (this.audioContext) { this.audioContext.close().catch(() => {}); this.audioContext = null; }

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
        elements.voiceBtn.classList.add('recording');
        updateStatus('listening', 'Dinliyorum...');
        state.isRecording = true;

        if (state.currentUserMsgId) { removeMessage(state.currentUserMsgId); state.currentUserMsgId = null; }

        try {
            if (this.audioContext) { await this.audioContext.close().catch(() => {}); this.audioContext = null; }

            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: { channelCount: 1, sampleRate: CONFIG.SAMPLE_RATE,
                         echoCancellation: true, noiseSuppression: true, autoGainControl: true }
            });

            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: CONFIG.SAMPLE_RATE
            });

            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            this.processor = this.audioContext.createScriptProcessor(CONFIG.AUDIO_CHUNK_SIZE, 1, 1);

            this.processor.onaudioprocess = (e) => {
                if (!this.isRecording) return;
                const pcmData = float32ToInt16(e.inputBuffer.getChannelData(0));
                sendWsMessage('audio', { data: arrayBufferToBase64(pcmData.buffer) });
            };

            source.connect(this.processor);
            this.processor.connect(this.audioContext.destination);

            sendWsMessage('start');
            handleStateChange('listening');
            showToast('success', 'Kayıt Başladı', 'Konuşmaya başlayın...');

        } catch (error) {
            console.error('VoiceManager Error:', error);
            showToast('error', 'Mikrofon Hatası', 'Erişim reddedildi veya hata oluştu');
            this.stop();
            handleStateChange('idle');
        }
    }
};

function initVoiceButton() { VoiceManager.init(); }
function handleVoiceToggle() { VoiceManager.handleToggle(); }
function startRecording() { VoiceManager.start(); }
function stopRecording() { VoiceManager.stop(); }

// ============================================
// Transcription Handling
// ============================================
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
        const msgEl = document.getElementById(state.currentUserMsgId);
        if (msgEl) {
            const textEl = msgEl.querySelector('.text p');
            if (textEl) { textEl.textContent = text; textEl.classList.remove('transcribing'); }
        }
    }
    state.currentUserMsgId = null;
    state.currentAssistantMsgId = addLoadingMessage();
}

// ============================================
// Answer Handling
// ============================================
function handleAnswer(text, sources) {
    if (state.currentAssistantMsgId) removeMessage(state.currentAssistantMsgId);

    const showSources = state.chatMode === 'rag' && sources.length > 0;
    state.currentAssistantMsgId = addAssistantMessage(text, showSources ? sources : [], true);

    state.audioQueue = [];
    state.fullAudioBuffer = null;

    // Konuşma turunu güncelle
    updateTurnCount(state.turnCount + 1);
}

let globalPlaybackContext = null;

function getPlaybackContext() {
    if (!globalPlaybackContext) {
        globalPlaybackContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (globalPlaybackContext.state === 'suspended') {
        globalPlaybackContext.resume().catch(() => {});
    }
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
            const arrayBuffer = base64ToArrayBuffer(base64Data);
            const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
            const source = ctx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(ctx.destination);
            const startTime = Math.max(ctx.currentTime, state.nextAudioTime);
            source.start(startTime);
            state.nextAudioTime = startTime + audioBuffer.duration;
        } catch (error) {
            console.error('Audio chunk error:', error);
        }
    });
}

function handleAudioComplete(fullAudioBase64) {
    state.fullAudioBuffer = fullAudioBase64;

    if (state.currentAssistantMsgId) {
        const msgEl = document.getElementById(state.currentAssistantMsgId);
        if (msgEl) {
            const audioDiv = msgEl.querySelector('.message-audio');
            if (audioDiv) {
                audioDiv.classList.remove('streaming');
                audioDiv.dataset.audio = fullAudioBase64;
                const playBtn = audioDiv.querySelector('.play-btn');
                if (playBtn) {
                    playBtn.classList.remove('playing');
                    playBtn.innerHTML = '<i class="fas fa-play"></i>';
                    playBtn.onclick = () => playStoredAudio(fullAudioBase64, playBtn);
                }
            }
        }
    }
    state.currentAssistantMsgId = null;
}

function handleError(message) {
    console.error('Server error:', message);
    showToast('error', 'Hata', message);

    if (state.isRecording) stopRecording();

    if (state.currentUserMsgId) {
        const msgEl = document.getElementById(state.currentUserMsgId);
        if (msgEl) {
            const textEl = msgEl.querySelector('.text p');
            if (textEl && !textEl.textContent.replace('|', '').trim()) {
                removeMessage(state.currentUserMsgId);
            }
        }
        state.currentUserMsgId = null;
    }

    if (state.currentAssistantMsgId) {
        removeMessage(state.currentAssistantMsgId);
        state.currentAssistantMsgId = null;
    }
    handleStateChange('idle');
}

function handleCanceled() {
    handleStateChange('idle');
    showToast('info', 'İptal Edildi', 'İşlem iptal edildi');
}

function playStoredAudio(base64Audio, btn) {
    const wave = btn.nextElementSibling;
    const icon = btn.querySelector('i');

    if (state.isPlaying) {
        if (state.playbackContext) { state.playbackContext.close(); state.playbackContext = null; }
        state.isPlaying = false;
        icon.className = 'fas fa-play';
        wave?.classList.remove('playing');
        return;
    }

    const audio = new Audio(`data:audio/wav;base64,${base64Audio}`);
    audio.onplay = () => { state.isPlaying = true; icon.className = 'fas fa-pause'; wave?.classList.add('playing'); };
    audio.onended = () => { state.isPlaying = false; icon.className = 'fas fa-play'; wave?.classList.remove('playing'); };
    audio.onerror = () => { showToast('error', 'Ses Hatası', 'Ses oynatılamadı'); icon.className = 'fas fa-play'; };
    audio.play();
}

// ============================================
// Navigation
// ============================================
function initNavigation() {
    elements.navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            switchView(item.dataset.view);
        });
    });
}

function switchView(viewName) {
    elements.navItems.forEach(item => item.classList.toggle('active', item.dataset.view === viewName));
    elements.views.forEach(view => view.classList.toggle('active', view.id === `${viewName}-view`));
}

// ============================================
// Text Input → /chat/query (tüm modlar)
// ============================================
function initTextInput() {
    elements.sendBtn.addEventListener('click', sendTextMessage);
    elements.textInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendTextMessage();
    });
}

async function sendTextMessage() {
    const text = elements.textInput.value.trim();
    if (!text) return;

    elements.textInput.value = '';
    addMessage('user', text);
    const loadingId = addLoadingMessage();

    try {
        const response = await fetch(`${CONFIG.API_BASE}/chat/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: text,
                session_id: state.sessionId,
                include_audio: true,
                mode: state.chatMode
            })
        });

        removeMessage(loadingId);

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Sorgu başarısız');
        }

        const data = await response.json();

        // Session kaydet
        if (data.session_id) saveSession(data.session_id);
        if (data.conversation_turn) updateTurnCount(data.conversation_turn);

        const showSources = state.chatMode === 'rag' && data.sources?.length > 0;
        addAssistantMessage(
            data.answer,
            showSources ? data.sources : [],
            false,
            data.audio_base64,
            data.metrics,
            data.rewritten_query
        );

    } catch (error) {
        removeMessage(loadingId);
        showToast('error', 'Hata', error.message);
        addMessage('assistant', 'Bir hata oluştu. Lütfen tekrar deneyin.');
    }
}

// ============================================
// Message Functions
// ============================================
function addMessage(type, content, isTranscribing = false) {
    const id = `msg-${Date.now()}`;
    const div = document.createElement('div');
    div.id = id;
    div.className = `message ${type}`;

    div.innerHTML = `
        <div class="message-content">
            <div class="text">
                <p class="${isTranscribing ? 'transcribing' : ''}">${
                    escapeHtml(content) || (isTranscribing ? '<span class="cursor">|</span>' : '')
                }</p>
            </div>
        </div>
    `;

    elements.messages.appendChild(div);
    scrollToBottom();
    return id;
}

function addAssistantMessage(
    text,
    sources = [],
    isStreaming = false,
    audioBase64 = null,
    metrics = null,
    rewrittenQuery = null
) {
    const id = `msg-${Date.now()}`;
    const div = document.createElement('div');
    div.id = id;
    div.className = 'message assistant';

    // Sources HTML (with score badges)
    let sourcesHtml = '';
    if (sources.length > 0) {
        const items = sources.map(s => {
            const score = s.score ?? 0;
            const pct = Math.round(score * 100);
            const cls = pct >= 70 ? 'high' : pct >= 45 ? 'mid' : 'low';
            return `<span class="source-item">
                <i class="fas fa-file-lines"></i>${escapeHtml(s.filename)}
                <span class="score-badge ${cls}">${pct}%</span>
            </span>`;
        }).join('');
        sourcesHtml = `<div class="sources"><div class="sources-title">📚 Kaynaklar</div>${items}</div>`;
    }

    // Rewritten query badge
    let rewriteHtml = '';
    if (rewrittenQuery) {
        rewriteHtml = `<div class="rewrite-badge">
            <i class="fas fa-wand-magic-sparkles"></i>
            Sorgu yeniden yazıldı: "<em>${escapeHtml(rewrittenQuery)}</em>"
        </div>`;
    }

    // Pipeline metrics
    let metricsHtml = '';
    if (metrics) {
        const parts = [];
        if (metrics.stt_ms != null)       parts.push(`<span class="metric-item"><i class="fas fa-microphone"></i>STT <span class="val">${metrics.stt_ms.toFixed(0)}ms</span></span>`);
        if (metrics.rewrite_ms != null)    parts.push(`<span class="metric-item"><i class="fas fa-pen"></i>Rewrite <span class="val">${metrics.rewrite_ms.toFixed(0)}ms</span></span>`);
        if (metrics.retrieval_ms != null)  parts.push(`<span class="metric-item"><i class="fas fa-magnifying-glass"></i>RAG <span class="val">${metrics.retrieval_ms.toFixed(0)}ms</span></span>`);
        if (metrics.llm_ms != null)        parts.push(`<span class="metric-item"><i class="fas fa-brain"></i>LLM <span class="val">${metrics.llm_ms.toFixed(0)}ms</span></span>`);
        if (metrics.tts_ms != null)        parts.push(`<span class="metric-item"><i class="fas fa-volume-high"></i>TTS <span class="val">${metrics.tts_ms.toFixed(0)}ms</span></span>`);
        if (metrics.docs_after_threshold != null) {
            parts.push(`<span class="metric-item"><i class="fas fa-filter"></i>Docs <span class="val">${metrics.docs_after_threshold}/${metrics.docs_retrieved ?? '?'}</span></span>`);
        }
        parts.push(`<span class="metric-item"><i class="fas fa-clock"></i>Toplam <span class="val">${metrics.total_ms.toFixed(0)}ms</span></span>`);

        if (parts.length > 0) {
            metricsHtml = `<div class="metrics-panel">${parts.join('')}</div>`;
        }
    }

    // Audio controls
    let audioHtml = '';
    if (isStreaming) {
        audioHtml = `
            <div class="message-audio streaming">
                <button class="play-btn"><i class="fas fa-volume-up"></i></button>
                <div class="audio-wave playing">
                    <span></span><span></span><span></span><span></span><span></span>
                </div>
            </div>
        `;
    } else if (audioBase64) {
        audioHtml = `
            <div class="message-audio" data-audio="${audioBase64}">
                <button class="play-btn" onclick="playStoredAudio('${audioBase64}', this)">
                    <i class="fas fa-play"></i>
                </button>
                <div class="audio-wave">
                    <span></span><span></span><span></span><span></span><span></span>
                </div>
            </div>
        `;
    }

    div.innerHTML = `
        <div class="message-content">
            <div class="text">
                <p>${formatText(text)}</p>
                ${audioHtml}
                ${rewriteHtml}
                ${sourcesHtml}
                ${metricsHtml}
            </div>
        </div>
    `;

    elements.messages.appendChild(div);
    scrollToBottom();

    if (audioBase64 && !isStreaming) {
        setTimeout(() => {
            const btn = div.querySelector('.play-btn');
            if (btn) btn.click();
        }, 300);
    }

    return id;
}

function addLoadingMessage() {
    const id = `loading-${Date.now()}`;
    const div = document.createElement('div');
    div.id = id;
    div.className = 'message assistant loading';
    div.innerHTML = `
        <div class="message-content">
            <div class="loading-dots"><span></span><span></span><span></span></div>
        </div>
    `;
    elements.messages.appendChild(div);
    scrollToBottom();
    return id;
}

function removeMessage(id) { document.getElementById(id)?.remove(); }

function scrollToBottom() { elements.messages.scrollTop = elements.messages.scrollHeight; }

function formatText(text) {
    return escapeHtml(text)
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function capitalize(text) {
    if (!text) return '';
    return text.charAt(0).toUpperCase() + text.slice(1);
}

// ============================================
// File Upload
// ============================================
function initFileUpload() {
    elements.uploadArea.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', (e) => uploadFiles(Array.from(e.target.files)));

    elements.uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.add('dragover');
    });
    elements.uploadArea.addEventListener('dragleave', () => elements.uploadArea.classList.remove('dragover'));
    elements.uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.remove('dragover');
        uploadFiles(Array.from(e.dataTransfer.files));
    });
}

async function uploadFiles(files) {
    for (const file of files) await uploadFile(file);
    loadDocuments();
}

async function uploadFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['pdf', 'docx', 'txt', 'md'].includes(ext)) {
        showToast('error', 'Desteklenmeyen Format', `${ext} formatı desteklenmiyor`);
        return;
    }

    elements.uploadProgress.classList.remove('hidden');
    elements.progressText.textContent = `Yükleniyor: ${file.name}`;
    elements.progressFill.style.width = '0%';

    try {
        const formData = new FormData();
        formData.append('file', file);

        let progress = 0;
        const interval = setInterval(() => {
            progress = Math.min(progress + 15, 90);
            elements.progressFill.style.width = `${progress}%`;
        }, 100);

        const response = await fetch(`${CONFIG.API_BASE}/documents/upload`, {
            method: 'POST',
            body: formData
        });

        clearInterval(interval);
        elements.progressFill.style.width = '100%';

        if (!response.ok) throw new Error('Yükleme başarısız');

        const data = await response.json();
        showToast('success', 'Başarılı', `${file.name} yüklendi (${data.chunk_count} parça)`);

    } catch (error) {
        showToast('error', 'Hata', error.message);
    } finally {
        setTimeout(() => elements.uploadProgress.classList.add('hidden'), 800);
    }
}

// ============================================
// Document Management
// ============================================
async function loadDocuments() {
    try {
        const response = await fetch(`${CONFIG.API_BASE}/documents`);
        const data = await response.json();
        state.documents = data.documents;
        renderDocuments();
        elements.docCount.textContent = `${state.documents.length} Belge`;
    } catch (error) {
        console.error('Load documents error:', error);
    }
}

function renderDocuments() {
    if (state.documents.length === 0) {
        elements.listContent.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-folder-open"></i>
                <p>Henüz belge yüklenmedi</p>
            </div>
        `;
        return;
    }

    const iconMap = { 'pdf': 'file-pdf', 'docx': 'file-word', 'txt': 'file-lines', 'md': 'file-code' };

    elements.listContent.innerHTML = state.documents.map(doc => `
        <div class="document-item">
            <div class="doc-icon ${doc.file_type}">
                <i class="fas fa-${iconMap[doc.file_type] || 'file'}"></i>
            </div>
            <div class="doc-info">
                <div class="doc-name">${escapeHtml(doc.filename)}</div>
                <div class="doc-meta">
                    <span><i class="fas fa-weight-hanging"></i> ${formatSize(doc.file_size)}</span>
                    <span><i class="fas fa-puzzle-piece"></i> ${doc.chunk_count} parça</span>
                </div>
            </div>
            <button class="doc-delete" onclick="deleteDocument('${doc.id}')">
                <i class="fas fa-trash-can"></i>
            </button>
        </div>
    `).join('');
}

function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function initDocumentActions() {
    elements.clearAllBtn.addEventListener('click', async () => {
        if (!confirm('Tüm belgeleri silmek istiyor musunuz?')) return;
        await fetch(`${CONFIG.API_BASE}/documents`, { method: 'DELETE' });
        loadDocuments();
        showToast('success', 'Başarılı', 'Tüm belgeler silindi');
    });
}

async function deleteDocument(id) {
    if (!confirm('Bu belgeyi silmek istiyor musunuz?')) return;
    await fetch(`${CONFIG.API_BASE}/documents/${id}`, { method: 'DELETE' });
    loadDocuments();
    showToast('success', 'Başarılı', 'Belge silindi');
}

// ============================================
// Utilities
// ============================================
function generateId() {
    return 'client_' + Math.random().toString(36).substring(2, 12);
}

function float32ToInt16(float32Array) {
    const int16 = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
        const s = Math.max(-1, Math.min(1, float32Array[i]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return int16;
}

function arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
    return btoa(binary);
}

function base64ToArrayBuffer(base64) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes.buffer;
}

function showToast(type, title, message) {
    const icons = {
        success: 'fa-circle-check',
        error: 'fa-circle-xmark',
        warning: 'fa-triangle-exclamation',
        info: 'fa-circle-info'
    };

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <i class="fas ${icons[type]}"></i>
        <div class="toast-content">
            <div class="toast-title">${escapeHtml(title)}</div>
            <div class="toast-message">${escapeHtml(message)}</div>
        </div>
        <button class="toast-close" onclick="this.parentElement.remove()">
            <i class="fas fa-xmark"></i>
        </button>
    `;

    elements.toastContainer.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// Heartbeat
setInterval(() => {
    if (state.isConnected) sendWsMessage('ping');
}, 30000);

// Particles
function createParticles() {
    const container = document.getElementById('particles');
    if (!container) return;
    const colors = ['#8b5cf6', '#06b6d4', '#10b981', '#f472b6'];
    for (let i = 0; i < 30; i++) {
        const p = document.createElement('div');
        p.className = 'particle';
        p.style.left = `${Math.random() * 100}%`;
        p.style.background = colors[Math.floor(Math.random() * colors.length)];
        p.style.animationDuration = `${15 + Math.random() * 20}s`;
        p.style.animationDelay = `${Math.random() * 20}s`;
        p.style.width = `${2 + Math.random() * 4}px`;
        p.style.height = p.style.width;
        container.appendChild(p);
    }
}
createParticles();

document.addEventListener('mousemove', (e) => {
    document.body.style.setProperty('--mouse-x', e.clientX / window.innerWidth);
    document.body.style.setProperty('--mouse-y', e.clientY / window.innerHeight);
});

// Global exports
window.playStoredAudio = playStoredAudio;
window.deleteDocument = deleteDocument;
