# Voice AI RAG — Sesli Soru-Cevap Sistemi

Tamamen **yerel STT/TTS** ile çalışan, GPU hızlandırmalı sesli yapay zeka asistanı. Belge yükleyin, sesinizle sorun, sesli yanıt alın. İki mod: belge tabanlı RAG veya serbest sohbet.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green?style=flat-square)
![Gemini](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-orange?style=flat-square)
![CUDA](https://img.shields.io/badge/GPU-CUDA%2012%2B-76b900?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

---

## Nedir Bu Proje?

Yüklediğiniz PDF, DOCX veya TXT belgelerini vektör veritabanına kaydeder. Siz sesinizle soru sorduğunuzda:

1. Sesiniz metne çevrilir (Faster Whisper **medium**, **CUDA**, float16 — yerel)
2. Soruya en yakın belge parçaları bulunur (ChromaDB + hybrid BM25+vektör arama)
3. Google Gemini bu parçalara dayanarak yanıt üretir ve token token akıtır
4. Yanıt cümle cümle TTS'e gönderilir — **ilk cümle TTS biter bitmez ses başlar**
5. Piper TTS WAV'lar sıralı zincirle çalınır, kelimeler üst üste gelmez

**İki çalışma modu:**
- **Belge RAG** — Yüklediğiniz belgelerden yanıt üretir
- **Serbest Sohbet** — Belgesiz, doğrudan Gemini ile konuşur

STT ve TTS tamamen yereldir. Yalnızca LLM için Gemini API anahtarı gerekir (ücretsiz kota mevcut).

---

## Mimari

```
┌────────────────────────────────────────────────────────────────────┐
│                          KULLANICI                                  │
│                    (Tarayıcı / Frontend)                            │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │   Mode Toggle: [ Belge RAG ] [ Serbest Sohbet ]              │  │
│  │   VAD → PCM/16kHz → WebSocket  |  Text → SSE stream          │  │
│  │   Audio: sentence_q zinciri → overlap yok                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
└──────────────────────────┬─────────────────────────────────────────┘
                           │  WebSocket / HTTP SSE
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│                       FastAPI Backend                               │
│  uvicorn (uvloop + httptools) · 4 worker · 32 thread pool          │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │              3-Aşamalı Streaming Pipeline                   │    │
│  │                                                            │    │
│  │  produce_sentences ──► sentence_q ──► consume_sentences    │    │
│  │       (LLM token)                        (Piper TTS)       │    │
│  │                                              │             │    │
│  │                                         audio_q            │    │
│  │                                              │             │    │
│  │                                        emit_audio          │    │
│  │                                   (cümle hazır → gönder)   │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌──────────────┐   ┌─────────────────┐   ┌────────────────────┐  │
│  │ Faster       │   │  RAGService     │   │  LLMService        │  │
│  │ Whisper STT  │   │  BM25 + Vector  │   │  Gemini 2.5 Flash  │  │
│  │ medium·CUDA  │   │  Hybrid Search  │   │  streaming tokens  │  │
│  │ float16      │   │  Reranker       │   │                    │  │
│  └──────────────┘   └─────────────────┘   └────────────────────┘  │
│                                                                     │
│  ┌──────────────┐   ┌─────────────────┐   ┌────────────────────┐  │
│  │ Piper TTS    │   │  ChromaDB       │   │  ConvService       │  │
│  │ tr_TR-dfki   │   │  HNSW Cosine    │   │  Session Memory    │  │
│  │ ONNX · CPU   │   │  Embeddings:    │   │  10 tur · 60 dk    │  │
│  │ (ORT CPU)    │   │  MiniLM · CUDA  │   │                    │  │
│  └──────────────┘   └─────────────────┘   └────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

### Streaming Ses Pipeline (3 aşama paralel)

```
LLM token → cümle birikir → sentence_q
                                  │
                            consume_sentences
                            (Piper TTS, CPU)
                                  │
                             audio_q → emit_audio → WebSocket → tarayıcı
                                                   (cümle hazır olunca)
```

İlk cümlenin TTS'i biter bitmez ses çalmaya başlar.  
Sonraki cümleler LLM hâlâ üretirken TTS'e girer, hiç bekleme olmaz.  
Tarayıcı `audioChain` ile cümleleri sıralı schedule eder → overlap yok.

### RAG Akışı

```
Belge Yükleme:
  PDF/DOCX/TXT ──► Parse ──► Parent-Child Chunk ──► BM25 + Embed ──► ChromaDB

Soru Sorma (RAG modu):
  Soru ──► Embed(CUDA) ──► BM25 + Cosine Hybrid ──► Reranker ──► Top-K
                                                                      │
  Soru + Context ──────────────────────────────────────► Gemini ──► Yanıt

Soru Sorma (Serbest mod):
  Soru + Konuşma Geçmişi ──────────────────────────────► Gemini ──► Yanıt
```

---

## Bileşenler

### STT — Faster Whisper

| Özellik        | Değer                                   |
|----------------|-----------------------------------------|
| Model          | `medium` (769 MB)                       |
| Cihaz          | **CUDA** (GPU)                          |
| Quantization   | **float16**                             |
| CPU threads    | 16 (fallback)                           |
| Dil            | Türkçe (tr)                             |
| Warm-up        | Startup'ta CUDA kernel JIT tetiklenir   |

**Model Seçenekleri:**

| Model    | Boyut   | GPU Hızı | Doğruluk  |
|----------|---------|----------|-----------|
| tiny     | 39 MB   | ~0.05s   | Düşük     |
| base     | 74 MB   | ~0.08s   | Orta      |
| small    | 273 MB  | ~0.12s   | İyi       |
| **medium** | **769 MB** | **~0.2s** | **Çok İyi** |
| large-v3 | 1550 MB | ~0.4s    | En İyi    |

### TTS — Piper (Yerel)

| Özellik        | Değer                              |
|----------------|------------------------------------|
| Model          | `tr_TR-dfki-medium.onnx`           |
| Çalışma        | ONNX Runtime CPU (ORT GPU yok)     |
| ONNX threads   | 8 (benchmark optimali)             |
| Execution mode | ORT_PARALLEL                       |
| Streaming      | Cümle bazlı — ilk cümle ~0.2s     |

### Embedding + Reranker

| Bileşen    | Model                                   | Cihaz   |
|------------|-----------------------------------------|---------|
| Embedding  | `paraphrase-multilingual-MiniLM-L12-v2` | **CUDA** |
| Reranker   | `cross-encoder/ms-marco-MiniLM-L-6-v2` | **CUDA** |

### LLM — Google Gemini

| Özellik    | Değer                         |
|------------|-------------------------------|
| Model      | `gemini-2.5-flash`            |
| Streaming  | Token-by-token SSE + WS       |
| Dil        | Türkçe sistem promptu         |

---

## Kurulum

### Ön Gereksinimler

- Python 3.10+
- NVIDIA GPU + CUDA 12+ (önerilir; CPU ile de çalışır)
- Google Gemini API anahtarı — [aistudio.google.com](https://aistudio.google.com/app/apikey)

### 1 — Klonla

```bash
git clone <repo-url>
cd Local-ASR-TTS-RAG
```

### 2 — Bağımlılıkları Yükle

```bash
cd backend
pip install -r requirements.txt
```

### 3 — Ortam Değişkenlerini Ayarla

```bash
cp .env.example .env
```

`.env` içine en az şunu ekleyin:

```env
GEMINI_API_KEY=your_key_here
```

### 4 — Piper TTS Modelini İndir

```bash
mkdir -p backend/models && cd backend/models

wget https://huggingface.co/rhasspy/piper-voices/resolve/main/tr/tr_TR/dfki/medium/tr_TR-dfki-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/tr/tr_TR/dfki/medium/tr_TR-dfki-medium.onnx.json
```

### 5 — Backend'i Başlat

```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

İlk başlatmada:
- Faster Whisper `medium` modeli indirilir (~769 MB)
- CUDA kernel warm-up çalışır (~5-20s, sadece bir kez)
- Tüm modeller paralel yüklenir

Tarayıcıda: **http://localhost:8000**  
API docs: **http://localhost:8000/docs**

---

## Konfigürasyon (`.env`)

```env
# ── Zorunlu ─────────────────────────────────────────────────────
GEMINI_API_KEY=your_key_here

# ── LLM ────────────────────────────────────────────────────────
GEMINI_MODEL=gemini-2.5-flash

# ── STT (GPU önerilir) ──────────────────────────────────────────
WHISPER_MODEL_SIZE=medium        # tiny | base | small | medium | large-v3
WHISPER_DEVICE=cuda              # cpu | cuda
WHISPER_COMPUTE_TYPE=float16     # int8 (CPU) | float16 (GPU)
WHISPER_CPU_THREADS=16           # CUDA'da etkisiz; CPU fallback için

# ── TTS ─────────────────────────────────────────────────────────
PIPER_MODEL_PATH=models/tr_TR-dfki-medium.onnx

# ── Embedding + Reranker ────────────────────────────────────────
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DEVICE=cuda            # cpu | cuda
RERANKER_DEVICE=cuda             # cpu | cuda

# ── RAG ─────────────────────────────────────────────────────────
CHUNK_SIZE=500
CHUNK_OVERLAP=50
RETRIEVAL_TOP_K=5
ENABLE_HYBRID_SEARCH=true        # BM25 + vektör karışımı
ENABLE_RERANKING=true
ENABLE_PARENT_CHILD=true         # Daha geniş bağlam için

# ── Konuşma Hafızası ────────────────────────────────────────────
MAX_CONVERSATION_HISTORY=10
SESSION_EXPIRY_MINUTES=60

# ── Debug ───────────────────────────────────────────────────────
DEBUG=false                      # true → 1 worker, reload açık
```

### CPU ile Çalıştırma

```env
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_MODEL_SIZE=small
EMBEDDING_DEVICE=cpu
RERANKER_DEVICE=cpu
```

---

## Frontend — Özellikler

| Özellik                  | Açıklama                                               |
|--------------------------|--------------------------------------------------------|
| **Mod Seçimi**           | Belge RAG ↔ Serbest Sohbet toggle (header'da)         |
| **VAD**                  | Otomatik sessizlik algılama (1500ms eşik)              |
| **Streaming Metin**      | LLM token'ları canlı yazılır                           |
| **Streaming Ses**        | İlk cümle TTS bitince çalmaya başlar                  |
| **Yanıt Süresi**         | Her mesajda `⚡ 2.3s` badge                           |
| **Kaynak Gösterimi**     | Hangi belgeden, yüzde kaç benzerlik                   |
| **Metriks Paneli**       | STT / RAG / LLM / TTS ms ayrı ayrı                   |
| **Ses Tekrar Çalma**     | Tüm cümleler birleşik WAV olarak saklanır             |
| **Hata Kurtarma**        | 45s frontend timeout, 90s backend timeout             |

---

## API Referansı

### WebSocket — Gerçek Zamanlı Ses

```
WS /api/ws/realtime/{client_id}
```

**İstemci → Sunucu:**
```jsonc
{"type": "start", "mode": "rag"}          // Dinlemeyi başlat
{"type": "audio", "data": "<base64-pcm>"} // Ses chunk'ı
{"type": "stop"}                           // Durdur ve işle
{"type": "cancel"}                         // İptal et
{"type": "ping"}                           // Heartbeat
```

**Sunucu → İstemci:**
```jsonc
{"type": "connected",    "data": {"client_id": "..."}}
{"type": "state",        "data": {"state": "idle|listening|processing|speaking"}}
{"type": "transcription","data": {"text": "...", "is_final": false}}
{"type": "user_message", "data": {"text": "..."}}
{"type": "answer_token", "data": {"text": "..."}}            // streaming
{"type": "audio_chunk",  "data": {"data": "<base64-wav>"}}  // cümle bazlı, sıralı
{"type": "answer",       "data": {"text": "...", "sources": [...], "total_ms": 1240}}
{"type": "audio_complete","data": {"full_audio": "<base64-wav>"}}
{"type": "error",        "data": {"message": "..."}}
```

### REST — Metin Sorgusu (SSE)

```
POST /api/chat/stream
```
```json
{"query": "Sorunuz", "session_id": "...", "mode": "rag"}
```

SSE olayları: `rewrite` → `sources` → `token` × N → `done`

### Belge Yönetimi

| Metod    | Endpoint                | Açıklama              |
|----------|-------------------------|-----------------------|
| `POST`   | `/api/documents/upload` | PDF/DOCX/TXT/MD yükle |
| `GET`    | `/api/documents`        | Belgeleri listele     |
| `DELETE` | `/api/documents/{id}`   | Belge sil             |
| `DELETE` | `/api/documents`        | Tümünü sil            |

---

## Proje Yapısı

```
Local-ASR-TTS-RAG/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI fabrika, lifespan, warm-up, thread pool
│   │   ├── config.py                # Pydantic Settings (.env okuma, tüm ayarlar)
│   │   ├── api/
│   │   │   ├── routes.py            # REST + SSE endpoint'leri
│   │   │   ├── websocket_routes.py  # WS endpoint, ConnectionManager
│   │   │   └── dependencies.py      # Singleton servis injection
│   │   ├── models/
│   │   │   ├── schemas.py           # Request/response Pydantic modelleri
│   │   │   └── exceptions.py        # Özel hata sınıfları
│   │   ├── services/
│   │   │   ├── realtime_service.py  # 3-aşamalı streaming pipeline
│   │   │   ├── rag_service.py       # ChromaDB, hybrid search, reranker
│   │   │   ├── llm_service.py       # Gemini / Ollama / OpenAI-compat
│   │   │   ├── document_service.py  # PDF/DOCX/TXT parse, parent-child chunk
│   │   │   ├── speech_service.py    # STT/TTS REST sarmalayıcı
│   │   │   ├── reranker_service.py  # Cross-encoder reranking
│   │   │   └── conversation_service.py # Session hafızası
│   │   └── utils/
│   │       └── audio_utils.py       # WAV/PCM yardımcıları
│   ├── models/                      # Piper .onnx buraya
│   ├── data/
│   │   ├── documents/               # Yüklenen dosyalar
│   │   └── chroma_db/               # ChromaDB kalıcı depolama
│   ├── requirements.txt
│   ├── .env
│   └── .env.example
└── frontend/
    ├── index.html                   # Mod toggle, ses kontrolleri, belge yönetimi
    ├── styles.css                   # Dark glassmorphism tema, animasyonlar
    └── app.js                       # VAD, WebSocket, SSE, streaming audio zinciri
```

---

## Sık Sorulan Sorular

**İlk başlatma neden uzun sürüyor?**  
Faster Whisper modeli indirilir ve CUDA kernel'leri derlenir (warm-up). Sonraki başlatmalarda bu adımlar atlanır.

**Ses "bu" deyip kesiliyorsa?**  
Eski sürümlerde WAV'ın byte bazlı bölünmesinden kaynaklanan bir bugdu. Şu an her cümle WAV'ı eksiksiz gönderilmektedir.

**"Selam" söyleyince takılıp kalıyorsa?**  
Startup warm-up bu sorunu çözer. İlk Whisper CUDA çağrısı kernel JIT compilation gerektiriyordu; artık başlangıçta tetikleniyor.

**GPU yoksa ne olur?**  
`.env`'de `WHISPER_DEVICE=cpu`, `WHISPER_COMPUTE_TYPE=int8`, `EMBEDDING_DEVICE=cpu`, `RERANKER_DEVICE=cpu` yapın. Her şey çalışır, biraz daha yavaş olur.

**Başka dil için TTS?**  
[Piper Voices](https://huggingface.co/rhasspy/piper-voices) sayfasından modeli indirip `PIPER_MODEL_PATH` ayarlayın.

---

## Bağımlılıklar

| Paket                   | Kullanım                            |
|-------------------------|-------------------------------------|
| `fastapi` + `uvicorn`   | Web framework + ASGI (uvloop)       |
| `google-generativeai`   | Gemini LLM API                      |
| `faster-whisper`        | Lokal STT (CTranslate2, CUDA)       |
| `piper-tts`             | Lokal TTS (ONNX CPU)                |
| `chromadb`              | Vektör veritabanı                   |
| `sentence-transformers` | Embedding (CUDA)                    |
| `rank-bm25`             | Keyword arama (hybrid)              |
| `pdfplumber` + `PyPDF2` | PDF parse                           |
| `python-docx`           | DOCX parse                          |
| `pydantic-settings`     | Tip güvenli konfigürasyon           |

---

## Lisans

MIT — Detaylar için [LICENSE](LICENSE) dosyasına bakın.
