# Voice AI RAG — Sesli Soru-Cevap Sistemi

Tamamen **yerel çalışan** (API bedava) bir sesli yapay zeka asistanı. Belge yükleyin, sesinizle sorun, sesli yanıt alın.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green?style=flat-square)
![Gemini](https://img.shields.io/badge/LLM-Gemini%202.0%20Flash-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

---

## Nedir Bu Proje?

Yüklediğiniz PDF, DOCX veya TXT belgelerini vektör veritabanına kaydeder. Siz sesinizle soru sorduğunuzda:

1. Sesiniz metne çevrilir (Faster Whisper, lokal)
2. Soruya en yakın belge parçaları bulunur (ChromaDB + embedding)
3. Google Gemini bu parçalara dayanarak yanıt üretir
4. Yanıt sese çevrilir (Piper TTS, lokal)

STT ve TTS tamamen lokaldir — internet bağlantısı veya API anahtarı gerekmez. Yalnızca LLM için Google Gemini API anahtarı gereklidir (ücretsiz kota mevcut).

---

## Mimari

```
┌─────────────────────────────────────────────────────────────────┐
│                         KULLANICI                               │
│                   (Tarayıcı / Frontend)                         │
└──────────────────────┬──────────────────────────────────────────┘
                       │  HTTP REST  /  WebSocket
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                             │
│                                                                 │
│  ┌─────────────┐   ┌──────────────┐   ┌────────────────────┐  │
│  │  REST API   │   │  WebSocket   │   │  Dependency        │  │
│  │  /api/*     │   │  /ws/realtime│   │  Injection         │  │
│  └──────┬──────┘   └──────┬───────┘   └────────────────────┘  │
│         │                 │                                     │
│         └────────┬────────┘                                     │
│                  ▼                                              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   SERVICE KATMANI                         │  │
│  │                                                           │  │
│  │  ┌─────────────────┐   ┌────────────────────────────┐   │  │
│  │  │  SpeechService  │   │     DocumentService        │   │  │
│  │  │                 │   │                            │   │  │
│  │  │  STT:           │   │  • PDF  (pdfplumber)       │   │  │
│  │  │  Faster Whisper │   │  • DOCX (python-docx)      │   │  │
│  │  │  (lokal, CPU)   │   │  • TXT / MD                │   │  │
│  │  │                 │   │  • Overlapping chunking    │   │  │
│  │  │  TTS:           │   │    (500 char, 50 overlap)  │   │  │
│  │  │  Piper TTS      │   └────────────┬───────────────┘   │  │
│  │  │  (lokal, ONNX)  │                │                   │  │
│  │  └────────┬────────┘                ▼                   │  │
│  │           │           ┌────────────────────────────┐   │  │
│  │           │           │       RAGService            │   │  │
│  │           │           │                            │   │  │
│  │           │           │  Embedding:                │   │  │
│  │           │           │  paraphrase-multilingual-  │   │  │
│  │           │           │  MiniLM-L12-v2             │   │  │
│  │           │           │                            │   │  │
│  │           │           │  VectorDB:                 │   │  │
│  │           │           │  ChromaDB (HNSW Cosine)    │   │  │
│  │           │           └────────────┬───────────────┘   │  │
│  │           │                        │                   │  │
│  │           │                        ▼                   │  │
│  │           │           ┌────────────────────────────┐   │  │
│  │           │           │       LLMService           │   │  │
│  │           │           │                            │   │  │
│  │           │           │  Google Gemini 2.0 Flash   │   │  │
│  │           │           │  Temperature: 0.7          │   │  │
│  │           │           │  Max tokens: 1000          │   │  │
│  │           │           │  System: Türkçe asistan    │   │  │
│  │           └───────────►  Context + Query → Answer  │   │  │
│  │                        └────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Ses Akışı (Voice Pipeline)

```
Kullanıcı Konuşur
      │
      ▼
┌─────────────┐
│ Faster      │  WAV (16kHz, 16-bit, mono)
│ Whisper STT │  Model: small / CPU / int8
│ (lokal)     │  Dil: Türkçe (tr)
└──────┬──────┘
       │ Transkript metni
       ▼
┌─────────────┐
│ ChromaDB    │  MiniLM-L12-v2 embedding
│ Similarity  │  HNSW cosine distance
│ Search      │  Top-K = 5 chunk
└──────┬──────┘
       │ Bağlam (context)
       ▼
┌─────────────┐
│ Gemini 2.0  │  System prompt + Context + Soru
│ Flash (API) │  → Türkçe cevap
└──────┬──────┘
       │ Yanıt metni
       ▼
┌─────────────┐
│ Piper TTS   │  ONNX model: tr_TR-dfki-medium
│ (lokal)     │  WAV çıktısı → Base64
└──────┬──────┘
       │ Audio (Base64 WAV)
       ▼
 Kullanıcı Dinler
```

### RAG (Retrieval Augmented Generation) Akışı

```
Belge Yükleme:
  PDF/DOCX/TXT ──► Parse ──► Chunk ──► Embed ──► ChromaDB

Soru Sorma:
  Soru ──► Embed ──► Cosine Similarity ──► Top-K Chunk ──► Context
                                                               │
  Soru + Context ──────────────────────────────────► Gemini ──► Yanıt
```

---

## LLM Yapısı (Detaylı)

### 1. Embedding Modeli

| Özellik        | Değer                                                  |
|----------------|--------------------------------------------------------|
| Model          | `paraphrase-multilingual-MiniLM-L12-v2`                |
| Boyut          | 384 boyut vektör                                       |
| Dil desteği    | 50+ dil (Türkçe dahil)                                 |
| Çalışma yeri   | Lokal CPU                                              |
| Kullanım amacı | Belge chunk'larını ve soruları vektöre çevirmek        |

### 2. Vektör Veritabanı (ChromaDB)

| Özellik         | Değer                                      |
|-----------------|--------------------------------------------|
| Depolama        | Disk (persistent)                          |
| İndeks          | HNSW (Hierarchical Navigable Small World)  |
| Benzerlik       | Cosine similarity                          |
| Score aralığı   | 0.0 (farklı) – 1.0 (özdeş)                |
| Chunk boyutu    | 500 karakter (50 karakter örtüşme)         |

### 3. LLM (Google Gemini 2.0 Flash)

| Özellik         | Değer                                      |
|-----------------|--------------------------------------------|
| Model ID        | `gemini-2.0-flash`                         |
| Temperature     | 0.7 (dengeli yaratıcılık)                  |
| Max tokens      | 1000                                       |
| Sistem dili     | Türkçe                                     |
| Kaynak politika | Yalnızca yüklü belgelerden cevap üretir    |

**Sistem Promptu:**
```
Sen yardımcı bir Türkçe asistansın.
Sana verilen kaynak belgelerine dayanarak soruları yanıtlıyorsun.

Kurallar:
1. Sadece verilen kaynaklardaki bilgilere dayanarak cevap ver
2. Eğer cevap kaynaklarda yoksa, "Bu konuda kaynaklarda bilgi bulamadım" de
3. Cevaplarını kısa ve öz tut
4. Doğal ve akıcı bir Türkçe kullan
5. Gerektiğinde kaynak belgeyi belirt
6. Emin olmadığın bilgileri tahmin etme
```

### 4. STT — Faster Whisper (Lokal)

| Özellik         | Değer                                      |
|-----------------|--------------------------------------------|
| Model           | `small` (273 MB)                           |
| Cihaz           | CPU                                        |
| Quantization    | int8 (bellek tasarrufu)                    |
| Dil             | Türkçe (tr)                                |
| Giriş formatı   | WAV (16kHz, 16-bit, mono)                  |
| API gereklimi?  | Hayır — tamamen lokal                      |

**Model Boyut Karşılaştırması:**

| Model    | Boyut  | Hız    | Doğruluk |
|----------|--------|--------|----------|
| tiny     | 39 MB  | Çok hızlı | Düşük |
| base     | 74 MB  | Hızlı  | Orta     |
| **small**| **273 MB** | **Orta** | **İyi** |
| medium   | 769 MB | Yavaş  | Çok iyi  |
| large-v3 | 1550 MB| En yavaş | En iyi |

### 5. TTS — Piper (Lokal)

| Özellik         | Değer                                      |
|-----------------|--------------------------------------------|
| Model           | `tr_TR-dfki-medium.onnx`                   |
| Çalışma şekli   | ONNX Runtime (CPU)                         |
| Çıktı formatı   | WAV                                        |
| API gereklimi?  | Hayır — tamamen lokal                      |
| Sesler          | tr_TR-dfki-medium (Nötr)                   |

---

## Kurulum

### Ön Gereksinimler

- Python 3.10+
- Google Gemini API anahtarı (ücretsiz: [aistudio.google.com](https://aistudio.google.com/app/apikey))
- ~1 GB disk alanı (modeller için)

### Adım 1 — Depoyu Klonlayın

```bash
git clone <repo-url>
cd local
```

### Adım 2 — Bağımlılıkları Yükleyin

```bash
cd backend
pip install -r requirements.txt
```

### Adım 3 — Ortam Değişkenlerini Ayarlayın

```bash
cp .env.example .env
```

`.env` dosyasını açıp yalnızca şu satırı doldurun:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Diğer tüm ayarlar varsayılan olarak çalışır.

### Adım 4 — Piper TTS Modelini İndirin

Türkçe ses modeli manuel indirilmeli:

```bash
# backend/ dizininde
mkdir -p models
cd models

# Linux / Mac
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/tr/tr_TR/dfki/medium/tr_TR-dfki-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/tr/tr_TR/dfki/medium/tr_TR-dfki-medium.onnx.json

# Windows (PowerShell)
Invoke-WebRequest -Uri "https://huggingface.co/rhasspy/piper-voices/resolve/main/tr/tr_TR/dfki/medium/tr_TR-dfki-medium.onnx" -OutFile "tr_TR-dfki-medium.onnx"
Invoke-WebRequest -Uri "https://huggingface.co/rhasspy/piper-voices/resolve/main/tr/tr_TR/dfki/medium/tr_TR-dfki-medium.onnx.json" -OutFile "tr_TR-dfki-medium.onnx.json"
```

Beklenen dizin yapısı:
```
backend/
└── models/
    ├── tr_TR-dfki-medium.onnx
    └── tr_TR-dfki-medium.onnx.json
```

### Adım 5 — Backend'i Başlatın

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

İlk çalıştırmada Faster Whisper modeli (~273 MB) otomatik indirilir. Bu birkaç dakika sürebilir.

### Adım 6 — Frontend'i Açın

```bash
cd frontend
python -m http.server 3000
```

Tarayıcıda açın: **http://localhost:3000**

API dokümantasyonu: **http://localhost:8000/docs**

---

## Proje Yapısı

```
local/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI uygulama fabrikası, lifespan
│   │   ├── config.py                # Pydantic Settings, .env okuma
│   │   ├── api/
│   │   │   ├── routes.py            # REST endpoint'leri
│   │   │   ├── websocket_routes.py  # WebSocket & ConnectionManager
│   │   │   └── dependencies.py      # Dependency injection (singleton servisler)
│   │   ├── models/
│   │   │   ├── schemas.py           # Pydantic request/response modelleri
│   │   │   └── exceptions.py        # Özel hata sınıfları
│   │   ├── services/
│   │   │   ├── llm_service.py       # Google Gemini entegrasyonu
│   │   │   ├── rag_service.py       # ChromaDB + embedding + similarity search
│   │   │   ├── document_service.py  # Dosya parse + chunking
│   │   │   ├── speech_service.py    # STT/TTS REST API için
│   │   │   └── realtime_service.py  # WebSocket için pipeline + streaming TTS
│   │   └── utils/
│   │       └── audio_utils.py       # WAV yardımcı fonksiyonlar
│   ├── data/
│   │   ├── documents/               # Yüklenen dosyalar (uuid.ext)
│   │   └── chroma_db/               # ChromaDB kalıcı depolama
│   ├── models/                      # Piper TTS .onnx modeli buraya gelir
│   ├── requirements.txt
│   ├── .env                         # (git'e eklenmez)
│   └── .env.example
└── frontend/
    ├── index.html
    ├── styles.css
    └── app.js
```

---

## API Referansı

### Health Check

```
GET /api/health
```

Tüm servislerin durumunu döner.

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "services": {
    "speech": true,
    "document": true,
    "rag": true,
    "llm": true,
    "document_count": 42
  }
}
```

### Belge Yönetimi

| Metod    | Endpoint                    | Açıklama                          |
|----------|-----------------------------|-----------------------------------|
| `POST`   | `/api/documents/upload`     | PDF/DOCX/TXT/MD yükle             |
| `GET`    | `/api/documents`            | Yüklü belgeleri listele           |
| `DELETE` | `/api/documents/{id}`       | Belirli bir belgeyi sil           |
| `DELETE` | `/api/documents`            | Tüm belgeleri sil                 |

**Yükleme yanıtı:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "rapor.pdf",
  "file_type": "pdf",
  "chunk_count": 24,
  "message": "Belge başarıyla yüklendi. 24 parça oluşturuldu.",
  "success": true
}
```

### Soru Sorma (REST)

#### Yazılı Soru (ses yanıtı opsiyonel)

```
POST /api/text/query
```

```json
{
  "query": "Şirketin 2024 geliri nedir?",
  "include_audio": true
}
```

Yanıt:
```json
{
  "query": "Şirketin 2024 geliri nedir?",
  "answer": "2024 yılında şirket geliri 45 milyon TL olarak gerçekleşmiştir.",
  "sources": [
    {
      "filename": "rapor.pdf",
      "content": "...ilgili paragraf...",
      "score": 0.92,
      "page": 12
    }
  ],
  "audio_base64": "UklGRiQA...",
  "processing_time_ms": 843.5
}
```

#### Yalnızca RAG Sorgusu (ses yok)

```
POST /api/rag/query
```

```json
{
  "query": "Proje teslim tarihi ne zaman?",
  "top_k": 5,
  "include_sources": true
}
```

#### Sesli Soru

```
POST /api/voice/query
Content-Type: multipart/form-data

audio: <WAV dosyası>
```

Yanıt:
```json
{
  "transcribed_text": "Proje teslim tarihi ne zaman?",
  "answer": "Proje 15 Haziran 2025 tarihinde teslim edilecektir.",
  "sources": [...],
  "audio_base64": "UklGRiQA...",
  "processing_time_ms": 1243.2
}
```

### WebSocket — Gerçek Zamanlı Ses Akışı

```
WS /ws/realtime/{client_id}
```

**İstemci → Sunucu:**

```jsonc
// Dinlemeyi başlat (mode: "rag" veya "free")
{"type": "start", "mode": "rag"}

// Ses chunk'ı gönder (Base64 PCM, 16kHz, 16-bit, mono)
{"type": "audio", "data": "<base64>"}

// Dinlemeyi durdur ve işle
{"type": "stop"}

// İptal et
{"type": "cancel"}

// Bağlantı testi
{"type": "ping"}
```

**Sunucu → İstemci:**

```jsonc
// Bağlantı kuruldu
{"type": "connected", "data": {"client_id": "abc123"}}

// Durum değişikliği
{"type": "state", "data": {"state": "idle|listening|processing|speaking"}}

// Transkript (anlık)
{"type": "transcription", "data": {"text": "proje...", "is_final": false}}

// Transkript (final)
{"type": "transcription", "data": {"text": "proje teslim tarihi ne zaman", "is_final": true}}

// AI yanıtı
{"type": "answer", "data": {"text": "15 Haziran 2025...", "sources": [...]}}

// TTS ses chunk'ı (streaming)
{"type": "audio_chunk", "data": {"data": "<base64>", "format": "wav"}}

// TTS tamamlandı (tam ses)
{"type": "audio_complete", "data": {"full_audio": "<base64>"}}

// Hata
{"type": "error", "data": {"message": "..."}}
```

---

## Konfigürasyon

Tüm ayarlar `backend/.env` dosyasında yapılır.

### Zorunlu

```env
GEMINI_API_KEY=your_key_here
```

### LLM Ayarları

```env
GEMINI_MODEL=gemini-2.0-flash    # veya gemini-1.5-pro, gemini-1.5-flash
```

### STT (Whisper) Ayarları

```env
WHISPER_MODEL_SIZE=small         # tiny | base | small | medium | large-v3
WHISPER_DEVICE=cpu               # cpu | cuda (GPU için)
WHISPER_COMPUTE_TYPE=int8        # int8 | float16 | float32
SPEECH_LANGUAGE=tr               # tr | en | de | ...
```

GPU kullanmak için (`cuda` varsa):
```env
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
```

### TTS (Piper) Ayarları

```env
PIPER_MODEL_PATH=models/tr_TR-dfki-medium.onnx
```

### RAG Ayarları

```env
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
CHUNK_SIZE=500                   # Chunk boyutu (karakter)
CHUNK_OVERLAP=50                 # Örtüşme miktarı
RETRIEVAL_TOP_K=5                # Kaç chunk getirilsin
```

Daha iyi doğruluk için:
```env
CHUNK_SIZE=800
CHUNK_OVERLAP=100
RETRIEVAL_TOP_K=8
```

---

## Sık Sorulan Sorular

**Whisper modeli indirilmiyor mu?**
İlk çalıştırmada internet gereklidir. Model `~/.cache/huggingface/` altına indirilir ve sonraki çalıştırmalarda tekrar indirilmez.

**Piper TTS çalışmıyor mu?**
`backend/models/` dizininde hem `.onnx` hem `.onnx.json` dosyasının olduğundan emin olun.

**Yanıtlar çok uzun veya kısa geliyor mu?**
`LLMService` içindeki `max_output_tokens` değerini ayarlayın (varsayılan: 1000).

**GPU hızlandırma?**
CUDA destekli GPU varsa `.env`'de `WHISPER_DEVICE=cuda` ve `WHISPER_COMPUTE_TYPE=float16` olarak değiştirin.

**Hangi diller desteklenir?**
STT için Faster Whisper 99 dili destekler. TTS için yalnızca Türkçe model yapılandırılmıştır (başka Piper modelleri eklenebilir). LLM herhangi bir dilde yanıt verebilir.

---

## Bağımlılıklar

| Paket                          | Kullanım                               |
|--------------------------------|----------------------------------------|
| `fastapi`                      | Web framework                          |
| `uvicorn`                      | ASGI sunucu                            |
| `google-generativeai`          | Gemini LLM API                         |
| `faster-whisper`               | Lokal STT (Whisper quantized)          |
| `piper-tts`                    | Lokal TTS (ONNX)                       |
| `chromadb`                     | Vektör veritabanı                      |
| `sentence-transformers`        | Embedding modeli                       |
| `pdfplumber` + `PyPDF2`        | PDF parse (fallback zinciri)           |
| `python-docx`                  | DOCX parse                             |
| `pydantic-settings`            | Tip güvenli konfigürasyon              |

---

## Lisans

MIT — Detaylar için [LICENSE](LICENSE) dosyasına bakın.
