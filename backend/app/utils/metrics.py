"""
Prometheus metrics — request counters, per-stage latency histograms, cache stats.
Imported lazily so missing prometheus_client doesn't crash the app.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram

    # ── Request counters ──────────────────────────────────────────────────────
    REQUEST_COUNT = Counter(
        "voiceai_requests_total",
        "Total API requests by endpoint and HTTP status",
        ["endpoint", "status"],
    )
    ERROR_COUNT = Counter(
        "voiceai_errors_total",
        "Total API errors by endpoint",
        ["endpoint", "error_type"],
    )

    # ── Per-stage latency (seconds) ───────────────────────────────────────────
    STT_LATENCY = Histogram(
        "voiceai_stt_seconds",
        "Whisper transcription latency",
        buckets=[0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0],
    )
    LLM_LATENCY = Histogram(
        "voiceai_llm_seconds",
        "LLM generation latency",
        ["provider"],
        buckets=[0.2, 0.5, 1.0, 2.0, 5.0, 15.0, 30.0, 60.0],
    )
    TTS_LATENCY = Histogram(
        "voiceai_tts_seconds",
        "TTS synthesis latency",
        ["backend"],
        buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 3.0, 8.0],
    )
    RETRIEVAL_LATENCY = Histogram(
        "voiceai_retrieval_seconds",
        "RAG retrieval latency",
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
    )
    TOTAL_LATENCY = Histogram(
        "voiceai_request_seconds",
        "Total request latency by endpoint",
        ["endpoint"],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 15.0, 30.0, 60.0],
    )

    # ── Semantic cache ────────────────────────────────────────────────────────
    CACHE_HITS = Counter(
        "voiceai_cache_hits_total",
        "Semantic cache hits",
    )
    CACHE_MISSES = Counter(
        "voiceai_cache_misses_total",
        "Semantic cache misses",
    )

    # ── System / resource gauges ──────────────────────────────────────────────
    ACTIVE_WS_CONNECTIONS = Gauge(
        "voiceai_active_websocket_connections",
        "Current number of open WebSocket connections",
    )
    DOCUMENT_CHUNKS = Gauge(
        "voiceai_document_chunks_total",
        "Total indexed document chunks in ChromaDB",
    )

    # ── Retry / circuit-breaker ───────────────────────────────────────────────
    LLM_RETRIES = Counter(
        "voiceai_llm_retries_total",
        "LLM calls retried due to transient errors",
        ["provider"],
    )
    CIRCUIT_OPEN_REJECTIONS = Counter(
        "voiceai_circuit_open_rejections_total",
        "Requests rejected because the circuit breaker is OPEN",
        ["provider"],
    )

    METRICS_AVAILABLE = True
    logger.info("Prometheus metrics initialized")

except ImportError:
    METRICS_AVAILABLE = False
    logger.warning(
        "prometheus_client not installed — /metrics endpoint disabled. "
        "Run: pip install prometheus-client"
    )

    # Stub objects so callers don't need to guard every call
    class _Noop:
        def labels(self, **_):
            return self
        def inc(self, *_, **__):
            pass
        def observe(self, *_, **__):
            pass
        def set(self, *_, **__):
            pass

    _noop = _Noop()
    REQUEST_COUNT = ERROR_COUNT = _noop
    STT_LATENCY = LLM_LATENCY = TTS_LATENCY = RETRIEVAL_LATENCY = TOTAL_LATENCY = _noop
    CACHE_HITS = CACHE_MISSES = _noop
    ACTIVE_WS_CONNECTIONS = DOCUMENT_CHUNKS = _noop
    LLM_RETRIES = CIRCUIT_OPEN_REJECTIONS = _noop


def record_pipeline_metrics(
    endpoint: str,
    total_ms: float,
    llm_provider: str = "gemini",
    tts_backend: str = "edge_tts",
    stt_ms: float = 0.0,
    llm_ms: float = 0.0,
    tts_ms: float = 0.0,
    retrieval_ms: float = 0.0,
    cache_hit: bool = False,
    status: int = 200,
) -> None:
    """Convenience helper — call once per request with stage timings."""
    REQUEST_COUNT.labels(endpoint=endpoint, status=str(status)).inc()
    TOTAL_LATENCY.labels(endpoint=endpoint).observe(total_ms / 1000)
    if stt_ms:
        STT_LATENCY.observe(stt_ms / 1000)
    if llm_ms:
        LLM_LATENCY.labels(provider=llm_provider).observe(llm_ms / 1000)
    if tts_ms:
        TTS_LATENCY.labels(backend=tts_backend).observe(tts_ms / 1000)
    if retrieval_ms:
        RETRIEVAL_LATENCY.observe(retrieval_ms / 1000)
    if cache_hit:
        CACHE_HITS.inc()
    else:
        CACHE_MISSES.inc()
