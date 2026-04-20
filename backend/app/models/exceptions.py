"""
Custom exceptions for the application
Özel hata sınıfları
"""


class VoiceAIException(Exception):
    """Base exception for Voice AI application"""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class SpeechServiceError(VoiceAIException):
    """Azure Speech Service hataları"""
    def __init__(self, message: str):
        super().__init__(f"Speech Service Error: {message}", status_code=503)


class TranscriptionError(SpeechServiceError):
    """Ses tanıma hataları"""
    def __init__(self, message: str = "Ses tanınamadı"):
        super().__init__(message)


class SynthesisError(SpeechServiceError):
    """Ses sentezi hataları"""
    def __init__(self, message: str = "Ses sentezlenemedi"):
        super().__init__(message)


class DocumentProcessingError(VoiceAIException):
    """Belge işleme hataları"""
    def __init__(self, message: str):
        super().__init__(f"Document Processing Error: {message}", status_code=400)


class UnsupportedFileTypeError(DocumentProcessingError):
    """Desteklenmeyen dosya türü"""
    def __init__(self, file_type: str):
        super().__init__(f"Desteklenmeyen dosya türü: {file_type}")


class RAGError(VoiceAIException):
    """RAG sistemi hataları"""
    def __init__(self, message: str):
        super().__init__(f"RAG Error: {message}", status_code=500)


class NoDocumentsError(RAGError):
    """Belge bulunamadı hatası"""
    def __init__(self):
        super().__init__("Hiç belge yüklenmemiş. Lütfen önce belge yükleyin.")


class LLMError(VoiceAIException):
    """LLM hataları"""
    def __init__(self, message: str):
        super().__init__(f"LLM Error: {message}", status_code=503)


class ConfigurationError(VoiceAIException):
    """Konfigürasyon hataları"""
    def __init__(self, message: str):
        super().__init__(f"Configuration Error: {message}", status_code=500)
