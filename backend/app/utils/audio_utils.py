"""
Audio Processing Utilities
Ses dosyası işleme yardımcı fonksiyonları
"""

import io
import wave
import struct
import base64
from typing import Optional, Tuple


class AudioProcessor:
    """
    Ses dosyası işleme utility'leri
    """
    
    # Desteklenen formatlar
    SUPPORTED_FORMATS = {'wav', 'webm', 'mp3', 'ogg'}
    
    # Azure Speech Service için gerekli format
    TARGET_SAMPLE_RATE = 16000
    TARGET_CHANNELS = 1
    TARGET_BITS_PER_SAMPLE = 16
    
    @staticmethod
    def validate_audio_format(audio_bytes: bytes) -> str:
        """
        Ses formatını tespit et
        
        Args:
            audio_bytes: Ses verisi
            
        Returns:
            Format string ('wav', 'webm', 'mp3', etc.)
        """
        # WAV header kontrolü
        if audio_bytes[:4] == b'RIFF' and audio_bytes[8:12] == b'WAVE':
            return 'wav'
        
        # WebM/Matroska header
        if audio_bytes[:4] == b'\x1a\x45\xdf\xa3':
            return 'webm'
        
        # MP3 header
        if audio_bytes[:3] == b'ID3' or audio_bytes[:2] == b'\xff\xfb':
            return 'mp3'
        
        # OGG header
        if audio_bytes[:4] == b'OggS':
            return 'ogg'
        
        return 'unknown'
    
    @staticmethod
    def get_wav_info(audio_bytes: bytes) -> dict:
        """
        WAV dosyası bilgilerini al
        """
        try:
            with io.BytesIO(audio_bytes) as audio_file:
                with wave.open(audio_file, 'rb') as wav:
                    return {
                        'channels': wav.getnchannels(),
                        'sample_rate': wav.getframerate(),
                        'bits_per_sample': wav.getsampwidth() * 8,
                        'duration_seconds': wav.getnframes() / wav.getframerate(),
                        'num_frames': wav.getnframes()
                    }
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def create_wav_header(
        num_samples: int,
        sample_rate: int = 16000,
        channels: int = 1,
        bits_per_sample: int = 16
    ) -> bytes:
        """
        WAV dosyası header'ı oluştur
        """
        bytes_per_sample = bits_per_sample // 8
        block_align = channels * bytes_per_sample
        byte_rate = sample_rate * block_align
        data_size = num_samples * block_align
        file_size = 36 + data_size
        
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',           # ChunkID
            file_size,         # ChunkSize
            b'WAVE',           # Format
            b'fmt ',           # Subchunk1ID
            16,                # Subchunk1Size (PCM)
            1,                 # AudioFormat (PCM = 1)
            channels,          # NumChannels
            sample_rate,       # SampleRate
            byte_rate,         # ByteRate
            block_align,       # BlockAlign
            bits_per_sample,   # BitsPerSample
            b'data',           # Subchunk2ID
            data_size          # Subchunk2Size
        )
        
        return header
    
    @staticmethod
    def pcm_to_wav(
        pcm_data: bytes,
        sample_rate: int = 16000,
        channels: int = 1,
        bits_per_sample: int = 16
    ) -> bytes:
        """
        Raw PCM verisini WAV formatına çevir
        """
        num_samples = len(pcm_data) // (bits_per_sample // 8)
        header = AudioProcessor.create_wav_header(
            num_samples, sample_rate, channels, bits_per_sample
        )
        return header + pcm_data
    
    @staticmethod
    def base64_to_bytes(base64_audio: str) -> bytes:
        """
        Base64 encoded audio'yu bytes'a çevir
        """
        # Data URL prefix'i temizle
        if ',' in base64_audio:
            base64_audio = base64_audio.split(',')[1]
        
        return base64.b64decode(base64_audio)
    
    @staticmethod
    def bytes_to_base64(audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
        """
        Audio bytes'ı data URL formatında Base64'e çevir
        """
        encoded = base64.b64encode(audio_bytes).decode('utf-8')
        return f"data:{mime_type};base64,{encoded}"
    
    @staticmethod
    def calculate_duration_from_base64(
        base64_audio: str,
        sample_rate: int = 16000,
        bits_per_sample: int = 16,
        channels: int = 1
    ) -> float:
        """
        Base64 audio'nun süresini hesapla (saniye)
        """
        audio_bytes = AudioProcessor.base64_to_bytes(base64_audio)
        
        # Header'ı çıkar (WAV için 44 byte)
        if audio_bytes[:4] == b'RIFF':
            audio_bytes = audio_bytes[44:]
        
        bytes_per_sample = bits_per_sample // 8
        num_samples = len(audio_bytes) // (bytes_per_sample * channels)
        
        return num_samples / sample_rate
    
    @staticmethod
    def normalize_audio(audio_bytes: bytes, target_amplitude: float = 0.8) -> bytes:
        """
        Ses seviyesini normalize et
        
        Args:
            audio_bytes: WAV formatında ses verisi
            target_amplitude: Hedef genlik (0.0 - 1.0)
            
        Returns:
            Normalize edilmiş WAV verisi
        """
        try:
            with io.BytesIO(audio_bytes) as audio_file:
                with wave.open(audio_file, 'rb') as wav:
                    params = wav.getparams()
                    frames = wav.readframes(wav.getnframes())
            
            # 16-bit signed samples
            samples = struct.unpack(f'<{len(frames)//2}h', frames)
            
            # Max amplitude bul
            max_amp = max(abs(s) for s in samples)
            
            if max_amp == 0:
                return audio_bytes
            
            # Normalize et
            scale = (32767 * target_amplitude) / max_amp
            normalized = [int(s * scale) for s in samples]
            normalized = [max(-32768, min(32767, s)) for s in normalized]
            
            # Yeni WAV oluştur
            normalized_bytes = struct.pack(f'<{len(normalized)}h', *normalized)
            
            output = io.BytesIO()
            with wave.open(output, 'wb') as wav_out:
                wav_out.setparams(params)
                wav_out.writeframes(normalized_bytes)
            
            return output.getvalue()
            
        except Exception:
            # Hata durumunda orijinal veriyi döndür
            return audio_bytes
