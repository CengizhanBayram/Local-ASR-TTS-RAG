"""
Piper TTS Türkçe model indirme scripti.
Çalıştır: python download_models.py
"""

import urllib.request
import os

MODELS_DIR = "models"
BASE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main"
    "/tr/tr_TR/dfki/medium"
)
FILES = [
    "tr_TR-dfki-medium.onnx",
    "tr_TR-dfki-medium.onnx.json",
]

os.makedirs(MODELS_DIR, exist_ok=True)

for filename in FILES:
    dest = os.path.join(MODELS_DIR, filename)
    if os.path.exists(dest):
        print(f"Already exists: {dest}")
        continue
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")

print("\nDone! Piper TTS model ready.")
