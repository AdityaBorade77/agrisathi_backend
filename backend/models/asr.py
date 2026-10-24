"""
asr.py — Automatic Speech Recognition
Marathi Audio → Marathi Text

Uses: ai4bharat/indic-conformer-600m-multilingual
Mode: local (loads model) or api (HF Inference API)
"""

import os
import io
import asyncio
import logging
import base64
import httpx
import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)

INFERENCE_MODE = os.getenv("INFERENCE_MODE", "api")
HF_TOKEN = os.getenv("HF_TOKEN", "")
DEVICE = os.getenv("DEVICE", "cpu")

ASR_MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
# Use router.huggingface.co (new Inference Providers endpoint — api-inference subdomain blocked on some networks)
HF_INFERENCE_URL = f"https://router.huggingface.co/hf-inference/models/{ASR_MODEL_ID}"

# ─── Local model (lazy loaded) ────────────────────────────────────────────────
_asr_model = None
_asr_processor = None


def _load_local_asr():
    """Load IndicConformer model locally (downloaded from HF Hub)."""
    global _asr_model, _asr_processor
    if _asr_model is not None:
        return
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForCTC
        logger.info("Loading ASR model locally: %s", ASR_MODEL_ID)
        _asr_processor = AutoProcessor.from_pretrained(
            ASR_MODEL_ID,
            token=HF_TOKEN or None,
            language="mr",
        )
        _asr_model = AutoModelForCTC.from_pretrained(
            ASR_MODEL_ID,
            token=HF_TOKEN or None,
        ).to(DEVICE)
        _asr_model.eval()
        logger.info("ASR model loaded successfully.")
    except Exception as e:
        logger.error("Failed to load ASR model: %s", e)
        raise


def _transcribe_local(audio_bytes: bytes) -> str:
    """Transcribe audio bytes using locally loaded model."""
    import torch
    _load_local_asr()

    # Decode audio
    audio_buffer = io.BytesIO(audio_bytes)
    audio_array, sample_rate = sf.read(audio_buffer)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    # Ensure mono
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    inputs = _asr_processor(
        audio_array,
        sampling_rate=sample_rate,
        return_tensors="pt",
        language="mr",
    ).to(DEVICE)

    with torch.no_grad():
        logits = _asr_model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    marathi_text = _asr_processor.batch_decode(predicted_ids)[0]
    return marathi_text.strip()


GOOGLE_STT_LANG_CODES = {"mr": "mr-IN", "hi": "hi-IN", "en": "en-IN"}


def _transcribe_api(audio_bytes: bytes, language: str = "mr") -> str:
    """Transcribe via SpeechRecognition (Google API) instead of HF Inference API."""
    import speech_recognition as sr
    import io
    import imageio_ffmpeg
    import subprocess
    import tempfile
    import os
    
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        
        # Write audio bytes to a temporary file because ffmpeg sometimes fails 
        # to parse WebM/MKV headers correctly from a non-seekable pipe (pipe:0).
        fd, temp_input_path = tempfile.mkstemp(suffix=".webm")
        with os.fdopen(fd, 'wb') as f:
            f.write(audio_bytes)
            
        try:
            command = [
                ffmpeg_exe,
                '-y',
                '-i', temp_input_path,
                '-ac', '1',
                '-ar', '16000',
                '-f', 'wav',
                'pipe:1'
            ]
            
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            wav_data, err_data = process.communicate()
            
            if process.returncode != 0:
                logger.error("FFmpeg conversion failed: %s", err_data.decode('utf-8', errors='ignore'))
                raise RuntimeError("FFmpeg conversion failed")
        finally:
            # Clean up the temporary file
            try:
                os.remove(temp_input_path)
            except OSError:
                pass
                
        wav_io = io.BytesIO(wav_data)
        
        r = sr.Recognizer()
        with sr.AudioFile(wav_io) as source:
            audio = r.record(source)
            
        google_lang = GOOGLE_STT_LANG_CODES.get(language, "mr-IN")
        marathi_text = r.recognize_google(audio, language=google_lang)
        return marathi_text
    except sr.UnknownValueError:
        return ""
    except Exception as e:
        logger.error("ASR API error: %s", e)
        raise RuntimeError(f"ASR API failed: {e}")


async def transcribe_marathi(audio_bytes: bytes, language: str = "mr") -> str:
    """
    Main entry: transcribe audio bytes to Marathi text.
    Handles both local and API modes.
    """
    if not audio_bytes:
        raise ValueError("Empty audio data received.")

    logger.info("Transcribing audio (%d bytes), mode=%s", len(audio_bytes), INFERENCE_MODE)

    # Run blocking work (ffmpeg, HTTP, model inference) off the event loop so
    # concurrent requests aren't serialized behind each other.
    if INFERENCE_MODE == "local":
        return await asyncio.to_thread(_transcribe_local, audio_bytes)
    else:
        return await asyncio.to_thread(_transcribe_api, audio_bytes, language)
