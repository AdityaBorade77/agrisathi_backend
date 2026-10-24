"""
translation.py — Bidirectional Translation
Marathi ↔ English using AI4Bharat IndicTrans2

Models:
  Marathi→English: ai4bharat/indictrans2-indic-en-dist-200M
  English→Marathi: ai4bharat/indictrans2-en-indic-dist-200M
"""

import re
import os
import asyncio
import logging
import httpx

from utils.text_cleaner import clean_for_translation

logger = logging.getLogger(__name__)

INFERENCE_MODE = os.getenv("INFERENCE_MODE", "api")
HF_TOKEN = os.getenv("HF_TOKEN", "")
DEVICE = os.getenv("DEVICE", "cpu")

MODEL_MR_TO_EN = "ai4bharat/indictrans2-indic-en-dist-200M"
MODEL_EN_TO_MR = "ai4bharat/indictrans2-en-indic-dist-200M"

# ─── IndicTrans2 language codes ───────────────────────────────────────────────
# IndicTrans2 uses ISO 639-1 BCP-47 style codes
LANG_MR = "mar_Deva"  # Marathi Devanagari
LANG_EN = "eng_Latn"  # English Latin

# ─── Local model (lazy loaded) ────────────────────────────────────────────────
_models = {}
_tokenizers = {}
_ip = None


def _load_local_translation(direction: str):
    """direction: 'mr2en' or 'en2mr'"""
    if direction in _models:
        return
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        model_id = MODEL_MR_TO_EN if direction == "mr2en" else MODEL_EN_TO_MR
        logger.info("Loading translation model: %s", model_id)
        _tokenizers[direction] = AutoTokenizer.from_pretrained(
            model_id, token=HF_TOKEN or None, trust_remote_code=True
        )
        _models[direction] = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, token=HF_TOKEN or None, trust_remote_code=True
        ).to(DEVICE)
        _models[direction].eval()
        logger.info("Translation model %s loaded.", direction)
    except Exception as e:
        logger.error("Failed to load translation model (%s): %s", direction, e)
        raise


def _translate_local(text: str, direction: str) -> str:
    """Translate using locally loaded IndicTrans2 model."""
    global _ip
    import torch
    from IndicTransToolkit import IndicProcessor
    if _ip is None:
        _ip = IndicProcessor(inference=True)

    _load_local_translation(direction)

    src_lang = LANG_MR if direction == "mr2en" else LANG_EN
    tgt_lang = LANG_EN if direction == "mr2en" else LANG_MR

    tokenizer = _tokenizers[direction]
    model = _models[direction]

    batch = _ip.preprocess_batch([text], src_lang=src_lang, tgt_lang=tgt_lang)
    inputs = tokenizer(batch, return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang) if not hasattr(tokenizer, 'lang_code_to_id') else tokenizer.lang_code_to_id[tgt_lang],
            max_length=256,
        )

    translated = tokenizer.batch_decode(generated, skip_special_tokens=True)
    postprocessed = _ip.postprocess_batch(translated, lang=tgt_lang)
    return postprocessed[0].strip()


GOOGLE_LANG_CODES = {"mr": "mr", "hi": "hi", "en": "en"}
MYMEMORY_LANG_CODES = {"mr": "mr-IN", "hi": "hi-IN", "en": "en-US"}

# Sentence-ending punctuation, including the Devanagari danda/double danda.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!।॥])\s+")


def _translate_mymemory_sentences(text: str, source: str, target: str) -> str:
    """Translate sentence by sentence via MyMemory, each piece capped at 400
    chars, instead of truncating the whole answer to ~450 chars. MyMemory's
    free tier is unreliable on long single requests; this keeps every
    sentence of a long answer instead of silently dropping the tail."""
    from deep_translator import MyMemoryTranslator

    mm_src = MYMEMORY_LANG_CODES.get(source, source)
    mm_tgt = MYMEMORY_LANG_CODES.get(target, target)
    translator = MyMemoryTranslator(source=mm_src, target=mm_tgt)

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()] or [text]
    translated = [translator.translate(s[:400]) or s[:400] for s in sentences]
    return " ".join(translated)


def _translate_api_generic(text: str, source: str, target: str, google_only: bool = False) -> str:
    """Translate via Google Translate (one request, up to 5000 chars).

    Falls back to MyMemory unless `google_only` is set, in which case a
    Google failure raises instead of falling back. source/target are 'mr',
    'hi' or 'en'.
    """
    from deep_translator import GoogleTranslator

    g_src = GOOGLE_LANG_CODES.get(source, source)
    g_tgt = GOOGLE_LANG_CODES.get(target, target)
    try:
        result = GoogleTranslator(source=g_src, target=g_tgt).translate(text[:4900]) or text
        logger.info("Translation engine=google %s→%s: %.60s...", source, target, result)
        return result
    except Exception as e:
        if google_only:
            logger.warning("Google translation failed (%s); google_only=True, not falling back", e)
            raise RuntimeError(f"Google translation failed: {e}")
        logger.warning("Google translation failed (%s); falling back to MyMemory", e)

    try:
        result = _translate_mymemory_sentences(text, source, target)
        logger.info("Translation engine=mymemory %s→%s: %.60s...", source, target, result)
        return result
    except Exception as e:
        logger.error("Translation API error: %s", e)
        raise RuntimeError(f"Translation API failed: {e}")


async def translate(text: str, source: str, target: str, google_only: bool = False) -> str:
    """Generic translation entry point. source/target: 'mr', 'hi' or 'en'.

    API mode: Google Translate first, MyMemory fallback — works for any pair.
    Local mode: only mr<->en is backed by a loaded IndicTrans2 model, so any
    pair involving 'hi' still falls through to API mode even when
    INFERENCE_MODE=local (there is no local Hindi model wired up).
    google_only=True raises instead of using MyMemory when Google fails.
    """
    if not text or not text.strip():
        return ""
    if source == target:
        return text.strip()

    # The English side is usually LLM output, which can contain markdown or
    # emojis that would confuse the translator — clean it before sending.
    cleaned_text = clean_for_translation(text) if source == "en" else text.strip()
    if not cleaned_text.strip():
        cleaned_text = text.strip()

    logger.info("Translating %s→%s: %.60s...", source, target, cleaned_text)

    if INFERENCE_MODE == "local" and {source, target} == {"mr", "en"}:
        direction = "mr2en" if source == "mr" else "en2mr"
        return await asyncio.to_thread(_translate_local, cleaned_text, direction)

    return await asyncio.to_thread(_translate_api_generic, cleaned_text, source, target, google_only)


async def translate_marathi_to_english(marathi_text: str) -> str:
    """Translate Marathi text → English text."""
    return await translate(marathi_text, "mr", "en")


async def translate_english_to_marathi(english_text: str) -> str:
    """Translate English text → Marathi text."""
    return await translate(english_text, "en", "mr")
