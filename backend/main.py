"""
main.py — FastAPI Backend
Marathi-English Speech Translator PWA

Routes:
  POST /api/speech-to-text   Audio → Marathi text + English translation
  POST /api/text-to-speech   English text → Marathi audio + word timings
  POST /api/disease          Leaf photo + crop → disease_api (port 5002), translated to a farmer message
  GET  /api/health           Health check
"""

import os
import json
import asyncio
import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Bypass WDAC blocking sklearn DLLs (sparsefuncs_fast.pyd)
import sys
from unittest.mock import MagicMock
sys.modules['sklearn.utils.sparsefuncs_fast'] = MagicMock()

# Import model modules
from models.asr import transcribe_marathi
from models.translation import translate, translate_marathi_to_english, translate_english_to_marathi
from models.tts import synthesize_marathi_speech
from models.db import init_db, create_chat, get_chats, get_chat_messages, add_message, delete_chat
from models.sarvam_client import (
    sarvam_chat,
    SarvamCapReachedError,
    SarvamHTTPError,
    SarvamNotConfiguredError,
)
from utils.text_cleaner import clean_for_translation

# ─── LLM Config ───────────────────────────────────────────────────────────────

# Third-party models on NVIDIA's free tier (DeepSeek, Gemma, GLM, Kimi) are
# often queued for minutes; NVIDIA's own Nemotron models respond in seconds.
LLM_MODEL = os.getenv("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "")
LLM_HEDGE_AFTER = float(os.getenv("LLM_HEDGE_AFTER", "8"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "40"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))
# "direct": no translation, LLM answers mr/hi/en directly (default, demo-safe).
# "split": translate to English, LLM answers English-only, translate back.
CHAT_MODE = os.getenv("CHAT_MODE", "direct")

# Pipeline B2 for mr/hi chat (en is always Nemotron-direct, unaffected by
# this): Nemotron answers the farmer's original mr/hi question in English,
# then sarvam-105b rewrites that English answer into natural mr/hi. "b2"
# (default) tries this; "nemotron" switches it off entirely and chat behaves
# exactly as before this phase (today's plain Nemotron-direct, _chat_direct).
CHAT_LLM_PROVIDER = os.getenv("CHAT_LLM_PROVIDER", "b2")
# Nemotron's English-answer step is capped at this many seconds (its own
# hedge-to-fallback-model logic runs inside this budget, same hedge pattern
# as _chat_direct/ _crop_llm_complete).
CHAT_NEMOTRON_TIMEOUT_S = float(os.getenv("CHAT_NEMOTRON_TIMEOUT_S", "8"))
# The mr/hi rewrite step (sarvam-105b) is capped at this many seconds,
# unless less than this is left of CHAT_TOTAL_TIMEOUT_S.
CHAT_REWRITE_TIMEOUT_S = float(os.getenv("CHAT_REWRITE_TIMEOUT_S", "4"))
# Whole b2 chain (Nemotron English answer + sarvam rewrite, or the pipeline C
# fallback) is capped at this many seconds total before giving up and
# falling back to today's existing plain-Nemotron-direct behaviour.
CHAT_TOTAL_TIMEOUT_S = float(os.getenv("CHAT_TOTAL_TIMEOUT_S", "14"))

# ─── Disease detection (forwards to Shubham's separate Flask service) ─────────

DISEASE_API_URL = os.getenv("DISEASE_API_URL", "http://127.0.0.1:5002")
DISEASE_NAMES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "disease_names.json")

_disease_names: Optional[dict] = None


def _load_disease_names() -> dict:
    """Load backend/data/disease_names.json once and cache it in memory."""
    global _disease_names
    if _disease_names is None:
        with open(DISEASE_NAMES_PATH, encoding="utf-8") as f:
            _disease_names = json.load(f)
    return _disease_names


def _disease_label(class_name: str, language: str) -> str:
    """Checked name for one of disease_api's 27 classes, e.g. 'Onion___Purple_Blotch'."""
    entry = _load_disease_names()["diseases"].get(class_name, {})
    if language != "en":
        lang_entry = entry.get(language)
        if lang_entry:
            return lang_entry["name"]
    return entry.get("en") or class_name.split("___", 1)[-1].replace("_", " ")


def _crop_label(crop: str, language: str) -> str:
    entry = _load_disease_names()["crops"].get(crop, {})
    if language != "en":
        lang_entry = entry.get(language)
        if lang_entry:
            return lang_entry["name"]
    return entry.get("en") or crop


def _disease_plural(class_name: str, language: str) -> bool:
    """Whether this class's name in this language is a plural noun (default False)."""
    if language == "en":
        return False
    entry = _load_disease_names()["diseases"].get(class_name, {})
    lang_entry = entry.get(language)
    return bool(lang_entry and lang_entry.get("plural", False))


def _disease_category(class_name: str) -> str:
    """fungal / bacterial / viral / healthy, from disease_names.json."""
    return _load_disease_names()["categories"].get(class_name, "healthy")


def _category_precautions(category: str, language: str) -> list:
    """General precaution lines for a disease category, in language (mr/hi unreviewed)."""
    entry = _load_disease_names()["precautions_by_category"].get(category)
    if not entry:
        return []
    if language == "en":
        return entry.get("en", [])
    lang_entry = entry.get(language)
    return lang_entry.get("lines", []) if lang_entry else entry.get("en", [])


def _precautions_title(language: str) -> str:
    titles = _load_disease_names().get("precautions_title", {})
    return titles.get(language) or titles.get("en", "Precautions")


def _build_disease_message(result: dict, language: str) -> str:
    """
    Build the farmer-facing message from disease_api's structured fields
    (status, top[0], top[1], is_healthy) using the checked name table and
    fixed templates — never by machine-translating disease_api's English
    `message`, since a wrong disease name could mislead a farmer (CLAUDE.md).

    The KVK note is a fixed, pre-translated sentence per language, appended
    whenever disease_api sends a non-empty `note` — disease_api currently
    only ever sends one note (for Onion), so this is keyed on presence, not
    on the note's actual English text. If disease_api starts sending other
    crop-specific notes with different meanings, this will need a real
    per-note translation table instead of one fixed sentence.
    """
    templates = _load_disease_names()["templates"][language]
    top = result["top"]
    best = top[0]

    if result["status"] == "not_sure":
        return templates["not_sure"]

    crop_label = _crop_label(result["crop"], language)
    if best["is_healthy"]:
        message = templates["healthy"].format(crop=crop_label)
    else:
        disease_label = _disease_label(best["class"], language)
        fmt_kwargs = {"crop": crop_label, "disease": disease_label}
        if "copula_singular" in templates:
            plural = _disease_plural(best["class"], language)
            fmt_kwargs["copula"] = templates["copula_plural"] if plural else templates["copula_singular"]

        if len(top) > 1 and top[1]["probability"] >= 0.25:
            second = templates["healthy_word"] if top[1]["is_healthy"] else _disease_label(top[1]["class"], language)
            message = templates["disease_two"].format(**fmt_kwargs, second=second)
        else:
            message = templates["disease_one"].format(**fmt_kwargs)

    if result.get("note"):
        message += " " + templates["kvk_note"]
    return message

FARMER_SYSTEM_PROMPT = (
    "You are Krishi Mitra, an agricultural advisor for small farmers in India. "
    "Always reply in {language} using simple, everyday words a farmer understands, "
    "like an experienced local farmer advisor talking to a farmer, not a textbook. "
    "Answer as exactly 5 points, each about 15-25 words (about 100 words "
    "total), on its own line starting with \"• \" "
    "(bullet, then a space) and ending with a full stop. Plain text only: "
    "no headings, no markdown, no numbering, no emojis, no tables. "
    "When the question is about a pest or disease problem, weave in -- in "
    "whatever order sounds most natural, not as a fixed template -- the "
    "likely pest or disease and the damage it causes, what exactly to "
    "check on the plant and when, a non-chemical action specific to this "
    "pest or disease (for example hand-picking larvae, removing the "
    "damaged parts), and monitoring or prevention specific to it (for "
    "example traps, pruning, rotation); if a spray is genuinely needed, "
    "name a suitable active ingredient with \"dose as on the label\". At "
    "least 3 of the 5 points must be specific to this exact pest or "
    "disease, not general farm hygiene such as spacing or drainage, "
    "unless that genuinely matters for this problem. Name the pest or "
    "disease by the common local name a farmer would recognize, never a "
    "scientific or Latin pathogen name or other textbook term -- chemical, "
    "fertilizer, and product names (for example sulfur, potassium, "
    "mancozeb, imidacloprid) are fine to use as usual. "
    "Be specific to THIS crop and THIS problem: name the likely cause from the "
    "symptoms described, and give crop-specific steps and timing -- not "
    "generic advice. Do not add a stock line (such as 'keep proper spacing' "
    "or 'contact KVK') unless it genuinely applies here. If the farmer's "
    "question does not say the crop's growth stage, just give advice that "
    "fits the likely stage -- never write a sentence announcing an "
    "assumption, guess, or missing detail (for example, never say "
    "\"I assume...\", \"गृहीत धरले...\", or \"you mentioned X but not Y\"); "
    "never say you assumed the crop or repeat it back as if it were unclear. "
    "Give non-chemical steps first: removing and destroying infected leaves, "
    "proper plant spacing, good drainage, crop rotation. If a spray is truly "
    "needed, you may name the active ingredient (for example, mancozeb), but "
    "only one registered for use in India, never a banned one -- and never "
    "give a dose, quantity, or mixing ratio, since a wrong amount can harm "
    "the crop or the farmer: say to use the dose on the product label. Seed "
    "rate, spacing, and fertilizer amounts may be given as rough, "
    "approximate figures, and you should suggest checking the farmer's Soil "
    "Health Card for exact fertilizer doses. For market-price questions, "
    "point the farmer to their local APMC market or the e-NAM portal rather "
    "than quoting a price yourself. If the question is not about farming, "
    "politely decline and say you can only help with farming questions. "
    "Your reply will be read aloud by a text-to-speech engine after the "
    "bullet markers are removed, so keep each point a short, plain sentence."
)

ORIGINAL_LANGUAGE_LABELS = {"mr": "Marathi", "hi": "Hindi"}

# Used by CHAT_MODE=direct to fill FARMER_SYSTEM_PROMPT's {language}.
LANGUAGE_NAMES = {"mr": "Marathi", "hi": "Hindi", "en": "English"}


def _add_full_stops_to_bullets(text: str) -> str:
    """Add a closing full stop to any "• " point that doesn't already
    end with one -- the prompt asks the LLM for this, but models don't
    always comply, and without it the points run together in the chat
    bubble and text-to-speech gets no pause between them. Always adds a
    plain period; an existing Devanagari danda ("।") still counts as
    already punctuated, so it is never doubled up."""
    if not text:
        return text
    end_chars = (".", "?", "!", "।")
    lines = text.split("\n")
    for i, line in enumerate(lines):
        trimmed = line.rstrip()
        if trimmed.lstrip().startswith("•") and trimmed and not trimmed.endswith(end_chars):
            trimmed = trimmed + "."
        lines[i] = trimmed
    return "\n".join(lines)

# ─── Farm glossary (fixed mr/hi terms, see CLAUDE.md on disease names) ────────

FARM_GLOSSARY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "farm_glossary.json")
_farm_glossary: Optional[dict] = None
_glossary_word_cache: dict[tuple[str, str], str] = {}

# Standard \w doesn't count Devanagari vowel signs/matras as word characters,
# so a plain \bword\b would wrongly match inside a longer word (e.g. "पान"
# inside "पानांची"). This class treats the whole Devanagari block as "word".
_DEVANAGARI_WORD_CHAR = r"[\wऀ-ॿ]"


def _load_farm_glossary() -> dict:
    """Load backend/data/farm_glossary.json once and cache it in memory."""
    global _farm_glossary
    if _farm_glossary is None:
        with open(FARM_GLOSSARY_PATH, encoding="utf-8") as f:
            _farm_glossary = json.load(f)
    return _farm_glossary


def _whole_word_present(word: str, text: str) -> bool:
    """True if `word` is in `text` as a separate word, not as a substring of
    a longer word."""
    pattern = rf"(?<!{_DEVANAGARI_WORD_CHAR}){re.escape(word)}(?!{_DEVANAGARI_WORD_CHAR})"
    return re.search(pattern, text) is not None


def _translated_glossary_word(term: str, language: str) -> Optional[str]:
    """Read-only cache lookup. The cache is filled once at server startup by
    _prefill_glossary_cache() — never during a request, so a slow or failing
    translation call can't add latency or risk to /api/chat."""
    return _glossary_word_cache.get((term, language))


async def _prefill_glossary_cache():
    """Fill _glossary_word_cache in the background after startup, one
    glossary term every 0.5s via Google Translate only (google_only=True —
    no MyMemory fallback here). A failed lookup is simply left out of the
    cache; _apply_farm_glossary then skips that term instead of retrying
    mid-request."""
    glossary = _load_farm_glossary()
    for language, terms in glossary.items():
        for term in terms:
            try:
                word = (await translate(term, "en", language, google_only=True)).strip().rstrip(".।")
                if word:
                    _glossary_word_cache[(term, language)] = word
                    logger.info("Glossary cache: %s/%s -> %s", term, language, word)
            except Exception as e:
                logger.warning("Glossary cache: %s/%s failed (%s); term will be skipped", term, language, e)
            await asyncio.sleep(0.5)


async def _apply_farm_glossary(translated_text: str, english_text: str, language: str) -> str:
    """
    Swap the general-purpose translator's wording for our fixed farm terms.

    For every English glossary term that appears in the LLM's English answer,
    translate that single word into `language` and see what word the
    translator chose. If it differs from our fixed glossary word, replace it
    in the translated text (whole-word only, so it can't corrupt a longer
    word that happens to contain the same letters).
    """
    terms = _load_farm_glossary().get(language)
    if not terms:
        return translated_text

    result = translated_text
    english_lower = english_text.lower()
    for term, fixed_word in terms.items():
        if not fixed_word or _whole_word_present(fixed_word, result):
            continue
        if not re.search(rf"\b{re.escape(term)}s?\b", english_lower):
            continue
        translator_word = _translated_glossary_word(term, language)
        if translator_word is None:
            continue
        if translator_word and translator_word != fixed_word:
            pattern = rf"(?<!{_DEVANAGARI_WORD_CHAR}){re.escape(translator_word)}(?!{_DEVANAGARI_WORD_CHAR})"
            result = re.sub(pattern, fixed_word, result)
    return result

_llm_client: Optional[AsyncOpenAI] = None

# ─── App Setup ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info("🚀 Marathi-English Speech Translator API starting...")
    init_db()
    logger.info("Inference mode: %s", os.getenv("INFERENCE_MODE", "api"))
    logger.info("TTS Voice: %s", os.getenv("TTS_VOICE", "Sunita"))
    yield
    logger.info("API shutting down.")


app = FastAPI(
    title="Marathi-English Speech Translator",
    description="PWA backend for Marathi ASR + IndicTrans2 + Indic Parler-TTS",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend to call the API
frontend_origin = os.getenv("FRONTEND_ORIGIN", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin] if frontend_origin != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class TextToSpeechRequest(BaseModel):
    english_text: Optional[str] = None
    # When the text is already Marathi (e.g. the LLM replied in Marathi),
    # send it here to skip the EN→MR translation step.
    marathi_text: Optional[str] = None
    # Generic path used by the frontend's language selector — skips
    # translation entirely and speaks `text` directly in `language`.
    text: Optional[str] = None
    language: Optional[str] = None
    voice: Optional[str] = None


class SpeechToTextResponse(BaseModel):
    marathi_text: str
    english_text: str
    success: bool = True


class ChatRequest(BaseModel):
    text: str
    reply_language: str = "mr"


class ChatResponse(BaseModel):
    response: str
    response_en: str
    question_en: str
    timings: dict
    translation_failed: bool = False
    success: bool = True


class TextToSpeechResponse(BaseModel):
    marathi_text: str
    audio_base64: str
    word_timings: list
    duration: float
    voice: str
    success: bool = True

class DiseaseTopGuess(BaseModel):
    class_: str = Field(alias="class")
    disease: str
    disease_en: str
    probability: float

    model_config = {"populate_by_name": True}


class DiseaseExplanation(BaseModel):
    cause: str
    today: str
    prevent: str


class DiseaseResponse(BaseModel):
    crop: str
    status: str
    message: str
    is_healthy: bool
    language: str
    note: str = ""
    precautions: list = []
    precautions_title: str = ""
    top: list[DiseaseTopGuess] = []
    # Set only when status=="ok" and the top guess isn't healthy. Always
    # fully populated then (LLM or rules fallback) -- see
    # _get_disease_explanation.
    explanation: Optional[DiseaseExplanation] = None
    # Which LLM produced it ("sarvam"/"nemotron"), or None for the rules
    # fallback. For logs/debug only.
    explanation_provider: Optional[str] = None


class CropRequest(BaseModel):
    taluka: str
    sowing_month: int
    irrigation: str
    previous_crop: Optional[str] = None
    soil_type: Optional[str] = None
    soil: Optional[dict] = None
    language: str = "mr"


class CropEvidenceRef(BaseModel):
    page: Optional[int] = None
    section: Optional[str] = None


class CropRecommendationOut(BaseModel):
    id: str
    name: str
    reason: str
    risk: str
    sowing_months: Optional[list] = None
    evidence_refs: list[CropEvidenceRef] = []


class CropResponse(BaseModel):
    success: bool = True
    language: str
    source: str  # "llm" or "rules"
    crops: list[CropRecommendationOut] = []
    weather: Optional[dict] = None
    disclaimer: str
    # Only set for the no-recommendations case (crops is then empty).
    message: Optional[str] = None
    # Which LLM actually produced "crops" when source=="llm" ("sarvam" or
    # "nemotron"); None for source=="rules". For logs/debug only -- the
    # frontend doesn't need to show this.
    provider: Optional[str] = None


class CreateChatRequest(BaseModel):
    title: str

class AddMessageRequest(BaseModel):
    role: str
    msg_type: str
    content: dict

# ─── Routes ───────────────────────────────────────────────────────────────────

async def _wake_service(base_url: str):
    """Send one GET to another service's /health so it starts waking up.
    On free hosting (Render) crop_api and disease_api sleep when idle and
    take about a minute to start; the frontend calls /api/health when the
    app opens, so they are usually awake by the time the farmer needs them.
    Errors are ignored: this is only a nudge."""
    try:
        async with httpx.AsyncClient(timeout=90) as client:
            await client.get(f"{base_url}/health")
    except Exception:
        pass


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    # Fire and forget: don't make the health check wait for them
    if os.getenv("WAKE_SERVICES_ON_HEALTH", "false").lower() == "true":
        asyncio.create_task(_wake_service(CROP_API_URL))
        asyncio.create_task(_wake_service(DISEASE_API_URL))
    return {
        "status": "ok",
        "inference_mode": os.getenv("INFERENCE_MODE", "api"),
        "tts_voice": os.getenv("TTS_VOICE", "Sunita"),
        "device": os.getenv("DEVICE", "cpu"),
        "hf_token_set": bool(os.getenv("HF_TOKEN")),
    }


@app.post("/api/speech-to-text", response_model=SpeechToTextResponse)
async def speech_to_text(audio: UploadFile = File(...), translate: bool = Form(True), language: str = Form("mr")):
    """
    Convert Marathi speech audio to English text.

    Pipeline:
      Audio (WebM/WAV/OGG) → IndicConformer → Marathi Text → IndicTrans2 → English Text
    """
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided.")

    logger.info("Received audio: %s (%s)", audio.filename, audio.content_type)

    try:
        audio_bytes = await audio.read()

        if len(audio_bytes) < 100:
            raise HTTPException(status_code=400, detail="Audio file is too small or empty.")

        # Step 1: Transcribe Marathi speech → Marathi text
        marathi_text = await transcribe_marathi(audio_bytes, language)
        logger.info("Transcribed: %s", marathi_text[:100])

        if not marathi_text.strip():
            return SpeechToTextResponse(
                marathi_text="",
                english_text="(No speech detected)",
                success=True,
            )

        # Step 2: Translate Marathi → English (optional — the LLM understands
        # Marathi directly, so the voice-chat flow skips this hop)
        english_text = ""
        if translate:
            english_text = await translate_marathi_to_english(marathi_text)
            logger.info("Translated: %s", english_text[:100])

        return SpeechToTextResponse(
            marathi_text=marathi_text,
            english_text=english_text,
        )

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error("Model error in speech-to-text: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in speech-to-text")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/api/text-to-speech", response_model=TextToSpeechResponse)
async def text_to_speech(request: TextToSpeechRequest):
    """
    Convert English text to Marathi speech audio.

    Pipeline:
      English Text → IndicTrans2 → Marathi Text → Indic Parler-TTS → Audio + Word Timings
    """
    source_text = (request.marathi_text or request.english_text or "").strip()
    if request.text:
        source_text = request.text.strip()

    if not source_text:
        raise HTTPException(status_code=400, detail="No text provided.")

    if len(source_text) > 2000:
        raise HTTPException(status_code=400, detail="Text too long (max 2000 chars).")

    logger.info("TTS request: %s", source_text[:100])

    try:
        # Step 1: Translate English → Marathi (skipped when Marathi was given,
        # or always skipped on the generic text+language path — the LLM
        # already answered in the target language, so nothing to translate).
        if request.text:
            marathi_text = clean_for_translation(source_text) or source_text
        elif request.marathi_text:
            marathi_text = clean_for_translation(source_text) or source_text
        else:
            marathi_text = await translate_english_to_marathi(source_text)
            logger.info("Translated to Marathi: %s", marathi_text[:100])

        if not marathi_text.strip():
            raise HTTPException(status_code=500, detail="Translation returned empty text.")

        # Step 2: Synthesize Marathi speech
        tts_result = await synthesize_marathi_speech(
            marathi_text, voice=request.voice, language=(request.language or "mr")
        )

        return TextToSpeechResponse(
            marathi_text=marathi_text,
            audio_base64=tts_result["audio_base64"],
            word_timings=tts_result["word_timings"],
            duration=tts_result["duration"],
            voice=tts_result["voice"],
        )

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error("Model error in text-to-speech: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in text-to-speech")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/api/disease", response_model=DiseaseResponse)
async def disease_check(
    image: UploadFile = File(...),
    crop: str = Form(...),
    language: str = Form("mr"),
):
    """
    Forward a leaf photo + crop to disease_api (a separate Flask service on
    port 5002 — see disease_api/app.py) and turn its structured result into
    a farmer-facing message in the requested language.
    """
    if not crop.strip():
        raise HTTPException(status_code=400, detail="Please choose a crop.")
    if not image.filename:
        raise HTTPException(status_code=400, detail="Please provide a leaf photo.")

    all_templates = _load_disease_names()["templates"]
    lang = language if language in all_templates else "en"

    image_bytes = await image.read()
    t_total_start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            disease_response = await client.post(
                f"{DISEASE_API_URL}/predict-disease",
                files={"image": (image.filename, image_bytes, image.content_type or "image/jpeg")},
                data={"crop": crop},
            )
    except httpx.ConnectError:
        logger.error(
            "disease_api unreachable at %s — start it with: cd disease_api && python app.py",
            DISEASE_API_URL,
        )
        raise HTTPException(status_code=503, detail=all_templates[lang]["service_unavailable"])
    except httpx.RequestError as e:
        logger.error("Error contacting disease_api: %s", e)
        raise HTTPException(status_code=503, detail=all_templates[lang]["service_unavailable"])

    if disease_response.status_code != 200:
        # Pass disease_api's own error body (bad crop, unreadable image, too large) straight through.
        return JSONResponse(status_code=disease_response.status_code, content=disease_response.json())

    result = disease_response.json()

    try:
        message = _build_disease_message(result, lang)
    except Exception:
        logger.exception("Unexpected error building disease message")
        raise HTTPException(status_code=500, detail="Could not build the result message.")

    top_guesses = [
        DiseaseTopGuess(**{
            "class": g["class"],
            "disease": _disease_label(g["class"], lang),
            "disease_en": _disease_label(g["class"], "en"),
            "probability": g["probability"],
        })
        for g in result["top"]
    ]
    # The returned note is our own translated KVK sentence, not disease_api's raw
    # English text — keyed on presence only (see _build_disease_message's docstring).
    note = all_templates[lang]["kvk_note"] if result.get("note") else ""

    precautions = []
    explanation = None
    explanation_provider = None
    llm_ms = 0
    if result["status"] == "ok":
        category = _disease_category(result["top"][0]["class"])
        if category != "healthy":
            precautions = _category_precautions(category, lang)
            t0 = time.perf_counter()
            explanation_dict, explanation_provider = await _get_disease_explanation(result, lang, category)
            llm_ms = round((time.perf_counter() - t0) * 1000)
            explanation = DiseaseExplanation(**explanation_dict)

    total_ms = round((time.perf_counter() - t_total_start) * 1000)
    logger.info(
        "Disease timings (ms): llm=%s total=%s provider=%s",
        llm_ms, total_ms, explanation_provider or "none",
    )

    return DiseaseResponse(
        crop=result["crop"],
        status=result["status"],
        message=message,
        is_healthy=result["top"][0]["is_healthy"],
        language=lang,
        note=note,
        precautions=precautions,
        precautions_title=_precautions_title(lang),
        top=top_guesses,
        explanation=explanation,
        explanation_provider=explanation_provider,
    )


def _get_llm_client() -> AsyncOpenAI:
    """Reuse one client so TCP/TLS connections to NVIDIA are kept alive between requests."""
    global _llm_client
    if _llm_client is None:
        _llm_client = AsyncOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY"),
            timeout=LLM_TIMEOUT,
            max_retries=0,
        )
    return _llm_client


def _llm_extra_body(model: str) -> dict:
    # Hybrid-reasoning models (DeepSeek, Nemotron) think by default; thinking
    # adds many seconds of latency and isn't needed for short advisory answers.
    if "deepseek" in model or "nemotron" in model:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}


async def _chat_direct(text: str, reply_language: str) -> ChatResponse:
    """CHAT_MODE=direct: no translation — the LLM sees the farmer's original
    text and is told to answer directly in mr/hi/en."""
    language_name = LANGUAGE_NAMES.get(reply_language, "English")
    system_prompt = FARMER_SYSTEM_PROMPT.format(language=language_name)
    if reply_language in ("mr", "hi"):
        system_prompt += (
            " Write only in Devanagari script. Do not use English letters. "
            "Use short, simple everyday words."
        )
    if reply_language == "mr":
        system_prompt += CHAT_MR_VOCAB_HINT

    logger.info(
        "LLM chat request (%s) CHAT_MODE=direct reply_language=%s question_en=%s",
        LLM_MODEL, reply_language, text[:100],
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    def complete(model: str):
        return asyncio.create_task(_get_llm_client().chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5,
            top_p=0.9,
            max_tokens=LLM_MAX_TOKENS,
            extra_body=_llm_extra_body(model),
        ))

    t_total_start = time.perf_counter()
    try:
        # Hedged request: if the primary model hasn't answered within
        # LLM_HEDGE_AFTER seconds (free-tier queueing), also ask the fallback
        # model and use whichever answers first.
        completion, failed = None, []
        pending = {complete(LLM_MODEL)}
        timeout = LLM_HEDGE_AFTER
        while completion is None:
            if not pending:
                raise failed[-1].exception()
            done, pending = await asyncio.wait(
                pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            for t in done:
                if t.exception():
                    failed.append(t)
                elif completion is None:
                    completion = t.result()
            if completion is None and timeout is not None and LLM_FALLBACK_MODEL:
                logger.warning("LLM %s slow or failed; also trying %s", LLM_MODEL, LLM_FALLBACK_MODEL)
                pending.add(complete(LLM_FALLBACK_MODEL))
            timeout = None  # hedge only once; then wait for whichever finishes
        for t in pending:
            t.cancel()

        response_text = (completion.choices[0].message.content or "").strip()
        response_text = _add_full_stops_to_bullets(response_text)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in chat")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    llm_ms = round((time.perf_counter() - t_total_start) * 1000)
    timings = {"translate_in": 0, "llm": llm_ms, "translate_out": 0, "total": llm_ms}
    logger.info("Chat timings (ms): %s", timings)

    return ChatResponse(
        response=response_text,
        response_en=response_text,
        question_en=text,
        timings=timings,
        translation_failed=False,
    )


# ─── Chat pipeline B2 (mr/hi only; CLAUDE.md "Chat LLM provider") ─────────────
# Nemotron answers the farmer's original mr/hi question in English (reusing
# FARMER_SYSTEM_PROMPT, same safety rules as _chat_direct), then sarvam-105b
# rewrites that English answer into natural mr/hi. Falls back to pipeline C
# (sarvam-105b answers directly in mr/hi) if Nemotron fails/is slow, or if
# the rewrite fails/is rejected and budget remains; falls back further to
# today's plain _chat_direct if nothing else worked in time. en is untouched
# — chat() below always sends it straight to _chat_direct, same as before.

# Local pest/disease words Nemotron has misread in practice when answering
# in English straight from the farmer's original mr/hi text (e.g. it read
# "आळी" as "aphid" instead of "caterpillar/larva" -- these don't all have a
# single correct English word, so a short hint in the prompt is cheaper and
# safer than trusting the model to already know them).
CHAT_PEST_GLOSSARY_HINT = (
    "Local pest/disease words the farmer may use, read them correctly: "
    "आळी/सुंडी = caterpillar or larva (e.g. bollworm); बोंडअळी = bollworm; "
    "मावा/माहू = aphid; तुडतुडे = jassid/leafhopper; फुलकिडे/थ्रिप्स = thrips; "
    "पांढरी माशी/सफेद मक्खी = whitefly; करपा = blight; भुरी = powdery mildew; "
    "डावणी = downy mildew."
)

CHAT_REWRITE_SYSTEM_PROMPT = (
    "Rewrite this farming advice in natural, simple {language} for a "
    "Nashik farmer. Keep every fact and number exactly. The text is a list "
    "of points, each starting with \"• \" (bullet, then a space) and "
    "ending with a full stop, on its own line -- keep exactly that many "
    "points, in the same order, each still starting with \"• \" and ending "
    "with a full stop, on its own line. Write all words, "
    "including product or active-ingredient names, in Devanagari script -- "
    "never add the English word in brackets alongside it. Use local "
    "farming words. Do not add or remove any information. Output only "
    "the rewritten text."
)

# Marathi-specific pest/disease vocabulary and phrasing mistakes seen in
# live testing: wrong word for a plant part, the same part named twice in
# one sentence, a wrong disposal method for damaged parts, or a diagnosis
# stated as certain. mr only -- matches this codebase's existing pattern
# (_apply_mr_word_fixes/mr_word_fixes.json) that Marathi output needs this
# extra correction and Hindi hasn't shown the same mistakes.
CHAT_MR_VOCAB_HINT = (
    " Use correct Marathi farmer words: square (cotton bud) is \"पाती\", "
    "flower is \"फुले\", boll is \"बोंड\", larva is \"अळी\", bollworm is "
    "\"बोंडअळी\", moth is \"पतंग\", leaf is \"पान\", vine is \"वेल\". Never "
    "name the same plant part twice in one sentence (for example, never "
    "\"बोंड आणि बोंडांना\" or \"बोंडे आणि कोवळी बोंडे\") -- name different "
    "plant parts instead. Damaged plant parts (leaves, squares, bolls, "
    "shoots) must be \"गोळा करून जाळा किंवा खोल गाडून टाका\"; only "
    "hand-picked larvae may be dropped into soapy water. For a diagnosis, "
    "say \"असू शकते\", never \"नक्की\". Use the common local disease name, "
    "never a scientific or Latin pathogen name or textbook term (never "
    "\"Alternaria porri\", \"Elsinoe ampelina\", or \"नॉन-अ‍ॅलियम\"): purple "
    "blotch is \"जांभळा करपा\", leaf curl virus is \"पर्णगुच्छ (बोकड्या)\", "
    "powdery mildew is \"भुरी\", downy mildew is \"डावणी\", anthracnose is "
    "\"करपा\". Crop rotation away from onion/garlic is \"कांदा-लसूण सोडून "
    "इतर पिकांची फेरपालट\". Chemical, fertilizer, and product names (for "
    "example सल्फर, पोटॅशियम, मॅन्कोझेब, इमिडाक्लोप्रिड), including the "
    "word \"डोस\", are fine to use as they are."
)

CHAT_MESSAGES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "chat_messages.json")
_chat_messages: Optional[dict] = None


def _load_chat_messages() -> dict:
    """Load backend/data/chat_messages.json once and cache it in memory."""
    global _chat_messages
    if _chat_messages is None:
        with open(CHAT_MESSAGES_PATH, encoding="utf-8") as f:
            _chat_messages = json.load(f)
    return _chat_messages


def _chat_fallback_message(reply_language: str) -> str:
    """Fixed mr/hi "couldn't answer right now" text (chat_messages.json,
    reviewed:false) used when pipeline B2 and pipeline C both fail/are
    unavailable -- see _chat_pipeline_b2. Never calls Nemotron again here:
    if Nemotron just failed or is slow, retrying it (_chat_direct's own long
    hedge/timeout) would only make the farmer wait longer for the same
    likely outcome."""
    return _load_chat_messages()["fallback_no_answer"].get(reply_language, "")


async def _chat_llm_complete(messages: list, total_timeout: float) -> str:
    """Hedged Nemotron call for chat's pipeline B2 English-answer step —
    same hedge pattern as _chat_direct (ask LLM_MODEL, also try
    LLM_FALLBACK_MODEL after LLM_HEDGE_AFTER seconds, use whichever answers
    first), but capped at `total_timeout` seconds total so the whole mr/hi
    chain stays inside CHAT_TOTAL_TIMEOUT_S. Raises asyncio.TimeoutError past
    that; the caller treats this as "Nemotron failed/slow" and moves on to
    pipeline C. Every task this call creates is cancelled and drained before
    returning (see _cancel_and_drain) so a losing hedge request never keeps
    running in the background."""
    client = _get_llm_client()
    all_tasks: list[asyncio.Task] = []

    def complete(model: str):
        task = asyncio.create_task(client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5,
            top_p=0.9,
            max_tokens=LLM_MAX_TOKENS,
            extra_body=_llm_extra_body(model),
        ))
        all_tasks.append(task)
        return task

    async def hedge():
        completion, failed = None, []
        pending = {complete(LLM_MODEL)}
        timeout = LLM_HEDGE_AFTER
        while completion is None:
            if not pending:
                raise failed[-1].exception()
            done, pending = await asyncio.wait(
                pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            for t in done:
                if t.exception():
                    failed.append(t)
                elif completion is None:
                    completion = t.result()
            if completion is None and timeout is not None and LLM_FALLBACK_MODEL:
                logger.warning("Chat LLM %s slow or failed; also trying %s", LLM_MODEL, LLM_FALLBACK_MODEL)
                pending.add(complete(LLM_FALLBACK_MODEL))
            timeout = None  # hedge only once; then wait for whichever finishes
        return completion

    try:
        completion = await asyncio.wait_for(hedge(), timeout=total_timeout)
        return (completion.choices[0].message.content or "").strip()
    finally:
        await _cancel_and_drain(all_tasks)


async def _chat_try_sarvam_direct(text: str, reply_language: str, timeout_s: float) -> Optional[str]:
    """Pipeline C: sarvam-105b answers the farmer's mr/hi question directly
    (same role/safety rules as _chat_direct's mr/hi prompt). Returns the raw
    answer text on success, or None -- after logging exactly why -- so the
    caller falls back to today's plain _chat_direct. Mirrors crop/disease's
    _crop_try_sarvam / _disease_try_sarvam."""
    language_name = LANGUAGE_NAMES.get(reply_language, "English")
    system_prompt = (
        FARMER_SYSTEM_PROMPT.format(language=language_name)
        + " Never give addresses, phone numbers, or office locations."
        + " Write only in Devanagari script. Do not use English letters. "
          "Use short, simple everyday words."
    )
    if reply_language == "mr":
        system_prompt += CHAT_MR_VOCAB_HINT
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
    try:
        raw_text = await sarvam_chat(messages, max_tokens=LLM_MAX_TOKENS, timeout_s=timeout_s)
    except SarvamNotConfiguredError:
        logger.info("Chat pipeline C unavailable (reason=not_configured)")
        return None
    except SarvamCapReachedError as e:
        logger.warning("Chat pipeline C unavailable (reason=cap_reached): %s", e)
        return None
    except SarvamHTTPError as e:
        logger.warning(
            "Chat pipeline C unavailable (reason=http_error status=%s): %s",
            e.status_code, e.body[:300],
        )
        return None
    except httpx.TimeoutException:
        logger.warning("Chat pipeline C unavailable (reason=timeout)")
        return None
    except Exception as e:
        logger.warning("Chat pipeline C unavailable (reason=error): %s", e)
        return None
    return raw_text.strip() or None


def _chat_mr_hi_valid(text: str, reply_language: str) -> bool:
    """True if `text` is non-empty and not wrong-language, using chat's
    relaxed 70%-Devanagari check (_chat_text_is_wrong_language) rather than
    crop/disease's stricter all-or-nothing rule."""
    return bool(text) and not _chat_text_is_wrong_language(text, reply_language)


async def _chat_pipeline_b2(text: str, reply_language: str) -> ChatResponse:
    """mr/hi only (chat() below routes en straight to _chat_direct). See the
    module comment above for the pipeline and its fallback order."""
    if CHAT_LLM_PROVIDER != "b2":
        return await _chat_direct(text, reply_language)

    language_name = LANGUAGE_NAMES.get(reply_language, "English")
    t_total_start = time.perf_counter()
    timings = {"nemotron_en_ms": 0, "rewrite_ms": 0, "pipeline_c_ms": 0, "total_ms": 0}

    # Step 1: Nemotron answers the original mr/hi question in English.
    english_answer = None
    t0 = time.perf_counter()
    try:
        system_prompt_en = (
            FARMER_SYSTEM_PROMPT.format(language="English")
            + " Never give addresses, phone numbers, or office locations."
            + " " + CHAT_PEST_GLOSSARY_HINT
        )
        messages = [{"role": "system", "content": system_prompt_en}, {"role": "user", "content": text}]
        answer = await _chat_llm_complete(messages, total_timeout=CHAT_NEMOTRON_TIMEOUT_S)
        english_answer = answer or None
    except Exception as e:
        logger.warning("Chat B2: Nemotron English step failed (%s)", e)
        english_answer = None
    timings["nemotron_en_ms"] = round((time.perf_counter() - t0) * 1000)

    # Step 2: sarvam-105b rewrites the English answer into natural mr/hi.
    # Budget is capped at whatever's left of CHAT_TOTAL_TIMEOUT_S -- no
    # minimum-floor override here (unlike crop/disease's shared helpers),
    # so this step can never push the whole chain past 14s.
    final_text = None
    if english_answer:
        elapsed = time.perf_counter() - t_total_start
        remaining = CHAT_TOTAL_TIMEOUT_S - elapsed
        if remaining > 0:
            rewrite_budget = min(CHAT_REWRITE_TIMEOUT_S, remaining)
            t0 = time.perf_counter()
            try:
                rewrite_system_prompt = CHAT_REWRITE_SYSTEM_PROMPT.format(language=language_name)
                if reply_language == "mr":
                    rewrite_system_prompt += CHAT_MR_VOCAB_HINT
                rewrite_messages = [
                    {"role": "system", "content": rewrite_system_prompt},
                    {"role": "user", "content": english_answer},
                ]
                rewritten = await sarvam_chat(rewrite_messages, max_tokens=LLM_MAX_TOKENS, timeout_s=rewrite_budget)
                rewritten = rewritten.strip()
                if _chat_mr_hi_valid(rewritten, reply_language):
                    final_text = _apply_mr_word_fixes(rewritten) if reply_language == "mr" else rewritten
                    final_text = _add_full_stops_to_bullets(final_text)
                else:
                    logger.warning("Chat B2: rewrite rejected (empty or wrong-language)")
            except SarvamNotConfiguredError:
                logger.info("Chat B2: rewrite unavailable (reason=not_configured)")
            except SarvamCapReachedError as e:
                logger.warning("Chat B2: rewrite unavailable (reason=cap_reached): %s", e)
            except SarvamHTTPError as e:
                logger.warning(
                    "Chat B2: rewrite unavailable (reason=http_error status=%s): %s",
                    e.status_code, e.body[:300],
                )
            except httpx.TimeoutException:
                logger.warning("Chat B2: rewrite timed out")
            except Exception as e:
                logger.warning("Chat B2: rewrite failed (%s)", e)
            timings["rewrite_ms"] = round((time.perf_counter() - t0) * 1000)

    path = "b2" if final_text else None

    # Step 3 (fallback): pipeline C, only if there's still budget left, and
    # capped at exactly what's left -- same reasoning as the rewrite step
    # above, so b2 + c together can never exceed CHAT_TOTAL_TIMEOUT_S.
    if final_text is None:
        elapsed = time.perf_counter() - t_total_start
        remaining = CHAT_TOTAL_TIMEOUT_S - elapsed
        if remaining > 0:
            t0 = time.perf_counter()
            pipeline_c_text = await _chat_try_sarvam_direct(text, reply_language, remaining)
            timings["pipeline_c_ms"] = round((time.perf_counter() - t0) * 1000)
            if _chat_mr_hi_valid(pipeline_c_text, reply_language):
                final_text = _apply_mr_word_fixes(pipeline_c_text) if reply_language == "mr" else pipeline_c_text
                final_text = _add_full_stops_to_bullets(final_text)
                path = "c"

    timings["total_ms"] = round((time.perf_counter() - t_total_start) * 1000)

    if final_text is None:
        # Both b2 and pipeline C failed/were unavailable. Nemotron likely
        # just failed or is slow, so retrying it via _chat_direct (its own
        # long hedge/timeout) would only make the farmer wait longer for the
        # same likely outcome -- a short fixed message instead.
        logger.info("Chat path=none (b2 and c both failed/unavailable) timings=%s", timings)
        fallback_text = _chat_fallback_message(reply_language)
        return ChatResponse(
            response=fallback_text,
            response_en="",
            question_en=text,
            timings={"translate_in": 0, "llm": timings["total_ms"], "translate_out": 0, "total": timings["total_ms"]},
            translation_failed=False,
        )

    logger.info("Chat path=%s timings=%s", path, timings)
    return ChatResponse(
        # Only pipeline B2 (path "b2") ever produces a real English answer;
        # pipeline C (path "c") answers mr/hi directly, so response_en is
        # left empty rather than filled with mr/hi text under an "_en" name.
        response=final_text,
        response_en=english_answer if path == "b2" else "",
        question_en=text,
        timings={"translate_in": 0, "llm": timings["total_ms"], "translate_out": 0, "total": timings["total_ms"]},
        translation_failed=False,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send the farmer's question to the NVIDIA-hosted LLM.

    CHAT_MODE=direct (default): mr/hi goes through pipeline B2
    (_chat_pipeline_b2 — Nemotron answers in English, sarvam-105b rewrites to
    mr/hi), with pipeline C (sarvam answers mr/hi directly) as its fallback
    and a short fixed message as the last resort if both fail; en always
    goes straight to _chat_direct, unaffected by this phase.
    CHAT_LLM_PROVIDER=nemotron switches pipeline B2/C off entirely and keeps
    using _chat_direct for mr/hi too, exactly as before this phase.

    CHAT_MODE=split (not the default, untouched by this phase): the LLM
    always answers in English — it must never write Marathi or Hindi itself,
    since that isn't checked the way disease names are. For mr/hi, the
    question is translated to English first and the English answer is
    translated back afterwards; for en, both hops are skipped.
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided.")

    reply_language = request.reply_language or "en"

    if CHAT_MODE == "direct":
        if reply_language in ("mr", "hi"):
            return await _chat_pipeline_b2(text, reply_language)
        return await _chat_direct(text, reply_language)

    needs_translation = reply_language != "en"
    translation_failed = False
    timings = {"translate_in": 0, "llm": 0, "translate_out": 0, "total": 0}
    t_total_start = time.perf_counter()

    # Step 1: translate the farmer's question to English (mr/hi only)
    question_en = text
    if needs_translation:
        t0 = time.perf_counter()
        try:
            translated_question = await translate(text, reply_language, "en")
            if translated_question.strip():
                question_en = translated_question
        except Exception as e:
            logger.warning("Question translation (%s→en) failed: %s", reply_language, e)
            translation_failed = True
        timings["translate_in"] = round((time.perf_counter() - t0) * 1000)

    logger.info(
        "LLM chat request (%s) reply_language=%s question_en=%s",
        LLM_MODEL, reply_language, question_en[:100],
    )

    if needs_translation:
        language_label = ORIGINAL_LANGUAGE_LABELS.get(reply_language, reply_language)
        user_content = (
            f"Farmer's original question ({language_label}): {text}\n"
            f"Machine translation to English (may contain mistakes): {question_en}\n"
            "Answer the farmer's real question in English."
        )
    else:
        user_content = question_en

    messages = [
        {"role": "system", "content": FARMER_SYSTEM_PROMPT.format(language="English")},
        {"role": "user", "content": user_content},
    ]

    def complete(model: str):
        return asyncio.create_task(_get_llm_client().chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5,
            top_p=0.9,
            max_tokens=LLM_MAX_TOKENS,
            extra_body=_llm_extra_body(model),
        ))

    t0 = time.perf_counter()
    try:
        # Hedged request: if the primary model hasn't answered within
        # LLM_HEDGE_AFTER seconds (free-tier queueing), also ask the fallback
        # model and use whichever answers first.
        completion, failed = None, []
        pending = {complete(LLM_MODEL)}
        timeout = LLM_HEDGE_AFTER
        while completion is None:
            if not pending:
                raise failed[-1].exception()
            done, pending = await asyncio.wait(
                pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            for t in done:
                if t.exception():
                    failed.append(t)
                elif completion is None:
                    completion = t.result()
            if completion is None and timeout is not None and LLM_FALLBACK_MODEL:
                logger.warning("LLM %s slow or failed; also trying %s", LLM_MODEL, LLM_FALLBACK_MODEL)
                pending.add(complete(LLM_FALLBACK_MODEL))
            timeout = None  # hedge only once; then wait for whichever finishes
        for t in pending:
            t.cancel()

        response_en = (completion.choices[0].message.content or "").strip()
        timings["llm"] = round((time.perf_counter() - t0) * 1000)
        logger.info("LLM response (English): %s", response_en[:100])
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in chat")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    # Step 3: translate the English answer to reply_language (mr/hi only)
    final_response = response_en
    if needs_translation:
        t0 = time.perf_counter()
        try:
            translated_answer = await translate(response_en, "en", reply_language)
            if translated_answer.strip():
                final_response = await _apply_farm_glossary(translated_answer, response_en, reply_language)
            else:
                translation_failed = True
        except Exception as e:
            logger.warning("Answer translation (en→%s) failed: %s", reply_language, e)
            translation_failed = True
        timings["translate_out"] = round((time.perf_counter() - t0) * 1000)

    timings["total"] = round((time.perf_counter() - t_total_start) * 1000)
    logger.info("Chat timings (ms): %s", timings)

    return ChatResponse(
        response=final_response,
        response_en=response_en,
        question_en=question_en,
        timings=timings,
        translation_failed=translation_failed,
    )


def pick_provider(module: str, lang: str) -> str:
    """Which LLM backend should answer this request.

    Rule for now: English always uses Nemotron (Sarvam is untested for
    English farmer answers); crop/disease mr/hi use Sarvam unless their own
    CROP_LLM_PROVIDER/DISEASE_LLM_PROVIDER switches it off. Chat's own mr/hi
    routing (pipeline B2/C, CHAT_LLM_PROVIDER) is separate -- see
    _chat_pipeline_b2 -- since it isn't a single explain-step call.
    """
    if lang == "en":
        return "nemotron"
    if module == "crop" and CROP_LLM_PROVIDER == "nemotron":
        return "nemotron"
    if module == "disease" and DISEASE_LLM_PROVIDER == "nemotron":
        return "nemotron"
    return "sarvam"


# ─── Crop recommendation (forwards to Shubham's crop_api, explained by Nemotron) ──

CROP_API_URL = os.getenv("CROP_API_URL", "http://127.0.0.1:5003")
# Seconds to wait for crop_api. 5 is plenty locally; on free hosting a
# sleeping crop_api needs ~1 minute to wake, so Render sets this higher.
CROP_API_TIMEOUT_S = float(os.getenv("CROP_API_TIMEOUT_S", "5"))
# Its own models (independent of /api/chat's LLM_MODEL/LLM_FALLBACK_MODEL):
# Super answered faster than Ultra in practice, so Super goes first and Ultra
# is only the hedge backup.
CROP_LLM_MODEL = os.getenv("CROP_LLM_MODEL", "nvidia/nemotron-3-super-120b-a12b")
CROP_LLM_FALLBACK_MODEL = os.getenv("CROP_LLM_FALLBACK_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
# Shorter total budget than /api/chat's LLM_TIMEOUT (40s) — if neither model
# has answered by CROP_LLM_TOTAL_TIMEOUT, _crop_llm_complete cancels both
# in-flight requests (see its docstring) and the caller falls back to rules.
CROP_LLM_HEDGE_AFTER = float(os.getenv("CROP_LLM_HEDGE_AFTER", "5"))
CROP_LLM_TOTAL_TIMEOUT = float(os.getenv("CROP_LLM_TOTAL_TIMEOUT", "10"))
# Higher than /api/chat's LLM_MAX_TOKENS (600) -- up to 3 crops' reason+risk
# in Devanagari script costs noticeably more tokens than the same text in
# English, and a cut-off reply can silently drop the later crops from the
# JSON (or make it invalid) just as easily as a real LLM error would.
CROP_LLM_MAX_TOKENS = int(os.getenv("CROP_LLM_MAX_TOKENS", "900"))
# "sarvam" (default) tries Sarvam's sarvam-105b first for mr/hi, falling back
# to Nemotron; "nemotron" switches Sarvam off entirely (e.g. once the daily
# free-tier cap is a concern, or to compare against the old behaviour).
CROP_LLM_PROVIDER = os.getenv("CROP_LLM_PROVIDER", "sarvam")
# For mr/hi, the whole Sarvam->Nemotron chain is capped at this many seconds
# total (SARVAM_TIMEOUT_S already caps Sarvam's own share, default 6s).
# Nemotron then gets whatever's left of this budget, down to
# CROP_LLM_MIN_FALLBACK_TIMEOUT_S -- never less, so a near-exhausted budget
# still gives Nemotron a real chance instead of failing instantly. English
# (Nemotron only, no Sarvam) is unaffected and keeps using
# CROP_LLM_TOTAL_TIMEOUT above.
CROP_PROVIDER_TOTAL_TIMEOUT_S = float(os.getenv("CROP_PROVIDER_TOTAL_TIMEOUT_S", "12"))
CROP_LLM_MIN_FALLBACK_TIMEOUT_S = float(os.getenv("CROP_LLM_MIN_FALLBACK_TIMEOUT_S", "3"))

CROP_MESSAGES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "crop_messages.json")
MR_WORD_FIXES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mr_word_fixes.json")
_crop_messages: Optional[dict] = None
_mr_word_fixes: Optional[dict] = None
_crop_llm_client_instance: Optional[AsyncOpenAI] = None

CROP_LLM_SYSTEM_PROMPT = (
    "You explain crop options to a Nashik farmer, using ONLY the crops, "
    "evidence, and the farmer's own inputs given below. Reply ONLY with "
    'JSON, no markdown, no extra text: {{"crops":[{{"id":"...","reason":'
    '"...","risk":"..."}}]}} in {language}. Return exactly one entry for '
    "EACH crop id listed, in the same order, using the ids exactly as "
    "given.\n"
    '"reason": 1-2 short sentences (at most about 25 words) on WHY this '
    "crop fits the farmer's OWN inputs -- the sowing season, the soil "
    "type, whether irrigation is available, and the rotation after the "
    "previous crop -- using only whichever of these the evidence given "
    "actually supports for this crop. Never put any uncertainty in "
    "reason.\n"
    '"risk": exactly 1 short sentence naming one real field risk for this '
    "crop (water shortage, bad weather, pests, or disease -- never a "
    "pesticide or fertilizer name or amount). If this crop's sowing "
    "window is only estimated from its season, or its data is marked not "
    "verified, say that in risk as well and tell the farmer to confirm "
    "with the local KVK.\n"
    "reason and risk must never be the same sentence or near-duplicate "
    "wording -- each must say something the other does not. Never "
    "mention pesticide or fertilizer names or amounts, yields, prices, "
    "profit, or success chances. Do not quote humidity or temperature "
    "numbers, or any specific month name, month number, or date -- the "
    "app shows the sowing months separately; you may name the general "
    "sowing season (e.g. this sowing season, rabi season) if it helps "
    "explain the fit. The evidence comes from a drought contingency "
    "plan; mention drought advice only if the weather shows a dry spell. "
    "Do not copy table fragments like 'No rabi crop'."
)

# Pesticide/chemical words the LLM must never use (copied from
# crop_api/scripts/build_chunks.py's CHEMICAL_WORDS, plus common Marathi/Hindi
# pesticide/fungicide/spraying words, since this LLM can answer in mr/hi
# directly — CLAUDE.md: a wrong dose or product name could harm the crop).
_CROP_UNSAFE_WORDS = [
    "spraying", "endosulfan", "monocrotophos", "phorate", "chloropyriphos",
    "dimethoate", "mancozeb", "quinolphos", "parathion",
    "कीटकनाशक", "फवारणी", "छिड़काव", "बुरशीनाशक", "फफूंदनाशक",
]
# Dose/price units immediately after a digit (ASCII or Devanagari ०-९), plus
# the % and ₹ symbols. "Rs"/"रुपये"/"रुपए" double as price markers.
#
# मिलीमीटर/मिमी/मि.मी (millimetre, used for rainfall, e.g. "०.८ मिलीमीटर
# पाऊस") and "mm" are deliberately NOT dose units — मिली is a prefix of
# मिलीमीटर, so it needs a negative lookahead to not swallow it; "ml"/"g" are
# matched only as whole words (not followed by another word character) so a
# plain word like "good" is never mistaken for the unit "g".
_CROP_DOSE_PRICE_UNITS_SUBSTRING = [
    "kg", "Rs", "मि.ली", "किग्रा", "किलो", "ग्रॅम",
    "ग्राम", "लिटर", "टक्के", "प्रतिशत", "रुपये", "रुपए",
]
_CROP_DOSE_PRICE_UNITS_WHOLE_WORD = ["ml", "g"]
_CROP_DOSE_PRICE_PATTERN = re.compile(
    r"[0-9०-९]+\.?[0-9०-९]*\s*(%|₹|मिली(?!मीटर)|"
    + "|".join(re.escape(u) for u in _CROP_DOSE_PRICE_UNITS_SUBSTRING)
    + "|"
    + "|".join(rf"{re.escape(u)}(?!\w)" for u in _CROP_DOSE_PRICE_UNITS_WHOLE_WORD)
    + r")"
)


def _crop_text_is_unsafe(text: str) -> bool:
    """True if `text` names a pesticide/fungicide or gives a dose/price —
    content the crop LLM must never produce. Matching text is replaced with
    the rules-based fallback text for that crop."""
    if not text:
        return False
    lower = text.lower()
    if any(word.lower() in lower for word in _CROP_UNSAFE_WORDS):
        return True
    return bool(_CROP_DOSE_PRICE_PATTERN.search(text))


_LATIN_WORD_PATTERN = re.compile(r"[A-Za-z]+")
# The only Latin-letter words allowed in an mr/hi reply -- both are proper
# nouns with no Devanagari form the farmer would recognize (Krishi Vigyan
# Kendra, Indian Council of Agricultural Research), and CROP_LLM_SYSTEM_PROMPT
# itself tells the LLM to point to "the local KVK".
_ALLOWED_LATIN_WORDS = {"KVK", "ICAR"}


def _crop_text_is_wrong_language(text: str, language: str) -> bool:
    """True if `language` is mr/hi but `text` contains any Latin-letter word
    other than KVK/ICAR -- the LLM answered in English, or mixed English
    into a Devanagari word (seen in practice: Nemotron wrote "विशेषज्ञanci",
    a Devanagari word with an English suffix glued on). Checking for any
    disallowed Latin run catches this even though count-based "mostly
    English" checks would miss it. A farmer reading mr/hi should never see
    this, so it is treated the same as unsafe content and replaced with the
    fallback text."""
    if language not in ("mr", "hi") or not text:
        return False
    return any(word.upper() not in _ALLOWED_LATIN_WORDS for word in _LATIN_WORD_PATTERN.findall(text))


_DEVANAGARI_CHAR_PATTERN = re.compile(r"[ऀ-ॿ]")


def _chat_text_is_wrong_language(text: str, language: str) -> bool:
    """Chat-only wrong-language check, relaxed from _crop_text_is_wrong_language
    above (which rejects on ANY disallowed Latin word). Real farmer chat
    questions mix in Latin-script crop/product names often enough that the
    strict rule was rejecting good mr/hi answers outright -- e.g. for
    "द्राक्षांच्या पानावर काळे ठिपके" both pipeline B2's rewrite and pipeline C
    got rejected, leaving the farmer with just the fixed fallback message.
    Chat instead accepts the text as long as at least 70% of its letters are
    Devanagari. Logs the first 80 chars and the Latin words found whenever it
    rejects, to make the next bad case easy to find."""
    if language not in ("mr", "hi") or not text:
        return False
    devanagari_count = len(_DEVANAGARI_CHAR_PATTERN.findall(text))
    latin_words = _LATIN_WORD_PATTERN.findall(text)
    latin_count = sum(len(w) for w in latin_words)
    total_letters = devanagari_count + latin_count
    if total_letters == 0:
        return False
    is_wrong = (devanagari_count / total_letters) < 0.70
    if is_wrong:
        logger.warning(
            "Chat wrong-language check rejected text (first 80 chars=%r, latin_words=%s)",
            text[:80], latin_words,
        )
    return is_wrong


def _load_crop_messages() -> dict:
    """Load backend/data/crop_messages.json once and cache it in memory."""
    global _crop_messages
    if _crop_messages is None:
        with open(CROP_MESSAGES_PATH, encoding="utf-8") as f:
            _crop_messages = json.load(f)
    return _crop_messages


def _crop_message(key: str, language: str) -> str:
    entry = _load_crop_messages()[key]
    return entry.get(language) or entry.get("en", "")


def _crop_reason_text(code: str, language: str) -> str:
    entry = _load_crop_messages()["reasons"].get(code)
    if not entry:
        return ""
    return entry.get(language) or entry.get("en", "")


def _load_mr_word_fixes() -> dict:
    """Load backend/data/mr_word_fixes.json once and cache it in memory."""
    global _mr_word_fixes
    if _mr_word_fixes is None:
        with open(MR_WORD_FIXES_PATH, encoding="utf-8") as f:
            _mr_word_fixes = json.load(f)["fixes"]
    return _mr_word_fixes


# Punctuation trimmed off each end of a token before comparing it to
# mr_word_fixes.json's keys -- a plain space-split, not a \b-based regex,
# because Devanagari dependent vowel signs (matras) are combining marks
# that Python's \w does not count as word characters, so \bword\b would
# split in the middle of a word like "बुवाई".
_MR_WORD_FIX_TRIM = ".,;:!?।\"'()[]{}…“”‘’"

# Marathi grammatical suffixes the crop LLM sometimes glues straight onto a
# Hindi word instead of leaving a space, e.g. "अक्टूबरमध्ये" (अक्टूबर +
# मध्ये) or "रबीच्या" (रबी + च्या). Checked only when the token's core isn't
# an exact match in mr_word_fixes.json on its own.
_MR_WORD_FIX_SUFFIXES = ("मध्ये", "साठी", "च्या", "नंतर", "पर्यंत", "ची", "चा", "चे", "ला", "त")


def _apply_mr_word_fixes(text: str) -> str:
    """Replace whole Hindi words the crop LLM sometimes uses with the
    correct Marathi word (e.g. सितंबर -> सप्टेंबर), only ever called for
    language=mr. Matches a standalone space-separated token with its
    surrounding punctuation stripped, never a substring of a longer word --
    except for a Marathi suffix from _MR_WORD_FIX_SUFFIXES glued directly
    onto the end of a known Hindi word, which is also fixed in place.

    A handful of fixes.json entries are multi-word phrases (e.g.
    "केव्हीके शी" -> "केव्हीकेशी", where the LLM wrongly puts a space inside
    "KVK"+"शी"). Those can't go through the per-token loop below (each
    token is one space-separated word), so they're replaced as plain
    substrings first."""
    if not text:
        return text
    fixes = _load_mr_word_fixes()

    for key, replacement in fixes.items():
        if " " in key:
            text = text.replace(key, replacement)

    def _fix_token(token: str) -> str:
        core = token.strip(_MR_WORD_FIX_TRIM)
        if not core:
            return token
        start = token.find(core)
        prefix, trailing = token[:start], token[start + len(core):]

        if core in fixes:
            return prefix + fixes[core] + trailing

        for key, replacement in fixes.items():
            if core.startswith(key) and core[len(key):] in _MR_WORD_FIX_SUFFIXES:
                return prefix + replacement + core[len(key):] + trailing

        return token

    return " ".join(_fix_token(tok) for tok in text.split(" "))


def _crop_display_name(rec: dict, language: str) -> str:
    names = rec.get("names") or {}
    return names.get(language) or names.get("en") or rec["id"]


def _fallback_crop_entry(rec: dict, language: str, fallback_reason: str) -> dict:
    """Build a crop's reason/risk from crop_messages.json's reason-code texts
    (used when the LLM is unavailable, invalid, or wrote unsafe/wrong-language
    content). `fallback_reason` is one of "missing", "unsafe", "wrong_language"
    — logged per crop by the /api/crop route so a drift like Nemotron
    answering one crop in English can be seen in the backend log."""
    texts = [_crop_reason_text(r["code"], language) for r in rec.get("reasons") or []]
    texts = [t for t in texts if t]
    reason_text = " ".join(texts) if texts else _crop_reason_text("not_verified", language)
    return {
        "id": rec["id"], "reason": reason_text, "risk": "",
        "source": "fallback", "fallback_reason": fallback_reason,
    }


def _get_crop_llm_client() -> AsyncOpenAI:
    """Same NVIDIA endpoint as chat's client; a separate key (NVIDIA_API_KEY_CROP)
    is used only if set, so crop traffic can be rate-limited independently."""
    crop_key = os.getenv("NVIDIA_API_KEY_CROP")
    if not crop_key:
        return _get_llm_client()
    global _crop_llm_client_instance
    if _crop_llm_client_instance is None:
        _crop_llm_client_instance = AsyncOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=crop_key,
            timeout=LLM_TIMEOUT,
            max_retries=0,
        )
    return _crop_llm_client_instance


async def _cancel_and_drain(tasks: list) -> None:
    """Cancel every task that is still running and await all of them so
    their results/exceptions are retrieved. Without this, a losing hedge
    request that's still in flight when we stop waiting on it (total
    timeout, or the other model already won) keeps running in the
    background; if it later raises, nobody ever reads that exception and
    Python logs "Task exception was never retrieved" once it's garbage
    collected — this is exactly how Ultra kept running to its own 40s
    client timeout after Super had already answered."""
    for t in tasks:
        if not t.done():
            t.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _crop_llm_complete(messages: list, total_timeout: Optional[float] = None) -> str:
    """Hedged Nemotron call for /api/crop: ask CROP_LLM_MODEL first; if it
    hasn't answered within CROP_LLM_HEDGE_AFTER seconds, also ask
    CROP_LLM_FALLBACK_MODEL and use whichever answers first (same hedge
    pattern as _chat_direct, but its own models, timing, and client —
    /api/chat's hedge is untouched). The whole call is capped at
    `total_timeout` seconds (default CROP_LLM_TOTAL_TIMEOUT) -- the mr/hi
    path passes a smaller value here when Sarvam already used up part of
    CROP_PROVIDER_TOTAL_TIMEOUT_S, so Nemotron only gets what's left of the
    12s chain budget. Raises asyncio.TimeoutError past that, which the
    caller treats as a fallback. Whatever happens (a model wins, both fail,
    or the total timeout fires), every task this call created is cancelled
    and drained before returning — see _cancel_and_drain."""
    client = _get_crop_llm_client()
    all_tasks: list[asyncio.Task] = []

    def complete(model: str):
        task = asyncio.create_task(client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            top_p=0.9,
            max_tokens=CROP_LLM_MAX_TOKENS,
            extra_body=_llm_extra_body(model),
        ))
        all_tasks.append(task)
        return task

    async def hedge():
        completion, failed = None, []
        pending = {complete(CROP_LLM_MODEL)}
        timeout = CROP_LLM_HEDGE_AFTER
        while completion is None:
            if not pending:
                raise failed[-1].exception()
            done, pending = await asyncio.wait(
                pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            for t in done:
                if t.exception():
                    failed.append(t)
                elif completion is None:
                    completion = t.result()
            if completion is None and timeout is not None and CROP_LLM_FALLBACK_MODEL:
                logger.warning("Crop LLM %s slow or failed; also trying %s", CROP_LLM_MODEL, CROP_LLM_FALLBACK_MODEL)
                pending.add(complete(CROP_LLM_FALLBACK_MODEL))
            timeout = None  # hedge only once; then wait for whichever finishes
        return completion

    try:
        completion = await asyncio.wait_for(hedge(), timeout=total_timeout if total_timeout is not None else CROP_LLM_TOTAL_TIMEOUT)
        return (completion.choices[0].message.content or "").strip()
    finally:
        await _cancel_and_drain(all_tasks)


def _crop_sarvam_response_format(rec_ids: list) -> dict:
    """Strict JSON schema for Sarvam's response_format: exactly one
    {id, reason, risk} entry per crop_api recommendation, id restricted to
    this request's ids. Forces one object per crop instead of the merged/
    truncated JSON sarvam-105b produced with a plain-text prompt (seen in
    backend/scripts/debug_crop_c.txt: duplicate "id" keys overwrote earlier
    crops into one object, and a trailing comma broke the parse)."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "crop_recommendations",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "crops": {
                        "type": "array",
                        "minItems": len(rec_ids),
                        "maxItems": len(rec_ids),
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "enum": rec_ids},
                                "reason": {"type": "string"},
                                "risk": {"type": "string"},
                            },
                            "required": ["id", "reason", "risk"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["crops"],
                "additionalProperties": False,
            },
        },
    }


async def _crop_try_sarvam(messages: list, rec_ids: list) -> Optional[str]:
    """Try Sarvam for the crop explanation. Returns the raw JSON text on
    success (already checked to parse with a "crops" list), or None -- after
    logging exactly why -- so the caller falls back to Nemotron."""
    response_format = _crop_sarvam_response_format(rec_ids)
    try:
        raw_text = await sarvam_chat(messages, response_format=response_format, max_tokens=CROP_LLM_MAX_TOKENS)
    except SarvamNotConfiguredError:
        logger.info("Crop LLM fallback sarvam->nemotron (reason=not_configured)")
        return None
    except SarvamCapReachedError as e:
        logger.warning("Crop LLM fallback sarvam->nemotron (reason=cap_reached): %s", e)
        return None
    except SarvamHTTPError as e:
        logger.warning(
            "Crop LLM fallback sarvam->nemotron (reason=http_error status=%s): %s",
            e.status_code, e.body[:300],
        )
        return None
    except httpx.TimeoutException:
        logger.warning("Crop LLM fallback sarvam->nemotron (reason=timeout)")
        return None
    except Exception as e:
        logger.warning("Crop LLM fallback sarvam->nemotron (reason=error): %s", e)
        return None

    if _parse_llm_crops_json(_strip_code_fence(raw_text)) is None:
        logger.warning("Crop LLM fallback sarvam->nemotron (reason=bad_json)")
        return None
    return raw_text


def _strip_code_fence(text: str) -> str:
    """Nemotron sometimes wraps JSON in a ```json ... ``` fence despite being
    told not to; strip it before parsing."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```\s*$", "", text).strip()
    return text


def _parse_llm_crops_json(raw_text: str) -> Optional[list]:
    try:
        data = json.loads(raw_text)
        crops = data.get("crops")
        return crops if isinstance(crops, list) else None
    except Exception:
        return None


def _apply_crop_validator(llm_crops: list, recommendations: list, language: str) -> list:
    """
    Turn the LLM's (possibly empty/partial/unsafe/wrong-language) crop list
    into one entry per crop_api recommendation, in crop_api's order:
      - an id the LLM invented (not in `recommendations`) is ignored
      - a reason/risk containing a dose, price, or pesticide word is replaced
        by that crop's rules-based fallback text (fallback_reason "unsafe")
      - a reason/risk mostly in Latin letters when mr/hi was requested is
        replaced the same way (fallback_reason "wrong_language")
      - any crop_api recommendation the LLM didn't return at all is
        backfilled with its rules-based fallback text (fallback_reason
        "missing")
    Each returned entry also carries "source" ("llm" or "fallback") and
    "fallback_reason" (None for "llm"), which the /api/crop route logs per
    crop so a language/content drift like Nemotron's can be seen in the log.
    """
    rec_by_id = {r["id"]: r for r in recommendations}
    validated_by_id: dict = {}

    for item in llm_crops:
        if not isinstance(item, dict):
            continue
        cid = item.get("id")
        if cid not in rec_by_id:
            continue
        reason = str(item.get("reason") or "").strip()
        risk = str(item.get("risk") or "").strip()
        if _crop_text_is_unsafe(reason) or _crop_text_is_unsafe(risk):
            validated_by_id[cid] = _fallback_crop_entry(rec_by_id[cid], language, "unsafe")
        elif _crop_text_is_wrong_language(reason, language) or _crop_text_is_wrong_language(risk, language):
            validated_by_id[cid] = _fallback_crop_entry(rec_by_id[cid], language, "wrong_language")
        else:
            if language == "mr":
                reason = _apply_mr_word_fixes(reason)
                risk = _apply_mr_word_fixes(risk)
            if risk == reason:
                # The LLM repeated the reason as the risk instead of giving
                # a real one -- keep the reason, replace only the risk.
                risk = _crop_reason_text("not_verified", language)
            validated_by_id[cid] = {
                "id": cid, "reason": reason, "risk": risk,
                "source": "llm", "fallback_reason": None,
            }

    for rec in recommendations:
        if rec["id"] not in validated_by_id:
            validated_by_id[rec["id"]] = _fallback_crop_entry(rec, language, "missing")

    return [validated_by_id[rec["id"]] for rec in recommendations]


# ─── Disease explanation (Sarvam/Nemotron) ─────────────────────────────────
# Phase 2: for an actual (non-healthy, non-not_sure) disease result, explain
# it in 3 parts -- cause, today's non-chemical action, and prevention next
# time. Same provider chain, 12s budget, daily cap, and logging as crop
# above; _crop_llm_complete, _crop_text_is_unsafe/_crop_text_is_wrong_language,
# and _apply_mr_word_fixes are reused as-is (generic despite the "crop"
# name). The disease name/crop name themselves are NEVER written by the
# LLM -- disease_check() always builds `message`/`top` from disease_api's
# class + the checked disease_names.json table, same as before this phase;
# the LLM only ever fills cause/today/prevent.

DISEASE_LLM_PROVIDER = os.getenv("DISEASE_LLM_PROVIDER", "sarvam")
# Below this top-guess probability (but still status=="ok", so above
# disease_api's own 0.7 not_sure cutoff), the prompt is told confidence is
# low and must tell the farmer to confirm with the local KVK too.
DISEASE_LOW_CONFIDENCE_THRESHOLD = float(os.getenv("DISEASE_LOW_CONFIDENCE_THRESHOLD", "0.85"))

DISEASE_LLM_SYSTEM_PROMPT = (
    "You explain a plant disease to a Nashik farmer, in {language}. A "
    "photo-checking tool has already diagnosed the crop and the disease "
    "name given below -- use them exactly as given; never rename the "
    "disease or suggest a different one. Reply ONLY with JSON, no "
    'markdown, no extra text: {{"cause":"...","today":"...","prevent":'
    '"..."}}.\n'
    '"cause": at most 2 short sentences on what this disease is and why '
    "it happens.\n"
    '"today": at most 2 short sentences on what to do today WITHOUT any '
    "chemical, pesticide, or fertilizer.\n"
    '"prevent": at most 2 short sentences on how to prevent it next '
    "time.\n"
    "Base today/prevent closely on the reference precaution lines given "
    "below -- stay close to their meaning, in simple words. Never mention "
    "a pesticide, fungicide, or fertilizer name or amount, a dose, a "
    "price, or a yield/success number."
)


def _cause_text(category: str, language: str) -> str:
    """Fixed "what it is / why it happens" sentence(s) for a disease
    category (backend/data/disease_names.json's cause_by_category), used as
    the rules-fallback for explanation.cause. Same en/mr-hi shape as
    _category_precautions."""
    entry = _load_disease_names().get("cause_by_category", {}).get(category)
    if not entry:
        return ""
    if language == "en":
        lines = entry.get("en", [])
    else:
        lang_entry = entry.get(language)
        lines = lang_entry.get("lines", []) if lang_entry else entry.get("en", [])
    return " ".join(lines)


def _disease_fallback_explanation(category: str, language: str) -> dict:
    """Rules-only explanation (no LLM): cause from cause_by_category, and
    today/prevent built from precautions_by_category's existing 3 checked
    lines -- line[0] (remove/destroy infected material) is "today", and
    line[1]+line[2] (spacing/drainage/tools, then "ask KVK for the right
    product") combined are "prevent". No line is dropped."""
    lines = _category_precautions(category, language)
    today = lines[0] if lines else ""
    prevent = " ".join(lines[1:]) if len(lines) > 1 else ""
    return {"cause": _cause_text(category, language), "today": today, "prevent": prevent}


def _disease_sarvam_response_format() -> dict:
    """Strict JSON schema for Sarvam: exactly cause/today/prevent, all
    required, no extra fields -- same strict-schema approach as crop's
    _crop_sarvam_response_format."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "disease_explanation",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "cause": {"type": "string"},
                    "today": {"type": "string"},
                    "prevent": {"type": "string"},
                },
                "required": ["cause", "today", "prevent"],
                "additionalProperties": False,
            },
        },
    }


def _parse_disease_explanation_json(raw_text: str) -> Optional[dict]:
    try:
        data = json.loads(_strip_code_fence(raw_text))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if not all(isinstance(data.get(k), str) and data.get(k).strip() for k in ("cause", "today", "prevent")):
        return None
    return {k: data[k].strip() for k in ("cause", "today", "prevent")}


async def _disease_try_sarvam(messages: list) -> Optional[str]:
    """Try Sarvam for the disease explanation. Returns the raw JSON text on
    success (already checked to parse with all 3 fields), or None -- after
    logging exactly why -- so the caller falls back to Nemotron. Mirrors
    crop's _crop_try_sarvam."""
    response_format = _disease_sarvam_response_format()
    try:
        raw_text = await sarvam_chat(messages, response_format=response_format, max_tokens=CROP_LLM_MAX_TOKENS)
    except SarvamNotConfiguredError:
        logger.info("Disease LLM fallback sarvam->nemotron (reason=not_configured)")
        return None
    except SarvamCapReachedError as e:
        logger.warning("Disease LLM fallback sarvam->nemotron (reason=cap_reached): %s", e)
        return None
    except SarvamHTTPError as e:
        logger.warning(
            "Disease LLM fallback sarvam->nemotron (reason=http_error status=%s): %s",
            e.status_code, e.body[:300],
        )
        return None
    except httpx.TimeoutException:
        logger.warning("Disease LLM fallback sarvam->nemotron (reason=timeout)")
        return None
    except Exception as e:
        logger.warning("Disease LLM fallback sarvam->nemotron (reason=error): %s", e)
        return None

    if _parse_disease_explanation_json(raw_text) is None:
        logger.warning("Disease LLM fallback sarvam->nemotron (reason=bad_json)")
        return None
    return raw_text


def _apply_disease_explanation_validator(parsed: dict, language: str) -> Optional[dict]:
    """None if any of the 3 fields is unsafe or wrong-language (the caller
    then uses the full rules fallback for all 3 -- same coarse,
    all-or-nothing granularity as crop's per-crop fallback), else the
    (mr-word-fixed) parsed dict."""
    texts = [parsed["cause"], parsed["today"], parsed["prevent"]]
    if any(_crop_text_is_unsafe(t) for t in texts):
        logger.warning("Disease LLM explanation rejected (unsafe content); using rules fallback")
        return None
    if any(_crop_text_is_wrong_language(t, language) for t in texts):
        logger.warning("Disease LLM explanation rejected (wrong language); using rules fallback")
        return None
    if language == "mr":
        return {k: _apply_mr_word_fixes(v) for k, v in parsed.items()}
    return parsed


def _build_disease_explanation_prompt(result: dict, language: str, category: str) -> list:
    crop_label = _crop_label(result["crop"], language)
    disease_label = _disease_label(result["top"][0]["class"], language)
    low_confidence = result["top"][0]["probability"] < DISEASE_LOW_CONFIDENCE_THRESHOLD

    language_name = LANGUAGE_NAMES.get(language, "English")
    system_prompt = DISEASE_LLM_SYSTEM_PROMPT.format(language=language_name)
    if low_confidence:
        system_prompt += (
            " The photo-checking tool was not fully sure, so tell the "
            "farmer to also confirm with the local KVK."
        )
    else:
        system_prompt += " Do not mention the KVK."
    if language in ("mr", "hi"):
        system_prompt += (
            " Write only in Devanagari script. Do not use English letters. "
            "Use short, simple everyday words."
        )

    reference_lines = _category_precautions(category, language)
    reference_text = "\n".join(f"- {line}" for line in reference_lines) if reference_lines else "(none)"
    user_content = (
        f"Crop: {crop_label}\nDiagnosed disease: {disease_label}\n\n"
        "Reference precaution lines for this disease category (checked "
        f"text; base today/prevent closely on these):\n{reference_text}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


async def _get_disease_explanation(result: dict, language: str, category: str) -> tuple[dict, Optional[str]]:
    """Always returns a full {cause, today, prevent} dict (never None) plus
    the provider that produced it ("sarvam"/"nemotron", or None for rules).
    Same Sarvam(<=6s) -> Nemotron(rest of 12s, floor 3s) -> rules chain as
    crop's /api/crop."""
    messages = _build_disease_explanation_prompt(result, language, category)
    provider = pick_provider("disease", language)

    parsed = None
    provider_used = None
    t0 = time.perf_counter()
    try:
        if provider == "sarvam":
            sarvam_text = await _disease_try_sarvam(messages)
            if sarvam_text is not None:
                parsed = _parse_disease_explanation_json(sarvam_text)
                provider_used = "sarvam"

            if parsed is None:
                elapsed = time.perf_counter() - t0
                nemotron_timeout = max(
                    CROP_PROVIDER_TOTAL_TIMEOUT_S - elapsed, CROP_LLM_MIN_FALLBACK_TIMEOUT_S
                )
                raw_text = await _crop_llm_complete(messages, total_timeout=nemotron_timeout)
                parsed = _parse_disease_explanation_json(raw_text)
                if parsed is not None:
                    provider_used = "nemotron"
        else:
            raw_text = await _crop_llm_complete(messages)
            parsed = _parse_disease_explanation_json(raw_text)
            if parsed is not None:
                provider_used = "nemotron"
    except asyncio.TimeoutError:
        logger.warning("Disease LLM: Nemotron timed out; using rules fallback")
        parsed = None
    except Exception as e:
        logger.warning("Disease LLM failed (%s); using rules fallback", e)
        parsed = None

    if parsed is not None:
        validated = _apply_disease_explanation_validator(parsed, language)
        if validated is not None:
            return validated, provider_used

    return _disease_fallback_explanation(category, language), None


@app.get("/api/crop/options")
async def crop_options(language: str = "en"):
    """
    One combined call for the Crop Advisor form: crop_api's talukas,
    soil types, and crops (id + names), fetched in parallel. Used to
    populate the form's dropdowns before the farmer submits /api/crop.
    """
    lang = language if language in LANGUAGE_NAMES else "en"
    try:
        async with httpx.AsyncClient(timeout=CROP_API_TIMEOUT_S) as client:
            talukas_resp, soil_resp, crops_resp = await asyncio.gather(
                client.get(f"{CROP_API_URL}/talukas"),
                client.get(f"{CROP_API_URL}/soil-types"),
                client.get(f"{CROP_API_URL}/crops"),
            )
    except httpx.ConnectError:
        logger.error(
            "crop_api unreachable at %s — start it with: cd crop_api && python app.py",
            CROP_API_URL,
        )
        raise HTTPException(status_code=503, detail=_crop_message("service_down", lang))
    except httpx.RequestError as e:
        logger.error("Error contacting crop_api: %s", e)
        raise HTTPException(status_code=503, detail=_crop_message("service_down", lang))

    if any(r.status_code != 200 for r in (talukas_resp, soil_resp, crops_resp)):
        logger.error("crop_api returned a non-200 status for /api/crop/options")
        raise HTTPException(status_code=503, detail=_crop_message("service_down", lang))

    return {
        "talukas": talukas_resp.json().get("talukas", []),
        "soil_types": soil_resp.json().get("soil_types", []),
        "crops": crops_resp.json().get("crops", []),
    }


@app.post("/api/crop", response_model=CropResponse)
async def crop_check(request: CropRequest):
    """
    Forward the farmer's inputs to crop_api (a separate Flask service on
    port 5003 — see crop_api/app.py) for rule-based recommendations, then ask
    Nemotron to explain them in the farmer's language. If the LLM is slow,
    unavailable, or writes anything unsafe, fall back to fixed rule-based
    text instead (source: "rules").
    """
    lang = request.language if request.language in LANGUAGE_NAMES else "en"
    t_total_start = time.perf_counter()

    crop_payload = {
        "taluka": request.taluka,
        "sowing_month": request.sowing_month,
        "irrigation": request.irrigation,
        "previous_crop": request.previous_crop,
        "soil_type": request.soil_type,
        "soil": request.soil or {},
    }

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=CROP_API_TIMEOUT_S) as client:
            crop_response = await client.post(f"{CROP_API_URL}/recommend-crop", json=crop_payload)
    except httpx.ConnectError:
        logger.error(
            "crop_api unreachable at %s — start it with: cd crop_api && python app.py",
            CROP_API_URL,
        )
        raise HTTPException(status_code=503, detail=_crop_message("service_down", lang))
    except httpx.RequestError as e:
        logger.error("Error contacting crop_api: %s", e)
        raise HTTPException(status_code=503, detail=_crop_message("service_down", lang))
    crop_api_ms = round((time.perf_counter() - t0) * 1000)

    if crop_response.status_code != 200:
        if 400 <= crop_response.status_code < 500:
            # Bad input (e.g. unknown taluka) -- pass crop_api's own errors
            # straight through, same pattern as /api/disease's error passthrough.
            logger.info("crop_api rejected the request (%s): %s", crop_response.status_code, crop_response.text[:200])
            return JSONResponse(status_code=crop_response.status_code, content=crop_response.json())
        logger.error("crop_api returned %s: %s", crop_response.status_code, crop_response.text[:200])
        raise HTTPException(status_code=503, detail=_crop_message("service_down", lang))

    body = crop_response.json()
    recommendations = body.get("recommendations") or []
    weather = body.get("weather")

    if not recommendations:
        total_ms = round((time.perf_counter() - t_total_start) * 1000)
        logger.info("Crop timings (ms): crop_api=%s llm=0 total=%s", crop_api_ms, total_ms)
        return CropResponse(
            language=lang,
            source="rules",
            crops=[],
            weather=weather,
            disclaimer=_crop_message("disclaimer", lang),
            message=_crop_message("no_crops", lang),
        )

    evidence_budget = 1200
    evidence_lines = []
    for rec in recommendations:
        for ev in rec.get("evidence") or []:
            if evidence_budget <= 0:
                break
            text = ev["text"][:evidence_budget]
            evidence_lines.append(f"[{rec['id']} p{ev.get('page')} {ev.get('section')}] {text}")
            evidence_budget -= len(text)
    evidence_text = "\n".join(evidence_lines)

    crop_lines = []
    for rec in recommendations:
        names = rec.get("names") or {}
        reasons = "; ".join(r["text"] for r in rec.get("reasons") or [])
        crop_lines.append(
            f"- id={rec['id']} name_en={names.get('en', rec['id'])} "
            f"sowing_months={rec.get('sowing_months')} reasons=[{reasons}]"
        )
    weather_line = f"Weather: {json.dumps(weather)}" if weather else "Weather: unavailable"
    farmer_inputs_line = (
        f"Farmer's own inputs: taluka={request.taluka}, sowing_month={request.sowing_month}, "
        f"irrigation={request.irrigation}, previous_crop={request.previous_crop or 'none'}, "
        f"soil_type={request.soil_type or 'unknown'}"
    )
    user_content = (
        farmer_inputs_line + "\n\n" +
        "Crops:\n" + "\n".join(crop_lines) + "\n" + weather_line +
        "\n\nEvidence (ICAR Nashik contingency plan):\n" + evidence_text
    )

    language_name = LANGUAGE_NAMES.get(lang, "English")
    crop_system_prompt = CROP_LLM_SYSTEM_PROMPT.format(language=language_name)
    if lang == "mr":
        crop_system_prompt += (
            " Use common Marathi farming words, for example पेरणी for "
            "sowing (not बुवाई), not Hindi. Example of the expected style: "
            '{"id":"chickpea","reason":"भारी जमिनीत व पाण्याची सुविधा '
            'असल्याने हे पीक या हंगामात चांगले येऊ शकते.","risk":"पावसाचा '
            'अंदाज चुकल्यास पाण्याचा ताण येऊ शकतो."}'
        )
    if lang in ("mr", "hi"):
        crop_system_prompt += (
            " Write only in Devanagari script. Do not use English letters. "
            "Use short, simple everyday words."
        )
    messages = [
        {"role": "system", "content": crop_system_prompt},
        {"role": "user", "content": user_content},
    ]

    rec_ids = [r["id"] for r in recommendations]
    provider = pick_provider("crop", lang)

    source = "rules"
    validated = None
    provider_used = None
    parsed = None
    llm_ms = 0
    nemotron_timeout_used = CROP_LLM_TOTAL_TIMEOUT
    t0 = time.perf_counter()
    try:
        if provider == "sarvam":
            sarvam_text = await _crop_try_sarvam(messages, rec_ids)
            if sarvam_text is not None:
                parsed = _parse_llm_crops_json(_strip_code_fence(sarvam_text))
                provider_used = "sarvam"

            if parsed is None:
                # Sarvam already spent part of the 12s chain budget; give
                # Nemotron only what's left of it (never less than
                # CROP_LLM_MIN_FALLBACK_TIMEOUT_S).
                elapsed = time.perf_counter() - t0
                nemotron_timeout_used = max(
                    CROP_PROVIDER_TOTAL_TIMEOUT_S - elapsed, CROP_LLM_MIN_FALLBACK_TIMEOUT_S
                )
                raw_text = await _crop_llm_complete(messages, total_timeout=nemotron_timeout_used)
                parsed = _parse_llm_crops_json(_strip_code_fence(raw_text))
                if parsed is not None:
                    provider_used = "nemotron"
        else:
            raw_text = await _crop_llm_complete(messages)
            parsed = _parse_llm_crops_json(_strip_code_fence(raw_text))
            if parsed is not None:
                provider_used = "nemotron"

        llm_ms = round((time.perf_counter() - t0) * 1000)
        if parsed is not None:
            validated = _apply_crop_validator(parsed, recommendations, lang)
            source = "llm"
    except asyncio.TimeoutError:
        llm_ms = round((time.perf_counter() - t0) * 1000)
        logger.warning("Crop LLM: Nemotron timed out after %.1fs; using rules fallback", nemotron_timeout_used)
    except Exception as e:
        llm_ms = round((time.perf_counter() - t0) * 1000)
        logger.warning("Crop LLM failed (%s); using rules fallback", e)

    if validated is None:
        provider_used = None
        validated = _apply_crop_validator([], recommendations, lang)

    validated_by_id = {c["id"]: c for c in validated}
    final_crops = []
    per_crop_log = []
    for rec in recommendations:
        v = validated_by_id[rec["id"]]
        seen_evidence = set()
        evidence_refs = []
        for e in rec.get("evidence") or []:
            key = (e.get("page"), e.get("section"))
            if key in seen_evidence:
                continue
            seen_evidence.add(key)
            evidence_refs.append({"page": e.get("page"), "section": e.get("section")})
        final_crops.append({
            "id": rec["id"],
            "name": _crop_display_name(rec, lang),
            "reason": v["reason"],
            "risk": v["risk"],
            "sowing_months": rec.get("sowing_months"),
            "evidence_refs": evidence_refs,
        })
        tag = v["source"] if v["source"] == "llm" else f"fallback({v['fallback_reason']})"
        per_crop_log.append(f"{rec['id']}={tag}")

    total_ms = round((time.perf_counter() - t_total_start) * 1000)
    logger.info(
        "Crop timings (ms): crop_api=%s llm=%s total=%s provider=%s",
        crop_api_ms, llm_ms, total_ms, provider_used or "none",
    )
    logger.info("Crop per-crop source: %s", ", ".join(per_crop_log))

    return CropResponse(
        language=lang,
        source=source,
        crops=final_crops,
        weather=weather,
        disclaimer=_crop_message("disclaimer", lang),
        provider=provider_used,
    )


@app.post("/api/chats")
def api_create_chat(request: CreateChatRequest):
    try:
        chat_id = create_chat(request.title)
        return {"chat_id": chat_id, "title": request.title}
    except Exception as e:
        logger.exception("Error creating chat")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chats")
def api_get_chats():
    try:
        chats = get_chats()
        return {"chats": chats}
    except Exception as e:
        logger.exception("Error getting chats")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chats/{chat_id}/messages")
def api_get_messages(chat_id: int):
    try:
        messages = get_chat_messages(chat_id)
        return {"messages": messages}
    except Exception as e:
        logger.exception("Error getting messages")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chats/{chat_id}/messages")
def api_add_message(chat_id: int, request: AddMessageRequest):
    try:
        msg_id = add_message(chat_id, request.role, request.msg_type, request.content)
        return {"message_id": msg_id, "success": True}
    except Exception as e:
        logger.exception("Error adding message")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
