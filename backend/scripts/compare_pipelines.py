"""
compare_pipelines.py — side-by-side test of 3 answer pipelines for PID-14.

Read-only research script. Does NOT change main.py, crop_api, or disease_api.
It only *imports* backend/main.py to reuse its existing Nemotron call code
(_chat_direct, _crop_llm_complete, _apply_crop_validator, the system prompts,
etc.) instead of copying that logic by hand. Importing main.py has no side
effects here: no server is started (uvicorn.run is behind
`if __name__ == "__main__":` in main.py) and no database file is touched
(init_db() only runs inside main.py's FastAPI lifespan, which never fires
just from importing the module).

Pipelines compared, for every question in test_questions.json:
  A — Nemotron answers directly in mr/hi (main.py's existing CHAT_MODE=direct
      behaviour for chat; for crop, main's existing /api/crop behaviour).
  B — translate mr/hi -> en (Sarvam), Nemotron answers in English, translate
      en -> mr/hi back (Sarvam). For crop, the crop LLM JSON's reason/risk
      text fields are translated field-by-field after the English call.
  C — sarvam-105b answers directly in mr/hi, same prompts as A.

Before running:
  - crop_api must already be running on http://127.0.0.1:5003 (for the
    "crop" module's rule-based recommendations) — start it yourself with
    `cd crop_api && python app.py`. This script never starts it.
  - backend/.env must have SARVAM_API_KEY and NVIDIA_API_KEY set.

Run:  cd backend && python scripts/compare_pipelines.py
Output: backend/scripts/compare_out.csv (UTF-8 with BOM, for Excel/Marathi).
"""

import os
import sys
import json
import time
import asyncio
import logging
import csv

import httpx

# ─── Make backend/main.py importable (this script lives in backend/scripts/) ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import main as backend_main  # noqa: E402  (backend/main.py — reused, never modified)

# Quieten the backend's own request-by-request logging so our prints stand out.
logging.getLogger().setLevel(logging.WARNING)

QUESTIONS_PATH = os.path.join(SCRIPT_DIR, "test_questions.json")
OUT_PATH = os.path.join(SCRIPT_DIR, "compare_out.csv")
PIPELINES = ["A", "B", "C"]

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"
SARVAM_CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"
SARVAM_CHAT_MODEL = "sarvam-105b"
SARVAM_LANG_CODES = {"mr": "mr-IN", "hi": "hi-IN", "en": "en-IN"}

API_SLEEP_SECONDS = 1  # "wait 1s between API calls" (CLAUDE.md-style rate-limit courtesy)


# ─── Hardcoded structured inputs (module: crop), matched to r01-r05's text ────
CROP_HARDCODED_INPUTS = {
    # Niphad, Oct, heavy black soil, well water (irrigation), previous=soybean
    "r01": {"taluka": "Niphad", "sowing_month": 10, "irrigation": "yes", "previous_crop": "soybean", "soil_type": "heavy"},
    # Sinnar, light soil, rain-only (no irrigation), June
    "r02": {"taluka": "Sinnar", "sowing_month": 6, "irrigation": "no", "previous_crop": None, "soil_type": "light"},
    # Malegaon, July, medium soil, rainfed, previous=cotton
    "r03": {"taluka": "Malegaon", "sowing_month": 7, "irrigation": "no", "previous_crop": "cotton", "soil_type": "medium"},
    # Dindori, Nov, medium soil, irrigated, previous=tomato
    "r04": {"taluka": "Dindori", "sowing_month": 11, "irrigation": "yes", "previous_crop": "tomato", "soil_type": "medium"},
    # Yeola, light soil, no water, August
    "r05": {"taluka": "Yeola", "sowing_month": 8, "irrigation": "no", "previous_crop": None, "soil_type": "light"},
}

# ─── Hardcoded disease_api classes (module: disease), matched to d01-d05's text ──
DISEASE_HARDCODED_CLASSES = {
    "d01": "Grape___Powdery_Mildew",       # "पांढरी भुकटी" = white powder
    "d02": "Tomato___Target_Spot",         # "गोल गोल दाग" = round spots
    "d03": "Onion___Purple_Blotch",        # "जांभळे डाग" = purple spots
    "d04": "Cotton___Curl_Virus",          # leaves curling + reddening
    "d05": "Maize___Gray_Leaf_Spot",       # "लांब करड्या रेषा" = long grey streaks
}


def _new_step_times():
    return {"translate_in": 0.0, "llm": 0.0, "translate_out": 0.0}


# ─── Sarvam API calls ──────────────────────────────────────────────────────────

async def sarvam_translate(client: httpx.AsyncClient, text: str, source_lang: str, target_lang: str) -> str:
    """POST https://api.sarvam.ai/translate — docs.sarvam.ai (Mayura/Translate guide).
    Uses sarvam-translate:v1 for its larger 2000-char cap (vs 1000 for mayura:v1),
    since LLM answers can run long."""
    if not text or not text.strip() or source_lang == target_lang:
        return text
    resp = await client.post(
        SARVAM_TRANSLATE_URL,
        headers={"Content-Type": "application/json", "api-subscription-key": SARVAM_API_KEY},
        json={
            "input": text[:2000],
            "source_language_code": SARVAM_LANG_CODES[source_lang],
            "target_language_code": SARVAM_LANG_CODES[target_lang],
            "model": "sarvam-translate:v1",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["translated_text"]


async def sarvam_chat(client: httpx.AsyncClient, system_prompt: str, user_content: str, max_tokens: int = 800) -> str:
    """POST https://api.sarvam.ai/v1/chat/completions, model sarvam-105b.
    reasoning_effort MUST be None, or the model spends its whole budget
    thinking and returns an empty answer (per CLAUDE.md task instructions)."""
    resp = await client.post(
        SARVAM_CHAT_URL,
        headers={"api-subscription-key": SARVAM_API_KEY},
        json={
            "model": SARVAM_CHAT_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": max_tokens,
            "reasoning_effort": None,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return (data["choices"][0]["message"].get("content") or "").strip()


# ─── Generic Nemotron hedge call (for prompts that don't go through an ────────
# existing main.py function, i.e. disease module + chat pipeline B's English
# step). Reuses main.py's client, extra_body, and hedge constants/pattern —
# main.py itself never factored this loop into a standalone function, so it
# is duplicated here rather than touching main.py. ────────────────────────────

async def hedge_chat_complete(messages: list, max_tokens: int = None) -> str:
    max_tokens = max_tokens or backend_main.LLM_MAX_TOKENS
    client = backend_main._get_llm_client()

    def complete(model: str):
        return asyncio.create_task(client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5,
            top_p=0.9,
            max_tokens=max_tokens,
            extra_body=backend_main._llm_extra_body(model),
        ))

    completion, failed = None, []
    pending = {complete(backend_main.LLM_MODEL)}
    timeout = backend_main.LLM_HEDGE_AFTER
    while completion is None:
        if not pending:
            raise failed[-1].exception()
        done, pending = await asyncio.wait(pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            if t.exception():
                failed.append(t)
            elif completion is None:
                completion = t.result()
        if completion is None and timeout is not None and backend_main.LLM_FALLBACK_MODEL:
            pending.add(complete(backend_main.LLM_FALLBACK_MODEL))
        timeout = None
    for t in pending:
        t.cancel()
    return (completion.choices[0].message.content or "").strip()


# ─── module: chat ───────────────────────────────────────────────────────────────

async def run_chat(client: httpx.AsyncClient, q: dict, pipeline: str) -> dict:
    lang, text = q["lang"], q["text"]
    step_times = _new_step_times()
    answer = ""
    try:
        if pipeline == "A":
            # Reuses main.py's existing CHAT_MODE=direct code exactly.
            resp = await backend_main._chat_direct(text, lang)
            answer = resp.response
            step_times["llm"] = resp.timings["llm"] / 1000
            await asyncio.sleep(API_SLEEP_SECONDS)

        elif pipeline == "B":
            t0 = time.perf_counter()
            text_en = await sarvam_translate(client, text, lang, "en")
            step_times["translate_in"] = time.perf_counter() - t0
            await asyncio.sleep(API_SLEEP_SECONDS)

            system_prompt = backend_main.FARMER_SYSTEM_PROMPT.format(language="English")
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text_en}]
            t0 = time.perf_counter()
            answer_en = await hedge_chat_complete(messages)
            step_times["llm"] = time.perf_counter() - t0
            await asyncio.sleep(API_SLEEP_SECONDS)

            t0 = time.perf_counter()
            answer = await sarvam_translate(client, answer_en, "en", lang)
            step_times["translate_out"] = time.perf_counter() - t0
            await asyncio.sleep(API_SLEEP_SECONDS)

        else:  # C
            language_name = backend_main.LANGUAGE_NAMES.get(lang, "English")
            system_prompt = backend_main.FARMER_SYSTEM_PROMPT.format(language=language_name)
            if lang in ("mr", "hi"):
                system_prompt += (
                    " Write only in Devanagari script. Do not use English letters. "
                    "Use short, simple everyday words."
                )
            t0 = time.perf_counter()
            answer = await sarvam_chat(client, system_prompt, text)
            step_times["llm"] = time.perf_counter() - t0
            await asyncio.sleep(API_SLEEP_SECONDS)

        return {"answer": answer, "step_times": step_times, "validator_ok": ""}
    except Exception as e:
        return {"answer": f"[ERROR] {type(e).__name__}: {e}", "step_times": step_times, "validator_ok": ""}


# ─── module: crop ───────────────────────────────────────────────────────────────

async def fetch_crop_recommendations(client: httpx.AsyncClient, qid: str):
    payload = {**CROP_HARDCODED_INPUTS[qid], "soil": {}}
    resp = await client.post(f"{backend_main.CROP_API_URL}/recommend-crop", json=payload, timeout=10)
    resp.raise_for_status()
    body = resp.json()
    return body.get("recommendations") or [], body.get("weather")


def build_crop_prompt(recommendations: list, weather, lang: str):
    """Duplicates backend/main.py's crop_check() prompt assembly (not
    factored into a reusable function there), reusing its constants
    (CROP_LLM_SYSTEM_PROMPT, LANGUAGE_NAMES) so wording stays in sync."""
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
    user_content = (
        "Crops:\n" + "\n".join(crop_lines) + "\n" + weather_line +
        "\n\nEvidence (ICAR Nashik contingency plan):\n" + evidence_text
    )

    language_name = backend_main.LANGUAGE_NAMES.get(lang, "English")
    system_prompt = backend_main.CROP_LLM_SYSTEM_PROMPT.format(language=language_name)
    if lang == "mr":
        system_prompt += (
            " Use common Marathi farming words, for example पेरणी for "
            "sowing (not बुवाई). Use Marathi month names and words, not "
            "Hindi. Example of the expected style: "
            '{"id":"chickpea","reason":"ऑक्टोबरमध्ये पेरणीसाठी योग्य पीक आहे.",'
            '"risk":"पाणी कमी असल्यास उत्पादन कमी होऊ शकते."}'
        )
    if lang in ("mr", "hi"):
        system_prompt += (
            " Write only in Devanagari script. Do not use English letters. "
            "Use short, simple everyday words."
        )
    return system_prompt, user_content


def format_crop_answer(validated: list, recommendations: list, lang: str) -> str:
    names = {rec["id"]: backend_main._crop_display_name(rec, lang) for rec in recommendations}
    return "\n".join(f"{names[c['id']]}: {c['reason']} ({c['risk']})" for c in validated)


async def run_crop(client: httpx.AsyncClient, q: dict, pipeline: str, crop_cache: dict) -> dict:
    lang = q["lang"]
    step_times = _new_step_times()
    try:
        if q["id"] not in crop_cache:
            crop_cache[q["id"]] = await fetch_crop_recommendations(client, q["id"])
        recommendations, weather = crop_cache[q["id"]]
        if not recommendations:
            return {"answer": "[crop_api returned no recommendations]", "step_times": step_times, "validator_ok": ""}

        if pipeline in ("A", "C"):
            system_prompt, user_content = build_crop_prompt(recommendations, weather, lang)
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
            t0 = time.perf_counter()
            if pipeline == "A":
                # Reuses main.py's existing hedged crop-LLM call exactly.
                raw_text = await backend_main._crop_llm_complete(messages)
            else:
                raw_text = await sarvam_chat(client, system_prompt, user_content, max_tokens=backend_main.CROP_LLM_MAX_TOKENS)
            step_times["llm"] = time.perf_counter() - t0
            await asyncio.sleep(API_SLEEP_SECONDS)

            parsed = backend_main._parse_llm_crops_json(backend_main._strip_code_fence(raw_text)) or []
            validated = backend_main._apply_crop_validator(parsed, recommendations, lang)

        else:  # B
            system_prompt_en, user_content = build_crop_prompt(recommendations, weather, "en")
            messages = [{"role": "system", "content": system_prompt_en}, {"role": "user", "content": user_content}]
            t0 = time.perf_counter()
            raw_text = await backend_main._crop_llm_complete(messages)
            step_times["llm"] = time.perf_counter() - t0
            await asyncio.sleep(API_SLEEP_SECONDS)

            parsed = backend_main._parse_llm_crops_json(backend_main._strip_code_fence(raw_text)) or []
            t0 = time.perf_counter()
            translated = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                reason_t = await sarvam_translate(client, str(item.get("reason") or ""), "en", lang)
                await asyncio.sleep(API_SLEEP_SECONDS)
                risk_t = await sarvam_translate(client, str(item.get("risk") or ""), "en", lang)
                await asyncio.sleep(API_SLEEP_SECONDS)
                translated.append({"id": item.get("id"), "reason": reason_t, "risk": risk_t})
            step_times["translate_out"] = time.perf_counter() - t0

            validated = backend_main._apply_crop_validator(translated, recommendations, lang)

        validator_ok = all(c["source"] == "llm" for c in validated)
        answer = format_crop_answer(validated, recommendations, lang)
        return {"answer": answer, "step_times": step_times, "validator_ok": validator_ok}
    except Exception as e:
        return {"answer": f"[ERROR] {type(e).__name__}: {e}", "step_times": step_times, "validator_ok": ""}


# ─── module: disease ────────────────────────────────────────────────────────────

def disease_en_names(class_name: str):
    crop, disease = class_name.split("___", 1)
    return crop, disease.replace("_", " ")


def build_disease_prompt(crop_en: str, disease_en: str, question_text: str, lang: str):
    """Reuses main.py's FARMER_SYSTEM_PROMPT (same no-pesticide-dose, no-brand,
    KVK-referral, TTS-safe-plain-text rules) for a use case main.py's own
    /api/disease route doesn't have: a free-text LLM explanation of a disease,
    to compare against /api/disease's fixed templates."""
    language_name = backend_main.LANGUAGE_NAMES.get(lang, "English")
    system_prompt = backend_main.FARMER_SYSTEM_PROMPT.format(language=language_name)
    if lang in ("mr", "hi"):
        system_prompt += (
            " Write only in Devanagari script. Do not use English letters. "
            "Use short, simple everyday words."
        )
    user_content = (
        f"Farmer's question: {question_text}\n"
        f"A photo-checking tool diagnosed the leaf as: {crop_en} - {disease_en}.\n"
        "Explain in simple words what this disease is and what general "
        "precautions the farmer should take."
    )
    return system_prompt, user_content


async def run_disease(client: httpx.AsyncClient, q: dict, pipeline: str) -> dict:
    lang, text = q["lang"], q["text"]
    class_name = DISEASE_HARDCODED_CLASSES[q["id"]]
    crop_en, disease_en = disease_en_names(class_name)
    step_times = _new_step_times()
    try:
        if pipeline == "A":
            system_prompt, user_content = build_disease_prompt(crop_en, disease_en, text, lang)
            t0 = time.perf_counter()
            answer = await hedge_chat_complete(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
            )
            step_times["llm"] = time.perf_counter() - t0
            await asyncio.sleep(API_SLEEP_SECONDS)

        elif pipeline == "B":
            t0 = time.perf_counter()
            text_en = await sarvam_translate(client, text, lang, "en")
            step_times["translate_in"] = time.perf_counter() - t0
            await asyncio.sleep(API_SLEEP_SECONDS)

            system_prompt_en, user_content_en = build_disease_prompt(crop_en, disease_en, text_en, "en")
            t0 = time.perf_counter()
            answer_en = await hedge_chat_complete(
                [{"role": "system", "content": system_prompt_en}, {"role": "user", "content": user_content_en}]
            )
            step_times["llm"] = time.perf_counter() - t0
            await asyncio.sleep(API_SLEEP_SECONDS)

            t0 = time.perf_counter()
            answer = await sarvam_translate(client, answer_en, "en", lang)
            step_times["translate_out"] = time.perf_counter() - t0
            await asyncio.sleep(API_SLEEP_SECONDS)

        else:  # C
            system_prompt, user_content = build_disease_prompt(crop_en, disease_en, text, lang)
            t0 = time.perf_counter()
            answer = await sarvam_chat(client, system_prompt, user_content)
            step_times["llm"] = time.perf_counter() - t0
            await asyncio.sleep(API_SLEEP_SECONDS)

        return {"answer": answer, "step_times": step_times, "validator_ok": ""}
    except Exception as e:
        return {"answer": f"[ERROR] {type(e).__name__}: {e}", "step_times": step_times, "validator_ok": ""}


# ─── driver ─────────────────────────────────────────────────────────────────────

async def run_comparison():
    if not SARVAM_API_KEY:
        print("SARVAM_API_KEY not set in backend/.env — aborting.")
        return
    if not os.getenv("NVIDIA_API_KEY"):
        print("NVIDIA_API_KEY not set in backend/.env — aborting.")
        return

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    rows = []
    crop_cache: dict = {}

    async with httpx.AsyncClient() as client:
        for q in questions:
            for pipeline in PIPELINES:
                print(f"Running {q['id']} ({q['module']}, {q['lang']}) pipeline {pipeline} ...")
                if q["module"] == "chat":
                    result = await run_chat(client, q, pipeline)
                elif q["module"] == "crop":
                    result = await run_crop(client, q, pipeline, crop_cache)
                elif q["module"] == "disease":
                    result = await run_disease(client, q, pipeline)
                else:
                    raise ValueError(f"Unknown module '{q['module']}' for question {q['id']}")

                total_sec = round(sum(result["step_times"].values()), 3)
                rows.append({
                    "id": q["id"],
                    "module": q["module"],
                    "lang": q["lang"],
                    "question": q["text"],
                    "pipeline": pipeline,
                    "answer": result["answer"],
                    "total_sec": total_sec,
                    "step_times": json.dumps({k: round(v, 3) for k, v in result["step_times"].items()}),
                    "validator_ok": result["validator_ok"],
                    "correct": "",
                    "language_ok": "",
                    "notes": "",
                })

    fieldnames = [
        "id", "module", "lang", "question", "pipeline", "answer", "total_sec",
        "step_times", "validator_ok", "correct", "language_ok", "notes",
    ]
    with open(OUT_PATH, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_PATH}")

    # ─── avg / slowest total_sec per pipeline per module ───────────────────────
    groups: dict = {}
    for r in rows:
        key = (r["module"], r["pipeline"])
        groups.setdefault(key, []).append(r["total_sec"])

    print("\nmodule    pipeline  avg_sec  slowest_sec")
    for (module, pipeline), values in sorted(groups.items()):
        avg_sec = round(sum(values) / len(values), 2)
        slowest_sec = round(max(values), 2)
        print(f"{module:<9} {pipeline:<9} {avg_sec:<8} {slowest_sec}")


if __name__ == "__main__":
    asyncio.run(run_comparison())
