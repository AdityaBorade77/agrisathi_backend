"""
compare_chat_b2.py — a second "split" variant for the chat module only:
pipeline B2 = Nemotron reads the farmer's ORIGINAL mr/hi question directly
and answers in English (no upfront translation step), then sarvam-105b
rewrites that English answer into natural mr/hi — instead of B's plain
Sarvam Translate round-trip. Compares against B on faithfulness/fluency.

Does NOT touch main.py or compare_pipelines.py — only imports
compare_pipelines.py to reuse its Nemotron hedge call and Sarvam chat call
(hedge_chat_complete, sarvam_chat) and its already-loaded env/constants
(backend_main.FARMER_SYSTEM_PROMPT, LANGUAGE_NAMES). Importing it runs no
API calls and starts no server — compare_pipelines.py only runs its
pipeline loop under `if __name__ == "__main__":`.

Only the 10 chat questions (c01-c10) from test_questions.json, pipeline B2
only — no crop, no disease, no A/B/C.

Run:  backend\\venv\\Scripts\\python.exe backend\\scripts\\compare_chat_b2.py
Output: backend/scripts/chat_B2.txt (UTF-8)
"""

import os
import sys
import json
import time
import asyncio

import httpx

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import compare_pipelines as cp  # noqa: E402 (reused, never modified)
backend_main = cp.backend_main  # backend/main.py, imported inside compare_pipelines.py

OUT_PATH = os.path.join(SCRIPT_DIR, "chat_B2.txt")

REWRITE_SYSTEM_PROMPT = (
    "Rewrite this farming advice in natural, simple {language} for a "
    "Nashik farmer. Keep every fact and number exactly. Use local farming "
    "words. Output only the rewritten text."
)


async def run_b2(client: httpx.AsyncClient, q: dict) -> dict:
    lang, text = q["lang"], q["text"]
    language_name = backend_main.LANGUAGE_NAMES.get(lang, "English")

    result = {
        "id": q["id"], "question": text, "english_answer": "", "final_answer": "",
        "nemotron_sec": 0.0, "sarvam_sec": 0.0, "total_sec": 0.0,
    }

    # Step 1: Nemotron reads the original mr/hi question, answers in English.
    system_prompt_en = (
        backend_main.FARMER_SYSTEM_PROMPT.format(language="English")
        + " Answer in simple English, 3-5 sentences."
    )
    messages = [{"role": "system", "content": system_prompt_en}, {"role": "user", "content": text}]
    try:
        t0 = time.perf_counter()
        english_answer = await cp.hedge_chat_complete(messages)
        result["nemotron_sec"] = round(time.perf_counter() - t0, 3)
        result["english_answer"] = english_answer
    except Exception as e:
        result["english_answer"] = f"[ERROR] {type(e).__name__}: {e}"
        result["total_sec"] = round(result["nemotron_sec"] + result["sarvam_sec"], 3)
        return result
    await asyncio.sleep(cp.API_SLEEP_SECONDS)

    # Step 2: sarvam-105b rewrites the English answer into natural mr/hi.
    rewrite_system_prompt = REWRITE_SYSTEM_PROMPT.format(language=language_name)
    try:
        t0 = time.perf_counter()
        final_answer = await cp.sarvam_chat(client, rewrite_system_prompt, english_answer)
        result["sarvam_sec"] = round(time.perf_counter() - t0, 3)
        result["final_answer"] = final_answer
    except Exception as e:
        result["final_answer"] = f"[ERROR] {type(e).__name__}: {e}"
    await asyncio.sleep(cp.API_SLEEP_SECONDS)

    result["total_sec"] = round(result["nemotron_sec"] + result["sarvam_sec"], 3)
    return result


async def run_comparison():
    if not cp.SARVAM_API_KEY:
        print("SARVAM_API_KEY not set in backend/.env — aborting.")
        return
    if not os.getenv("NVIDIA_API_KEY"):
        print("NVIDIA_API_KEY not set in backend/.env — aborting.")
        return

    with open(cp.QUESTIONS_PATH, encoding="utf-8") as f:
        questions = [q for q in json.load(f) if q["module"] == "chat"]

    results = []
    async with httpx.AsyncClient() as client:
        for q in questions:
            print(f"Running {q['id']} pipeline B2 ...")
            results.append(await run_b2(client, q))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write("=" * 70 + "\n")
            f.write(f"id: {r['id']}\n")
            f.write(f"question: {r['question']}\n")
            f.write(f"nemotron_sec: {r['nemotron_sec']}\n")
            f.write(f"english_answer: {r['english_answer']}\n")
            f.write(f"sarvam_sec: {r['sarvam_sec']}\n")
            f.write(f"final_answer: {r['final_answer']}\n")
            f.write(f"total_sec: {r['total_sec']}\n")
        f.write("=" * 70 + "\n")
    print(f"\nWrote {len(results)} rows to {OUT_PATH}")

    totals = [r["total_sec"] for r in results]
    avg_sec = round(sum(totals) / len(totals), 2) if totals else 0.0
    slowest_sec = round(max(totals), 2) if totals else 0.0
    print(f"avg total_sec: {avg_sec}")
    print(f"slowest total_sec: {slowest_sec}")


if __name__ == "__main__":
    asyncio.run(run_comparison())
