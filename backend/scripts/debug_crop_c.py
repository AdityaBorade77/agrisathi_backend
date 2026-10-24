"""
debug_crop_c.py — isolates why the crop module's pipeline C (sarvam-105b,
direct mr/hi) gets validator rejections, for r01 and r05 only.

Does NOT touch main.py or compare_pipelines.py — only imports
compare_pipelines.py to reuse its exact input/prompt-building code
(CROP_HARDCODED_INPUTS, fetch_crop_recommendations, build_crop_prompt,
sarvam_chat) so this test sends the identical prompt compare_pipelines.py
sends, not a re-typed copy of it. Importing compare_pipelines.py (and the
main.py it imports) runs no API calls and starts no server — both only
execute their pipeline/server code under `if __name__ == "__main__":`.

Only 2 Sarvam calls total: r01 then r05, pipeline C only, no Nemotron.

Before running: crop_api must already be running on http://127.0.0.1:5003.

Run:  backend\\venv\\Scripts\\python.exe backend\\scripts\\debug_crop_c.py
Output: backend/scripts/debug_crop_c.txt (UTF-8)
"""

import os
import sys
import json
import asyncio

import httpx

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import compare_pipelines as cp  # noqa: E402 (reused, never modified)
backend_main = cp.backend_main  # backend/main.py, imported inside compare_pipelines.py

QIDS = ["r01", "r05"]
QUESTIONS_PATH = cp.QUESTIONS_PATH
OUT_PATH = os.path.join(SCRIPT_DIR, "debug_crop_c.txt")


def diagnose_crop_items(raw_text: str, recommendations: list, lang: str) -> str:
    """Walks the same checks as backend_main._apply_crop_validator(), but
    reports WHY each item was accepted/rejected instead of silently
    substituting fallback text."""
    stripped = backend_main._strip_code_fence(raw_text)
    parsed = backend_main._parse_llm_crops_json(stripped)

    lines = []
    if parsed is None:
        lines.append(
            "JSON parse FAILED — _parse_llm_crops_json() returned None "
            "(raw text isn't valid JSON, or has no 'crops' list)."
        )
        parsed = []
    else:
        lines.append(f"JSON parsed OK — {len(parsed)} item(s) in 'crops'.")

    rec_by_id = {r["id"]: r for r in recommendations}
    seen_ids = set()

    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            lines.append(f"  item[{i}]: not a JSON object ({type(item).__name__}) — ignored.")
            continue

        cid = item.get("id")
        if cid not in rec_by_id:
            lines.append(
                f"  item[{i}]: id={cid!r} — UNKNOWN crop id "
                f"(not one of crop_api's recommended ids: {list(rec_by_id)}) — ignored entirely."
            )
            continue

        seen_ids.add(cid)
        reason = str(item.get("reason") or "").strip()
        risk = str(item.get("risk") or "").strip()
        problems = []

        for field_name, text in (("reason", reason), ("risk", risk)):
            if not text:
                problems.append(f"{field_name} is empty")
                continue
            lower = text.lower()
            unsafe_words = [w for w in backend_main._CROP_UNSAFE_WORDS if w.lower() in lower]
            if unsafe_words:
                problems.append(f"{field_name}: pesticide/chemical word(s) {unsafe_words}")
            dose_match = backend_main._CROP_DOSE_PRICE_PATTERN.search(text)
            if dose_match:
                problems.append(f"{field_name}: dose/price pattern matched {dose_match.group(0)!r}")
            if lang in ("mr", "hi"):
                latin_words = [
                    w for w in backend_main._LATIN_WORD_PATTERN.findall(text)
                    if w.upper() not in backend_main._ALLOWED_LATIN_WORDS
                ]
                if latin_words:
                    problems.append(f"{field_name}: Latin word(s) not allowed {latin_words}")

        if problems:
            lines.append(
                f"  id={cid}: REJECTED by validator -> replaced with rules fallback text. "
                f"Reason(s): " + "; ".join(problems)
            )
        elif risk == reason:
            lines.append(
                f"  id={cid}: risk text == reason text -> validator keeps the reason but "
                f"replaces risk with the 'not_verified' fallback line."
            )
        else:
            lines.append(f"  id={cid}: ACCEPTED as-is (source=llm).")

    missing = [rid for rid in rec_by_id if rid not in seen_ids]
    if missing:
        lines.append(
            f"  MISSING from LLM output entirely (never returned, or only returned with "
            f"an unknown id) -> fallback(missing): {missing}"
        )

    return "\n".join(lines)


async def debug_one(client: httpx.AsyncClient, q: dict, out_lines: list):
    qid, lang = q["id"], q["lang"]
    out_lines.append("=" * 70)
    out_lines.append(f"{qid}  ({q['module']}, {lang})  pipeline C (sarvam-105b)")
    out_lines.append(f"Hardcoded input: {cp.CROP_HARDCODED_INPUTS[qid]}")
    out_lines.append("=" * 70)

    recommendations, weather = await cp.fetch_crop_recommendations(client, qid)
    if not recommendations:
        out_lines.append("crop_api returned NO recommendations for this input — nothing to test.")
        out_lines.append("")
        return

    system_prompt, user_content = cp.build_crop_prompt(recommendations, weather, lang)
    out_lines.append(f"Recommended crop ids from crop_api: {[r['id'] for r in recommendations]}")
    out_lines.append("")

    try:
        raw_text = await cp.sarvam_chat(
            client, system_prompt, user_content, max_tokens=backend_main.CROP_LLM_MAX_TOKENS
        )
    except Exception as e:
        out_lines.append(f"Sarvam chat call FAILED: {type(e).__name__}: {e}")
        out_lines.append("")
        return

    out_lines.append("--- RAW Sarvam text ---")
    out_lines.append(raw_text)
    out_lines.append("")
    out_lines.append("--- Validator diagnosis ---")
    out_lines.append(diagnose_crop_items(raw_text, recommendations, lang))
    out_lines.append("")


async def run_debug():
    if not cp.SARVAM_API_KEY:
        print("SARVAM_API_KEY not set in backend/.env — aborting.")
        return

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = {q["id"]: q for q in json.load(f)}

    out_lines = []
    async with httpx.AsyncClient() as client:
        for i, qid in enumerate(QIDS):
            if i > 0:
                await asyncio.sleep(cp.API_SLEEP_SECONDS)
            await debug_one(client, questions[qid], out_lines)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(run_debug())
