"""
sarvam_client.py — minimal client for Sarvam AI's sarvam-105b chat model
(https://api.sarvam.ai/v1/chat/completions), used for mr/hi crop
explanations (see CLAUDE.md, Module 2). Nemotron (backend/main.py) stays
the fallback, and the only path for English.

Usage cap: a free-tier key can run out mid-day, so every call attempt is
counted in backend/data/sarvam_usage.json (git-ignored) against
SARVAM_MAX_REQUESTS_PER_DAY, reset at IST midnight. Once the cap is hit,
sarvam_chat() raises SarvamCapReachedError immediately -- before making any
HTTP call -- so the caller can fall back to Nemotron without waiting.
"""

import os
import json
import time
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

SARVAM_CHAT_URL = "https://api.sarvam.ai/v1/chat/completions"
SARVAM_MODEL = "sarvam-105b"
IST = timezone(timedelta(hours=5, minutes=30))

USAGE_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sarvam_usage.json")
)

# Guards the usage-counter file against two concurrent requests both
# reading "299" and both writing "300" (we'd then let through a 301st call).
_usage_lock = asyncio.Lock()


class SarvamError(Exception):
    """Base class for every error sarvam_chat() raises."""


class SarvamNotConfiguredError(SarvamError):
    """SARVAM_API_KEY isn't set in backend/.env."""


class SarvamCapReachedError(SarvamError):
    """SARVAM_MAX_REQUESTS_PER_DAY already used up today (IST)."""


class SarvamHTTPError(SarvamError):
    """Sarvam answered with a non-2xx status."""

    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"Sarvam HTTP {status_code}: {body[:300]}")


def _today_ist() -> str:
    return datetime.now(IST).date().isoformat()


def _read_usage() -> dict:
    try:
        with open(USAGE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("date") == _today_ist() and isinstance(data.get("count"), int):
            return data
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {"date": _today_ist(), "count": 0}


def _write_usage(data: dict) -> None:
    os.makedirs(os.path.dirname(USAGE_PATH), exist_ok=True)
    with open(USAGE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)


async def _reserve_request_slot() -> None:
    """Raise SarvamCapReachedError if today's (IST) count is already at
    SARVAM_MAX_REQUESTS_PER_DAY, else count this attempt and continue.
    Counts every attempt, not just successes, so a run of failed calls
    can't be used to dodge the daily cap."""
    max_per_day = int(os.getenv("SARVAM_MAX_REQUESTS_PER_DAY", "300"))
    async with _usage_lock:
        usage = _read_usage()
        if usage["count"] >= max_per_day:
            raise SarvamCapReachedError(
                f"Sarvam daily cap reached ({usage['count']}/{max_per_day} for {usage['date']})"
            )
        usage["count"] += 1
        _write_usage(usage)


async def sarvam_chat(
    messages: list,
    response_format: Optional[dict] = None,
    max_tokens: int = 800,
    timeout_s: Optional[float] = None,
) -> str:
    """
    Call sarvam-105b with `messages` (OpenAI-style chat messages) and return
    the assistant's text. `response_format`, if given, is passed through
    unchanged (e.g. an OpenAI-style strict json_schema) so the caller can
    force structured output. `timeout_s`, if given, overrides the default
    SARVAM_TIMEOUT_S env setting for this one call only (used by chat's
    pipeline B2/C, whose own per-step budgets can be smaller than crop/
    disease's shared default).

    reasoning_effort is ALWAYS None -- sarvam-105b otherwise spends its whole
    token budget "thinking" and returns an empty answer (seen while testing
    in backend/scripts/check_sarvam.py).

    Raises SarvamNotConfiguredError, SarvamCapReachedError, SarvamHTTPError,
    or an httpx.TimeoutException/httpx.RequestError on failure. The key is
    never logged.
    """
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise SarvamNotConfiguredError("SARVAM_API_KEY is not set in backend/.env")

    await _reserve_request_slot()

    timeout = timeout_s if timeout_s is not None else float(os.getenv("SARVAM_TIMEOUT_S", "6"))
    payload = {
        "model": SARVAM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "reasoning_effort": None,
        "temperature": 0.2,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                SARVAM_CHAT_URL,
                headers={"api-subscription-key": api_key},
                json=payload,
            )
    except httpx.TimeoutException:
        latency_ms = round((time.perf_counter() - t0) * 1000)
        logger.warning("Sarvam call failed: timed out after %sms", latency_ms)
        raise
    except httpx.RequestError as e:
        latency_ms = round((time.perf_counter() - t0) * 1000)
        logger.warning("Sarvam call failed: %s (after %sms)", e, latency_ms)
        raise

    latency_ms = round((time.perf_counter() - t0) * 1000)

    if resp.status_code != 200:
        logger.warning(
            "Sarvam call failed: HTTP %s after %sms. Body: %s",
            resp.status_code, latency_ms, resp.text[:300],
        )
        raise SarvamHTTPError(resp.status_code, resp.text)

    data = resp.json()
    usage = data.get("usage") or {}
    logger.info(
        "Sarvam call ok: %sms prompt_tokens=%s completion_tokens=%s",
        latency_ms, usage.get("prompt_tokens"), usage.get("completion_tokens"),
    )
    return (data["choices"][0]["message"].get("content") or "").strip()
