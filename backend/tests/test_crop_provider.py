# Tests for /api/crop's Sarvam-first routing for mr/hi (CROP_LLM_PROVIDER,
# pick_provider, _crop_try_sarvam). No real network calls: crop_api and
# Sarvam are both reached through httpx, so we patch httpx.AsyncClient.post
# and dispatch by URL; the Nemotron path goes through main._crop_llm_complete,
# which we patch directly (same pattern as test_crop_route.py).
import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from main import app
from models import sarvam_client

client = TestClient(app)

CROP_REQUEST_BODY_MR = {
    "taluka": "Niphad",
    "sowing_month": 10,
    "irrigation": "yes",
    "previous_crop": "maize",
    "soil_type": "heavy",
    "language": "mr",
}

CROP_REQUEST_BODY_EN = {**CROP_REQUEST_BODY_MR, "language": "en"}

FAKE_RECOMMENDATIONS = [
    {
        "id": "wheat",
        "names": {"en": "Wheat", "mr": "गहू", "hi": "गेहूं", "reviewed": False},
        "sowing_months": [10, 11],
        "score": 115,
        "reasons": [{"code": "good_rotation", "text": "Good to grow after maize"}],
        "evidence": [{"page": 3, "section": "2.1.1", "text": "Wheat grows well in heavy soil."}],
    },
    {
        "id": "chickpea",
        "names": {"en": "Chickpea", "mr": "हरभरा", "hi": "चना", "reviewed": False},
        "sowing_months": [10],
        "score": 100,
        "reasons": [{"code": "sowing_unknown", "text": "Sowing window not verified"}],
        "evidence": [],
    },
]

FAKE_CROP_API_RESPONSE = {
    "success": True,
    "recommendations": FAKE_RECOMMENDATIONS,
    "source": "rules",
    "weather": {"rain_last_14_days_mm": 5.0},
}

SARVAM_CROPS_JSON_MR = (
    '{"crops":[{"id":"wheat","reason":"गव्हासाठी जमीन आणि हवामान अनुकूल आहे.","risk":"जास्त थंडी हानी करू शकते."},'
    '{"id":"chickpea","reason":"हरभरा कमी पाण्यात येतो.","risk":"किडीचा धोका कमी आहे."}]}'
)

NEMOTRON_CROPS_JSON_MR = (
    '{"crops":[{"id":"wheat","reason":"गव्हासाठी चांगली जमीन आहे.","risk":"कमी पावसाचा धोका."},'
    '{"id":"chickpea","reason":"कमी पाण्यात येणारे पीक आहे.","risk":"किडीचा धोका कमी."}]}'
)

NEMOTRON_CROPS_JSON_EN = (
    '{"crops":[{"id":"wheat","reason":"Good soil and weather for wheat.","risk":"Minor cold risk."},'
    '{"id":"chickpea","reason":"Grows well with little water.","risk":"Low pest risk."}]}'
)


def _crop_api_response():
    request = httpx.Request("POST", "http://127.0.0.1:5003/recommend-crop")
    return httpx.Response(200, json=FAKE_CROP_API_RESPONSE, request=request)


def _sarvam_response(content: str, status_code: int = 200):
    request = httpx.Request("POST", sarvam_client.SARVAM_CHAT_URL)
    body = {"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 10, "completion_tokens": 20}}
    return httpx.Response(status_code, json=body, request=request)


def _dispatch_post(calls: list, sarvam_result):
    """Sync side_effect for httpx.AsyncClient.post: answers crop_api's
    /recommend-crop with FAKE_CROP_API_RESPONSE and Sarvam's chat endpoint
    with `sarvam_result` (a canned httpx.Response, or an Exception instance
    to raise). Records every URL hit in `calls` so tests can assert whether
    Sarvam/crop_api were actually called."""
    def _dispatch(url, **kwargs):
        url_str = str(url)
        calls.append(url_str)
        if url_str.endswith("/recommend-crop"):
            return _crop_api_response()
        if url_str == sarvam_client.SARVAM_CHAT_URL:
            if isinstance(sarvam_result, Exception):
                raise sarvam_result
            return sarvam_result
        raise AssertionError(f"Unexpected POST to {url}")
    return _dispatch


@pytest.fixture
def sarvam_configured(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key-not-real")


def test_sarvam_success_is_used_and_nemotron_is_not_called(sarvam_configured):
    calls = []
    nemotron_mock = AsyncMock()
    with patch.object(httpx.AsyncClient, "post", AsyncMock(
            side_effect=_dispatch_post(calls, _sarvam_response(SARVAM_CROPS_JSON_MR)))), \
         patch.object(main, "_crop_llm_complete", nemotron_mock):
        response = client.post("/api/crop", json=CROP_REQUEST_BODY_MR)
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "llm"
    assert body["provider"] == "sarvam"
    wheat = next(c for c in body["crops"] if c["id"] == "wheat")
    assert wheat["reason"] == "गव्हासाठी जमीन आणि हवामान अनुकूल आहे."
    nemotron_mock.assert_not_awaited()
    assert any(u == sarvam_client.SARVAM_CHAT_URL for u in calls)


def test_sarvam_broken_json_falls_back_to_nemotron(sarvam_configured):
    calls = []
    # Malformed JSON seen in practice (backend/scripts/debug_crop_c.txt):
    # a trailing comma before the closing bracket.
    broken_json = '{"crops":[{"id":"wheat","reason":"ok","risk":"ok",}]}'
    with patch.object(httpx.AsyncClient, "post", AsyncMock(
            side_effect=_dispatch_post(calls, _sarvam_response(broken_json)))), \
         patch.object(main, "_crop_llm_complete", AsyncMock(return_value=NEMOTRON_CROPS_JSON_MR)):
        response = client.post("/api/crop", json=CROP_REQUEST_BODY_MR)
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "llm"
    assert body["provider"] == "nemotron"
    wheat = next(c for c in body["crops"] if c["id"] == "wheat")
    assert wheat["reason"] == "गव्हासाठी चांगली जमीन आहे."


def test_sarvam_timeout_falls_back_to_nemotron(sarvam_configured):
    calls = []
    with patch.object(httpx.AsyncClient, "post", AsyncMock(
            side_effect=_dispatch_post(calls, httpx.ReadTimeout("timed out")))), \
         patch.object(main, "_crop_llm_complete", AsyncMock(return_value=NEMOTRON_CROPS_JSON_MR)):
        response = client.post("/api/crop", json=CROP_REQUEST_BODY_MR)
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "llm"
    assert body["provider"] == "nemotron"


def test_both_sarvam_and_nemotron_failing_falls_back_to_rules(sarvam_configured):
    calls = []
    with patch.object(httpx.AsyncClient, "post", AsyncMock(
            side_effect=_dispatch_post(calls, httpx.ReadTimeout("timed out")))), \
         patch.object(main, "_crop_llm_complete", AsyncMock(side_effect=RuntimeError("nemotron down"))):
        response = client.post("/api/crop", json=CROP_REQUEST_BODY_MR)
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "rules"
    assert body["provider"] is None
    wheat = next(c for c in body["crops"] if c["id"] == "wheat")
    assert wheat["reason"] == main._crop_reason_text("good_rotation", "mr")


def test_cap_reached_skips_sarvam_entirely_and_uses_nemotron(sarvam_configured, monkeypatch):
    monkeypatch.setenv("SARVAM_MAX_REQUESTS_PER_DAY", "0")
    calls = []
    with patch.object(httpx.AsyncClient, "post", AsyncMock(
            side_effect=_dispatch_post(calls, _sarvam_response(SARVAM_CROPS_JSON_MR)))), \
         patch.object(main, "_crop_llm_complete", AsyncMock(return_value=NEMOTRON_CROPS_JSON_MR)):
        response = client.post("/api/crop", json=CROP_REQUEST_BODY_MR)
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "llm"
    assert body["provider"] == "nemotron"
    # The cap must stop sarvam_chat() before it ever calls Sarvam's HTTP API.
    assert not any(u == sarvam_client.SARVAM_CHAT_URL for u in calls)


def test_english_request_uses_nemotron_only_sarvam_never_tried(sarvam_configured):
    calls = []
    with patch.object(httpx.AsyncClient, "post", AsyncMock(
            side_effect=_dispatch_post(calls, _sarvam_response(SARVAM_CROPS_JSON_MR)))), \
         patch.object(main, "_crop_llm_complete", AsyncMock(return_value=NEMOTRON_CROPS_JSON_EN)):
        response = client.post("/api/crop", json=CROP_REQUEST_BODY_EN)
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "llm"
    assert body["provider"] == "nemotron"
    assert not any(u == sarvam_client.SARVAM_CHAT_URL for u in calls)


def test_sarvam_timeout_reduces_nemotron_timeout(sarvam_configured):
    """When Sarvam eats part of the 12s chain budget before timing out,
    Nemotron must be called with a smaller total_timeout (12s minus what
    Sarvam spent), not the full default -- so the farmer isn't kept waiting
    the full budget twice."""
    calls = []
    sarvam_delay_s = 0.5

    async def slow_sarvam_then_timeout(url, **kwargs):
        url_str = str(url)
        calls.append(url_str)
        if url_str.endswith("/recommend-crop"):
            return _crop_api_response()
        if url_str == sarvam_client.SARVAM_CHAT_URL:
            await asyncio.sleep(sarvam_delay_s)
            raise httpx.ReadTimeout("timed out")
        raise AssertionError(f"Unexpected POST to {url}")

    captured_timeouts = []

    async def fake_crop_llm_complete(messages, total_timeout=None):
        captured_timeouts.append(total_timeout)
        return NEMOTRON_CROPS_JSON_MR

    with patch.object(httpx.AsyncClient, "post", AsyncMock(side_effect=slow_sarvam_then_timeout)), \
         patch.object(main, "_crop_llm_complete", fake_crop_llm_complete):
        response = client.post("/api/crop", json=CROP_REQUEST_BODY_MR)

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "nemotron"

    assert len(captured_timeouts) == 1
    used_timeout = captured_timeouts[0]
    assert used_timeout is not None
    # Clearly reduced from the full 12s budget (Sarvam spent ~0.5s), but
    # still well above the 3s floor.
    assert used_timeout < main.CROP_PROVIDER_TOTAL_TIMEOUT_S - 0.3
    assert used_timeout > main.CROP_LLM_MIN_FALLBACK_TIMEOUT_S
