# Tests for /api/disease's Sarvam-first explanation (cause/today/prevent),
# Phase 2 of the Sarvam integration (see test_crop_provider.py for Phase 1).
# No real network calls: disease_api and Sarvam are both reached through
# httpx, so we patch httpx.AsyncClient.post and dispatch by URL; the
# Nemotron path goes through main._crop_llm_complete (the shared, generic
# hedge-call helper), which we patch directly.
import asyncio
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

FAKE_DISEASE_API_RESPONSE_OK = {
    "crop": "Onion",
    "status": "ok",
    "message": "The most likely problem is Purple Blotch.",
    "note": "",
    "precautions": [],
    "threshold": 0.7,
    "top": [
        {"class": "Onion___Purple_Blotch", "disease": "Purple Blotch", "is_healthy": False, "probability": 0.92},
        {"class": "Onion___Alternaria", "disease": "Alternaria", "is_healthy": False, "probability": 0.04},
    ],
}

FAKE_DISEASE_API_RESPONSE_LOW_CONFIDENCE = {
    **FAKE_DISEASE_API_RESPONSE_OK,
    "top": [
        {"class": "Onion___Purple_Blotch", "disease": "Purple Blotch", "is_healthy": False, "probability": 0.75},
        {"class": "Onion___Alternaria", "disease": "Alternaria", "is_healthy": False, "probability": 0.1},
    ],
}

FAKE_DISEASE_API_RESPONSE_HEALTHY = {
    "crop": "Onion",
    "status": "ok",
    "message": "The onion leaf looks healthy.",
    "note": "",
    "precautions": [],
    "threshold": 0.7,
    "top": [{"class": "Onion___Healthy", "disease": "Healthy", "is_healthy": True, "probability": 0.98}],
}

FAKE_DISEASE_API_RESPONSE_NOT_SURE = {
    "crop": "Onion",
    "status": "not_sure",
    "message": "Not sure -- please try another photo.",
    "note": "",
    "precautions": [],
    "threshold": 0.7,
    "top": [{"class": "Onion___Purple_Blotch", "disease": "Purple Blotch", "is_healthy": False, "probability": 0.5}],
}

SARVAM_EXPLANATION_JSON_MR = (
    '{"cause":"हा बुरशीजन्य रोग आहे.","today":"रोगट पाने काढून टाका.","prevent":"झाडांमध्ये योग्य अंतर ठेवा."}'
)
NEMOTRON_EXPLANATION_JSON_MR = (
    '{"cause":"हा रोग बुरशीमुळे होतो.","today":"रोगट भाग काढून टाका.","prevent":"पाणी पानांवर देऊ नका."}'
)
NEMOTRON_EXPLANATION_JSON_EN = (
    '{"cause":"This is a fungal disease.","today":"Remove infected leaves.","prevent":"Keep plants well spaced."}'
)


def _disease_api_response(body):
    request = httpx.Request("POST", "http://127.0.0.1:5002/predict-disease")
    return httpx.Response(200, json=body, request=request)


def _sarvam_response(content: str, status_code: int = 200):
    request = httpx.Request("POST", sarvam_client.SARVAM_CHAT_URL)
    body = {"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 10, "completion_tokens": 20}}
    return httpx.Response(status_code, json=body, request=request)


def _dispatch_post(calls, sarvam_result, disease_body=FAKE_DISEASE_API_RESPONSE_OK):
    """Sync side_effect for httpx.AsyncClient.post: answers disease_api's
    /predict-disease with `disease_body` and Sarvam's chat endpoint with
    `sarvam_result` (a canned httpx.Response, or an Exception to raise)."""
    def _dispatch(url, **kwargs):
        url_str = str(url)
        calls.append(url_str)
        if url_str.endswith("/predict-disease"):
            return _disease_api_response(disease_body)
        if url_str == sarvam_client.SARVAM_CHAT_URL:
            if isinstance(sarvam_result, Exception):
                raise sarvam_result
            return sarvam_result
        raise AssertionError(f"Unexpected POST to {url}")
    return _dispatch


def _post_disease(language="mr"):
    return client.post(
        "/api/disease",
        files={"image": ("leaf.jpg", b"fake-bytes", "image/jpeg")},
        data={"crop": "Onion", "language": language},
    )


@pytest.fixture
def sarvam_configured(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key-not-real")


def test_sarvam_success_is_used_and_nemotron_is_not_called(sarvam_configured):
    calls = []
    nemotron_mock = AsyncMock()
    with patch.object(httpx.AsyncClient, "post", AsyncMock(
            side_effect=_dispatch_post(calls, _sarvam_response(SARVAM_EXPLANATION_JSON_MR)))), \
         patch.object(main, "_crop_llm_complete", nemotron_mock):
        response = _post_disease("mr")
    assert response.status_code == 200
    body = response.json()
    assert body["explanation_provider"] == "sarvam"
    assert body["explanation"]["cause"] == "हा बुरशीजन्य रोग आहे."
    nemotron_mock.assert_not_awaited()
    assert any(u == sarvam_client.SARVAM_CHAT_URL for u in calls)


def test_sarvam_broken_json_falls_back_to_nemotron(sarvam_configured):
    calls = []
    broken_json = '{"cause":"ok","today":"ok",}'  # trailing comma -- invalid JSON
    with patch.object(httpx.AsyncClient, "post", AsyncMock(
            side_effect=_dispatch_post(calls, _sarvam_response(broken_json)))), \
         patch.object(main, "_crop_llm_complete", AsyncMock(return_value=NEMOTRON_EXPLANATION_JSON_MR)):
        response = _post_disease("mr")
    assert response.status_code == 200
    body = response.json()
    assert body["explanation_provider"] == "nemotron"
    assert body["explanation"]["cause"] == "हा रोग बुरशीमुळे होतो."


def test_sarvam_timeout_reduces_nemotron_timeout(sarvam_configured):
    async def slow_sarvam_then_timeout(url, **kwargs):
        url_str = str(url)
        if url_str.endswith("/predict-disease"):
            return _disease_api_response(FAKE_DISEASE_API_RESPONSE_OK)
        if url_str == sarvam_client.SARVAM_CHAT_URL:
            await asyncio.sleep(0.5)
            raise httpx.ReadTimeout("timed out")
        raise AssertionError(f"Unexpected POST to {url}")

    captured_timeouts = []

    async def fake_crop_llm_complete(messages, total_timeout=None):
        captured_timeouts.append(total_timeout)
        return NEMOTRON_EXPLANATION_JSON_MR

    with patch.object(httpx.AsyncClient, "post", AsyncMock(side_effect=slow_sarvam_then_timeout)), \
         patch.object(main, "_crop_llm_complete", fake_crop_llm_complete):
        response = _post_disease("mr")

    assert response.status_code == 200
    body = response.json()
    assert body["explanation_provider"] == "nemotron"
    assert len(captured_timeouts) == 1
    used_timeout = captured_timeouts[0]
    assert used_timeout is not None
    assert used_timeout < main.CROP_PROVIDER_TOTAL_TIMEOUT_S - 0.3
    assert used_timeout > main.CROP_LLM_MIN_FALLBACK_TIMEOUT_S


def test_both_sarvam_and_nemotron_failing_falls_back_to_rules(sarvam_configured):
    calls = []
    with patch.object(httpx.AsyncClient, "post", AsyncMock(
            side_effect=_dispatch_post(calls, httpx.ReadTimeout("timed out")))), \
         patch.object(main, "_crop_llm_complete", AsyncMock(side_effect=RuntimeError("nemotron down"))):
        response = _post_disease("mr")
    assert response.status_code == 200
    body = response.json()
    assert body["explanation_provider"] is None
    assert body["explanation"]["today"] == main._category_precautions("fungal", "mr")[0]


def test_cap_reached_skips_sarvam_entirely_and_uses_nemotron(sarvam_configured, monkeypatch):
    monkeypatch.setenv("SARVAM_MAX_REQUESTS_PER_DAY", "0")
    calls = []
    with patch.object(httpx.AsyncClient, "post", AsyncMock(
            side_effect=_dispatch_post(calls, _sarvam_response(SARVAM_EXPLANATION_JSON_MR)))), \
         patch.object(main, "_crop_llm_complete", AsyncMock(return_value=NEMOTRON_EXPLANATION_JSON_MR)):
        response = _post_disease("mr")
    assert response.status_code == 200
    body = response.json()
    assert body["explanation_provider"] == "nemotron"
    assert not any(u == sarvam_client.SARVAM_CHAT_URL for u in calls)


def test_english_request_uses_nemotron_only_sarvam_never_tried(sarvam_configured):
    calls = []
    with patch.object(httpx.AsyncClient, "post", AsyncMock(
            side_effect=_dispatch_post(calls, _sarvam_response(SARVAM_EXPLANATION_JSON_MR)))), \
         patch.object(main, "_crop_llm_complete", AsyncMock(return_value=NEMOTRON_EXPLANATION_JSON_EN)):
        response = _post_disease("en")
    assert response.status_code == 200
    body = response.json()
    assert body["explanation_provider"] == "nemotron"
    assert not any(u == sarvam_client.SARVAM_CHAT_URL for u in calls)


def test_unsafe_pesticide_content_falls_back_to_rules(sarvam_configured):
    unsafe_json = '{"cause":"ठीक आहे.","today":"रोगासाठी फवारणी करा.","prevent":"ठीक आहे."}'
    with patch.object(httpx.AsyncClient, "post", AsyncMock(
            side_effect=_dispatch_post([], _sarvam_response(unsafe_json)))), \
         patch.object(main, "_crop_llm_complete", AsyncMock(return_value=NEMOTRON_EXPLANATION_JSON_MR)):
        response = _post_disease("mr")
    assert response.status_code == 200
    body = response.json()
    # "फवारणी" (spraying) in "today" -> whole explanation falls back to rules,
    # even though "cause"/"prevent" were themselves fine.
    assert body["explanation_provider"] is None


def test_healthy_result_never_calls_any_llm(sarvam_configured):
    calls = []
    with patch.object(httpx.AsyncClient, "post", AsyncMock(
            side_effect=_dispatch_post(
                calls, _sarvam_response(SARVAM_EXPLANATION_JSON_MR), FAKE_DISEASE_API_RESPONSE_HEALTHY
            ))):
        response = _post_disease("mr")
    assert response.status_code == 200
    body = response.json()
    assert body["explanation"] is None
    assert body["explanation_provider"] is None
    assert not any(u == sarvam_client.SARVAM_CHAT_URL for u in calls)


def test_not_sure_result_never_calls_any_llm(sarvam_configured):
    calls = []
    with patch.object(httpx.AsyncClient, "post", AsyncMock(
            side_effect=_dispatch_post(
                calls, _sarvam_response(SARVAM_EXPLANATION_JSON_MR), FAKE_DISEASE_API_RESPONSE_NOT_SURE
            ))):
        response = _post_disease("mr")
    assert response.status_code == 200
    body = response.json()
    assert body["explanation"] is None
    assert not any(u == sarvam_client.SARVAM_CHAT_URL for u in calls)


def test_low_confidence_tells_llm_to_mention_kvk(sarvam_configured):
    captured_messages = []

    async def fake_sarvam_capture(messages, **kwargs):
        captured_messages.append(messages)
        raise RuntimeError("force rules fallback -- this test only checks the prompt sent")

    with patch.object(main, "sarvam_chat", fake_sarvam_capture), \
         patch.object(httpx.AsyncClient, "post", AsyncMock(
             side_effect=_dispatch_post([], httpx.ReadTimeout("n/a"), FAKE_DISEASE_API_RESPONSE_LOW_CONFIDENCE))), \
         patch.object(main, "_crop_llm_complete", AsyncMock(return_value=NEMOTRON_EXPLANATION_JSON_MR)):
        response = _post_disease("mr")
    assert response.status_code == 200
    assert len(captured_messages) == 1
    system_prompt = captured_messages[0][0]["content"]
    assert "confirm with the local KVK" in system_prompt


def test_high_confidence_does_not_mention_kvk_in_prompt(sarvam_configured):
    captured_messages = []

    async def fake_sarvam_capture(messages, **kwargs):
        captured_messages.append(messages)
        raise RuntimeError("force rules fallback -- this test only checks the prompt sent")

    with patch.object(main, "sarvam_chat", fake_sarvam_capture), \
         patch.object(httpx.AsyncClient, "post", AsyncMock(
             side_effect=_dispatch_post([], httpx.ReadTimeout("n/a"), FAKE_DISEASE_API_RESPONSE_OK))), \
         patch.object(main, "_crop_llm_complete", AsyncMock(return_value=NEMOTRON_EXPLANATION_JSON_MR)):
        response = _post_disease("mr")
    assert response.status_code == 200
    assert len(captured_messages) == 1
    system_prompt = captured_messages[0][0]["content"]
    assert "Do not mention the KVK" in system_prompt
    assert "confirm with the local KVK" not in system_prompt
