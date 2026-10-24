# Tests for /api/chat's pipeline B2 for mr/hi (CHAT_LLM_PROVIDER,
# _chat_pipeline_b2): Nemotron answers the farmer's original mr/hi question
# in English, sarvam-105b rewrites it to natural mr/hi, with a pipeline C
# (sarvam answers directly in mr/hi) and today's plain _chat_direct as
# fallbacks. No real network calls: the Nemotron English-answer step goes
# through main._chat_llm_complete (patched directly, same pattern as
# test_crop_provider.py's main._crop_llm_complete); Sarvam is reached
# through httpx, so httpx.AsyncClient.post is patched and dispatches
# canned responses in call order.
import asyncio
import os
import sys
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from main import app, ChatResponse
from models import sarvam_client

client = TestClient(app)

CHAT_REQUEST_BODY_MR = {"text": "माझ्या कापसाच्या पानांवर डाग आहेत", "reply_language": "mr"}
CHAT_REQUEST_BODY_EN = {"text": "My cotton leaves have spots", "reply_language": "en"}

ENGLISH_ANSWER = "Remove the affected leaves and keep the plants well spaced. Contact your local KVK for the right product."
REWRITTEN_MR_OK = "रोगट पाने काढून टाका आणि झाडांमध्ये योग्य अंतर ठेवा. योग्य औषधासाठी स्थानिक केव्हीकेशी संपर्क करा."
PIPELINE_C_MR_OK = "रोगट पाने काढून झाडे निरोगी ठेवा. योग्य सल्ल्यासाठी केव्हीकेशी संपर्क करा."

# The fixed "couldn't answer" message main._chat_fallback_message("mr")
# returns when both pipeline B2 and pipeline C fail/are unavailable --
# read from the real data file so this test doesn't duplicate its text.
FALLBACK_TEXT_MR = main._chat_fallback_message("mr")

NEMOTRON_DIRECT_RESPONSE = ChatResponse(
    response="आजचा सल्ला उपलब्ध नाही (नेमोट्रॉन डायरेक्ट).",
    response_en="आजचा सल्ला उपलब्ध नाही (नेमोट्रॉन डायरेक्ट).",
    question_en=CHAT_REQUEST_BODY_MR["text"],
    timings={"translate_in": 0, "llm": 0, "translate_out": 0, "total": 0},
    translation_failed=False,
)

EN_DIRECT_RESPONSE = ChatResponse(
    response="Remove the affected leaves and contact your local KVK.",
    response_en="Remove the affected leaves and contact your local KVK.",
    question_en=CHAT_REQUEST_BODY_EN["text"],
    timings={"translate_in": 0, "llm": 0, "translate_out": 0, "total": 0},
    translation_failed=False,
)


def _sarvam_response(content: str, status_code: int = 200):
    request = httpx.Request("POST", sarvam_client.SARVAM_CHAT_URL)
    body = {"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 10, "completion_tokens": 20}}
    return httpx.Response(status_code, json=body, request=request)


def _dispatch_sarvam_sequence(calls: list, responses: list):
    """Sync side_effect for httpx.AsyncClient.post: each call to Sarvam's
    chat endpoint returns the next item of `responses` in order (an
    httpx.Response, or an Exception instance to raise)."""
    it = iter(responses)

    def _dispatch(url, **kwargs):
        calls.append(str(url))
        try:
            result = next(it)
        except StopIteration:
            raise AssertionError(f"Unexpected extra Sarvam call: {url}")
        if isinstance(result, Exception):
            raise result
        return result

    return _dispatch


@pytest.fixture
def sarvam_configured(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key-not-real")


def test_b2_success_rewrite_used_pipeline_c_not_tried(sarvam_configured):
    calls = []
    with patch.object(main, "_chat_llm_complete", AsyncMock(return_value=ENGLISH_ANSWER)), \
         patch.object(httpx.AsyncClient, "post", AsyncMock(
             side_effect=_dispatch_sarvam_sequence(calls, [_sarvam_response(REWRITTEN_MR_OK)]))):
        response = client.post("/api/chat", json=CHAT_REQUEST_BODY_MR)
    assert response.status_code == 200
    body = response.json()
    assert body["response"] == REWRITTEN_MR_OK
    assert body["response_en"] == ENGLISH_ANSWER
    assert body["translation_failed"] is False
    assert len(calls) == 1  # only the rewrite call, no pipeline C


def test_nemotron_timeout_falls_back_to_pipeline_c(sarvam_configured):
    calls = []
    with patch.object(main, "_chat_llm_complete", AsyncMock(side_effect=asyncio.TimeoutError())), \
         patch.object(httpx.AsyncClient, "post", AsyncMock(
             side_effect=_dispatch_sarvam_sequence(calls, [_sarvam_response(PIPELINE_C_MR_OK)]))):
        response = client.post("/api/chat", json=CHAT_REQUEST_BODY_MR)
    assert response.status_code == 200
    body = response.json()
    assert body["response"] == PIPELINE_C_MR_OK
    # No real English answer exists on the pipeline C path -- must be
    # empty, not filled with the mr text under an "_en" name.
    assert body["response_en"] == ""
    assert len(calls) == 1  # only pipeline C's direct call


def test_rewrite_rejected_falls_back_to_pipeline_c(sarvam_configured):
    calls = []
    # The rewrite call "succeeds" HTTP-wise but answers in English instead
    # of mr -- caught by the wrong-language check, same as crop/disease.
    bad_rewrite = "Remove the affected leaves and keep plants spaced."
    with patch.object(main, "_chat_llm_complete", AsyncMock(return_value=ENGLISH_ANSWER)), \
         patch.object(httpx.AsyncClient, "post", AsyncMock(
             side_effect=_dispatch_sarvam_sequence(
                 calls, [_sarvam_response(bad_rewrite), _sarvam_response(PIPELINE_C_MR_OK)]
             ))):
        response = client.post("/api/chat", json=CHAT_REQUEST_BODY_MR)
    assert response.status_code == 200
    body = response.json()
    assert body["response"] == PIPELINE_C_MR_OK
    assert body["response_en"] == ""
    assert len(calls) == 2  # rewrite attempt, then pipeline C


def test_nemotron_and_pipeline_c_both_fail_uses_fixed_message(sarvam_configured):
    """Both steps failing must NOT call _chat_direct again (Nemotron likely
    just failed/is slow; retrying it would only make the farmer wait
    longer) -- it must return the short fixed message instead."""
    calls = []
    with patch.object(main, "_chat_llm_complete", AsyncMock(side_effect=RuntimeError("nemotron down"))), \
         patch.object(httpx.AsyncClient, "post", AsyncMock(
             side_effect=_dispatch_sarvam_sequence(calls, [httpx.ReadTimeout("timed out")]))), \
         patch.object(main, "_chat_direct", AsyncMock(return_value=NEMOTRON_DIRECT_RESPONSE)) as chat_direct_mock:
        response = client.post("/api/chat", json=CHAT_REQUEST_BODY_MR)
    assert response.status_code == 200
    body = response.json()
    assert body["response"] == FALLBACK_TEXT_MR
    assert body["response_en"] == ""
    chat_direct_mock.assert_not_awaited()


def test_chat_llm_provider_nemotron_skips_b2_and_c_entirely(sarvam_configured, monkeypatch):
    """CHAT_LLM_PROVIDER=nemotron is the one exception: chat for mr/hi keeps
    using _chat_direct exactly as before this phase, never touching pipeline
    B2/C."""
    monkeypatch.setattr(main, "CHAT_LLM_PROVIDER", "nemotron")
    with patch.object(main, "_chat_direct", AsyncMock(return_value=NEMOTRON_DIRECT_RESPONSE)) as chat_direct_mock, \
         patch.object(main, "_chat_llm_complete", AsyncMock()) as chat_llm_complete_mock, \
         patch.object(httpx.AsyncClient, "post", AsyncMock(
             side_effect=AssertionError("Sarvam must not be called when CHAT_LLM_PROVIDER=nemotron"))):
        response = client.post("/api/chat", json=CHAT_REQUEST_BODY_MR)
    assert response.status_code == 200
    body = response.json()
    assert body["response"] == NEMOTRON_DIRECT_RESPONSE.response
    chat_direct_mock.assert_awaited_once()
    chat_llm_complete_mock.assert_not_awaited()


def test_english_request_uses_chat_direct_pipeline_b2_never_tried():
    with patch.object(main, "_chat_direct", AsyncMock(return_value=EN_DIRECT_RESPONSE)) as chat_direct_mock, \
         patch.object(main, "_chat_pipeline_b2", AsyncMock()) as pipeline_b2_mock:
        response = client.post("/api/chat", json=CHAT_REQUEST_BODY_EN)
    assert response.status_code == 200
    body = response.json()
    assert body["response"] == EN_DIRECT_RESPONSE.response
    chat_direct_mock.assert_awaited_once()
    pipeline_b2_mock.assert_not_awaited()


def test_cap_reached_skips_sarvam_entirely_and_uses_fixed_message(sarvam_configured, monkeypatch):
    monkeypatch.setenv("SARVAM_MAX_REQUESTS_PER_DAY", "0")
    calls = []
    with patch.object(main, "_chat_llm_complete", AsyncMock(return_value=ENGLISH_ANSWER)), \
         patch.object(httpx.AsyncClient, "post", AsyncMock(
             side_effect=_dispatch_sarvam_sequence(calls, []))), \
         patch.object(main, "_chat_direct", AsyncMock(return_value=NEMOTRON_DIRECT_RESPONSE)) as chat_direct_mock:
        response = client.post("/api/chat", json=CHAT_REQUEST_BODY_MR)
    assert response.status_code == 200
    body = response.json()
    assert body["response"] == FALLBACK_TEXT_MR
    assert body["response_en"] == ""
    chat_direct_mock.assert_not_awaited()
    # The cap must stop sarvam_chat() before it ever calls Sarvam's HTTP API,
    # for both the rewrite attempt and the pipeline C attempt.
    assert calls == []


# ─── _add_full_stops_to_bullets ──────────────────────────────────────────
# Bullet points sometimes come back from the LLM without a trailing full
# stop, which runs the points together in the chat bubble and gives
# text-to-speech no pause between them.

def test_add_full_stops_adds_missing_period():
    text = "• पहिला मुद्दा\n• Second point"
    result = main._add_full_stops_to_bullets(text)
    assert result == "• पहिला मुद्दा.\n• Second point."


def test_add_full_stops_leaves_existing_punctuation_alone():
    text = "• Already ends with a period.\n• Already a question?\n• Already excited!\n• पूर्ण वाक्य आहे।"
    result = main._add_full_stops_to_bullets(text)
    assert result == text


def test_add_full_stops_never_doubles_up_the_danda():
    text = "• ही माहिती पूर्ण आहे।"
    result = main._add_full_stops_to_bullets(text)
    assert result == text
    assert not result.endswith("।.")


def test_add_full_stops_ignores_non_bullet_lines():
    text = "Just a plain sentence with no bullet marker"
    result = main._add_full_stops_to_bullets(text)
    assert result == text


def test_add_full_stops_handles_empty_text():
    assert main._add_full_stops_to_bullets("") == ""
    assert main._add_full_stops_to_bullets(None) is None
