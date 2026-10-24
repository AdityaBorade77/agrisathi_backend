# Tests for /api/crop (Module 2 Phase 4b). No real network calls: crop_api
# is reached through httpx, so we patch httpx.AsyncClient.post; the Nemotron
# call goes through main._crop_llm_complete, which we patch directly.
import asyncio
import gc
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

client = TestClient(app)

CROP_REQUEST_BODY = {
    "taluka": "Niphad",
    "sowing_month": 10,
    "irrigation": "yes",
    "previous_crop": "maize",
    "soil_type": "heavy",
    "language": "mr",
}

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


def _mock_crop_api_response(json_body, status_code=200):
    request = httpx.Request("POST", "http://127.0.0.1:5003/recommend-crop")
    return httpx.Response(status_code, json=json_body, request=request)


def _mock_crop_api_get(url, *args, **kwargs):
    """Fake GET responses for /api/crop/options' three parallel calls to
    crop_api (talukas, soil-types, crops), keyed by the request URL."""
    request = httpx.Request("GET", url)
    if str(url).endswith("/talukas"):
        return httpx.Response(200, json={
            "talukas": [{"en": "Niphad", "mr": "निफाड", "hi": "निफाड", "reviewed": False}],
        }, request=request)
    if str(url).endswith("/soil-types"):
        return httpx.Response(200, json={
            "soil_types": [{"id": "heavy", "en": "Heavy soil", "mr": "भारी जमीन", "hi": "भारी मिट्टी", "reviewed": False}],
        }, request=request)
    if str(url).endswith("/crops"):
        return httpx.Response(200, json={
            "crops": [{"id": "wheat", "names": {"en": "Wheat", "mr": "गहू", "hi": "गेहूं", "reviewed": False}}],
        }, request=request)
    return httpx.Response(404, json={}, request=request)


def test_crop_options_combines_crop_api_endpoints():
    with patch.object(httpx.AsyncClient, "get", AsyncMock(side_effect=_mock_crop_api_get)):
        response = client.get("/api/crop/options?language=mr")
    assert response.status_code == 200
    body = response.json()
    assert body["talukas"][0]["en"] == "Niphad"
    assert body["soil_types"][0]["id"] == "heavy"
    assert body["crops"][0]["id"] == "wheat"


def test_crop_options_service_down_when_crop_api_unreachable():
    with patch.object(httpx.AsyncClient, "get", AsyncMock(side_effect=httpx.ConnectError("refused"))):
        response = client.get("/api/crop/options?language=mr")
    assert response.status_code == 503
    assert response.json()["detail"] == main._crop_message("service_down", "mr")


def test_service_down_when_crop_api_unreachable():
    with patch.object(httpx.AsyncClient, "post", AsyncMock(side_effect=httpx.ConnectError("refused"))):
        response = client.post("/api/crop", json=CROP_REQUEST_BODY)
    assert response.status_code == 503
    assert response.json()["detail"] == main._crop_message("service_down", "mr")


def test_crop_api_400_is_passed_through_not_service_down():
    bad_input_response = _mock_crop_api_response(
        {"success": False, "errors": ["Unknown taluka 'Nowhere'."]}, status_code=400
    )
    with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=bad_input_response)):
        response = client.post("/api/crop", json={**CROP_REQUEST_BODY, "taluka": "Nowhere"})
    assert response.status_code == 400
    assert response.json() == {"success": False, "errors": ["Unknown taluka 'Nowhere'."]}


def test_crop_api_500_is_still_service_down():
    server_error_response = _mock_crop_api_response({"error": "boom"}, status_code=500)
    with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=server_error_response)):
        response = client.post("/api/crop", json=CROP_REQUEST_BODY)
    assert response.status_code == 503
    assert response.json()["detail"] == main._crop_message("service_down", "mr")


def test_rain_in_millimetres_is_not_treated_as_a_dose():
    assert main._crop_text_is_unsafe("०.८ मिलीमीटर पाऊस झाला आहे.") is False


def test_marathi_dose_in_milliliters_is_unsafe():
    assert main._crop_text_is_unsafe("२ मिली वापरा.") is True


def test_english_word_containing_unit_letter_is_not_caught():
    assert main._crop_text_is_unsafe("2 good crops for this soil.") is False


def test_no_crops_message_when_recommendations_empty():
    empty_response = _mock_crop_api_response({"success": True, "recommendations": [], "weather": None})
    with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=empty_response)):
        response = client.post("/api/crop", json=CROP_REQUEST_BODY)
    assert response.status_code == 200
    body = response.json()
    assert body["crops"] == []
    assert body["source"] == "rules"
    assert body["message"] == main._crop_message("no_crops", "mr")


def test_valid_llm_reply_is_used_as_is():
    crop_api_response = _mock_crop_api_response(FAKE_CROP_API_RESPONSE)
    llm_json = (
        '{"crops":[{"id":"wheat","reason":"गव्हासाठी जमीन आणि हवामान अनुकूल आहे.","risk":"जास्त थंडी हानी करू शकते."},'
        '{"id":"chickpea","reason":"हरभरा कमी पाण्यात येतो.","risk":"किडीचा धोका कमी आहे."}]}'
    )
    with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=crop_api_response)), \
         patch.object(main, "_crop_llm_complete", AsyncMock(return_value=llm_json)):
        response = client.post("/api/crop", json=CROP_REQUEST_BODY)
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "llm"
    wheat = next(c for c in body["crops"] if c["id"] == "wheat")
    assert wheat["reason"] == "गव्हासाठी जमीन आणि हवामान अनुकूल आहे."
    assert wheat["name"] == "गहू"
    assert wheat["sowing_months"] == [10, 11]


def test_crop_prompt_includes_the_farmers_own_inputs():
    # The LLM's "reason" is supposed to explain the fit using the farmer's
    # own season/soil/irrigation/rotation -- it can only do that if those
    # inputs are actually sent in the prompt, not just crop_api's own
    # (already-derived) per-crop reasons.
    crop_api_response = _mock_crop_api_response(FAKE_CROP_API_RESPONSE)
    captured_messages = []

    async def fake_llm_complete(messages, total_timeout=None):
        captured_messages.append(messages)
        return (
            '{"crops":[{"id":"wheat","reason":"चांगली जमीन आहे.","risk":"कमी धोका आहे."},'
            '{"id":"chickpea","reason":"कमी पाण्यात येतो.","risk":"किडीचा धोका कमी."}]}'
        )

    with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=crop_api_response)), \
         patch.object(main, "_crop_llm_complete", fake_llm_complete):
        response = client.post("/api/crop", json=CROP_REQUEST_BODY)

    assert response.status_code == 200
    assert len(captured_messages) == 1
    user_content = captured_messages[0][1]["content"]
    assert f"taluka={CROP_REQUEST_BODY['taluka']}" in user_content
    assert f"sowing_month={CROP_REQUEST_BODY['sowing_month']}" in user_content
    assert f"irrigation={CROP_REQUEST_BODY['irrigation']}" in user_content
    assert f"previous_crop={CROP_REQUEST_BODY['previous_crop']}" in user_content
    assert f"soil_type={CROP_REQUEST_BODY['soil_type']}" in user_content


def test_llm_reply_with_unknown_crop_id_is_ignored():
    crop_api_response = _mock_crop_api_response(FAKE_CROP_API_RESPONSE)
    llm_json = (
        '{"crops":[{"id":"wheat","reason":"चांगली जमीन.","risk":"कमी धोका."},'
        '{"id":"mango","reason":"आंब्यासाठी चांगले.","risk":"काही नाही."}]}'
    )
    with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=crop_api_response)), \
         patch.object(main, "_crop_llm_complete", AsyncMock(return_value=llm_json)):
        response = client.post("/api/crop", json=CROP_REQUEST_BODY)
    assert response.status_code == 200
    body = response.json()
    ids = [c["id"] for c in body["crops"]]
    assert "mango" not in ids
    assert ids == ["wheat", "chickpea"]
    # chickpea wasn't in the LLM reply at all -> backfilled with fallback text
    chickpea = next(c for c in body["crops"] if c["id"] == "chickpea")
    assert chickpea["reason"] == main._crop_reason_text("sowing_unknown", "mr")


def test_llm_reply_with_pesticide_word_falls_back_for_that_crop():
    crop_api_response = _mock_crop_api_response(FAKE_CROP_API_RESPONSE)
    llm_json = (
        '{"crops":[{"id":"wheat","reason":"चांगली उगवण होईल.","risk":"रोगासाठी फवारणी करा."},'
        '{"id":"chickpea","reason":"कमी पाण्यात येतो.","risk":"किडीचा धोका कमी."}]}'
    )
    with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=crop_api_response)), \
         patch.object(main, "_crop_llm_complete", AsyncMock(return_value=llm_json)):
        response = client.post("/api/crop", json=CROP_REQUEST_BODY)
    assert response.status_code == 200
    body = response.json()
    wheat = next(c for c in body["crops"] if c["id"] == "wheat")
    # "फवारणी" (spraying) in the risk -> whole crop replaced with fallback text
    assert wheat["reason"] == main._crop_reason_text("good_rotation", "mr")
    assert wheat["risk"] == ""


def test_llm_reply_with_marathi_dose_falls_back_for_that_crop():
    crop_api_response = _mock_crop_api_response(FAKE_CROP_API_RESPONSE)
    llm_json = (
        '{"crops":[{"id":"wheat","reason":"चांगली उगवण होईल.","risk":"प्रति एकर २ मिली वापरा."},'
        '{"id":"chickpea","reason":"कमी पाण्यात येतो.","risk":"किडीचा धोका कमी."}]}'
    )
    with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=crop_api_response)), \
         patch.object(main, "_crop_llm_complete", AsyncMock(return_value=llm_json)):
        response = client.post("/api/crop", json=CROP_REQUEST_BODY)
    assert response.status_code == 200
    body = response.json()
    wheat = next(c for c in body["crops"] if c["id"] == "wheat")
    # "२ मिली" (a Devanagari-digit dose) in the risk -> fallback for wheat only
    assert wheat["reason"] == main._crop_reason_text("good_rotation", "mr")
    assert wheat["risk"] == ""
    chickpea = next(c for c in body["crops"] if c["id"] == "chickpea")
    assert chickpea["reason"] == "कमी पाण्यात येतो."


def test_wrong_language_detector():
    assert main._crop_text_is_wrong_language("Onion is suitable for sowing in October.", "mr") is True
    assert main._crop_text_is_wrong_language("कांदा ऑक्टोबरमध्ये पेरणीसाठी योग्य आहे.", "mr") is False
    assert main._crop_text_is_wrong_language("This is fine in English.", "en") is False


def test_wrong_language_detector_catches_latin_glued_onto_devanagari_word():
    # Seen in production: Nemotron wrote "विशेषज्ञanci" -- a Devanagari word
    # with an English suffix glued on, with no space/boundary between them.
    assert main._crop_text_is_wrong_language("स्थानिक विशेषज्ञanci सल्ला घ्या.", "mr") is True


def test_wrong_language_detector_allows_kvk_and_icar():
    assert main._crop_text_is_wrong_language("स्थानिक KVK शी संपर्क करा.", "mr") is False
    assert main._crop_text_is_wrong_language("ICAR च्या अहवालानुसार हे पीक योग्य आहे.", "hi") is False


def test_mr_word_fixes_replaces_whole_words_only():
    text = "सितंबर आणि अक्टूबर मध्ये रबी फसल ची बुवाई करा."
    fixed = main._apply_mr_word_fixes(text)
    assert fixed == "सप्टेंबर आणि ऑक्टोबर मध्ये रब्बी पीक ची पेरणी करा."


def test_mr_word_fixes_leaves_unlisted_words_alone():
    text = "कांदा ऑक्टोबरमध्ये पेरणीसाठी योग्य आहे."
    assert main._apply_mr_word_fixes(text) == text


def test_mr_word_fixes_strips_punctuation_before_matching():
    assert main._apply_mr_word_fixes("सीजन.") == "हंगाम."
    assert main._apply_mr_word_fixes("(खरीफ)") == "(खरीप)"


def test_mr_word_fixes_handles_marathi_suffix_glued_onto_hindi_word():
    assert main._apply_mr_word_fixes("अक्टूबरमध्ये") == "ऑक्टोबरमध्ये"
    assert main._apply_mr_word_fixes("रबीच्या") == "रब्बीच्या"
    assert main._apply_mr_word_fixes("फसलसाठी पाणी द्या.") == "पीकसाठी पाणी द्या."


def test_mr_word_fixes_does_not_match_an_unlisted_suffix():
    # "अक्टूबरनुसार" has a suffix ("नुसार") that is not in the allowed list,
    # so it must be left alone rather than guessed at.
    text = "अक्टूबरनुसार पेरणी करा."
    assert main._apply_mr_word_fixes(text) == text


def test_mr_word_fixes_handles_alternate_hindi_spellings():
    # सितम्बर/नवम्बर/अक्टोबर are alternate (explicit-consonant) spellings of
    # the same Hindi months as सितंबर/नवंबर/अक्टूबर, seen from the LLM too.
    assert main._apply_mr_word_fixes("सितम्बर") == "सप्टेंबर"
    assert main._apply_mr_word_fixes("अक्टोबर") == "ऑक्टोबर"


def test_mr_word_fixes_handles_multiword_phrase():
    # "केव्हीके शी" is a multi-word fixes.json key (the LLM wrongly puts a
    # space inside "KVK"+"शी") -- it can't go through the per-token loop,
    # so _apply_mr_word_fixes must also do a plain substring pass for it.
    text = "आपल्या जवळच्या केव्हीके शी संपर्क करा."
    assert main._apply_mr_word_fixes(text) == "आपल्या जवळच्या केव्हीकेशी संपर्क करा."


def test_crop_validator_applies_multiword_mr_word_fix_to_llm_text():
    recs = [{"id": "chickpea", "names": {"en": "Chickpea"}, "reasons": []}]
    llm_crops = [{"id": "chickpea", "reason": "शंका असल्यास केव्हीके शी संपर्क करा.", "risk": "कमी धोका."}]
    result = main._apply_crop_validator(llm_crops, recs, "mr")
    chickpea = result[0]
    assert chickpea["source"] == "llm"
    assert chickpea["reason"] == "शंका असल्यास केव्हीकेशी संपर्क करा."
    assert main._apply_mr_word_fixes("नवम्बर") == "नोव्हेंबर"


def test_crop_validator_applies_mr_word_fixes_to_llm_text():
    recs = [{"id": "chickpea", "names": {"en": "Chickpea"}, "reasons": []}]
    llm_crops = [{"id": "chickpea", "reason": "रबी सीजन मध्ये बुवाई करा.", "risk": "कमी धोका."}]
    result = main._apply_crop_validator(llm_crops, recs, "mr")
    chickpea = result[0]
    assert chickpea["source"] == "llm"
    assert chickpea["reason"] == "रब्बी हंगाम मध्ये पेरणी करा."


def test_crop_validator_does_not_apply_mr_word_fixes_for_hi():
    # The fixes are Hindi-word -> Marathi-word; they must never run when the
    # farmer actually asked for Hindi.
    recs = [{"id": "chickpea", "names": {"en": "Chickpea"}, "reasons": []}]
    llm_crops = [{"id": "chickpea", "reason": "रबी सीजन मध्ये बुवाई करा.", "risk": "कमी धोका."}]
    result = main._apply_crop_validator(llm_crops, recs, "hi")
    chickpea = result[0]
    assert chickpea["source"] == "llm"
    assert chickpea["reason"] == "रबी सीजन मध्ये बुवाई करा."


def test_crop_validator_replaces_risk_with_not_verified_when_equal_to_reason():
    # The LLM repeated the reason as the risk -- reason must stay, risk
    # must become the not_verified fallback text, source stays "llm".
    recs = [{"id": "chickpea", "names": {"en": "Chickpea"}, "reasons": []}]
    llm_crops = [{"id": "chickpea", "reason": "कमी पाण्यात येतो.", "risk": "कमी पाण्यात येतो."}]
    result = main._apply_crop_validator(llm_crops, recs, "mr")
    chickpea = result[0]
    assert chickpea["source"] == "llm"
    assert chickpea["reason"] == "कमी पाण्यात येतो."
    assert chickpea["risk"] == main._crop_reason_text("not_verified", "mr")


def test_crop_validator_keeps_risk_when_different_from_reason():
    recs = [{"id": "chickpea", "names": {"en": "Chickpea"}, "reasons": []}]
    llm_crops = [{"id": "chickpea", "reason": "कमी पाण्यात येतो.", "risk": "किडीचा धोका कमी."}]
    result = main._apply_crop_validator(llm_crops, recs, "mr")
    chickpea = result[0]
    assert chickpea["risk"] == "किडीचा धोका कमी."


def test_llm_reply_in_english_falls_back_for_that_crop_when_mr_requested():
    # Reproduces the production bug: Nemotron answered wheat in English
    # despite language=mr, while getting chickpea right.
    crop_api_response = _mock_crop_api_response(FAKE_CROP_API_RESPONSE)
    llm_json = json.dumps({"crops": [
        {"id": "wheat", "reason": "Wheat is suitable for sowing after maize.", "risk": "Minor cold risk."},
        {"id": "chickpea", "reason": "कमी पाण्यात येतो.", "risk": "किडीचा धोका कमी."},
    ]})
    with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=crop_api_response)), \
         patch.object(main, "_crop_llm_complete", AsyncMock(return_value=llm_json)):
        response = client.post("/api/crop", json=CROP_REQUEST_BODY)
    assert response.status_code == 200
    body = response.json()
    wheat = next(c for c in body["crops"] if c["id"] == "wheat")
    assert wheat["reason"] == main._crop_reason_text("good_rotation", "mr")
    assert wheat["risk"] == ""
    chickpea = next(c for c in body["crops"] if c["id"] == "chickpea")
    assert chickpea["reason"] == "कमी पाण्यात येतो."


def test_all_three_crops_present_and_in_language_are_all_source_llm():
    recommendations = FAKE_RECOMMENDATIONS + [{
        "id": "cabbage",
        "names": {"en": "Cabbage", "mr": "कोबी", "hi": "पत्ता गोभी", "reviewed": False},
        "sowing_months": None,
        "score": 90,
        "reasons": [{"code": "not_verified", "text": "This crop's data is not verified yet"}],
        "evidence": [],
    }]
    crop_api_response = _mock_crop_api_response({
        "success": True, "recommendations": recommendations, "weather": None,
    })
    llm_json = json.dumps({"crops": [
        {"id": "wheat", "reason": "चांगली जमीन आणि हवामान.", "risk": "थंडीचा थोडा धोका."},
        {"id": "chickpea", "reason": "कमी पाण्यात येणारे पीक.", "risk": "किडीचा धोका कमी."},
        {"id": "cabbage", "reason": "थंड हवामानात चांगले येते.", "risk": "जास्त उष्णतेचा धोका."},
    ]})
    with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=crop_api_response)), \
         patch.object(main, "_crop_llm_complete", AsyncMock(return_value=llm_json)):
        response = client.post("/api/crop", json=CROP_REQUEST_BODY)
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "llm"
    reasons = {c["id"]: c["reason"] for c in body["crops"]}
    assert reasons == {
        "wheat": "चांगली जमीन आणि हवामान.",
        "chickpea": "कमी पाण्यात येणारे पीक.",
        "cabbage": "थंड हवामानात चांगले येते.",
    }


def test_crop_llm_timeout_cancels_and_drains_all_pending_tasks():
    """Regression test for the production bug: when the total timeout fires
    while both the primary and hedge requests are still in flight, every
    task _crop_llm_complete created must be cancelled and awaited so none
    of them later logs "Task exception was never retrieved" (this is what
    happened to Ultra, which kept running to its own 40s client timeout
    after Super had already answered)."""
    captured_messages = []

    async def fake_create(*, model, **kwargs):
        # Neither model finishes before our (patched, short) total timeout;
        # if left running, each would eventually raise like a real backend
        # error/timeout would.
        await asyncio.sleep(0.3)
        raise RuntimeError(f"{model} backend error")

    fake_completions = type("FakeCompletions", (), {"create": staticmethod(fake_create)})()
    fake_chat = type("FakeChat", (), {"completions": fake_completions})()
    fake_client = type("FakeClient", (), {"chat": fake_chat})()

    async def run():
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda loop, context: captured_messages.append(context.get("message")))

        with patch.object(main, "_get_crop_llm_client", return_value=fake_client), \
             patch.object(main, "CROP_LLM_HEDGE_AFTER", 0.05), \
             patch.object(main, "CROP_LLM_TOTAL_TIMEOUT", 0.1):
            with pytest.raises(asyncio.TimeoutError):
                await main._crop_llm_complete([{"role": "user", "content": "hi"}])

        # Give any not-yet-cancelled task a chance to actually raise, then
        # force collection of anything no longer referenced.
        await asyncio.sleep(0.4)
        gc.collect()
        await asyncio.sleep(0)

    asyncio.run(run())
    assert not any(m and "never retrieved" in m for m in captured_messages)


def test_llm_timeout_falls_back_to_rules_source():
    crop_api_response = _mock_crop_api_response(FAKE_CROP_API_RESPONSE)
    with patch.object(httpx.AsyncClient, "post", AsyncMock(return_value=crop_api_response)), \
         patch.object(main, "_crop_llm_complete", AsyncMock(side_effect=asyncio.TimeoutError())):
        response = client.post("/api/crop", json=CROP_REQUEST_BODY)
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "rules"
    wheat = next(c for c in body["crops"] if c["id"] == "wheat")
    assert wheat["reason"] == main._crop_reason_text("good_rotation", "mr")
