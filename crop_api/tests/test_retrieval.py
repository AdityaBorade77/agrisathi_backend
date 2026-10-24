# Tests for retrieval.py. Uses a small fake chunks list and fake crop_terms
# here, not the real chunks.json / crop_rules.json, so these tests do not
# depend on the real ICAR extraction. No network calls.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import retrieve  # noqa: E402

FAKE_CHUNKS = [
    {"id": 0, "page": 6, "section": "2.1.1", "text": "Pearl millet Shallow red soils protective irrigation at sowing."},
    {"id": 1, "page": 8, "section": "2.1.1", "text": "Onion sprinkler irrigation suggested for late kharif onion in deep black soils."},
    {"id": 2, "page": 5, "section": "1.12", "text": "Sowing window for Onion Rabi-Irrigated is 3rd week of September to 2nd week of October."},
    {"id": 3, "page": 9, "section": "2.1.2", "text": "Wheat irrigated at critical growth stages through sprinkler irrigation in medium soils."},
    {"id": 4, "page": 2, "section": None, "text": "District rainfall and land use pattern for Nashik district (no crop mentioned here)."},
]

FAKE_CROP_TERMS = {
    "onion_rabi": ["Onion"],
    "wheat": ["Wheat"],
}


def test_retrieve_only_returns_chunks_with_search_term():
    results = retrieve("wheat", k=3, chunks=FAKE_CHUNKS, crop_terms=FAKE_CROP_TERMS)
    assert len(results) == 1
    assert "Wheat" in results[0]["text"]


def test_retrieve_respects_k():
    results = retrieve("onion_rabi", k=1, chunks=FAKE_CHUNKS, crop_terms=FAKE_CROP_TERMS)
    assert len(results) == 1


def test_retrieve_returns_all_matches_up_to_k():
    results = retrieve("onion_rabi", k=3, chunks=FAKE_CHUNKS, crop_terms=FAKE_CROP_TERMS)
    # both onion chunks (id 1, id 2) contain "Onion"; the wheat-only and
    # rainfall chunks must not appear.
    ids = {(r["page"], r["section"]) for r in results}
    assert len(results) == 2
    assert (9, "2.1.2") not in ids  # wheat chunk
    assert (2, None) not in ids  # rainfall chunk, no crop term at all


def test_retrieve_returns_empty_for_unknown_crop():
    results = retrieve("durian", k=3, chunks=FAKE_CHUNKS, crop_terms=FAKE_CROP_TERMS)
    assert results == []


def test_retrieve_extra_terms_affect_ranking_not_inclusion():
    # Extra terms that match nothing must not add results beyond chunks
    # that already contain a real search term.
    results = retrieve("onion_rabi", extra_terms=["monsoon", "drought"], k=5,
                        chunks=FAKE_CHUNKS, crop_terms=FAKE_CROP_TERMS)
    assert len(results) == 2


def test_api_evidence_endpoint():
    from app import app as flask_app

    client = flask_app.test_client()
    reply = client.get("/evidence?crop=onion_rabi")
    assert reply.status_code == 200
    data = reply.get_json()
    assert data["crop"] == "onion_rabi"
    assert isinstance(data["evidence"], list)


def test_api_evidence_rejects_unknown_crop():
    from app import app as flask_app

    client = flask_app.test_client()
    reply = client.get("/evidence?crop=durian")
    assert reply.status_code == 400


def test_api_recommend_crop_includes_evidence():
    from unittest.mock import patch

    from app import app as flask_app

    client = flask_app.test_client()
    with patch("app.get_weather", return_value=None):
        reply = client.post("/recommend-crop", json={
            "taluka": "Niphad", "sowing_month": 11, "irrigation": "yes",
        })
    assert reply.status_code == 200
    recommendations = reply.get_json()["recommendations"]
    assert all("evidence" in rec for rec in recommendations)
