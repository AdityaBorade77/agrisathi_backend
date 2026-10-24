# Tests for the rule engine. Uses a small fake rules list here, not the
# real crop_rules.json, so these tests do not depend on Shubham's real data.
# No real network calls: anything that would reach the weather API is mocked.
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rules import load_rules, recommend  # noqa: E402

FAKE_RULES = [
    {
        "id": "wheat",
        "names": {"en": "Wheat", "mr": "गहू", "hi": "गेहूं", "reviewed": False},
        "season": "rabi",
        "talukas": ["Niphad", "Sinnar"],
        "sowing_months": [10, 11],
        "irrigation": "required",
        "soil_types": ["medium", "heavy"],
        "ph_min": 6.0,
        "ph_max": 7.5,
        "avoid_after": ["wheat"],
        "good_after": None,
        "source": ["test"],
        "verified": True,
    },
    {
        "id": "bajra",
        "names": {"en": "Pearl millet", "mr": "बाजरी", "hi": "बाजरा", "reviewed": False},
        "season": "kharif",
        "talukas": "all",
        "sowing_months": [6, 7],
        "irrigation": "rainfed",
        "soil_types": ["light"],
        "ph_min": None,
        "ph_max": None,
        "avoid_after": None,
        "good_after": None,
        "source": ["test"],
        "verified": False,
    },
    {
        "id": "onion_rabi",
        "names": {"en": "Onion (Rabi)", "mr": "कांदा (रब्बी)", "hi": "प्याज (रबी)", "reviewed": False},
        "season": "rabi",
        "talukas": "all",
        "sowing_months": None,
        "irrigation": "optional",
        "soil_types": None,
        "ph_min": None,
        "ph_max": None,
        "avoid_after": None,
        "good_after": ["maize"],
        "source": ["test"],
        "verified": True,
    },
    {
        "id": "groundnut",
        "names": {"en": "Groundnut", "mr": "भुईमूग", "hi": "मूंगफली", "reviewed": False},
        "season": "kharif",
        "talukas": "all",
        "sowing_months": None,
        "irrigation": "rainfed",
        "soil_types": None,
        "ph_min": None,
        "ph_max": None,
        "avoid_after": None,
        "good_after": None,
        "source": ["test"],
        "verified": False,
    },
    {
        "id": "chickpea",
        "names": {"en": "Chickpea", "mr": "हरभरा", "hi": "चना", "reviewed": False},
        "season": "rabi",
        "talukas": "all",
        "sowing_months": None,
        "irrigation": "rainfed",
        "soil_types": None,
        "ph_min": None,
        "ph_max": None,
        "avoid_after": None,
        "good_after": None,
        "source": ["test"],
        "verified": False,
    },
]


def test_taluka_hard_rule_removes_crop():
    results = recommend("Malegaon", 11, "yes", None, rules=FAKE_RULES)
    ids = [r["id"] for r in results]
    assert "wheat" not in ids  # Malegaon is not in wheat's taluka list


def test_month_hard_rule_removes_crop():
    results = recommend("Niphad", 3, "yes", None, rules=FAKE_RULES)
    ids = [r["id"] for r in results]
    assert "wheat" not in ids  # month 3 is not in wheat's sowing_months


def test_irrigation_hard_rule_removes_crop():
    results = recommend("Niphad", 11, "no", None, rules=FAKE_RULES)
    ids = [r["id"] for r in results]
    assert "wheat" not in ids  # wheat needs irrigation, farmer has none


def test_avoid_after_hard_rule_removes_crop():
    results = recommend("Niphad", 11, "yes", "wheat", rules=FAKE_RULES)
    ids = [r["id"] for r in results]
    assert "wheat" not in ids  # can't grow wheat right after wheat


def test_soil_type_hard_rule_removes_crop():
    results = recommend("Niphad", 11, "yes", None, soil_type="light", rules=FAKE_RULES)
    ids = [r["id"] for r in results]
    assert "wheat" not in ids  # wheat needs medium/heavy soil, farmer has light


def test_soil_type_none_does_not_remove_crop():
    # No soil_type given at all -> soil_types check is skipped, not a failure.
    results = recommend("Niphad", 11, "yes", None, rules=FAKE_RULES)
    ids = [r["id"] for r in results]
    assert "wheat" in ids


def test_ph_soft_rule_lowers_score():
    in_range = recommend("Niphad", 11, "yes", None, soil={"ph": 7.0}, rules=FAKE_RULES)
    out_of_range = recommend("Niphad", 11, "yes", None, soil={"ph": 9.0}, rules=FAKE_RULES)

    wheat_in_range = next(r for r in in_range if r["id"] == "wheat")
    wheat_out_of_range = next(r for r in out_of_range if r["id"] == "wheat")

    assert wheat_out_of_range["score"] < wheat_in_range["score"]
    assert any(r["code"] == "ph_out_of_range" for r in wheat_out_of_range["reasons"])


def test_good_after_bonus_raises_score():
    # month 10 -- onion_rabi has no sowing_months of its own, but season
    # "rabi" puts it inside the season-window hard rule (9, 10, 11).
    after_maize = recommend("Dindori", 10, "yes", "maize", rules=FAKE_RULES)
    after_nothing = recommend("Dindori", 10, "yes", None, rules=FAKE_RULES)

    onion_after_maize = next(r for r in after_maize if r["id"] == "onion_rabi")
    onion_after_nothing = next(r for r in after_nothing if r["id"] == "onion_rabi")

    assert onion_after_maize["score"] > onion_after_nothing["score"]
    assert any(r["code"] == "good_rotation" for r in onion_after_maize["reasons"])


def test_sowing_unknown_reason_when_months_null():
    # onion_rabi has sowing_months=None and season "rabi" (season window
    # 9, 10, 11), so month 10 keeps it and should still flag the reason.
    results = recommend("Dindori", 10, "yes", None, rules=FAKE_RULES)
    onion = next(r for r in results if r["id"] == "onion_rabi")
    assert any(r["code"] == "sowing_unknown" for r in onion["reasons"])


def test_dry_spell_lowers_score_for_kharif_crop_in_sowing_month():
    dry = recommend("Dindori", 6, "yes", None, weather={"rain_last_14_days_mm": 1.0}, rules=FAKE_RULES)
    wet = recommend("Dindori", 6, "yes", None, weather={"rain_last_14_days_mm": 10.0}, rules=FAKE_RULES)

    bajra_dry = next(r for r in dry if r["id"] == "bajra")
    bajra_wet = next(r for r in wet if r["id"] == "bajra")

    assert bajra_dry["score"] < bajra_wet["score"]
    assert any(r["code"] == "dry_spell" for r in bajra_dry["reasons"])
    assert not any(r["code"] == "dry_spell" for r in bajra_wet["reasons"])


def test_no_dry_spell_for_rabi_crop():
    # wheat is rabi -- a dry spell reading should never apply to it.
    results = recommend("Niphad", 11, "yes", None, weather={"rain_last_14_days_mm": 1.0}, rules=FAKE_RULES)
    wheat = next(r for r in results if r["id"] == "wheat")
    assert not any(r["code"] == "dry_spell" for r in wheat["reasons"])


def test_no_dry_spell_when_weather_is_none():
    results = recommend("Dindori", 6, "yes", None, weather=None, rules=FAKE_RULES)
    bajra = next(r for r in results if r["id"] == "bajra")
    assert not any(r["code"] == "dry_spell" for r in bajra["reasons"])


def test_null_fields_are_skipped_not_treated_as_failure():
    # bajra has every soft/hard field null or "all" except irrigation/months;
    # giving no soil data at all must not remove or penalise it for pH.
    results = recommend("Dindori", 6, "yes", None, rules=FAKE_RULES)
    bajra = next(r for r in results if r["id"] == "bajra")
    assert not any(r["code"].startswith("ph_") for r in bajra["reasons"])


def test_max_three_results():
    # month 10 -- FAKE_RULES[2] (onion_rabi) needs it to pass the season
    # window hard rule (season "rabi" -> 9, 10, 11), so the cap is actually
    # exercised instead of every copy being dropped.
    many_rules = [dict(FAKE_RULES[2], id=f"crop{i}") for i in range(5)]
    results = recommend("Dindori", 10, "yes", None, rules=many_rules)
    assert len(results) <= 3


def test_empty_list_when_nothing_qualifies():
    results = recommend("Dindori", 1, "yes", None, rules=[FAKE_RULES[0]])
    assert results == []


def test_result_includes_names_and_sowing_months():
    results = recommend("Niphad", 11, "yes", None, rules=FAKE_RULES)
    wheat = next(r for r in results if r["id"] == "wheat")
    assert wheat["names"] == {"en": "Wheat", "mr": "गहू", "hi": "गेहूं", "reviewed": False}
    assert wheat["sowing_months"] == [10, 11]


def test_result_sowing_months_is_none_when_unverified():
    # month 10 -- must stay inside onion_rabi's season window (see above).
    results = recommend("Dindori", 10, "yes", None, rules=FAKE_RULES)
    onion = next(r for r in results if r["id"] == "onion_rabi")
    assert onion["sowing_months"] is None


def test_season_window_hard_rule_drops_kharif_crop_outside_window():
    # groundnut: sowing_months=None, season="kharif" -> window 6, 7, 8.
    # October (10) is outside it, so it must not be suggested.
    results = recommend("Dindori", 10, "yes", None, rules=FAKE_RULES)
    ids = [r["id"] for r in results]
    assert "groundnut" not in ids


def test_season_window_hard_rule_allows_rabi_crop_inside_window():
    # chickpea: sowing_months=None, season="rabi" -> window 9, 10, 11.
    # October (10) is inside it, so it must be allowed.
    results = recommend("Dindori", 10, "yes", None, rules=FAKE_RULES)
    ids = [r["id"] for r in results]
    assert "chickpea" in ids


def test_season_window_hard_rule_allows_kharif_crop_inside_window():
    # Same groundnut crop, but June (6) is inside its kharif window.
    results = recommend("Dindori", 6, "yes", None, rules=FAKE_RULES)
    ids = [r["id"] for r in results]
    assert "groundnut" in ids


def test_season_window_reason_added_when_season_known():
    results = recommend("Dindori", 10, "yes", None, rules=FAKE_RULES)
    chickpea = next(r for r in results if r["id"] == "chickpea")
    assert any(r["code"] == "season_window" for r in chickpea["reasons"])


def test_season_window_reason_not_added_when_season_unknown():
    # bajra has sowing_months=[6, 7] (not null), so the season-window
    # estimate never applies to it and the reason must not appear.
    results = recommend("Dindori", 6, "yes", None, rules=FAKE_RULES)
    bajra = next(r for r in results if r["id"] == "bajra")
    assert not any(r["code"] == "season_window" for r in bajra["reasons"])


def test_onion_split_in_real_data():
    # Phase 2 replaced the single "onion" crop with two seasonal entries.
    real_rules = load_rules()
    ids = [c["id"] for c in real_rules]
    assert "onion" not in ids
    assert "onion_kharif" in ids
    assert "onion_rabi" in ids


def test_api_rejects_bad_input():
    from app import app as flask_app

    client = flask_app.test_client()

    bad_taluka = client.post("/recommend-crop", json={
        "taluka": "Nowhere", "sowing_month": 11, "irrigation": "yes",
    })
    assert bad_taluka.status_code == 400
    assert bad_taluka.get_json()["success"] is False

    bad_month = client.post("/recommend-crop", json={
        "taluka": "Niphad", "sowing_month": 13, "irrigation": "yes",
    })
    assert bad_month.status_code == 400

    bad_irrigation = client.post("/recommend-crop", json={
        "taluka": "Niphad", "sowing_month": 11, "irrigation": "maybe",
    })
    assert bad_irrigation.status_code == 400

    bad_ph = client.post("/recommend-crop", json={
        "taluka": "Niphad", "sowing_month": 11, "irrigation": "yes",
        "soil": {"ph": 20},
    })
    assert bad_ph.status_code == 400

    bad_soil_type = client.post("/recommend-crop", json={
        "taluka": "Niphad", "sowing_month": 11, "irrigation": "yes",
        "soil_type": "rocky",
    })
    assert bad_soil_type.status_code == 400


def test_api_accepts_valid_soil_type():
    from app import app as flask_app

    client = flask_app.test_client()
    with patch("app.get_weather", return_value=None):
        reply = client.post("/recommend-crop", json={
            "taluka": "Niphad", "sowing_month": 11, "irrigation": "yes",
            "soil_type": "medium",
        })
    assert reply.status_code == 200
    assert reply.get_json()["success"] is True


def test_api_soil_types_endpoint():
    from app import app as flask_app

    client = flask_app.test_client()
    reply = client.get("/soil-types")
    assert reply.status_code == 200
    ids = [s["id"] for s in reply.get_json()["soil_types"]]
    assert ids == ["light", "medium", "heavy"]


def test_api_crops_endpoint():
    from app import app as flask_app

    client = flask_app.test_client()
    reply = client.get("/crops")
    assert reply.status_code == 200
    crops = reply.get_json()["crops"]
    ids = [c["id"] for c in crops]
    assert "wheat" in ids
    assert "maize" in ids
    wheat = next(c for c in crops if c["id"] == "wheat")
    assert wheat["names"]["en"] == "Wheat"


def test_api_returns_200_with_weather_null_when_weather_fails():
    from app import app as flask_app

    client = flask_app.test_client()
    with patch("app.get_weather", return_value=None):
        reply = client.post("/recommend-crop", json={
            "taluka": "Niphad", "sowing_month": 11, "irrigation": "yes",
        })
    assert reply.status_code == 200
    data = reply.get_json()
    assert data["success"] is True
    assert data["weather"] is None
    assert data["weather_note"] == "Weather unavailable"


def test_weather_route_reports_unavailable_when_weather_fails():
    from app import app as flask_app

    client = flask_app.test_client()
    with patch("app.get_weather", return_value=None):
        reply = client.get("/weather?taluka=Niphad")
    assert reply.status_code == 200
    assert reply.get_json()["weather"] is None


def test_weather_route_rejects_unknown_taluka():
    from app import app as flask_app

    client = flask_app.test_client()
    reply = client.get("/weather?taluka=Nowhere")
    assert reply.status_code == 400
