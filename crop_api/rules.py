# Pure-Python rule engine for crop recommendation (Module 2, Phase 2).
# No Flask, no I/O here, so it is easy to unit test.
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RULES_FILE = os.path.join(BASE_DIR, "data", "crop_rules.json")

STARTING_SCORE = 100
PH_PENALTY = 20
UNVERIFIED_PENALTY = 10
GOOD_ROTATION_BONUS = 15
DRY_SPELL_PENALTY = 10
MAX_RESULTS = 3

# ICAR 2011 sec 2.1.1: less than 2.5 mm rain over 2 weeks counts as a dry spell.
DRY_SPELL_RAIN_MM = 2.5

# Estimate used only when a crop's own sowing_months is null but its season
# is known: ICAR 2011 sec 1.12's general kharif/rabi sowing windows, not a
# value stated per crop -- see crop_api/data/SOURCES.md.
SEASON_SOWING_WINDOWS = {"kharif": [6, 7, 8], "rabi": [9, 10, 11]}


def load_rules(path=DEFAULT_RULES_FILE):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["crops"]


def recommend(taluka, sowing_month, irrigation, previous_crop, soil=None, soil_type=None,
              weather=None, rules=None):
    if rules is None:
        rules = load_rules()
    soil = soil or {}

    results = []
    for crop in rules:
        # Hard rules: fail one and the crop is dropped, not scored.
        talukas = crop.get("talukas")
        if talukas is not None and talukas != "all" and taluka not in talukas:
            continue

        sowing_months = crop.get("sowing_months")
        season_window = None
        if sowing_months is None:
            season_window = SEASON_SOWING_WINDOWS.get(crop.get("season"))
        effective_sowing_months = sowing_months if sowing_months is not None else season_window
        if effective_sowing_months is not None and sowing_month not in effective_sowing_months:
            continue

        if crop.get("irrigation") == "required" and irrigation == "no":
            continue

        avoid_after = crop.get("avoid_after")
        if avoid_after is not None and previous_crop in avoid_after:
            continue

        soil_types = crop.get("soil_types")
        if soil_types is not None and soil_type is not None and soil_type not in soil_types:
            continue

        # Soft rules: crop stays, score changes, reason explains why.
        score = STARTING_SCORE
        reasons = []

        ph = soil.get("ph")
        ph_min, ph_max = crop.get("ph_min"), crop.get("ph_max")
        if ph is not None and ph_min is not None and ph_max is not None:
            if ph_min <= ph <= ph_max:
                reasons.append({"code": "ph_ok", "text": f"Soil pH {ph} is within {ph_min}-{ph_max}"})
            else:
                score -= PH_PENALTY
                reasons.append({"code": "ph_out_of_range", "text": f"Soil pH {ph} is outside {ph_min}-{ph_max}"})

        if sowing_months is None:
            reasons.append({"code": "sowing_unknown", "text": "Sowing window not verified"})
            if season_window is not None:
                reasons.append({
                    "code": "season_window",
                    "text": "Sowing month estimated from season",
                })

        good_after = crop.get("good_after")
        if good_after is not None and previous_crop in good_after:
            score += GOOD_ROTATION_BONUS
            reasons.append({"code": "good_rotation", "text": f"Good to grow after {previous_crop}"})

        if weather is not None and crop.get("season") == "kharif" \
                and sowing_months is not None and sowing_month in sowing_months \
                and weather.get("rain_last_14_days_mm") is not None \
                and weather["rain_last_14_days_mm"] < DRY_SPELL_RAIN_MM:
            score -= DRY_SPELL_PENALTY
            reasons.append({
                "code": "dry_spell",
                "text": "Less than 2.5 mm rain in the last 14 days. ICAR defines this as a "
                        "dry spell; sowing may need protective irrigation.",
            })

        if not crop.get("verified", False):
            score -= UNVERIFIED_PENALTY
            reasons.append({"code": "not_verified", "text": "This crop's data is not verified yet"})

        results.append({
            "id": crop["id"],
            "names": crop.get("names"),
            "sowing_months": sowing_months,
            "score": score,
            "reasons": reasons,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:MAX_RESULTS]
