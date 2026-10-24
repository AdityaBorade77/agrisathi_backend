# Crop recommendation API (Module 2, Phase 4a - rule engine + weather + BM25 evidence).
# Run:  python app.py
import json
import os

from flask import Flask, jsonify, request

from retrieval import retrieve
from rules import load_rules, recommend
from weather import get_weather

EVIDENCE_CHAR_BUDGET = 2000
# How the ICAR plan describes each soil_type (sec 1.12 / 2.1.x), used as
# extra BM25 query terms -- never shown to the farmer, only used to rank.
SOIL_TYPE_WORDS = {"light": "shallow red", "medium": "medium red black", "heavy": "deep black"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TALUKAS_FILE = os.path.join(DATA_DIR, "talukas.json")
SOIL_TYPES_FILE = os.path.join(DATA_DIR, "soil_types.json")
# Render (and other hosts) tell the app which port to use via $PORT;
# on your laptop it stays 5003.
PORT = int(os.getenv("PORT", "5003"))

app = Flask(__name__)

print("Loading crop rules ...")
crop_rules = load_rules()
with open(TALUKAS_FILE, encoding="utf-8") as f:
    talukas = json.load(f)["talukas"]
with open(SOIL_TYPES_FILE, encoding="utf-8") as f:
    soil_types = json.load(f)["soil_types"]
taluka_by_name = {t["en"].lower(): t for t in talukas}
soil_type_ids = {s["id"] for s in soil_types}
crop_by_id = {c["id"]: c for c in crop_rules}
# used by /health as a cheap, known-good point to test the weather API with
HEALTH_CHECK_TALUKA = next((t for t in talukas if t.get("latitude") is not None), None)
print(f"Crop rules ready. {len(crop_rules)} crops, {sum(c['verified'] for c in crop_rules)} verified.")


def weather_for_taluka(taluka):
    """taluka is the matched entry from talukas.json, or None."""
    if taluka is None or taluka.get("latitude") is None or taluka.get("longitude") is None:
        return None
    return get_weather(taluka["latitude"], taluka["longitude"])


def error(messages, code=400):
    if isinstance(messages, str):
        messages = [messages]
    return jsonify({"success": False, "errors": messages}), code


@app.get("/health")
def health():
    weather_ok = weather_for_taluka(HEALTH_CHECK_TALUKA) is not None
    return jsonify({
        "status": "ok",
        "crops": len(crop_rules),
        "verified_crops": sum(c["verified"] for c in crop_rules),
        "weather_ok": weather_ok,
    })


@app.get("/talukas")
def get_talukas():
    return jsonify({"talukas": talukas})


@app.get("/soil-types")
def get_soil_types():
    return jsonify({"soil_types": soil_types})


@app.get("/crops")
def get_crops():
    return jsonify({"crops": [{"id": c["id"], "names": c.get("names")} for c in crop_rules]})


@app.get("/evidence")
def evidence_route():
    crop_id = request.args.get("crop") or ""
    if crop_id not in crop_by_id:
        return error(f"Unknown crop '{crop_id}'.")
    chunks = retrieve([crop_id], k=3)
    return jsonify({
        "crop": crop_id,
        "evidence": [{"page": c["page"], "section": c["section"], "text": c["text"]} for c in chunks],
    })


@app.get("/weather")
def weather_route():
    taluka_name = request.args.get("taluka") or ""
    taluka = taluka_by_name.get(taluka_name.lower())
    if taluka is None:
        return error(f"Unknown taluka '{taluka_name}'.")

    weather = weather_for_taluka(taluka)
    if weather is None:
        return jsonify({"weather": None, "weather_note": "Weather unavailable"})
    return jsonify({"weather": weather})


@app.post("/recommend-crop")
def recommend_crop():
    body = request.get_json(silent=True)
    if body is None:
        return error("Send a JSON body.")

    errors = []

    taluka = body.get("taluka")
    if not taluka or taluka.lower() not in taluka_by_name:
        errors.append(f"Unknown taluka '{taluka}'.")

    sowing_month = body.get("sowing_month")
    if not isinstance(sowing_month, int) or not (1 <= sowing_month <= 12):
        errors.append("sowing_month must be a number from 1 to 12.")

    irrigation = body.get("irrigation")
    if irrigation not in ("yes", "no"):
        errors.append("irrigation must be 'yes' or 'no'.")

    soil = body.get("soil") or {}
    ph = soil.get("ph")
    if ph is not None and not (0 <= ph <= 14):
        errors.append("soil.ph must be between 0 and 14.")

    soil_type = body.get("soil_type")
    if soil_type is not None and soil_type not in soil_type_ids:
        errors.append(f"Unknown soil_type '{soil_type}'.")

    if errors:
        return error(errors)

    previous_crop = body.get("previous_crop")
    weather = weather_for_taluka(taluka_by_name.get(taluka.lower()))
    recommendations = recommend(
        taluka=taluka,
        sowing_month=sowing_month,
        irrigation=irrigation,
        previous_crop=previous_crop,
        soil=soil,
        soil_type=soil_type,
        weather=weather,
        rules=crop_rules,
    )

    evidence_budget = EVIDENCE_CHAR_BUDGET
    for rec in recommendations:
        crop = crop_by_id.get(rec["id"])
        extra_terms = []
        if crop and crop.get("season"):
            extra_terms.append(crop["season"])
        if soil_type:
            extra_terms.append(SOIL_TYPE_WORDS.get(soil_type, soil_type))

        rec["evidence"] = []
        for chunk in retrieve([rec["id"]], extra_terms=extra_terms, k=2):
            if evidence_budget <= 0:
                break
            text = chunk["text"][:evidence_budget]
            rec["evidence"].append({"page": chunk["page"], "section": chunk["section"], "text": text})
            evidence_budget -= len(text)

    reply = {"success": True, "recommendations": recommendations, "source": "rules", "weather": weather}
    if weather is None:
        reply["weather_note"] = "Weather unavailable"
    return jsonify(reply)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT, debug=False)
