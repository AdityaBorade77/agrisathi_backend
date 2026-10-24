# Plant disease API (Module 3).  Run:  python app.py
import json
import os

from flask import Flask, jsonify, request
from PIL import Image, ImageOps, UnidentifiedImageError

from disease_logic import predict_for_crop, split_class_name
from disease_model import DiseaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
PRECAUTIONS_FILE = os.path.join(BASE_DIR, "precautions.json")   # optional, filled in later
THRESHOLD = 0.7        # below this top probability the app says "not sure"
# Render (and other hosts) tell the app which port to use via $PORT;
# on your laptop it stays 5002.
PORT = int(os.getenv("PORT", "5002"))
MAX_UPLOAD_MB = 10

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

print("Loading model ...")
disease_model = DiseaseModel(MODEL_DIR)
precautions = {}
if os.path.exists(PRECAUTIONS_FILE):
    with open(PRECAUTIONS_FILE, encoding="utf-8") as f:
        precautions = json.load(f)
print("Model ready. Crops:", ", ".join(disease_model.crops))


def error(message, code=400):
    return jsonify({"error": message, "supported_crops": disease_model.crops}), code


@app.get("/health")
def health():
    return jsonify({"status": "ok", "classes": len(disease_model.class_names), "crops": disease_model.crops})


@app.get("/crops")
def crops():
    listing = {}
    for name in disease_model.class_names:
        crop, disease = split_class_name(name)
        listing.setdefault(crop, []).append(disease)
    return jsonify({"crops": listing})


@app.post("/predict-disease")
def predict_disease():
    crop = (request.form.get("crop") or "").strip()
    if not crop:
        return error("Please send the crop name in the 'crop' field.")
    file = request.files.get("image")
    if file is None:
        return error("Please send a leaf photo in the 'image' field.")

    try:
        image = Image.open(file.stream)
        image = ImageOps.exif_transpose(image).convert("RGB")   # fixes sideways phone photos
    except (UnidentifiedImageError, OSError):
        return error("That file is not a readable image (use JPG or PNG).")

    logits = disease_model.logits(image)
    try:
        result = predict_for_crop(logits, disease_model.class_names, crop, THRESHOLD, precautions=precautions)
    except ValueError:
        return error(f"'{crop}' is not a supported crop.")
    return jsonify(result)


@app.errorhandler(413)
def too_big(_):
    return error(f"Image is too large (limit {MAX_UPLOAD_MB} MB).", 413)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT, debug=False)
