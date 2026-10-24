# Crop recommendation API (Module 2, Phase 1)

A rule-based crop recommender for Nashik district farmers. No AI model, no
LLM, no weather data yet — just hard/soft rules over a data file. English
only for now.

**Important:** `data/crop_rules.json` has only crop ids and names filled
in. Every agronomic field (talukas, sowing months, irrigation, pH range,
rotation) is `null` with `"source": "TODO", "verified": false`, because no
real agronomic values have been entered yet. Until Shubham fills these in
from the ICAR Nashik District Agricultural Contingency Plan, Soil Health
Card data, and MPKV Agromet bulletins, the API will return few or no
recommendations — that is expected, not a bug.

## Setup (Windows PowerShell)

```powershell
cd crop_api
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run the server

```powershell
cd crop_api
venv\Scripts\activate
python app.py
```

Server starts on `http://127.0.0.1:5003`.

## Run the tests

```powershell
cd crop_api
venv\Scripts\activate
pytest
```

## Endpoints

- `GET /health` — `{"status": "ok", "crops": <count>, "verified_crops": <count>}`
- `GET /talukas` — the 15 Nashik talukas with en/mr/hi names
- `POST /recommend-crop` — JSON body:

```json
{
  "taluka": "Niphad",
  "sowing_month": 11,
  "irrigation": "yes",
  "previous_crop": "soybean",
  "soil": {"ph": 7.6, "ec": 0.3, "oc": 0.5, "n": 180, "p": 15, "k": 300}
}
```

`soil` is optional; every field inside it is optional. Bad input (unknown
taluka, month not 1-12, irrigation not `yes`/`no`, pH not 0-14) returns
HTTP 400 with `{"success": false, "errors": [...]}`.

On success: `{"success": true, "recommendations": [...], "source": "rules"}`,
with up to 3 crops sorted by score, each with a list of short reason codes
(e.g. `ph_out_of_range`, `not_verified`).
