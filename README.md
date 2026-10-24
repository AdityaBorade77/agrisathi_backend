# agrisathi_backend

Deploy copy of the AgriSaathi backend (PID-14, K. K. Wagh IEER, Nashik) for
Render's free plan. **Do not edit here**: change the code in the main repo
(Multilingual-Integration, branch `integration`), commit, then run
`python deploy/export_backend.py` there and push this repo.

Services (see `render.yaml`):
- `backend/`     FastAPI: chat, speech-to-text, translation, text-to-speech
- `crop_api/`    Flask: crop recommendation (ICAR Nashik 2011 rules)
- `disease_api/` Flask + torch: leaf disease model (model downloaded at build)
- `frontend/`    the PWA (static site); also opened by the Android app (TWA)

Secrets (API keys) are set in the Render dashboard, never in this repo.
