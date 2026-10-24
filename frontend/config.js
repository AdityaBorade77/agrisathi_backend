/* ============================================================
   config.js — where the frontend finds the backend
   ============================================================
   This is the ONLY file to edit when the backend moves.

   - On your laptop: "http://localhost:8000"
   - After hosting:  your server address, e.g. "https://agrisaathi-api.example.com"
     (no "/" at the end, and it must be https:// for the microphone
     and "Install app" to work on phones)

   Never put API keys or tokens here: everything in the frontend
   can be read by anyone who opens the app.
   ============================================================ */

// Opened from your laptop (localhost) -> your local backend.
// Opened anywhere else (the hosted site, the Android app) -> the backend on Render.
(function () {
  const LOCAL_BACKEND = "http://localhost:8000";
  const HOSTED_BACKEND = "https://agrisathi-backend-n0qq.onrender.com";

  const host = window.location.hostname;
  const isLocal = host === "localhost" || host === "127.0.0.1";

  window.AGRISAATHI_CONFIG = {
    apiUrl: isLocal ? LOCAL_BACKEND : HOSTED_BACKEND,
  };
})();
