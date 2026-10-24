# Weather lookup from the free Open-Meteo forecast API (no API key needed).
# Pure data fetch -- never raises, so a weather outage never breaks a
# crop recommendation. Results are cached per (lat, lon) for 1 hour so a
# burst of requests for the same taluka does not hammer the free API.
import time
from datetime import datetime, timezone

import requests

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_SECONDS = 3
CACHE_SECONDS = 60 * 60

_cache = {}  # (lat, lon) -> (fetched_at_monotonic, result_dict)


def _fetch(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m",
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min",
        "past_days": 14,
        "forecast_days": 7,
        "timezone": "Asia/Kolkata",
    }
    reply = requests.get(FORECAST_URL, params=params, timeout=TIMEOUT_SECONDS)
    reply.raise_for_status()
    data = reply.json()

    current = data["current"]
    daily = data["daily"]
    dates = daily["time"]
    precip = daily["precipitation_sum"]
    tmax = daily["temperature_2m_max"]
    tmin = daily["temperature_2m_min"]

    today = datetime.now(timezone.utc).astimezone().date().isoformat()
    # past_days/forecast_days both count from "today", so split the daily
    # arrays at the first date that is not in the past.
    split = next((i for i, d in enumerate(dates) if d >= today), len(dates))

    return {
        "current_temp_c": current["temperature_2m"],
        "current_humidity": current["relative_humidity_2m"],
        "rain_last_14_days_mm": round(sum(precip[:split]), 1),
        "rain_next_7_days_mm": round(sum(precip[split:]), 1),
        "max_temp_next_7_days_c": max(tmax[split:]) if tmax[split:] else None,
        "min_temp_next_7_days_c": min(tmin[split:]) if tmin[split:] else None,
        "source": "Open-Meteo",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def get_weather(lat, lon):
    if lat is None or lon is None:
        return None

    key = (round(lat, 4), round(lon, 4))
    cached = _cache.get(key)
    if cached is not None and time.monotonic() - cached[0] < CACHE_SECONDS:
        return cached[1]

    try:
        result = _fetch(lat, lon)
    except (requests.RequestException, KeyError, ValueError, IndexError):
        return None

    _cache[key] = (time.monotonic(), result)
    return result
