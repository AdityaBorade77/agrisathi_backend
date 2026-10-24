# One-off script: look up lat/lon for each Nashik taluka headquarters town
# using the free Open-Meteo geocoding API, and save them into talukas.json.
# Run:  python scripts/fetch_taluka_coords.py
#
# Shows a result table first and asks before writing, so a bad match
# (wrong state, wrong country) never gets silently saved as real data.
import json
import os
import sys
import time

import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TALUKAS_FILE = os.path.join(BASE_DIR, "data", "talukas.json")
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

# Some taluka names are not the best search term for the headquarters town.
# Tried in order, after the taluka's own name, until one gives a Nashik-district match.
SEARCH_NAME_FALLBACKS = {
    "Baglan": ["Satana"],
    "Trimbakeshwar": ["Trimbak"],
    "Peth": ["Peint"],
    "Kalwan": ["Kalvan"],
    "Chandwad": ["Chandvad", "Chandor"],
    "Deola": ["Devla", "Deoli"],
}

REQUIRED_DISTRICT = "Nashik"  # admin2, as returned by Open-Meteo/GeoNames

# admin3 (sub-taluka) usually matches the taluka's own English name exactly.
# Known alternate spellings go here so a real match is not rejected by accident.
ADMIN3_ALIASES = {
    "Yeola": ["Yevla"],
    "Kalwan": ["Kalvan"],
    "Chandwad": ["Chandvad", "Chandor"],
    "Deola": ["Devla"],
}


def search(name, retries=3):
    for attempt in range(retries):
        try:
            reply = requests.get(GEOCODE_URL, params={"name": name, "count": 10}, timeout=10)
            reply.raise_for_status()
            return reply.json().get("results") or []
        except (requests.RequestException, ValueError):
            if attempt < retries - 1:
                time.sleep(1)
    return []


def is_real_match(taluka_en, place):
    if place.get("country_code") != "IN" or place.get("admin1") != "Maharashtra" \
            or place.get("admin2") != REQUIRED_DISTRICT:
        return False
    # admin3 must name the taluka itself, not just any place inside Nashik
    # district -- otherwise a same-named hamlet in a different taluka (seen
    # for "Deola", filed under admin3 "Trimbakeshwar") gets accepted as if
    # it were the real taluka headquarters.
    admin3 = place.get("admin3")
    return admin3 == taluka_en or admin3 in ADMIN3_ALIASES.get(taluka_en, [])


def lookup(taluka_en):
    """Try the taluka's own name, then its fallbacks, until one search term
    returns a result that is really inside this taluka (see is_real_match).
    Returns (place_or_None, search_name_used)."""
    for name in [taluka_en] + SEARCH_NAME_FALLBACKS.get(taluka_en, []):
        for place in search(name):
            if is_real_match(taluka_en, place):
                return place, name
        time.sleep(0.3)  # be gentle on the free API between fallback attempts
    return None, taluka_en


def main():
    with open(TALUKAS_FILE, encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for taluka in data["talukas"]:
        place, searched_as = lookup(taluka["en"])
        if place is None:
            print(f"NOT FOUND inside Nashik district: {taluka['en']}")
            rows.append((taluka["en"], searched_as, None, None, None))
        else:
            rows.append((taluka["en"], searched_as, place["latitude"], place["longitude"], place.get("admin3")))
        taluka["_fetched_place"] = place  # temporary, used below, not saved as-is

    print(f"\n{'Taluka':<16}{'Searched as':<14}{'Latitude':<12}{'Longitude':<12}{'admin3 (sub-taluka)'}")
    for en, searched, lat, lon, admin3 in rows:
        lat_s = f"{lat:.4f}" if lat is not None else "-"
        lon_s = f"{lon:.4f}" if lon is not None else "-"
        print(f"{en:<16}{searched:<14}{lat_s:<12}{lon_s:<12}{admin3 or '-'}")

    if "--save" not in sys.argv:
        print("\nDry run only (table shown above). Re-run with --save to write to talukas.json.")
        return

    for taluka in data["talukas"]:
        place = taluka.pop("_fetched_place")
        if place is None:
            taluka["latitude"] = None
            taluka["longitude"] = None
            taluka["coords_source"] = None
        else:
            taluka["latitude"] = place["latitude"]
            taluka["longitude"] = place["longitude"]
            taluka["coords_source"] = "Open-Meteo geocoding"

    with open(TALUKAS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"\nSaved to {TALUKAS_FILE}")


if __name__ == "__main__":
    main()
