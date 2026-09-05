"""Extract location mentions and geocode them via OSM Nominatim."""

import argparse
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from geopy.geocoders import Nominatim

load_dotenv()

FINAL_DIR = Path(__file__).resolve().parent.parent / "data" / "final"


def geocode_all(locations, user_agent):
    geolocator = Nominatim(user_agent=user_agent)
    results = {}
    for loc in locations:
        try:
            geocoded = geolocator.geocode(f"{loc}, Nepal")
        except Exception:
            geocoded = None
        results[loc] = (geocoded.latitude, geocoded.longitude) if geocoded else (None, None)
        time.sleep(1)  # Nominatim rate limit
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-path", default=FINAL_DIR / "test.csv")
    parser.add_argument("--out", default=FINAL_DIR / "test_geocoded.csv")
    args = parser.parse_args()

    user_agent = os.environ.get("NOMINATIM_USER_AGENT", "nepali-crisis-triage")

    raise NotImplementedError(
        "run NER over text column to extract location spans, then geocode_all() them"
    )


if __name__ == "__main__":
    main()
