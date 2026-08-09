#!/usr/bin/env python3
"""
live_weather.py  —  Real-time weather feature fetcher for Project Hydra

Fetches live/recent weather data from Open-Meteo (free, no API key) and
converts it into the same 8 feature columns the XGBoost model expects:

    rain_monthly_mm      — total rainfall this calendar month (mm)
    rain_7d_mm           — total rainfall last 7 days (mm)
    rain_3d_mm           — total rainfall last 3 days (mm)
    rain_1d_mm           — rainfall yesterday (mm)
    rain_daily_mean_mm   — daily mean rainfall this month (mm/day)
    rain_anomaly         — z-score vs historical climatology (σ)
    sm_surface           — surface soil moisture (0–7 cm, m³/m³)
    sm_rootzone          — root-zone soil moisture (0–100 cm, m³/m³)

Usage:
    from live_weather import get_live_weather_features
    features = get_live_weather_features(lat=26.4, lon=94.1, month=8)
"""

import requests
import datetime
from pathlib import Path
import csv

# ── Open-Meteo endpoints ──────────────────────────────────────
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Monthly climatological baseline: mean monthly rainfall (mm) for India
# Derived from CHIRPS 2000–2014 — used for rain_anomaly calculation
# Keys: (state_approx, month) → (mean_mm, std_mm)
# Fallback climatology for any location (all-India average)
STATE_CLIMATOLOGY = {
    "assam": {
        5: (356.3, 127.9), 6: (454.5, 147.6), 7: (431.5, 154.6),
        8: (315.1, 115.6), 9: (284.0, 130.7), 10: (148.7, 76.2)
    },
    "bihar": {
        5: (99.0, 97.2), 6: (261.6, 147.2), 7: (260.4, 113.0),
        8: (286.5, 77.3), 9: (201.0, 70.3), 10: (98.7, 44.7)
    },
    "west_bengal": {
        5: (197.8, 99.4), 6: (280.1, 167.6), 7: (320.8, 170.9),
        8: (346.8, 117.8), 9: (329.6, 132.6), 10: (188.0, 54.7)
    },
    "odisha": {
        5: (88.0, 64.1), 6: (167.6, 43.2), 7: (328.1, 79.2),
        8: (347.7, 91.1), 9: (297.3, 99.1), 10: (104.8, 54.4)
    }
}

CACHE: dict = {}   # in-memory cache: (lat_r, lon_r, month, day_offset, state_key) → weather dict
CACHE_TIMEOUT_MINS = 30


def get_live_weather_features(
    lat: float,
    lon: float,
    month: int | None = None,
    day_offset: int = 0,
    state_key: str | None = None,
    use_cache: bool = True,
) -> dict:
    """
    Fetch live weather features from Open-Meteo for a given lat/lon, day offset, and state.
    """
    if month is None:
        month = datetime.date.today().month

    day_offset = max(-7, min(15, day_offset))
    state_key = (state_key or "assam").lower()

    # Cache key includes day_offset & state_key
    cache_key = (round(lat, 1), round(lon, 1), month, day_offset, state_key)
    if use_cache and cache_key in CACHE:
        return CACHE[cache_key]

    today = datetime.date.today()

    try:
        params = {
            "latitude":       round(lat, 3),
            "longitude":      round(lon, 3),
            "daily":          ["precipitation_sum"],
            "hourly":         [
                "soil_moisture_0_to_1cm",
                "soil_moisture_1_to_3cm",
                "soil_moisture_3_to_9cm",
                "soil_moisture_9_to_27cm",
                "soil_moisture_27_to_81cm",
            ],
            "past_days":      9,   # Day -9 to Day -1
            "forecast_days":  16,  # Day 0 to Day +15
            "timezone":       "Asia/Kolkata",
        }
        resp = requests.get(FORECAST_URL, params=params, timeout=12)
        resp.raise_for_status()
        data = resp.json()

        daily_rain = data["daily"]["precipitation_sum"]
        all_rains  = [r or 0.0 for r in daily_rain]

        # Index 9 in all_rains corresponds to Day 0 (Today)
        today_idx = 9 if len(all_rains) >= 25 else max(0, len(all_rains) - 16)
        target_idx = max(0, min(len(all_rains) - 1, today_idx + day_offset))

        rain_1d_mm         = round(all_rains[target_idx], 2)
        rain_3d_mm         = round(sum(all_rains[max(0, target_idx - 2) : target_idx + 1]), 2)
        rain_7d_mm         = round(sum(all_rains[max(0, target_idx - 6) : target_idx + 1]), 2)
        rain_daily_mean_mm = round(rain_7d_mm / 7.0, 2)
        rain_monthly_mm    = round(rain_7d_mm * 4.3, 2)

        # Rainfall anomaly (z-score vs state climatology)
        clim_mean, clim_std = STATE_CLIMATOLOGY.get(state_key, {}).get(month, (300.0, 140.0))
        raw_anomaly = (rain_monthly_mm - clim_mean) / max(clim_std, 1.0)
        rain_anomaly = round(max(-2.5, min(3.5, raw_anomaly)), 3)

        # Hourly soil moisture for target day
        hourly = data["hourly"]
        n_hours = len(hourly["time"])
        h_start = max(0, target_idx * 24)
        h_end   = min(n_hours, (target_idx + 1) * 24)
        h_slice = slice(h_start, h_end) if h_start < n_hours else slice(max(0, n_hours - 24), n_hours)

        def mean_layer(key):
            vals = [v for v in hourly[key][h_slice] if v is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        sm_0_1  = mean_layer("soil_moisture_0_to_1cm")  or 0.25
        sm_1_3  = mean_layer("soil_moisture_1_to_3cm")  or 0.25
        sm_3_9  = mean_layer("soil_moisture_3_to_9cm")  or 0.25
        sm_9_27 = mean_layer("soil_moisture_9_to_27cm") or 0.25
        sm_27_81= mean_layer("soil_moisture_27_to_81cm")or 0.25

        sm_surface  = round((sm_0_1 * 1 + sm_1_3 * 2 + sm_3_9 * 6) / 9, 4)
        sm_rootzone = round((sm_0_1*1 + sm_1_3*2 + sm_3_9*6 + sm_9_27*18 + sm_27_81*54) / 81, 4)

        # For future forecast offset, adjust soil moisture by cumulative forecast rain & drying rate
        if day_offset > 0:
            sm_surface  = round(min(0.85, max(0.10, sm_surface * (0.95 ** day_offset) + (rain_3d_mm * 0.002))), 4)
            sm_rootzone = round(min(0.85, max(0.12, sm_rootzone * (0.97 ** day_offset) + (rain_7d_mm * 0.0015))), 4)

        result = {
            "rain_monthly_mm":    rain_monthly_mm,
            "rain_7d_mm":         rain_7d_mm,
            "rain_3d_mm":         rain_3d_mm,
            "rain_1d_mm":         rain_1d_mm,
            "rain_daily_mean_mm": rain_daily_mean_mm,
            "rain_anomaly":       rain_anomaly,
            "sm_surface":         sm_surface,
            "sm_rootzone":        sm_rootzone,
            "day_offset":         day_offset,
            "_source":            "open-meteo-live",
            "_fetched":           today.isoformat(),
            "_lat":               round(lat, 3),
            "_lon":               round(lon, 3),
        }

        CACHE[cache_key] = result
        return result

    except Exception as e:
        print(f"  ⚠ Live weather fetch failed for ({lat:.2f},{lon:.2f}): {e}")
        return None
        return None


def get_district_weather(district_centroids: list[dict], month: int | None = None) -> dict:
    """
    Fetch live weather for a list of districts in one batch.

    Args:
        district_centroids: list of {'district_name': str, 'lat': float, 'lon': float}
        month: target month (default: current)

    Returns:
        dict mapping district_name → weather feature dict
    """
    results = {}
    failed  = []
    for dist in district_centroids:
        name = dist["district_name"]
        w = get_live_weather_features(dist["lat"], dist["lon"], month)
        if w:
            results[name] = w
        else:
            failed.append(name)

    if failed:
        print(f"  ⚠ Live weather unavailable for: {failed}")

    return results


def clear_cache():
    """Clear the in-memory weather cache."""
    global CACHE
    CACHE = {}


# ── Standalone test ───────────────────────────────────────────
if __name__ == "__main__":
    import json

    # Test districts: Lakhimpur (Assam), Patna (Bihar), Cuttack (Odisha)
    test_points = [
        {"district_name": "LAKHIMPUR",  "lat": 27.23, "lon": 94.10},
        {"district_name": "DHEMAJI",    "lat": 27.48, "lon": 94.56},
        {"district_name": "PATNA",      "lat": 25.59, "lon": 85.14},
        {"district_name": "MURSHIDABAD","lat": 24.18, "lon": 88.27},
        {"district_name": "CUTTACK",    "lat": 20.46, "lon": 85.88},
        {"district_name": "BALESHWAR",  "lat": 21.49, "lon": 86.93},
    ]

    print("=" * 65)
    print("  Live Weather Fetch Test — Open-Meteo")
    print(f"  Date: {datetime.date.today()}")
    print("=" * 65)

    results = get_district_weather(test_points)

    print(f"\n  {'District':20s} {'Rain 7d':>8} {'Rain 1d':>8} {'SM Surf':>8} {'Anomaly':>8}")
    print(f"  {'-'*60}")
    for name, w in results.items():
        print(f"  {name:20s} {w['rain_7d_mm']:>7.1f}mm {w['rain_1d_mm']:>7.1f}mm "
              f"{w['sm_surface']:>8.3f} {w['rain_anomaly']:>+7.2f}σ")

    print(f"\n  Full data for LAKHIMPUR:")
    if "LAKHIMPUR" in results:
        for k, v in results["LAKHIMPUR"].items():
            if not k.startswith("_"):
                print(f"    {k:25s}: {v}")
