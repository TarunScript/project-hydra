"""
download_idm_archive.py
=======================
Downloads the full weekly CDI archive from the India Drought Monitor GitHub repo.
265 weekly files from July 14, 2021 to present.

Format: lat lon cdi_value (space-separated, 0.25-degree grid, all-India)
Source: https://github.com/wcl-iitgn/IndianDroughtMonitor/tree/main/data/Drough_TS
"""

import os
import requests
import json
import time
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/wcl-iitgn/IndianDroughtMonitor/main/data"
API_URL = "https://api.github.com/repos/wcl-iitgn/IndianDroughtMonitor/contents/data/Drough_TS"
CACHE_DIR = Path(__file__).parent / "data" / "idm_archive"

def download_archive():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Get file listing from GitHub API
    print("Fetching IDM archive file listing...")
    r = requests.get(API_URL, timeout=30)
    r.raise_for_status()
    files = r.json()
    cdi_files = [f for f in files if f['name'].startswith('CDI_') and f['name'].endswith('.txt')]
    cdi_files.sort(key=lambda x: x['name'])

    print(f"Found {len(cdi_files)} weekly CDI files")
    print(f"  First: {cdi_files[0]['name']}")
    print(f"  Last:  {cdi_files[-1]['name']}")

    # Download each file
    downloaded = 0
    skipped = 0
    for i, f in enumerate(cdi_files):
        out_path = CACHE_DIR / f['name']
        if out_path.exists() and out_path.stat().st_size > 1000:
            skipped += 1
            continue

        url = f"{BASE_URL}/Drough_TS/{f['name']}"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            out_path.write_text(resp.text)
            downloaded += 1
            if downloaded % 20 == 0:
                print(f"  Downloaded {downloaded} files ({i+1}/{len(cdi_files)})...")
        except Exception as e:
            print(f"  FAILED: {f['name']} - {e}")

        # Be polite to GitHub
        if downloaded % 50 == 0:
            time.sleep(1)

    # Also download current and future CDI
    for extra in ['Current_CDI.txt', 'Future_CDI_7day.txt', 'Future_CDI_15day.txt', 'Future_CDI_30day.txt']:
        out_path = CACHE_DIR / extra
        url = f"{BASE_URL}/{extra}"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            out_path.write_text(resp.text)
            print(f"  Downloaded {extra}")
        except Exception as e:
            print(f"  FAILED: {extra} - {e}")

    total = downloaded + skipped
    print(f"\nDone: {downloaded} downloaded, {skipped} cached, {total} total")
    print(f"Archive dir: {CACHE_DIR}")

    # Quick validation
    all_files = sorted(CACHE_DIR.glob("CDI_*.txt"))
    print(f"\nValidation: {len(all_files)} CDI files in archive")
    if all_files:
        # Parse one to check format
        with open(all_files[0]) as fh:
            lines = [l.strip() for l in fh if l.strip() and not l.startswith('#')]
            parts = lines[0].split()
            print(f"  Format check: {len(parts)} columns (lat lon cdi)")
            print(f"  Grid points per file: ~{len(lines)}")
            # Count points in our 4 regions
            regions = {
                'marathwada': (17.5, 20.5, 75.0, 78.5),
                'bundelkhand': (23.1, 26.5, 78.1, 81.5),
                'rayalaseema': (12.5, 16.25, 76.9, 79.9),
                'saurashtra_kutch': (20.7, 24.7, 68.2, 72.0),
            }
            for rname, (lat_min, lat_max, lon_min, lon_max) in regions.items():
                count = 0
                for line in lines:
                    p = line.split()
                    if len(p) >= 3:
                        lat, lon = float(p[0]), float(p[1])
                        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                            count += 1
                print(f"  IDM grid points in {rname}: {count}")


if __name__ == "__main__":
    download_archive()
