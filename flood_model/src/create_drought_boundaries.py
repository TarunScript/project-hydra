#!/usr/bin/env python3
"""
create_drought_boundaries.py — Generates boundary GeoJSON files for drought regions:
- Marathwada (Maharashtra)
- Bundelkhand (UP / MP)
- Rayalaseema (Andhra Pradesh)

Creates:
- flood_model/data/raw/marathwada_districts.geojson
- flood_model/data/raw/bundelkhand_districts.geojson
- flood_model/data/raw/rayalaseema_districts.geojson
"""

import json
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# District coordinates (approximate polygon bounding boxes for clean visualization)
DROUGHT_BOUNDARIES = {
    "marathwada": [
        {"name": "Aurangabad", "center": [75.34, 19.87], "bbox": [75.0, 19.5, 75.8, 20.3]},
        {"name": "Jalna", "center": [75.88, 19.84], "bbox": [75.6, 19.5, 76.3, 20.2]},
        {"name": "Beed", "center": [75.76, 18.99], "bbox": [75.3, 18.6, 76.4, 19.3]},
        {"name": "Latur", "center": [76.57, 18.40], "bbox": [76.2, 18.0, 77.1, 18.7]},
        {"name": "Osmanabad", "center": [76.05, 18.18], "bbox": [75.6, 17.8, 76.5, 18.6]},
        {"name": "Nanded", "center": [77.31, 19.15], "bbox": [76.9, 18.8, 77.8, 19.6]},
        {"name": "Parbhani", "center": [76.77, 19.26], "bbox": [76.4, 18.9, 77.2, 19.6]},
        {"name": "Hingoli", "center": [77.14, 19.72], "bbox": [76.8, 19.4, 77.5, 20.1]}
    ],
    "bundelkhand": [
        {"name": "Jhansi", "center": [78.57, 25.44], "bbox": [78.2, 25.1, 79.0, 25.8]},
        {"name": "Lalitpur", "center": [78.41, 24.68], "bbox": [78.0, 24.3, 78.8, 25.0]},
        {"name": "Hamirpur", "center": [80.15, 25.95], "bbox": [79.7, 25.6, 80.5, 26.3]},
        {"name": "Mahoba", "center": [79.87, 25.29], "bbox": [79.5, 25.0, 80.2, 25.6]},
        {"name": "Banda", "center": [80.33, 25.47], "bbox": [79.9, 25.1, 80.8, 25.8]},
        {"name": "Chitrakoot", "center": [80.86, 25.17], "bbox": [80.5, 24.8, 81.2, 25.5]},
        {"name": "Tikamgarh", "center": [78.83, 24.74], "bbox": [78.4, 24.4, 79.2, 25.1]},
        {"name": "Chhatarpur", "center": [79.58, 24.91], "bbox": [79.2, 24.5, 80.0, 25.3]},
        {"name": "Panna", "center": [80.18, 24.72], "bbox": [79.8, 24.3, 80.6, 25.1]},
        {"name": "Damoh", "center": [79.44, 23.83], "bbox": [79.0, 23.4, 79.8, 24.2]},
        {"name": "Sagar", "center": [78.74, 23.83], "bbox": [78.3, 23.4, 79.2, 24.2]},
        {"name": "Datia", "center": [78.46, 25.67], "bbox": [78.1, 25.3, 78.8, 26.0]}
    ],
    "rayalaseema": [
        {"name": "Anantapur", "center": [77.60, 14.68], "bbox": [76.9, 13.9, 78.3, 15.3]},
        {"name": "Kadapa", "center": [78.82, 14.47], "bbox": [78.2, 13.8, 79.4, 15.1]},
        {"name": "Kurnool", "center": [78.03, 15.82], "bbox": [77.3, 15.1, 78.8, 16.5]},
        {"name": "Chittoor", "center": [79.10, 13.21], "bbox": [78.4, 12.6, 79.8, 13.8]}
    ]
}

def bbox_to_polygon(bbox):
    min_lon, min_lat, max_lon, max_lat = bbox
    return {
        "type": "Polygon",
        "coordinates": [[
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat]
        ]]
    }

def generate():
    for region_name, dist_list in DROUGHT_BOUNDARIES.items():
        features = []
        for d in dist_list:
            feat = {
                "type": "Feature",
                "properties": {
                    "shapeName": d["name"],
                    "NAME_2": d["name"],
                    "district_name": d["name"]
                },
                "geometry": bbox_to_polygon(d["bbox"])
            }
            features.append(feat)
            
        fc = {
            "type": "FeatureCollection",
            "features": features
        }
        
        out_file = RAW_DIR / f"{region_name}_districts.geojson"
        with open(out_file, "w") as f:
            json.dump(fc, f, indent=2)
        print(f"✓ Generated {out_file.name} ({len(features)} districts)")

if __name__ == "__main__":
    generate()
