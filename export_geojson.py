import os
import json
import pandas as pd
from pathlib import Path

def get_risk_attributes(score):
    if score <= 0.2:
        return "Low", "#00CC00"
    elif score <= 0.4:
        return "Moderate", "#FFCC00"
    elif score <= 0.6:
        return "High", "#FF6600"
    elif score <= 0.8:
        return "Severe", "#FF0000"
    else:
        return "Extreme", "#990000"

def create_polygon(lat, lon, delta=0.0225): # Half of 0.045
    return [
        [
            [lon - delta, lat - delta],
            [lon + delta, lat - delta],
            [lon + delta, lat + delta],
            [lon - delta, lat + delta],
            [lon - delta, lat - delta]
        ]
    ]

def main():
    base_dir = Path(r"c:\Users\riyav\project-hydra")
    data_dir = base_dir / "data"
    out_dir = base_dir / "output" / "geojson"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    projections_file = data_dir / "drought_projections.csv"
    if not projections_file.exists():
        print(f"Projections file not found at {projections_file}. Please run project_drought_risk.py first.")
        return
        
    print(f"Loading projections from {projections_file}...")
    df = pd.read_csv(projections_file)
    
    # Ensure necessary columns are present
    if 'date' not in df.columns:
        df['date'] = '2025-07-01' # Fallback date
    if 'region' not in df.columns:
        df['region'] = 'unknown_region'
        
    grouped = df.groupby(['region', 'date'])
    
    for (region, date), group in grouped:
        features = []
        for _, row in group.iterrows():
            lat = row['lat']
            lon = row['lon']
            risk_score = row.get('current_risk', 0.0)
            risk_level, risk_color = get_risk_attributes(risk_score)
            
            # Construct properties
            properties = {
                "cell_id": row.get('cell_id', f"{region}_{lat}_{lon}"),
                "lat": lat,
                "lon": lon,
                "date": date,
                "risk_score": round(risk_score, 4),
                "risk_level": risk_level,
                "risk_color": risk_color,
                "rain_deficit_30d_mm": round(row.get('rain_30d_deficit_mm', 0.0), 4),
                "ndvi_anomaly": round(row.get('ndvi_anomaly', 0.0), 4),
                "soil_moisture_rootzone": round(row.get('sm_rootzone', 0.0), 4),
                "dry_spell_days": row.get('dry_spell_days', 0),
                "projection_7d_risk": round(row.get('risk_7d', 0.0), 4),
                "projection_15d_risk": round(row.get('risk_15d', 0.0), 4),
                "projection_label": row.get('projection_label', 'Trend projection (not a forecast)')
            }
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": create_polygon(lat, lon)
                },
                "properties": properties
            }
            features.append(feature)
            
        feature_collection = {
            "type": "FeatureCollection",
            "features": features
        }
        
        # Save to file
        safe_date = str(date).replace(":", "-").replace("/", "-")
        out_filename = out_dir / f"drought_risk_{region}_{safe_date}.geojson"
        
        with open(out_filename, 'w') as f:
            json.dump(feature_collection, f, indent=2)
            
        print(f"Exported {len(features)} cells to {out_filename}")

if __name__ == "__main__":
    main()
