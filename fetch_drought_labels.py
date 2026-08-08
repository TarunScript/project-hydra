import os
import sys
import pandas as pd
import numpy as np
import requests
import warnings
from datetime import datetime, timedelta

# Suppress warnings
warnings.filterwarnings('ignore')

DATA_DIR = 'data'
FEATURES_FILE = os.path.join(DATA_DIR, 'drought_features_all_regions.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'drought_features_labeled.csv')

# District mapping for the demo regions
REGION_DISTRICTS = {
    'marathwada': ['Aurangabad', 'Beed', 'Osmanabad', 'Hingoli', 'Jalna', 'Latur', 'Nanded', 'Parbhani'],
    'bundelkhand': ['Banda', 'Chitrakoot', 'Hamirpur', 'Jalaun', 'Jhansi', 'Lalitpur', 'Mahoba', 
                    'Chhatarpur', 'Damoh', 'Datia', 'Panna', 'Sagar', 'Tikamgarh'],
    'rayalaseema': ['Anantapur', 'Chittoor', 'Kadapa', 'Kurnool']
}

# Label encoding for Combined Drought Index (CDI)
CDI_MAPPING = {
    'No Drought': 0.0,
    'D0': 0.2, # Abnormally dry
    'D1': 0.4, # Moderate
    'D2': 0.6, # Severe
    'D3': 0.8, # Extreme
    'D4': 1.0  # Exceptional
}

def attempt_fetch_real_data():
    """Attempt to pull CDI data from India Drought Monitor."""
    url = "https://indiadroughtmonitor.in/data" # Dummy endpoint for attempt
    print(f"Attempting to fetch live drought labels from {url}...")
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            # Assuming successful response contains JSON or CSV
            print("Successfully accessed endpoint, but data format parsing not implemented.")
            return False
        else:
            print(f"Failed to fetch data (Status Code: {response.status_code}).")
            return False
    except requests.RequestException as e:
        print(f"Connection failed: {e}")
        return False

def generate_synthetic_labels(df):
    """
    MANUAL FALLBACK: Generate synthetic CDI labels based on feature data proxies.
    Uses rainfall deficit and NDVI anomaly to approximate drought severity.
    """
    print("\n" + "="*60)
    print("WARNING: USING SYNTHETIC/PROXY LABELS FOR DROUGHT SEVERITY.")
    print("Live API access failed or unavailable. Generating labels based on")
    print("rain_90d_deficit_mm and ndvi_anomaly.")
    print("="*60 + "\n")
    
    # Calculate a composite proxy score (lower is worse drought)
    # Normalize features roughly
    
    # Fill NAs
    df['rain_90d_deficit_mm'] = df.get('rain_90d_deficit_mm', pd.Series(np.zeros(len(df)))).fillna(0)
    df['ndvi_anomaly'] = df.get('ndvi_anomaly', pd.Series(np.zeros(len(df)))).fillna(0)
    
    # A simple formula: large negative rain deficit and negative NDVI anomaly means high drought risk
    # This is a heuristic purely for generating synthetic labels for the hackathon
    proxy_score = df['rain_90d_deficit_mm'] + (df['ndvi_anomaly'] * 100)
    
    # Map to CDI classes based on quantiles to get a spread of classes
    # More negative proxy_score -> worse drought
    conditions = [
        (proxy_score < -150),       # D4
        (proxy_score < -100),       # D3
        (proxy_score < -50),        # D2
        (proxy_score < -20),        # D1
        (proxy_score < 0)           # D0
    ]
    choices = [1.0, 0.8, 0.6, 0.4, 0.2]
    
    df['risk_score'] = np.select(conditions, choices, default=0.0)
    df['is_synthetic_label'] = True
    
    return df

def map_latlon_to_district(lat, lon, region):
    """
    Mock function to map a lat/lon point to a district name.
    In a real scenario, this would use a spatial join with a district shapefile.
    Here we just assign a random district from the region for demonstration.
    """
    districts = REGION_DISTRICTS.get(region.lower(), ['Unknown'])
    # Deterministic pseudo-random choice based on lat/lon
    idx = int((lat + lon) * 100) % len(districts)
    return districts[idx]

def main():
    # Try loading combined file first, fallback to individual files
    if os.path.exists(FEATURES_FILE):
        print(f"Loading features from {FEATURES_FILE}...")
        df = pd.read_csv(FEATURES_FILE)
    else:
        print(f"Combined file not found at {FEATURES_FILE}")
        print("Looking for individual feature CSVs...")
        import glob
        csv_files = glob.glob(os.path.join(DATA_DIR, 'drought_features_*_20*.csv'))
        if not csv_files:
            # Also try the test file
            test_file = os.path.join(DATA_DIR, 'drought_features_marathwada_test.csv')
            if os.path.exists(test_file):
                csv_files = [test_file]
            else:
                print("Error: No feature CSV files found. Run build_drought_features.py first.")
                sys.exit(1)
        
        print(f"Found {len(csv_files)} feature files, concatenating...")
        dfs = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(dfs, ignore_index=True)
        print(f"Combined: {len(df)} rows from {len(csv_files)} files")
    
    print(f"Loaded {len(df)} rows x {len(df.columns)} columns")
    
    # 1 & 2: Attempt to fetch real labels, otherwise fallback to synthetic
    success = attempt_fetch_real_data()
    
    if not success:
        df = generate_synthetic_labels(df)
    
    # Rename to match training script's expected column name
    df.rename(columns={'risk_score': 'drought_risk_score'}, inplace=True)
    
    # Map districts
    print("Mapping coordinates to districts...")
    if 'region' in df.columns and 'lat' in df.columns and 'lon' in df.columns:
        df['district'] = df.apply(lambda row: map_latlon_to_district(row['lat'], row['lon'], row['region']), axis=1)
    else:
        print("Warning: Missing lat, lon, or region columns. Cannot map districts properly.")
        df['district'] = 'Unknown'

    print(f"Saving labeled features to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("\nDone! Labeling complete.")
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nLabel distribution (drought_risk_score):")
    print(df['drought_risk_score'].value_counts().sort_index())
    print(f"\nRegions: {df['region'].unique().tolist() if 'region' in df.columns else 'N/A'}")
    print(f"Dates: {df['date'].nunique() if 'date' in df.columns else 'N/A'} unique")

if __name__ == "__main__":
    main()

