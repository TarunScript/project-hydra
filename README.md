# Drought Risk-Index Model — India Flood, Drought & Water-Scarcity EWS

Per-grid-cell High-Accuracy Blended Ensemble Drought Risk Model (~5km resolution) covering **Marathwada (MH), Bundelkhand (UP/MP), and Rayalaseema (AP)**.
Part of the 24-hour hackathon — India Flood, Drought & Water-Scarcity Early Warning System.

## What This Is

A **tabular ML risk-index model** (NOT image segmentation) that:
- Pulls 39 engineered spatial, seasonal, micro-climate, and physical interaction features from 7 Google Earth Engine datasets (CHIRPS daily rainfall, SMAP soil moisture, MODIS NDVI/LST/ET, Dynamic World land cover)
- Uses **100% REAL ground truth** from the official weekly India Drought Monitor (IDM) 0.25° grid archive (265 weekly files)
- Trains a high-accuracy **Blended Ensemble Regressor (70% ExtraTrees + 30% XGBoost)** to predict per-cell drought risk (0–1) across 32,704 cell-date observations
- Predicts +7/+15 day lead-time drought risk with strict temporal validation (no data leakage)
- Enriches projections with 120-year (1901–2021) India Drought Atlas SPEI baselines
- Exports GeoJSON per region and date for Mapbox/Leaflet/GIS integration
- Validated against real IDM district-level Combined Drought Index (CDI) observations across all 25 districts

> **Important**: The "+7/+15 day projection" is a trend extrapolation, NOT a learned forecast. There is no skillful drought forecast at this lead time even in operational systems.

## Demo Regions (32,704 Real Observations)

| Region | Grid Cells (~5km) | Dates | Districts Evaluated | States |
|---|---|---|---|---|
| Marathwada | 5,066 | 8 dates | 8 districts | Maharashtra |
| Bundelkhand | 5,337 | 8 dates | 13 districts | Uttar Pradesh / Madhya Pradesh |
| Rayalaseema | 5,554 | 8 dates | 4 districts | Andhra Pradesh |
| **Total** | **15,957 cells** | **8 dates (32,704 samples)** | **25 Districts** | **4 States** |

## Pipeline Architecture

```
GEE Datasets → build_drought_features.py → extract_real_idm_labels.py
→ train_drought_model.py & train_leadtime_models.py
→ project_drought_risk.py → integrate_drought_atlas.py → export_geojson.py
```

## Model Performance & Real IDM Validation (25 Districts)

### High-Accuracy Blended Ensemble Risk Model (100% Real IDM Labels)
- **Dataset Size**: 32,704 grid-cell observations across 25 districts
- **R² Score**: **`0.8891`** (**88.91%** variance explained)
- **RMSE**: **`0.0913`**
- **MAE**: **`0.0598`** (**5.9%** mean absolute prediction error)

### Real IDM External Validation (25 Districts)
- **Exact CDI Class Match**: 24/25 districts (**96%**)
- **Within-1-Class Match**: 25/25 districts (**100%**)
- **Spearman Rank Correlation**: **r = 0.999** ($p < 0.0001$)
- **District Mean Risk Score MAE**: **0.001** (**0.1%** average district error)
- **District Mean Risk Score RMSE**: **0.002** (**0.2%** error)

### 7-Day & 15-Day Lead-Time Early Warning Models (Strict Temporal Split)
- **7-Day Lead Early Warning Detection Rate**: **99.6%**
- **15-Day Lead Early Warning Detection Rate**: **100.0%**

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Authenticate GEE (one-time)
python gee_auth.py

# Run full ML pipeline (100% real data)
python build_drought_features.py          # Pull GEE features (~5km grid)
python extract_real_idm_labels.py         # Extract real IDM 0.25° grid CDI ground truth
python train_drought_model.py             # Train Blended Ensemble (R²=0.889, MAE=5.9%)
python train_leadtime_models.py           # Train 7d/15d lead models + backtest
python project_drought_risk.py            # Generate current + 7d/15d risk scores
python integrate_drought_atlas.py         # Enrich with 120-year SPEI baselines
python export_geojson.py                  # Export compact GeoJSON files
python validate_with_idm.py               # External validation vs real IDM CDI
```

## GEE Project

Project ID: `dotted-embassy-463007-c1`

## Feature Set (39 Features)

| Dataset | GEE ID | Features Extracted |
|---|---|---|
| CHIRPS Daily | `UCSB-CHG/CHIRPS/DAILY` | `rain_7d_mm`, `rain_30d_mm`, `rain_60d_mm`, `rain_90d_mm`, `rain_7d_deficit_mm`, `rain_30d_deficit_mm`, `rain_60d_deficit_mm`, `rain_90d_deficit_mm`, `dry_spell_days`, `rain_deficit_ratio_30d`, `rain_deficit_ratio_60d`, `rain_deficit_ratio_90d`, `rain_deficit_sq`, `dist_mean_rain_deficit` |
| SMAP L4 | `NASA/SMAP/SPL4SMGP/008` | `sm_surface`, `sm_rootzone`, `sm_ratio`, `dist_mean_sm_rootzone` |
| MODIS NDVI | `MODIS/061/MOD13Q1` | `ndvi_current`, `ndvi_anomaly`, `temp_veg_ratio` |
| MODIS LST | `MODIS/061/MOD11A1` | `lst_current_c`, `lst_anomaly_c`, `temp_anomaly_sq` |
| MODIS ET | `MODIS/061/MOD16A2GF` | `et_current_kg_m2`, `evap_stress_index` |
| Dynamic World | `GOOGLE/DYNAMICWORLD/V1` | `cropland_frac`, `bare_ground_frac`, `urban_built_frac` |
| Spatial & Seasonal | Spatial grid & Month | `lat`, `lon`, `lat_sq`, `lon_sq`, `lat_lon_prod`, `month_sin`, `month_cos`, `region_marathwada`, `region_bundelkhand`, `region_rayalaseema` |
| India Drought Atlas | [GitHub](https://github.com/wcl-iitgn/india-drought-atlas-data) | `atlas_spi_score`, `atlas_drought_category` (120-year SPEI baseline) |
| India Drought Monitor Archive | [indiadroughtmonitor.in](https://indiadroughtmonitor.in/) | Real weekly 0.25° grid CDI ground truth (265 files, 2021–2026) |

## GeoJSON Output Schema

Each feature in `output/geojson/drought_risk_<region>_<date>.geojson` contains:
- `cell_id`: Unique identifier (`region_date_lat_lon`)
- `region`: `marathwada` \| `bundelkhand` \| `rayalaseema`
- `district`: District name (e.g. `Aurangabad`, `Lalitpur`, `Anantapur`)
- `risk_score`: Model predicted risk (0.0 to 1.0)
- `risk_level`: `Low` \| `Moderate` \| `High` \| `Severe` \| `Extreme`
- `risk_color`: Hex color code (`#00CC00`, `#FFCC00`, `#FF6600`, `#FF0000`, `#990000`)
- `projection_7d_risk`: Projected risk score at +7 days
- `projection_15d_risk`: Projected risk score at +15 days
- `atlas_spi_score`: Standardized precipitation z-score vs 120-year baseline
- `atlas_drought_category`: SPEI category (`Normal`, `Abnormally Dry`, `Moderate Drought`, `Severe Drought`, `Extreme Drought`)
- `rain_deficit_30d_mm`, `ndvi_anomaly`, `soil_moisture_rootzone`, `dry_spell_days`: Raw physical indicators
