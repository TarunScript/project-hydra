# Drought Risk-Index Model — India Flood, Drought & Water-Scarcity EWS

Per-grid-cell XGBoost drought risk model (~5km resolution) for Marathwada, Maharashtra.
Part of the 24-hour hackathon — India Flood, Drought & Water-Scarcity Early Warning System.

## What This Is

A **tabular ML risk-index model** (NOT image segmentation) that:
- Pulls 20 features from 7 Google Earth Engine datasets (CHIRPS, SMAP, MODIS NDVI/LST/ET, Dynamic World)
- Trains an XGBoost regressor to predict per-cell drought risk (0–1)
- Projects risk forward +7/+15 days via climatological trend extrapolation
- Enriches projections with 120-year India Drought Atlas SPEI baselines
- Exports GeoJSON per date for Mapbox/Leaflet frontend
- Validates against real India Drought Monitor (IDM) district-level CDI data

> **Important**: The "+7/+15 day projection" is a trend extrapolation, NOT a learned forecast. There is no skillful drought forecast at this lead time even in operational systems.

## Demo Regions

| Region | Coverage | States |
|---|---|---|
| Marathwada | 5,066 cells (~5km) | Maharashtra |
| Bundelkhand | *Planned* | UP / MP |
| Rayalaseema | *Planned* | AP |

## Pipeline

```
GEE Datasets → build_drought_features.py → fetch_drought_labels.py
→ train_drought_model.py → project_drought_risk.py
→ integrate_drought_atlas.py → export_geojson.py
```

## Validation

```
validate_with_idm.py   # External validation against real IDM CDI data
```
IDM validation results: `data/validation/idm_validation_report.csv`

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Authenticate GEE (one-time)
python gee_auth.py

# Run full pipeline
python build_drought_features.py          # ~45 min — pulls GEE data
python fetch_drought_labels.py            # Label integration
python train_drought_model.py             # XGBoost training
python project_drought_risk.py            # +7/+15 day projections
python integrate_drought_atlas.py         # 120-year Atlas enrichment
python export_geojson.py                  # Frontend GeoJSON

# External validation (IDM data required)
python validate_with_idm.py
```

## GEE Project

Project ID: `dotted-embassy-463007-c1`

## Model Performance (Marathwada, synthetic labels)

| Metric | Value |
|---|---|
| RMSE | 0.031 |
| MAE | 0.014 |
| R² | 0.992 |
| IDM District Match (exact) | 5/8 (62%) |
| IDM District Match (±1 class) | 6/8 (75%) |

## GeoJSON Schema

Each cell feature includes:
- `risk_score` (0-1), `risk_level`, `risk_color`
- `projection_7d_risk`, `projection_15d_risk`
- `atlas_spi_score` (SPEI-based z-score vs. 120-year baseline)
- `atlas_drought_category` (Normal / Abnormally Dry / Moderate / Severe / Extreme Drought)
- Raw features: `rain_deficit_30d_mm`, `ndvi_anomaly`, `soil_moisture_rootzone`, `dry_spell_days`

## Data Sources

| Dataset | GEE ID | Description |
|---|---|---|
| CHIRPS Daily | `UCSB-CHG/CHIRPS/DAILY` | Rainfall + deficit vs. 2000-2022 climatology |
| SMAP L4 | `NASA/SMAP/SPL4SMGP/008` | Soil moisture (surface + root-zone) |
| MODIS NDVI | `MODIS/061/MOD13Q1` | Vegetation index + anomaly |
| MODIS LST | `MODIS/061/MOD11A1` | Land surface temperature + anomaly |
| MODIS ET | `MODIS/061/MOD16A2GF` | Evapotranspiration (gap-filled) |
| Dynamic World | `GOOGLE/DYNAMICWORLD/V1` | Land cover fractions |
| India Drought Atlas | [GitHub](https://github.com/wcl-iitgn/india-drought-atlas-data) | 120-year SPEI baseline |
| India Drought Monitor | [indiadroughtmonitor.in](https://indiadroughtmonitor.in/) | Real CDI (external validation only) |
