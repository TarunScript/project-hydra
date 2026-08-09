# Drought Model — Person C's build (Section 5.2)

Implements the drought half of the plan: GEE feature pipeline → historical
label join → XGBoost training → trend+climatology forecast projection →
GeoJSON export for the frontend timeline slider → alert trigger logic.

Note: the `project-hydra` repo currently only has a placeholder README on
`main` — drop these files into a `drought/` folder there (or wherever the
flood pipeline actually lands) and adjust the relative imports if you
reorganize into subpackages.

## Files

| File | Role |
|---|---|
| `config.py` | Regions, grid scale, dataset IDs, thresholds — single source of truth |
| `gee_setup.py` | GEE auth/init (Section 4) |
| `climatology.py` | Long-term baselines for rainfall/NDVI/LST anomalies |
| `feature_pipeline.py` | Pulls all Section 5.2 features into a per-cell CSV |
| `labels.py` | India Drought Monitor CDI labels, joined to grid cells by district |
| `train_model.py` | Trains XGBoost (or Random Forest) regressor → 0-1 risk score |
| `forecast_projection.py` | Trend + climatology projection for 7/15/30/60/90-day horizons |
| `alert_logic.py` | Risk-level thresholds + alert message generation (Section 7) |
| `export_geojson.py` | Precomputed per-day/per-horizon GeoJSON for the map (Section 8) |

## Run order (hour-by-hour, matches Section 10's "C" column)

```bash
pip install -r requirements.txt

# Hour 0-2: label sourcing (do this in parallel with A's pipeline coming online)
# -> manually download a few weeks of India Drought Monitor CDI tables for your
#    demo region's districts into data/manual_drought_monitor/*.csv
#    columns needed: district, date, cdi_class

# Hour 2-8: feature engineering + training
python gee_setup.py YOUR-CLOUD-PROJECT-ID
python feature_pipeline.py --region marathwada --project YOUR-CLOUD-PROJECT-ID
python labels.py --region marathwada --grid-geojson data/grid_marathwada.geojson \
    --district-boundaries data/marathwada_districts.geojson
python train_model.py --region marathwada

# Hour 8-10: alert trigger logic (this is also usable by the flood model —
# see alert_logic.py's docstring)
python forecast_projection.py --region marathwada

# Hour 10-16: hand off to D (frontend) via GeoJSON
python export_geojson.py --region marathwada \
    --grid-geojson data/grid_marathwada.geojson \
    --projection-csv data/drought_projection_marathwada.csv
```

## Labels: real endpoint, no manual download needed

`labels.py` pulls directly from India Drought Monitor's public data files —
confirmed working, no scraping, no district shapefile:

- `https://indiadroughtmonitor.in/data/Current_CDI.txt` — latest week
- `https://indiadroughtmonitor.in/data/Drough_TS/CDI_YYYYMMDD.txt` — any
  past week (site states coverage back to July 2021)

Both are plain text, one row per 0.25° grid point: `lat lon cdi_value`
(drier = more negative; `NaN NaN NaN` marks no-data cells). This is a raw
CDI value, not a pre-baked D0–D4 class, so `labels.py` buckets it into
classes using standard SPI-style thresholds — that bucketing is our own
choice, not an official IDM boundary, so say that plainly if you cite
"D0–D4" in the pitch. Because it's already a lat/lon grid, there's no
district join at all — `snap_labels_to_grid_cells()` just nearest-point-
matches each of your feature grid's cells to the closest CDI point.

**Archive weeks land on Wednesdays only** — confirmed by cross-checking
`India_Drought_Area_Timeseries.txt` (one row per real archive week).
`labels.py` auto-snaps any date you pass to the nearest Wednesday and
prints what it snapped to, so you don't need to hunt for valid dates by
hand:
```bash
python3 labels.py --region marathwada --dates 20240815 20240822
# -> snaps to 20240814 and 20240821 automatically, prints the snap
```
If you want the exact date without snapping, check
`India_Drought_Area_Timeseries.txt` yourself first (columns are
`year month day <national stats...>`, one row per real week).

## What still needs a human decision, per the plan's own flags

- **GRACE feature is regional, not per-cell** (Section 3.5's own caveat) —
  `groundwater_trend_feature()` broadcasts one value across the whole grid.
  Say this plainly if it shows up as an important feature in the pitch.
- **Climatology baseline** defaults to an in-pipeline CHIRPS/MODIS mean
  (1991-2020). Swap in `climatology.load_drought_atlas_climatology()` if you
  clone the India Drought Atlas repo and want the more citation-backed
  1901-2021 baseline instead.
- **Time-based train/test split** in `train_model.py` avoids leaking future
  weeks into training — don't switch this to a random split just to get a
  better-looking validation number; it would be dishonest given how the
  model is actually used (projecting forward).
