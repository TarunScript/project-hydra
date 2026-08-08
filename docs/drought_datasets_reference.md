# Drought Model — Dataset Reference

Part of the India Flood, Drought & Water-Scarcity Early Warning System hackathon build.
This file covers only the **drought model's** datasets (Section 5.2 / 3 of the master plan).

---

## 1. Rainfall / Precipitation Deficit

| Purpose | Source | GEE ID / Endpoint | Resolution & Cadence | Notes |
|---|---|---|---|---|
| Historical daily rainfall, 7/30/60/90-day deficit calcs | CHIRPS | `UCSB-CHG/CHIRPS/DAILY` (or newer `UCSB-CHC/CHIRPS/V3/DAILY_SAT`) | 0.05° (~5.5 km), daily, 1981–present | Primary rainfall-deficit feature source |
| Live / current rainfall | Open-Meteo Forecast API | `https://api.open-meteo.com/v1/forecast` | ~1–11 km, hourly | Free, no API key, no rate-limit headaches |

## 2. Soil Moisture

| Purpose | Source | GEE ID | Resolution & Cadence |
|---|---|---|---|
| Surface + root-zone soil moisture | NASA SMAP L4 | `NASA/SMAP/SPL4SMGP/008`, bands `sm_surface`, `sm_rootzone` | ~9–11 km, every 3 hours |

## 3. Vegetation Stress

| Purpose | Source | GEE ID | Resolution & Cadence |
|---|---|---|---|
| NDVI/EVI (current + anomaly vs. climatology) | MODIS | `MODIS/061/MOD13Q1` | 250 m, 16-day composite |

## 4. Temperature

| Purpose | Source | GEE ID | Resolution & Cadence |
|---|---|---|---|
| Land surface temperature / anomaly | MODIS | `MODIS/061/MOD11A1` | 1 km, daily |

## 5. Evapotranspiration

| Purpose | Source | GEE ID | Resolution & Cadence | Note |
|---|---|---|---|---|
| Evapotranspiration | MODIS | `MODIS/061/MOD16A2GF` | 500 m, 8-day, 2000–present | ⚠️ Use the **gap-filled** version — `MOD16A2` (non-GF) only covers 2021+ |

## 6. Dry-Spell Duration

Derived feature — no separate dataset. Compute consecutive dry days directly from CHIRPS daily rainfall.

## 7. Groundwater / Storage Trend (optional)

| Purpose | Source | GEE ID | Resolution & Cadence | Note |
|---|---|---|---|---|
| Groundwater/storage trend | NASA GRACE | `NASA/GRACE/MASS_GRIDS_V04/LAND` | ~300 km cells, monthly | ⚠️ Very coarse — regional/basin trend only, not a per-cell feature. State this plainly if used. |

## 8. Land Cover

| Purpose | Source | GEE ID | Resolution & Cadence |
|---|---|---|---|
| Land cover, urban/built-up % | Dynamic World | `GOOGLE/DYNAMICWORLD/V1` | 10 m, updates every 2–5 days |

---

## 9. Labels (Ground Truth) — the important part

| Need | Source | Link | What it gives you |
|---|---|---|---|
| **Drought labels — best option, and it's live** | India Drought Monitor, IIT Gandhinagar | https://indiadroughtmonitor.in/ | Live, weekly, district-level Combined Drought Index (5-class: D0 abnormally dry → D4 exceptional), downloadable, archive from July 2021–present. Use as both training ground truth and to sanity-check model output. |
| **Drought labels — long-term climatology for anomaly baselines** | India Drought Atlas, IIT Gandhinagar | https://github.com/wcl-iitgn/india-drought-atlas-data | 0.05° gridded monthly precipitation & temperature, 1901–2021 — exactly what's needed to compute "how abnormal is this" anomalies. |

---

## 10. Quick Reference — All Links

**Google Earth Engine**
- Sign up: https://earthengine.google.com/signup/
- Access/registration guide: https://developers.google.com/earth-engine/guides/access
- Noncommercial use info: https://earthengine.google.com/noncommercial/
- Code Editor: https://code.earthengine.google.com
- Python install: https://developers.google.com/earth-engine/guides/python_install
- Auth guide: https://developers.google.com/earth-engine/guides/auth

**Forecast APIs**
- Open-Meteo Forecast: https://open-meteo.com/en/docs

**India-specific drought data**
- India Drought Monitor: https://indiadroughtmonitor.in/
- India Drought Atlas data repo: https://github.com/wcl-iitgn/india-drought-atlas-data

---

## 11. Modeling Note

There is **no skillful 60–90 day drought forecast**, even in operational systems. Frame the "future" drought projection honestly as a **trend + climatology projection** (current rainfall/NDVI deficit trajectory compared against the India Drought Atlas climatology) — not a hard forecast. This is the standard the master plan recommends stating plainly in the pitch.

**Modeling approach:** XGBoost or Random Forest classifier/regressor, trained on per-grid-cell feature tables (Sections 1–8 above) with India Drought Monitor labels (Section 9) as the target. Same-day-trainable on a laptop, no GPU needed.
