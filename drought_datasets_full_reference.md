# Drought Model — Full Dataset Reference

Project Hydra — India Flood, Drought & Water-Scarcity Early Warning System.
All datasets below are real, publicly documented sources — no synthetic/fabricated data.
Feature datasets already cover all of India — only the bounding box (region) changes per demo area.

## 1. Feature Datasets (Google Earth Engine)

| Purpose | GEE Dataset ID | Catalog Link |
|---|---|---|
| Rainfall (CHIRPS Daily) — deficit vs. climatology | `UCSB-CHG/CHIRPS/DAILY` | https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY |
| Soil moisture, surface + root-zone (NASA SMAP L4) | `NASA/SMAP/SPL4SMGP/008` | https://developers.google.com/earth-engine/datasets/catalog/NASA_SMAP_SPL4SMGP_008 |
| NDVI, current + anomaly (MODIS) | `MODIS/061/MOD13Q1` | https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD13Q1 |
| Land surface temperature (MODIS) | `MODIS/061/MOD11A1` | https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A1 |
| Evapotranspiration, gap-filled (MODIS) | `MODIS/061/MOD16A2GF` | https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD16A2GF |
| Land cover fractions (Dynamic World) | `GOOGLE/DYNAMICWORLD/V1` | https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1 |

## 2. Forecast Feature (for near-term / 7-day horizon)

| Purpose | Link |
|---|---|
| Open-Meteo Forecast API — 16-day precipitation forecast, free, no key | https://open-meteo.com/en/docs |
| Open-Meteo Flood API — GloFAS river discharge forecast, up to ~210 days out | https://open-meteo.com/en/docs/flood-api |

## 3. Labels — Real Ground Truth (critical for training)

| Purpose | Link | Notes |
|---|---|---|
| India Drought Monitor — live + weekly archive district-level CDI (D0–D4) | https://indiadroughtmonitor.in/ | Archive from July 2021–present. **Verify historical/bulk download exists before building a lagged-training pipeline.** |
| India Drought Atlas — 120-year monthly precipitation & temperature climatology | https://github.com/wcl-iitgn/india-drought-atlas-data | 1901–2021, 0.05° grid. Used for anomaly/climatology baselines. |

## 4. Google Earth Engine — Access Setup

| Step | Link |
|---|---|
| Sign up / register Cloud Project | https://earthengine.google.com/signup/ |
| Access/registration guide | https://developers.google.com/earth-engine/guides/access |
| Noncommercial use info (free tier) | https://earthengine.google.com/noncommercial/ |
| Python API install | https://developers.google.com/earth-engine/guides/python_install |
| Auth guide | https://developers.google.com/earth-engine/guides/auth |

## 5. Demo Region Definitions (verified boundaries)

### Marathwada, Maharashtra — already built & validated
Districts: Aurangabad, Beed, Hingoli, Jalna, Latur, Nanded, Osmanabad, Parbhani.

### Bundelkhand, UP / MP
Bounding box: lat 23.1°N to 26.5°N, lon 78.1°E to 81.5°E. Area ~70,000 km².
Districts (13): Jhansi, Lalitpur, Jalaun, Hamirpur, Mahoba, Banda, Chitrakoot (Uttar Pradesh); Datia, Tikamgarh, Chhatarpur, Panna, Sagar, Damoh (Madhya Pradesh).

### Rayalaseema, Andhra Pradesh
Bounding box: lat 12.5°N to 16.25°N, lon 76.9°E to 79.9°E. Area ~81,000 km².
Historical 4 districts: Kurnool, Anantapur, Kadapa, Chittoor.

**Caution:** Andhra Pradesh redrew district boundaries in 2022 and again in Dec 2025. The historical 4 districts are now split across roughly 9 (Anantapuramu, Annamayya, Chittoor, Kurnool, Nandyal, Sri Sathya Sai, Tirupati, YSR Kadapa, and a disputed Markapuram). Before joining India Drought Monitor labels, confirm which district-name vintage the IDM data actually uses and build an explicit old-name-to-new-name mapping (same pattern already used for Aurangabad → Chhatrapati Sambhajinagar in the Marathwada pipeline).

### Saurashtra / Kutch, Gujarat
Bounding box: lat 20.7°N to 24.7°N, lon 68.2°E to 72.0°E. Saurashtra ~61,000 km² plus Kutch district.

**Note:** Kutch and Saurashtra can show opposite conditions in the same season (one in drought, one flooding), per 2026 monsoon reporting. Do not assume uniform risk across this combined area — this is expected real variation, not a data error.

## 6. Key Principle

The feature datasets (CHIRPS, SMAP, MODIS, Dynamic World) and label sources (India Drought Monitor, India Drought Atlas) already cover all of India — they are national/global satellite and monitoring products, not region-specific datasets. Expanding to a new region only requires changing the bounding box (area of interest) passed to Google Earth Engine, not sourcing new datasets. The real work in expanding regions is verifying district-label coverage and name-mapping consistency, not finding new data.
