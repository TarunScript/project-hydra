# Project Hydra — India Flood & Drought Early Warning System

Interactive geospatial dashboard for flood and drought risk analysis across India, built with React + MapLibre GL + Express.

## Quick Start

```bash
# Install dependencies
npm install

# Start the API server (port 3001, runs in MOCK mode)
npm run server

# In another terminal, start the frontend (port 5173)
npm run dev
```

Open http://localhost:5173 in your browser.

## Architecture

| Layer | Stack | Port |
|-------|-------|------|
| **Frontend** | React 19 + Vite 8 + MapLibre GL JS | 5173 |
| **Backend API** | Express.js (Node 18+) | 3001 |
| **Map Tiles** | CartoDB Dark Matter / ESRI Satellite (free, no key needed) | — |

## Features

- 🗺️ Interactive dark-theme map with risk grid overlays
- 🔍 Global city/district search (OpenStreetMap Nominatim)
- 🛰️ Dark Map ↔ Satellite View toggle
- 📊 Risk analysis detail panel with environmental factors
- 🔬 **Emerging Factors Panel** — surfaces real-world developments that could affect local flood/drought risk (see below)
- 📲 Simulated SMS alert trigger

---

## 🔬 Emerging Factors Panel

### What it does
Given a selected map location, the panel surfaces **real, sourced, recent** developments near that location that could plausibly affect local flood/drought risk — things that static satellite or weather data can't see.

### Categories monitored (strictly enforced)
1. New/expanding data centers or AI infrastructure
2. New/expanding semiconductor fabrication plants
3. Lithium/rare-earth/critical-mineral extraction projects
4. Green hydrogen production or direct air capture facilities
5. Groundwater extraction trends or new industrial water permits
6. Major upstream dam/reservoir/irrigation changes
7. Large-scale land-use change (deforestation, urban expansion)

### Important: This is NOT a risk score
The panel output is **informational context**, displayed alongside (but never blended into) the flood/drought risk scores. The API response structure makes this explicit:

```json
{
  "emerging_factors": { "findings": [...], ... },
  "flood_risk": null,
  "drought_risk": null
}
```

The risk fields are `null` because the flood and drought models are being built on separate branches. When those branches are ready, the `null` values get replaced with real model outputs.

### API Endpoint

```
GET /api/emerging-factors?lat=26.15&lon=92.65&location_name=Assam
```

### Current mode: MOCK
The panel ships with hardcoded mock data for **Assam (Brahmaputra Basin)** and **Marathwada (Latur)**. No API key is needed to demo.

### Switching to live search
1. Copy `.env.example` to `.env`
2. Add your Perplexity API key: `PERPLEXITY_API_KEY=pplx-xxxx`
3. Restart the server: `npm run server`

The code auto-detects the key and switches from mock to live Perplexity Sonar search — no code changes required.

---

## 🔗 Branch Integration Guide

This feature lives on the `feature/emerging-factors-panel` branch and is designed to merge cleanly into `main` once the other branches are ready.

### Files added by this branch
```
server/
  index.js                    ← Express API server
  emerging-factors.js         ← Emerging factors service (mock + live API)
src/
  EmergingFactorsPanel.jsx    ← React panel component
  EmergingFactorsPanel.css    ← Panel styles
.env.example                  ← API key placeholder
```

### How to merge with the flood/drought model branches
1. Merge `shanmuk-frontend` into `main` first (base UI)
2. Merge `feature/emerging-factors-panel` (this branch)
3. Merge the flood model branch and drought model branch
4. In the combined codebase, update `server/emerging-factors.js` to import the real risk score functions and replace:
   ```js
   flood_risk: null,    // → flood_risk: await getFloodRisk(lat, lon),
   drought_risk: null,  // → drought_risk: await getDroughtRisk(lat, lon),
   ```
5. Set `PERPLEXITY_API_KEY` in `.env` to enable live web search

### No merge conflicts expected
- The backend (`server/`) is entirely new — no overlap with existing files
- The frontend component (`EmergingFactorsPanel.jsx`) is self-contained
- Only `App.jsx` has minor additions (import + toggle button + panel render)
- `package.json` has two new dependencies (`express`, `cors`) and one new script (`server`)
