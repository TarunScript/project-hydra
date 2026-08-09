# 🌊 Project Hydra — India Flood & Drought Early Warning System

> Real-time flood risk prediction across 4 Indian states using XGBoost ML, live weather data from Open-Meteo, and an interactive map dashboard.

![Project Hydra](https://img.shields.io/badge/HackVerse-2.0-blue) ![Python](https://img.shields.io/badge/Python-3.10+-green) ![React](https://img.shields.io/badge/React-19-61dafb) ![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Troubleshooting](#troubleshooting)
- [Tech Stack](#tech-stack)

---

## Overview

Project Hydra predicts flood risk at the **district level** and **5km grid cell level** across **Assam, Bihar, West Bengal, and Odisha** using:

- **XGBoost classifier** trained on 430K+ satellite-derived feature samples
- **24 geospatial features** (elevation, rainfall, soil moisture, water occurrence, etc.)
- **Live weather integration** from Open-Meteo API with climatological fallback
- **Interactive MapLibre GL** dashboard with dark/satellite map styles
- **Timeline slider** for historical (-7d) to forecast (+15d) risk views

---

## Architecture

```
┌─────────────────────────┐     ┌──────────────────────────────┐
│   React Frontend        │────▶│  Flask API (Python)          │
│   (Vite + MapLibre GL)  │     │  Port 5001                   │
│   Port 5173             │     │  XGBoost model predictions   │
│                         │     │  Live weather from Open-Meteo│
│                         │     └──────────────────────────────┘
│                         │
│                         │     ┌──────────────────────────────┐
│                         │────▶│  Express API (Node.js)       │
│                         │     │  Port 3001                   │
└─────────────────────────┘     │  Emerging factors panel      │
                                └──────────────────────────────┘
```

---

## Prerequisites

Before you begin, make sure you have:

| Tool | Version | Check Command |
|------|---------|--------------|
| **Python** | 3.10+ | `python3 --version` |
| **Node.js** | 18+ | `node --version` |
| **npm** | 9+ | `npm --version` |
| **pip** | 21+ | `pip --version` |
| **Git** | Any | `git --version` |

---

## Quick Start

> **Clone and run everything in 4 commands:**

```bash
# 1. Clone the repository
git clone https://github.com/TarunScript/project-hydra.git
cd project-hydra

# 2. Install frontend dependencies
npm install

# 3. Install Python backend dependencies (in a virtual environment)
cd flood_model
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd ..

# 4. Start all servers (run each in a separate terminal)
```

### Terminal 1 — Python ML Backend (Port 5001)
```bash
cd flood_model
source .venv/bin/activate
python3 src/api_server.py
```

### Terminal 2 — Node.js Express Server (Port 3001)
```bash
cd server
npm install
node index.js
```

### Terminal 3 — React Frontend (Port 5173)
```bash
npm run dev
```

### 🌐 Open the app
Navigate to **http://localhost:5173** in your browser.

---

## Detailed Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/TarunScript/project-hydra.git
cd project-hydra
```

### Step 2: Frontend Setup

```bash
# Install Node.js dependencies
npm install

# Verify it works
npm run dev
# → Should show: VITE ready at http://localhost:5173/
# Press Ctrl+C to stop for now
```

### Step 3: Python Backend Setup

```bash
cd flood_model

# Create a Python virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

# Install all Python dependencies
pip install -r requirements.txt

# Verify: Start the API server
python3 src/api_server.py
# → Should show: 🌊 Project Hydra — Flood Risk API
# → Running on http://localhost:5001
```

### Step 4: Express Server Setup (Optional — for Emerging Factors panel)

```bash
cd server
npm install
node index.js
# → Should show: 🌊 Project Hydra API server running on port 3001
```

### Step 5: Verify Everything

Open your browser and check:

| URL | Expected |
|-----|----------|
| http://localhost:5173 | React dashboard with map |
| http://localhost:5001/api/health | `{"status": "ok", ...}` |
| http://localhost:3001/api/health | `{"status": "ok", ...}` |

---

## Project Structure

```
project-hydra/
├── src/                          # React Frontend (Vite)
│   ├── main.jsx                  # Entry point
│   ├── App.jsx                   # Main dashboard (map, timeline, panels)
│   ├── App.css                   # Dashboard styles
│   ├── LandingPage.jsx           # Landing page component
│   ├── LandingPage.css           # Landing page styles
│   ├── EmergingFactorsPanel.jsx  # AI-powered emerging factors
│   ├── EmergingFactorsPanel.css  # Emerging factors styles
│   └── index.css                 # Global styles & design system
│
├── flood_model/                  # Python ML Backend
│   ├── src/
│   │   ├── api_server.py         # Flask API server (main backend)
│   │   ├── live_weather.py       # Open-Meteo live weather fetcher
│   │   ├── train_flood_model.py  # XGBoost model training script
│   │   ├── validate_model.py     # Model validation & metrics
│   │   ├── pull_features.py      # GEE feature extraction (Assam)
│   │   ├── pull_features_multistate.py  # Multi-state features
│   │   ├── build_training_table.py      # Training data assembly
│   │   ├── build_labels.py              # Flood label generation
│   │   └── build_district_static_multistate.py
│   ├── models/
│   │   ├── flood_model_multistate.json  # Trained XGBoost model (4-state)
│   │   ├── flood_model.json             # Assam-only model
│   │   └── flood_model_multistate_meta.json
│   ├── data/
│   │   ├── raw/                  # District GeoJSON boundaries
│   │   └── features/             # Extracted feature CSVs (gitignored)
│   └── requirements.txt          # Python dependencies
│
├── server/                       # Node.js Express Server
│   ├── index.js                  # Express API entry point
│   └── emerging-factors.js       # Emerging factors service
│
├── index.html                    # Vite HTML entry point
├── package.json                  # Node.js dependencies
├── vite.config.js                # Vite configuration
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## API Endpoints

### Flask Backend (Port 5001)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check & model info |
| `GET` | `/api/risk-grid/flood?month=7&day=0` | District-level flood risk GeoJSON |
| `GET` | `/api/risk-grid/flood/cells?month=7&day=0` | 5km grid cell risk GeoJSON |
| `GET` | `/api/districts?month=7` | List of districts with risk scores |
| `GET` | `/api/feature-importance` | XGBoost feature importance |

**Query Parameters:**
- `month` (5-10): Monsoon month (default: current)
- `day` (-7 to 15): Day offset for historical/forecast (default: 0)
- `min_risk` (0.0-1.0): Filter cells by minimum risk score
- `max_cells` (int): Cap number of returned cells (default: 3000)

### Express Backend (Port 3001)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/emerging-factors?lat=26&lon=92` | AI emerging risk factors |

---

## Troubleshooting

### ❌ `ModuleNotFoundError: No module named 'numpy'`
You're not in the virtual environment. Run:
```bash
cd flood_model
source .venv/bin/activate
pip install -r requirements.txt
```

### ❌ `EADDRINUSE: Port 5001 already in use`
Another process is using port 5001. Kill it:
```bash
lsof -ti:5001 | xargs kill -9     # macOS/Linux
```

### ❌ `EADDRINUSE: Port 5173 already in use`
```bash
lsof -ti:5173 | xargs kill -9     # macOS/Linux
```

### ❌ Open-Meteo API rate limit (HTTP 429)
The free tier allows 10,000 requests/day. The limit resets at midnight UTC (5:30 AM IST).
The app will automatically fall back to **climatological baseline data** — the map still works, just with historical averages instead of live weather.

### ❌ `FileNotFoundError: training_table_multistate.csv`
The large training CSV files are gitignored (>100MB). You need to either:
1. Generate them by running the feature extraction pipeline, or
2. Get them from a team member

### ❌ Frontend shows "Failed to fetch" / CORS error
Make sure the Flask backend is running on port 5001 before opening the frontend.

### ❌ Map not loading / blank screen
Check browser console (F12) for errors. Common fixes:
- Ensure all 3 servers are running
- Hard refresh: `Ctrl+Shift+R` / `Cmd+Shift+R`

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 19, Vite 8, MapLibre GL JS, Lucide Icons |
| **ML Backend** | Python 3, Flask, XGBoost, Pandas, GeoPandas |
| **Weather Data** | Open-Meteo API (free, no key required) |
| **Satellite Data** | Google Earth Engine (CHIRPS, MODIS, ERA5-Land) |
| **Express Backend** | Node.js, Express 5 |
| **Map Tiles** | CARTO Dark, Esri World Imagery |

---

## Team

Built for **HackVerse 2.0** 🚀

---

## License

MIT License — see [LICENSE](LICENSE) for details.
