import React, { useEffect, useRef, useState, useCallback } from 'react';
import { BrowserRouter, Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import * as maplibregl from 'maplibre-gl';
import EmergingFactorsPanel from './EmergingFactorsPanel.jsx';
import './EmergingFactorsPanel.css';
import LandingPage from './LandingPage.jsx';

// API Backend URL — connects to the Flask server serving real XGBoost predictions
const API_BASE = 'http://localhost:5001/api';

// ─── Calendar Date Helper ────────────────────────────────────────
const getCalendarDate = (dayOffset, format = 'short') => {
  const d = new Date();
  d.setDate(d.getDate() + dayOffset);
  if (format === 'full') {
    return d.toLocaleDateString('en-IN', { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' });
  }
  return d.toLocaleDateString('en-IN', { month: 'short', day: 'numeric' });
};

const getTimelineLabel = (dayOffset) => {
  if (dayOffset === 0) return `Today · ${getCalendarDate(0)}`;
  if (dayOffset < 0) return `${getCalendarDate(dayOffset)} (${dayOffset}d)`;
  return `${getCalendarDate(dayOffset)} (+${dayOffset}d)`;
};

const getDayBadge = (dayOffset) => {
  if (dayOffset < 0) return 'Historical';
  if (dayOffset === 0) return 'Live · Today';
  return 'Forecast';
};

// ─── Regions ─────────────────────────────────────────────────────
const DEMO_REGIONS = {
  all_floods: {
    name: 'All Flood States (4-State Model)',
    center: [87.5, 23.5],
    zoom: 6.0
  },
  assam: {
    name: 'Assam',
    center: [92.8, 26.15],
    zoom: 8.0
  },
  bihar: {
    name: 'Bihar',
    center: [85.8, 25.8],
    zoom: 8.0
  },
  west_bengal: {
    name: 'West Bengal',
    center: [87.8, 24.0],
    zoom: 7.5
  },
  odisha: {
    name: 'Odisha',
    center: [84.8, 20.5],
    zoom: 7.5
  },
  marathwada: {
    name: 'Marathwada (Drought Case)',
    center: [76.5, 19.15],
    zoom: 8.5
  },
  rayalaseema: {
    name: 'Rayalaseema (Drought Zone)',
    center: [78.2, 14.5],
    zoom: 8.5
  },
  kerala: {
    name: 'Kerala (Monsoon Inundation)',
    center: [76.3, 9.9],
    zoom: 8.5
  }
};

// ─── Map Styles ──────────────────────────────────────────────────
const MAP_STYLES = {
  dark: {
    version: 8,
    sources: {
      'carto-dark': {
        type: 'raster',
        tiles: [
          'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
          'https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
          'https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
          'https://d.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png'
        ],
        tileSize: 256,
        attribution: '&copy; OpenStreetMap &copy; CARTO'
      },
      'carto-labels': {
        type: 'raster',
        tiles: [
          'https://a.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}.png',
          'https://b.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}.png',
          'https://c.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}.png',
          'https://d.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}.png'
        ],
        tileSize: 256
      }
    },
    layers: [
      {
        id: 'carto-dark-layer',
        type: 'raster',
        source: 'carto-dark',
        minzoom: 0,
        maxzoom: 19
      },
      {
        id: 'carto-labels-layer',
        type: 'raster',
        source: 'carto-labels',
        minzoom: 0,
        maxzoom: 19
      }
    ]
  },
  satellite: {
    version: 8,
    sources: {
      'esri-satellite': {
        type: 'raster',
        tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
        tileSize: 256,
        attribution: 'Esri, Maxar, Earthstar Geographics'
      },
      'esri-labels': {
        type: 'raster',
        tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}'],
        tileSize: 256
      }
    },
    layers: [
      {
        id: 'esri-satellite-layer',
        type: 'raster',
        source: 'esri-satellite',
        minzoom: 0,
        maxzoom: 19
      },
      {
        id: 'esri-labels-layer',
        type: 'raster',
        source: 'esri-labels',
        minzoom: 0,
        maxzoom: 19
      }
    ]
  }
};

// ─── Dashboard Component ─────────────────────────────────────────
function Dashboard() {
  const navigate = useNavigate();
  const mapContainer = useRef(null);
  const map = useRef(null);
  const [features, setFeatures] = useState([]);
  const [selectedCell, setSelectedCell] = useState(null);
  const [timelineDay, setTimelineDay] = useState(0);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [detailOpen, setDetailOpen] = useState(false);
  const [currentStyle, setCurrentStyle] = useState('dark');
  const [selectedRegion, setSelectedRegion] = useState('all_floods');
  const [isLoading, setIsLoading] = useState(true);
  const [viewMode, setViewMode] = useState('districts');
  const [weatherWarning, setWeatherWarning] = useState(null);

  // Emerging Factors panel state
  const [efpOpen, setEfpOpen] = useState(false);

  // Search state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);

  const selectedRegionRef = useRef(selectedRegion);
  selectedRegionRef.current = selectedRegion;
  const timelineDayRef = useRef(timelineDay);
  timelineDayRef.current = timelineDay;

  // ── Fetch real risk data from Flask API ────────────────────────
  const fetchRiskData = useCallback(async (region = selectedRegion, month = null, mode = viewMode, day = timelineDay) => {
    try {
      setIsLoading(true);
      const monthParam = month !== null ? `month=${month}` : 'month=7';
      const dayParam = `day=${day}`;
      const endpointRegion = ['all_floods', 'assam', 'bihar', 'west_bengal', 'odisha'].includes(region) ? 'flood' : region;

      let url;
      if (mode === 'zones') {
        let bboxParams = '';
        if (map.current) {
          const bounds = map.current.getBounds();
          bboxParams = `&minlon=${bounds.getWest().toFixed(4)}&maxlon=${bounds.getEast().toFixed(4)}&minlat=${bounds.getSouth().toFixed(4)}&maxlat=${bounds.getNorth().toFixed(4)}`;
        }
        url = `${API_BASE}/risk-grid/${endpointRegion}/cells?${monthParam}&${dayParam}&min_risk=0.0&max_cells=3000${bboxParams}`;
      } else {
        url = `${API_BASE}/risk-grid/${endpointRegion}?${monthParam}&${dayParam}`;
      }

      let res;
      try {
        res = await fetch(url);
      } catch (err) {
        const altUrl = url.includes('localhost')
          ? url.replace('localhost', '127.0.0.1')
          : url.replace('127.0.0.1', 'localhost');
        console.warn(`Primary fetch failed, retrying with fallback URL: ${altUrl}`);
        res = await fetch(altUrl);
      }
      const data = await res.json();

      setWeatherWarning(data.metadata?.weather_warning || null);

      if (data.features && data.features.length > 0) {
        setFeatures(data.features);
        setSelectedCell(data.features[0].properties);
        setDetailOpen(true);
        console.log(`✓ Loaded ${data.features.length} ${mode} (day=${day}) | total=${data.features.length}`);
      }
    } catch (err) {
      console.error('Failed to fetch risk data from API:', err);
    } finally {
      setIsLoading(false);
    }
  }, [selectedRegion, viewMode, timelineDay]);

  // Re-fetch when region, viewMode, or timelineDay changes
  useEffect(() => {
    fetchRiskData(selectedRegion, null, viewMode, timelineDay);
  }, [selectedRegion, viewMode, timelineDay]);

  // ── Risk Counts for Stat Pills ─────────────────────────────────
  const riskCounts = React.useMemo(() => {
    const counts = { severe: 0, high: 0, moderate: 0, low: 0 };
    features.forEach(f => {
      const level = f.properties?.risk_level;
      if (level && counts[level] !== undefined) counts[level]++;
    });
    return counts;
  }, [features]);

  // ── Map Layers ─────────────────────────────────────────────────
  const addRiskLayers = () => {
    if (!map.current) return;

    const geojsonData = {
      type: 'FeatureCollection',
      features: features
    };

    if (!map.current.getSource('risk-grid')) {
      map.current.addSource('risk-grid', {
        type: 'geojson',
        data: geojsonData
      });
    } else {
      map.current.getSource('risk-grid').setData(geojsonData);
    }

    if (!map.current.getLayer('risk-layer')) {
      map.current.addLayer({
        id: 'risk-layer',
        type: 'fill',
        source: 'risk-grid',
        paint: {
          'fill-color': [
            'interpolate', ['linear'],
            ['coalesce', ['to-number', ['get', 'risk_score'], 0], 0],
            0.0, '#22c55e',
            0.15, '#22c55e',
            0.30, '#eab308',
            0.50, '#f97316',
            0.70, '#ef4444'
          ],
          'fill-opacity': 0.65,
          'fill-outline-color': '#ffffff'
        }
      });

      map.current.on('click', 'risk-layer', (e) => {
        if (e.features && e.features[0]) {
          const props = e.features[0].properties;
          // Parse factors if it's a JSON string from maplibre
          if (typeof props.factors === 'string') {
            try { props.factors = JSON.parse(props.factors); } catch {}
          }
          setSelectedCell(props);
          setDetailOpen(true);
        }
      });

      map.current.on('mouseenter', 'risk-layer', () => {
        map.current.getCanvas().style.cursor = 'pointer';
      });
      map.current.on('mouseleave', 'risk-layer', () => {
        map.current.getCanvas().style.cursor = '';
      });
    }
  };

  // Initialize map
  useEffect(() => {
    if (map.current) return;

    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: MAP_STYLES.dark,
      center: DEMO_REGIONS.all_floods.center,
      zoom: DEMO_REGIONS.all_floods.zoom
    });

    map.current.addControl(new maplibregl.NavigationControl(), 'top-left');

    map.current.on('load', () => {
      map.current.resize();
      addRiskLayers();
    });
    map.current.on('styledata', addRiskLayers);

    // Auto-switch to zone view on zoom in
    map.current.on('zoomend', () => {
      const zoom = map.current.getZoom();
      if (zoom >= 9.5 && viewMode !== 'zones') {
        setViewMode('zones');
      } else if (zoom < 8.5 && viewMode !== 'districts') {
        setViewMode('districts');
      }
    });

    setTimeout(() => {
      if (map.current) map.current.resize();
    }, 500);
  }, []);

  // Update map data when features change
  useEffect(() => {
    if (map.current && map.current.getSource('risk-grid')) {
      map.current.getSource('risk-grid').setData({
        type: 'FeatureCollection',
        features: features
      });
    }
  }, [features]);

  const changeMapStyle = (styleKey) => {
    setCurrentStyle(styleKey);
    if (map.current) {
      map.current.setStyle(MAP_STYLES[styleKey]);
    }
  };

  const handleRegionChange = (key) => {
    setSelectedRegion(key);
    const reg = DEMO_REGIONS[key];
    if (reg && map.current) {
      map.current.flyTo({
        center: reg.center,
        zoom: reg.zoom,
        duration: 1800
      });
    }
  };

  // ── Search ─────────────────────────────────────────────────────
  const handleSearchInput = async (e) => {
    const val = e.target.value;
    setSearchQuery(val);

    if (val.trim().length < 2) {
      setSearchResults([]);
      return;
    }

    setIsSearching(true);
    try {
      const res = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(val)}&limit=5`);
      const data = await res.json();
      setSearchResults(data);
    } catch (err) {
      console.error('Location search failed:', err);
    } finally {
      setIsSearching(false);
    }
  };

  const handleSelectPlace = (place) => {
    const lon = parseFloat(place.lon);
    const lat = parseFloat(place.lat);
    const placeName = place.display_name.split(',')[0];

    if (map.current) {
      map.current.flyTo({
        center: [lon, lat],
        zoom: 10,
        duration: 2000
      });
    }

    setSearchQuery(placeName);
    setSearchResults([]);
  };

  // ── Risk Level Color Helper ────────────────────────────────────
  const getRiskColor = (level) => {
    switch (level) {
      case 'severe': return 'var(--risk-severe)';
      case 'high': return 'var(--risk-high)';
      case 'moderate': return 'var(--risk-moderate)';
      default: return 'var(--risk-low)';
    }
  };

  return (
    <div id="app">
      {/* Top Header / Stats Bar */}
      <header className="header">
        <div className="header__brand" style={{ cursor: 'pointer' }} onClick={() => navigate('/')} title="Return to Landing Page">
          <span className="header__logo">🌊</span>
          <div>
            <h1 className="header__title">PROJECT HYDRA</h1>
            <div className="header__subtitle">India Flood & Drought EWS</div>
          </div>
        </div>

        <div className="header__stats">
          <div className="stat-pill stat-pill--danger">
            <span>Severe:</span>
            <span className="stat-pill__value">{riskCounts.severe} Districts</span>
          </div>
          <div className="stat-pill stat-pill--warning">
            <span>High Risk:</span>
            <span className="stat-pill__value">{riskCounts.high} Districts</span>
          </div>
          <div className="stat-pill">
            <span>Moderate:</span>
            <span className="stat-pill__value">{riskCounts.moderate} Districts</span>
          </div>
          <div className="stat-pill stat-pill--safe">
            <span>Low:</span>
            <span className="stat-pill__value">{riskCounts.low} Districts</span>
          </div>
          <div className="stat-pill">
            <span>📅</span>
            <span className="stat-pill__value">{getCalendarDate(timelineDay, 'full')}</span>
          </div>
        </div>

        <div className="header__controls">
          {/* Location Search Bar */}
          <div className="search-container">
            <span className="search-icon">🔍</span>
            <input
              type="text"
              className="search-input"
              placeholder="Search any city or district..."
              value={searchQuery}
              onChange={handleSearchInput}
            />
            {searchResults.length > 0 && (
              <div className="search-results">
                {searchResults.map((place) => (
                  <div
                    key={place.place_id}
                    className="search-result-item"
                    onClick={() => handleSelectPlace(place)}
                  >
                    <span><strong>{place.display_name.split(',')[0]}</strong></span>
                    <span className="search-result-sub">{place.display_name}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Map Theme Toggle */}
          <select
            className="select-control"
            value={currentStyle}
            onChange={(e) => changeMapStyle(e.target.value)}
          >
            <option value="dark">🗺️ Dark Map</option>
            <option value="satellite">🛰️ Satellite</option>
          </select>

          {/* Region Selector */}
          <select
            className="select-control"
            value={selectedRegion}
            onChange={(e) => handleRegionChange(e.target.value)}
          >
            {Object.entries(DEMO_REGIONS).map(([key, item]) => (
              <option key={key} value={key}>{item.name}</option>
            ))}
          </select>

          {/* View Mode Toggle */}
          <select
            className="select-control"
            value={viewMode}
            onChange={(e) => setViewMode(e.target.value)}
          >
            <option value="districts">🏛️ District View</option>
            <option value="zones">🔬 5km Grid</option>
          </select>
        </div>
      </header>

      {/* Main Map Content Area */}
      <div className="main-content">
        {/* Weather API Degradation / Rate Limit Warning Toast Banner */}
        {weatherWarning && (
          <div style={{
            position: 'absolute', top: '16px', left: '50%', transform: 'translateX(-50%)',
            zIndex: 950, background: 'rgba(234, 179, 8, 0.18)', border: '1px solid rgba(234, 179, 8, 0.5)',
            backdropFilter: 'blur(16px)', borderRadius: '9999px', padding: '6px 18px',
            fontSize: '0.75rem', color: '#fef08a', display: 'flex', alignItems: 'center', gap: '8px',
            boxShadow: '0 8px 24px rgba(0,0,0,0.4)', fontWeight: 500
          }}>
            <span>⚠️</span>
            <span>{weatherWarning}</span>
            <button onClick={() => setWeatherWarning(null)} style={{ background: 'none', border: 'none', color: '#fef08a', cursor: 'pointer', marginLeft: '6px', fontSize: '0.9rem' }}>✕</button>
          </div>
        )}
        {/* Loading Overlay */}
        {isLoading && (
          <div style={{
            position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
            background: 'rgba(2, 6, 16, 0.7)', zIndex: 999,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            backdropFilter: 'blur(4px)'
          }}>
            <div style={{ textAlign: 'center', color: 'var(--text-accent)' }}>
              <div className="spinner" style={{ margin: '0 auto 12px' }}></div>
              <div style={{ fontSize: 'var(--text-sm)' }}>Loading predictions for {getCalendarDate(timelineDay, 'full')}...</div>
            </div>
          </div>
        )}

        {/* Floating Toggle Button */}
        <button className="sidebar-toggle" onClick={() => setSidebarOpen(!sidebarOpen)}>
          {sidebarOpen ? '✕' : '☰'}
        </button>

        {/* Map Container */}
        <div className="map-container">
          <div id="map" ref={mapContainer} />

          {/* Emerging Factors toggle button */}
          <button
            className="efp-toggle-btn"
            onClick={() => setEfpOpen(!efpOpen)}
            title="Emerging Factors Panel"
          >
            🔬 {efpOpen ? 'Close' : 'Factors'}
          </button>

          {/* Map Legend */}
          <div className="map-legend">
            <div className="map-legend__title">Risk Index Scale</div>
            <div className="map-legend__items">
              <div className="map-legend__item">
                <div className="map-legend__color" style={{ background: 'var(--risk-severe)' }}></div>
                <span>Severe (≥ 0.60)</span>
              </div>
              <div className="map-legend__item">
                <div className="map-legend__color" style={{ background: 'var(--risk-high)' }}></div>
                <span>High (0.35 – 0.60)</span>
              </div>
              <div className="map-legend__item">
                <div className="map-legend__color" style={{ background: 'var(--risk-moderate)' }}></div>
                <span>Moderate (0.15 – 0.35)</span>
              </div>
              <div className="map-legend__item">
                <div className="map-legend__color" style={{ background: 'var(--risk-low)' }}></div>
                <span>Low (&lt; 0.15)</span>
              </div>
            </div>
          </div>
        </div>

        {/* Emerging Factors Panel */}
        {(() => {
          const matchedFeature = features.find(f => f.properties.id === selectedCell?.id);
          const coords = matchedFeature?.geometry?.coordinates?.[0];
          let cLat = 26.15, cLon = 92.8;
          if (coords && coords.length > 0) {
            cLat = coords.reduce((s, c) => s + c[1], 0) / coords.length;
            cLon = coords.reduce((s, c) => s + c[0], 0) / coords.length;
          }
          return (
            <EmergingFactorsPanel
              lat={cLat}
              lon={cLon}
              locationName={selectedCell?.region || 'Selected Location'}
              isOpen={efpOpen}
              onClose={() => setEfpOpen(false)}
            />
          );
        })()}

        {/* Detail Panel (Slide-in Inspect Overlay) */}
        {selectedCell && (
          <div className={`detail-panel ${detailOpen ? 'detail-panel--open' : ''}`}>
            <div className="detail-panel__header">
              <h2 className="detail-panel__title">Risk Analysis</h2>
              <button className="detail-panel__close" onClick={() => setDetailOpen(false)}>✕</button>
            </div>

            <div className="detail-panel__body">
              <div className="risk-gauge">
                <div className="risk-gauge__circle" style={{
                  '--gauge-color': getRiskColor(selectedCell.risk_level),
                  '--gauge-pct': `${(selectedCell.risk_score || 0) * 100}%`
                }}>
                  <span className="risk-gauge__value">{selectedCell.risk_score}</span>
                </div>
                <div className="risk-gauge__label" style={{ color: getRiskColor(selectedCell.risk_level) }}>
                  {selectedCell.risk_level} Risk
                </div>
              </div>

              <div className="detail-section">
                <div className="detail-section__title">Target Region</div>
                <p style={{ fontWeight: 600 }}>{selectedCell.region}</p>
                <span className={`model-badge model-badge--${selectedCell.model_type || 'flood'}`} style={{ marginTop: '6px' }}>
                  {(selectedCell.model_type || 'flood')} Model
                </span>
              </div>

              {/* Forecast Date */}
              <div className="detail-section">
                <div className="detail-section__title">📅 Forecast Date</div>
                <p style={{ fontWeight: 600, color: 'var(--text-accent)', fontSize: 'var(--text-lg)' }}>
                  {getCalendarDate(timelineDay, 'full')}
                </p>
                <span style={{ color: 'var(--text-muted)', fontSize: 'var(--text-xs)' }}>
                  {getDayBadge(timelineDay)} · Day {timelineDay > 0 ? `+${timelineDay}` : timelineDay}
                </span>
              </div>

              <div className="detail-section">
                <div className="detail-section__title">Environmental Factors</div>
                {selectedCell.factors && Object.entries(typeof selectedCell.factors === 'string' ? JSON.parse(selectedCell.factors) : selectedCell.factors).map(([key, value]) => (
                  <div className="factor-row" key={key}>
                    <span className="factor-row__label">{key.replace(/_/g, ' ').toUpperCase()}</span>
                    <span className="factor-row__value">{value}</span>
                  </div>
                ))}
              </div>

              <div className="detail-section">
                <div className="detail-section__title">Alert Advisory</div>
                <div className="detail-alert-preview">
                  <div><strong>Predicted Event:</strong> {getCalendarDate(selectedCell.days_to_event || 0, 'full')}</div>
                  <div className="detail-alert-preview__action">
                    {selectedCell.alert_message}
                  </div>
                </div>
                <button className="btn-sms" onClick={() => alert(`Simulated SMS: "${selectedCell.alert_message}"`)}>
                  📲 Test Trigger SMS Alert
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Right Sidebar / Active Alert Cards Panel */}
        <aside className={`sidebar ${!sidebarOpen ? 'sidebar--collapsed' : ''}`}>
          <div className="sidebar__header">
            <div className="sidebar__title">
              Active Alerts <span className="sidebar__badge">{features.length} LIVE</span>
            </div>
            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', marginTop: '4px' }}>
              📅 {getCalendarDate(timelineDay, 'full')} · {getDayBadge(timelineDay)}
            </div>
          </div>

          <div className="sidebar__content">
            {features.filter(f => f.properties.risk_level === 'severe' || f.properties.risk_level === 'high').slice(0, 20).map((feature) => {
              const p = feature.properties;
              return (
                <div
                  key={p.id}
                  className={`alert-card alert-card--${p.risk_level}`}
                  onClick={() => {
                    setSelectedCell(p);
                    setDetailOpen(true);
                    const coords = feature.geometry?.coordinates?.[0]?.[0];
                    if (map.current && coords) {
                      map.current.flyTo({ center: coords, zoom: 9, duration: 1500 });
                    }
                  }}
                >
                  <div className="alert-card__header">
                    <span className={`alert-card__level alert-card__level--${p.risk_level}`}>
                      {p.risk_level}
                    </span>
                    <span className="alert-card__days">{getCalendarDate(timelineDay)}</span>
                  </div>
                  <div className="alert-card__region">{p.region}</div>
                  <div className="alert-card__action">{p.alert_message}</div>
                  <div className="alert-card__meta">
                    <span className="alert-card__score">SCORE: {p.risk_score}</span>
                    <span className="alert-card__type">{(p.model_type || 'FLOOD').toUpperCase()}</span>
                  </div>
                </div>
              );
            })}
            {features.filter(f => f.properties.risk_level === 'severe' || f.properties.risk_level === 'high').length === 0 && (
              <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)', fontSize: 'var(--text-sm)' }}>
                ✅ No severe or high risk alerts for {getCalendarDate(timelineDay, 'full')}
              </div>
            )}
          </div>
        </aside>
      </div>

      {/* Timeline Bar */}
      <div className="timeline-bar">
        <div className="timeline-bar__controls">
          <button className={`timeline-btn ${timelineDay === -7 ? 'timeline-btn--active' : ''}`} onClick={() => setTimelineDay(-7)}>-7d</button>
          <button className={`timeline-btn ${timelineDay === 0 ? 'timeline-btn--active' : ''}`} onClick={() => setTimelineDay(0)}>Today</button>
          <button className={`timeline-btn ${timelineDay === 7 ? 'timeline-btn--active' : ''}`} onClick={() => setTimelineDay(7)}>+7d</button>
          <button className={`timeline-btn ${timelineDay === 15 ? 'timeline-btn--active' : ''}`} onClick={() => setTimelineDay(15)}>+15d</button>
        </div>

        <div className="timeline-bar__slider-wrap">
          <input
            type="range"
            min="-7"
            max="15"
            value={timelineDay}
            onChange={(e) => setTimelineDay(Number(e.target.value))}
            className="timeline-slider"
          />
          <div className="timeline-bar__dates">
            <span className="timeline-bar__date-label">{getCalendarDate(-7)} (Past)</span>
            <span className="timeline-bar__date-label timeline-bar__date-label--today">Today · {getCalendarDate(0)}</span>
            <span className="timeline-bar__date-label timeline-bar__date-label--forecast">{getCalendarDate(15)} (Forecast)</span>
          </div>
        </div>

        <div className="timeline-bar__current-date">
          📅 {getCalendarDate(timelineDay, 'full')} · Day {timelineDay > 0 ? `+${timelineDay}` : timelineDay}
          <span className="timeline-bar__forecast-badge">{getDayBadge(timelineDay)}</span>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/welcome" element={<LandingPage />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}