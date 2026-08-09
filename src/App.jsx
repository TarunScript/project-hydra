import React, { useEffect, useRef, useState } from 'react';
import * as maplibregl from 'maplibre-gl';

// API Backend URL — connects to the Flask server serving real XGBoost predictions
const API_BASE = 'http://localhost:5001/api';

// Features will be loaded from the API
const INITIAL_FEATURES = [];


const DEMO_REGIONS = {
  all_floods: {
    name: 'All Flood States (Unified 4-State Model)',
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

export default function App() {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const [features, setFeatures] = useState(INITIAL_FEATURES);
  const [selectedCell, setSelectedCell] = useState(null);
  const [timelineDay, setTimelineDay] = useState(0);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [detailOpen, setDetailOpen] = useState(false);
  const [currentStyle, setCurrentStyle] = useState('dark');
  const [selectedRegion, setSelectedRegion] = useState('all_floods');
  const [isLoading, setIsLoading] = useState(true);
  const [viewMode, setViewMode] = useState('districts'); // start with districts; switch to zones on demand

  // Search state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);

  const timelineDayRef = useRef(timelineDay);
  timelineDayRef.current = timelineDay;

  // Fetch real risk data from the Flask API
  const fetchRiskData = async (region = selectedRegion, month = null, mode = viewMode, day = timelineDay) => {
    try {
      setIsLoading(true);
      const monthParam = month !== null ? `month=${month}` : 'month=7';
      const dayParam = `day=${day}`;
      const endpointRegion = ['all_floods', 'assam', 'bihar', 'west_bengal', 'odisha'].includes(region) ? 'flood' : region;

      let url;
      if (mode === 'zones') {
        // Get current map bbox for zone filtering
        let bboxParams = '';
        if (map.current) {
          const bounds = map.current.getBounds();
          bboxParams = `&minlon=${bounds.getWest().toFixed(4)}&maxlon=${bounds.getEast().toFixed(4)}&minlat=${bounds.getSouth().toFixed(4)}&maxlat=${bounds.getNorth().toFixed(4)}`;
        }
        url = `${API_BASE}/risk-grid/${endpointRegion}/cells?${monthParam}&${dayParam}&min_risk=0.0&max_cells=3000${bboxParams}`;
      } else {
        // District-level polygons
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

      if (data.features && data.features.length > 0) {
        // Compute relative_score (0-1) for colour ramp — avoids all-red when range is narrow
        const rawScores = data.features.map(f => f.properties.risk_score || 0);
        const minS = Math.min(...rawScores);
        const maxS = Math.max(...rawScores);
        const range = maxS - minS || 0.001;
        const normalised = data.features.map(f => ({
          ...f,
          properties: {
            ...f.properties,
            relative_score: parseFloat(((f.properties.risk_score - minS) / range).toFixed(3)),
          }
        }));
        setFeatures(normalised);
        setSelectedCell(normalised[0].properties);
        setDetailOpen(true);
        console.log(`✓ Loaded ${normalised.length} ${mode} (day=${day}) | score range [${minS.toFixed(3)}, ${maxS.toFixed(3)}]`);
      }
    } catch (err) {
      console.error('Failed to fetch risk data from API:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Re-fetch when region, viewMode, or timelineDay changes
  useEffect(() => {
    fetchRiskData(selectedRegion, null, viewMode, timelineDay);
  }, [selectedRegion, viewMode, timelineDay]);

  // Compute dynamic color stops from loaded feature scores for relative coloring
  const getColorStops = (feats) => {
    if (!feats || feats.length === 0) return { low: 0.0, mid: 0.5, high: 0.9 };
    const scores = feats.map(f => f.properties.risk_score || 0).sort((a, b) => a - b);
    return {
      low: scores[Math.floor(scores.length * 0.1)],
      mid: scores[Math.floor(scores.length * 0.5)],
      high: scores[Math.floor(scores.length * 0.9)],
    };
  };

  const featuresRef = useRef(features);
  const currentStyleRef = useRef(currentStyle);
  featuresRef.current = features;
  currentStyleRef.current = currentStyle;

  const addRiskLayers = () => {
    if (!map.current) return;

    const geojsonData = {
      type: 'FeatureCollection',
      features: featuresRef.current || []
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
      // Find label layer to place risk-layer UNDER labels so place names and markings stay visible
      const beforeId = map.current.getLayer('esri-labels-layer') ? 'esri-labels-layer' :
                       (map.current.getLayer('carto-labels-layer') ? 'carto-labels-layer' : undefined);

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
          'fill-opacity': currentStyleRef.current === 'satellite' ? 0.45 : 0.65,
          'fill-outline-color': currentStyleRef.current === 'satellite' ? 'rgba(255,255,255,0.7)' : 'rgba(255,255,255,0.2)'
        }
      }, beforeId);

      map.current.on('click', 'risk-layer', (e) => {
        if (e.features && e.features[0]) {
          setSelectedCell(e.features[0].properties);
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

  const viewModeRef = useRef(viewMode);
  const selectedRegionRef = useRef(selectedRegion);
  viewModeRef.current = viewMode;
  selectedRegionRef.current = selectedRegion;

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

    // Auto re-fetch 5km zones when user finishes panning or zooming the map
    map.current.on('moveend', () => {
      if (viewModeRef.current === 'zones') {
        fetchRiskData(selectedRegionRef.current, null, 'zones', timelineDayRef.current);
      }
    });

    setTimeout(() => {
      if (map.current) map.current.resize();
    }, 500);
  }, []);

  useEffect(() => {
    if (map.current && map.current.getSource('risk-grid')) {
      map.current.getSource('risk-grid').setData({
        type: 'FeatureCollection',
        features: features
      });

      // color is driven by relative_score (0-1), already embedded in feature properties
    }
  }, [features]);

  const changeMapStyle = (styleKey) => {
    setCurrentStyle(styleKey);
    currentStyleRef.current = styleKey;
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

  const handleSelectPlace = async (place) => {
    const lon = parseFloat(place.lon);
    const lat = parseFloat(place.lat);
    const placeName = place.display_name.split(',')[0];

    setSearchQuery(placeName);
    setSearchResults([]);

    if (map.current) {
      map.current.flyTo({ center: [lon, lat], zoom: 10, duration: 1500 });
    }

    // Fetch real model zones within ~0.6 degrees around the searched place
    const pad = 0.6;
    try {
      setIsLoading(true);
      const url = `${API_BASE}/risk-grid/flood/cells?month=7&min_risk=0.0&max_cells=3000` +
        `&minlon=${(lon - pad).toFixed(4)}&maxlon=${(lon + pad).toFixed(4)}` +
        `&minlat=${(lat - pad).toFixed(4)}&maxlat=${(lat + pad).toFixed(4)}`;
      const res = await fetch(url);
      const data = await res.json();
      if (data.features && data.features.length > 0) {
        setFeatures(data.features);
        setSelectedCell(data.features[0].properties);
        setDetailOpen(true);
        setViewMode('zones');
      } else {
        setSelectedCell({
          id: `search-${Date.now()}`, region: `${placeName} Region`,
          model_type: 'flood', risk_score: null, risk_level: 'moderate',
          days_to_event: null,
          alert_message: `No flood model data for ${placeName}. Model covers Assam (2015-2023).`,
          factors: {},
        });
        setDetailOpen(true);
      }
    } catch (err) {
      console.error('Zone fetch failed:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div id="app">
      {/* Top Header / Stats Bar */}
      <header className="header">
        <div className="header__brand">
          <span className="header__logo">🌊</span>
          <div>
            <h1 className="header__title">PROJECT HYDRA</h1>
            <div className="header__subtitle">India Flood & Drought EWS</div>
          </div>
        </div>

        <div className="header__stats">
          <div className="stat-pill stat-pill--danger">
            <span>Severe Risks:</span>
            <span className="stat-pill__value">{
              features.filter(f => (!selectedRegion || selectedRegion === 'all_floods' || f.properties.state === selectedRegion || (f.properties.region && f.properties.region.toLowerCase().includes(selectedRegion.replace('_', ' ')))) && f.properties.risk_level === 'severe').length
            } {viewMode === 'zones' ? 'Zones' : 'Districts'}</span>
          </div>
          <div className="stat-pill stat-pill--warning">
            <span>High Risk:</span>
            <span className="stat-pill__value">{
              features.filter(f => (!selectedRegion || selectedRegion === 'all_floods' || f.properties.state === selectedRegion || (f.properties.region && f.properties.region.toLowerCase().includes(selectedRegion.replace('_', ' ')))) && f.properties.risk_level === 'high').length
            } {viewMode === 'zones' ? 'Zones' : 'Districts'}</span>
          </div>
          <div className="stat-pill">
            <span>Active Region:</span>
            <span className="stat-pill__value">{DEMO_REGIONS[selectedRegion]?.name.split(' ')[0]}</span>
          </div>
          {isLoading && (
            <div className="stat-pill">
              <span>⏳ Loading model data...</span>
            </div>
          )}
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
            <option value="dark">🗺️ Dark Map (Labeled)</option>
            <option value="satellite">🛰️ Satellite View (ESRI)</option>
          </select>

          {/* Preset Demo Regions */}
          <select 
            className="select-control"
            value={selectedRegion}
            onChange={(e) => handleRegionChange(e.target.value)}
          >
            {Object.entries(DEMO_REGIONS).map(([key, item]) => (
              <option key={key} value={key}>{item.name}</option>
            ))}
          </select>

          {/* Zone vs District toggle */}
          <div style={{ display: 'flex', gap: '4px', background: 'rgba(255,255,255,0.07)', borderRadius: '8px', padding: '3px' }}>
            <button
              onClick={() => setViewMode('zones')}
              style={{
                padding: '5px 12px', borderRadius: '6px', border: 'none', cursor: 'pointer', fontSize: '12px', fontWeight: 600,
                background: viewMode === 'zones' ? 'var(--accent)' : 'transparent',
                color: viewMode === 'zones' ? '#fff' : 'var(--text-muted)',
                transition: 'all 0.2s',
              }}
            >🗺️ 5km Zones</button>
            <button
              onClick={() => setViewMode('districts')}
              style={{
                padding: '5px 12px', borderRadius: '6px', border: 'none', cursor: 'pointer', fontSize: '12px', fontWeight: 600,
                background: viewMode === 'districts' ? 'var(--accent)' : 'transparent',
                color: viewMode === 'districts' ? '#fff' : 'var(--text-muted)',
                transition: 'all 0.2s',
              }}
            >📍 Districts</button>
          </div>
        </div>
      </header>

      {/* Main Map Content Area */}
      <div className="main-content">
        {/* Floating Toggle Button */}
        <button className="sidebar-toggle" onClick={() => setSidebarOpen(!sidebarOpen)}>
          {sidebarOpen ? '✕' : '☰'}
        </button>

        {/* Map Container */}
        <div className="map-container">
          <div id="map" ref={mapContainer} />

          {/* Map Legend */}
          <div className="map-legend">
            <div className="map-legend__title">Risk Index Scale</div>
            <div className="map-legend__items">
              <div className="map-legend__item">
                <div className="map-legend__color" style={{ background: 'var(--risk-severe)' }}></div>
                <span>Severe (0.8 - 1.0)</span>
              </div>
              <div className="map-legend__item">
                <div className="map-legend__color" style={{ background: 'var(--risk-high)' }}></div>
                <span>High (0.6 - 0.8)</span>
              </div>
              <div className="map-legend__item">
                <div className="map-legend__color" style={{ background: 'var(--risk-moderate)' }}></div>
                <span>Moderate (0.4 - 0.6)</span>
              </div>
              <div className="map-legend__item">
                <div className="map-legend__color" style={{ background: 'var(--risk-low)' }}></div>
                <span>Low (&lt; 0.4)</span>
              </div>
            </div>
          </div>
        </div>

        {/* Detail Panel (Slide-in Inspect Overlay) */}
        {selectedCell && (
          <div className={`detail-panel ${detailOpen ? 'detail-panel--open' : ''}`}>
            <div className="detail-panel__header">
              <h2 className="detail-panel__title">Cell Risk Analysis</h2>
              <button className="detail-panel__close" onClick={() => setDetailOpen(false)}>✕</button>
            </div>

            <div className="detail-panel__body">
              <div className="risk-gauge">
                <div className="risk-gauge__circle" style={{
                  '--gauge-color': selectedCell.risk_level === 'severe' ? 'var(--risk-severe)' : 'var(--risk-high)',
                  '--gauge-pct': `${selectedCell.risk_score * 100}%`
                }}>
                  <span className="risk-gauge__value">{selectedCell.risk_score}</span>
                </div>
                <div className="risk-gauge__label" style={{ color: selectedCell.risk_level === 'severe' ? 'var(--risk-severe)' : 'var(--risk-high)' }}>
                  {selectedCell.risk_level} Risk
                </div>
              </div>

              <div className="detail-section">
                <div className="detail-section__title">Target Region</div>
                <p style={{ fontWeight: 600 }}>{selectedCell.region}</p>
                <span className={`model-badge model-badge--${selectedCell.model_type}`} style={{ marginTop: '6px' }}>
                  {selectedCell.model_type} Model
                </span>
              </div>

              <div className="detail-section">
                <div className="detail-section__title">Environmental Factors</div>
                {selectedCell.factors && Object.entries(typeof selectedCell.factors === 'string' ? JSON.parse(selectedCell.factors) : selectedCell.factors).map(([key, value]) => (
                  <div className="factor-row" key={key}>
                    <span className="factor-row__label">{key.replace('_', ' ').toUpperCase()}</span>
                    <span className="factor-row__value">{value}</span>
                  </div>
                ))}
              </div>

              <div className="detail-section">
                <div className="detail-section__title">Simulated Alert Advisory</div>
                <div className="detail-alert-preview">
                  <div><strong>Days to Event:</strong> {selectedCell.days_to_event} Days Out</div>
                  <div className="detail-alert-preview__action">
                    {selectedCell.alert_message}
                  </div>
                </div>
                <button className="btn-sms" onClick={() => alert(`Simulated SMS Sent to test numbers: "${selectedCell.alert_message}"`)}>
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
              Active Alerts <span className="sidebar__badge">{
                features.filter(f => !selectedRegion || selectedRegion === 'all_floods' || f.properties.state === selectedRegion || (f.properties.region && f.properties.region.toLowerCase().includes(selectedRegion.replace('_', ' ')))).length
              } LIVE</span>
            </div>
          </div>

          <div className="sidebar__content">
            {features
              .filter(f => !selectedRegion || selectedRegion === 'all_floods' || f.properties.state === selectedRegion || (f.properties.region && f.properties.region.toLowerCase().includes(selectedRegion.replace('_', ' '))))
              .map((feature) => {
                const p = feature.properties;
                return (
                  <div
                    key={p.id}
                    className={`alert-card alert-card--${p.risk_level}`}
                    onClick={() => {
                      setSelectedCell(p);
                      setDetailOpen(true);
                      // Fly to feature center
                      const coords = feature.geometry.coordinates[0][0];
                      if (map.current) {
                        map.current.flyTo({ center: coords, zoom: 9, duration: 1500 });
                      }
                    }}
                  >
                    <div className="alert-card__header">
                      <span className={`alert-card__level alert-card__level--${p.risk_level}`}>
                        {p.risk_level}
                      </span>
                      <span className="alert-card__days">{p.days_to_event}d forecast</span>
                    </div>
                    <div className="alert-card__region">{p.region}</div>
                    <div className="alert-card__action">{p.alert_message}</div>
                    <div className="alert-card__meta">
                      <span className="alert-card__score">SCORE: {p.risk_score}</span>
                      <span className="alert-card__type">{p.model_type ? p.model_type.toUpperCase() : 'FLOOD'}</span>
                    </div>
                  </div>
                );
              })}
          </div>
        </aside>
      </div>

      {/* Timeline Bar (Section 8) */}
      <div className="timeline-bar">
        <div className="timeline-bar__controls">
          <button className="timeline-btn" onClick={() => setTimelineDay(0)}>Today</button>
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
            <span className="timeline-bar__date-label">-7 Days (Past)</span>
            <span className="timeline-bar__date-label timeline-bar__date-label--today">Today (Day 0)</span>
            <span className="timeline-bar__date-label timeline-bar__date-label--forecast">+15 Days (Forecast)</span>
          </div>
        </div>

        <div className="timeline-bar__current-date">
          Day Selected: {timelineDay > 0 ? `+${timelineDay}` : timelineDay}
          {timelineDay > 0 && <span className="timeline-bar__forecast-badge">Forecast</span>}
        </div>
      </div>
    </div>
  );
}