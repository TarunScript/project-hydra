import React, { useEffect, useRef, useState } from 'react';
import * as maplibregl from 'maplibre-gl';
import EmergingFactorsPanel from './EmergingFactorsPanel.jsx';
import './EmergingFactorsPanel.css';

// Mock GeoJSON data matching design system's risk levels
const INITIAL_FEATURES = [
  {
    type: 'Feature',
    geometry: {
      type: 'Polygon',
      coordinates: [[[92.5, 26.0], [92.8, 26.0], [92.8, 26.3], [92.5, 26.3], [92.5, 26.0]]]
    },
    properties: {
      id: 'cell-1',
      region: 'Assam - Brahamputra Basin #1',
      model_type: 'flood',
      risk_score: 0.88,
      risk_level: 'severe',
      days_to_event: 2,
      alert_message: 'Severe flood risk predicted in 48 hours due to high upstream discharge. Evacuate low-lying areas.',
      factors: {
        rainfall_7d: '185 mm',
        soil_moisture: '82%',
        discharge: '4,200 m³/s'
      }
    }
  },
  {
    type: 'Feature',
    geometry: {
      type: 'Polygon',
      coordinates: [[[92.8, 26.0], [93.1, 26.0], [93.1, 26.3], [92.8, 26.3], [92.8, 26.0]]]
    },
    properties: {
      id: 'cell-2',
      region: 'Assam - Brahamputra Basin #2',
      model_type: 'flood',
      risk_score: 0.64,
      risk_level: 'high',
      days_to_event: 5,
      alert_message: 'High flood advisory. Prepare water storage and move assets to higher ground.',
      factors: {
        rainfall_7d: '110 mm',
        soil_moisture: '68%',
        discharge: '2,900 m³/s'
      }
    }
  },
  {
    type: 'Feature',
    geometry: {
      type: 'Polygon',
      coordinates: [[[76.3, 19.0], [76.7, 19.0], [76.7, 19.3], [76.3, 19.3], [76.3, 19.0]]]
    },
    properties: {
      id: 'cell-3',
      region: 'Marathwada - Latur Drought Zone',
      model_type: 'drought',
      risk_score: 0.78,
      risk_level: 'high',
      days_to_event: 12,
      alert_message: 'Severe water deficit & crop stress forecasted in next 12 days. Trigger irrigation allocation.',
      factors: {
        rainfall_deficit: '-58%',
        soil_moisture: '21%',
        temp_anomaly: '+3.2°C'
      }
    }
  }
];

const DEMO_REGIONS = {
  assam: {
    name: 'Assam (Flood Case)',
    center: [92.8, 26.15],
    zoom: 8.5
  },
  marathwada: {
    name: 'Marathwada (Drought Case)',
    center: [76.5, 19.15],
    zoom: 8.5
  },
  bihar: {
    name: 'Bihar (Kosi Flood Zone)',
    center: [86.8, 25.8],
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
  const [selectedCell, setSelectedCell] = useState(INITIAL_FEATURES[0].properties);
  const [timelineDay, setTimelineDay] = useState(0);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [detailOpen, setDetailOpen] = useState(true);
  const [currentStyle, setCurrentStyle] = useState('dark');
  const [selectedRegion, setSelectedRegion] = useState('assam');

  // Emerging Factors panel state
  const [efpOpen, setEfpOpen] = useState(false);

  // Search state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);

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
            'match',
            ['get', 'risk_level'],
            'severe', '#ef4444',
            'high', '#f97316',
            'moderate', '#eab308',
            '#22c55e'
          ],
          'fill-opacity': 0.65,
          'fill-outline-color': '#ffffff'
        }
      });

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

  useEffect(() => {
    if (map.current) return;

    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: MAP_STYLES.dark,
      center: DEMO_REGIONS.assam.center,
      zoom: DEMO_REGIONS.assam.zoom
    });

    map.current.addControl(new maplibregl.NavigationControl(), 'top-left');

    map.current.on('load', () => {
      map.current.resize();
      addRiskLayers();
    });
    map.current.on('styledata', addRiskLayers);

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

    // Dynamically create an interactive risk polygon for searched city
    const newFeature = {
      type: 'Feature',
      geometry: {
        type: 'Polygon',
        coordinates: [[
          [lon - 0.12, lat - 0.12],
          [lon + 0.12, lat - 0.12],
          [lon + 0.12, lat + 0.12],
          [lon - 0.12, lat + 0.12],
          [lon - 0.12, lat - 0.12]
        ]]
      },
      properties: {
        id: `search-${Date.now()}`,
        region: `${placeName} Region Risk Cell`,
        model_type: 'flood',
        risk_score: 0.74,
        risk_level: 'high',
        days_to_event: 4,
        alert_message: `Moderate-High hydrological risk detected for ${placeName}. Live sensors connected.`,
        factors: {
          rainfall_7d: '135 mm',
          soil_moisture: '74%',
          discharge: '3,100 m³/s'
        }
      }
    };

    setFeatures((prev) => [newFeature, ...prev]);
    setSelectedCell(newFeature.properties);
    setDetailOpen(true);
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
            <span className="stat-pill__value">1 Grid</span>
          </div>
          <div className="stat-pill stat-pill--warning">
            <span>High Risk:</span>
            <span className="stat-pill__value">2 Grids</span>
          </div>
          <div className="stat-pill">
            <span>Active Region:</span>
            <span className="stat-pill__value">{DEMO_REGIONS[selectedRegion]?.name.split(' ')[0]}</span>
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

        {/* Emerging Factors Panel */}
        {(() => {
          // Compute centroid from the first matching feature
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
              Active Alerts <span className="sidebar__badge">{features.length} LIVE</span>
            </div>
          </div>

          <div className="sidebar__content">
            {features.map((feature) => {
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
                    <span className="alert-card__type">{p.model_type.toUpperCase()}</span>
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