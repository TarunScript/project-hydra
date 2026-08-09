import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as maplibregl from 'maplibre-gl';

// Available real drought GeoJSON dates
const AVAILABLE_DATES = [
  { value: '2023-08-01', label: 'Aug 2023 (Monsoon)' },
  { value: '2023-11-01', label: 'Nov 2023 (Post-Monsoon)' },
  { value: '2024-05-01', label: 'May 2024 (Pre-Monsoon)' },
  { value: '2024-08-01', label: 'Aug 2024 (Monsoon)' },
  { value: '2024-11-01', label: 'Nov 2024 (Post-Monsoon)' },
];

const DEMO_REGIONS = {
  marathwada: {
    name: 'Marathwada (Drought Zone)',
    center: [76.0, 19.0],
    zoom: 7.5
  },
  india: {
    name: 'India Overview',
    center: [78.9, 22.5],
    zoom: 4.5
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
          'https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png'
        ],
        tileSize: 256,
        attribution: '&copy; OpenStreetMap &copy; CARTO'
      }
    },
    layers: [
      {
        id: 'carto-dark-layer',
        type: 'raster',
        source: 'carto-dark',
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
      }
    },
    layers: [
      {
        id: 'esri-satellite-layer',
        type: 'raster',
        source: 'esri-satellite',
        minzoom: 0,
        maxzoom: 19
      }
    ]
  }
};

// Risk color interpolation
function getRiskColor(score) {
  if (score <= 0.1) return '#1a9641';
  if (score <= 0.2) return '#66bd63';
  if (score <= 0.3) return '#a6d96a';
  if (score <= 0.4) return '#d9ef8b';
  if (score <= 0.5) return '#fee08b';
  if (score <= 0.6) return '#fdae61';
  if (score <= 0.7) return '#f46d43';
  if (score <= 0.8) return '#d73027';
  return '#a50026';
}

function getRiskLevel(score) {
  if (score <= 0.2) return 'low';
  if (score <= 0.4) return 'moderate';
  if (score <= 0.6) return 'high';
  if (score <= 0.8) return 'severe';
  return 'extreme';
}

function getAlertMessage(score, props) {
  if (score <= 0.2) return 'No significant drought stress detected. Normal conditions.';
  if (score <= 0.4) return `Moderate drought stress. Rainfall deficit: ${props.rain_deficit_30d_mm?.toFixed(0) || '?'}mm. Monitor crop water needs.`;
  if (score <= 0.6) return `High drought risk. NDVI anomaly: ${props.ndvi_anomaly?.toFixed(2) || '?'}. Activate irrigation reserves.`;
  if (score <= 0.8) return `Severe drought conditions. Soil moisture critically low (${(props.soil_moisture_rootzone * 100)?.toFixed(0) || '?'}%). Trigger water rationing.`;
  return `EXTREME drought emergency. ${props.dry_spell_days?.toFixed(0) || '?'} consecutive dry days. Immediate relief deployment required.`;
}

export default function App() {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const [selectedCell, setSelectedCell] = useState(null);
  const [detailOpen, setDetailOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [currentStyle, setCurrentStyle] = useState('dark');
  const [selectedRegion, setSelectedRegion] = useState('marathwada');
  const [selectedDate, setSelectedDate] = useState('2024-11-01');
  const [loading, setLoading] = useState(false);
  const [alertCells, setAlertCells] = useState([]);
  const [stats, setStats] = useState(null);
  const [geojsonData, setGeojsonData] = useState(null);

  // Load GeoJSON for selected date
  const loadDroughtData = useCallback(async (date) => {
    setLoading(true);
    try {
      const url = `/geojson/drought_risk_marathwada_${date}.geojson`;
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Failed to load ${url}`);
      const data = await response.json();

      // Compute stats
      const features = data.features;
      const scores = features.map(f => f.properties.risk_score);
      const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
      const maxScore = Math.max(...scores);
      const levels = { low: 0, moderate: 0, high: 0, severe: 0, extreme: 0 };
      features.forEach(f => {
        const lvl = getRiskLevel(f.properties.risk_score);
        levels[lvl]++;
      });

      setStats({
        totalCells: features.length,
        meanRisk: mean.toFixed(3),
        maxRisk: maxScore.toFixed(3),
        levels
      });

      // Extract alert-worthy cells (risk > 0.15)
      const alerts = features
        .filter(f => f.properties.risk_score > 0.15)
        .sort((a, b) => b.properties.risk_score - a.properties.risk_score)
        .slice(0, 20)
        .map((f, i) => {
          const p = f.properties;
          return {
            ...f,
            properties: {
              ...p,
              id: `alert-${i}`,
              risk_level: getRiskLevel(p.risk_score),
              model_type: 'drought',
              region: `${p.cell_id?.split('_')[0] || 'Marathwada'} (${p.lat?.toFixed(2)}, ${p.lon?.toFixed(2)})`,
              alert_message: getAlertMessage(p.risk_score, p),
              days_to_event: p.projection_7d_risk > p.risk_score ? 3 : 7,
              factors: {
                'Rain Deficit (30d)': `${p.rain_deficit_30d_mm?.toFixed(1) || '?'} mm`,
                'NDVI Anomaly': p.ndvi_anomaly?.toFixed(3) || '?',
                'Soil Moisture': `${(p.soil_moisture_rootzone * 100)?.toFixed(1) || '?'}%`,
                'Dry Spell': `${p.dry_spell_days?.toFixed(0) || '?'} days`,
                'Risk +7d': p.projection_7d_risk?.toFixed(3) || '?',
                'Risk +15d': p.projection_15d_risk?.toFixed(3) || '?',
                'Atlas SPI': p.atlas_spi_score?.toFixed(2) || 'N/A',
                'Atlas Category': p.atlas_drought_category || 'N/A'
              }
            }
          };
        });

      setAlertCells(alerts);
      setGeojsonData(data);
      setLoading(false);
      return data;
    } catch (err) {
      console.error('Failed to load drought data:', err);
      setLoading(false);
      return null;
    }
  }, []);

  // Initialize map
  useEffect(() => {
    if (map.current) return;

    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: MAP_STYLES[currentStyle],
      center: DEMO_REGIONS.marathwada.center,
      zoom: DEMO_REGIONS.marathwada.zoom,
      minZoom: 3,
      maxZoom: 15
    });

    map.current.addControl(new maplibregl.NavigationControl(), 'bottom-right');

    map.current.on('load', async () => {
      const data = await loadDroughtData(selectedDate);
      if (data) {
        addDroughtLayer(data);
      }
    });
  }, []);

  // Add drought choropleth layer to map
  const addDroughtLayer = useCallback((data) => {
    if (!map.current) return;

    // Remove old layers/sources
    if (map.current.getLayer('drought-fill')) map.current.removeLayer('drought-fill');
    if (map.current.getLayer('drought-outline')) map.current.removeLayer('drought-outline');
    if (map.current.getLayer('drought-highlight')) map.current.removeLayer('drought-highlight');
    if (map.current.getSource('drought-data')) map.current.removeSource('drought-data');

    map.current.addSource('drought-data', {
      type: 'geojson',
      data: data
    });

    // Filled choropleth
    map.current.addLayer({
      id: 'drought-fill',
      type: 'fill',
      source: 'drought-data',
      paint: {
        'fill-color': [
          'interpolate',
          ['linear'],
          ['get', 'risk_score'],
          0.0, '#1a9641',
          0.1, '#66bd63',
          0.2, '#a6d96a',
          0.3, '#d9ef8b',
          0.4, '#fee08b',
          0.5, '#fdae61',
          0.6, '#f46d43',
          0.7, '#d73027',
          0.8, '#a50026',
          1.0, '#67001f'
        ],
        'fill-opacity': 0.75
      }
    });

    // Cell outlines
    map.current.addLayer({
      id: 'drought-outline',
      type: 'line',
      source: 'drought-data',
      paint: {
        'line-color': '#ffffff',
        'line-width': 0.2,
        'line-opacity': 0.3
      }
    });

    // Hover highlight
    map.current.addLayer({
      id: 'drought-highlight',
      type: 'line',
      source: 'drought-data',
      paint: {
        'line-color': '#00ffff',
        'line-width': 2,
        'line-opacity': 0.8
      },
      filter: ['==', 'cell_id', '']
    });

    // Click handler
    map.current.on('click', 'drought-fill', (e) => {
      if (e.features.length === 0) return;
      const p = e.features[0].properties;
      setSelectedCell({
        ...p,
        risk_level: getRiskLevel(p.risk_score),
        model_type: 'drought',
        region: `Marathwada (${p.lat?.toFixed(2)}, ${p.lon?.toFixed(2)})`,
        alert_message: getAlertMessage(p.risk_score, p),
        days_to_event: p.projection_7d_risk > p.risk_score ? 3 : 7,
        factors: JSON.stringify({
          'Rain Deficit (30d)': `${p.rain_deficit_30d_mm?.toFixed(1) || '?'} mm`,
          'NDVI Anomaly': `${p.ndvi_anomaly?.toFixed(3) || '?'}`,
          'Soil Moisture': `${(p.soil_moisture_rootzone * 100)?.toFixed(1) || '?'}%`,
          'Dry Spell': `${p.dry_spell_days || '?'} days`,
          'Risk +7d': `${p.projection_7d_risk?.toFixed(3) || '?'}`,
          'Risk +15d': `${p.projection_15d_risk?.toFixed(3) || '?'}`,
          'Atlas SPI': `${p.atlas_spi_score || 'N/A'}`,
          'Atlas Category': `${p.atlas_drought_category || 'N/A'}`
        })
      });
      setDetailOpen(true);

      // Highlight cell
      map.current.setFilter('drought-highlight', ['==', 'cell_id', p.cell_id]);
    });

    // Hover cursor
    map.current.on('mouseenter', 'drought-fill', () => {
      map.current.getCanvas().style.cursor = 'pointer';
    });
    map.current.on('mouseleave', 'drought-fill', () => {
      map.current.getCanvas().style.cursor = '';
      map.current.setFilter('drought-highlight', ['==', 'cell_id', '']);
    });

    // Tooltip on hover
    const popup = new maplibregl.Popup({
      closeButton: false,
      closeOnClick: false,
      className: 'risk-popup'
    });

    map.current.on('mousemove', 'drought-fill', (e) => {
      if (e.features.length === 0) return;
      const p = e.features[0].properties;
      const html = `
        <div style="font-family: 'Inter', sans-serif; font-size: 12px; line-height: 1.4;">
          <div style="font-weight: 700; color: ${getRiskColor(p.risk_score)}; font-size: 14px;">
            Risk: ${p.risk_score?.toFixed(3)} (${p.risk_level})
          </div>
          <div style="margin-top: 4px; color: #ccc;">
            Rain Deficit: ${p.rain_deficit_30d_mm?.toFixed(1)} mm<br/>
            NDVI: ${p.ndvi_anomaly?.toFixed(3)}<br/>
            Soil: ${(p.soil_moisture_rootzone * 100)?.toFixed(1)}%
          </div>
        </div>
      `;
      popup.setLngLat(e.lngLat).setHTML(html).addTo(map.current);
    });

    map.current.on('mouseleave', 'drought-fill', () => {
      popup.remove();
    });
  }, []);

  // Handle date change
  const handleDateChange = async (date) => {
    setSelectedDate(date);
    const data = await loadDroughtData(date);
    if (data && map.current && map.current.getSource('drought-data')) {
      map.current.getSource('drought-data').setData(data);
    } else if (data) {
      addDroughtLayer(data);
    }
  };

  // Handle region change
  const handleRegionChange = (region) => {
    setSelectedRegion(region);
    const r = DEMO_REGIONS[region];
    if (map.current && r) {
      map.current.flyTo({ center: r.center, zoom: r.zoom, duration: 1500 });
    }
  };

  // Handle map style change
  const changeMapStyle = (style) => {
    setCurrentStyle(style);
    if (map.current) {
      map.current.setStyle(MAP_STYLES[style]);
      map.current.once('style.load', async () => {
        const data = await loadDroughtData(selectedDate);
        if (data) addDroughtLayer(data);
      });
    }
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header__brand">
          <div className="header__logo">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2L2 7l10 5 10-5-10-5z"/>
              <path d="M2 17l10 5 10-5"/>
              <path d="M2 12l10 5 10-5"/>
            </svg>
          </div>
          <div>
            <div className="header__title">Project HYDRA</div>
            <div className="header__subtitle">Drought Early Warning System</div>
          </div>
        </div>

        <div className="header__controls">
          {/* Date Selector */}
          <select
            className="select-control"
            value={selectedDate}
            onChange={(e) => handleDateChange(e.target.value)}
          >
            {AVAILABLE_DATES.map(d => (
              <option key={d.value} value={d.value}>{d.label}</option>
            ))}
          </select>

          {/* Map Style */}
          <select
            className="select-control"
            value={currentStyle}
            onChange={(e) => changeMapStyle(e.target.value)}
          >
            <option value="dark">Dark Map</option>
            <option value="satellite">Satellite View</option>
          </select>

          {/* Region */}
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

      {/* Main Content */}
      <div className="main-content">
        {/* Loading overlay */}
        {loading && (
          <div className="loading-overlay">
            <div className="loading-spinner"></div>
            <div className="loading-text">Loading drought risk data...</div>
          </div>
        )}

        {/* Sidebar toggle */}
        <button className="sidebar-toggle" onClick={() => setSidebarOpen(!sidebarOpen)}>
          {sidebarOpen ? '\u2715' : '\u2630'}
        </button>

        {/* Map */}
        <div className="map-container">
          <div id="map" ref={mapContainer} />

          {/* Stats Banner */}
          {stats && (
            <div className="stats-banner">
              <div className="stats-item">
                <div className="stats-value">{stats.totalCells.toLocaleString()}</div>
                <div className="stats-label">Grid Cells</div>
              </div>
              <div className="stats-item">
                <div className="stats-value" style={{ color: getRiskColor(parseFloat(stats.meanRisk)) }}>
                  {stats.meanRisk}
                </div>
                <div className="stats-label">Mean Risk</div>
              </div>
              <div className="stats-item">
                <div className="stats-value" style={{ color: getRiskColor(parseFloat(stats.maxRisk)) }}>
                  {stats.maxRisk}
                </div>
                <div className="stats-label">Max Risk</div>
              </div>
              <div className="stats-item">
                <div className="stats-value" style={{ color: '#f46d43' }}>
                  {(stats.levels.high || 0) + (stats.levels.severe || 0) + (stats.levels.extreme || 0)}
                </div>
                <div className="stats-label">High+ Cells</div>
              </div>
            </div>
          )}

          {/* Legend */}
          <div className="map-legend">
            <div className="map-legend__title">Drought Risk Index</div>
            <div className="map-legend__items">
              <div className="map-legend__item">
                <div className="map-legend__color" style={{ background: '#a50026' }}></div>
                <span>Extreme (0.8 - 1.0)</span>
              </div>
              <div className="map-legend__item">
                <div className="map-legend__color" style={{ background: '#d73027' }}></div>
                <span>Severe (0.6 - 0.8)</span>
              </div>
              <div className="map-legend__item">
                <div className="map-legend__color" style={{ background: '#f46d43' }}></div>
                <span>High (0.4 - 0.6)</span>
              </div>
              <div className="map-legend__item">
                <div className="map-legend__color" style={{ background: '#fee08b' }}></div>
                <span>Moderate (0.2 - 0.4)</span>
              </div>
              <div className="map-legend__item">
                <div className="map-legend__color" style={{ background: '#1a9641' }}></div>
                <span>Low (0.0 - 0.2)</span>
              </div>
            </div>
            <div className="map-legend__note">
              Model: XGBoost | R&sup2;=0.867 | Spearman r=0.929
            </div>
          </div>
        </div>

        {/* Detail Panel */}
        {selectedCell && (
          <div className={`detail-panel ${detailOpen ? 'detail-panel--open' : ''}`}>
            <div className="detail-panel__header">
              <h2 className="detail-panel__title">Cell Risk Analysis</h2>
              <button className="detail-panel__close" onClick={() => setDetailOpen(false)}>{'\u2715'}</button>
            </div>

            <div className="detail-panel__body">
              <div className="risk-gauge">
                <div className="risk-gauge__circle" style={{
                  '--gauge-color': getRiskColor(selectedCell.risk_score),
                  '--gauge-pct': `${selectedCell.risk_score * 100}%`
                }}>
                  <span className="risk-gauge__value">{selectedCell.risk_score?.toFixed(3)}</span>
                </div>
                <div className="risk-gauge__label" style={{ color: getRiskColor(selectedCell.risk_score) }}>
                  {selectedCell.risk_level?.toUpperCase()} Risk
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
                    <span className="factor-row__label">{key}</span>
                    <span className="factor-row__value">{value}</span>
                  </div>
                ))}
              </div>

              <div className="detail-section">
                <div className="detail-section__title">Alert Advisory</div>
                <div className="detail-alert-preview">
                  <div><strong>Projection:</strong> {selectedCell.days_to_event} days</div>
                  <div className="detail-alert-preview__action">
                    {selectedCell.alert_message}
                  </div>
                </div>
                <div style={{ marginTop: '8px', fontSize: '11px', color: '#888', fontStyle: 'italic' }}>
                  Trend projection only - not a forecast
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Right Sidebar - Active Alerts */}
        <aside className={`sidebar ${!sidebarOpen ? 'sidebar--collapsed' : ''}`}>
          <div className="sidebar__header">
            <div className="sidebar__title">
              Active Alerts <span className="sidebar__badge">{alertCells.length} LIVE</span>
            </div>
          </div>

          <div className="sidebar__content">
            {alertCells.map((feature) => {
              const p = feature.properties;
              return (
                <div
                  key={p.id}
                  className={`alert-card alert-card--${p.risk_level}`}
                  onClick={() => {
                    setSelectedCell(p);
                    setDetailOpen(true);
                    const coords = feature.geometry.coordinates[0][0];
                    if (map.current) {
                      map.current.flyTo({ center: coords, zoom: 10, duration: 1500 });
                    }
                  }}
                >
                  <div className="alert-card__header">
                    <span className={`alert-card__level alert-card__level--${p.risk_level}`}>
                      {p.risk_level}
                    </span>
                    <span className="alert-card__days">{p.days_to_event}d outlook</span>
                  </div>
                  <div className="alert-card__region">{p.region}</div>
                  <div className="alert-card__action">{p.alert_message}</div>
                  <div className="alert-card__meta">
                    <span className="alert-card__score">RISK: {p.risk_score?.toFixed(3)}</span>
                    <span className="alert-card__type">DROUGHT</span>
                  </div>
                </div>
              );
            })}
            {alertCells.length === 0 && !loading && (
              <div style={{ padding: '24px', color: '#888', textAlign: 'center', fontSize: '13px' }}>
                No high-risk alerts for this date.
              </div>
            )}
          </div>
        </aside>
      </div>

      {/* Timeline Bar */}
      <div className="timeline-bar">
        <div className="timeline-bar__controls">
          <button className="timeline-btn timeline-btn--active">Drought Model</button>
        </div>

        <div className="timeline-bar__slider-wrap">
          <div className="timeline-bar__dates" style={{ display: 'flex', gap: '8px', justifyContent: 'center', flexWrap: 'wrap' }}>
            {AVAILABLE_DATES.map(d => (
              <button
                key={d.value}
                className={`timeline-btn ${selectedDate === d.value ? 'timeline-btn--active' : ''}`}
                onClick={() => handleDateChange(d.value)}
              >
                {d.label}
              </button>
            ))}
          </div>
        </div>

        <div className="timeline-bar__current-date">
          Viewing: {selectedDate}
          <span className="timeline-bar__forecast-badge" style={{ marginLeft: '8px' }}>
            Real IDM Labels | Spearman r=0.929
          </span>
        </div>
      </div>
    </div>
  );
}