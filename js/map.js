/**
 * map.js — Leaflet map initialization, risk overlay, and interaction
 * Project Hydra — India Flood & Drought EWS
 */

const HydraMap = (() => {
  let map = null;
  let riskLayer = null;
  let selectedCellLayer = null;
  let currentRegionId = 'assam';

  // ── Tile layers ──
  const TILE_LAYERS = {
    dark: {
      url: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
      name: 'Dark',
    },
    satellite: {
      url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      attribution: '&copy; Esri, Maxar, Earthstar Geographics',
      name: 'Satellite',
    },
    terrain: {
      url: 'https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png',
      attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
      name: 'Terrain',
    },
  };

  // ── Risk Color Scale ──
  function getRiskColor(score) {
    if (score >= 0.75) return '#ef4444';  // Severe — red
    if (score >= 0.5) return '#f97316';   // High — orange
    if (score >= 0.25) return '#eab308';  // Moderate — amber
    return '#22c55e';                     // Low — green
  }

  function getRiskOpacity(score) {
    return 0.3 + score * 0.5;
  }

  // ── Initialize Map ──
  function init() {
    const region = DataService.REGIONS[currentRegionId];

    map = L.map('map', {
      center: region.center,
      zoom: region.zoom,
      zoomControl: false,
      attributionControl: true,
    });

    // Add zoom control to bottom-left
    L.control.zoom({ position: 'bottomleft' }).addTo(map);

    // Add dark tile layer
    L.tileLayer(TILE_LAYERS.dark.url, {
      attribution: TILE_LAYERS.dark.attribution,
      maxZoom: 18,
      subdomains: 'abcd',
    }).addTo(map);

    // Click handler for map background — close detail panel if not clicking a cell
    map.on('click', (e) => {
      // Use a short delay to let the feature click handler set the flag first
      setTimeout(() => {
        if (!e.originalEvent._cellClicked) {
          DetailPanel.close();
        }
      }, 50);
    });

    return map;
  }

  // ── Render Risk Grid ──
  function renderRiskGrid(geojson, fitBounds) {
    // Remove existing risk layer
    if (riskLayer) {
      map.removeLayer(riskLayer);
      riskLayer = null;
    }

    if (!geojson || !geojson.features || geojson.features.length === 0) return;

    riskLayer = L.geoJSON(geojson, {
      style: (feature) => {
        const score = feature.properties.risk_score;
        return {
          fillColor: getRiskColor(score),
          fillOpacity: getRiskOpacity(score),
          color: getRiskColor(score),
          weight: 0.5,
          opacity: 0.4,
        };
      },
      onEachFeature: (feature, layer) => {
        // Hover effect
        layer.on('mouseover', (e) => {
          const l = e.target;
          l.setStyle({
            weight: 2,
            opacity: 0.9,
            fillOpacity: getRiskOpacity(feature.properties.risk_score) + 0.15,
          });
          l.bringToFront();
          showTooltip(e, feature.properties);
        });

        layer.on('mouseout', (e) => {
          riskLayer.resetStyle(e.target);
          hideTooltip();
        });

        // Click — open detail panel
        layer.on('click', (e) => {
          console.log('Cell clicked:', feature.properties.id, feature.properties.risk_score);
          e.originalEvent._cellClicked = true;
          L.DomEvent.stopPropagation(e);
          highlightCell(e.target);
          DetailPanel.open(feature.properties, currentRegionId);
        });
      },
    }).addTo(map);

    // Auto-fit map to the risk grid bounds (only on first load / region change)
    if (fitBounds && riskLayer.getBounds().isValid()) {
      map.fitBounds(riskLayer.getBounds(), { padding: [20, 20], maxZoom: 9 });
    }
  }

  // ── Highlight selected cell ──
  function highlightCell(layer) {
    if (selectedCellLayer) {
      riskLayer.resetStyle(selectedCellLayer);
    }
    selectedCellLayer = layer;
    layer.setStyle({
      weight: 3,
      color: '#38bdf8',
      opacity: 1,
      fillOpacity: 0.8,
    });
    layer.bringToFront();
  }

  // ── Tooltip ──
  let tooltipEl = null;

  function showTooltip(e, props) {
    if (!tooltipEl) {
      tooltipEl = document.createElement('div');
      tooltipEl.className = 'map-tooltip';
      tooltipEl.style.cssText = `
        position: fixed;
        z-index: 2000;
        padding: 8px 12px;
        background: rgba(15, 23, 42, 0.95);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        backdrop-filter: blur(12px);
        pointer-events: none;
        font-family: var(--font-sans);
        font-size: 12px;
        color: #f1f5f9;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        transition: opacity 0.15s;
      `;
      document.body.appendChild(tooltipEl);
    }

    const riskColor = getRiskColor(props.risk_score);
    tooltipEl.innerHTML = `
      <div style="font-weight:700; margin-bottom:3px;">
        <span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${riskColor};margin-right:6px;"></span>
        ${props.risk_category.toUpperCase()}
      </div>
      <div style="color:#94a3b8;">Risk Score: <span style="color:${riskColor};font-weight:600;font-family:'JetBrains Mono',monospace;">${props.risk_score.toFixed(3)}</span></div>
      <div style="color:#64748b;font-size:11px;margin-top:2px;">${props.lat.toFixed(2)}°N, ${props.lng.toFixed(2)}°E</div>
    `;

    tooltipEl.style.opacity = '1';
    tooltipEl.style.left = (e.originalEvent.clientX + 16) + 'px';
    tooltipEl.style.top = (e.originalEvent.clientY - 10) + 'px';
  }

  function hideTooltip() {
    if (tooltipEl) {
      tooltipEl.style.opacity = '0';
    }
  }

  // ── Move tooltip with mouse ──
  document.addEventListener('mousemove', (e) => {
    if (tooltipEl && tooltipEl.style.opacity === '1') {
      tooltipEl.style.left = (e.clientX + 16) + 'px';
      tooltipEl.style.top = (e.clientY - 10) + 'px';
    }
  });

  // ── Change region ──
  function setRegion(regionId) {
    const region = DataService.REGIONS[regionId];
    if (!region) return;
    currentRegionId = regionId;
    map.flyTo(region.center, region.zoom, { duration: 1.5 });
  }

  function getRegionId() {
    return currentRegionId;
  }

  function getMap() {
    return map;
  }

  return {
    init,
    renderRiskGrid,
    setRegion,
    getRegionId,
    getMap,
    getRiskColor,
  };
})();
