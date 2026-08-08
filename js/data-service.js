/**
 * data-service.js — Data abstraction layer for Project Hydra
 * 
 * Loads mock GeoJSON data now, designed to swap to real API fetch later
 * with zero UI changes. All data access goes through this module.
 */

const DataService = (() => {
  // ── Configuration ──
  const REGIONS = {
    assam: {
      id: 'assam',
      name: 'Assam',
      type: 'flood',
      center: [26.2, 92.9],
      zoom: 7,
      bounds: [[24.1, 89.7], [28.2, 96.0]],
    },
    marathwada: {
      id: 'marathwada',
      name: 'Marathwada',
      type: 'drought',
      center: [19.0, 76.5],
      zoom: 8,
      bounds: [[17.5, 74.5], [20.5, 78.5]],
    },
  };

  // ── Mock Data Generator ──
  // Generates realistic GeoJSON grid cells with risk scores

  function seededRandom(seed) {
    let s = seed;
    return function () {
      s = (s * 16807) % 2147483647;
      return (s - 1) / 2147483646;
    };
  }

  function generateGridCells(region, dateOffset) {
    const bounds = region.bounds;
    const cellSize = region.type === 'flood' ? 0.15 : 0.12;
    const cells = [];

    const latMin = bounds[0][0], latMax = bounds[1][0];
    const lngMin = bounds[0][1], lngMax = bounds[1][1];

    const seed = Math.abs(dateOffset * 1000 + region.id.charCodeAt(0) * 100);
    const rng = seededRandom(seed + 42);

    // Create hotspot centers for realistic clustering
    const hotspots = [];
    const numHotspots = 2 + Math.floor(rng() * 3);
    for (let i = 0; i < numHotspots; i++) {
      hotspots.push({
        lat: latMin + rng() * (latMax - latMin),
        lng: lngMin + rng() * (lngMax - lngMin),
        intensity: 0.4 + rng() * 0.6,
        radius: 0.8 + rng() * 1.5,
      });
    }

    let cellId = 0;
    for (let lat = latMin; lat < latMax; lat += cellSize) {
      for (let lng = lngMin; lng < lngMax; lng += cellSize) {
        // Base risk from hotspot proximity
        let risk = 0.05 + rng() * 0.1;
        for (const hs of hotspots) {
          const dist = Math.sqrt((lat - hs.lat) ** 2 + (lng - hs.lng) ** 2);
          if (dist < hs.radius) {
            risk += hs.intensity * (1 - dist / hs.radius) * 0.7;
          }
        }

        // Add temporal variation — risk increases toward forecast
        const temporalFactor = region.type === 'flood'
          ? 1 + Math.sin(dateOffset * 0.5) * 0.3
          : 1 + dateOffset * 0.02;

        risk = Math.min(1, Math.max(0, risk * temporalFactor + rng() * 0.08));

        // Generate contributing factors
        const factors = region.type === 'flood'
          ? {
            rainfall_7d: Math.round((50 + risk * 200 + rng() * 40) * 10) / 10,
            soil_moisture: Math.round((0.3 + risk * 0.5 + rng() * 0.1) * 100) / 100,
            flow_accumulation: Math.round(500 + risk * 8000 + rng() * 1000),
            elevation: Math.round(20 + rng() * 200),
            slope: Math.round((1 + rng() * 15) * 10) / 10,
            distance_to_river: Math.round((0.5 + rng() * 20) * 10) / 10,
          }
          : {
            rainfall_deficit: Math.round((risk * -150 - rng() * 30) * 10) / 10,
            ndvi_anomaly: Math.round((-0.1 - risk * 0.4 + rng() * 0.05) * 1000) / 1000,
            soil_moisture: Math.round((0.4 - risk * 0.3 + rng() * 0.05) * 100) / 100,
            temp_anomaly: Math.round((risk * 4 + rng() * 1.5) * 10) / 10,
            dry_spell_days: Math.round(risk * 45 + rng() * 10),
            et_anomaly: Math.round((risk * -2 + rng() * 0.5) * 10) / 10,
          };

        cells.push({
          type: 'Feature',
          properties: {
            id: `${region.id}-cell-${cellId++}`,
            risk_score: Math.round(risk * 1000) / 1000,
            risk_category: getRiskCategory(risk),
            region: region.name,
            model_type: region.type,
            lat: Math.round((lat + cellSize / 2) * 1000) / 1000,
            lng: Math.round((lng + cellSize / 2) * 1000) / 1000,
            factors,
          },
          geometry: {
            type: 'Polygon',
            coordinates: [[
              [lng, lat],
              [lng + cellSize, lat],
              [lng + cellSize, lat + cellSize],
              [lng, lat + cellSize],
              [lng, lat],
            ]],
          },
        });
      }
    }

    return {
      type: 'FeatureCollection',
      features: cells,
    };
  }

  function getRiskCategory(score) {
    if (score >= 0.75) return 'severe';
    if (score >= 0.5) return 'high';
    if (score >= 0.25) return 'moderate';
    return 'low';
  }

  // ── Date Helpers ──
  const TODAY = new Date();
  TODAY.setHours(0, 0, 0, 0);

  function getDateRange() {
    const dates = [];
    for (let i = -14; i <= 15; i++) {
      const d = new Date(TODAY);
      d.setDate(d.getDate() + i);
      dates.push({
        date: d,
        offset: i,
        label: formatDate(d),
        isToday: i === 0,
        isForecast: i > 0,
        forecastDay: i > 0 ? `Day +${i}` : null,
      });
    }
    return dates;
  }

  function formatDate(date) {
    return date.toLocaleDateString('en-IN', {
      day: 'numeric',
      month: 'short',
    });
  }

  // ── Data Cache ──
  const cache = new Map();

  function getCacheKey(regionId, dateOffset) {
    return `${regionId}:${dateOffset}`;
  }

  // ── Public API ──
  async function getRiskGrid(regionId, dateOffset = 0) {
    const key = getCacheKey(regionId, dateOffset);

    if (cache.has(key)) {
      return cache.get(key);
    }

    const region = REGIONS[regionId];
    if (!region) throw new Error(`Unknown region: ${regionId}`);

    // TODO: Replace with real API fetch when backend is ready
    // const response = await fetch(`/api/risk-grid/${regionId}?date_offset=${dateOffset}`);
    // const data = await response.json();

    // For now: generate mock data
    const data = generateGridCells(region, dateOffset);
    cache.set(key, data);
    return data;
  }

  function getAlerts(geojson) {
    if (!geojson || !geojson.features) return [];

    const ACTIONS = {
      severe: '⚠️ EVACUATE immediately. Move to higher ground. Follow local authority instructions.',
      high: '🔶 PREPARE — Store water, move valuables to higher ground, keep emergency kit ready.',
      moderate: '📋 MONITOR conditions. Stay informed via local weather alerts.',
      low: 'No action needed. Conditions normal.',
    };

    const alerts = geojson.features
      .filter(f => f.properties.risk_score >= 0.25)
      .sort((a, b) => b.properties.risk_score - a.properties.risk_score)
      .slice(0, 20)
      .map((f, i) => {
        const p = f.properties;
        const daysToEvent = p.risk_category === 'severe' ? Math.floor(Math.random() * 3)
          : p.risk_category === 'high' ? 3 + Math.floor(Math.random() * 4)
            : 7 + Math.floor(Math.random() * 8);

        return {
          id: `alert-${i}`,
          cellId: p.id,
          region: p.region,
          riskScore: p.risk_score,
          riskCategory: p.risk_category,
          modelType: p.model_type,
          daysToEvent,
          action: ACTIONS[p.risk_category],
          lat: p.lat,
          lng: p.lng,
        };
      });

    return alerts;
  }

  function getStats(geojson) {
    if (!geojson || !geojson.features) {
      return { totalCells: 0, severeCount: 0, highCount: 0, moderateCount: 0, lowCount: 0, avgRisk: 0 };
    }

    const features = geojson.features;
    let severeCount = 0, highCount = 0, moderateCount = 0, lowCount = 0;
    let totalRisk = 0;

    for (const f of features) {
      const cat = f.properties.risk_category;
      if (cat === 'severe') severeCount++;
      else if (cat === 'high') highCount++;
      else if (cat === 'moderate') moderateCount++;
      else lowCount++;
      totalRisk += f.properties.risk_score;
    }

    return {
      totalCells: features.length,
      severeCount,
      highCount,
      moderateCount,
      lowCount,
      avgRisk: Math.round((totalRisk / features.length) * 1000) / 1000,
      atRiskCount: severeCount + highCount,
    };
  }

  function getTrendData(regionId, cellId) {
    // Generate a mini trend across dates for a given cell
    const region = REGIONS[regionId];
    if (!region) return [];

    const points = [];
    for (let offset = -14; offset <= 15; offset++) {
      const grid = generateGridCells(region, offset);
      const cell = grid.features.find(f => f.properties.id === cellId);
      if (cell) {
        const d = new Date(TODAY);
        d.setDate(d.getDate() + offset);
        points.push({
          date: d,
          offset,
          riskScore: cell.properties.risk_score,
        });
      }
    }
    return points;
  }

  return {
    REGIONS,
    getDateRange,
    getRiskGrid,
    getAlerts,
    getStats,
    getTrendData,
    getRiskCategory,
    formatDate,
    TODAY,
  };
})();
