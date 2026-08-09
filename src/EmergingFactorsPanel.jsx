/**
 * EmergingFactorsPanel.jsx — Standalone research panel component
 *
 * Surfaces real-world emerging factors near a map location that could
 * affect flood/drought risk. Designed to work independently of the
 * risk model branches.
 *
 * Props:
 *   lat       (number)  — latitude of selected location
 *   lon       (number)  — longitude of selected location
 *   locationName (string) — human-readable location name
 *   isOpen    (boolean) — whether the panel is visible
 *   onClose   (function) — callback to close the panel
 */

import React, { useEffect, useState } from 'react';

const API_BASE = 'http://localhost:3001';

// Category icons for visual distinction
const CATEGORY_ICONS = {
  'New/expanding data centers or AI infrastructure': '🖥️',
  'New/expanding semiconductor fabrication plants': '🔬',
  'Lithium/rare-earth/critical-mineral extraction projects': '⛏️',
  'Green hydrogen production or direct air capture facilities': '💨',
  'Groundwater extraction trends or new industrial water permits': '💧',
  'Major upstream dam/reservoir/irrigation changes': '🏗️',
  'Large-scale land-use change (deforestation, urban expansion)': '🌳'
};

// Short labels for categories
const CATEGORY_SHORT = {
  'New/expanding data centers or AI infrastructure': 'Data Centers / AI Infra',
  'New/expanding semiconductor fabrication plants': 'Semiconductor Fabs',
  'Lithium/rare-earth/critical-mineral extraction projects': 'Critical Minerals',
  'Green hydrogen production or direct air capture facilities': 'Green Hydrogen / DAC',
  'Groundwater extraction trends or new industrial water permits': 'Groundwater',
  'Major upstream dam/reservoir/irrigation changes': 'Dam / Reservoir',
  'Large-scale land-use change (deforestation, urban expansion)': 'Land-Use Change'
};

export default function EmergingFactorsPanel({ lat, lon, locationName, isOpen, onClose }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!isOpen || lat == null || lon == null) return;

    let cancelled = false;
    setLoading(true);
    setError(null);

    fetch(`${API_BASE}/api/emerging-factors?lat=${lat}&lon=${lon}&location_name=${encodeURIComponent(locationName || '')}`)
      .then(res => {
        if (!res.ok) throw new Error(`Server error ${res.status}`);
        return res.json();
      })
      .then(json => {
        if (!cancelled) setData(json);
      })
      .catch(err => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => { cancelled = true; };
  }, [lat, lon, isOpen]);

  if (!isOpen) return null;

  const factors = data?.emerging_factors;
  const findings = factors?.findings || [];

  return (
    <div className="efp-panel">
      <div className="efp-panel__header">
        <div>
          <h2 className="efp-panel__title">🔬 Emerging Factors</h2>
          <span className="efp-panel__subtitle">Informational — not a risk score</span>
        </div>
        <button className="efp-panel__close" onClick={onClose}>✕</button>
      </div>

      <div className="efp-panel__body">
        {/* Location context */}
        <div className="efp-location">
          <span className="efp-location__icon">📍</span>
          <div>
            <div className="efp-location__name">{locationName || `${lat}, ${lon}`}</div>
            <div className="efp-location__coords">{lat?.toFixed(2)}°N, {lon?.toFixed(2)}°E · ~50km radius</div>
          </div>
        </div>

        {/* Risk placeholder notice */}
        <div className="efp-risk-placeholder">
          <div className="efp-risk-placeholder__row">
            <span>Flood Risk Score:</span>
            <span className="efp-risk-placeholder__value">—</span>
            <span className="efp-risk-placeholder__tag">Model not connected</span>
          </div>
          <div className="efp-risk-placeholder__row">
            <span>Drought Risk Score:</span>
            <span className="efp-risk-placeholder__value">—</span>
            <span className="efp-risk-placeholder__tag">Model not connected</span>
          </div>
        </div>

        {/* Loading state */}
        {loading && (
          <div className="efp-loading">
            <div className="spinner"></div>
            <span>Searching for emerging factors...</span>
          </div>
        )}

        {/* Error state */}
        {error && (
          <div className="efp-error">
            <span>⚠️ {error}</span>
            <div className="efp-error__hint">Make sure the API server is running: <code>npm run server</code></div>
          </div>
        )}

        {/* No findings state */}
        {!loading && !error && factors?.no_findings && (
          <div className="efp-empty">
            <div className="efp-empty__icon">✅</div>
            <div className="efp-empty__text">No emerging factors found</div>
            <div className="efp-empty__sub">
              No recent developments matching the 7 monitored categories were found within ~50km of this location.
              This is a valid result — it means no unusual activity was detected.
            </div>
          </div>
        )}

        {/* Findings list */}
        {!loading && !error && findings.length > 0 && (
          <>
            <div className="efp-count">
              {findings.length} finding{findings.length !== 1 ? 's' : ''} detected
              {factors?.data_source === 'mock' && (
                <span className="efp-mock-badge">DEMO DATA</span>
              )}
            </div>

            {findings.map((f, idx) => (
              <div key={idx} className="efp-finding">
                <div className="efp-finding__header">
                  <span className="efp-finding__icon">{CATEGORY_ICONS[f.category] || '📋'}</span>
                  <span className="efp-finding__category">{CATEGORY_SHORT[f.category] || f.category}</span>
                  {f.source_date && (
                    <span className="efp-finding__date">{f.source_date}</span>
                  )}
                </div>
                <div className="efp-finding__summary">{f.summary}</div>
                <div className="efp-finding__relevance">
                  <span className="efp-finding__relevance-label">Relevance:</span> {f.relevance}
                </div>
                <a
                  href={f.source_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="efp-finding__source"
                >
                  🔗 {new URL(f.source_url).hostname.replace('www.', '')}
                </a>
              </div>
            ))}
          </>
        )}

        {/* Data source footer */}
        {!loading && factors && (
          <div className="efp-footer">
            <span>Source: {factors.data_source === 'mock' ? 'Mock data (demo)' : 'Perplexity Sonar (live search)'}</span>
            {factors.queried_at && <span>{new Date(factors.queried_at).toLocaleString()}</span>}
          </div>
        )}
      </div>
    </div>
  );
}
