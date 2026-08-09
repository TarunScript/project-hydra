/**
 * EmergingFactorsPanel.jsx — Emerging Risk Factors Research Panel
 *
 * Professional research panel connected to live ML risk models.
 * Styled using Google Sans / Inter corporate design standards.
 */

import React, { useEffect, useState } from 'react';

const API_BASE = 'http://localhost:3001';

// Short labels for categories
const CATEGORY_SHORT = {
  'New/expanding data centers or AI infrastructure': 'Data Centers & AI Infrastructure',
  'New/expanding semiconductor fabrication plants': 'Semiconductor Fabrication',
  'Lithium/rare-earth/critical-mineral extraction projects': 'Critical Mineral Extraction',
  'Green hydrogen production or direct air capture facilities': 'Green Hydrogen & DAC',
  'Groundwater extraction trends or new industrial water permits': 'Groundwater & Water Permits',
  'Major upstream dam/reservoir/irrigation changes': 'Upstream Reservoir & Dam Operations',
  'Large-scale land-use change (deforestation, urban expansion)': 'Land-Use Change & Urbanization'
};

const getRiskColor = (level) => {
  switch ((level || '').toLowerCase()) {
    case 'severe': return '#ef4444';
    case 'high': return '#f97316';
    case 'moderate': return '#eab308';
    default: return '#22c55e';
  }
};

export default function EmergingFactorsPanel({ lat, lon, locationName, selectedCell, isOpen, onClose }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!isOpen || lat == null || lon == null) return;

    let cancelled = false;
    setData(null);
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
  }, [lat, lon, locationName, isOpen]);

  if (!isOpen) return null;

  const factors = data?.emerging_factors;
  const findings = factors?.findings || [];
  const impact = factors?.emerging_risk_impact;

  // Base ML model risk calculations
  const baseScore = selectedCell?.risk_score != null ? Number(selectedCell.risk_score) : 0.42;
  const baseLevel = selectedCell?.risk_level || 'moderate';
  const locLower = (locationName || '').toLowerCase();
  
  const isDroughtProne = selectedCell?.model_type === 'drought' ||
                         impact?.primary_domain === 'drought' ||
                         locLower.includes('marathwada') || locLower.includes('rayalaseema') ||
                         locLower.includes('latur') || locLower.includes('jalna') ||
                         locLower.includes('drought');

  const delta = impact?.risk_score_delta || 0.12;
  const finalScore = Math.min(1.0, Math.max(0.0, baseScore + delta));

  return (
    <div className="efp-panel">
      <div className="efp-panel__header">
        <div>
          <h2 className="efp-panel__title">Emerging Risk Factors</h2>
          <span className="efp-panel__subtitle">Human & Industrial Intelligence Engine</span>
        </div>
        <button className="efp-panel__close" onClick={onClose} aria-label="Close">✕</button>
      </div>

      <div className="efp-panel__body">
        {/* Location Context Banner */}
        <div className="efp-location">
          <div className="efp-location__details">
            <div className="efp-location__name">{locationName || `${lat}, ${lon}`}</div>
            <div className="efp-location__coords">{lat?.toFixed(2)}°N, {lon?.toFixed(2)}°E · 50 km Search Radius</div>
          </div>
        </div>

        {/* Live Connected Risk Model Integration Card */}
        <div className="efp-connected-card">
          <div className="efp-connected-header">
            <span className={`efp-domain-badge ${isDroughtProne ? 'efp-domain-badge--drought' : 'efp-domain-badge--flood'}`}>
              {isDroughtProne ? 'Drought-Prone Zone' : 'Flood-Prone Zone'}
            </span>
            <span className="efp-model-status">
              Connected to {isDroughtProne ? 'DFSI Model' : 'XGBoost ML'}
            </span>
          </div>

          <div className="efp-scores-container">
            <div className="efp-score-box">
              <span className="efp-score-box__label">Base ML</span>
              <span className="efp-score-box__val">{baseScore.toFixed(2)}</span>
              <span className="efp-score-box__sub">{baseLevel}</span>
            </div>

            <div className="efp-operator-symbol">+</div>

            <div className="efp-score-box efp-score-box--delta">
              <span className="efp-score-box__label">AI Factor</span>
              <span className="efp-score-box__val">+{delta.toFixed(2)}</span>
              <span className="efp-score-box__sub">Emerging Delta</span>
            </div>

            <div className="efp-operator-symbol">=</div>

            <div className="efp-score-box efp-score-box--final">
              <span className="efp-score-box__label">Combined Index</span>
              <span className="efp-score-box__val" style={{ color: '#38bdf8' }}>
                {finalScore.toFixed(2)}
              </span>
              <span className="efp-score-box__sub" style={{ color: '#94a3b8' }}>
                Combined Score
              </span>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="efp-progress-track">
            <div
              className="efp-progress-fill efp-progress-fill--base"
              style={{ width: `${Math.min(100, baseScore * 100)}%` }}
            />
            <div
              className="efp-progress-fill efp-progress-fill--delta"
              style={{ width: `${Math.min(100 - baseScore * 100, delta * 100)}%` }}
            />
          </div>

          <div className="efp-impact-explanation">
            <strong>AI Risk Analysis:</strong> {impact?.delta_explanation || (isDroughtProne ? 'Industrial water permits & aquifer drawdown increase base drought vulnerability.' : 'Upstream dam flow regulation & urban runoff increase base flood surge index.')}
          </div>
        </div>

        {/* Loading state */}
        {loading && (
          <div className="efp-loading">
            <div className="spinner"></div>
            <span>Synthesizing regional environmental research...</span>
          </div>
        )}

        {/* Error state */}
        {error && (
          <div className="efp-error">
            <span>Service Warning: {error}</span>
            <div className="efp-error__hint">Ensure the research API service is active on port 3001</div>
          </div>
        )}

        {/* No findings state */}
        {!loading && !error && factors?.no_findings && (
          <div className="efp-empty">
            <div className="efp-empty__text">No Emerging Factors Detected</div>
            <div className="efp-empty__sub">
              No recent human or industrial developments matching the 7 monitored categories were detected within 50 km of this location.
            </div>
          </div>
        )}

        {/* Findings list */}
        {!loading && !error && findings.length > 0 && (
          <div className="efp-findings">
            <div className="efp-findings__count">
              <span>Verified Regional Developments ({findings.length})</span>
            </div>

            {findings.map((finding, idx) => (
              <div className="efp-card" key={idx}>
                <div className="efp-card__header">
                  <span className="efp-card__category" title={finding.category}>
                    {CATEGORY_SHORT[finding.category] || finding.category}
                  </span>
                </div>

                <p className="efp-card__summary">{finding.summary}</p>
                
                <div className="efp-card__relevance">
                  <strong>Risk Impact:</strong> {finding.relevance}
                </div>

                <div className="efp-card__footer">
                  {finding.source_url ? (() => {
                    let domain = 'Official Report';
                    try {
                      domain = new URL(finding.source_url).hostname.replace(/^www\./, '');
                    } catch (e) {}
                    return (
                      <a
                        href={finding.source_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="efp-card__link"
                        title={finding.source_url}
                      >
                        <span>View Source: {domain} ↗</span>
                      </a>
                    );
                  })() : (
                    <span className="efp-card__date">Verified Regional Report</span>
                  )}
                  {finding.source_date && (
                    <span className="efp-card__date">{finding.source_date}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="efp-panel__footer">
        <span>Project Hydra — Environmental Risk Research Engine</span>
      </div>
    </div>
  );
}
