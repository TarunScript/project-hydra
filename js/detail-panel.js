/**
 * detail-panel.js — Cell detail / inspect panel (slide-in)
 * Project Hydra — India Flood & Drought EWS
 */

const DetailPanel = (() => {
  let panelEl, bodyEl;
  let isOpen = false;

  function init() {
    panelEl = document.getElementById('detail-panel');
    bodyEl = document.getElementById('detail-panel-body');

    document.getElementById('detail-panel-close').addEventListener('click', close);
  }

  function open(cellProps, regionId) {
    if (!panelEl) return;

    isOpen = true;
    panelEl.classList.add('detail-panel--open');

    const riskColor = HydraMap.getRiskColor(cellProps.risk_score);
    const gaugePct = Math.round(cellProps.risk_score * 100);

    bodyEl.innerHTML = `
      <!-- Risk Gauge -->
      <div class="risk-gauge">
        <div class="risk-gauge__circle" style="--gauge-color: ${riskColor}; --gauge-pct: ${gaugePct}%; background: ${riskColor}15;">
          <span class="risk-gauge__value" style="color: ${riskColor};">${cellProps.risk_score.toFixed(2)}</span>
          <span class="risk-gauge__label" style="color: ${riskColor};">${cellProps.risk_category}</span>
        </div>
      </div>

      <!-- Location -->
      <div class="detail-section">
        <div class="detail-section__title">📍 Location</div>
        <div class="factor-row">
          <span class="factor-row__label">Coordinates</span>
          <span class="factor-row__value">${cellProps.lat.toFixed(3)}°N, ${cellProps.lng.toFixed(3)}°E</span>
        </div>
        <div class="factor-row">
          <span class="factor-row__label">Region</span>
          <span class="factor-row__value">${cellProps.region}</span>
        </div>
        <div class="factor-row">
          <span class="factor-row__label">Model</span>
          <span class="factor-row__value">
            <span class="model-badge model-badge--${cellProps.model_type}">${cellProps.model_type}</span>
          </span>
        </div>
      </div>

      <!-- Contributing Factors -->
      <div class="detail-section">
        <div class="detail-section__title">📊 Contributing Factors</div>
        ${renderFactors(cellProps.factors, cellProps.model_type)}
      </div>

      <!-- Risk Trend Chart -->
      <div class="detail-section">
        <div class="detail-section__title">📈 Risk Trend (30-day)</div>
        <div class="trend-chart">
          <canvas id="trend-canvas"></canvas>
        </div>
      </div>

      <!-- Alert Preview -->
      <div class="detail-section">
        <div class="detail-section__title">🔔 Alert Preview</div>
        <div class="detail-alert-preview">
          ${renderAlertPreview(cellProps)}
        </div>
      </div>

      <!-- Simulated SMS Button -->
      <button class="btn-sms" id="btn-send-sms" data-cell="${cellProps.id}">
        📱 Simulate SMS Alert
      </button>
    `;

    // Render trend chart
    setTimeout(() => renderTrendChart(regionId, cellProps.id, riskColor), 50);

    // SMS button handler
    document.getElementById('btn-send-sms').addEventListener('click', (e) => {
      simulateSMS(cellProps);
      e.target.innerHTML = '✅ SMS Sent (Simulated)';
      e.target.style.background = 'linear-gradient(135deg, #22c55e 0%, #16a34a 100%)';
      setTimeout(() => {
        e.target.innerHTML = '📱 Simulate SMS Alert';
        e.target.style.background = '';
      }, 2000);
    });
  }

  function close() {
    if (!panelEl) return;
    isOpen = false;
    panelEl.classList.remove('detail-panel--open');
  }

  function renderFactors(factors, modelType) {
    const FACTOR_LABELS = {
      // Flood
      rainfall_7d: { label: 'Rainfall (7-day)', unit: 'mm', icon: '🌧️' },
      soil_moisture: { label: 'Soil Moisture', unit: '', icon: '💧' },
      flow_accumulation: { label: 'Flow Accumulation', unit: '', icon: '🌊' },
      elevation: { label: 'Elevation', unit: 'm', icon: '⛰️' },
      slope: { label: 'Slope', unit: '°', icon: '📐' },
      distance_to_river: { label: 'Distance to River', unit: 'km', icon: '🏞️' },
      // Drought
      rainfall_deficit: { label: 'Rainfall Deficit', unit: 'mm', icon: '☀️' },
      ndvi_anomaly: { label: 'NDVI Anomaly', unit: '', icon: '🌿' },
      temp_anomaly: { label: 'Temp Anomaly', unit: '°C', icon: '🌡️' },
      dry_spell_days: { label: 'Dry Spell', unit: 'days', icon: '🏜️' },
      et_anomaly: { label: 'ET Anomaly', unit: 'mm/day', icon: '💨' },
    };

    let html = '';
    for (const [key, value] of Object.entries(factors)) {
      const meta = FACTOR_LABELS[key] || { label: key, unit: '', icon: '📋' };
      const displayValue = typeof value === 'number'
        ? (Number.isInteger(value) ? value.toLocaleString() : value.toFixed(2))
        : value;

      html += `
        <div class="factor-row">
          <span class="factor-row__label">${meta.icon} ${meta.label}</span>
          <span class="factor-row__value">${displayValue} ${meta.unit}</span>
        </div>
      `;
    }
    return html;
  }

  function renderAlertPreview(props) {
    if (props.risk_score < 0.25) {
      return '<div style="color: var(--risk-low); font-size: var(--text-sm);">✅ No alert — risk is low.</div>';
    }

    const actions = {
      severe: '⚠️ EVACUATE immediately. Move to higher ground.',
      high: '🔶 PREPARE — Store water, move valuables.',
      moderate: '📋 MONITOR conditions closely.',
    };

    const daysEstimate = props.risk_category === 'severe' ? '0–3' :
      props.risk_category === 'high' ? '3–7' : '7–15';

    return `
      <div style="margin-bottom: 8px;">
        <span class="alert-card__level alert-card__level--${props.risk_category}" style="font-size: 11px;">
          ${props.risk_category.toUpperCase()}
        </span>
        <span style="color: var(--text-muted); font-size: 11px; margin-left: 8px;">
          Est. ${daysEstimate} days to event
        </span>
      </div>
      <div class="detail-alert-preview__action" style="
        background: var(--risk-${props.risk_category}-bg);
        border-left-color: var(--risk-${props.risk_category});
      ">
        ${actions[props.risk_category] || 'Monitor conditions.'}
      </div>
    `;
  }

  function renderTrendChart(regionId, cellId, riskColor) {
    const canvas = document.getElementById('trend-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * 2;
    canvas.height = rect.height * 2;
    ctx.scale(2, 2);

    const w = rect.width;
    const h = rect.height;

    const trendData = DataService.getTrendData(regionId, cellId);
    if (trendData.length === 0) return;

    const padding = { top: 10, right: 10, bottom: 20, left: 10 };
    const plotW = w - padding.left - padding.right;
    const plotH = h - padding.top - padding.bottom;

    // Draw grid lines
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.08)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = padding.top + (plotH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(w - padding.right, y);
      ctx.stroke();
    }

    // Draw "today" line
    const todayIdx = trendData.findIndex(t => t.offset === 0);
    if (todayIdx >= 0) {
      const x = padding.left + (todayIdx / (trendData.length - 1)) * plotW;
      ctx.strokeStyle = 'rgba(59, 130, 246, 0.3)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, h - padding.bottom);
      ctx.stroke();
      ctx.setLineDash([]);

      // "Today" label
      ctx.fillStyle = '#3b82f6';
      ctx.font = '9px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Today', x, h - 4);
    }

    // Draw risk line
    ctx.beginPath();
    ctx.strokeStyle = riskColor;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    trendData.forEach((point, i) => {
      const x = padding.left + (i / (trendData.length - 1)) * plotW;
      const y = padding.top + (1 - point.riskScore) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw fill gradient
    const gradient = ctx.createLinearGradient(0, padding.top, 0, h - padding.bottom);
    gradient.addColorStop(0, riskColor + '30');
    gradient.addColorStop(1, riskColor + '05');

    ctx.lineTo(padding.left + plotW, h - padding.bottom);
    ctx.lineTo(padding.left, h - padding.bottom);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Draw dots at endpoints
    [0, trendData.length - 1].forEach(i => {
      const x = padding.left + (i / (trendData.length - 1)) * plotW;
      const y = padding.top + (1 - trendData[i].riskScore) * plotH;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fillStyle = riskColor;
      ctx.fill();
    });
  }

  function simulateSMS(cellProps) {
    const msg = `[Project Hydra Alert] ${cellProps.risk_category.toUpperCase()} ${cellProps.model_type} risk at ${cellProps.lat.toFixed(2)}°N, ${cellProps.lng.toFixed(2)}°E (${cellProps.region}). Risk Score: ${cellProps.risk_score.toFixed(2)}. Please take necessary precautions.`;
    console.log('📱 Simulated SMS:', msg);
    showToast(`SMS simulated to residents near ${cellProps.lat.toFixed(2)}°N`, 'success');
  }

  return {
    init,
    open,
    close,
  };
})();
