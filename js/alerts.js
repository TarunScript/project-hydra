/**
 * alerts.js — Alert panel rendering
 * Project Hydra — India Flood & Drought EWS
 */

const AlertPanel = (() => {
  let containerEl, badgeEl;

  function init() {
    containerEl = document.getElementById('alert-list');
    badgeEl = document.getElementById('alert-count-badge');
  }

  function render(alerts) {
    if (!containerEl) return;

    // Update badge count
    const severeHighCount = alerts.filter(a => a.riskCategory === 'severe' || a.riskCategory === 'high').length;
    badgeEl.textContent = severeHighCount;
    badgeEl.style.display = severeHighCount > 0 ? 'inline' : 'none';

    if (alerts.length === 0) {
      containerEl.innerHTML = `
        <div class="empty-state">
          <div class="empty-state__icon">✅</div>
          <div class="empty-state__text">No active alerts for this date</div>
        </div>
      `;
      return;
    }

    containerEl.innerHTML = alerts.map((alert, i) => `
      <div class="alert-card alert-card--${alert.riskCategory}" 
           data-cell-id="${alert.cellId}" 
           data-lat="${alert.lat}" 
           data-lng="${alert.lng}"
           style="animation-delay: ${i * 0.05}s"
           id="alert-${alert.id}">
        <div class="alert-card__header">
          <span class="alert-card__level alert-card__level--${alert.riskCategory}">
            ${getIcon(alert.riskCategory)} ${alert.riskCategory}
          </span>
          <span class="alert-card__days">${alert.daysToEvent}d out</span>
        </div>
        <div class="alert-card__region">${alert.region} — ${alert.lat.toFixed(2)}°N</div>
        <div class="alert-card__action">${alert.action}</div>
        <div class="alert-card__meta">
          <span class="alert-card__score">Risk: ${alert.riskScore.toFixed(3)}</span>
          <span class="alert-card__type">${alert.modelType}</span>
        </div>
      </div>
    `).join('');

    // Add click handlers to zoom to cell
    containerEl.querySelectorAll('.alert-card').forEach(card => {
      card.addEventListener('click', () => {
        const lat = parseFloat(card.dataset.lat);
        const lng = parseFloat(card.dataset.lng);
        const map = HydraMap.getMap();
        map.flyTo([lat, lng], 10, { duration: 1 });

        // Highlight the card
        containerEl.querySelectorAll('.alert-card').forEach(c => c.style.outline = 'none');
        card.style.outline = '1px solid rgba(59, 130, 246, 0.5)';
      });
    });
  }

  function getIcon(category) {
    switch (category) {
      case 'severe': return '🔴';
      case 'high': return '🟠';
      case 'moderate': return '🟡';
      case 'low': return '🟢';
      default: return '⚪';
    }
  }

  return {
    init,
    render,
  };
})();
