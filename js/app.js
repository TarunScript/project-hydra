/**
 * app.js — Main application entry point
 * Project Hydra — India Flood & Drought EWS
 * 
 * Orchestrates all modules: Map, Timeline, Alerts, DetailPanel, StatsBar
 */

// ── Toast notification system ──
function showToast(message, type = 'success') {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = `toast toast--${type}`;
  toast.textContent = message;
  container.appendChild(toast);

  requestAnimationFrame(() => {
    toast.classList.add('toast--visible');
  });

  setTimeout(() => {
    toast.classList.remove('toast--visible');
    setTimeout(() => toast.remove(), 400);
  }, 3000);
}

// ── Main App ──
const App = (() => {
  let currentRegionId = 'assam';
  let currentDateOffset = 0;

  async function init() {
    // 1. Initialize map
    HydraMap.init();

    // 2. Initialize sub-modules
    AlertPanel.init();
    DetailPanel.init();
    StatsBar.init();

    // 3. Initialize timeline with callback
    Timeline.init(onDateChange);

    // 4. Setup controls
    setupControls();

    // 5. Load initial data (fitBounds = true to auto-zoom to grid)
    await loadData(currentRegionId, 0, true);

    // 6. Show welcome toast
    showToast('🌊 Project Hydra loaded — Assam flood region', 'success');
  }

  async function loadData(regionId, dateOffset, fitBounds = false) {
    try {
      const geojson = await DataService.getRiskGrid(regionId, dateOffset);

      // Render risk overlay on map
      HydraMap.renderRiskGrid(geojson, fitBounds);

      // Update alerts
      const alerts = DataService.getAlerts(geojson);
      AlertPanel.render(alerts);

      // Update stats
      const stats = DataService.getStats(geojson);
      StatsBar.update(stats);
    } catch (err) {
      console.error('Failed to load risk data:', err);
      showToast('Failed to load risk data', 'error');
    }
  }

  async function onDateChange(offset, dateInfo) {
    currentDateOffset = offset;
    await loadData(currentRegionId, offset);
  }

  function setupControls() {
    // Region selector
    const regionSelect = document.getElementById('region-select');
    regionSelect.addEventListener('change', async (e) => {
      currentRegionId = e.target.value;
      HydraMap.setRegion(currentRegionId);

      // Update model badge
      const region = DataService.REGIONS[currentRegionId];
      const modelBadge = document.getElementById('model-badge');
      modelBadge.className = `model-badge model-badge--${region.type}`;
      modelBadge.textContent = region.type === 'flood' ? '🌊 Flood' : '☀️ Drought';

      await loadData(currentRegionId, currentDateOffset, true);
      showToast(`Switched to ${region.name} (${region.type})`, 'success');
    });

    // Model type selector
    const modelSelect = document.getElementById('model-select');
    modelSelect.addEventListener('change', (e) => {
      const type = e.target.value;
      // Find first region matching this type
      const matchingRegion = Object.values(DataService.REGIONS).find(r => r.type === type);
      if (matchingRegion) {
        regionSelect.value = matchingRegion.id;
        regionSelect.dispatchEvent(new Event('change'));
      }
    });

    // Sidebar toggle
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebar-toggle');
    sidebarToggle.addEventListener('click', () => {
      sidebar.classList.toggle('sidebar--collapsed');
      sidebarToggle.innerHTML = sidebar.classList.contains('sidebar--collapsed') ? '◀' : '▶';
    });
  }

  return { init };
})();

// ── Boot ──
document.addEventListener('DOMContentLoaded', () => {
  App.init();
});
