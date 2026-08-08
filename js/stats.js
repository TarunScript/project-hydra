/**
 * stats.js — Header stats display
 * Project Hydra — India Flood & Drought EWS
 */

const StatsBar = (() => {
  let severeEl, atRiskEl, totalEl, avgRiskEl;

  function init() {
    severeEl = document.getElementById('stat-severe');
    atRiskEl = document.getElementById('stat-at-risk');
    totalEl = document.getElementById('stat-total');
    avgRiskEl = document.getElementById('stat-avg-risk');
  }

  function update(stats) {
    if (!severeEl) return;

    animateValue(severeEl, parseInt(severeEl.textContent) || 0, stats.severeCount, 300);
    animateValue(atRiskEl, parseInt(atRiskEl.textContent) || 0, stats.atRiskCount, 300);
    animateValue(totalEl, parseInt(totalEl.textContent) || 0, stats.totalCells, 300);

    if (avgRiskEl) {
      avgRiskEl.textContent = stats.avgRisk.toFixed(3);
    }
  }

  function animateValue(el, start, end, duration) {
    if (start === end) {
      el.textContent = end;
      return;
    }

    const range = end - start;
    const startTime = performance.now();

    function step(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // easeOutCubic
      el.textContent = Math.round(start + range * eased);

      if (progress < 1) {
        requestAnimationFrame(step);
      }
    }

    requestAnimationFrame(step);
  }

  return {
    init,
    update,
  };
})();
