/**
 * timeline.js — Timeline slider with date navigation and auto-play
 * Project Hydra — India Flood & Drought EWS
 */

const Timeline = (() => {
  let dates = [];
  let currentIndex = 14; // Index 14 = today (offset 0)
  let isPlaying = false;
  let playInterval = null;
  let onDateChange = null;

  // ── DOM Elements ──
  let sliderEl, currentDateEl, forecastBadgeEl, playBtn;

  function init(callback) {
    onDateChange = callback;
    dates = DataService.getDateRange();
    currentIndex = dates.findIndex(d => d.isToday);

    sliderEl = document.getElementById('timeline-slider');
    currentDateEl = document.getElementById('current-date-display');
    forecastBadgeEl = document.getElementById('forecast-badge');
    playBtn = document.getElementById('btn-play');

    // Setup slider
    sliderEl.min = 0;
    sliderEl.max = dates.length - 1;
    sliderEl.value = currentIndex;

    // Render tick labels
    renderDateLabels();

    // Event listeners
    sliderEl.addEventListener('input', (e) => {
      setIndex(parseInt(e.target.value));
    });

    document.getElementById('btn-prev').addEventListener('click', () => {
      if (currentIndex > 0) setIndex(currentIndex - 1);
    });

    document.getElementById('btn-next').addEventListener('click', () => {
      if (currentIndex < dates.length - 1) setIndex(currentIndex + 1);
    });

    playBtn.addEventListener('click', togglePlay);

    document.getElementById('btn-today').addEventListener('click', () => {
      const todayIdx = dates.findIndex(d => d.isToday);
      setIndex(todayIdx);
    });

    // Initial update
    updateDisplay();
  }

  function renderDateLabels() {
    const container = document.getElementById('timeline-date-labels');
    container.innerHTML = '';

    // Show only a few key labels to avoid clutter
    const keyIndices = [0, 7, 14, 21, dates.length - 1];
    const sliderWidth = sliderEl.offsetWidth || 600;

    keyIndices.forEach((idx) => {
      if (idx >= dates.length) return;
      const d = dates[idx];
      const label = document.createElement('span');
      label.className = 'timeline-bar__date-label';
      if (d.isToday) label.classList.add('timeline-bar__date-label--today');
      if (d.isForecast) label.classList.add('timeline-bar__date-label--forecast');
      label.textContent = d.isToday ? '● Today' : d.label;
      container.appendChild(label);
    });
  }

  function setIndex(index) {
    if (index < 0 || index >= dates.length) return;
    currentIndex = index;
    sliderEl.value = index;
    updateDisplay();

    if (onDateChange) {
      const d = dates[currentIndex];
      onDateChange(d.offset, d);
    }
  }

  function updateDisplay() {
    const d = dates[currentIndex];
    const fullDate = d.date.toLocaleDateString('en-IN', {
      weekday: 'short',
      day: 'numeric',
      month: 'short',
      year: 'numeric',
    });
    currentDateEl.textContent = fullDate;

    // Forecast badge
    if (d.isForecast) {
      forecastBadgeEl.textContent = d.forecastDay;
      forecastBadgeEl.style.display = 'inline';
    } else if (d.isToday) {
      forecastBadgeEl.textContent = 'TODAY';
      forecastBadgeEl.style.display = 'inline';
      forecastBadgeEl.style.background = 'rgba(59, 130, 246, 0.15)';
      forecastBadgeEl.style.color = '#3b82f6';
    } else {
      forecastBadgeEl.style.display = 'none';
    }

    // Update slider gradient fill
    const pct = (currentIndex / (dates.length - 1)) * 100;
    const todayPct = (14 / (dates.length - 1)) * 100;

    // Color the track: historical (blue) → forecast (amber)
    sliderEl.style.background = `linear-gradient(to right, 
      #1e3a5f 0%, 
      #3b82f6 ${Math.min(pct, todayPct)}%, 
      #3b82f6 ${todayPct}%, 
      ${pct > todayPct ? '#eab308' : '#1e293b'} ${pct}%, 
      #1e293b ${pct}%, 
      #1e293b 100%)`;
  }

  function togglePlay() {
    if (isPlaying) {
      stopPlay();
    } else {
      startPlay();
    }
  }

  function startPlay() {
    isPlaying = true;
    playBtn.classList.add('timeline-btn--active');
    playBtn.innerHTML = '⏸';

    playInterval = setInterval(() => {
      if (currentIndex >= dates.length - 1) {
        setIndex(0);
      } else {
        setIndex(currentIndex + 1);
      }
    }, 800);
  }

  function stopPlay() {
    isPlaying = false;
    playBtn.classList.remove('timeline-btn--active');
    playBtn.innerHTML = '▶';
    if (playInterval) {
      clearInterval(playInterval);
      playInterval = null;
    }
  }

  function getCurrentDate() {
    return dates[currentIndex];
  }

  function getCurrentOffset() {
    return dates[currentIndex].offset;
  }

  return {
    init,
    setIndex,
    getCurrentDate,
    getCurrentOffset,
    stopPlay,
  };
})();
