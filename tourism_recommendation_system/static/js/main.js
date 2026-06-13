/* ═══════════════════════════════════════════════════════
   TourMate AI — Main JavaScript
   ═══════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
  initNavbar();
  initMatchBars();
  initCounters();
  initLoadingOverlay();
  initTooltips();
  initPlotlyCharts();
  initRatingSlider();
  initSearchSuggestions();
  initScrollReveal();
});

/* ── Navbar ──────────────────────────────────────────── */
function initNavbar() {
  const navbar = document.querySelector('.navbar');
  const toggle = document.querySelector('.navbar-toggle');
  const links  = document.querySelector('.navbar-links');

  if (navbar) {
    window.addEventListener('scroll', () => {
      navbar.classList.toggle('scrolled', window.scrollY > 20);
    });
  }
  if (toggle && links) {
    toggle.addEventListener('click', () => {
      links.classList.toggle('open');
      const spans = toggle.querySelectorAll('span');
      const isOpen = links.classList.contains('open');
      if (spans[0]) spans[0].style.transform = isOpen ? 'rotate(45deg) translate(5px, 5px)' : '';
      if (spans[1]) spans[1].style.opacity   = isOpen ? '0' : '1';
      if (spans[2]) spans[2].style.transform = isOpen ? 'rotate(-45deg) translate(5px, -5px)' : '';
    });
  }

  // Set active link
  const path = window.location.pathname;
  document.querySelectorAll('.navbar-links a').forEach(a => {
    a.classList.toggle('active', a.getAttribute('href') === path);
  });
}

/* ── Match Score Bars ────────────────────────────────── */
function initMatchBars() {
  const bars = document.querySelectorAll('.rec-match-fill');
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const el = entry.target;
        const target = el.dataset.width || el.style.width;
        el.style.width = '0%';
        requestAnimationFrame(() => {
          setTimeout(() => { el.style.width = target; }, 80);
        });
        observer.unobserve(el);
      }
    });
  }, { threshold: 0.3 });
  bars.forEach(b => observer.observe(b));
}

/* ── Animated Counters ───────────────────────────────── */
function initCounters() {
  const counters = document.querySelectorAll('.counter');
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        animateCounter(entry.target);
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.5 });
  counters.forEach(c => observer.observe(c));
}

function animateCounter(el) {
  const target = parseFloat(el.dataset.target || el.textContent);
  const suffix = el.dataset.suffix || '';
  const prefix = el.dataset.prefix || '';
  const decimals = el.dataset.decimals ? parseInt(el.dataset.decimals) : 0;
  const duration = 1500;
  const start = performance.now();

  function step(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = eased * target;
    el.textContent = prefix + current.toFixed(decimals) + suffix;
    if (progress < 1) requestAnimationFrame(step);
    else el.textContent = prefix + target.toFixed(decimals) + suffix;
  }
  requestAnimationFrame(step);
}

/* ── Loading Overlay ─────────────────────────────────── */
function initLoadingOverlay() {
  const form = document.querySelector('.recommendation-form');
  const overlay = document.getElementById('loadingOverlay');
  if (form && overlay) {
    form.addEventListener('submit', () => {
      overlay.classList.add('active');
    });
  }
}

/* ── Tooltips ────────────────────────────────────────── */
function initTooltips() {
  document.querySelectorAll('[data-tooltip]').forEach(el => {
    el.style.position = 'relative';
    el.addEventListener('mouseenter', () => {
      const tip = document.createElement('div');
      tip.className = 'tooltip-bubble';
      tip.textContent = el.dataset.tooltip;
      tip.style.cssText = `
        position:absolute;bottom:calc(100% + 8px);left:50%;
        transform:translateX(-50%);
        background:#1e293b;color:white;
        padding:6px 12px;border-radius:8px;font-size:.78rem;
        white-space:nowrap;z-index:9999;pointer-events:none;
        box-shadow:0 4px 12px rgba(0,0,0,.15);
      `;
      el.appendChild(tip);
    });
    el.addEventListener('mouseleave', () => {
      el.querySelector('.tooltip-bubble')?.remove();
    });
  });
}

/* ── Plotly Charts ────────────────────────────────────── */
function initPlotlyCharts() {
  document.querySelectorAll('.plotly-chart').forEach(el => {
    const rawJson = el.dataset.chart;
    if (!rawJson) return;
    try {
      const fig = JSON.parse(rawJson);
      Plotly.newPlot(el, fig.data, {
        ...fig.layout,
        font: { family: 'DM Sans, system-ui, sans-serif' },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        margin: fig.layout?.margin || { l: 10, r: 10, t: 50, b: 10 },
      }, { responsive: true, displayModeBar: false });
    } catch (e) {
      el.innerHTML = '<div class="text-muted text-center p-4">Chart unavailable</div>';
    }
  });
}

/* ── Rating Slider ───────────────────────────────────── */
function initRatingSlider() {
  const slider = document.getElementById('minRating');
  const label  = document.getElementById('ratingLabel');
  if (slider && label) {
    function update() {
      const v = parseFloat(slider.value);
      label.textContent = v.toFixed(1) + ' ★';
      label.style.color = v >= 4.5 ? '#10b981' : v >= 4.0 ? '#0ea5e9' : '#f59e0b';
    }
    slider.addEventListener('input', update);
    update();
  }
}

/* ── Search Suggestions ──────────────────────────────── */
function initSearchSuggestions() {
  const destInput = document.getElementById('attractionName');
  if (!destInput) return;

  let timeout;
  let dropdown;

  destInput.addEventListener('input', () => {
    clearTimeout(timeout);
    const val = destInput.value.trim();
    if (val.length < 2) { removeDropdown(); return; }
    timeout = setTimeout(() => fetchSuggestions(val), 300);
  });

  destInput.addEventListener('blur', () => {
    setTimeout(removeDropdown, 200);
  });

  function fetchSuggestions(q) {
    fetch(`/api/suggest?q=${encodeURIComponent(q)}`)
      .then(r => r.json())
      .then(data => showDropdown(data.suggestions || []))
      .catch(() => {});
  }

  function showDropdown(items) {
    removeDropdown();
    if (!items.length) return;
    dropdown = document.createElement('div');
    dropdown.className = 'suggestions-dropdown';
    dropdown.style.cssText = `
      position:absolute;top:100%;left:0;right:0;
      background:white;border:1.5px solid #e2e8f0;
      border-radius:14px;box-shadow:0 8px 24px rgba(0,0,0,.1);
      z-index:500;max-height:240px;overflow-y:auto;margin-top:4px;
    `;
    items.forEach(item => {
      const div = document.createElement('div');
      div.style.cssText = 'padding:10px 16px;cursor:pointer;font-size:.9rem;transition:background .15s;';
      div.innerHTML = `<span style="margin-right:8px">${item.icon || '📍'}</span>${item.name}
        <span style="font-size:.75rem;color:#94a3b8;margin-left:8px">${item.category || ''}</span>`;
      div.addEventListener('mouseenter', () => div.style.background = '#f0f9ff');
      div.addEventListener('mouseleave', () => div.style.background = 'white');
      div.addEventListener('click', () => {
        destInput.value = item.name;
        removeDropdown();
      });
      dropdown.appendChild(div);
    });
    const wrapper = destInput.closest('.search-wrapper') || destInput.parentElement;
    wrapper.style.position = 'relative';
    wrapper.appendChild(dropdown);
  }

  function removeDropdown() {
    dropdown?.remove();
    dropdown = null;
  }
}

/* ── Scroll Reveal ───────────────────────────────────── */
function initScrollReveal() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

  document.querySelectorAll('.scroll-reveal').forEach((el, i) => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = `opacity .5s ${i * 0.05}s ease, transform .5s ${i * 0.05}s ease`;
    observer.observe(el);
  });
}

/* ── Destination Search Filter ───────────────────────── */
function filterDestinations() {
  const search = document.getElementById('destSearch')?.value.toLowerCase() || '';
  const cards  = document.querySelectorAll('.dest-card-wrapper');
  let visible  = 0;
  cards.forEach(card => {
    const text = card.textContent.toLowerCase();
    const show = !search || text.includes(search);
    card.style.display = show ? '' : 'none';
    if (show) visible++;
  });
  const countEl = document.getElementById('destCount');
  if (countEl) countEl.textContent = visible;
}

/* ── Copy to clipboard ────────────────────────────────── */
function copyToClipboard(text, btn) {
  navigator.clipboard.writeText(text).then(() => {
    const orig = btn.textContent;
    btn.textContent = '✓ Copied!';
    btn.style.color = '#10b981';
    setTimeout(() => { btn.textContent = orig; btn.style.color = ''; }, 2000);
  });
}

/* ── Share destination ────────────────────────────────── */
function shareDestination(name, url) {
  if (navigator.share) {
    navigator.share({ title: name + ' — TourMate AI', url: url });
  } else {
    copyToClipboard(url, event.target);
  }
}

/* ── API: Suggest endpoint (for autocomplete) ─────────── */
// Registered on Flask as /api/suggest via route (see app.py addition)

/* ── Insight Expand Toggle ────────────────────────────── */
function toggleInsight(id) {
  const panel = document.getElementById(id);
  if (!panel) return;
  const isHidden = panel.style.display === 'none' || !panel.style.display;
  panel.style.display = isHidden ? 'block' : 'none';
  const btn = document.querySelector(`[onclick="toggleInsight('${id}')"]`);
  if (btn) btn.textContent = isHidden ? '▲ Hide AI Insight' : '✨ Show AI Insight';
}
