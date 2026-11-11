const _MOON = '<svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M13.5 9.5A5.5 5.5 0 0 1 6.5 2.5a5.5 5.5 0 1 0 7 7Z"/></svg>';
const _SUN = '<svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><circle cx="8" cy="8" r="3.2"/><path d="M8 1.5v1.8M8 12.7v1.8M1.5 8h1.8M12.7 8h1.8M3.4 3.4l1.3 1.3M11.3 11.3l1.3 1.3M12.6 3.4l-1.3 1.3M4.7 11.3l-1.3 1.3"/></svg>';

// The browser's own light or dark setting.
function _browserScheme() {
  try {
    return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  } catch {
    return "light";
  }
}

export function currentTheme() {
  return document.documentElement.getAttribute("data-theme") || "light";
}

function _applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  syncGlyphs();
}

export function setTheme(theme) {
  _applyTheme(theme);
  try {
    localStorage.setItem("agent-console-theme", JSON.stringify({ theme, scheme: _browserScheme() }));
  } catch { /* saving can fail */ }
}

export function toggleTheme() {
  setTheme(currentTheme() === "dark" ? "light" : "dark");
}

export function syncGlyphs() {
  const dark = currentTheme() === "dark";
  const svg = dark ? _SUN : _MOON;
  const label = dark ? "Switch to light theme" : "Switch to dark theme";
  document.querySelectorAll("[data-theme-toggle]").forEach((b) => {
    b.innerHTML = svg;
    b.title = label;
    b.setAttribute("aria-label", label);
  });
}

let _schemeWatched = false;

export function initTheme() {
  syncGlyphs();
  // Theme buttons made later still work.
  document.addEventListener("click", (ev) => {
    const t = ev.target;
    if (t && t.closest && t.closest("[data-theme-toggle]")) toggleTheme();
  });
  if (_schemeWatched) return;
  _schemeWatched = true;
  // A change in the browser setting takes the page back and clears the saved pick.
  const mq = matchMedia("(prefers-color-scheme: dark)");
  mq.addEventListener("change", () => {
    try { localStorage.removeItem("agent-console-theme"); } catch { /* saving can fail */ }
    _applyTheme(_browserScheme());
  });
}
