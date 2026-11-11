// The script for the standalone /view pages. The console app never loads it.
// These pages load nothing else, so this file keeps its own icons, zoom, and small
// helpers, and it may import only the files below.

import { copyToClipboard } from "./clipboard.js";
import { initAudioCard } from "./audio-card.js";
import { mountPdfViewer } from "./pdf-viewer.js";

const _POLL_MS = 1000;
const _ZOOM_MIN = 0.1;
const _ZOOM_MAX = 8;
const _ZOOM_STEP = 0.25;
const _THUMB_RETRY_MS = 2000;
const _THUMB_RETRY_MAX = 10;
const _OFFICE_EXTS = new Set(["doc", "docx", "ppt", "pptx"]);
const _SHEET_EXTS = new Set(["xlsx", "xls"]);

// The moon shows in light mode and the sun shows in dark mode.
const _MOON = '<svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M13.5 9.5A5.5 5.5 0 0 1 6.5 2.5a5.5 5.5 0 1 0 7 7Z"/></svg>';
const _SUN = '<svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><circle cx="8" cy="8" r="3.2"/><path d="M8 1.5v1.8M8 12.7v1.8M1.5 8h1.8M12.7 8h1.8M3.4 3.4l1.3 1.3M11.3 11.3l1.3 1.3M12.6 3.4l-1.3 1.3M4.7 11.3l-1.3 1.3"/></svg>';

// The icons the server asks for by name.
const _ICON_ALERT = '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="8" cy="8" r="6"/><path d="M8 5v4M8 11.2v.3"/></svg>';
const _ICON_BOX = '<svg width="22" height="22" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"><path d="m8 2 5.5 3.2v5.6L8 14l-5.5-3.2V5.2L8 2Z"/><path d="M2.5 5.2 8 8.4l5.5-3.2M8 8.4V14"/></svg>';
const _ICON_DOWNLOAD = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M8 2.5v7M5 6.5l3 3 3-3M3 12.5h10"/></svg>';
// These pages load no app.css, so each of these icons carries its own size.
const _ICON_REFRESH = '<svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M13 8a5 5 0 1 1-1.7-3.7"/><path d="M13 2.5v3h-3"/></svg>';
const _ICON_COPY = '<svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><rect x="5.5" y="5.5" width="8" height="8" rx="1.5"/><path d="M10.5 5.5v-2a1 1 0 0 0-1-1h-6a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2"/></svg>';
// The presentation toolbar's open-in-new-tab action. Its size comes from the button it sits in.
const _ICON_OPEN = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M6.5 3.5H3v9.5h9.5V9.5"/><path d="M9.5 2.5h4v4M13 3 8.5 7.5"/></svg>';

const _ICONS = { alert: _ICON_ALERT, box: _ICON_BOX, download: _ICON_DOWNLOAD, open: _ICON_OPEN };

const _CONVERTING = {
  office: ["Converting with LibreOffice…",
    "Large documents can take up to a minute. You can leave — the result is cached if it finishes."],
  sheet: ["Reading with the built-in sheet reader…",
    "This usually finishes in a moment. You can leave — the result is cached if it finishes."],
};

function _esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function _fmtDur(ms) {
  if (!Number.isFinite(ms) || ms < 0) ms = 0;
  if (ms < 10000) return (ms / 1000).toFixed(1) + "s";
  const s = Math.floor(ms / 1000);
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (d > 0) return d + "d " + h + "h";
  if (h > 0) return h + "h " + m + "m";
  if (m > 0) return m + "m " + sec + "s";
  return sec + "s";
}

function _extOf(name) {
  const i = name.lastIndexOf(".");
  return i > 0 ? name.slice(i + 1).toLowerCase() : "";
}

function _baseName(path) {
  const i = path.lastIndexOf("/");
  return i >= 0 ? path.slice(i + 1) : path;
}

function _plural(n, word) {
  return (n == null ? "" : n.toLocaleString("en-US")) + " " + word + (n === 1 ? "" : "s");
}

function _encPath(path) {
  return path.split("/").map(encodeURIComponent).join("/");
}

// The file path this page shows. It is null when the URL names no file.
function _viewPath() {
  if (!location.pathname.startsWith("/view/")) return null;
  try {
    return decodeURIComponent(location.pathname.slice("/view/".length));
  } catch {
    return null; // A URL we cannot read names no file.
  }
}

// The file path one part of the page shows. Its own data-path, then the nearest
// ancestor's, then the URL. On a presentation the URL names no file.
function _filePath(el) {
  if (el.dataset.path != null) return el.dataset.path;
  let node = el.parentElement;
  while (node && node !== document.body) {
    if (node.dataset.path != null) return node.dataset.path;
    node = node.parentElement;
  }
  return _viewPath();
}

// True when this page shows a run result. The same script runs both kinds of page.
const _RESULT = location.pathname.startsWith("/view/result/");

// True when this page is a presentation. Its URL names an id, not a file.
const _PRESENTATION = location.pathname.startsWith("/view/presentation/");
async function _parse(res) {
  let body = null;
  try { body = await res.json(); } catch { body = null; }
  if (!res.ok) {
    const err = new Error((body && body.error) || res.statusText);
    err.status = res.status;
    throw err;
  }
  return body;
}

function _getJSON(url) {
  return fetch(url, { headers: { Accept: "application/json" } }).then(_parse);
}

function _postJSON(url) {
  return fetch(url, { method: "POST", headers: { Accept: "application/json" } }).then(_parse);
}

// ============ theme ============
// The theme is saved under the same name the console uses, as the pick paired
// with the browser setting it was made under.

function _browserScheme() {
  try {
    return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  } catch {
    return "light";
  }
}

function _currentTheme() {
  return document.documentElement.getAttribute("data-theme") || "light";
}

function _initTheme() {
  // The page already carries a theme before this runs. A click saves the new pick.
  const sync = () => {
    const dark = _currentTheme() === "dark";
    const svg = dark ? _SUN : _MOON;
    const label = dark ? "Switch to light theme" : "Switch to dark theme";
    document.querySelectorAll("[data-vp-theme]").forEach((b) => {
      b.innerHTML = svg;
      b.title = label;
      b.setAttribute("aria-label", label);
    });
  };
  const apply = (theme) => {
    document.documentElement.setAttribute("data-theme", theme);
    sync();
  };
  sync();
  document.addEventListener("click", (ev) => {
    const t = ev.target;
    if (!t || !t.closest || !t.closest("[data-vp-theme]")) return;
    const theme = _currentTheme() === "dark" ? "light" : "dark";
    apply(theme);
    try {
      localStorage.setItem("agent-console-theme", JSON.stringify({ theme, scheme: _browserScheme() }));
    } catch { /* saving can fail */ }
  });
  // A change in the browser setting takes the page back and clears the saved pick.
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    try { localStorage.removeItem("agent-console-theme"); } catch { /* saving can fail */ }
    apply(_browserScheme());
  });
}

// ============ icons and top bar buttons ============

function _initIcons() {
  document.querySelectorAll("[data-vp-icon]").forEach((el) => {
    const svg = _ICONS[el.getAttribute("data-vp-icon")];
    if (svg) el.innerHTML = svg;
  });
}

function _initActions() {
  // Buttons made later still work.
  document.addEventListener("click", (ev) => {
    const t = ev.target;
    if (!t || !t.closest) return;
    if (t.closest("[data-vp-back]")) {
      history.back();
      return;
    }
    const retry = t.closest("[data-vp-retry]");
    if (retry) {
      _retry(retry);
      return;
    }
    const raw = t.closest("[data-vp-copyraw]");
    if (raw) _copyRaw(raw.getAttribute("data-vp-copyraw"));
  });
}

async function _copyRaw(path) {
  try {
    const res = await fetch("/dav/" + _encPath(path));
    if (!res.ok) return;
    await copyToClipboard(await res.text());
  } catch {
    // This page says nothing when a copy fails.
  }
}

// ============ office well top bar ============

// The size and shape of an icon button. These pages have no app.css to set it.
const _ICON_BTN_STYLE =
  "width:28px;height:28px;display:inline-flex;align-items:center;justify-content:center;flex:none";

function _chip(label, dashed) {
  return '<span class="chip' + (dashed ? " dashed" : "") + '">' + _esc(label) + "</span>";
}

function _retryBtn(path, title) {
  return '<button class="icon-btn" type="button" style="' + _ICON_BTN_STYLE + '" title="' + _esc(title)
    + '" data-vp-retry="' + _esc("/api/view/" + _encPath(path) + "/retry") + '">' + _ICON_REFRESH + "</button>";
}

function _copyRawBtn(path) {
  return '<button class="icon-btn" type="button" style="' + _ICON_BTN_STYLE + '" title="Copy raw file"'
    + ' data-vp-copyraw="' + _esc(path) + '">' + _ICON_COPY + "</button>";
}

// The bar above a converted file. A sheet also gets a column count and a copy button.
function _officeTopbar(d) {
  const ext = _extOf(d.name || "");
  if (d.mode === "doc" && _OFFICE_EXTS.has(ext)) {
    const bar = document.createElement("header");
    bar.className = "well-topbar";
    bar.innerHTML = _chip(ext) + (d.provenance ? _chip(d.provenance, true) : "")
      + _retryBtn(d.path, "Re-convert");
    return bar;
  }
  if (d.mode === "table" && _SHEET_EXTS.has(ext)) {
    const bar = document.createElement("header");
    bar.className = "well-topbar";
    bar.innerHTML = _chip(ext) + (d.provenance ? _chip(d.provenance, true) : "")
      + _chip(_plural(d.column_count, "column"))
      + '<span style="display:flex;align-items:center;gap:8px">' + _copyRawBtn(d.path)
      + _retryBtn(d.path, "Re-read") + "</span>";
    return bar;
  }
  return null;
}

async function _initOfficeTopBars() {
  if (_PRESENTATION) return; // The body's toolbar is the page's only top bar.
  const path = _viewPath();
  if (path == null) return;
  const ext = _extOf(_baseName(path));
  if (!_OFFICE_EXTS.has(ext) && !_SHEET_EXTS.has(ext)) return;
  const well = document.querySelector(".vp-well-doc, .vp-well-table");
  if (!well) return;
  let d;
  try {
    d = await _getJSON("/api/view/" + _encPath(path));
  } catch {
    return; // The well still shows without a top bar.
  }
  const bar = _officeTopbar(d);
  if (bar && well.isConnected) well.prepend(bar);
}

// ============ converting state ============

function _initConvertingEl(el, path) {
  const sheet = _SHEET_EXTS.has(_extOf(_baseName(path)));
  const [label, note] = _CONVERTING[sheet ? "sheet" : "office"];
  el.innerHTML =
    '<div class="cv-spin"></div>' +
    '<div class="cv-label">' + label + "</div>" +
    '<div class="cv-elapsed">0.0s</div>' +
    '<div class="cv-note">' + note + "</div>";
  const enc = _encPath(path);
  const elapsed = el.querySelector(".cv-elapsed");
  const t0 = Date.now();
  let stopped = false;
  let timer = null;
  const ticker = setInterval(() => {
    elapsed.textContent = _fmtDur(Date.now() - t0);
  }, _POLL_MS);
  const stop = () => {
    stopped = true;
    clearInterval(ticker);
    if (timer) clearInterval(timer);
  };
  const tick = async () => {
    // The descriptor itself is the signal: keep checking until it stops converting.
    let next;
    try {
      next = await _getJSON("/api/view/" + enc);
    } catch (err) {
      // The file went away while it was being converted.
      if (err && err.status === 404) { stop(); _blockGone(el); return; }
      return; // Keep checking when the network hiccups.
    }
    if (stopped) return;
    if (next.mode === "converting") return; // Still working, keep checking.
    stop();
    // Do nothing when this part of the page is already gone.
    if (!el.isConnected) return;
    _swapIn(el, next, path);
  };
  timer = setInterval(tick, _POLL_MS);
}

function _initConverting() {
  document.querySelectorAll(".vp-converting").forEach((el) => {
    const path = _filePath(el);
    if (path == null) return;
    _initConvertingEl(el, path);
  });
}

// Asks the server to convert the file again, then shows what comes back.
async function _retry(btn) {
  let next;
  try {
    next = await _postJSON(btn.getAttribute("data-vp-retry"));
  } catch {
    return; // Leave the page as it is, the button still works.
  }
  const well = btn.closest(".vp-convert-failed, .vp-well");
  const path = well ? _filePath(well) : null;
  if (!well || path == null || !well.isConnected) return;
  _swapIn(well, next, path);
}

// ============ redraw after a conversion ============

// The markup comes from the server and is trusted, so it goes on the page as is.
function _appendFragment(container, fragment) {
  if (!fragment) return;
  const el = document.createElement("div");
  el.innerHTML = fragment;
  while (el.firstChild) container.appendChild(el.firstChild);
}

// Builds the markup for one file. It must match what the server builds.
function _renderBody(d) {
  const mode = d.mode;
  if (mode === "doc" || mode === "code" || mode === "table" || mode === "plain") {
    const well = document.createElement("div");
    well.className = "vp-well vp-well-" + mode;
    const bar = _officeTopbar(d);
    if (bar) well.appendChild(bar);
    _appendFragment(well, d.fragment);
    return well;
  }
  if (mode === "raw") {
    const well = document.createElement("div");
    well.className = "vp-well vp-well-raw";
    if (d.fragment != null) _appendFragment(well, d.fragment);
    else well.textContent = d.note || "";
    return well;
  }
  if (mode === "native") {
    if (d.kind === "image") {
      const stage = document.createElement("div");
      stage.className = "img-stage vp-native";
      const img = document.createElement("img");
      img.alt = "";
      img.dataset.vpZoom = "";
      img.src = d.url;
      stage.appendChild(img);
      return stage;
    }
    const wrap = document.createElement("div");
    wrap.className = "vp-native";
    const media = document.createElement(d.kind === "video" ? "video" : "audio");
    media.setAttribute("controls", "");
    media.src = d.url;
    wrap.appendChild(media);
    return wrap;
  }
  if (mode === "frame") {
    const wrap = document.createElement("div");
    wrap.className = "vp-native";
    if (d.kind === "pdf" || d.kind === "office_ppt") {
      // The pdf viewer mounts into this container.
      const host = document.createElement("div");
      host.className = "vp-frame vp-pdf";
      host.dataset.pdfSrc = d.url;
      wrap.appendChild(host);
      return wrap;
    }
    const ifr = document.createElement("iframe");
    ifr.className = "vp-frame";
    ifr.src = d.url;
    if (d.kind === "html") ifr.setAttribute("sandbox", "allow-same-origin");
    wrap.appendChild(ifr);
    return wrap;
  }
  if (mode === "converting") {
    const el = document.createElement("div");
    el.className = "vp-converting";
    return el;
  }
  if (mode === "convert_failed") {
    const el = document.createElement("div");
    el.className = "vp-convert-failed";
    el.innerHTML =
      "<p>Couldn't convert this file</p>" +
      '<p class="vp-err">' + _esc(d.error || "") + "</p>" +
      '<div class="vp-state-actions">' +
      '<button class="btn ghost" type="button" data-vp-retry="'
      + _esc("/api/view/" + _encPath(d.path) + "/retry") + '">Retry</button>' +
      '<a class="btn ghost" href="/dav/dl?path=' + _encPath(d.path) + '">Download</a>' +
      "</div>";
    return el;
  }
  const el = document.createElement("div");
  el.className = "vp-unsupported";
  const card = document.createElement("div");
  card.className = "vp-state-card";
  card.innerHTML = '<span class="vp-state-ic">' + _ICON_BOX + "</span>";
  const title = document.createElement("h1");
  title.className = "vp-state-title";
  title.textContent = d.name || "";
  card.appendChild(title);
  const sub = document.createElement("p");
  sub.className = "vp-state-sub";
  sub.textContent = "This file type can't be previewed — download it to view.";
  card.appendChild(sub);
  const actions = document.createElement("div");
  actions.className = "vp-state-actions";
  actions.innerHTML =
    '<a class="btn ghost" href="/dav/dl?path=' + _encPath(d.path) + '">Download</a>' +
    '<a class="btn ghost" data-vp-back>Go back</a>';
  card.appendChild(actions);
  el.appendChild(card);
  return el;
}

function _swapIn(el, next, path) {
  const section = el.closest(".vp-presentation-body");
  if (section) {
    const body = _renderPresentationBody(next, section.dataset.path, section.dataset.dlHref);
    if (body == null) {
      section.remove();
      return;
    }
    section.replaceWith(body);
    _initIcons(); // the toolbar's data-vp-icon spans are filled like on load
    const converting = body.querySelector(".vp-converting");
    if (converting) _initConvertingEl(converting, path);
    const audio = body.querySelector(".audio-card[data-src]");
    if (audio) initAudioCard(audio, audio.dataset.src, audio.dataset.name);
    _initPdfs(body);
    return;
  }
  const body = _renderBody(next);
  if (body == null) {
    el.remove();
    return;
  }
  el.replaceWith(body);
  if (body.classList.contains("vp-converting")) _initConvertingEl(body, path);
  const audio = body.querySelector(".audio-card[data-src]");
  if (audio) initAudioCard(audio, audio.dataset.src, audio.dataset.name);
  const img = body.querySelector("img[data-vp-zoom]");
  if (img) initImageZoom(img, img.parentElement);
  _initPdfs(body);
}

// Mounts the inline PDF containers the swapped-in markup carries.
function _initPdfs(root) {
  root.querySelectorAll(".vp-pdf[data-pdf-src]").forEach((host) =>
    mountPdfViewer(host, host.dataset.pdfSrc));
}

// The toolbar row of the presentation body. It must match what the server builds.
function _presentationToolbar(d, dlHref) {
  return '<div class="well-topbar">'
    + '<span class="chip">' + _esc(d.kind_label || "") + "</span>"
    + '<span class="well-note">' + _esc(_bytes(d.size)) + "</span>"
    + '<span class="well-actions">'
    + '<a class="vp-icon-btn" href="/view/' + _encPath(d.path) + '" target="_blank" '
    + 'title="Open full-screen" aria-label="Open full-screen"><span data-vp-icon="open"></span></a>'
    + '<a class="vp-icon-btn" href="' + _esc(dlHref) + '" title="Download" '
    + 'aria-label="Download"><span data-vp-icon="download"></span></a>'
    + "</span></div>";
}

// Builds the presentation body section. It must match what the server builds.
function _renderPresentationBody(d, path, dlHref) {
  if (d.mode === "unsupported") return null; // The file stays a card in the grid.
  let inner;
  let cls = "vp-well vp-presentation-body";
  if (d.mode === "doc" || d.mode === "code" || d.mode === "table" || d.mode === "plain") {
    cls += " vp-well-" + d.mode;
    inner = d.fragment || "";
  } else if (d.mode === "raw") {
    cls += " vp-well-raw";
    inner = d.fragment != null ? d.fragment : _esc(d.note || "");
  } else if (d.mode === "native") {
    if (d.kind === "image") inner = '<div class="img-stage"><img src="' + _esc(d.url) + '" alt=""></div>';
    else if (d.kind === "video") inner = '<div class="media-stage"><video controls src="' + _esc(d.url) + '"></video></div>';
    else inner = '<div class="audio-card" data-src="' + _esc(d.url) + '" data-name="' + _esc(d.name || "") + '"></div>';
  } else if (d.mode === "frame") {
    if (d.kind === "pdf" || d.kind === "office_ppt") {
      // The pdf viewer mounts into this container.
      inner = '<div class="vp-frame vp-pdf" data-pdf-src="' + _esc(d.url) + '"></div>';
    } else {
      inner = '<iframe class="frame-body" src="' + _esc(d.url) + '"'
        + (d.kind === "html" ? ' sandbox="allow-same-origin"' : "") + "></iframe>";
    }
  } else if (d.mode === "converting") {
    inner = '<div class="vp-converting"></div>';
  } else if (d.mode === "convert_failed") {
    inner = '<div class="vp-convert-failed"><p>Couldn\'t convert this file</p>'
      + '<p class="vp-err">' + _esc(d.error || "") + "</p>"
      + '<div class="vp-state-actions">'
      + '<button class="btn ghost" type="button" data-vp-retry="'
      + _esc("/api/view/" + _encPath(d.path) + "/retry") + '">Retry</button>'
      + '<a class="btn ghost" href="' + _esc(dlHref) + '">Download</a></div></div>';
  } else {
    return null;
  }
  const section = document.createElement("div");
  section.className = cls;
  section.innerHTML = _presentationToolbar(d, dlHref) + inner;
  section.dataset.path = path;
  section.dataset.dlHref = dlHref;
  return section;
}

// The body section when its file is gone mid-session. Matches the server state.
function _bodyGoneEl(path) {
  const section = document.createElement("div");
  section.className = "vp-well vp-presentation-body";
  if (path != null) section.dataset.path = path;
  section.innerHTML =
    '<div class="vp-body-gone">' +
    '<span class="vp-state-ic">' + _ICON_BOX + "</span>" +
    '<p class="vp-body-gone-title">The message content is no longer available</p>' +
    '<p class="vp-body-gone-sub">Files are retained for 30 days, then removed.</p>' +
    "</div>";
  return section;
}

// Fills in the audio card when the body is an audio file.
function _initPresentationBody() {
  document.querySelectorAll(".vp-presentation-body .audio-card[data-src]").forEach((a) =>
    initAudioCard(a, a.dataset.src, a.dataset.name));
}

// A file that is gone mid-session shows its GC'd state in place of its
// preview. Outside a feed block the whole page goes to the expired page.
function _blockGone(el) {
  const section = el.closest(".vp-presentation-body");
  if (section) {
    section.replaceWith(_bodyGoneEl(_filePath(section)));
    return;
  }
  const block = el.closest(".vp-file-block");
  if (!block) {
    _expiredInPlace();
    return;
  }
  block.classList.remove("vp-file-present");
  el.replaceWith(_miniStateEl());
}

function _miniStateEl() {
  const wrap = document.createElement("div");
  wrap.innerHTML =
    '<div class="mini-state">' +
    '<span class="vp-state-ic">' + _ICON_ALERT + "</span>" +
    '<p class="mini-state-title">No longer in the resource.</p>' +
    '<p class="vp-state-sub">Garbage-collected (30-day retention).</p>' +
    "</div>";
  return wrap.firstChild;
}

// Shows the expired page in place when the file is gone.
function _expiredInPlace() {
  const main = document.querySelector("main.vp-main");
  if (!main) return;
  const expired = document.createElement("main");
  expired.className = "vp-main vp-expired";
  expired.innerHTML =
    '<div class="vp-expired-inner">' +
    '<span class="vp-expired-ic">' + _ICON_ALERT + "</span>" +
    '<h1 class="vp-expired-title">This link is no longer valid</h1>' +
    '<p class="vp-expired-sub">The file may have been garbage-collected — files are retained '
    + "for 30 days and then removed. Ask the sender for a fresh link.</p>" +
    '<button class="btn ghost" type="button" data-vp-back>Go back</button>' +
    "</div>";
  main.replaceWith(expired);
  // The expired page has no download button.
  const soloCluster = document.querySelector(".vp-solo-cluster");
  if (soloCluster) soloCluster.remove();
}

// ============ image zoom ============

// The zoom jobs that run when the window is resized.
const _zoomResizes = new Set();
let _resizeWired = false;

function initImageZoom(img, stage) {
  let frame = img.parentElement;
  if (!frame || !frame.classList.contains("img-frame")) {
    frame = document.createElement("div");
    frame.className = "img-frame";
    img.parentNode.insertBefore(frame, img);
    frame.appendChild(img);
  }
  let scale = 1;
  let fitScale = 1;
  let mode = "fit";
  let ready = false;

  const cluster = document.createElement("div");
  cluster.className = "zoom-cluster";
  cluster.innerHTML =
    '<button type="button" data-zoom="fit" title="Fit to stage">Fit</button>' +
    '<button type="button" data-zoom="one" title="Actual size">1:1</button>' +
    '<span class="zoom-sep"></span>' +
    '<button type="button" data-zoom="out" title="Zoom out">−</button>' +
    '<span class="zoom-pct">100%</span>' +
    '<button type="button" data-zoom="in" title="Zoom in">+</button>';
  stage.appendChild(cluster);
  const pct = cluster.querySelector(".zoom-pct");
  const btnFit = cluster.querySelector('[data-zoom="fit"]');
  const btnOne = cluster.querySelector('[data-zoom="one"]');

  const clamp = (s) => Math.min(_ZOOM_MAX, Math.max(_ZOOM_MIN, s));

  function computeFit() {
    fitScale = clamp(Math.min(stage.clientWidth / img.naturalWidth,
      stage.clientHeight / img.naturalHeight, 1));
  }

  function apply() {
    // The wrapper is sized to the zoomed image.
    frame.style.width = Math.round(img.naturalWidth * scale) + "px";
    frame.style.height = Math.round(img.naturalHeight * scale) + "px";
    // The image is centered while it fits. Once it is bigger than the stage it is
    // pinned to the corner, or the top left cannot be scrolled to.
    const overflows = (img.naturalWidth * scale > stage.clientWidth + 1) ||
      (img.naturalHeight * scale > stage.clientHeight + 1);
    frame.style.margin = overflows ? "0" : "auto";
    pct.textContent = Math.round(scale * 100) + "%";
    btnFit.classList.toggle("active", mode === "fit");
    btnOne.classList.toggle("active", mode === "one");
  }

  function setScale(s, m) {
    scale = clamp(s);
    mode = m;
    apply();
  }

  function onReady() {
    if (ready || !img.naturalWidth || !img.naturalHeight) return;
    ready = true;
    computeFit();
    setScale(fitScale, "fit");
  }

  img.addEventListener("load", onReady);
  if (img.complete) onReady();

  cluster.addEventListener("click", (ev) => {
    const btn = ev.target.closest("button");
    if (!btn || !ready) return;
    const z = btn.dataset.zoom;
    if (z === "fit") {
      computeFit();
      setScale(fitScale, "fit");
    } else if (z === "one") setScale(1, "one");
    else if (z === "in") setScale(scale + _ZOOM_STEP, "");
    else if (z === "out") setScale(scale - _ZOOM_STEP, "");
  });

  if (!_resizeWired) {
    _resizeWired = true;
    window.addEventListener("resize", () => {
      for (const fn of [..._zoomResizes]) fn();
    });
  }
  const onResize = () => {
    if (!stage.isConnected || !ready) {
      _zoomResizes.delete(onResize);
      return;
    }
    if (mode !== "fit") return;
    computeFit();
    setScale(fitScale, "fit");
  };
  _zoomResizes.add(onResize);
}

function _initZoom() {
  document.querySelectorAll("img[data-vp-zoom]").forEach((img) => {
    if (img.parentElement) initImageZoom(img, img.parentElement);
  });
}

// ============ card thumbnail retry ============

function _initPresentationThumbs() {
  const grid = document.querySelector(".thumb-grid");
  if (!grid) return;
  for (const card of grid.querySelectorAll(".thumb-card:not(.vp-card-gone)")) {
    const glyph = card.querySelector(":scope > .thumb-1610 > .thumb-glyph");
    if (!glyph) continue;
    // The card's path comes from its open link.
    const open = card.querySelector('a[href^="/view/"]');
    if (!open) continue;
    _retryThumb(glyph, open.getAttribute("href").slice("/view/".length));
  }
}

// Asks for the small preview again every two seconds, up to ten times.
// It stops for good once the file turns out to be missing.
function _retryThumb(glyph, rel) {
  const url = "/api/resources/thumb?path=" + encodeURIComponent(rel);
  let tries = 0;
  const timer = setInterval(async () => {
    tries += 1;
    let res;
    try {
      res = await fetch(url);
    } catch {
      if (tries >= _THUMB_RETRY_MAX) clearInterval(timer);
      return;
    }
    if (res.status === 404) {
      clearInterval(timer);
      return;
    }
    if (res.ok) {
      clearInterval(timer);
      if (glyph.isConnected) {
        const img = document.createElement("img");
        img.className = "thumb-img done";
        img.alt = "";
        img.src = url;
        glyph.replaceWith(img);
      }
      return;
    }
    if (tries >= _THUMB_RETRY_MAX) clearInterval(timer);
  }, _THUMB_RETRY_MS);
}

// ============ result mode ============
// The result page is body-only, so it only wires the theme, the icons, and
// the top bar actions.

if (_RESULT) {
  _initTheme();
  _initIcons();
  _initActions();
} else {
  _initTheme();
  _initIcons();
  _initActions();
  _initConverting();
  _initPresentationBody();
  _initOfficeTopBars();
  _initZoom();
  _initPresentationThumbs();
}
