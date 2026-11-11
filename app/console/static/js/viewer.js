import { getJSON, postJSON, encodePath, davHref, dlHref } from "./api.js";
import { esc, fmtDur, startTicker } from "./format.js";
import { icon, fileGlyph } from "./ui.js";
import { copyText, toast } from "./toast.js";
import { mountPdfViewer } from "./pdf-viewer.js";

const _POLL_MS = 1000;
const _CSV_WINDOW = 8;
const _ZOOM_MIN = 0.1;
const _ZOOM_MAX = 8;
const _ZOOM_STEP = 0.25;
const _DATA_KINDS = new Set(["json", "jsonl", "yaml", "xml"]);
const _SHEET_KINDS = new Set(["xlsx", "xls"]);
const _OFFICE_KINDS = new Set(["office_doc", "office_ppt", "xlsx", "xls"]);

// Each file path and the box that shows it right now.
const _wells = new Map();
// The zoom jobs that run when the window is resized.
const _zoomResizes = new Set();
let _resizeWired = false;
let _renderSeq = 0;

export function renderView(descriptor, container, onDescriptor) {
  container.replaceChildren();
  container._viewToken = ++_renderSeq;
  _registerWell(descriptor.path, container);
  const build = _BUILDERS[descriptor.mode];
  if (!build) {
    _unsupported(descriptor, container);
    return;
  }
  build(descriptor, container, onDescriptor);
}

export function showWellSkeleton(container) {
  container.replaceChildren();
  const el = document.createElement("div");
  el.className = "well-skeleton";
  container.appendChild(el);
}

export async function retryConvert(path) {
  const well = _wells.get(path);
  try {
    const descriptor = await postJSON("/api/view/" + encodePath(path) + "/retry");
    if (well && well.isConnected) renderView(descriptor, well);
  } catch (err) {
    toast((err && err.message) || "Retry failed", { error: true });
  }
}

export async function copyRaw(path) {
  try {
    const res = await fetch(davHref(path));
    if (!res.ok) throw new Error("HTTP " + res.status);
    copyText(await res.text(), "Raw file copied");
  } catch {
    toast("Copy failed", { error: true });
  }
}

const _BUILDERS = {
  doc: _doc,
  code: _code,
  plain: _plain,
  table: _table,
  native: _native,
  frame: _frame,
  raw: _raw,
  unsupported: _unsupported,
  converting: _converting,
  convert_failed: _convertFailed,
};

function _registerWell(path, container) {
  for (const [p, c] of _wells) {
    if (p === path || c === container) _wells.delete(p);
  }
  if (path) _wells.set(path, container);
}

// The markup comes from the server and is trusted, so it goes on the page as is.
function _fragment(container, fragment) {
  if (!fragment) return;
  const el = document.createElement("div");
  el.innerHTML = fragment;
  while (el.firstChild) container.appendChild(el.firstChild);
}

function _chip(label, dashed) {
  return '<span class="chip' + (dashed ? " dashed" : "") + '">' + esc(label) + "</span>";
}

function _note(label) {
  return '<span class="well-note">' + esc(label) + "</span>";
}

function _iconBtn(name, title, id) {
  return '<button class="icon-btn" type="button" title="' + esc(title) + '" id="' + id + '">' + icon(name) + "</button>";
}

function _download(d) {
  return '<a class="btn ghost" href="' + esc(dlHref(d.path)) + '">Download</a>';
}

// The right end of a top bar. It only sits right when it is the last thing in the bar.
function _right(html) {
  return '<span style="display:flex;align-items:center;gap:8px">' + html + "</span>";
}

function _topbar(html) {
  const bar = document.createElement("header");
  bar.className = "well-topbar";
  bar.innerHTML = html;
  return bar;
}

function _extOf(name) {
  const i = name.lastIndexOf(".");
  return i > 0 ? name.slice(i + 1).toLowerCase() : "";
}

function _plural(n, word) {
  return (n == null ? "" : n.toLocaleString("en-US")) + " " + word + (n === 1 ? "" : "s");
}

function _officeBar(d, c) {
  if (!_OFFICE_KINDS.has(d.kind)) return;
  c.appendChild(_topbar(
    _chip(_extOf(d.name)) +
    (d.provenance ? _chip(d.provenance, true) : "") +
    _iconBtn("refresh", "Re-convert", "wv-refresh")));
  c.querySelector("#wv-refresh").addEventListener("click", () => retryConvert(d.path));
}

function _doc(d, c) {
  _officeBar(d, c);
  _fragment(c, d.fragment);
}

function _plain(d, c) {
  _fragment(c, d.fragment);
}

function _code(d, c) {
  const data = _DATA_KINDS.has(d.kind);
  let html = _chip(data ? d.kind : d.language || "text");
  if (!data && d.line_count != null) html += _chip(_plural(d.line_count, "line"));
  if (d.provenance) html += _chip(d.provenance, true);
  if (d.note) html += _chip(d.note, true);
  let right = "";
  if (d.kind !== "text") right += _note("Detected by extension");
  right += _iconBtn("copy", "Copy raw text", "wv-copy");
  c.appendChild(_topbar(html + _right(right)));
  c.querySelector("#wv-copy").addEventListener("click", () => copyRaw(d.path));
  _fragment(c, d.fragment);
}

function _table(d, c) {
  const sheet = _SHEET_KINDS.has(d.kind);
  let html = _chip(d.kind);
  if (d.provenance) html += _chip(d.provenance, true);
  html += _chip(_plural(d.column_count, "column"));
  const copy = _iconBtn("copy", "Copy raw file", "wv-copy");
  // For a csv the copy button sits on the left. A sheet keeps its buttons on the right.
  const icons = sheet ? _right(copy + _iconBtn("refresh", "Re-read", "wv-refresh"))
    : '<span style="margin-left:0">' + copy + "</span>";
  c.appendChild(_topbar(html + icons));
  c.querySelector("#wv-copy").addEventListener("click", () => copyRaw(d.path));
  if (sheet) c.querySelector("#wv-refresh").addEventListener("click", () => retryConvert(d.path));
  // The box that holds the table and the row bar. The show all rows class goes here.
  const body = document.createElement("div");
  body.style.cssText = "flex:1;min-height:0;display:flex;flex-direction:column";
  const frag = document.createElement("div");
  frag.innerHTML = d.fragment || "";
  while (frag.firstChild) body.appendChild(frag.firstChild);
  if (d.row_count > _CSV_WINDOW) body.appendChild(_csvBar(d.row_count, body));
  c.appendChild(body);
}

function _csvBar(rowCount, body) {
  const bar = document.createElement("div");
  bar.className = "csv-bar";
  const chev = icon("chevDown").replace("<svg ", '<svg class="csv-chev" ');
  bar.innerHTML = chev +
    "Showing " + _CSV_WINDOW + " of " + rowCount.toLocaleString("en-US") + " rows — " +
    '<button type="button">Show all</button>';
  bar.querySelector("button").addEventListener("click", () => {
    const all = body.classList.toggle("csv-all");
    bar.querySelector("button").textContent = all ? "Show less" : "Show all";
  });
  return bar;
}

function _native(d, c) {
  if (d.kind === "image") {
    const stage = document.createElement("div");
    stage.className = "img-stage";
    const frame = document.createElement("div");
    frame.className = "img-frame";
    const img = document.createElement("img");
    img.alt = "";
    frame.appendChild(img);
    stage.appendChild(frame);
    c.appendChild(stage);
    img.addEventListener("error", () => _brokenImage(d, c));
    img.src = d.url;
    initImageZoom(img, stage);
    return;
  }
  if (d.kind === "video") {
    c.innerHTML = '<video class="pv-video" controls src="' + esc(d.url) + '"></video>';
    return;
  }
  c.innerHTML = '<div class="audio-well"><audio controls src="' + esc(d.url) + '"></audio></div>';
}

function _brokenImage(d, c) {
  c.innerHTML =
    '<div class="img-stage"><div style="margin:auto;display:flex;flex-direction:column;align-items:center;gap:12px">' +
    "<span style=\"font-size:13px;color:rgba(255,255,255,.65)\">Can't display this image</span>" +
    '<a class="btn solid" href="' + esc(dlHref(d.path)) + '">Download</a>' +
    "</div></div>";
}

function _frame(d, c) {
  if (d.kind === "html") {
    c.appendChild(_topbar(_chip("html") + _chip("scripts disabled — open raw to execute", true)));
  }
  _officeBar(d, c);
  const well = document.createElement("div");
  well.className = "frame-well";
  well.style.cssText = "flex:1;min-height:0;display:flex;flex-direction:column";
  if (d.kind === "pdf" || d.kind === "office_ppt") {
    // The pdf viewer mounts into this container.
    const host = document.createElement("div");
    host.className = "vp-pdf";
    well.appendChild(host);
    c.appendChild(well);
    mountPdfViewer(host, d.url);
    return;
  }
  const ifr = document.createElement("iframe");
  ifr.src = d.url;
  if (d.kind === "html") ifr.setAttribute("sandbox", "allow-same-origin");
  well.appendChild(ifr);
  c.appendChild(well);
}

function _raw(d, c) {
  let html = d.note ? _chip(d.note, true) : "";
  html += _download(d);
  c.appendChild(_topbar(html));
  _fragment(c, d.fragment);
}

function _unsupported(d, c) {
  c.innerHTML =
    '<div class="vp-unsupported">' +
    fileGlyph(d.name) +
    "<span>" + esc(d.name) + "</span>" +
    _download(d) +
    "</div>";
}

const _CONVERTING = {
  office: ["Converting with LibreOffice…",
    "Large documents can take up to a minute. You can leave — the result is cached if it finishes."],
  sheet: ["Reading with the built-in sheet reader…",
    "This usually finishes in a moment. You can leave — the result is cached if it finishes."],
};

function _converting(d, c, onDescriptor) {
  const sheet = _SHEET_KINDS.has(d.kind);
  const [label, note] = _CONVERTING[sheet ? "sheet" : "office"];
  c.innerHTML =
    '<div class="converting-well">' +
    '<div class="cv-spin"></div>' +
    '<div class="cv-label">' + label + "</div>" +
    '<div class="cv-elapsed">0.0s</div>' +
    '<div class="cv-note">' + note + "</div>" +
    "</div>";
  const elapsed = c.querySelector(".cv-elapsed");
  const t0 = Date.now();
  const stopTicker = startTicker(() => {
    elapsed.textContent = fmtDur(Date.now() - t0);
  }, _POLL_MS);
  const token = c._viewToken;
  let stopped = false;
  let inFlight = false;
  const stop = () => {
    stopped = true;
    stopTicker();
    clearInterval(timer);
  };
  const tick = async () => {
    // Stop once the box is off the page.
    if (stopped || !c.isConnected || c._viewToken !== token) {
      stop();
      return;
    }
    if (inFlight) return; // The last check has not come back yet.
    inFlight = true;
    // The descriptor itself is the signal: keep checking until it stops converting.
    let next;
    try {
      next = await getJSON("/api/view/" + encodePath(d.path));
    } catch (err) {
      // The file went away while it was being converted.
      if (err && err.status === 404) { stop(); _gone(d, c); return; }
      inFlight = false;
      return; // Keep checking when the network hiccups.
    }
    inFlight = false;
    if (stopped || !c.isConnected || c._viewToken !== token) {
      stop();
      return;
    }
    if (next.mode === "converting") return; // Still working, keep checking.
    stop();
    if (typeof onDescriptor === "function") onDescriptor(next);
    renderView(next, c);
  };
  const timer = setInterval(tick, _POLL_MS);
}

function _convertFailed(d, c) {
  c.innerHTML =
    '<div class="vp-convert-failed">' +
    '<p style="display:flex;align-items:center;gap:8px">' + icon("alert") + "Couldn't convert this file</p>" +
    '<p class="vp-err">' + esc(d.error || "") + "</p>" +
    '<button class="btn ghost" type="button" id="wv-retry">Retry</button>' +
    _download(d) +
    "</div>";
  c.querySelector("#wv-retry").addEventListener("click", () => retryConvert(d.path));
}

function _gone(d, c) {
  c.innerHTML =
    '<div class="pv-error">' + icon("alert") +
    '<span class="t1">File no longer exists</span>' +
    '<span class="t2">It may have been garbage-collected (30-day retention).</span>' +
    '<button class="btn ghost" type="button" id="wv-gone-refresh">Refresh</button></div>';
  c.querySelector("#wv-gone-refresh").addEventListener("click", async () => {
    try {
      const next = await getJSON("/api/view/" + encodePath(d.path));
      renderView(next, c);
    } catch {
      _gone(d, c);
    }
  });
}

export function initImageZoom(img, stage) {
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
