import { esc } from "./format.js";

export const FAIL_BANNER_AT = 3;
export const DEBOUNCE_MS = 150;
export const SKELETON_MS = 600;

export const ICONS = {
  edit: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m10.5 2 3.5 3.5-8.5 8.5H2v-3.5Z"/><path d="m8.5 4 3.5 3.5"/></svg>',
  trash: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M2.5 4h11M6 4V2h4v2M4 4l.5 10h7L12 4M6.5 6.5v5M9.5 6.5v5"/></svg>',
  prompt: '<svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 5 4 3-4 3"/><path d="M9 12h4"/></svg>',
  rows: '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M3 4.5h10M3 8h10M3 11.5h6"/></svg>',
  folder: '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M2 4.5c0-.6.4-1 1-1h3l1.5 1.5H13c.6 0 1 .4 1 1v5.5c0 .6-.4 1-1 1H3c-.6 0-1-.4-1-1V4.5Z"/></svg>',
  open: '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M6.5 3.5H3v9.5h9.5V9.5"/><path d="M9.5 2.5h4v4M13 3 8.5 7.5"/></svg>',
  check: '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8.5 6.5 12 13 4.5"/></svg>',
  folderSm: '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="#6f7d90" stroke-width="1.5" stroke-linejoin="round"><path d="M2 4.5c0-.6.4-1 1-1h3l1.5 1.5H13c.6 0 1 .4 1 1v5.5c0 .6-.4 1-1 1H3c-.6 0-1-.4-1-1V4.5Z"/></svg>',
  monitor: '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="12" height="8.5" rx="1"/><path d="M6 14h4M8 11.5V14"/></svg>',
  monitorOff: '<svg width="26" height="26" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="12" height="8.5" rx="1"/><path d="M6 14h4M8 11.5V14"/><path d="m4 5 6 5M10 5 4 10"/></svg>',
  search: '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="7" cy="7" r="4.5"/><path d="m13.5 13.5-3-3"/></svg>',
  chevR: '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m6 4 4 4-4 4"/></svg>',
  chevL: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m10 4-4 4 4 4"/></svg>',
  chevDown: '<svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m4 6 4 4 4-4"/></svg>',
  chevUp: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m4 10 4-4 4 4"/></svg>',
  caret: '<svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 4 4 4-4 4"/></svg>',
  copy: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><rect x="5.5" y="5.5" width="8" height="8" rx="1.5"/><path d="M10.5 5.5v-2a1 1 0 0 0-1-1h-6a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2"/></svg>',
  download: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M8 2.5v7M5 6.5l3 3 3-3M3 12.5h10"/></svg>',
  upload: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M8 10.5v-7M5 6l3-3 3 3M3 12.5h10"/></svg>',
  plus: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M8 3.5v9M3.5 8h9"/></svg>',
  x: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M4 4l8 8M12 4l-8 8"/></svg>',
  box: '<svg width="22" height="22" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"><path d="m8 2 5.5 3.2v5.6L8 14l-5.5-3.2V5.2L8 2Z"/><path d="M2.5 5.2 8 8.4l5.5-3.2M8 8.4V14"/></svg>',
  terminal: '<svg width="20" height="20" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="12" height="10" rx="1.5"/><path d="m4.5 6.5 2 1.75-2 1.75M8.5 10.5h3"/></svg>',
  alert: '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="8" cy="8" r="6"/><path d="M8 5v4M8 11.2v.3"/></svg>',
  refresh: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M13 8a5 5 0 1 1-1.7-3.7"/><path d="M13 2.5v3h-3"/></svg>',
  maximize: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M2.5 6V2.5H6M10 2.5h3.5V6M13.5 10v3.5H10M6 13.5H2.5V10"/></svg>',
  minimize: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M2.5 2.5H6V6M13.5 2.5H10V6M13.5 13.5V10H10M6 13.5V10H2.5"/></svg>',
  viewer: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M10 3.5H5A2 2 0 0 0 3 5.5V11a2 2 0 0 0 2 2h5.5a2 2 0 0 0 2-2V8"/><path d="M9.5 2.5h4v4"/><path d="M6.5 9.5 13 3"/></svg>',
  pause: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M6 4v8M10 4v8"/></svg>',
  play: '<svg viewBox="0 0 16 16" fill="none"><path d="M5.5 3.5v9l8-4.5z" fill="currentColor" stroke="none"/></svg>',
  arrowDown: '<svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3v9M4.5 8.5l3.5 3.5L11.5 8.5"/></svg>',
  corner: '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v7a2 2 0 0 0 2 2h7"/><path d="m9 9 3 3-3 3"/></svg>',
  tool: '<svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="8" cy="8" r="5.5"/><path d="M5.5 8.5 8 6l2.5 2.5"/></svg>',
  image: '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"><rect x="2.5" y="3" width="11" height="10" rx="1"/><circle cx="5.5" cy="6" r="1"/><path d="m2.5 11 3-2.5 3 2.5 2.5-2 4.5 3"/></svg>',
  audio: '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M6 12V4.5L12 3v7.5"/><circle cx="4.5" cy="12" r="1.5"/><circle cx="10.5" cy="10.5" r="1.5"/></svg>',
  doc: '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"><path d="M4 2.5h5L12 5.5v8H4v-11Z"/><path d="M9 2.5v3h3M5.8 8.5h4.4M5.8 10.5h4.4"/></svg>',
  paperPlane: '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2 7 8.5M14 2l-4.5 12-2.5-5.5L2 6 14 2Z"/></svg>',
  paperPlaneEmpty: '<svg width="22" height="22" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round" opacity=".35"><path d="M14 2 7 8.5M14 2l-4.5 12-2.5-5.5L2 6 14 2Z"/></svg>',
};

export function icon(name) {
  return ICONS[name] || "";
}

export function emptyState(root, iconName, title, sub, actionLabel, actionHref) {
  const el = document.createElement("div");
  el.className = "empty";
  const action = actionLabel
    ? (actionHref
        ? '<a class="btn ghost" data-spa href="' + esc(actionHref) + '">' + esc(actionLabel) + "</a>"
        : '<button class="btn ghost">' + esc(actionLabel) + "</button>")
    : "";
  el.innerHTML =
    '<span class="empty-ic">' + icon(iconName) + "</span>" +
    '<span class="t-title">' + esc(title) + "</span>" +
    (sub ? '<span class="t-sub">' + esc(sub) + "</span>" : "") +
    action;
  root.appendChild(el);
  return el;
}

export function skeletonRows(container, n) {
  const wrap = document.createElement("div");
  wrap.className = "skel";
  for (let i = 0; i < n; i++) {
    const row = document.createElement("div");
    row.className = "skel-row";
    wrap.appendChild(row);
  }
  container.appendChild(wrap);
  return wrap;
}

export function setLastPolled(root, lastPollLocal) {
  for (const n of root.querySelectorAll(".lp")) n.textContent = lastPollLocal || "—";
}

const _VALUE_CAP = 80;

function _cap(v) {
  return v.length > _VALUE_CAP ? v.slice(0, _VALUE_CAP) + "…" : v;
}

export function paramSummary(params) {
  if (!params || typeof params !== "object") return "";
  const parts = [];
  for (const key of Object.keys(params)) {
    const v = params[key];
    if (v === null || v === undefined) continue;
    let val;
    if (typeof v === "string") val = '"' + _cap(v) + '"';
    else if (typeof v === "number" || typeof v === "boolean") val = String(v);
    else val = _cap(JSON.stringify(v));
    parts.push('<span class="pk">' + esc(key) + "=</span><span class=\"pv\">" + esc(val) + "</span>");
  }
  return parts.join(" ");
}

export function paramTitle(params) {
  if (!params) return "";
  try { return JSON.stringify(params); } catch { return ""; }
}

export function fileClass(name) {
  const ext = name.includes(".") ? name.split(".").pop().toLowerCase() : "";
  if (["png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"].includes(ext)) return "image";
  if (["opus", "mp3", "wav", "m4a", "ogg", "flac", "aac"].includes(ext)) return "audio";
  if (["mp4", "webm", "mov", "mkv", "avi"].includes(ext)) return "video";
  if (["txt", "md", "markdown", "json", "jsonl", "csv", "log", "xml", "yaml", "yml", "toml", "ini", "cfg", "conf", "html", "css", "py", "js", "ts", "sh", "sql", "c", "h", "go", "rs", "java"].includes(ext)) return "text";
  return "binary";
}

const _TINT_IMAGE = "#7aa2f7";
const _TINT_AUDIO = "#9d7cd8";
const _TINT_DOC = "#5f87d6";
const _TINT_PY = "#8bd46f";
const _TINT_ZIP = "#c9a227";

export function fileGlyph(name) {
  const ext = name.includes(".") ? name.split(".").pop().toLowerCase() : "";
  const cls = fileClass(name);
  let color = "currentColor";
  let body = ICONS.doc;
  if (cls === "image") { color = _TINT_IMAGE; body = ICONS.image; }
  else if (cls === "audio") { color = _TINT_AUDIO; body = ICONS.audio; }
  else if (cls === "video") { color = _TINT_DOC; body = ICONS.image; }
  else if (["pdf", "docx", "doc", "xlsx", "xls", "pptx"].includes(ext)) { color = _TINT_DOC; body = ICONS.doc; }
  else if (ext === "py") { color = _TINT_PY; body = ICONS.doc; }
  else if (["zip", "tar", "gz"].includes(ext)) { color = _TINT_ZIP; body = ICONS.doc; }
  else if (cls === "text") { color = _TINT_DOC; body = ICONS.doc; }
  if (color !== "currentColor") body = body.replace('stroke="currentColor"', 'stroke="' + color + '"');
  return body;
}
