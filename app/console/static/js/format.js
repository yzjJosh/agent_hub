const _MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const _DATE_REL_MS = 48 * 3600 * 1000;

let _offset = 0;
let _offsetKnown = false;
let _skew = 0;
let _skewKnown = false;

export function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

export function parseUtc(iso) {
  return iso ? Date.parse(iso) : NaN;
}

export function noteServerTime(utcIso, localStr) {
  if (!utcIso || !localStr) return;
  const utc = Date.parse(utcIso);
  const local = Date.parse(localStr.replace(" ", "T") + "Z");
  if (Number.isFinite(utc) && Number.isFinite(local)) {
    _offset = local - utc;
    _offsetKnown = true;
    // How far the server clock is ahead of this browser's clock.
    _skew = utc - Date.now();
    _skewKnown = true;
  }
}

export function serverNow() {
  return _skewKnown ? Date.now() + _skew : Date.now();
}

function _asUtc(ms) {
  return new Date(ms + (_offsetKnown ? _offset : 0));
}

function _pad(n) {
  return String(n).padStart(2, "0");
}

export function fmtClock(ms) {
  const d = _asUtc(ms);
  return _pad(d.getUTCHours()) + ":" + _pad(d.getUTCMinutes()) + ":" + _pad(d.getUTCSeconds());
}

export function fmtDur(ms) {
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

export function relLabel(ageMs) {
  if (!Number.isFinite(ageMs) || ageMs < 0) return "just now";
  const s = Math.floor(ageMs / 1000);
  if (s < 60) return s + "s ago";
  const m = Math.floor(s / 60);
  if (m < 60) return m + "m ago";
  const h = Math.floor(m / 60);
  if (h < 48) return h + "h ago";
  return Math.floor(h / 24) + "d ago";
}

// The month and day from the server's own time text.
// The browser's time zone never changes what it shows.
export function monthDayFromLocal(localStr) {
  if (!localStr || localStr.length < 10) return "";
  return _MONTHS[Number(localStr.slice(5, 7)) - 1] + " " + Number(localStr.slice(8, 10));
}

export function monthDayHmFromLocal(localStr) {
  if (!localStr || localStr.length < 16) return "";
  return monthDayFromLocal(localStr) + " " + localStr.slice(11, 16);
}

// A short age like 3h ago for recent times, otherwise the month and day.
// The time comes in as seconds.
export function dateLabel(mtime, mtimeLocal) {
  const t = mtime * 1000;
  if (!Number.isFinite(t)) return "";
  return serverNow() - t < _DATE_REL_MS ? relLabel(serverNow() - t) : monthDayFromLocal(mtimeLocal);
}

export function fmtBytes(n) {
  if (n < 1024) return n + " B";
  if (n < 1048576) return Math.round(n / 1024) + " KB";
  return (n / 1048576).toFixed(1) + " MB";
}

export function fmtBytesPair(loaded, total) {
  if (total >= 1048576) return (loaded / 1048576).toFixed(1) + "/" + (total / 1048576).toFixed(1) + " MB";
  if (total >= 1024) return Math.round(loaded / 1024) + "/" + Math.round(total / 1024) + " KB";
  return loaded + "/" + total + " B";
}

export function fmtBytesShort(n) {
  if (n < 1024) return n + "B";
  if (n < 1048576) {
    const k = n / 1024;
    return (k >= 10 ? String(Math.round(k)) : k.toFixed(1)) + "K";
  }
  const m = n / 1048576;
  return (m >= 10 ? String(Math.round(m)) : m.toFixed(1)) + "M";
}

export function fmtBytesExact(n) {
  return n.toLocaleString("en-US") + " bytes";
}

export function countLines(text) {
  if (!text) return 0;
  return text.split("\n").length - 1 + (text.endsWith("\n") ? 0 : 1);
}

export function fileCountLabel(n) {
  return n + (n === 1 ? " file" : " files");
}

export function startTicker(cb, ms = 1000) {
  cb();
  const t = setInterval(cb, ms);
  return () => clearInterval(t);
}
