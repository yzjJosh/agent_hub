// The folder gallery. It draws the cards for one folder and keeps them fresh.
// Picking a card sends a "gallery:select" event, and the Open button sends
// "gallery:open" instead. Both events bubble up, so the screen can catch them.
import { getJSON, postJSON, dlHref } from "./api.js";
import { esc, fmtBytes, dateLabel, fileCountLabel, startTicker } from "./format.js";
import { icon, fileGlyph } from "./ui.js";

const _POLL_MS = 2000;
const _SELECT_EVENT = "gallery:select";
const _OPEN_EVENT = "gallery:open";
const _PREPARING = "Preparing preview…";
const _GONE_NOTE = "no longer exists";
const _GONE_WARN = "Removed by 30-day retention — Refresh to drop it";
const _FAILED_WARN = "Thumbnail unavailable — open to view";
const _EMPTY_TITLE = "Empty folder";
const _EMPTY_SUB = "Files appear here when the agent writes to this folder.";

// The picture area for one file.
// It shows the small preview when there is one, and a file glyph when there is not.
export function thumbSrc(entry) {
  const status = entry.thumb ? entry.thumb.status : "none";
  const glyph = '<span class="thumb-glyph">' + fileGlyph(entry.name) + "</span>";
  if (status === "done") {
    return '<img class="thumb-img done" src="' + esc(entry.thumb.url) + '" alt="">';
  }
  if (status === "failed") {
    return glyph + '<span class="thumb-failed-badge" title="Thumbnail generation failed">!</span>';
  }
  if (status === "none") return glyph;
  return glyph + '<span class="thumb-preparing">' + _PREPARING + "</span>";
}

// The compact 16:10 slot for picker rows and cards. It shares the state
// vocabulary with the labeled slot above, drops the text pill, and shimmers
// on pending as well as generating.
export function compactSlot(entry) {
  const status = entry.thumb ? entry.thumb.status : "none";
  const glyph = '<span class="thumb-glyph">' + fileGlyph(entry.name) + "</span>";
  const cls = status === "pending" || status === "generating" ? "thumb-1610 slot-prep" : "thumb-1610";
  if (status === "done") {
    return '<span class="' + cls + '"><img class="thumb-img done" src="' + esc(entry.thumb.url) + '" alt=""></span>';
  }
  if (status === "failed") {
    return '<span class="' + cls + '">' + glyph + '<span class="thumb-failed-badge" title="Thumbnail generation failed">!</span></span>';
  }
  return '<span class="' + cls + '">' + glyph + "</span>";
}

function _goneMap(gonePaths) {
  const m = new Map();
  if (gonePaths instanceof Map) {
    for (const [p, e] of gonePaths) m.set(p, e);
  } else if (gonePaths) {
    for (const p of gonePaths) m.set(p, { name: p.split("/").pop(), path: p, mtime: 0, type: "file" });
  }
  return m;
}

function _fileMeta(e) {
  return [e.size != null ? fmtBytes(e.size) : "", dateLabel(e.mtime, e.mtime_local)].filter(Boolean).join(" · ");
}

function _folderMeta(e) {
  return fileCountLabel(e.file_count) + " · open folder";
}

function _cardShell(path, extraClass) {
  const card = document.createElement("div");
  card.className = "thumb-card" + (extraClass ? " " + extraClass : "");
  card.dataset.path = path;
  card.tabIndex = 0;
  card.setAttribute("role", "button");
  card.title = path;
  return card;
}

function _bindActivate(card, entry) {
  const act = (eventName) =>
    card.dispatchEvent(new CustomEvent(eventName, { bubbles: true, detail: entry }));
  card.addEventListener("click", (ev) => {
    if (ev.target.closest("a")) return; // The Download link works on its own.
    ev.preventDefault();
    // The Open button asks for full screen. The rest of the card only picks it.
    act(ev.target.closest(".open") ? _OPEN_EVENT : _SELECT_EVENT);
  });
  card.addEventListener("keydown", (ev) => {
    if (ev.target !== card || (ev.key !== "Enter" && ev.key !== " ")) return;
    ev.preventDefault();
    act(_SELECT_EVENT);
  });
}

function _fileCard(e) {
  const generating = e.thumb && e.thumb.status === "generating";
  const failed = e.thumb && e.thumb.status === "failed";
  const card = _cardShell(e.path, generating ? "generating" : "");
  card.innerHTML =
    '<div class="thumb-1610">' + thumbSrc(e) + "</div>" +
    '<div class="thumb-name">' + esc(e.name) + "</div>" +
    '<div class="thumb-meta">' + esc(_fileMeta(e)) + "</div>" +
    (failed ? '<div class="thumb-warn">' + _FAILED_WARN + "</div>" : "") +
    '<div class="thumb-actions">' +
      '<button type="button" class="open" title="Open in full screen — this console">' + icon("maximize") + "</button>" +
      '<a href="' + esc(dlHref(e.path)) + '" title="Download">' + icon("download") + "</a>" +
    "</div>";
  _bindActivate(card, e);
  return card;
}

function _folderCard(e) {
  const card = _cardShell(e.path);
  card.innerHTML =
    '<div class="thumb-1610"><span class="thumb-glyph">' + icon("folder") + "</span></div>" +
    '<div class="thumb-name">' + esc(e.name) + "</div>" +
    '<div class="thumb-meta">' + esc(_folderMeta(e)) + "</div>" +
    '<div class="thumb-actions">' +
      '<button type="button" class="open" title="Open folder in full screen">' + icon("maximize") + "</button>" +
    "</div>";
  _bindActivate(card, e);
  return card;
}

function _goneCard(e) {
  const card = document.createElement("div");
  card.className = "thumb-card gone";
  card.dataset.path = e.path;
  card.title = e.path;
  card.innerHTML =
    '<div class="thumb-1610"><span class="thumb-glyph">' + fileGlyph(e.name) + "</span></div>" +
    '<div class="thumb-name">' + esc(e.name) + "</div>" +
    '<div class="thumb-meta">' + _GONE_NOTE + "</div>" +
    '<div class="thumb-warn">' + _GONE_WARN + "</div>";
  return card;
}

function _emptyNote() {
  const d = document.createElement("div");
  d.className = "gallery-empty";
  d.innerHTML = icon("folder") + '<span class="t1">' + _EMPTY_TITLE + "</span><span class=\"t2\">" + _EMPTY_SUB + "</span>";
  return d;
}

const _fileStamp = (e) => "f|" + (e.mtime || 0) + "|" + (e.thumb ? e.thumb.status : "none");
const _dirStamp = (e) => "d|" + (e.mtime || 0) + "|" + (e.file_count || 0);

export function sortEntries(entries) {
  const dirs = entries.filter((e) => e.type === "dir").sort((a, b) => a.name.localeCompare(b.name));
  const files = entries.filter((e) => e.type !== "dir").sort((a, b) => (b.mtime || 0) - (a.mtime || 0));
  return [...dirs, ...files];
}

// Draws the card grid. Folders come first by name, then files with the newest first.
// gonePaths names the files that are gone now, and they still get a card.
// With nothing to show it draws a short empty note.
// onMounted is called once the grid is on the page.
export function renderGallery(entries, gonePaths, container, opts = {}) {
  const gone = _goneMap(gonePaths);
  const effective = [
    ...entries.filter((e) => e.type === "dir"),
    ...entries.filter((e) => e.type === "file" && !gone.has(e.path)),
    ...[...gone].map(([p, g]) => ({ ...g, path: p })),
  ];
  const sorted = sortEntries(effective);
  if (sorted.length === 0) {
    container.replaceChildren(_emptyNote());
    return;
  }
  const grid = document.createElement("div");
  grid.className = "gallery-grid";
  const prev = new Map();
  const oldGrid = container.querySelector(":scope > .gallery-grid");
  if (oldGrid) {
    for (const c of oldGrid.querySelectorAll(":scope > .thumb-card")) prev.set(c.dataset.path, c);
  }
  const place = (path, stamp, build) => {
    const old = prev.get(path);
    if (old && old.dataset.stamp === stamp) {
      grid.appendChild(old);
      return;
    }
    const card = build();
    card.dataset.stamp = stamp;
    grid.appendChild(card);
  };
  for (const e of sorted) {
    if (e.type === "dir") {
      place(e.path, _dirStamp(e), () => _folderCard(e));
    } else {
      const isGone = gone.has(e.path);
      place(e.path, isGone ? "g|" + (e.mtime || 0) : _fileStamp(e), () => (isGone ? _goneCard(e) : _fileCard(e)));
    }
  }
  container.replaceChildren(grid);
  if (opts.onMounted) requestAnimationFrame(() => opts.onMounted());
}

// Watches one folder and tells the caller what changed there.
// It looks again every two seconds and stops once nothing is left to wait for.
// Call the handle it gives back to stop it when the screen goes away.
export function initGalleryPoll(container, dirPath, handlers = {}) {
  const { onEntries, onGone, onFolderGone } = handlers;
  let live = null; // The file paths seen last time.
  const declaredGone = new Set();
  let inFlight = false;
  let stop = null;
  const stopAll = () => { if (stop) stop(); };
  const tick = async () => {
    if (inFlight) return;
    inFlight = true;
    let body;
    try {
      body = await getJSON("/api/resources?" + new URLSearchParams({ path: dirPath }));
    } catch (err) {
      inFlight = false;
      if (err && err.status === 404) {
        stopAll();
        if (onFolderGone) onFolderGone();
      }
      return;
    }
    inFlight = false;
    const entries = body.entries || [];
    const nowLive = new Set(entries.filter((e) => e.type === "file").map((e) => e.path));
    if (live) {
      for (const p of live) {
        if (!nowLive.has(p)) {
          declaredGone.add(p);
          if (onGone) onGone(p);
        }
      }
    }
    for (const p of declaredGone) if (nowLive.has(p)) declaredGone.delete(p);
    live = nowLive;
    if (onEntries) onEntries(entries);
    const hasWork = entries.some((e) =>
      e.type === "file" && e.thumb && (e.thumb.status === "pending" || e.thumb.status === "generating"));
    if (hasWork) postJSON("/api/resources/thumbs?" + new URLSearchParams({ path: dirPath })).catch(() => {});
    if (!hasWork && declaredGone.size === 0) stopAll();
  };
  stop = startTicker(tick, _POLL_MS);
  return stopAll;
}
