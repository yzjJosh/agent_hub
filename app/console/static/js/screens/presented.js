import { getJSON } from "../api.js";
import { getState, on } from "../state.js";
import { esc, parseUtc, serverNow, relLabel, fmtBytes, fileCountLabel, startTicker } from "../format.js";
import { icon, fileGlyph, emptyState, skeletonRows, setLastPolled, DEBOUNCE_MS } from "../ui.js";
import { copyText } from "../toast.js";

const _POLL_MS = 2000;
const _MIN_SKELETON_MS = 300;
const _GC_NOTE = "no longer available — garbage-collected (30 days)";

// N = 1 takes the file name, a package under one top-level folder takes the
// folder name, anything else names the lead file and counts the rest.
function _deriveTitle(files) {
  if (!files || files.length === 0) return "";
  if (files.length === 1) return files[0].name;
  const tops = new Set(files.map((f) => f.path.split("/")[0] || f.name));
  if (tops.size === 1) return [...tops][0];
  return files[0].name + " +" + (files.length - 1) + " more";
}

export async function renderPresented(el, _params) {
  const s = getState();
  const isPhone = window.matchMedia("(max-width: 767px)").matches;
  let _destroyed = false;
  let _ready = false;
  let _lastSig = null;
  let _searchQ = "";
  let _debounce = null;

  el.innerHTML =
    '<div class="page-head">' +
      '<span class="page-title">Presented by Agent</span>' +
      '<span class="t3" style="font-size:var(--text-sm)">Last polled <span class="mono lp"></span></span>' +
    "</div>" +
    '<div class="pd-toolbar toolbar"><div class="input-wrap">' + icon("search") +
      '<input class="input" id="pd-q" type="text" placeholder="Search by file name or path…" autocomplete="off" spellcheck="false">' +
    "</div></div>" +
    '<div class="pd-list" id="pd-list"></div>';

  const list = el.querySelector("#pd-list");
  const qInput = el.querySelector("#pd-q");

  function _age(p) {
    return relLabel(serverNow() - parseUtc(p.presented_at));
  }

  function _matches(p) {
    if (!_searchQ) return true;
    const q = _searchQ.toLowerCase();
    const title = (p.title && p.title.trim()) || _deriveTitle(p.files);
    if (title.toLowerCase().includes(q)) return true;
    return p.files.some((f) => f.name.toLowerCase().includes(q) || f.path.toLowerCase().includes(q));
  }

  // The lead is the first file that is still there, so a partially
  // garbage-collected package keeps showing what remains.
  function _thumbHtml(p) {
    if (p.state === "expired") return icon("alert");
    const files = p.files || [];
    const lead = files.find((f) => f.present) || files[0];
    if (!lead) return icon("box");
    if (lead.present && lead.thumb && lead.thumb.status === "done") {
      return '<img class="pd-well-img" src="' + esc(lead.thumb.url) + '" alt="">';
    }
    return fileGlyph(lead.name);
  }

  // On a phone the age rides along on the meta line.
  function _metaHtml(p) {
    if (p.state === "expired") {
      return '<span class="pd-gc">' + _GC_NOTE + "</span>";
    }
    const count = p.state === "partial"
      ? '<span class="pd-partial">' + p.present_count + " of " + p.file_count + " files available</span>"
      : fileCountLabel(p.file_count);
    let html = count + " · " + fmtBytes(p.total_size);
    if (p.kind === "solo" && p.files[0]) html += " · " + esc(p.files[0].kind);
    if (isPhone) html += " · " + esc(_age(p));
    return html;
  }

  function _rowHtml(p) {
    const title = (p.title && p.title.trim()) || _deriveTitle(p.files);
    const age = esc(_age(p));
    const well = '<div class="pd-well">' + _thumbHtml(p) + "</div>";
    if (p.state === "expired") {
      const body =
        '<div class="pd-body">' +
        '<div class="pd-title">' + esc(title) + "</div>" +
        '<div class="pd-meta">' + _metaHtml(p) + "</div>" +
        (isPhone ? '<div class="pd-age-line mono">' + age + "</div>" : "") +
        "</div>";
      return '<div class="pd-row pd-row-expired" title="' + esc(title) + " · " + esc(p.presented_at_local) + '">' +
        well + body +
        (isPhone ? "" : '<span class="pd-age mono">' + age + "</span>") +
        "</div>";
    }
    const body =
      '<div class="pd-body">' +
      '<div class="pd-title" title="' + esc(title) + " · " + esc(p.presented_at_local) + " · " + esc(p.url) + '">' + esc(title) + "</div>" +
      '<div class="pd-meta">' + _metaHtml(p) + "</div>" +
      "</div>";
    const copyBtn = '<button class="pd-copy icon-btn" type="button" data-copy-url="' + esc(p.url) +
      '" title="Copy link" aria-label="Copy link for ' + esc(title) + '">' + icon("copy") + "</button>";
    return '<a class="pd-row" href="' + esc(p.url) + '" target="_blank" rel="noopener" title="Open presentation (new tab)">' +
      well + body +
      (isPhone ? "" : '<span class="pd-age mono">' + age + "</span>") +
      copyBtn +
      "</a>";
  }

  function _sig(p) {
    return p.id + "|" + p.state + "|" + p.present_count + "|" +
      p.files.map((f) => f.present + f.size).join(",");
  }

  function _render(items) {
    if (items.length === 0) {
      list.replaceChildren();
      if (_searchQ) {
        const d = document.createElement("div");
        d.className = "t-nomatch";
        d.innerHTML = 'No matches for <span class="q">"' + esc(_searchQ) + '"</span>';
        list.appendChild(d);
      } else {
        emptyState(list, "paperPlaneEmpty", "No presentations yet",
          "The agent hasn't presented anything. New presentations appear here automatically.");
      }
      return;
    }
    list.innerHTML = items.map(_rowHtml).join("");
  }

  async function _fetch() {
    if (_destroyed) return;
    let body;
    try {
      body = await getJSON("/api/presentations");
    } catch {
      // The console-wide banner is the error UI. The list keeps its last rows.
      return;
    }
    if (_destroyed) return;
    const items = (body.presentations || []).filter(_matches);
    // The query rides along so an empty list still re-renders when the
    // search text changes.
    const sig = _searchQ + "||" + items.map(_sig).join("|");
    if (sig === _lastSig) return;
    _lastSig = sig;
    _render(items);
  }

  const off = on((kind) => {
    if (!_ready) return;
    if (kind === "changes") _fetch();
    setLastPolled(el, s.lastPollLocal);
  });
  const stopTick = startTicker(() => {
    if (!_ready) return;
    _fetch();
    setLastPolled(el, s.lastPollLocal);
  }, _POLL_MS);

  setLastPolled(el, s.lastPollLocal);
  const t0 = Date.now();
  skeletonRows(list, 6);
  await _fetch();
  const rest = _MIN_SKELETON_MS - (Date.now() - t0);
  if (rest > 0 && !_destroyed) await new Promise((r) => setTimeout(r, rest));
  if (_destroyed) return;
  list.querySelectorAll(".skel").forEach((n) => n.remove());
  _ready = true;

  qInput.addEventListener("input", () => {
    clearTimeout(_debounce);
    _debounce = setTimeout(() => {
      _searchQ = qInput.value.trim();
      _fetch();
    }, DEBOUNCE_MS);
  });

  el.addEventListener("click", (ev) => {
    const btn = ev.target.closest("[data-copy-url]");
    if (!btn) return;
    ev.preventDefault();
    ev.stopPropagation();
    copyText(btn.dataset.copyUrl, "Link copied");
  });

  return {
    destroy() {
      _destroyed = true;
      off();
      stopTick();
      clearTimeout(_debounce);
    },
  };
}
