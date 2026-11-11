// The result feed holds the body block, the FILES divider, and one block per file.
// The file wells use the file-viewer page's markup. Same classes and state cards, no topbar, no provenance strip.
import { getJSON, postJSON, ApiError, dlHref, encodePath } from "../api.js";
import { esc, fmtBytes, fileCountLabel } from "../format.js";
import { icon } from "../ui.js";
import { toast } from "../toast.js";
import { initAudioCard } from "../audio-card.js";
import { mountPdfViewer } from "../pdf-viewer.js";

const _MENU_KINDS = [
  ["text", "text"], ["markdown", "markdown"], ["code", "code"], ["json", "JSON"],
  ["csv", "CSV"], ["html", "HTML"], ["xml", "XML"], ["yaml", "YAML"],
];
const _SHEET_KINDS = new Set(["xlsx", "xls"]);
const _RAW_COLLAPSE = 4096;    // bytes shown first in Raw view
const _RAW_WINDOW = 524288;    // bytes added each time Raw view loads more
const _CSV_WINDOW = 8;         // Rows shown before the reveal bar. fragments.py has the same number and must match.
const _LINE_WINDOW = 1000;     // Lines the server sends per code page. _WINDOW_LINES in result_descriptor.py must match.
const _POLL_MS = 1000;
const _CONVERTING = {
  office: ["Converting with LibreOffice…",
    "Large documents can take up to a minute. You can leave — the result is cached if it finishes."],
  sheet: ["Reading with the built-in sheet reader…",
    "This usually finishes in a moment. You can leave — the result is cached if it finishes."],
};

function _isPhone() {
  return matchMedia("(max-width: 767px)").matches;
}

function _fragment(container, fragment) {
  if (!fragment) return;
  const el = document.createElement("div");
  el.innerHTML = fragment;
  while (el.firstChild) container.appendChild(el.firstChild);
}

function _note(label) {
  const el = document.createElement("div");
  el.className = "res-note";
  el.textContent = label;
  return el;
}

function _stateCard(ic, title, sub) {
  const el = document.createElement("div");
  el.className = "feed-state";
  el.innerHTML =
    '<span class="vp-state-ic">' + icon(ic) + "</span>" +
    '<h1 class="vp-state-title">' + esc(title) + "</h1>" +
    '<p class="vp-state-sub">' + esc(sub) + "</p>";
  return el;
}

export function mountResult(mount, detail, pendingMessage) {
  const enc = encodeURIComponent(detail.id);
  const feedBase = "/api/tasks/" + enc + "/result/feed";
  const viewBase = "/api/tasks/" + enc + "/result/view";
  const dlRes = "/api/tasks/" + enc + "/result/download";
  const viewResult = "/view/result/" + enc;

  let seq = 0;             // only the newest feed or body load is used
  let bodyBlock = null;    // the body block in the feed
  let bodyDescriptor = null;
  let bodyKind = "";       // the explicit kind, or "" for auto-detect
  let bodyView = "preview";
  let lineOffset = 0;      // the next line to load in a code body
  let filesN = 0;
  let rawState = null;
  let activeMenu = null;   // the open kind popover and its close function
  const audioCards = new Set();

  function stopAudio() {
    for (const card of audioCards) card.stop();
    audioCards.clear();
  }
  function stopBlockTimers(root) {
    for (const el of root.querySelectorAll(".vp-converting")) {
      if (el._timer) clearInterval(el._timer);
    }
  }

  // ---- the feed ----

  async function render() {
    seq++;
    const s = seq;
    closeMenu();
    stopAudio();
    stopBlockTimers(mount);
    if (detail.task.status === "running") {
      mount.replaceChildren(pendingEl());
      return;
    }
    mount.replaceChildren(_skeleton());
    let feed;
    try {
      feed = await getJSON(feedBase);
    } catch (err) {
      if (s !== seq) return;
      if (err instanceof ApiError && (err.code === "result_not_found" || err.status === 404)) {
        mount.replaceChildren(noResultEl());
      } else {
        mount.replaceChildren(feedErrorEl(err));
      }
      return;
    }
    if (s !== seq) return;
    bodyKind = "";
    bodyView = "preview";
    lineOffset = 0;
    filesN = feed.files.length;
    mount.replaceChildren(buildFeed(feed));
  }
  let renderP = render();

  function _skeleton() {
    const el = document.createElement("div");
    el.className = "well-skeleton";
    return el;
  }

  function buildFeed(feed) {
    const root = document.createElement("div");
    root.className = "feedwrap";
    bodyBlock = bodyBlockEl(feed.body, feed.files.length);
    root.appendChild(bodyBlock);
    if (feed.files.length > 0) {
      const div = document.createElement("div");
      div.className = "vp-feed-files";
      div.textContent = "FILES · " + feed.files.length;
      root.appendChild(div);
      for (const f of feed.files) root.appendChild(fileBlock(f));
    }
    return root;
  }

  // ---- states with no feed ----

  function pendingEl() {
    const el = document.createElement("div");
    el.className = "res-pending";
    if (pendingMessage) {
      el.innerHTML = icon("box") +
        '<span class="t1">' + esc(pendingMessage) + "</span>" +
        '<span class="t2"></span>';
      return el;
    }
    el.innerHTML = icon("box") +
      '<span class="t1">Result available when the run completes</span>' +
      '<span class="t2">Progress is streaming in the Log tab.</span>';
    return el;
  }

  function noResultEl() {
    const el = document.createElement("div");
    el.className = "res-nores";
    el.innerHTML = icon("box") +
      '<span class="t1">No result</span>' +
      '<span class="t2">Results are stored only for completed runs.</span>';
    return el;
  }

  function feedErrorEl(err) {
    const el = _stateCard("alert", "Cannot load result", (err && err.message) || String(err));
    const acts = document.createElement("div");
    acts.className = "vp-state-actions";
    acts.innerHTML = '<button class="btn ghost danger" type="button" data-act="retry">Retry</button>';
    el.appendChild(acts);
    el.querySelector('[data-act="retry"]').addEventListener("click", () => { renderP = render(); });
    return el;
  }

  // ---- body block ----

  function bodyBlockEl(d, nFiles) {
    const block = document.createElement("div");
    block.className = "res-body-block";
    bodyDescriptor = d;
    if (d.size === 0) {
      // The empty body has no content to give a kind, so no chip and no actions.
      // With files there is no empty text block — the feed starts at the divider.
      block.appendChild(bareHead(nFiles > 0 ? fileCountLabel(nFiles) : (_isPhone() ? "0 B" : "0 bytes")));
      if (nFiles === 0) block.appendChild(_stateCard("box", "No result content", "This run stored an empty result."));
      return block;
    }
    if (d.mode === "legacy") {
      block.appendChild(legacyEl(d));
      return block;
    }
    block.appendChild(bodyHead(d));
    block.appendChild(bodyWell(d));
    return block;
  }

  function headRow() {
    const head = document.createElement("div");
    head.className = "vp-file-head res-body-head";
    return head;
  }

  function _act(href, ic, title, external) {
    const a = document.createElement("a");
    a.className = "vp-file-act";
    a.href = href;
    if (external) { a.target = "_blank"; a.rel = "noopener"; }
    a.title = title;
    a.setAttribute("aria-label", title);
    a.innerHTML = icon(ic);
    return a;
  }

  // The degraded head: a bare label, with no open-full-screen action.
  function bareHead(label) {
    const head = headRow();
    const span = document.createElement("span");
    span.className = "res-bare";
    span.textContent = label;
    head.appendChild(span);
    return head;
  }

  function legacyEl(d) {
    const size = _isPhone() ? "" : " · " + esc(fmtBytes(d.size));
    const wrap = document.createElement("div");
    wrap.className = "vp-body-legacy";
    wrap.innerHTML =
      '<div class="feed-state">' +
      '<span class="vp-state-ic">' + icon("box") + "</span>" +
      '<h1 class="vp-state-title">result.bin' + size + "</h1>" +
      '<p class="vp-legacy-label">Legacy binary result</p>' +
      '<p class="vp-state-sub">This run predates the text-body migration — the stored body is binary. Use Download.</p>' +
      "</div>" +
      '<a class="btn ghost" href="' + dlRes + '">Download</a>';
    return wrap;
  }

  function chipLabel(d) {
    return d.kind_label || d.kind || "";
  }

  function bodyHead(d) {
    const head = headRow();
    const wrap = document.createElement("span");
    wrap.className = "res-chip-wrap";
    if (d.declared_kind) {
      // The tool fixed the kind, so the chip is a label, not a control.
      const fixed = document.createElement("span");
      fixed.className = "res-chip res-chip-static";
      fixed.setAttribute("aria-label", "Preview kind");
      fixed.title = "Preview kind (fixed by the tool)";
      fixed.innerHTML = '<span class="res-chip-label">' + esc(chipLabel(d)) + "</span>";
      wrap.appendChild(fixed);
    } else {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "res-chip";
      btn.title = "Preview kind";
      btn.innerHTML = '<span class="res-chip-label">' + esc(chipLabel(d)) + "</span>" + icon("caret");
      btn.addEventListener("click", (ev) => { ev.stopPropagation(); toggleMenu(d, btn, wrap); });
      wrap.appendChild(btn);
      const sel = document.createElement("select");
      sel.className = "res-chip-select";
      sel.setAttribute("aria-label", "Preview kind");
      sel.innerHTML = '<option value="">Auto-detect</option>' +
        _MENU_KINDS.map(([v, l]) => '<option value="' + v + '">' + l + "</option>").join("");
      sel.value = bodyKind;
      sel.addEventListener("change", () => applyKind(sel.value));
      wrap.appendChild(sel);
      const face = document.createElement("span");
      face.className = "res-chip-face";
      face.setAttribute("aria-hidden", "true");
      face.innerHTML = '<span class="res-chip-label">' + esc(chipLabel(d)) + "</span>" + icon("caret");
      wrap.appendChild(face);
    }
    head.appendChild(wrap);
    const acts = document.createElement("div");
    acts.className = "vp-file-actions";
    acts.appendChild(_act(viewResult + bodyQuery(), "maximize", "Open full-screen", true));
    acts.appendChild(_act(dlRes, "download", "Download"));
    head.appendChild(acts);
    return head;
  }

  // ---- the kind menu (desktop popover) ----

  function _menuRow(label, value, right, active) {
    return '<div class="kind-menu-row' + (active ? " active" : "") + '" data-kind="' + value + '">' +
      '<span class="k-check">' + icon("check") + "</span>" +
      "<span>" + label + "</span>" +
      (right ? '<span class="k-detected">' + esc(right) + "</span>" : "") +
      "</div>";
  }

  function toggleMenu(d, btn, wrap) {
    if (activeMenu && activeMenu.wrap === wrap && activeMenu.menu.isConnected) {
      activeMenu.close();
      return;
    }
    closeMenu();
    const menu = document.createElement("div");
    menu.className = "kind-menu";
    let html = _menuRow("Auto-detect", "", d.detected_kind || "", !bodyKind);
    for (const [value, label] of _MENU_KINDS) html += _menuRow(label, value, "", bodyKind === value);
    menu.innerHTML = html;
    const lastFocus = document.activeElement;
    function onOutside(ev) { if (!menu.contains(ev.target)) closeMenu(); }
    function onEsc(ev) { if (ev.key === "Escape") closeMenu(); }
    function close() {
      activeMenu = null;
      menu.remove();
      btn.classList.remove("open");
      document.removeEventListener("mousedown", onOutside);
      document.removeEventListener("keydown", onEsc);
      if (lastFocus) lastFocus.focus();
    }
    menu.querySelectorAll(".kind-menu-row").forEach((row) => {
      row.addEventListener("click", () => { close(); applyKind(row.dataset.kind); });
    });
    btn.classList.add("open");
    wrap.appendChild(menu);
    document.addEventListener("mousedown", onOutside);
    document.addEventListener("keydown", onEsc);
    activeMenu = { menu, wrap, close };
  }

  function closeMenu() {
    if (activeMenu) { activeMenu.close(); activeMenu = null; }
  }

  function applyKind(kind) {
    bodyKind = kind;
    lineOffset = 0;
    bodyView = "preview";
    fetchBody();
  }

  // Re-fetches only the body and swaps its block in place. The files are untouched.
  async function fetchBody() {
    seq++;
    const s = seq;
    let d;
    try {
      d = await getJSON(viewBase + bodyQuery());
    } catch (err) {
      if (s !== seq) return;
      toast("Load result failed: " + ((err && err.message) || err), { error: true });
      return;
    }
    if (s !== seq || !bodyBlock || !bodyBlock.isConnected) return;
    lineOffset = d.truncated ? (d.line_offset != null ? d.line_offset + _LINE_WINDOW : _LINE_WINDOW) : 0;
    const next = bodyBlockEl(d, filesN);
    bodyBlock.replaceWith(next);
    bodyBlock = next;
  }

  function bodyQuery() {
    const q = new URLSearchParams();
    if (bodyKind) q.set("kind", bodyKind);
    const s = q.toString();
    return s ? "?" + s : "";
  }

  // ---- body well: Preview and Raw ----

  function bodyWell(d) {
    const well = document.createElement("div");
    const mode = d.mode === "doc" ? "doc" : d.mode === "table" ? "table" : d.mode === "code" ? "code" : d.mode === "plain" ? "plain" : "raw";
    well.className = "vp-well vp-well-" + mode + " res-body-well";
    well.appendChild(microtab());
    if (bodyView === "raw") {
      rawState = { off: 0, acc: "", full: false, el: null };
      const pre = document.createElement("pre");
      pre.className = "raw-pre";
      well.appendChild(pre);
      rawState.el = pre;
      _rawFetch(d, well);
      return well;
    }
    if (d.note) well.appendChild(_note(d.note));
    _fragment(well, d.fragment);
    if (mode === "table") wireCsvReveal(well, d);
    if (mode === "code" && d.truncated) addLineWindowBar(well, d);
    return well;
  }

  function microtab() {
    const el = document.createElement("div");
    el.className = "res-microtab";
    el.innerHTML =
      '<span class="res-microtab-pill">' +
      '<button type="button" data-view="preview"' + (bodyView === "preview" ? ' class="active"' : "") + ">Preview</button>" +
      '<button type="button" data-view="raw"' + (bodyView === "raw" ? ' class="active"' : "") + ">Raw</button>" +
      "</span>";
    el.querySelector('[data-view="preview"]').addEventListener("click", () => setBodyView("preview"));
    el.querySelector('[data-view="raw"]').addEventListener("click", () => setBodyView("raw"));
    return el;
  }

  function setBodyView(view) {
    if (view === bodyView) return;
    bodyView = view;
    swapBodyWell();
  }

  function swapBodyWell() {
    if (!bodyBlock || !bodyBlock.isConnected || !bodyDescriptor) return;
    rawState = null;
    const old = bodyBlock.querySelector(".res-body-well");
    if (old) old.remove();
    bodyBlock.appendChild(bodyWell(bodyDescriptor));
  }

  // ---- line windows ----

  function addLineWindowBar(well, d) {
    const bar = document.createElement("button");
    bar.type = "button";
    bar.className = "res-bar";
    bar.innerHTML = '<span class="res-chev">' + icon("chevDown") + "</span>" +
      '<span class="res-label">Show more — ' + d.line_count.toLocaleString("en-US") + " lines total</span>";
    bar.addEventListener("click", async () => {
      bar.disabled = true;
      let next;
      try {
        const q = new URLSearchParams();
        if (bodyKind) q.set("kind", bodyKind);
        q.set("line_offset", String(lineOffset));
        next = await getJSON(viewBase + "?" + q.toString());
      } catch {
        bar.disabled = false;
        toast("Couldn't load more — showing first " + lineOffset + " lines", { error: true });
        return;
      }
      if (!well.isConnected) return;
      _appendSlice(well, next);
      lineOffset = next.line_offset != null ? next.line_offset + _LINE_WINDOW : lineOffset + _LINE_WINDOW;
      if (!next.truncated) bar.remove();
      else bar.disabled = false;
    });
    well.appendChild(bar);
  }

  function _appendSlice(well, d) {
    if (!d.fragment) return;
    const tmp = document.createElement("div");
    tmp.innerHTML = d.fragment;
    const pre = well.querySelector("pre.code-block");
    if (pre) {
      // New lines must go inside the block that is already there.
      while (tmp.firstChild) pre.appendChild(tmp.firstChild);
    } else {
      while (tmp.firstChild) well.appendChild(tmp.firstChild);
    }
  }

  // ---- raw mode: byte windows from the /result endpoint ----

  async function _rawFetch(d, well) {
    const st = rawState;
    try {
      const res = await getJSON("/api/tasks/" + enc + "/result?offset=" + st.off +
        "&max_bytes=" + (st.off === 0 ? _RAW_COLLAPSE : _RAW_WINDOW));
      if (rawState !== st || !st.el || !st.el.isConnected) return;
      st.acc += res.content;
      st.off = res.offset + res.content_bytes;
      st.full = !res.truncated;
      st.el.textContent = st.acc;
      const old = well.querySelector(".res-bar");
      if (old) old.remove();
      if (!st.full) {
        const bar = document.createElement("button");
        bar.type = "button";
        bar.className = "res-bar";
        bar.innerHTML = '<span class="res-chev">' + icon("chevDown") + "</span>" +
          '<span class="res-label">Show more — ' + fmtBytes(d.size) + " total</span>";
        bar.addEventListener("click", () => { if (!bar.disabled) { bar.disabled = true; _rawFetch(d, well); } });
        well.appendChild(bar);
      }
    } catch (err) {
      if (rawState === st) toast("Couldn't load more — showing first " + st.off + " bytes", { error: true });
    }
  }

  // The server sends the table with the extra rows hidden. The reveal bar is made here.
  function wireCsvReveal(well, d) {
    const table = well.querySelector(".csv-table");
    if (!table || !d || d.row_count == null || d.row_count <= _CSV_WINDOW) return;
    const bar = document.createElement("div");
    bar.className = "csv-bar";
    const chev = icon("chevDown").replace("<svg ", '<svg class="csv-chev" ');
    bar.innerHTML = chev +
      "Showing " + _CSV_WINDOW + " of " + d.row_count.toLocaleString("en-US") + " rows — " +
      '<button type="button">Show all</button>';
    bar.querySelector("button").addEventListener("click", () => {
      const all = table.classList.toggle("csv-all");
      bar.querySelector("button").textContent = all ? "Show less" : "Show all";
    });
    well.appendChild(bar);
  }

  // ---- file blocks ----

  function fileBlock(entry) {
    const block = document.createElement("div");
    const d = entry.present ? entry.descriptor : null;
    const binary = d && d.mode === "unsupported";
    block.className = "vp-file-block" +
      (entry.present ? " vp-file-present" : "") +
      (binary ? " vp-file-binary" : "");
    block.dataset.path = entry.path;
    block.appendChild(fileHead(entry));
    if (!entry.present) {
      block.appendChild(miniStateEl());
      return block;
    }
    if (entry.error === "load_error") {
      block.appendChild(loadErrorEl(entry.path));
      return block;
    }
    const well = fileWell(d);
    if (well) block.appendChild(well);
    return block;
  }

  function fileHead(entry) {
    const head = document.createElement("div");
    head.className = "vp-file-head";
    const idb = document.createElement("div");
    idb.className = "vp-file-id";
    const size = entry.size != null ? fmtBytes(entry.size) : null;
    const mtime = entry.mtime_local || null;
    const name = document.createElement("span");
    name.className = "vp-file-name";
    name.textContent = entry.name;
    name.title = entry.name;
    const row = document.createElement("div");
    row.className = "vp-name-row";
    row.appendChild(name);
    const d = entry.descriptor || null;
    if (d && d.mode === "native" && d.kind === "audio") {
      const chip = document.createElement("span");
      chip.className = "badge media";
      chip.textContent = "audio";
      row.appendChild(chip);
    }
    idb.appendChild(row);
    const path = document.createElement("span");
    path.className = "vp-file-path";
    path.textContent = entry.path;
    path.title = entry.path;
    idb.appendChild(path);
    if (size || mtime) {
      const meta = document.createElement("span");
      meta.className = "vp-file-meta";
      meta.textContent = size || "";
      if (size && mtime) {
        const t = document.createElement("span");
        t.className = "vp-file-mtime";
        t.textContent = " · " + mtime;
        meta.appendChild(t);
      }
      idb.appendChild(meta);
    }
    head.appendChild(idb);
    const acts = document.createElement("div");
    acts.className = "vp-file-actions";
    acts.appendChild(_act("/view/" + encodePath(entry.path), "open", "Open full-screen", true));
    const res = _act("/resources?select=" + encodePath(entry.path), "folder", "Open in Resources");
    res.dataset.spa = "";
    acts.appendChild(res);
    acts.appendChild(_act(dlHref(entry.path), "download", "Download"));
    head.appendChild(acts);
    return head;
  }

  function fileWell(d) {
    const mode = d.mode;
    if (mode === "doc" || mode === "code" || mode === "table" || mode === "plain") {
      const well = document.createElement("div");
      well.className = "vp-well vp-well-" + mode;
      _fragment(well, d.fragment);
      return well;
    }
    if (mode === "raw") {
      const well = document.createElement("div");
      well.className = "vp-well vp-well-raw";
      if (d.fragment != null) _fragment(well, d.fragment);
      else well.textContent = d.note || "";
      return well;
    }
    if (mode === "native") {
      if (d.kind === "image") {
        const stage = document.createElement("div");
        stage.className = "media-stage";
        const img = document.createElement("img");
        img.alt = "";
        img.src = d.url;
        stage.appendChild(img);
        return stage;
      }
      if (d.kind === "video") {
        const stage = document.createElement("div");
        stage.className = "media-stage";
        const v = document.createElement("video");
        v.setAttribute("controls", "");
        v.src = d.url;
        stage.appendChild(v);
        return stage;
      }
      const stage = document.createElement("div");
      stage.className = "media-stage media-stage-audio";
      const card = document.createElement("div");
      card.className = "audio-card";
      audioCards.add(initAudioCard(card, d.url, d.name));
      stage.appendChild(card);
      return stage;
    }
    if (mode === "frame") {
      const well = document.createElement("div");
      well.className = "vp-well vp-well-frame";
      if (d.kind === "pdf") {
        // The pdf viewer mounts into this container.
        const host = document.createElement("div");
        host.className = "vp-pdf";
        well.appendChild(host);
        mountPdfViewer(host, d.url);
        return well;
      }
      const ifr = document.createElement("iframe");
      ifr.src = d.url;
      if (d.kind === "html") ifr.setAttribute("sandbox", "allow-same-origin");
      well.appendChild(ifr);
      return well;
    }
    if (mode === "converting") {
      const el = document.createElement("div");
      el.className = "vp-converting";
      el.dataset.path = d.path || "";
      wireConverting(el, d);
      return el;
    }
    if (mode === "convert_failed") {
      const el = document.createElement("div");
      el.className = "vp-convert-failed";
      el.innerHTML =
        "<p>Couldn't convert this file</p>" +
        '<p class="vp-err">' + esc(d.error || "") + "</p>" +
        '<div class="vp-state-actions">' +
        '<button class="btn ghost" type="button" data-act="retry">Retry</button>' +
        '<a class="btn ghost" href="' + dlHref(d.path) + '">Download</a></div>';
      el.querySelector('[data-act="retry"]').addEventListener("click", () => retryConvert(d.path, el));
      return el;
    }
    return null; // A file that cannot be rendered is its head row alone.
  }

  // The file that is gone mid-session shows its GC'd state in place of its well.
  function miniStateEl() {
    const el = document.createElement("div");
    el.className = "mini-state";
    el.innerHTML =
      '<span class="vp-state-ic">' + icon("alert") + "</span>" +
      '<p class="mini-state-title">No longer in the resource.</p>' +
      '<p class="vp-state-sub">Garbage-collected (30-day retention).</p>';
    return el;
  }

  function loadErrorEl(path) {
    const el = _stateCard("alert", "Couldn't load file", "The file exists, but its preview failed to build.");
    const acts = document.createElement("div");
    acts.className = "vp-state-actions";
    acts.innerHTML = '<button class="btn ghost" type="button" data-act="retry">Retry</button>';
    el.appendChild(acts);
    el.querySelector('[data-act="retry"]').addEventListener("click", () => refetchFile(path, el));
    return el;
  }

  function blockGone(el) {
    const block = el.closest(".vp-file-block");
    if (!block) return;
    block.classList.remove("vp-file-present");
    el.replaceWith(miniStateEl());
  }

  // Swaps a block's state or well for a freshly described one.
  function swapBlockContent(el, next) {
    const block = el.closest(".vp-file-block");
    const well = fileWell(next);
    if (well == null) {
      el.remove();
      if (block) block.classList.add("vp-file-binary");
      return;
    }
    if (block) block.classList.remove("vp-file-binary");
    el.replaceWith(well);
  }

  // Waits for the conversion to finish, then draws the new well in place.
  function wireConverting(el, d) {
    const [label, note] = _SHEET_KINDS.has(d.kind) ? _CONVERTING.sheet : _CONVERTING.office;
    el.innerHTML =
      '<div class="cv-spin"></div>' +
      '<div class="cv-label">' + label + "</div>" +
      '<div class="cv-note">' + note + "</div>";
    const viewPath = "/api/view/" + encodePath(d.path);
    const tick = async () => {
      if (!el.isConnected) { clearInterval(el._timer); return; }
      let next;
      try {
        next = await getJSON(viewPath);
      } catch (err) {
        // The file went away while it was being converted.
        if (err instanceof ApiError && err.status === 404) { clearInterval(el._timer); blockGone(el); return; }
        return; // Keep checking when the network hiccups.
      }
      if (!el.isConnected) { clearInterval(el._timer); return; }
      if (next.mode === "converting") return; // Still working, keep checking.
      clearInterval(el._timer);
      swapBlockContent(el, next);
    };
    el._timer = setInterval(tick, _POLL_MS);
  }

  // Asks the server to convert the file again, then shows what comes back.
  async function retryConvert(path, el) {
    let next;
    try {
      next = await postJSON("/api/view/" + encodePath(path) + "/retry");
    } catch {
      toast("Retry failed", { error: true });
      return;
    }
    if (!el.isConnected) return;
    swapBlockContent(el, next);
  }

  // Builds the file's preview again, then shows what comes back. A 404 means
  // the file is gone, so the block settles on the expired state.
  async function refetchFile(path, el) {
    let next;
    try {
      next = await getJSON("/api/view/" + encodePath(path));
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) { blockGone(el); return; }
      return;
    }
    if (!el.isConnected) return;
    swapBlockContent(el, next);
  }

  return {
    onTerminal() { renderP = render(); },
  };
}
