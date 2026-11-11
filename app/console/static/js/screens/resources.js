import { getJSON, postJSON, davHref, dlHref, encodePath } from "../api.js";
import { esc, fmtBytes, fmtBytesShort, monthDayHmFromLocal, dateLabel, fileCountLabel } from "../format.js";
import { icon, fileGlyph } from "../ui.js";
import { copyText, toast } from "../toast.js";
import { getState } from "../state.js";
import { syncGlyphs } from "../theme.js";
import { renderView, showWellSkeleton } from "../viewer.js";
import { renderGallery, initGalleryPoll, compactSlot, sortEntries } from "../gallery.js";
import { watchThumbs } from "../tree-thumbs.js";
import { entryView, entryState, pushEntry, replaceEntry } from "../router.js";

const _SEARCH_DEBOUNCE_MS = 200;
const _SEARCH_SKELETON_MS = 300;

// The meta line under the selected folder name. The size is left out when it is zero.
// The entry comes from the folder's own row in its parent listing.
function folderHeadMeta(e) {
  const parts = [fileCountLabel(e.file_count)];
  if (e.total_size) parts.push(fmtBytesShort(e.total_size));
  if (e.mtime_local) parts.push("Modified " + monthDayHmFromLocal(e.mtime_local));
  return parts.join(" · ");
}

export async function renderResources(el, _params) {
  el.innerHTML =
    '<div class="r-split">' +
      '<div class="tree-pane">' +
        '<div class="tree-top">' +
          '<div class="input-wrap">' + icon("search") +
            '<input class="input" id="r-search" type="text" placeholder="Search files…" autocomplete="off" spellcheck="false">' +
          "</div>" +
          '<button class="icon-btn" id="r-refresh" title="Refresh tree">' + icon("refresh") + "</button>" +
        "</div>" +
        '<div class="crumbs" id="r-crumbs"></div>' +
        '<div class="tree" id="r-tree"></div>' +
      "</div>" +
      '<div class="preview-pane" id="r-preview">' +
        '<div class="pv-empty">' + icon("folder") +
          '<span class="t1">Select a file</span>' +
          '<span class="t2">Choose a file from the tree to preview it here.</span>' +
        "</div>" +
      "</div>" +
    "</div>";

  const tree = el.querySelector("#r-tree");
  const crumbs = el.querySelector("#r-crumbs");
  const preview = el.querySelector("#r-preview");
  const searchIn = el.querySelector("#r-search");
  const refreshBtn = el.querySelector("#r-refresh");
  const cache = new Map(); // Entries for each path, the empty path is the root.
  const open = new Set();  // Folder paths that are open.
  let selected = null;     // The selected file or folder.
  let view = entryView();
  let previewEntry = false; // True when the open preview has its own history entry.
  let searchMode = false;
  let matches = [];
  let truncated = false;
  let hoverPath = null;
  let searchSeq = 0;

  function fileMeta(e) {
    return fmtBytes(e.size) + (e.mtime ? " · " + dateLabel(e.mtime, e.mtime_local) : "");
  }

  function folderMeta(e, topLevel) {
    return fileCountLabel(e.file_count) +
      (topLevel && e.total_size ? " · " + fmtBytesShort(e.total_size) : "");
  }

  function folderGlyph(isOpen) {
    return isOpen
      ? icon("folderSm").replace("#6f7d90", "#8b95a7")
      : icon("folderSm");
  }

  function folderRow(e, topLevel, depth) {
    const p = e.path;
    const row = document.createElement("div");
    row.className = "t-row folder" + (selected && selected.type === "dir" && selected.path === p ? " selected" : "");
    row.dataset.path = p;
    row.style.setProperty("--t-depth", depth);
    const isOpen = open.has(p);
    row.innerHTML =
      '<button class="t-caret' + (isOpen ? " open" : "") + '" title="' + (isOpen ? "Collapse" : "Expand") + '">' + icon("caret") + "</button>" +
      folderGlyph(isOpen) +
      '<span class="t-name">' + esc(e.name) + "/</span>" +
      (e.name === "upload" ? '<span class="rw-pill" title="The only writable folder (WebDAV PUT)">RW</span>' : "") +
      '<span class="t-meta">' + esc(folderMeta(e, topLevel)) + "</span>";
    const box = document.createElement("div");
    box.className = "t-children" + (isOpen ? " open" : "");
    box.dataset.children = p;
    row.querySelector(".t-caret").addEventListener("click", (ev) => {
      ev.stopPropagation();
      toggleFolder(p, row, box);
    });
    row.addEventListener("click", (ev) => {
      ev.stopPropagation();
      toggleFolder(p, row, box);
      selectFolder(e);
    });
    row.addEventListener("mouseenter", () => { hoverPath = p; });
    row.addEventListener("mouseleave", () => { if (hoverPath === p) hoverPath = null; });
    return { row, box, p };
  }

  async function toggleFolder(p, row, box) {
    if (open.has(p)) {
      open.delete(p);
      box.classList.remove("open");
      row.querySelector(".t-caret").classList.remove("open");
      return;
    }
    open.add(p);
    box.classList.add("open");
    row.querySelector(".t-caret").classList.add("open");
    if (cache.has(p)) { fillChildren(p, box); watch.start(); return; }
    box.replaceChildren();
    for (let i = 0; i < 3; i++) {
      const sk = document.createElement("div");
      sk.className = "t-skel-row";
      box.appendChild(sk);
    }
    const t0 = Date.now();
    try {
      const body = await getJSON("/api/resources?" + new URLSearchParams({ path: p }));
      cache.set(p, body.entries);
      const rest = _SEARCH_SKELETON_MS - (Date.now() - t0);
      if (rest > 0) await new Promise((r) => setTimeout(r, rest));
      fillChildren(p, box);
      watch.start();
    } catch (err) {
      open.delete(p);
      box.classList.remove("open");
      row.querySelector(".t-caret").classList.remove("open");
      box.replaceChildren();
      toast("Load folder failed: " + (err.message || err), { error: true });
    }
  }

  async function openPathChain(dirPath, seq) {
    let dir = "";
    for (const name of dirPath.split("/")) {
      dir = dir ? dir + "/" + name : name;
      try {
        if (!cache.has(dir)) {
          const body = await getJSON("/api/resources?" + new URLSearchParams({ path: dir }));
          cache.set(dir, body.entries);
        }
      } catch {
        return dir;
      }
      if (seq !== searchSeq) return dir;
      open.add(dir);
    }
    return dir;
  }

  function fillChildren(p, box) {
    box.replaceChildren();
    const depth = p.split("/").length;
    for (const e of sortEntries(cache.get(p) || [])) {
      if (e.type === "dir") {
        const { row, box: childBox } = folderRow(e, false, depth);
        box.appendChild(row);
        box.appendChild(childBox);
        if (open.has(e.path)) fillChildren(e.path, childBox);
      } else {
        box.appendChild(fileRow(e, depth));
      }
    }
  }

  function fileRow(e, depth) {
    const row = document.createElement("div");
    row.className = "t-row file" + (selected && selected.type === "file" && selected.path === e.path ? " selected" : "");
    row.dataset.path = e.path;
    row.dataset.thumb = e.thumb ? e.thumb.status : "none";
    row.style.setProperty("--t-depth", depth);
    row.innerHTML =
      '<span class="t-caret leaf"></span>' + compactSlot(e) +
      '<span class="t-name">' + esc(e.name) + "</span>" +
      '<span class="t-meta">' + esc(fileMeta(e)) + "</span>";
    row.querySelector(".t-name").title = e.path;
    row.addEventListener("click", (ev) => {
      ev.stopPropagation();
      selectFile(e);
    });
    row.addEventListener("mouseenter", () => { hoverPath = e.path; });
    row.addEventListener("mouseleave", () => { if (hoverPath === e.path) hoverPath = null; });
    return row;
  }

  function renderTree() {
    tree.replaceChildren();
    for (const e of sortEntries(cache.get("") || [])) {
      if (e.type === "dir") {
        const { row, box } = folderRow(e, true, 0);
        tree.appendChild(row);
        tree.appendChild(box);
        if (open.has(e.path)) fillChildren(e.path, box);
      } else {
        tree.appendChild(fileRow(e, 0));
      }
    }
    watch.start();
  }

  function renderSearch() {
    tree.replaceChildren();
    if (matches.length === 0) {
      const d = document.createElement("div");
      d.className = "t-nomatch";
      d.innerHTML = 'No matches for <span class="q">"' + esc(searchIn.value.trim()) + '"</span>';
      tree.appendChild(d);
      watch.start();
      return;
    }
    for (const m of matches) {
      const anc = m.matched_ancestor_paths || [];
      let prefix = "";
      for (let i = 0; i < anc.length; i++) {
        prefix = prefix ? prefix + "/" + anc[i] : anc[i];
        const row = document.createElement("div");
        row.className = "t-row folder dimmed";
        row.style.setProperty("--t-depth", i);
        row.innerHTML =
          '<span class="t-caret leaf"></span>' + folderGlyph(open.has(prefix)) +
          '<span class="t-name">' + esc(anc[i]) + "/</span>";
        tree.appendChild(row);
      }
      if (m.type === "dir") {
        const row = document.createElement("div");
        row.className = "t-row folder";
        row.style.setProperty("--t-depth", anc.length);
        row.innerHTML =
          '<span class="t-caret leaf"></span>' + folderGlyph(false) +
          '<span class="t-name">' + esc(m.name) + "/</span>";
        row.addEventListener("click", async (ev) => {
          ev.stopPropagation();
          searchIn.value = "";
          searchMode = false;
          const seq = ++searchSeq;
          await openPathChain(m.path, seq);
          if (seq !== searchSeq) return;
          renderTree();
        });
        tree.appendChild(row);
      } else {
        tree.appendChild(fileRow(m, anc.length));
      }
    }
    if (truncated) {
      const d = document.createElement("div");
      d.className = "t-trunc";
      d.textContent = "Results truncated";
      tree.appendChild(d);
    }
    watch.start();
  }

  function renderCrumbs() {
    crumbs.replaceChildren();
    const seg = (label, path) => {
      const s = document.createElement("span");
      s.className = "seg";
      s.textContent = label;
      s.title = "Go to " + label;
      s.addEventListener("click", () => {
        searchIn.value = "";
        searchSeq += 1;
        searchMode = false;
        backToTree();
        open.add(path);
        renderTree();
        renderCrumbs();
      });
      return s;
    };
    const sep = () => {
      const s = document.createElement("span");
      s.className = "sep";
      s.textContent = "/";
      return s;
    };
    if (!selected) return; // The crumb row stays empty at the root.
    const parent = selected.path.split("/").slice(0, -1);
    crumbs.appendChild(seg("resource", ""));
    parent.forEach((segName, i) => {
      crumbs.appendChild(sep());
      if (i === parent.length - 1) {
        const c = document.createElement("span");
        c.className = "cur";
        c.textContent = segName;
        crumbs.appendChild(c);
      } else {
        crumbs.appendChild(seg(segName, parent.slice(0, i + 1).join("/")));
      }
    });
  }

  // File names can hold characters that a CSS selector cannot match.
  function rowByPath(cls, path) {
    for (const row of tree.querySelectorAll(".t-row." + cls)) {
      if (row.dataset.path === path) return row;
    }
    return null;
  }

  function selectFile(entry, push = true) {
    selected = entry;
    for (const row of tree.querySelectorAll(".t-row.selected")) row.classList.remove("selected");
    const row = rowByPath("file", entry.path);
    if (row) row.classList.add("selected");
    renderCrumbs();
    previewSelect(entry);
    openPreview(entry, push);
  }

  // Picks a folder and shows it in the preview pane.
  // Only one file or folder is selected at a time.
  function selectFolder(entry, push = true) {
    selected = entry;
    for (const row of tree.querySelectorAll(".t-row.selected")) row.classList.remove("selected");
    const row = rowByPath("folder", entry.path);
    if (row) row.classList.add("selected");
    renderCrumbs();
    previewSelectFolder(entry);
    openPreview(entry, push);
  }

  // Clears the preview pane. A running folder poll stops here.
  function renderPreviewEmpty() {
    for (const fn of cleanups.splice(0)) fn();
    pvSeq += 1;
    folderRefresh = null;
    lastDescriptor = null;
    document.body.classList.remove("r-preview");
    preview.innerHTML =
      '<div class="pv-empty">' + icon("folder") +
      '<span class="t1">Select a file</span>' +
      '<span class="t2">Choose a file from the tree to preview it here.</span></div>';
  }

  const cleanups = [];
  let pvSeq = 0;

  function previewSelect(entry) {
    for (const fn of cleanups.splice(0)) fn();
    const seq = ++pvSeq;
    folderRefresh = null;
    lastDescriptor = null;
    const viewUrl = (getState().davBase || location.origin) + "/view/" + encodePath(entry.path);
    preview.innerHTML =
      '<header class="pv-mbar">' +
        '<button class="icon-btn" id="pv-back" title="Back to list">' + icon("chevL") + "</button>" +
        '<span class="pv-mtitle">' + esc(entry.name) + "</span>" +
        '<span class="spacer"></span>' +
        '<button class="icon-btn" data-theme-toggle title="Switch theme"></button>' +
      "</header>" +
      '<div class="pv-m-meta"><span class="pv-m-path">' + esc(entry.path) +
        '<button class="icon-btn" id="pv-copypath-m" title="Copy path">' + icon("copy") + "</button></span>" +
        '<div class="pv-m-sub">' + esc(fileMeta(entry)) + "</div></div>" +
      '<div class="pv-head">' +
        '<span class="pv-tile">' + fileGlyph(entry.name) + "</span>" +
        '<div class="pv-id">' +
          '<div class="pv-name">' + esc(entry.name) + "</div>" +
          '<div class="pv-path"><span class="pv-path-t" title="' + esc(entry.path) + '">' + esc(entry.path) + "</span>" +
            '<button class="icon-btn" id="pv-copypath" title="Copy path">' + icon("copy") + "</button></div>" +
          '<div class="pv-meta" id="pv-meta">' + esc(fileMeta(entry)) +
            (entry.mtime_local ? " · Modified " + monthDayHmFromLocal(entry.mtime_local) : "") + "</div>" +
        "</div>" +
        '<div class="pv-actions">' +
          '<a class="btn solid" id="pv-dl" href="' + esc(dlHref(entry.path)) + '">Download</a>' +
          '<button class="btn ghost" id="pv-expand" title="Expand to full screen (f)">' + icon("maximize") + '<span class="lbl">Expand</span></button>' +
          '<a class="btn ghost" id="pv-openviewer" href="/view/' + esc(encodePath(entry.path)) + '" target="_blank" rel="noopener" title="Open the resource page in a new tab">' + icon("viewer") + '<span class="lbl">Open viewer</span></a>' +
          '<button class="btn ghost" id="pv-copylink">Copy link</button>' +
          '<a class="btn ghost" id="pv-openraw" href="' + esc(davHref(entry.path)) + '" target="_blank" rel="noopener" style="display:none">Open raw</a>' +
        "</div>" +
      "</div>" +
      '<div class="pv-well" id="pv-well"><div class="pv-note">' + icon("box") + "Loading preview…</div></div>" +
      '<div class="pv-m-actions">' +
        '<a class="btn solid" href="' + esc(dlHref(entry.path)) + '">Download</a>' +
        '<a class="icon-btn" id="pv-openviewer-m" href="/view/' + esc(encodePath(entry.path)) + '" title="Open the resource page">' + icon("viewer") + "</a>" +
      "</div>";
    syncGlyphs();
    el.querySelector("#pv-back").addEventListener("click", backToTree);
    el.querySelector("#pv-copypath").addEventListener("click", () => copyText(entry.path, "Path copied"));
    el.querySelector("#pv-copypath-m").addEventListener("click", () => copyText(entry.path, "Path copied"));
    el.querySelector("#pv-expand").addEventListener("click", () => toggleFullScreen());
    el.querySelector("#pv-copylink").addEventListener("click", () => copyText(viewUrl, "Link copied"));
    renderWell(entry, seq);
    applyFullScreenChrome();
  }

  // Shows a folder in the preview pane with a gallery of what is inside.
  // Files that go away stay in the gallery with their last known info.
  function previewSelectFolder(entry) {
    for (const fn of cleanups.splice(0)) fn();
    const seq = ++pvSeq;
    lastDescriptor = null;
    const goneSet = new Map();
    let lastEntries = [];
    const dir = entry.path;
    const parent = dir.split("/").slice(0, -1).join("/");

    function renderShell(fe) {
      preview.innerHTML =
        '<header class="pv-mbar">' +
          '<button class="icon-btn" id="pv-back" title="Back to list">' + icon("chevL") + "</button>" +
          '<span class="pv-mtitle">' + esc(fe.name) + "</span>" +
          '<span class="spacer"></span>' +
          '<button class="icon-btn" data-theme-toggle title="Switch theme"></button>' +
        "</header>" +
        '<div class="pv-head">' +
          '<span class="pv-tile">' + icon("folder") + "</span>" +
          '<div class="pv-id">' +
            '<div class="pv-name">' + esc(fe.name) + "/</div>" +
            '<div class="pv-path">' + esc(fe.path) + "</div>" +
            '<div class="pv-meta">' + esc(folderHeadMeta(fe)) +
              (fe.name === "upload" ? ' <span class="rw-pill" title="The only writable folder (WebDAV PUT)">RW</span>' : "") +
            "</div>" +
          "</div>" +
          '<div class="pv-actions">' +
            '<button class="btn ghost" id="pv-expand" title="Expand to full screen (f)">' + icon("maximize") + '<span class="lbl">Expand</span></button>' +
            '<button class="icon-btn" id="pv-refresh" title="Refresh">' + icon("refresh") + "</button>" +
          "</div>" +
        "</div>" +
        '<div class="pv-well" id="pv-well"></div>';
      preview.querySelector("#pv-back").addEventListener("click", backToTree);
      preview.querySelector("#pv-refresh").addEventListener("click", refreshFolder);
      preview.querySelector("#pv-expand").addEventListener("click", () => toggleFullScreen());
      preview.querySelector("#pv-well").addEventListener("gallery:select", onCardSelect);
      preview.querySelector("#pv-well").addEventListener("gallery:open", onCardOpen);
      syncGlyphs();
      applyFullScreenChrome();
    }

    function selectCardEntry(e) {
      if (e.type === "dir") selectFolder(e);
      else selectFile(e);
    }

    function onCardSelect(ev) {
      if (seq !== pvSeq) return;
      selectCardEntry(ev.detail);
    }

    // Opens a card in full screen. While already full screen only the
    // selection changes.
    function onCardOpen(ev) {
      if (seq !== pvSeq) return;
      selectCardEntry(ev.detail);
      enterFullScreen();
    }

    // Shown when the folder is gone. Refresh tries again.
    function renderFolderGone() {
      preview.innerHTML =
        '<div class="pv-error">' + icon("alert") +
        '<span class="t1">Folder no longer exists</span>' +
        '<span class="t2">It may have been garbage-collected (30-day retention).</span>' +
        '<button class="btn ghost" id="pv-refresh">Refresh</button></div>';
      preview.querySelector("#pv-refresh").addEventListener("click", refreshFolder);
      applyFullScreenChrome();
    }

    // Loads the folder again. It stays on the gone screen when the folder is
    // still missing.
    async function refreshFolder() {
      const before = pvSeq;
      goneSet.clear();
      let fe = null;
      try {
        const pbody = await getJSON("/api/resources?" + new URLSearchParams({ path: parent }));
        cache.set(parent, pbody.entries);
        fe = pbody.entries.find((e) => e.path === dir && e.type === "dir") || null;
      } catch { fe = null; }
      if (before !== pvSeq) return;
      if (!fe) fe = (cache.get(parent) || []).find((e) => e.path === dir && e.type === "dir") || entry;
      previewSelectFolder(fe);
    }

    async function mountGallery() {
      const well = preview.querySelector("#pv-well");
      if (!well) return;
      well.innerHTML =
        '<div class="gallery-grid">' +
        [0, 1, 2, 3, 4, 5].map(() =>
          '<div class="thumb-card skeleton"><div class="thumb-1610"></div>' +
          '<div class="thumb-name"></div><div class="thumb-meta"></div></div>').join("") +
        "</div>";
      let body;
      try {
        body = await getJSON("/api/resources?" + new URLSearchParams({ path: dir }));
      } catch (err) {
        if (seq !== pvSeq) return;
        if (err && err.status === 404) { renderFolderGone(); return; }
        toast("Load folder failed: " + (err.message || err), { error: true });
        return;
      }
      if (seq !== pvSeq) return;
      cache.set(dir, body.entries);
      lastEntries = body.entries;
      renderGallery(body.entries, goneSet, well, {
        onMounted: () => {
          if (seq !== pvSeq) return;
          postJSON("/api/resources/thumbs?" + new URLSearchParams({ path: dir })).catch(() => {});
        },
      });
      const stop = initGalleryPoll(well, dir, {
        onGone: (p) => {
          const last = lastEntries.find((e) => e.path === p && e.type === "file");
          if (last) goneSet.set(p, last);
        },
        onEntries: (entries) => {
          if (seq !== pvSeq) return;
          for (const e of entries) goneSet.delete(e.path);
          lastEntries = entries;
          renderGallery(entries, goneSet, preview.querySelector("#pv-well"));
          applyFullScreenChrome();
        },
        onFolderGone: () => { if (seq === pvSeq) renderFolderGone(); },
      });
      cleanups.push(stop);
    }

    folderRefresh = refreshFolder;
    renderShell(entry);
    mountGallery();
  }

  // Full screen is a layout state, not a route.
  // Going in and out does not touch the well, so a preview that is still
  // loading keeps going.
  let fullScreen = false;
  let preFullCollapsed = false;
  let lastDescriptor = null; // Info about the shown file. It decides whether the f key works.
  let folderRefresh = null;  // Loads the selected folder again, null when a file is selected.

  // The buttons on the right side of the full screen bar.
  function fsToolbarCluster() {
    if (selected.type === "dir") {
      return folderRefresh
        ? '<button class="icon-btn" id="fs-refresh" title="Refresh">' + icon("refresh") + "</button>"
        : "";
    }
    return (
      '<button class="btn ghost" id="fs-copylink">Copy link</button>' +
      '<a class="btn ghost" id="fs-openraw" href="' + esc(davHref(selected.path)) + '" target="_blank" rel="noopener" style="display:none">Open raw</a>' +
      '<a class="btn solid" id="fs-dl" href="' + esc(dlHref(selected.path)) + '">Download</a>'
    );
  }

  function wireFsCluster(bar) {
    const copylink = bar.querySelector("#fs-copylink");
    if (copylink) {
      const viewUrl = (getState().davBase || location.origin) + "/view/" + encodePath(selected.path);
      copylink.addEventListener("click", () => copyText(viewUrl, "Link copied"));
    }
    const copypath = bar.querySelector("#fs-copypath");
    if (copypath) copypath.addEventListener("click", () => copyText(selected.path, "Path copied"));
    const refresh = bar.querySelector("#fs-refresh");
    if (refresh) refresh.addEventListener("click", () => { if (folderRefresh) folderRefresh(); });
    // Sets the Open raw button from the file info loaded so far.
    applyFrameRaw(bar.querySelector("#fs-openraw"));
  }

  // Puts the full screen bar back and points the URL at the selected entry.
  // Call this after every preview re-render, or the old header shows up on
  // top of the bar.
  function applyFullScreenChrome() {
    if (!fullScreen || !selected) return;
    if (!preview.querySelector(".fs-toolbar")) {
      const bar = document.createElement("div");
      bar.className = "fs-toolbar";
      // The bar holds three parts so it still fits on a narrow screen.
      bar.innerHTML =
        '<button class="btn ghost" id="fs-exit" title="Exit full screen (Esc)">' + icon("minimize") + "Exit full screen</button>" +
        '<span class="fs-id">' +
          '<span class="fs-name">' + esc(selected.name) + (selected.type === "dir" ? "/" : "") + "</span>" +
          '<span class="fs-sub" title="' + esc(selected.path) + '">' + esc(selected.path) + "</span>" +
          (selected.type === "file" ? '<button class="icon-btn" id="fs-copypath" title="Copy path">' + icon("copy") + "</button>" : "") +
        "</span>" +
        '<span class="spacer"></span>' +
        '<span class="fs-cluster">' + fsToolbarCluster() + "</span>";
      preview.insertBefore(bar, preview.querySelector("#pv-well"));
      bar.querySelector("#fs-exit").addEventListener("click", toggleFullScreen);
      wireFsCluster(bar);
    }
    for (const h of preview.querySelectorAll(".pv-mbar, .pv-head")) h.style.display = "none";
    replaceEntry(entryState(), "/resources?" + new URLSearchParams({ select: selected.path, full: "1" }).toString());
  }

  function enterFullScreen() {
    if (fullScreen || !selected) return;
    // Full screen never opens on a small screen.
    if (window.matchMedia("(max-width: 767px)").matches) return;
    preFullCollapsed = document.documentElement.classList.contains("sb-collapsed");
    fullScreen = true;
    document.body.classList.add("full-screen");
    document.documentElement.classList.add("sb-collapsed");
    applyFullScreenChrome();
  }

  function clearFullScreenChrome() {
    fullScreen = false;
    document.body.classList.remove("full-screen");
    document.documentElement.classList.toggle("sb-collapsed", preFullCollapsed);
    const bar = preview.querySelector(".fs-toolbar");
    if (bar) bar.remove();
  }

  function exitFullScreen() {
    if (!fullScreen) return;
    clearFullScreenChrome();
    for (const h of preview.querySelectorAll(".pv-mbar, .pv-head")) h.style.display = "";
    const qs = new URLSearchParams();
    if (selected) qs.set("select", selected.path);
    replaceEntry(entryState(), "/resources" + (qs.toString() ? "?" + qs.toString() : ""));
  }

  function toggleFullScreen() {
    if (fullScreen) exitFullScreen();
    else enterFullScreen();
  }

  // Shows the preview. A new preview gets its own history entry.
  // Switching entries replaces the current one.
  function openPreview(entry, push) {
    document.body.classList.add("r-preview");
    if (!push) {
      view.selection = entry.path;
      return;
    }
    if (previewEntry) {
      replaceEntry({ resource: entry.path });
      view.selection = entry.path;
      return;
    }
    pushEntry({ resource: entry.path });
    previewEntry = true;
    view = entryView();
    view.selection = entry.path;
  }

  // The saved selection is a path. Its entry comes from the cached listing.
  function _entryByPath(path) {
    if (!path) return null;
    const parent = path.split("/").slice(0, -1).join("/");
    const entries = cache.get(parent) || [];
    return entries.find((e) => e.path === path) || null;
  }

  function closePreview() {
    previewEntry = false;
    view.selection = null;
    selected = null;
    for (const row of tree.querySelectorAll(".t-row.selected")) row.classList.remove("selected");
    replaceEntry({}, "/resources");
    renderCrumbs();
    renderPreviewEmpty();
  }

  function backToTree() {
    if (previewEntry) history.back();
    else closePreview();
  }

  // The note shown when a file is gone.
  function goneNote(withRefresh) {
    return (
      '<div class="pv-error">' + icon("alert") +
      '<span class="t1">File no longer exists</span>' +
      '<span class="t2">It may have been garbage-collected (30-day retention).</span>' +
      (withRefresh ? '<button class="btn ghost" id="pv-refresh">Refresh</button>' : "") +
      "</div>"
    );
  }

  function goneState(well, entry) {
    well.innerHTML = goneNote(true);
    well.querySelector("#pv-refresh").addEventListener("click", async () => {
      try {
        const dir = entry.path.split("/").slice(0, -1).join("/");
        const body = await getJSON("/api/resources?" + new URLSearchParams({ path: dir }));
        cache.set(dir, body.entries);
        const still = body.entries.find((e) => e.path === entry.path && e.type === "file");
        if (still) {
          selectFile(still);
        } else {
          goneState(well, entry);
        }
      } catch (err) {
        // Leaves full screen when the folder is gone too.
        if (fullScreen && err && err.status === 404) exitFullScreen();
        goneState(well, entry);
      }
    });
  }

  async function renderWell(entry, seq) {
    const well = preview.querySelector("#pv-well");
    showWellSkeleton(well);
    let descriptor;
    try {
      descriptor = await getJSON("/api/view/" + encodePath(entry.path));
    } catch (err) {
      if (seq !== pvSeq) return;
      if (err && err.status === 404) { goneState(well, entry); return; }
      loadErrorState(well, entry);
      return;
    }
    if (seq !== pvSeq) return;
    renderView(descriptor, well, (fresh) => {
      if (seq !== pvSeq) return;
      applyDescriptorActions(fresh);
      applyFullScreenChrome();
    });
    applyDescriptorActions(descriptor);
    applyFullScreenChrome();
  }

  // True when a file has nothing to expand. The Expand button is hidden and
  // the f key does nothing.
  function expandInert(descriptor) {
    return (
      descriptor.mode === "unsupported" ||
      (descriptor.mode === "raw" && descriptor.fragment === null)
    );
  }

  // Shows the Open raw button only for a file that is shown in a frame.
  function applyFrameRaw(openRaw) {
    const isFrame = lastDescriptor && lastDescriptor.mode === "frame";
    if (openRaw) openRaw.style.display = isFrame ? "" : "none";
  }

  // Updates the header buttons that depend on the loaded file info.
  function applyDescriptorActions(descriptor) {
    lastDescriptor = descriptor;
    const expand = preview.querySelector("#pv-expand");
    const expandHidden = expandInert(descriptor);
    if (expand) {
      expand.style.display = expandHidden ? "none" : "";
      if (!expandHidden) expand.disabled = descriptor.mode === "converting";
    }
    applyFrameRaw(preview.querySelector("#pv-openraw"));
    applyFrameRaw(preview.querySelector("#fs-openraw"));
  }

  function loadErrorState(well, entry) {
    well.innerHTML =
      '<div class="pv-error">' + icon("alert") +
      '<span class="t1">Couldn\'t load preview</span>' +
      '<span class="t2">The preview could not be loaded. Try again.</span>' +
      '<button class="btn ghost" id="pv-retry">Retry</button></div>';
    well.querySelector("#pv-retry").addEventListener("click", () => renderWell(entry, ++pvSeq));
  }

  let debounce = null;
  searchIn.addEventListener("input", () => {
    clearTimeout(debounce);
    debounce = setTimeout(async () => {
      const q = searchIn.value.trim();
      if (!q) {
        searchSeq += 1;
        searchMode = false;
        renderTree();
        return;
      }
      const seq = ++searchSeq;
      try {
        const body = await getJSON("/api/resources/search?" + new URLSearchParams({ q, limit: "200" }));
        if (seq !== searchSeq) return;
        matches = body.matches;
        truncated = Boolean(body.truncated);
        searchMode = true;
        renderSearch();
      } catch (err) {
        if (seq !== searchSeq) return;
        toast("Search failed: " + (err.message || err), { error: true });
      }
    }, _SEARCH_DEBOUNCE_MS);
  });

  refreshBtn.addEventListener("click", async () => {
    refreshBtn.classList.add("spin");
    setTimeout(() => refreshBtn.classList.remove("spin"), 400);
    cache.clear();
    try {
      const body = await getJSON("/api/resources");
      cache.set("", body.entries);
      for (const p of [...open]) {
        const b2 = await getJSON("/api/resources?" + new URLSearchParams({ path: p }));
        cache.set(p, b2.entries);
      }
      if (!searchMode) renderTree();
    } catch (err) {
      toast("Refresh failed: " + (err.message || err), { error: true });
    }
  });

  document.addEventListener("keydown", onKey);
  function onKey(ev) {
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    const t = ev.target;
    const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
    if (ev.key === "/" && !typing) {
      ev.preventDefault();
      searchIn.focus();
      return;
    }
    if (typing || searchMode) return;
    if (ev.key === "f") {
      // The f key does nothing for a file with nothing to expand.
      if (!fullScreen && selected && selected.type === "file" && lastDescriptor &&
        expandInert(lastDescriptor)) return;
      toggleFullScreen();
      return;
    }
    if (ev.key === "Escape" && fullScreen) toggleFullScreen();
    const files = [...tree.querySelectorAll(".t-row.file")].filter((r) => !r.closest(".t-children:not(.open)"));
    if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
      ev.preventDefault();
      const idx = files.findIndex((r) => r.classList.contains("selected"));
      const next = ev.key === "ArrowDown" ? Math.min(files.length - 1, idx + 1) : Math.max(0, idx <= 0 ? 0 : idx - 1);
      const entry = entryByPath(files[next] ? files[next].dataset.path : null);
      if (entry) selectFile(entry);
    }
    if ((ev.key === "ArrowRight" || ev.key === "ArrowLeft") && hoverPath) {
      const row = rowByPath("folder", hoverPath);
      const box = row ? row.nextElementSibling : null;
      if (row && box) toggleFolder(hoverPath, row, box);
    }
  }
  function entryByPath(p) {
    if (!p) return null;
    for (const entries of cache.values()) {
      const hit = entries.find((e) => e.path === p && e.type === "file");
      if (hit) return hit;
    }
    return null;
  }

  const _GONE_NOTE = "no longer exists";

  // The folder a file row lives in. Root files live in the empty path.
  function rowThumbDir(row) {
    return row.dataset.path.split("/").slice(0, -1).join("/");
  }

  // The folders whose visible file rows are still preparing.
  // Rows inside a collapsed folder are out of scope.
  function thumbDirsInScope() {
    const dirs = new Set();
    for (const row of tree.querySelectorAll('.t-row.file[data-thumb="pending"], .t-row.file[data-thumb="generating"]')) {
      if (row.closest(".t-children:not(.open)")) continue;
      dirs.add(rowThumbDir(row));
    }
    return dirs;
  }

  // Swaps a row's slot in place. The row keeps its position, scroll, and
  // selection. Only the slot and the meta line change.
  function applyDirThumbs(dir, entries) {
    cache.set(dir, entries);
    const byPath = new Map(entries.map((e) => [e.path, e]));
    for (const row of tree.querySelectorAll(".t-row.file")) {
      if (row.closest(".t-children:not(.open)")) continue;
      if (rowThumbDir(row) !== dir) continue;
      const meta = row.querySelector(".t-meta");
      const e = byPath.get(row.dataset.path);
      if (!e) {
        // The file went away while the folder was open. It stays, dimmed.
        row.classList.add("dimmed");
        row.dataset.thumb = "none";
        if (meta) meta.textContent = _GONE_NOTE;
        continue;
      }
      row.classList.remove("dimmed");
      row.dataset.thumb = e.thumb ? e.thumb.status : "none";
      if (meta) meta.textContent = fileMeta(e);
      const slot = row.querySelector(".thumb-1610");
      if (slot) {
        const wrap = document.createElement("span");
        wrap.innerHTML = compactSlot(e);
        slot.replaceWith(wrap.firstElementChild);
      }
    }
  }

  const watch = watchThumbs({
    active: () => true,
    dirsInScope: thumbDirsInScope,
    applyDir: applyDirThumbs,
  });

  // The entry state names the preview this entry represents. The URL query
  // still names a push-free selection, and the entry view the saved
  // selection of a revisited entry.
  const mountState = entryState();
  previewEntry = Boolean(mountState.resource);
  const _selParams = new URLSearchParams(location.search);
  const want = mountState.resource
    ? (view.selection || mountState.resource)
    : (_selParams.get("select") || _selParams.get("file") || view.selection);
  const wantFull = _selParams.get("full") === "1";
  const wantAncestors = want ? want.split("/").slice(0, -1) : [];
  try {
    const body = await getJSON("/api/resources");
    cache.set("", body.entries);
    const wantDir = wantAncestors.length ? await openPathChain(wantAncestors.join("/"), searchSeq) : "";
    if (!searchMode) renderTree();
    renderCrumbs();
    const parentEntries = want ? cache.get(wantDir) || [] : [];
    const file = parentEntries.find((e) => e.path === want && e.type === "file");
    const dir = parentEntries.find((e) => e.path === want && e.type === "dir");
    if (file) {
      // The link opens the preview on page load, without a history entry.
      selectFile(file, false);
      if (wantFull) enterFullScreen();
    } else if (dir) {
      selectFolder(dir, false);
      if (wantFull) enterFullScreen();
    } else if (want) {
      // The linked entry is not there any more.
      preview.innerHTML = goneNote(false);
    }
  } catch (err) {
    tree.innerHTML = '<div class="t-nomatch">Cannot load resource: ' + esc(err.message || "") + "</div>";
  }

  return {
    destroy() {
      clearTimeout(debounce);
      searchSeq += 1;
      watch.stop();
      document.removeEventListener("keydown", onKey);
      for (const fn of cleanups.splice(0)) fn();
      if (fullScreen) {
        // Leaving the screen while full screen turns it off. The app has
        // already moved to the next URL, so nothing here touches it.
        clearFullScreenChrome();
      }
      document.body.classList.remove("r-preview");
    },
    onPop() {
      if (!previewEntry) return false;
      previewEntry = false;
      replaceEntry({}, "/resources");
      view = entryView();
      const savedSel = _entryByPath(view.selection || "");
      if (savedSel) {
        if (savedSel.type === "dir") selectFolder(savedSel, false);
        else selectFile(savedSel, false);
      } else {
        selected = null;
        for (const row of tree.querySelectorAll(".t-row.selected")) row.classList.remove("selected");
        renderCrumbs();
        renderPreviewEmpty();
      }
      return true;
    },
  };
}
