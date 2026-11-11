// The pdf canvas viewer. The console frame wells call mountPdfViewer
// directly, and the /view pages load this module, which mounts every
// data-pdf-src container on its own.

// pdfjs/ holds the pdfjs-dist legacy build.
import * as pdfjsLib from "./pdfjs/pdf.min.mjs";

const _WORKER_SRC = "/assets/js/pdfjs/pdf.worker.min.mjs";
const _ZOOM_MIN = 0.5;
const _ZOOM_MAX = 4;
const _ZOOM_STEP = 0.25;
const _PAGES_PAD_X = 12;    // must match the horizontal padding of .pv-pages
const _ROOT_MARGIN = "600px 0px"; // pages render ahead of the viewport
const _TOP_BAND = 40;        // a page counts as current within this band
const _MAX_CANVAS_PIXELS = 16_000_000;
const _MAX_CANVAS_SIDE = 16_384;

// Resolves a destination to its explicit form, or null when it is not a valid dest array.
async function _resolveDest(doc, dest) {
  try {
    const explicit = typeof dest === "string" ? await doc.getDestination(dest) : await dest;
    return Array.isArray(explicit) ? explicit : null;
  } catch {
    return null;
  }
}

// The 1-based page number a destination array points at, or null.
async function _destPageNum(doc, explicit) {
  const [pageRef] = explicit;
  if (pageRef && typeof pageRef === "object") {
    try {
      return doc.cachedPageNumber(pageRef) || (await doc.getPageIndex(pageRef)) + 1;
    } catch {
      return null;
    }
  }
  if (Number.isInteger(pageRef)) return pageRef + 1;
  return null;
}

// The XY offset of an XYZ destination in css px from the top of its page, or 0.
function _destOffset(item, explicit, scale) {
  const [, dest, left, top] = explicit;
  if (dest?.name !== "XYZ" || typeof left !== "number" || typeof top !== "number") return 0;
  const vp = item.page.getViewport({ scale: 1 });
  return vp.convertToViewportPoint(left, top)[1] * scale;
}

// The page a named action moves to, or null when the action has no page effect.
function _namedPage(action, current, count) {
  if (action === "NextPage") return Math.min(current + 1, count);
  if (action === "PrevPage") return Math.max(current - 1, 1);
  if (action === "FirstPage") return 1;
  if (action === "LastPage") return count;
  return null;
}

// The resize jobs that run when the window is resized.
const _resizeJobs = new Set();
let _resizeWired = false;

// Builds the document and mounts the toolbar, the scroll area, and the
// page placeholders. A fetch or parse failure settles on the error state.
export async function mountPdfViewer(container, davUrl) {
  if (container._pv) return; // one viewer per container
  container._pv = true;
  pdfjsLib.GlobalWorkerOptions.workerSrc = _WORKER_SRC;
  container.replaceChildren();
  const bar = _toolbar();
  container.appendChild(bar);
  const pages = document.createElement("div");
  pages.className = "pv-pages";
  container.appendChild(pages);
  let doc;
  try {
    const res = await fetch(davUrl);
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = new Uint8Array(await res.arrayBuffer());
    doc = await pdfjsLib.getDocument({ data }).promise;
  } catch {
    bar.remove();
    _showError(pages, davUrl);
    return;
  }
  const teardown = _startViewer(doc, pages, bar);
  _watchRemoval(container, teardown);
}

function _toolbar() {
  const bar = document.createElement("div");
  bar.className = "pv-toolbar";
  bar.innerHTML =
    '<span class="pv-toolbar-page">Page 1 / 1</span>' +
    '<span class="pv-toolbar-actions">' +
    '<button type="button" class="pv-zoom-btn" data-pv="fit" title="Fit to width">Fit</button>' +
    '<button type="button" class="pv-zoom-btn" data-pv="out" title="Zoom out">−</button>' +
    '<span class="pv-zoom-pct">100%</span>' +
    '<button type="button" class="pv-zoom-btn" data-pv="in" title="Zoom in">+</button>' +
    "</span>";
  return bar;
}

function _errorBox(text) {
  const box = document.createElement("div");
  box.className = "pv-error";
  const msg = document.createElement("p");
  msg.textContent = text;
  box.appendChild(msg);
  return box;
}

function _showError(pages, davUrl) {
  const box = _errorBox("Couldn't load this PDF");
  const link = document.createElement("a");
  link.href = davUrl;
  link.textContent = "Open the raw file";
  box.appendChild(link);
  pages.appendChild(box);
}

function _showPageError(item) {
  item.el.replaceChildren(_errorBox("Couldn't render this page"));
  item.canvas = null;
}

function _startViewer(doc, pages, bar) {
  const pageEl = bar.querySelector(".pv-toolbar-page");
  const pctEl = bar.querySelector(".pv-zoom-pct");
  const items = [];
  const inFlight = new Set();
  const rendering = new Set();
  let zoom = 1;      // the zoom over the fit-to-width scale
  let fit = 1;       // css px per pdf unit at zoom 1
  let pageWidth = null;
  let current = 1;

  pageEl.textContent = "Page 1 / " + doc.numPages;
  for (let i = 1; i <= doc.numPages; i++) {
    const wrap = document.createElement("div");
    wrap.className = "pv-page";
    wrap._pvIndex = i - 1;
    pages.appendChild(wrap);
    items.push({ el: wrap, page: null, canvas: null, ann: null, rendered: false });
  }

  // Sizes an unrendered page so the scroll area carries its real geometry.
  async function _ensureHeight(item) {
    if (item.el.style.height) return;
    if (item.page == null) item.page = await doc.getPage(item.el._pvIndex + 1);
    const vp = item.page.getViewport({ scale: fit * zoom });
    item.el.style.height = Math.round(vp.height) + "px";
    updateCurrent();
  }

  // Sizes every page up to num, then scrolls that page to the top.
  async function _goto(num, offset = 0) {
    if (num < 1 || num > doc.numPages) return;
    await Promise.all(items.slice(0, num).map(_ensureHeight));
    pages.scrollTop = items[num - 1].el.offsetTop + offset;
  }

  const linkService = {
    getAnchorUrl: (anchor) => anchor,
    getDestinationHash: (dest) => {
      const str = typeof dest === "string" ? dest : Array.isArray(dest) ? JSON.stringify(dest) : "";
      return str ? "#" + escape(str) : "";
    },
    addLinkAttributes: (link, url, newWindow = false) => {
      link.href = url;
      if (newWindow) link.target = "_blank";
      link.rel = "noopener noreferrer nofollow";
    },
    goToDestination: async (dest) => {
      const explicit = await _resolveDest(doc, dest);
      if (explicit == null) return;
      const num = await _destPageNum(doc, explicit);
      if (num == null) return;
      await _ensureHeight(items[num - 1]);
      await _goto(num, _destOffset(items[num - 1], explicit, fit * zoom));
    },
    executeNamedAction: async (action) => {
      const num = _namedPage(action, current, doc.numPages);
      if (num != null) await _goto(num);
    },
    getAttachmentContent: async (id) => {
      try {
        return await doc.getAttachmentContent(id);
      } catch {
        return null;
      }
    },
    executeSetOCGState: () => {},
  };

  // The fit-to-width scale of the first page, against the visible area.
  async function refit() {
    if (pageWidth == null) {
      const first = await doc.getPage(1);
      pageWidth = first.getViewport({ scale: 1 }).width;
    }
    fit = Math.max(pages.clientWidth - _PAGES_PAD_X * 2, 1) / pageWidth;
    applyScale();
  }

  function applyScale() {
    for (const task of inFlight) task.cancel();
    inFlight.clear();
    for (const item of items) item.rendered = false;
    pctEl.textContent = Math.round(zoom * 100) + "%";
    updateCurrent();
    // Re-observing re-fires the initial state for each page.
    obs.disconnect();
    for (const item of items) obs.observe(item.el);
  }

  async function renderPage(item) {
    if (item.rendered || rendering.has(item)) return;
    rendering.add(item);
    try {
      if (item.page == null) item.page = await doc.getPage(item.el._pvIndex + 1);
      if (!pages.isConnected) return;
      const cssVp = item.page.getViewport({ scale: fit * zoom });
      const maxScale = Math.min(
        Math.sqrt(_MAX_CANVAS_PIXELS / (cssVp.width * cssVp.height)),
        _MAX_CANVAS_SIDE / cssVp.width,
        _MAX_CANVAS_SIDE / cssVp.height,
      );
      const outputScale = Math.min(window.devicePixelRatio || 1, maxScale);
      const vp = item.page.getViewport({ scale: fit * zoom * outputScale });
      if (item.canvas == null) {
        item.canvas = document.createElement("canvas");
        item.el.replaceChildren(item.canvas);
      }
      item.el.style.height = Math.round(cssVp.height) + "px";
      updateCurrent();
      item.canvas.width = Math.round(vp.width);
      item.canvas.height = Math.round(vp.height);
      item.canvas.style.width = Math.round(cssVp.width) + "px";
      item.canvas.style.height = Math.round(cssVp.height) + "px";
      const task = item.page.render({ canvasContext: item.canvas.getContext("2d"), viewport: vp });
      inFlight.add(task);
      let failed = false;
      let cancelled = false;
      try {
        await task.promise;
      } catch (err) {
        if (err instanceof pdfjsLib.RenderingCancelledException) cancelled = true;
        else failed = true;
      } finally {
        inFlight.delete(task);
      }
      if (failed) {
        _showPageError(item);
        return;
      }
      if (cancelled) return;
      item.rendered = true;
      await _renderAnnotations(item);
    } finally {
      rendering.delete(item);
    }
  }

  // Builds a fresh annotation layer over the canvas so links and form
  // fields land where the pdf places them.
  async function _renderAnnotations(item) {
    item.ann?.remove();
    const ann = document.createElement("div");
    ann.className = "pv-ann";
    ann.style.setProperty("--total-scale-factor", fit * zoom);
    ann.style.setProperty("--scale-round-x", "1px");
    ann.style.setProperty("--scale-round-y", "1px");
    item.ann = ann;
    item.el.append(ann);
    try {
      const layer = new pdfjsLib.AnnotationLayer({
        div: ann,
        page: item.page,
        viewport: item.page.getViewport({ scale: fit * zoom }),
        linkService,
      });
      const anns = await item.page.getAnnotations();
      await layer.render({ annotations: anns });
    } catch {
      ann.remove();
      item.ann = null;
    }
  }

  // A page draws once it enters or approaches the viewport.
  const obs = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) renderPage(items[e.target._pvIndex]);
  }, { root: pages, rootMargin: _ROOT_MARGIN });
  for (const item of items) obs.observe(item.el);

  function updateCurrent() {
    const y = pages.scrollTop + _TOP_BAND;
    let cur = 1;
    for (const item of items) {
      // A page without a height has no known position yet.
      if (!item.el.style.height) break;
      if (item.el.offsetTop <= y) cur = item.el._pvIndex + 1;
    }
    if (cur !== current) {
      current = cur;
      pageEl.textContent = "Page " + cur + " / " + doc.numPages;
    }
  }
  pages.addEventListener("scroll", updateCurrent, { passive: true });

  const onBarClick = (ev) => {
    const btn = ev.target.closest("button");
    if (!btn) return;
    if (btn.dataset.pv === "fit") zoom = 1;
    else if (btn.dataset.pv === "in") zoom = Math.min(_ZOOM_MAX, zoom + _ZOOM_STEP);
    else if (btn.dataset.pv === "out") zoom = Math.max(_ZOOM_MIN, zoom - _ZOOM_STEP);
    else return;
    applyScale();
  };
  bar.addEventListener("click", onBarClick);

  if (!_resizeWired) {
    _resizeWired = true;
    window.addEventListener("resize", () => {
      for (const fn of [..._resizeJobs]) fn();
    });
  }
  const onResize = () => {
    if (!pages.isConnected) {
      _resizeJobs.delete(onResize);
      return;
    }
    refit();
  };
  _resizeJobs.add(onResize);

  refit();

  const teardown = () => {
    obs.disconnect();
    for (const task of inFlight) task.cancel();
    inFlight.clear();
    pages.removeEventListener("scroll", updateCurrent);
    bar.removeEventListener("click", onBarClick);
    _resizeJobs.delete(onResize);
    doc.destroy();
  };
  return teardown;
}

// Ends the viewer once its container leaves the document.
function _watchRemoval(container, teardown) {
  const mo = new MutationObserver(() => {
    if (!container.isConnected) {
      mo.disconnect();
      container._pv = false;
      teardown();
    }
  });
  mo.observe(document.body, { childList: true, subtree: true });
}

// Mounts the containers the server rendered. The console SPA carries none.
function _autoInit() {
  document.querySelectorAll("[data-pdf-src]").forEach((el) => {
    mountPdfViewer(el, el.dataset.pdfSrc);
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", _autoInit);
} else {
  _autoInit();
}
