import { icon } from "../ui.js";

// The foldable bubble. The label row is the fold control, the clip is a
// measured inline height on the .foldable wrapper, and every visible state
// is a class on the card.
const _PREVIEW_LINES = 6;
const _FOLD_BAND = 16;
const _FOLD_THRESHOLD = 8;
const _FOLD_MS = 150;
const _RESIZE_MS = 150;
const _REDUCED = "(prefers-reduced-motion: reduce)";
const _FRAGMENT_ROOTS = [".plain-pre", ".md-doc", ".code-block", ".raw-pre", ".csv-table"];

let _foldSeq = 0;

export function attachFoldable(card, body, labelText) {
  const label = document.createElement("button");
  label.type = "button";
  label.className = "conv-label plain";
  label.disabled = true;
  const text = document.createElement("span");
  text.className = "conv-lbl";
  text.textContent = labelText;
  label.appendChild(text);

  _foldSeq += 1;
  const wrap = document.createElement("div");
  wrap.className = "foldable";
  wrap.id = "conv-fold-" + _foldSeq;
  card.insertBefore(label, body);
  card.insertBefore(wrap, body);
  wrap.appendChild(body);

  let state = "UNSETTLED";
  let view = null;
  let prefersExpanded = false;
  let lines = 0;
  let lineH = 0;
  let chevron = null;
  let countEl = null;
  let resizeTimer = null;
  let resizeOn = false;
  let swapTimer = null;
  let endTimer = null;
  let finish = null;
  let destroyed = false;

  // ---- measurement ----

  function _contentNode() {
    for (const sel of _FRAGMENT_ROOTS) {
      const n = body.querySelector(sel);
      if (n) return n;
    }
    return body;
  }

  function _firstText(node) {
    for (const n of _textNodes(node)) {
      if (n.nodeValue && n.nodeValue.trim()) return n;
    }
    return null;
  }

  function _textNodes(node) {
    const out = [];
    const walk = (n) => {
      if (n.nodeType === 3) {
        out.push(n);
        return;
      }
      for (const c of n.childNodes || []) walk(c);
    };
    walk(node);
    return out;
  }

  // The bounding box of the first text line is a one-character range at
  // the head of the first non-empty text node.
  function _lineHeight(node) {
    const first = _firstText(node);
    if (first) {
      const range = document.createRange();
      range.setStart(first, 0);
      range.setEnd(first, 1);
      const rect = range.getBoundingClientRect();
      if (rect && rect.height > 0) return rect.height;
    }
    // An images-only fragment has no text to probe. The node's computed
    // line height stands in, and the division below is a height ratio.
    const style = getComputedStyle(node);
    return parseFloat(style.lineHeight) || parseFloat(style.fontSize);
  }

  function _padding(node) {
    const style = getComputedStyle(node);
    return (parseFloat(style.paddingTop) || 0) + (parseFloat(style.paddingBottom) || 0);
  }

  function _measure() {
    const node = _contentNode();
    const lh = _lineHeight(node);
    const n = Math.max(0, Math.round((node.scrollHeight - _padding(node)) / lh));
    return { lines: n, lineH: lh };
  }

  function _clip(lh) {
    return _PREVIEW_LINES * lh + _FOLD_BAND;
  }

  // ---- presentation ----

  function _showControl() {
    label.classList.remove("plain");
    label.disabled = false;
    if (!chevron) {
      chevron = document.createElement("span");
      chevron.className = "chev";
      chevron.setAttribute("aria-hidden", "true");
      chevron.innerHTML = icon("caret");
      label.insertBefore(chevron, text);
    }
    label.setAttribute("aria-controls", wrap.id);
  }

  function _hideControl() {
    label.classList.add("plain");
    label.disabled = true;
    label.classList.remove("open");
    if (chevron) chevron.remove();
    chevron = null;
    if (countEl) countEl.remove();
    countEl = null;
    label.removeAttribute("aria-expanded");
    label.removeAttribute("aria-controls");
  }

  function _showCount() {
    if (countEl) countEl.remove();
    countEl = document.createElement("span");
    countEl.className = "hidden-count";
    const n = lines - _PREVIEW_LINES;
    countEl.textContent = n + (n === 1 ? " line hidden" : " lines hidden");
    label.appendChild(countEl);
  }

  function _restingFolded() {
    prefersExpanded = false;
    view = "folded";
    card.classList.add("folded");
    label.classList.remove("open");
    _showControl();
    label.setAttribute("aria-expanded", "false");
    _showCount();
  }

  // Re-measure and apply the fresh count to the card's visible state. The
  // clip resets to six rendered lines of the current layout, the count
  // updates, and a threshold crossing either way gains or loses the control
  // together with its ARIA. An expanded card is left visually alone.
  function _apply() {
    const m = _measure();
    lines = m.lines;
    lineH = m.lineH;
    if (m.lines <= _FOLD_THRESHOLD) {
      view = "plain";
      _hideControl();
      card.classList.remove("folded");
      wrap.style.height = "";
      wrap.style.overflow = "";
      return;
    }
    if (prefersExpanded) {
      view = "expanded";
      card.classList.remove("folded");
      wrap.style.height = "";
      wrap.style.overflow = "";
      _showControl();
      label.classList.add("open");
      label.setAttribute("aria-expanded", "true");
      if (countEl) {
        countEl.remove();
        countEl = null;
      }
      return;
    }
    _restingFolded();
    wrap.style.height = _clip(m.lineH) + "px";
  }

  // ---- lifecycle ----

  function settle(opts) {
    if (destroyed || state === "SETTLED") return;
    state = "SETTLED";
    _registerResize();
    const m = _measure();
    lines = m.lines;
    lineH = m.lineH;
    if (m.lines <= _FOLD_THRESHOLD) {
      view = "plain";
      return;
    }
    const animate = !!(opts && opts.animate) && !_reduced();
    if (animate) _animate("folded");
    else {
      _restingFolded();
      wrap.style.height = _clip(m.lineH) + "px";
    }
  }

  // The fold state applies at once, and again after the debounce, once the
  // new fragment's layout has settled.
  function onSwapped() {
    if (destroyed || state !== "SETTLED") return;
    _apply();
    if (swapTimer) clearTimeout(swapTimer);
    swapTimer = setTimeout(() => {
      swapTimer = null;
      if (destroyed || state !== "SETTLED") return;
      _apply();
    }, _RESIZE_MS);
  }

  function toggle() {
    if (destroyed || state !== "SETTLED" || lines <= _FOLD_THRESHOLD) return;
    if (view === "folded") _animate("expanded");
    else if (view === "expanded") _animate("folded");
  }

  function _onResize() {
    if (resizeTimer) clearTimeout(resizeTimer);
    resizeTimer = setTimeout(_apply, _RESIZE_MS);
  }

  function _registerResize() {
    if (resizeOn) return;
    resizeOn = true;
    window.addEventListener("resize", _onResize);
  }

  function _reduced() {
    return matchMedia(_REDUCED).matches;
  }

  // The resting state classes, control, ARIA, and count apply at the start
  // of the transition. The height chases between explicit pixel endpoints.
  function _animate(target) {
    const m = _measure();
    lines = m.lines;
    lineH = m.lineH;
    if (target === "folded") {
      _restingFolded();
      const full = wrap.scrollHeight;
      wrap.style.height = full + "px";
      _afterFrame(() => {
        wrap.style.height = _clip(m.lineH) + "px";
      });
    } else {
      _restingExpanded();
      const start = wrap.style.height || wrap.scrollHeight + "px";
      const full = wrap.scrollHeight + "px";
      wrap.style.overflow = "hidden";
      wrap.style.height = start;
      _afterFrame(() => {
        wrap.style.height = full;
      });
    }
    _endSignal(() => {
      if (target === "expanded") {
        wrap.style.height = "";
        wrap.style.overflow = "";
      }
    });
  }

  function _restingExpanded() {
    prefersExpanded = true;
    view = "expanded";
    card.classList.remove("folded");
    label.classList.add("open");
    _showControl();
    label.setAttribute("aria-expanded", "true");
    if (countEl) {
      countEl.remove();
      countEl = null;
    }
  }

  function _afterFrame(fn) {
    requestAnimationFrame(fn);
  }

  // The transition is complete on transitionend or the _FOLD_MS timeout,
  // whichever comes first, and the end state is applied exactly once.
  function _endSignal(apply) {
    if (finish) finish();
    let done = false;
    finish = () => {
      if (done) return;
      done = true;
      wrap.removeEventListener("transitionend", finish);
      if (endTimer) clearTimeout(endTimer);
      endTimer = null;
      finish = null;
      apply();
    };
    wrap.addEventListener("transitionend", finish);
    endTimer = setTimeout(finish, _FOLD_MS);
  }

  function destroy() {
    if (destroyed) return;
    destroyed = true;
    if (finish) finish();
    if (endTimer) clearTimeout(endTimer);
    endTimer = null;
    if (resizeTimer) clearTimeout(resizeTimer);
    if (swapTimer) clearTimeout(swapTimer);
    swapTimer = null;
    if (resizeOn) window.removeEventListener("resize", _onResize);
    label.removeEventListener("click", toggle);
  }

  label.addEventListener("click", toggle);

  return { settle, toggle, onSwapped, destroy };
}
