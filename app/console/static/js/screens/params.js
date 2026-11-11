import { icon, fileGlyph } from "../ui.js";
import { getJSON } from "../api.js";
import { parameterKind } from "../parameter-kind.js";
import { mountParameterText } from "../parameter-text.js";
import { mountParameterList } from "../parameter-list.js";
import { renderResourceCard } from "../resource-card.js";

const _CHIPS_INLINE_MAX = 6;
// Params each tool declares, kept by param name. Each tool is fetched once per page load.
const _schemaCache = new Map();

export function loadToolSchema(tool) {
  if (_schemaCache.has(tool)) return Promise.resolve(_schemaCache.get(tool));
  return getJSON("/api/tools/" + encodeURIComponent(tool) + "/schema").then((body) => {
    const schema = new Map();
    for (const p of body.params) schema.set(p.name, p);
    _schemaCache.set(tool, schema);
    return schema;
  }).catch(() => null);
}

function _deepEq(a, b) {
  if (a === b) return true;
  if (typeof a !== "object" || typeof b !== "object" || a === null || b === null) return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  const ka = Object.keys(a);
  const kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  return ka.every((k) => Object.prototype.hasOwnProperty.call(b, k) && _deepEq(a[k], b[k]));
}

function _atDefault(key, value, schema) {
  const p = schema && schema.get(key);
  if (!p || p.required) return false;
  return _deepEq(value, p.default);
}

function _keyEl(key, dim) {
  const keyEl = document.createElement("span");
  keyEl.className = "p-key";
  keyEl.dataset.parameterLabel = key;
  keyEl.textContent = key;
  if (dim) {
    const badge = document.createElement("span");
    badge.className = "p-chip";
    badge.textContent = "default";
    keyEl.appendChild(badge);
  }
  return keyEl;
}

// The collapsed file-list pill. Its preview is a view of the first card's
// fact. The real PNG shows when the thumbnail is done, and the first file's
// tinted glyph shows otherwise. The first card's 2-second probe is what
// flips it. The pill owns no timer of its own.
function _fileSummary(n, key, firstName) {
  const sum = document.createElement("span");
  sum.className = "pv-sum";
  sum.setAttribute("role", "button");
  sum.setAttribute("tabindex", "0");
  sum.setAttribute("aria-expanded", "false");
  sum.setAttribute("aria-label", "Expand value of " + key);
  const preview = document.createElement("span");
  preview.className = "sum-thumb";
  sum.appendChild(preview);
  const num = document.createElement("b");
  num.className = "n";
  num.textContent = String(n);
  sum.appendChild(num);
  const unit = document.createElement("span");
  unit.className = "sum-u";
  unit.textContent = "files";
  sum.appendChild(unit);
  const chev = document.createElement("span");
  chev.className = "p-chev";
  chev.innerHTML = icon("chevDown");
  sum.appendChild(chev);
  function paint(fact) {
    if (fact && fact.thumb && fact.thumb.status === "done" && fact.thumb.url) {
      const img = document.createElement("img");
      img.src = fact.thumb.url;
      img.alt = "";
      img.addEventListener("error", () => paint(null), { once: true });
      preview.replaceChildren(img);
    } else {
      preview.innerHTML = fileGlyph(firstName);
    }
  }
  paint(null);
  return { el: sum, preview: paint };
}

// Resource cards open the file viewer.
function _fileRow(key, value, dim, stops) {
  const items = value === null || value === "" ? [] : Array.isArray(value) ? value : [value];
  const n = items.length;
  const row = document.createElement("div");
  row.className = (value === null ? "p-row p-file p-file-empty" : "p-row p-list p-file") + (dim ? " p-dim" : "");
  row.setAttribute("role", "group");
  row.setAttribute("aria-label", key + " parameter");
  row.appendChild(_keyEl(key, dim));
  const val = document.createElement("span");
  val.className = "p-val";
  val.dataset.parameterValue = key;
  const cards = document.createElement("div");
  cards.className = "res-cards";
  let sum = null;
  let paintPreview = null;
  if (n === 0) {
    const empty = document.createElement("span");
    empty.className = Array.isArray(value) ? "pv-zero" : "p-text";
    empty.textContent = Array.isArray(value) ? "0 files" : "unset";
    val.appendChild(empty);
  } else {
    const collapsed = n > _CHIPS_INLINE_MAX;
    if (collapsed && typeof items[0] === "string" && items[0] !== "") {
      const built = _fileSummary(n, key, items[0].split("/").pop());
      sum = built.el;
      paintPreview = built.preview;
      val.appendChild(sum);
    } else if (collapsed) {
      // No card for the first item. The generic-glyph pill is the fallback.
      sum = _fileSummary(n, key, "").el;
      val.appendChild(sum);
    }
    for (let i = 0; i < n; i++) {
      const item = items[i];
      if (typeof item !== "string" || item === "") {
        stops.push(mountParameterText(cards, JSON.stringify(item)).stop);
        continue;
      }
      const opts = {};
      if (i === 0 && paintPreview) opts.onFact = paintPreview;
      stops.push(renderResourceCard(cards, item, opts).stop);
    }
    if (collapsed) cards.style.display = "none";
    val.appendChild(cards);
  }
  row.appendChild(val);
  if (sum) {
    const toggle = () => {
      const open = row.classList.toggle("open");
      cards.style.display = open ? "" : "none";
      sum.setAttribute("aria-expanded", open ? "true" : "false");
      sum.setAttribute("aria-label", (open ? "Collapse value of " : "Expand value of ") + key);
    };
    sum.addEventListener("click", toggle);
    sum.addEventListener("keydown", (ev) => {
      if (ev.key !== "Enter" && ev.key !== " ") return;
      ev.preventDefault();
      toggle();
    });
  }
  return row;
}

function _rowFor(key, value, dim, schema, stops) {
  const param = schema && schema.get(key);
  const kind = parameterKind(param, value);
  if (kind === "resource-list" && Array.isArray(value)) return _fileRow(key, value, dim, stops);
  if (kind === "resource" && (value === null || typeof value === "string")) return _fileRow(key, value, dim, stops);
  const row = document.createElement("div");
  row.className = "p-row parameter-row" + (dim ? " p-dim" : "");
  row.setAttribute("role", "group");
  row.setAttribute("aria-label", key + " parameter");
  row.appendChild(_keyEl(key, dim));
  const body = document.createElement("div");
  body.className = "p-val";
  row.appendChild(body);
  if (kind === "string-list" && Array.isArray(value) && value.every((item) => typeof item === "string")) {
    stops.push(mountParameterList(body, value, { label: key }).stop);
  } else {
    const text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
    const href = kind === "text" && param?.kind === "url" && typeof value === "string" && /^https?:\/\//.test(value) ? value : undefined;
    stops.push(mountParameterText(body, text, { href }).stop);
  }
  return row;
}

function _footer(count, dimRows) {
  const foot = document.createElement("div");
  foot.className = "p-footer";
  foot.setAttribute("role", "button");
  foot.setAttribute("tabindex", "0");
  const label = document.createElement("span");
  label.className = "p-foot-text";
  const chev = document.createElement("span");
  chev.className = "p-chev";
  chev.innerHTML = icon("chevDown");
  foot.appendChild(label);
  foot.appendChild(chev);
  const setLabel = (isShown) => {
    label.textContent = count + (count === 1 ? " param" : " params") + " at default · " + (isShown ? "hide" : "show");
    foot.setAttribute("aria-expanded", isShown ? "true" : "false");
  };
  setLabel(false);
  const toggle = () => {
    const shown = foot.classList.toggle("open");
    setLabel(shown);
    for (const r of dimRows) r.classList.toggle("p-hidden", !shown);
  };
  foot.addEventListener("click", toggle);
  foot.addEventListener("keydown", (ev) => {
    if (ev.target !== foot || (ev.key !== "Enter" && ev.key !== " ")) return;
    ev.preventDefault();
    toggle();
  });
  return foot;
}

export function mountParams(well, countEl, params, schema) {
  const stops = [];
  const entries = [];
  if (params && typeof params === "object") {
    for (const key of Object.keys(params)) {
      const v = params[key];
      if (v === undefined) continue;
      entries.push({ key, value: v, dim: _atDefault(key, v, schema) });
    }
  }
  if (entries.length === 0) {
    countEl.textContent = "";
    well.innerHTML = '<div class="p-empty">' + icon("corner") + " No parameters — this run stored no params.</div>";
  } else {
    const visible = entries.filter((e) => !e.dim);
    const dimRows = entries.filter((e) => e.dim).map((e) => _rowFor(e.key, e.value, true, schema, stops));
    countEl.textContent = visible.length ? String(visible.length) : "";
    const rows = document.createElement("div");
    rows.className = "p-rows";
    for (const e of visible) rows.appendChild(_rowFor(e.key, e.value, false, schema, stops));
    for (const r of dimRows) {
      r.classList.add("p-hidden");
      rows.appendChild(r);
    }
    if (dimRows.length > 0) rows.appendChild(_footer(dimRows.length, dimRows));
    well.replaceChildren(rows);
  }
  return () => {
    for (const stop of stops) stop();
  };
}
