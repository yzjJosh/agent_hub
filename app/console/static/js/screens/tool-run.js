import { getJSON, postJSONBody, ApiError, davHref } from "../api.js";
import { esc, fmtBytes, fmtBytesPair } from "../format.js";
import { icon, paramSummary, emptyState, skeletonRows, SKELETON_MS } from "../ui.js";
import { toast } from "../toast.js";
import { navigate } from "../router.js";
import { syncGlyphs } from "../theme.js";
import { compactSlot, sortEntries } from "../gallery.js";
import { watchThumbs } from "../tree-thumbs.js";
import { renderResourceCard } from "../resource-card.js";

import { parameterKind } from "../parameter-kind.js";
import { mountParameterList } from "../parameter-list.js";
import { mountParameterText } from "../parameter-text.js";

const _MIN_SUBMIT_MS = 200;
const _UPLOAD_MAX = 100 * 1024 * 1024;
const _TA_MAX_PX = 252; // the height of a box 12 rows tall
const _SEARCH_DEBOUNCE_MS = 150;
const _TREE_SKELETON_MS = 300;
const _POP_MAX = 560;
const _POP_MIN = 220;

function _isPhone() {
  return matchMedia("(max-width: 767px)").matches;
}

// The grey hint for a field with no default. It is the first sentence of the param
// description, cut to 40 letters.
function _placeholder(desc) {
  const text = (desc || "").replace(/\s+/g, " ").trim();
  const masked = text.replace(/\b[eE]\.g\.|\b[iI]\.e\./g, (m) => m.replace(/\./g, "·"));
  const end = masked.search(/[.!?](?:\s|$)/);
  const sentence = end === -1 ? text : text.slice(0, end + 1);
  return sentence.length > 40 ? sentence.slice(0, 40).trimEnd() + "…" : sentence;
}

export async function renderToolRun(el, params) {
  const tool = params.tool;
  const from = new URLSearchParams(location.search).get("from");

  el.innerHTML =
    '<header class="pushbar">' +
      '<a class="icon-btn" data-spa href="/tools/' + encodeURIComponent(tool) + '" title="Back to task list">' + icon("chevL") + "</a>" +
      '<span class="bar-title">' + esc(tool) + "</span>" +
      '<span class="spacer"></span>' +
      '<button class="icon-btn" id="rr-m-cancel" title="Cancel">' + icon("x") + "</button>" +
      '<button class="icon-btn" data-theme-toggle title="Switch theme"></button>' +
    "</header>" +
    '<div class="tt-head">' +
      '<a class="icon-btn" data-spa href="/tools/' + encodeURIComponent(tool) + '" title="Back to task list">' + icon("chevL") + "</a>" +
      '<span class="tile">' + icon("tool") + "</span>" +
      '<span class="tt-name" id="rr-name">' + esc(tool) + '<span class="tt-sub" id="rr-sub"></span></span>' +
      '<span class="spacer"></span>' +
      '<span class="rr-pill">manual run</span>' +
    "</div>" +
    '<div class="rr-scroll"><div class="rr-col">' +
      '<div id="rr-about"></div>' +
      '<div class="rr-form" id="rr-form"></div>' +
      '<div class="rr-banner hidden" id="rr-banner" role="alert">' + icon("alert") + '<span class="msg" id="rr-banner-msg"></span></div>' +
      '<div class="rr-well hidden" id="rr-well"></div>' +
      '<div class="rr-bar">' +
        '<span class="rr-preview" id="rr-preview"></span>' +
        '<span class="spacer"></span>' +
        '<button class="btn ghost" id="rr-cancel">Cancel</button>' +
        '<button class="btn solid" id="rr-run" disabled>' + icon("play") + "<span class=\"rr-run-t\">Run tool</span></button>" +
      "</div>" +
    "</div></div>";
  syncGlyphs();
  document.body.classList.add("pushed");

  const form = el.querySelector("#rr-form");
  const preview = el.querySelector("#rr-preview");
  const banner = el.querySelector("#rr-banner");
  const bannerMsg = el.querySelector("#rr-banner-msg");
  const well = el.querySelector("#rr-well");
  const runBtn = el.querySelector("#rr-run");
  const cancelBtn = el.querySelector("#rr-cancel");
  const runLabel = () => (runBtn.innerHTML = icon("play") + '<span class="rr-run-t">Run tool</span>');

  let schema = null;
  const fields = []; // one entry per field, a date pair counts as one
  let optionalOpen = !_isPhone();
  let validatedOnce = false;
  let submitting = false;
  let formReady = false;

  function notExposed() {
    el.replaceChildren();
    emptyState(el, "alert", "Tool not exposed in the console",
      "This tool cannot be started from the console.", "Back to Tools", "/tools");
  }

  function notFound() {
    el.replaceChildren();
    emptyState(el, "alert", "Tool not found or expired",
      "This tool no longer exists in the registry.", "Back to Tools", "/tools");
  }

  function showBanner(msg) {
    bannerMsg.textContent = msg;
    banner.classList.remove("hidden");
    if (banner.focus) banner.focus();
  }

  function hideBanner() {
    banner.classList.add("hidden");
  }

  function sameAsDefault(p, v) {
    if (p.default === null || p.default === undefined) return false;
    return Array.isArray(v) ? JSON.stringify(v) === JSON.stringify(p.default) : v === p.default;
  }

  function addVal(body, p, v) {
    if (v === null || v === undefined || v === "") return;
    if (typeof v === "number" && Number.isNaN(v)) return;
    if (sameAsDefault(p, v)) return;
    body[p.name] = v;
  }

  function buildBody() {
    const body = {};
    for (const f of fields) {
      addVal(body, f.p, f.read());
      if (f.p2) addVal(body, f.p2, f.p2.read());
    }
    return body;
  }

  function refreshPreview() {
    const sum = paramSummary(buildBody());
    preview.innerHTML = sum ? sum : '<span class="rr-none">No parameters set yet</span>';
  }

  function fieldError(f, msg) {
    f.errEl.textContent = msg || "";
    f.errEl.classList.toggle("hidden", !msg);
    f.wrap.classList.toggle("f-bad", !!msg);
    const ctl = f.errTarget || f.control;
    if (!ctl) return;
    const base = (f.ariaDescribedBy || []).join(" ");
    if (msg) ctl.setAttribute("aria-describedby", (base ? base + " " : "") + f.errEl.id);
    else if (base) ctl.setAttribute("aria-describedby", base);
    else ctl.removeAttribute("aria-describedby");
  }

  function onRevalidate() {
    updateOptionalCount();
    if (!validatedOnce) return;
    validate();
    refreshPreview();
  }

  function validate() {
    let firstBad = null;
    const mark = (f, msg) => {
      fieldError(f, msg);
      if (msg && !firstBad) firstBad = f;
    };
    for (const f of fields) {
      fieldError(f, null);
      if (f.incompatible) mark(f, f.p.name + " has an unsupported saved value.");
      for (const pf of [f.p, f.p2]) {
        if (!pf) continue;
        const read = pf === f.p ? f.read : pf.read;
        const v = read();
        let msg = null;
        if (pf.required && (v === null || v === undefined || v === "")) {
          msg = pf.name + " is required.";
        } else if (v !== null && v !== undefined && v !== "") {
          if (typeof v === "number" && Number.isNaN(v)) {
            msg = pf.name + " must be an integer.";
          } else if (pf.type === "integer") {
            if (!Number.isInteger(v)) msg = pf.name + " must be an integer.";
            else if (pf.min !== null && v < pf.min) msg = pf.name + " must be ≥ " + pf.min + ".";
            else if (pf.max !== null && v > pf.max) msg = pf.name + " must be ≤ " + pf.max + ".";
          } else if (pf.kind === "url" && !/^https?:\/\//.test(String(v))) {
            msg = pf.name + " must start with http:// or https://";
          }
        }
        if (msg) mark(f, msg);
      }
      // A pair field has one error spot, so both of its params show the error there.
    }
    for (const rule of schema.rules || []) {
      if (!rule.startsWith("order:")) continue;
      const parts = rule.split(":");
      const fa = fields.find((f) => f.p.name === parts[1] || (f.p2 && f.p2.name === parts[1]));
      const fb = fields.find((f) => f.p.name === parts[2] || (f.p2 && f.p2.name === parts[2]));
      if (!fa || !fb) continue;
      const read = (f, n) => (f.p.name === n ? f.read() : f.p2.read());
      const va = read(fa, parts[1]);
      const vb = read(fb, parts[2]);
      if (va === null || va === undefined || va === "" || vb === null || vb === undefined || vb === "") continue;
      const isDates = fa.p.kind === "date" && fb.p.kind === "date";
      const bothNum = Number.isFinite(Number(va)) && Number.isFinite(Number(vb));
      // Numbers are compared as numbers, everything else as text.
      const inverted = isDates ? String(va) > String(vb) : bothNum ? Number(va) > Number(vb) : String(va) > String(vb);
      if (inverted) {
        const msg = isDates
          ? parts[2] + " is before " + parts[1] + "."
          : parts[1] + " is greater than " + parts[2] + ".";
        mark(isDates ? fb : fa, msg);
      }
    }
    if (firstBad && firstBad.control) firstBad.control.focus();
    return !firstBad;
  }

  // ---------- field builders ----------

  function labelRow(f) {
    const p = f.p;
    const name = f.p2 ? p.name + " / " + f.p2.name : p.name;
    return '<span class="f-name">' + esc(name) +
      (p.required ? '<span class="f-req">*</span>' : "") +
      (p.default !== null ? '<span class="f-default">default</span>' : "") + "</span>" +
      '<span class="f-spacer"></span>' +
      (f.sw ? '<span class="f-val mono" data-fval></span>' : "");
  }

  function descRow(f) {
    const d = f.p.description || "";
    return d ? '<div class="f-desc" title="' + esc(d) + '">' + esc(d) + "</div>" : "";
  }

  function makeField(f) {
    f.wrap = document.createElement("div");
    f.wrap.className = "f-field";
    f.wrap.dataset.parameterField = f.p.name;
    f.wrap.innerHTML =
      '<div class="f-label">' + labelRow(f) + "</div>" +
      descRow(f) +
      '<div class="f-err hidden" role="alert"></div>';
    f.errEl = f.wrap.querySelector(".f-err");
    f.errEl.id = "ferr-" + f.p.name + (f.p2 ? "-" + f.p2.name : "");
    f.labelEl = f.wrap.querySelector(".f-label");
    f.controlBox = f.wrap.appendChild(document.createElement("div"));
    f.controlBox.className = "f-control";
    f.wrap.insertBefore(f.controlBox, f.errEl);
    form.appendChild(f.wrap);
  }

  function wire(f, node) {
    node.addEventListener("input", () => { refreshPreview(); onRevalidate(); });
    node.addEventListener("change", () => { refreshPreview(); onRevalidate(); });
  }

  function textish(f, mono) {
    const p = f.p;
    const input = document.createElement("input");
    input.type = "text";
    input.className = "f-input" + (mono ? " mono" : "");
    if (p.default == null && p.description) input.placeholder = _placeholder(p.description);
    if (p.default !== null) input.value = String(p.default);
    input.setAttribute("aria-required", p.required ? "true" : "false");
    wire(f, input);
    f.controlBox.appendChild(input);
    f.control = input;
    f.read = () => {
      const v = input.value.trim();
      return v === "" ? null : v;
    };
    f.set = (v) => { input.value = v === null || v === undefined ? "" : String(v); };
  }

  function longtext(f) {
    const p = f.p;
    const ta = document.createElement("textarea");
    ta.className = "f-input f-ta";
    if (p.default == null && p.description) ta.placeholder = _placeholder(p.description);
    if (p.default !== null) ta.value = String(p.default);
    ta.setAttribute("aria-required", p.required ? "true" : "false");
    const grow = () => {
      ta.style.height = "auto";
      ta.style.height = Math.min(ta.scrollHeight, _TA_MAX_PX) + "px";
    };
    ta.addEventListener("input", grow);
    wire(f, ta);
    f.controlBox.appendChild(ta);
    grow();
    f.control = ta;
    f.read = () => {
      const v = ta.value.trim();
      return v === "" ? null : v;
    };
    f.set = (v) => {
      ta.value = v === null || v === undefined ? "" : String(v);
      grow();
    };
  }

  function numberField(f) {
    const p = f.p;
    const input = document.createElement("input");
    input.type = "number";
    input.className = "f-input f-num";
    if (p.step !== null) input.step = String(p.step);
    else if (p.type === "integer") input.step = "1";
    if (p.min !== null) input.min = String(p.min);
    if (p.max !== null) input.max = String(p.max);
    input.setAttribute("inputmode", p.type === "integer" ? "numeric" : "decimal");
    if (p.default == null && p.description) input.placeholder = _placeholder(p.description);
    if (p.default !== null) input.value = String(p.default);
    input.setAttribute("aria-required", p.required ? "true" : "false");
    wire(f, input);
    f.controlBox.appendChild(input);
    f.control = input;
    f.read = () => {
      if (input.value.trim() === "") return null;
      return Number(input.value);
    };
    f.set = (v) => { input.value = v === null || v === undefined ? "" : String(v); };
  }

  function boolField(f) {
    const p = f.p;
    let val = p.default !== null ? !!p.default : false;
    const sw = document.createElement("button");
    sw.type = "button";
    sw.className = "sw" + (val ? " on" : "");
    sw.setAttribute("role", "switch");
    sw.setAttribute("aria-checked", String(val));
    sw.setAttribute("aria-label", p.name);
    sw.innerHTML = '<span class="sw-knob"></span>';
    const valEl = f.wrap.querySelector("[data-fval]");
    const host = _isPhone() ? f.labelEl : f.controlBox;
    sw.addEventListener("click", () => {
      val = !val;
      sw.classList.toggle("on", val);
      sw.setAttribute("aria-checked", String(val));
      valEl.textContent = String(val);
      refreshPreview();
      onRevalidate();
    });
    host.appendChild(sw);
    f.control = sw;
    valEl.textContent = String(val);
    f.read = () => val;
    f.set = (v) => {
      val = !!v;
      sw.classList.toggle("on", val);
      sw.setAttribute("aria-checked", String(val));
      valEl.textContent = String(val);
    };
  }

  function enumField(f) {
    const p = f.p;
    const sel = document.createElement("select");
    sel.className = "f-input";
    const ph = document.createElement("option");
    ph.value = "";
    ph.textContent = "Select…";
    sel.appendChild(ph);
    for (const v of p.enum) {
      const o = document.createElement("option");
      o.value = v;
      o.textContent = v;
      sel.appendChild(o);
    }
    if (p.default !== null) sel.value = String(p.default);
    sel.setAttribute("aria-required", p.required ? "true" : "false");
    wire(f, sel);
    f.controlBox.appendChild(sel);
    f.control = sel;
    f.read = () => (sel.value === "" ? null : sel.value);
    f.set = (v) => { if (v !== null && v !== undefined) sel.value = String(v); };
  }

  function dateInput(f, p) {
    const input = document.createElement("input");
    input.type = "date";
    input.className = "f-input mono f-date";
    input.setAttribute("aria-required", p.required ? "true" : "false");
    if (p.default !== null) input.value = String(p.default);
    wire(f, input);
    return input;
  }

  function dateField(f) {
    const p = f.p;
    const input = dateInput(f, p);
    f.controlBox.appendChild(input);
    f.control = input;
    f.read = () => (input.value === "" ? null : input.value);
    f.set = (v) => { input.value = v === null || v === undefined ? "" : String(v); };
  }

  function resourceField(f) {
    const p = f.p;
    let val = p.default !== null && p.default !== "" ? String(p.default) : null;
    let valFact = null;
    let card = null;
    const cards = document.createElement("div");
    cards.className = "res-cards";
    const hint = document.createElement("span");
    hint.className = "f-chip-hint";
    hint.textContent = "No file selected.";
    cards.appendChild(hint);
    f.controlBox.appendChild(cards);
    const choose = document.createElement("button");
    choose.type = "button";
    choose.className = "btn ghost res-choose-btn";
    choose.innerHTML = icon("plus") + "Choose file…";
    choose.addEventListener("click", () => f.picker.open(choose));
    f.controlBox.appendChild(choose);
    const renderCard = () => {
      if (card) {
        card.stop();
        card = null;
      }
      cards.replaceChildren(hint);
      hint.classList.toggle("hidden", val !== null);
      if (val === null) return;
      card = renderResourceCard(cards, valFact !== null ? valFact : val, {
        onRemove: () => {
          val = null;
          valFact = null;
          renderCard();
          fieldError(f, null);
          refreshPreview();
          onRevalidate();
        },
      });
    };
    f.controlBox.classList.add("res-field");
    f.control = choose;
    f.read = () => val;
    f.set = (v) => {
      val = v === null || v === undefined || v === "" ? null : String(v);
      valFact = null; // A prefilled value is a bare path. The card fetches its fact.
      renderCard();
    };
    renderCard();
    f.stop = () => {
      if (card) {
        card.stop();
        card = null;
      }
    };
    f.picker = createPicker(f.controlBox, {
      multi: false,
      onPick: (items) => {
        const item = items[0];
        val = typeof item === "string" ? item : item.path;
        valFact = typeof item === "string" ? null : item;
        renderCard();
        fieldError(f, null);
        refreshPreview();
        onRevalidate();
      },
    });
  }

  function resourceListField(f) {
    const p = f.p;
    let vals = [];
    const facts = new Map();
    let cardStops = [];
    const cards = document.createElement("div");
    cards.className = "res-cards";
    const hint = document.createElement("span");
    hint.className = "f-chip-hint";
    hint.textContent = "No files selected.";
    cards.appendChild(hint);
    f.controlBox.appendChild(cards);
    const add = document.createElement("button");
    add.type = "button";
    add.className = "btn ghost res-choose-btn";
    add.innerHTML = icon("plus") + "Add resource…";
    add.addEventListener("click", () => f.picker.open(add));
    f.controlBox.appendChild(add);
    const renderCards = () => {
      for (const s of cardStops) s();
      cardStops = [];
      cards.replaceChildren(hint);
      hint.classList.toggle("hidden", vals.length > 0);
      for (const v of vals) {
        const card = renderResourceCard(cards, facts.has(v) ? facts.get(v) : v, {
          onRemove: () => {
            vals = vals.filter((x) => x !== v);
            facts.delete(v);
            renderCards();
            refreshPreview();
            onRevalidate();
          },
        });
        cardStops.push(card.stop);
      }
    };
    f.controlBox.classList.add("res-field");
    f.control = add;
    f.read = () => (vals.length === 0 ? null : vals.slice());
    f.set = (v) => {
      vals = Array.isArray(v) ? v.slice() : [];
      facts.clear(); // Prefilled values are bare paths. The cards fetch their facts.
      renderCards();
    };
    renderCards();
    f.stop = () => {
      for (const s of cardStops) s();
      cardStops = [];
    };
    f.picker = createPicker(f.controlBox, {
      multi: true,
      onPick: (items) => {
        for (const item of items) {
          const path = typeof item === "string" ? item : item.path;
          if (vals.includes(path)) continue;
          vals.push(path);
          if (typeof item !== "string") facts.set(path, item);
        }
        renderCards();
        refreshPreview();
        onRevalidate();
      },
    });
  }

  function listField(f) {
    const list = mountParameterList(f.controlBox, [], {
      editable: true, label: f.p.name,
      onChange: () => { fieldError(f, null); refreshPreview(); onRevalidate(); },
      onEditingChange: (active) => {
        f.editing = active;
        runBtn.disabled = !formReady || submitting || fields.some((field) => field.editing) || active;
      },
    });
    f.control = list.control;
    f.control.setAttribute("aria-required", f.p.required ? "true" : "false");
    const desc = f.wrap.querySelector(".f-desc");
    if (desc) {
      desc.id = "fdesc-" + f.p.name;
      f.ariaDescribedBy = [desc.id];
      f.control.setAttribute("aria-describedby", desc.id);
    }
    f.commit = list.commit;
    f.read = () => { const values = list.read(); return values.length ? values : null; };
    f.set = (value) => list.set(value || []);
    f.stop = list.stop;
  }

  function fallbackField(f, value) {
    if (f.stop) f.stop();
    if (f.picker) { f.picker.destroy(); f.picker = null; }
    if (f.control) f.control.remove();
    f.labelEl.querySelector("[data-fval]")?.remove();
    f.controlBox.replaceChildren();
    const text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
    f.stop = mountParameterText(f.controlBox, text).stop;
    f.read = () => value;
    f.control = null;
    f.commit = null;
    f.incompatible = true;
  }

  function buildField(p, values) {
    const f = { p, read: () => null, set: () => {} };
    const kind = parameterKind(p);
    if (p.type === "boolean" && kind === "scalar") f.sw = true;
    makeField(f);
    if (kind === "string-list") listField(f);
    else if (kind === "resource-list") resourceListField(f);
    else if (kind === "resource") resourceField(f);
    else if (kind === "scalar" && p.type === "boolean") boolField(f);
    else if ((kind === "text" || kind === "scalar") && p.enum) enumField(f);
    else if (kind === "scalar") numberField(f);
    else if (kind === "text" && p.kind === "longtext") longtext(f);
    else if (kind === "text" && p.kind === "date") dateField(f);
    else if (kind === "text") textish(f, p.kind !== "text");
    else fallbackField(f, p.default);
    if (f.control && ["text", "scalar"].includes(kind)) f.control.setAttribute("aria-label", p.name);
    const set = f.set;
    f.set = (value) => {
      const compatible = value == null ||
        (["string-list", "resource-list"].includes(kind) ? Array.isArray(value) && value.every((item) => typeof item === "string") :
          ["text", "resource"].includes(kind) ? typeof value === "string" :
            kind === "scalar" ? typeof value === (p.type === "boolean" ? "boolean" : "number") : false);
      if (!compatible) fallbackField(f, value);
      else {
        set(value);
        const isEmpty = value === "" || (["string-list", "resource-list"].includes(kind) && Array.isArray(value) && value.length === 0);
        const expected = isEmpty ? null : value;
        if (value != null && JSON.stringify(f.read()) !== JSON.stringify(expected)) fallbackField(f, value);
      }
    };
    const value = Object.hasOwn(values, p.name) ? values[p.name] : p.default;
    if (Object.hasOwn(values, p.name) || value != null) f.set(value);
    return f;
  }

  // ---------- about block ----------

  function renderAbout() {
    const host = el.querySelector("#rr-about");
    const text = (schema.description || "").trim();
    if (!text) { host.remove(); return; }
    const phone = _isPhone();
    const lines = text.split("\n");
    let bodyHtml = "";
    let block = null;
    const flushBlock = () => {
      if (!block) return;
      const base = Math.min(...block.map((l) => l.length - l.trimStart().length));
      bodyHtml += '<div class="rr-about-pre">' + block.map((l) => {
        if (!phone) return '<div class="ln">' + esc(l) + "</div>";
        const stripped = l.slice(base);
        const m = stripped.match(/^(\S+)\s{2,}(\S.*)$/);
        if (!m) return '<div class="ln">' + esc(stripped) + "</div>";
        const rest = m[2].startsWith(":") ? " " + m[2] : ": " + m[2];
        return '<div class="ln"><span class="nm">' + esc(m[1]) + "</span><span class=\"rst\">" + esc(rest) + "</span></div>";
      }).join("") + "</div>";
      block = null;
    };
    for (let i = 0; i < lines.length; i += 1) {
      const ln = lines[i];
      if (ln.trim() !== "" && /^\s/.test(ln)) {
        if (!block) block = [];
        block.push(ln);
        continue;
      }
      flushBlock();
      if (ln === "") continue;
      const next = lines[i + 1] || "";
      const isLead = /^\s/.test(next) && next.trim() !== "";
      bodyHtml += '<p class="rr-about-' + (isLead ? "lead" : "p") + '">' + esc(ln) + "</p>";
    }
    flushBlock();
    host.className = "rr-about";
    host.innerHTML =
      '<div class="rr-about-hd">' +
        '<button class="chev" type="button" title="Collapse" aria-expanded="true" aria-controls="rr-about-body">' +
          icon("chevDown") + "</button>" +
        '<span class="rr-about-lbl">About</span>' +
      "</div>" +
      '<div class="rr-about-body" id="rr-about-body">' + bodyHtml + "</div>";
    host.querySelector(".chev").addEventListener("click", () => {
      const closed = host.classList.toggle("closed");
      const btn = host.querySelector(".chev");
      btn.title = closed ? "Expand" : "Collapse";
      btn.setAttribute("aria-expanded", String(!closed));
    });
  }

  // ---------- form assembly ----------

  function renderForm(values) {
    const all = schema.params;
    form.replaceChildren();
    if (all.length === 0) {
      form.innerHTML =
        '<div class="rr-noparams">' + icon("box") +
          '<span class="t-title">This tool takes no parameters.</span>' +
          '<span class="t-sub">It answers with instructions — just start the run.</span>' +
        "</div>";
      refreshPreview();
      return;
    }
    const entries = all.map((p) => ({ p }));
    const req = entries.filter((e) => e.p.required);
    const opt = entries.filter((e) => !e.p.required);
    if (req.length) {
      const g = document.createElement("div");
      g.className = "rr-group";
      g.innerHTML = '<div class="rr-glabel">Required</div>';
      form.appendChild(g);
      for (const { p } of req) {
        const f = buildField(p, values);
        g.appendChild(f.wrap);
        fields.push(f);
      }
    }
    if (opt.length) {
      const g = document.createElement("div");
      g.className = "rr-group" + (optionalOpen ? "" : " closed");
      g.innerHTML =
        '<div class="rr-glabel">' +
          '<button class="rr-chev" type="button" title="' + (optionalOpen ? "Collapse" : "Expand") + '">' + icon("chevDown") + "</button>" +
          '<span class="rr-gt">Optional <span class="n" data-optn></span></span>' +
        "</div>" +
        '<div class="rr-gbody"></div>';
      const body = g.querySelector(".rr-gbody");
      g.querySelector(".rr-chev").addEventListener("click", () => {
        optionalOpen = !optionalOpen;
        g.classList.toggle("closed", !optionalOpen);
        g.querySelector(".rr-chev").title = optionalOpen ? "Collapse" : "Expand";
      });
      for (const { p } of opt) {
        const f = buildField(p, values);
        f.optGroup = true;
        body.appendChild(f.wrap);
        fields.push(f);
      }
      form.appendChild(g);
    }
    updateOptionalCount();
    refreshPreview();
  }

  function updateOptionalCount() {
    const n = el.querySelector("[data-optn]");
    if (!n) return;
    const optFields = fields.filter((f) => f.optGroup);
    const set = optFields.reduce((acc, f) => {
      const count = (p, read) => {
        const v = read();
        return v !== null && v !== undefined && v !== "" && !sameAsDefault(p, v) ? 1 : 0;
      };
      return acc + count(f.p, f.read) + (f.p2 ? count(f.p2, f.p2.read) : 0);
    }, 0);
    const total = optFields.reduce((acc, f) => acc + 1 + (f.p2 ? 1 : 0), 0);
    n.textContent = "· " + (set > 0 ? set + " set" : total);
  }

  // ---------- prefill from a prior run ----------

  async function prefill() {
    if (!from) return {};
    try {
      const body = await getJSON("/api/tasks/" + encodeURIComponent(from));
      return body.task && body.task.params || {};
    } catch {
      toast("Could not load the source run — starting with defaults", { error: true });
      return {};
    }
  }

  // ---------- submit ----------

  async function doRun() {
    if (!formReady || submitting || fields.some((f) => f.editing)) return;
    // A typed-but-uncommitted list draft joins the value before validation and the request.
    for (const fld of fields) if (fld.commit && fld.commit() === false) return;
    if (!validate()) return;
    hideBanner();
    well.classList.add("hidden");
    submitting = true;
    validatedOnce = true;
    runBtn.disabled = true;
    cancelBtn.disabled = true;
    runBtn.innerHTML = '<span class="br-spin" style="width:14px;height:14px;border-width:2px"></span>Starting…';
    const t0 = Date.now();
    try {
      const body = await postJSONBody("/api/tools/" + encodeURIComponent(tool) + "/call", buildBody());
      const wait = _MIN_SUBMIT_MS - (Date.now() - t0);
      if (wait > 0) await new Promise((r) => setTimeout(r, wait));
      if (body && body.task_id) {
        if (body.status !== "completed") {
          toast("Run started · " + body.task_id.slice(0, 8));
        }
        navigate("/task/" + encodeURIComponent(body.task_id));
        return;
      }
      const text = (body && typeof body.result === "string") ? body.result : JSON.stringify(body);
      well.innerHTML = '<pre class="rr-well-t">' + esc(text) + "</pre>";
      well.classList.remove("hidden");
      toast("Completed without a run");
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : "The call failed — try again.";
      showBanner(msg);
      toast("Run failed: " + msg, { error: true });
    } finally {
      submitting = false;
      runBtn.disabled = !formReady || fields.some((field) => field.editing);
      cancelBtn.disabled = false;
      runLabel();
    }
  }

  function onCancel() {
    if (history.length > 1) history.back();
    else navigate("/tools/" + encodeURIComponent(tool));
  }

  runBtn.addEventListener("click", doRun);
  cancelBtn.addEventListener("click", onCancel);
  el.querySelector("#rr-m-cancel").addEventListener("click", onCancel);
  el.querySelector(".rr-col").addEventListener("keydown", (ev) => {
    if (ev.key !== "Enter") return;
    const t = ev.target;
    if (t && (t.tagName === "TEXTAREA" || t.tagName === "BUTTON" || t.classList.contains("lst-in") || t.closest(".pick-pop"))) return;
    ev.preventDefault();
    doRun();
  });

  skeletonRows(form, 6);
  const t0 = Date.now();
  let schemaBody = null;
  try {
    schemaBody = await getJSON("/api/tools/" + encodeURIComponent(tool) + "/schema");
  } catch (err) {
    if (err instanceof ApiError && err.status === 403) notExposed();
    else if (err instanceof ApiError && err.status === 404) notFound();
    else {
      el.replaceChildren();
      emptyState(el, "alert", "Cannot load tool schema", err.message || err);
    }
    return { destroy() { document.body.classList.remove("pushed"); } };
  }
  const rest = SKELETON_MS - (Date.now() - t0);
  if (rest > 0) await new Promise((r) => setTimeout(r, rest));
  schema = schemaBody;
  el.querySelector("#rr-sub").textContent = schema.summary || "";
  renderAbout();
  renderForm(await prefill());
  formReady = true;
  runBtn.disabled = submitting || fields.some((field) => field.editing);

  return {
    destroy() {
      document.body.classList.remove("pushed");
      for (const fld of fields) {
        if (fld.picker) fld.picker.destroy();
        if (fld.stop) fld.stop();
      }
    },
  };
}

// ---------- resource picker ----------

function createPicker(anchor, opts) {
  const multi = !!opts.multi;
  const phone = _isPhone();
  const cache = new Map();
  const entryMap = new Map();
  const openSet = new Set();
  const selected = new Set();
  let singleSel = null;
  let searchSeq = 0;
  let rootLoaded = false;
  let lastFocus = null;
  let inFlight = null;
  let batchCancelled = false;

  const pop = document.createElement("div");
  pop.className = "pick-pop" + (phone ? " full" : "");
  pop.innerHTML =
    '<div class="pick-mbar">' +
      '<button class="icon-btn" data-act="mback" title="Back">' + icon("chevL") + "</button>" +
      '<span class="pick-mtitle">Choose from Resources</span>' +
      '<span class="spacer"></span>' +
      '<button class="icon-btn" data-act="mupload" title="Upload to upload/">' + icon("upload") + "</button>" +
      (multi
        ? '<button class="btn solid" data-act="madd" disabled>Add (0)</button>'
        : '<button class="btn solid" data-act="mchoose" disabled>Choose</button>') +
    "</div>" +
    '<div class="pick-head">' +
      '<span class="pick-title">Choose from Resources</span>' +
      '<div class="input-wrap pick-searchw">' + icon("search") +
        '<input class="input" type="text" placeholder="Search files…" autocomplete="off" spellcheck="false">' +
      "</div>" +
      '<button class="icon-btn" title="Upload to upload/">' + icon("upload") + "</button>" +
      '<input type="file" multiple class="pick-file" hidden>' +
    "</div>" +
    '<div class="pick-msearch"><div class="input-wrap">' + icon("search") +
      '<input class="input pick-msin" type="text" placeholder="Search files…" autocomplete="off" spellcheck="false">' +
    "</div></div>" +
    '<div class="pick-prog hidden"></div>' +
    '<div class="pick-body"><div class="pick-tree"></div></div>' +
    '<div class="pick-drop" hidden><span>Drop to upload</span></div>' +
    '<div class="pick-foot">' +
      (multi
        ? '<span class="pick-n mono">0 selected</span><span class="spacer"></span>' +
          '<button class="btn ghost" data-act="cancel">Cancel</button>' +
          '<button class="btn solid" data-act="add" disabled>Add (0)</button>'
        : '<span class="pick-n mono"></span><span class="spacer"></span>' +
          '<button class="btn ghost" data-act="cancel">Cancel</button>' +
          '<button class="btn solid" data-act="choose" disabled>Choose</button>') +
    "</div>";
  anchor.appendChild(pop);

  const tree = pop.querySelector(".pick-tree");
  const searchIn = pop.querySelector(".pick-searchw .input");
  const mSearchIn = pop.querySelector(".pick-msin");
  const uploadBtn = pop.querySelector(".pick-head .icon-btn");
  const fileIn = pop.querySelector(".pick-file");
  const prog = pop.querySelector(".pick-prog");
  const footN = pop.querySelector(".pick-n");
  const addBtn = pop.querySelector('[data-act="add"]');
  const chooseBtn = pop.querySelector('[data-act="choose"]');
  const cancelBtn = pop.querySelector('[data-act="cancel"]');
  const mback = pop.querySelector('[data-act="mback"]');
  const mupload = pop.querySelector('[data-act="mupload"]');
  const madd = pop.querySelector('[data-act="madd"]');
  const mchoose = pop.querySelector('[data-act="mchoose"]');
  const drop = pop.querySelector(".pick-drop");

  function updateFoot() {
    if (multi) {
      footN.textContent = selected.size + " selected";
      addBtn.textContent = "Add (" + selected.size + ")";
      addBtn.disabled = selected.size === 0;
      madd.textContent = "Add (" + selected.size + ")";
      madd.disabled = selected.size === 0;
    } else {
      footN.textContent = singleSel ? singleSel.split("/").pop() : "";
      chooseBtn.disabled = !singleSel;
      mchoose.disabled = !singleSel;
    }
  }

  function close() {
    batchCancelled = true;
    watch.stop();
    abortUpload();
    pop.classList.remove("show");
    document.removeEventListener("mousedown", onOutside);
    document.removeEventListener("keydown", onEsc);
    document.body.classList.remove("pick-lock");
    if (lastFocus) lastFocus.focus();
  }

  function onOutside(ev) {
    if (!pop.contains(ev.target)) close();
  }

  function onEsc(ev) {
    if (ev.key === "Escape") close();
  }

  function open(trigger) {
    lastFocus = trigger;
    const r = trigger.getBoundingClientRect();
    // The picker opens on whichever side has more room.
    const spaceAbove = r.top - 64;
    const spaceBelow = window.innerHeight - 70 - r.bottom;
    const up = spaceAbove >= spaceBelow;
    pop.classList.toggle("up", up);
    pop.style.maxHeight = Math.max(_POP_MIN, Math.min(_POP_MAX, up ? spaceAbove : spaceBelow)) + "px";
    pop.classList.add("show");
    if (phone) document.body.classList.add("pick-lock");
    document.addEventListener("mousedown", onOutside);
    document.addEventListener("keydown", onEsc);
    if (!rootLoaded) loadRoot();
    (phone ? mSearchIn : searchIn).focus();
  }

  function rowHtml(e, depth, prefix) {
    const dir = e.type === "dir";
    const isOpen = openSet.has(e.path);
    const nameHtml = prefix
      ? '<span class="t-name t-path" title="' + esc(e.path) + '"><span class="t-prefix">' + esc(prefix) + "</span>" + esc(e.name) + "</span>"
      : '<span class="t-name">' + esc(e.name) + (dir ? "/" : "") + "</span>";
    return (
      (dir ? '<button class="t-caret' + (isOpen ? " open" : "") + '" title="' + (isOpen ? "Collapse" : "Expand") + '">' + icon("caret") + "</button>" : '<span class="pick-pad"></span>') +
      (dir ? icon("folderSm") : compactSlot(e)) +
      nameHtml +
      (dir && e.name === "upload" ? '<span class="rw-pill">RW</span>' : "") +
      (dir
        ? '<span class="t-meta">' + (e.file_count || 0) + ((e.file_count === 1) ? " file" : " files") + "</span>"
        : '<span class="t-meta">' + fmtBytes(e.size || 0) + "</span>") +
      (multi && !dir ? '<span class="pick-cb" role="checkbox" aria-checked="false"><span class="tick"></span></span>' : "")
    );
  }

  function fileRow(e, depth, prefix) {
    const row = document.createElement("div");
    row.className = "pick-row";
    row.dataset.path = e.path;
    row.dataset.thumb = e.type === "file" && e.thumb ? e.thumb.status : "none";
    row.dataset.depth = depth;
    if (prefix) row.dataset.prefix = prefix;
    row.style.setProperty("--t-depth", depth);
    row.innerHTML = rowHtml(e, depth, prefix);
    if (e.type === "file") entryMap.set(e.path, e);
    const select = () => {
      if (multi) {
        if (selected.has(e.path)) selected.delete(e.path);
        else selected.add(e.path);
        const cb = row.querySelector(".pick-cb");
        cb.classList.toggle("on", selected.has(e.path));
        cb.setAttribute("aria-checked", String(selected.has(e.path)));
        updateFoot();
      } else {
        singleSel = e.path;
        tree.querySelectorAll(".pick-row.on").forEach((n) => n.classList.remove("on"));
        row.classList.add("on");
        updateFoot();
      }
    };
    row.addEventListener("click", (ev) => {
      if (ev.target.closest(".t-caret")) return;
      if (e.type === "dir") {
        toggleFolder(e, row);
        return;
      }
      select();
    });
    const caret = row.querySelector(".t-caret");
    if (caret) caret.addEventListener("click", (ev) => {
      ev.stopPropagation();
      toggleFolder(e, row);
    });
    return row;
  }

  function statusRow(text, depth, retry) {
    const row = document.createElement("div");
    row.className = "pick-status";
    row.style.setProperty("--t-depth", depth);
    row.innerHTML = esc(text) + (retry ? '<button class="btn ghost sm" type="button">Retry</button>' : "");
    if (retry) row.querySelector("button").addEventListener("click", retry);
    return row;
  }

  async function loadKids(p, box) {
    const depth = 1 + p.split("/").length;
    box.replaceChildren();
    for (let i = 0; i < 3; i++) {
      const sk = document.createElement("div");
      sk.className = "pick-skel";
      box.appendChild(sk);
    }
    const t0 = Date.now();
    try {
      const body = await getJSON("/api/resources?" + new URLSearchParams({ path: p }));
      cache.set(p, body.entries);
      const rest = _TREE_SKELETON_MS - (Date.now() - t0);
      if (rest > 0) await new Promise((r) => setTimeout(r, rest));
      fillKids(p, box);
    } catch {
      box.replaceChildren(statusRow("Could not open " + p + "/", depth, () => loadKids(p, box)));
    }
  }

  async function toggleFolder(e, row) {
    const p = e.path;
    if (openSet.has(p)) {
      openSet.delete(p);
      row.classList.remove("open");
      row.querySelector(".t-caret").classList.remove("open");
      const box = tree.querySelector('[data-children="' + CSS.escape(p) + '"]');
      if (box) box.classList.remove("open");
      return;
    }
    openSet.add(p);
    row.classList.add("open");
    row.querySelector(".t-caret").classList.add("open");
    const box = document.createElement("div");
    box.className = "pick-kids open";
    box.dataset.children = p;
    row.after(box);
    if (cache.has(p)) {
      fillKids(p, box);
      return;
    }
    await loadKids(p, box);
  }

  function fillKids(p, box) {
    const depth = 1 + p.split("/").length;
    const entries = cache.get(p);
    if (entries.length === 0) {
      box.replaceChildren(statusRow("Empty folder.", depth));
      return;
    }
    box.replaceChildren();
    for (const e of sortEntries(entries)) box.appendChild(fileRow(e, depth));
    watch.start();
  }

  // Swap one row's art in place. The row element keeps its position, scroll,
  // and selection. Only the slot and meta come from the fresh entry.
  function refreshRow(row, e) {
    row.innerHTML = rowHtml(e, Number(row.dataset.depth), row.dataset.prefix || "");
    row.dataset.thumb = e.thumb ? e.thumb.status : "none";
    entryMap.set(e.path, e);
  }

  // One short poll while any row is still preparing. It asks the server to
  // make the thumbnails, re-lists the folders that still have work, and
  // re-renders the rows in place. It stops when nothing is pending or the
  // picker closes.
  const watch = watchThumbs({
    active: () => pop.isConnected && pop.classList.contains("show"),
    dirsInScope: () => {
      const dirs = new Set();
      for (const row of tree.querySelectorAll('.pick-row[data-thumb="pending"], .pick-row[data-thumb="generating"]')) {
        dirs.add(row.dataset.path.split("/").slice(0, -1).join("/"));
      }
      return dirs;
    },
    applyDir: (d, entries) => {
      cache.set(d, entries);
      const byPath = new Map(entries.map((x) => [x.path, x]));
      for (const row of tree.querySelectorAll(".pick-row")) {
        const e = byPath.get(row.dataset.path);
        if (e && e.type === "file") refreshRow(row, e);
      }
      syncSelection();
    },
  });

  async function loadRoot() {
    rootLoaded = true;
    tree.replaceChildren();
    for (let i = 0; i < 4; i++) {
      const sk = document.createElement("div");
      sk.className = "pick-skel";
      tree.appendChild(sk);
    }
    const t0 = Date.now();
    try {
      const body = await getJSON("/api/resources?" + new URLSearchParams({ path: "" }));
      const rest = _TREE_SKELETON_MS - (Date.now() - t0);
      if (rest > 0) await new Promise((r) => setTimeout(r, rest));
      tree.replaceChildren();
      for (const e of sortEntries(body.entries)) tree.appendChild(fileRow(e, 0));
      watch.start();
    } catch {
      tree.innerHTML = '<div class="pick-empty">Could not load the resource.</div>';
    }
  }

  async function search(q) {
    const seq = ++searchSeq;
    if (!q) {
      loadRoot();
      return;
    }
    tree.replaceChildren();
    for (let i = 0; i < 3; i++) {
      const sk = document.createElement("div");
      sk.className = "pick-skel";
      tree.appendChild(sk);
    }
    try {
      const body = await getJSON("/api/resources/search?" + new URLSearchParams({ q }));
      if (seq !== searchSeq) return;
      tree.replaceChildren();
      const files = body.matches.filter((m) => m.type === "file");
      if (files.length === 0) {
        tree.innerHTML = '<div class="pick-empty">No matches for "' + esc(q) + '"</div>';
        return;
      }
      for (const m of files) {
        const e = { name: m.name, path: m.path, type: "file", size: m.size || 0, kind: m.kind, thumb: m.thumb };
        const prefix = m.path.includes("/") ? m.path.slice(0, m.path.lastIndexOf("/") + 1) : "";
        tree.appendChild(fileRow(e, e.path.split("/").length, prefix));
      }
      watch.start();
    } catch {
      if (seq !== searchSeq) return;
      tree.innerHTML = '<div class="pick-empty">Search failed.</div>';
    }
  }

  const bindSearchInput = (input) => {
    let debounce = null;
    input.addEventListener("input", () => {
      clearTimeout(debounce);
      debounce = setTimeout(() => search(input.value.trim()), _SEARCH_DEBOUNCE_MS);
    });
  };
  bindSearchInput(searchIn);
  bindSearchInput(mSearchIn);

  // ---------- upload ----------

  function existsInUpload(name) {
    return getJSON("/api/resources?" + new URLSearchParams({ path: "upload" }))
      .then((body) => body.entries.some((e) => e.name === name && e.type === "file"))
      .catch(() => false);
  }

  function uploadFile(file) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      inFlight = { xhr, file };
      xhr.open("PUT", davHref("upload/" + file.name));
      xhr.upload.onprogress = (ev) => {
        if (!ev.lengthComputable) return;
        setProgPct(Math.round((ev.loaded / ev.total) * 100), ev.loaded, ev.total);
      };
      xhr.onload = () => { inFlight = null; (xhr.status < 300 ? resolve() : reject(new Error("HTTP " + xhr.status))); };
      xhr.onerror = () => { inFlight = null; reject(new Error("upload failed")); };
      xhr.onabort = () => { inFlight = null; reject(new Error("aborted")); };
      xhr.send(file);
    });
  }

  function showProg(file) {
    prog.replaceChildren();
    prog.classList.remove("fail", "hidden");
    prog.innerHTML =
      icon("upload") +
      '<span class="prog-name" title="' + esc(file.name) + '">' + esc(file.name) + "</span>" +
      '<span class="prog-bar"><span class="fill"></span></span>' +
      '<span class="prog-pct mono"></span>' +
      '<button class="prog-x" title="Cancel upload">' + icon("x") + "</button>";
    prog.querySelector(".prog-x").addEventListener("click", cancelUpload);
  }

  function setProgPct(pct, loaded, total) {
    const fill = prog.querySelector(".prog-bar .fill");
    if (fill) fill.style.width = pct + "%";
    const pctEl = prog.querySelector(".prog-pct");
    if (pctEl) pctEl.textContent = pct + "% · " + fmtBytesPair(loaded, total);
  }

  function hideProg() {
    prog.classList.add("hidden");
    prog.classList.remove("fail");
    prog.replaceChildren();
  }

  function setProgFail(msg, file) {
    prog.classList.remove("hidden");
    prog.classList.add("fail");
    prog.innerHTML =
      icon("upload") +
      '<span class="prog-fail">' + esc(msg) + "</span>" +
      (file ? '<button class="btn ghost sm prog-retry" type="button">Retry</button>' : "");
    if (file) prog.querySelector(".prog-retry").addEventListener("click", () => uploadOne(file));
  }

  function deletePartial(name) {
    fetch(davHref("upload/" + name), { method: "DELETE" }).catch(() => {});
  }

  function abortUpload() {
    const uf = inFlight;
    if (!uf) return;
    inFlight = null;
    uf.xhr.abort();
    deletePartial(uf.file.name);
    hideProg();
    toast("Upload cancelled.");
  }

  function cancelUpload() {
    if (!inFlight) return;
    batchCancelled = true;
    abortUpload();
    updateFoot();
  }

  function pulseUpload() {
    const btn = phone ? mupload : uploadBtn;
    btn.classList.add("pulse");
    btn.addEventListener("animationend", () => btn.classList.remove("pulse"), { once: true });
  }

  function syncSelection() {
    tree.querySelectorAll(".pick-row.on").forEach((n) => n.classList.remove("on"));
    if (multi) {
      for (const cb of tree.querySelectorAll(".pick-cb")) {
        const path = cb.closest(".pick-row").dataset.path;
        const on = selected.has(path);
        cb.classList.toggle("on", on);
        cb.setAttribute("aria-checked", String(on));
      }
    } else if (singleSel) {
      const row = tree.querySelector('.pick-row[data-path="' + CSS.escape(singleSel) + '"]');
      if (row) row.classList.add("on");
    }
  }

  async function refreshUploadFolder() {
    const upRow = tree.querySelector('.pick-row[data-path="upload"]');
    if (!upRow) return;
    openSet.add("upload");
    upRow.classList.add("open");
    const caret = upRow.querySelector(".t-caret");
    if (caret) caret.classList.add("open");
    let box = tree.querySelector('[data-children="upload"]');
    if (!box) {
      box = document.createElement("div");
      box.className = "pick-kids open";
      box.dataset.children = "upload";
      upRow.after(box);
    } else {
      box.classList.add("open");
    }
    const body = await getJSON("/api/resources?" + new URLSearchParams({ path: "upload" }));
    cache.set("upload", body.entries);
    fillKids("upload", box);
  }

  async function uploadOne(file) {
    const exists = await existsInUpload(file.name);
    if (exists) toast("upload/" + file.name + " already exists — overwriting");
    showProg(file);
    if (addBtn) addBtn.disabled = true;
    try {
      await uploadFile(file);
      hideProg();
      toast("Uploaded to upload/" + file.name);
      cache.delete("");
      if (multi) selected.add("upload/" + file.name);
      else singleSel = "upload/" + file.name;
      await refreshUploadFolder();
      syncSelection();
    } catch (err) {
      if (err.message === "aborted") return;
      setProgFail("Upload failed — try again", file);
      toast("Upload failed: " + (err.message || err), { error: true });
    } finally {
      updateFoot();
    }
  }

  async function handleFiles(files) {
    batchCancelled = false;
    for (const file of files) {
      if (batchCancelled) return;
      if (file.size > _UPLOAD_MAX) {
        toast(file.name + " is too large (max 100 MB)", { error: true });
        setProgFail("Too large (max 100 MB)");
        pulseUpload();
        continue;
      }
      await uploadOne(file);
    }
  }

  uploadBtn.addEventListener("click", () => fileIn.click());
  mupload.addEventListener("click", () => fileIn.click());
  fileIn.addEventListener("change", () => {
    handleFiles([...fileIn.files]);
    fileIn.value = "";
  });

  if (!phone) {
    pop.addEventListener("dragover", (ev) => {
      ev.preventDefault();
      drop.hidden = false;
    });
    pop.addEventListener("dragleave", (ev) => {
      if (!pop.contains(ev.relatedTarget)) drop.hidden = true;
    });
    pop.addEventListener("drop", (ev) => {
      ev.preventDefault();
      drop.hidden = true;
      handleFiles([...ev.dataTransfer.files]);
    });
  }

  // A committed file is handed to the field as its full fact when the row
  // knows it, and as a bare path when it does not.
  function entryFor(p) {
    return entryMap.get(p) || p;
  }

  // ---------- footer actions ----------

  cancelBtn.addEventListener("click", close);
  mback.addEventListener("click", close);
  if (addBtn) {
    const doAdd = () => {
      opts.onPick([...selected].map(entryFor));
      selected.clear();
      updateFoot();
      close();
    };
    addBtn.addEventListener("click", doAdd);
    madd.addEventListener("click", doAdd);
  }
  if (chooseBtn) {
    const doChoose = () => {
      if (!singleSel) return;
      opts.onPick([entryFor(singleSel)]);
      close();
    };
    chooseBtn.addEventListener("click", doChoose);
    mchoose.addEventListener("click", doChoose);
  }

  function destroy() {
    batchCancelled = true;
    watch.stop();
    pop.remove();
  }

  return { open, close, destroy };
}
