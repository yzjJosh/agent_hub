import { getState, on } from "../state.js";
import { getJSON } from "../api.js";
import { esc, parseUtc, serverNow, fmtDur, relLabel, startTicker } from "../format.js";
import { icon, skeletonRows, setLastPolled, FAIL_BANNER_AT, DEBOUNCE_MS, SKELETON_MS } from "../ui.js";
import { navigate } from "../router.js";

export async function renderTools(el, _params) {
  const s = getState();
  el.innerHTML =
    '<div class="page-head">' +
      '<span class="page-title">Tools</span>' +
      '<span class="t3" style="font-size:var(--text-sm)">Last polled <span class="mono lp"></span></span>' +
    "</div>" +
    '<div class="l1-toolbar toolbar"><div class="input-wrap">' + icon("search") +
      '<input class="input" id="l1-q" type="text" placeholder="Search tools…" autocomplete="off" spellcheck="false">' +
    "</div></div>" +
    '<div class="pause-banner hidden" id="l1-pause">Live updates paused — retrying <span class="mono"></span></div>' +
    '<div class="l1-scroll">' +
      '<div class="l1-colhead">' +
        '<span class="h-tool">Tool</span><span class="h-run">Running</span><span class="h-comp">Completed</span>' +
        '<span class="h-fail">Failed</span><span class="h-canc">Cancelled</span><span class="h-last">Last activity</span>' +
        '<span class="h-chev"></span>' +
      "</div>" +
      '<div id="l1-list"></div>' +
      '<div class="l1-mob" id="l1m-list"></div>' +
      '<div class="empty hidden" id="l1-empty">' +
        '<span class="empty-ic">' + icon("search") + "</span>" +
        '<span class="t-title">No tools match</span>' +
        '<button class="btn ghost" id="l1-clear">Clear search</button>' +
      "</div>" +
    "</div>";

  const list = el.querySelector("#l1-list");
  const mlist = el.querySelector("#l1m-list");
  const emptyBox = el.querySelector("#l1-empty");
  const pause = el.querySelector("#l1-pause");
  const pauseMeta = pause.querySelector(".mono");
  const qInput = el.querySelector("#l1-q");
  const rows = new Map(); // one entry per tool name
  let hover = null;
  let pendingLayout = false;
  let q = "";
  let ready = false;

  function rowHtml(name, t) {
    return '<span class="c-name"><span class="tname' + (t.running + t.completed + t.failed + t.cancelled === 0 ? " quiet" : "") + '">' + esc(name) + "</span>" +
      '<span class="tdesc">' + esc(t.description || "") + "</span></span>" +
      '<span class="c-run"></span><span class="c-stat completed"></span><span class="c-stat failed"></span>' +
      '<span class="c-stat cancelled"></span><span class="c-last"></span>' +
      '<span class="c-chev">' +
        '<button class="icon-btn l1-play" type="button" title="Run ' + esc(name) + '">' + icon("play") + "</button>" +
        icon("chevR") +
      "</span>";
  }

  function cardHtml(name, t) {
    const total = t.running + t.completed + t.failed + t.cancelled;
    return '<span class="m-l1"><span class="m-name' + (total === 0 ? " quiet" : "") + '">' + esc(name) + "</span>" +
      '<span class="m-runpill hidden" data-runpill><span class="s-dot running"></span><span data-runcount></span><span class="m-rlabel">running</span></span>' +
      '<span class="spacer"></span><span class="m-last"></span></span>' +
      '<span class="m-l2">' +
        '<span class="m-none hidden" data-mnone>No runs yet</span>' +
        '<span class="m-stat" data-mc="completed"><span class="s-dot completed"></span><span data-mn></span><span class="m-slabel">completed</span></span>' +
        '<span class="m-stat" data-mc="failed"><span class="s-dot failed"></span><span data-mn></span><span class="m-slabel">failed</span></span>' +
        '<span class="m-stat" data-mc="cancelled"><span class="s-dot cancelled"></span><span data-mn></span><span class="m-slabel">cancelled</span></span>' +
      "</span>";
  }

  function build(name) {
    const t = s.tools.get(name);
    const a = document.createElement("a");
    a.className = "l1-row";
    a.setAttribute("data-spa", "");
    a.href = "/tools/" + encodeURIComponent(name);
    a.innerHTML = rowHtml(name, t);
    const c = document.createElement("a");
    c.className = "m-card";
    c.setAttribute("data-spa", "");
    c.href = "/tools/" + encodeURIComponent(name);
    c.innerHTML = cardHtml(name, t);
    rows.set(name, { row: a, card: c, q: (name + " " + (t.description || "")).toLowerCase() });
    a.querySelector(".l1-play").addEventListener("click", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      navigate("/tools/" + encodeURIComponent(name) + "/run");
    });
    a.addEventListener("mouseenter", () => { hover = name; });
    a.addEventListener("mouseleave", () => {
      if (hover === name) {
        hover = null;
        if (pendingLayout) place();
      }
    });
    updateRow(name);
  }

  function updateRow(name) {
    const t = s.tools.get(name);
    const r = rows.get(name);
    if (!t || !r) return;
    const total = t.running + t.completed + t.failed + t.cancelled;
    a11y(r.row, name, t);
    a11y(r.card, name, t);
    const run = r.row.querySelector(".c-run");
    if (t.running > 0) {
      run.className = "c-run live";
      run.innerHTML = '<span class="run-pill"><span class="dot running"></span>' + t.running + "</span>";
    } else {
      run.className = "c-run";
      run.textContent = "0";
    }
    for (const k of ["completed", "failed", "cancelled"]) {
      const cell = r.row.querySelector(".c-stat." + k);
      cell.textContent = String(t[k]);
      cell.classList.toggle("on", t[k] > 0);
    }
    setLast(r.row.querySelector(".c-last"), t);
    setLastM(r.card, t);
    const pill = r.card.querySelector("[data-runpill]");
    pill.classList.toggle("hidden", t.running === 0);
    pill.querySelector("[data-runcount]").textContent = String(t.running);
    for (const k of ["completed", "failed", "cancelled"]) {
      const m = r.card.querySelector('[data-mc="' + k + '"]');
      m.querySelector("[data-mn]").textContent = String(t[k]);
      m.classList.toggle("zero", t[k] === 0);
    }
    r.card.querySelector("[data-mnone]").classList.toggle("hidden", total !== 0);
  }

  function a11y(node, name, t) {
    node.setAttribute("aria-label", name + ", " + t.running + " running");
  }

  function setLast(cell, t) {
    if (t.running > 0) {
      cell.className = "c-last live";
      cell.textContent = fmtDur(serverNow() - parseUtc(t.running_since || t.last_activity));
      cell.removeAttribute("title");
    } else if (t.last_activity) {
      cell.className = "c-last";
      cell.textContent = relLabel(serverNow() - parseUtc(t.last_activity));
      cell.title = t.last_activity_local || "";
    } else {
      cell.className = "c-last none";
      cell.textContent = "No runs yet";
      cell.removeAttribute("title");
    }
  }

  function setLastM(card, t) {
    const cell = card.querySelector(".m-last");
    if (t.running > 0) {
      cell.className = "m-last live";
      cell.textContent = fmtDur(serverNow() - parseUtc(t.running_since || t.last_activity));
    } else if (t.last_activity) {
      cell.className = "m-last";
      cell.textContent = relLabel(serverNow() - parseUtc(t.last_activity));
    } else {
      cell.className = "m-last";
      cell.textContent = "No runs yet";
    }
  }

  function sortNames() {
    return [...s.tools.keys()].sort((a, b) => {
      const ra = s.tools.get(a).running > 0 ? 1 : 0;
      const rb = s.tools.get(b).running > 0 ? 1 : 0;
      if (ra !== rb) return rb - ra;
      return a < b ? -1 : a > b ? 1 : 0;
    });
  }

  // A row under the mouse never moves. The move happens once the mouse leaves.
  function place() {
    const order = sortNames();
    if (hover) {
      const cur = [...list.children].map((n) => n.getAttribute("data-name"));
      const targetIdx = order.indexOf(hover);
      if (targetIdx !== cur.indexOf(hover)) {
        pendingLayout = true;
        return;
      }
    }
    pendingLayout = false;
    for (const n of order) {
      list.appendChild(rows.get(n).row);
      mlist.appendChild(rows.get(n).card);
      rows.get(n).row.setAttribute("data-name", n);
      rows.get(n).card.setAttribute("data-name", n);
    }
    applyFilter();
  }

  function applyFilter() {
    let shownCount = 0;
    for (const [name, r] of rows) {
      const hit = !q || r.q.includes(q);
      r.row.classList.toggle("hidden", !hit);
      r.card.classList.toggle("hidden", !hit);
      if (hit) shownCount += 1;
    }
    emptyBox.classList.toggle("hidden", shownCount !== 0 || !q);
  }

  function updatePause() {
    if (s.failStreak >= FAIL_BANNER_AT) {
      pauseMeta.textContent = "last good poll " + (s.lastPollLocal || "—");
      pause.classList.remove("hidden");
    } else {
      pause.classList.add("hidden");
    }
  }

  el.querySelector("#l1-clear").addEventListener("click", () => {
    q = "";
    qInput.value = "";
    applyFilter();
  });
  let debounce = null;
  qInput.addEventListener("input", () => {
    clearTimeout(debounce);
    debounce = setTimeout(() => {
      q = qInput.value.trim().toLowerCase();
      applyFilter();
    }, DEBOUNCE_MS);
  });

  const off = on((kind) => {
    if (!ready) return;
    if (kind === "changes") {
      for (const name of s.tools.keys()) updateRow(name);
      place();
    }
    updatePause();
    setLastPolled(el, s.lastPollLocal);
  });
  const stopTick = startTicker(() => {
    if (!ready) return;
    for (const [name, t] of s.tools) if (t.running > 0) setLast(rows.get(name).row.querySelector(".c-last"), t);
    for (const [name, t] of s.tools) if (t.running > 0) setLastM(rows.get(name).card, t);
    setLastPolled(el, s.lastPollLocal);
  });

  setLastPolled(el, s.lastPollLocal);
  if (s.tools.size === 0) {
    skeletonRows(list, 6);
    skeletonRows(mlist, 6);
    const t0 = Date.now();
    try {
      const body = await getJSON("/api/tools");
      for (const t of body.tools) s.tools.set(t.name, t);
    } catch { /* the error shows in the top banner */ }
    const rest = SKELETON_MS - (Date.now() - t0);
    if (rest > 0) await new Promise((r) => setTimeout(r, rest));
    list.querySelectorAll(".skel").forEach((n) => n.remove());
    mlist.querySelectorAll(".skel").forEach((n) => n.remove());
  }
  ready = true;
  for (const name of s.tools.keys()) build(name);
  place();
  updatePause();

  return {
    destroy() {
      off();
      stopTick();
      clearTimeout(debounce);
    },
  };
}
