import { getState, on } from "../state.js";
import { getJSON, ApiError } from "../api.js";
import { esc, parseUtc, serverNow, fmtClock, relLabel, fmtDur, startTicker } from "../format.js";
import { icon, paramSummary, paramTitle, skeletonRows, emptyState, setLastPolled, DEBOUNCE_MS, SKELETON_MS } from "../ui.js";
import { copyText, toast } from "../toast.js";
import { syncGlyphs } from "../theme.js";
import { navigate } from "../router.js";

const _PAGE = 200;

export async function renderToolTasks(el, params) {
  const s = getState();
  const tool = params.tool;
  el.innerHTML =
    '<header class="pushbar">' +
      '<a class="icon-btn" data-spa href="/tools" title="Back to Tools">' + icon("chevL") + "</a>" +
      '<span class="bar-title">' + esc(tool) + "</span>" +
      '<span class="spacer"></span>' +
      '<button class="icon-btn" data-theme-toggle title="Switch theme"></button>' +
    "</header>" +
    '<div class="tt-head">' +
      '<a class="icon-btn" data-spa href="/tools" title="Back to Tools">' + icon("chevL") + "</a>" +
      '<span class="tile">' + icon("tool") + "</span>" +
      '<span class="tt-name">' + esc(tool) + '<span class="tt-sub" id="tt-sub"></span></span>' +
      '<span class="spacer"></span>' +
      '<button class="btn solid tt-run" type="button">' + icon("play") + "<span>Run tool</span></button>" +
      '<span class="tt-lp t3">Last polled <span class="mono lp"></span></span>' +
    "</div>" +
    '<div class="tt-strip" id="tt-strip"></div>' +
    '<div class="tt-toolbar">' +
      '<div class="input-wrap">' + icon("search") +
        '<input class="input" id="tt-q" type="text" placeholder="Search task id, or params…" autocomplete="off" spellcheck="false">' +
      "</div>" +
      '<select class="tt-select" id="tt-status" aria-label="Status filter">' +
        '<option value="">All</option><option value="completed">completed</option>' +
        '<option value="failed">failed</option><option value="cancelled">cancelled</option>' +
      "</select>" +
    "</div>" +
    '<div class="tt-m-head" id="tt-m-head"></div>' +
    '<button class="btn solid tt-run-m" type="button">' + icon("play") + "<span>Run tool</span></button>" +
    '<div class="tt-m-search"><div class="input-wrap">' + icon("search") +
      '<input class="input" id="tt-mq" type="text" placeholder="Search task id, or params…" autocomplete="off" spellcheck="false">' +
    "</div></div>" +
    '<div class="tt-chips" id="tt-chips">' +
      '<button class="chip active" data-v="">All</button>' +
      '<button class="chip" data-v="completed"><span class="cdot" style="background:var(--completed)"></span>completed</button>' +
      '<button class="chip" data-v="failed"><span class="cdot" style="background:var(--failed)"></span>failed</button>' +
      '<button class="chip" data-v="cancelled"><span class="cdot" style="background:var(--cancelled)"></span>cancelled</button>' +
    "</div>" +
    '<div class="t-group" id="g-run" style="display:none"><span class="s-dot running"></span>Running <span class="n" id="n-run">· 0</span></div>' +
    '<div class="t-group" id="g-hist">History <span class="n" id="n-hist">· 0</span></div>' +
    '<div class="t-list" id="tt-list"></div>' +
    '<div class="load-older hidden" id="tt-load"><button class="btn ghost" id="tt-load-btn">Showing most recent ' + _PAGE + " — Load older</button></div>";
  syncGlyphs();
  document.body.classList.add("pushed");

  const list = el.querySelector("#tt-list");
  const gRun = el.querySelector("#g-run");
  const gHist = el.querySelector("#g-hist");
  const nRun = el.querySelector("#n-run");
  const nHist = el.querySelector("#n-hist");
  const loadWrap = el.querySelector("#tt-load");
  const loadBtn = el.querySelector("#tt-load-btn");
  const runHref = "/tools/" + encodeURIComponent(tool) + "/run";
  el.querySelector(".tt-run").addEventListener("click", () => navigate(runHref));
  el.querySelector(".tt-run-m").addEventListener("click", () => navigate(runHref));
  const data = new Map();
  const rows = new Map();
  let nextCursor = null;
  let q = "";
  let status = "";
  let hover = null;
  let pendingLayout = false;
  let ready = false;

  function notFound() {
    ready = false;
    el.replaceChildren();
    emptyState(el, "alert", "Tool not found or expired",
      "This tool no longer exists in the registry.", "Back to Tools", "/tools");
  }

  function renderLoadError(err) {
    gRun.style.display = "none";
    gHist.style.display = "none";
    list.replaceChildren();
    emptyState(list, "alert", "Cannot load tasks", err.message || err);
  }

  function rowHtml(t) {
    const st = esc(t.status);
    return '<span class="c-status"><span class="s-dot ' + st + '"></span>' +
      '<span class="badge ' + st + '">' + st + "</span></span>" +
      '<span class="c-id" title="Copy task id">' + esc(t.task_id.slice(0, 8)) + "</span>" +
      '<span class="c-when"></span><span class="c-dur"></span>' +
      '<span class="c-params">' + paramSummary(t.params) + "</span>" +
      '<span class="c-chev">' + icon("chevR") + "</span>";
  }

  function cardHtml(t) {
    const st = esc(t.status);
    return '<span class="mt-l1"><span class="badge ' + st + '">' + st + "</span>" +
      '<span class="c-id" title="Copy task id">' + esc(t.task_id.slice(0, 8)) + "</span>" +
      '<span class="spacer"></span><span class="mt-elapsed"></span></span>' +
      '<span class="mt-l2"><span class="mt-when"></span></span>' +
      '<span class="mt-l3">' + paramSummary(t.params) + "</span>";
  }

  function upsert(t, isNew) {
    const prev = rows.get(t.task_id);
    if (!prev) {
      const href = "/task/" + encodeURIComponent(t.task_id);
      const a = document.createElement("a");
      a.className = "t-row" + (isNew ? " fade-in" : "");
      a.setAttribute("data-spa", "");
      a.href = href;
      a.title = paramTitle(t.params);
      a.innerHTML = rowHtml(t);
      const c = document.createElement("a");
      c.className = "m-tcard";
      c.setAttribute("data-spa", "");
      c.href = href;
      c.innerHTML = cardHtml(t);
      for (const node of [a, c]) {
        node.dataset.tid = t.task_id;
        node.querySelector(".c-id").addEventListener("click", (ev) => {
          ev.preventDefault();
          ev.stopPropagation();
          copyText(t.task_id, "Task id copied");
        });
        node.addEventListener("mouseenter", () => { hover = t.task_id; });
        node.addEventListener("mouseleave", () => {
          if (hover === t.task_id) {
            hover = null;
            if (pendingLayout) layout();
          }
        });
      }
      rows.set(t.task_id, { row: a, card: c, status: t.status, search: (t.task_id + " " + paramTitle(t.params)).toLowerCase() });
      return;
    }
    if (prev.status !== t.status) {
      for (const node of [prev.row, prev.card]) {
        const b = node.querySelector(".badge");
        b.className = "badge " + t.status;
        b.textContent = t.status;
        b.classList.add("flip");
        const dot = node.querySelector(".s-dot");
        if (dot) dot.className = "s-dot " + t.status;
      }
      prev.status = t.status;
    }
    prev.search = (t.task_id + " " + paramTitle(t.params)).toLowerCase();
  }

  function cellUpdate(t) {
    const r = rows.get(t.task_id);
    if (!r) return;
    const started = parseUtc(t.started_at);
    const ended = parseUtc(t.ended_at);
    const dur = t.status === "running" ? fmtDur(serverNow() - started)
      : (Number.isFinite(ended) ? fmtDur(ended - started) : "");
    const rel = relLabel(serverNow() - started);
    const deskWhen = r.row.querySelector(".c-when");
    const deskDur = r.row.querySelector(".c-dur");
    const mWhen = r.card.querySelector(".mt-when");
    const mDur = r.card.querySelector(".mt-elapsed");
    if (t.status === "running") {
      deskWhen.className = "c-when live";
      deskWhen.textContent = fmtClock(started);
      deskWhen.removeAttribute("title");
      deskDur.className = "c-dur live";
      deskDur.textContent = dur;
      mWhen.className = "mt-when live";
      mWhen.textContent = fmtClock(started);
      mDur.className = "mt-elapsed live";
      mDur.textContent = dur;
    } else {
      deskWhen.className = "c-when";
      deskWhen.textContent = rel;
      deskWhen.title = t.started_at_local || "";
      deskDur.className = "c-dur";
      deskDur.textContent = dur;
      mWhen.className = "mt-when";
      mWhen.textContent = (t.started_at_local || "").slice(11) + " · " + rel;
      mDur.className = "mt-elapsed";
      mDur.textContent = dur;
    }
  }

  function ordered() {
    const byNewest = (a, b) => (a.started_at < b.started_at ? 1 : a.started_at > b.started_at ? -1 : 0);
    return [...data.values()].sort(byNewest);
  }

  function visibleHistory() {
    return ordered().filter((t) =>
      t.status !== "running" &&
      (!status || t.status === status) &&
      (!q || rows.get(t.task_id).search.includes(q)));
  }

  // A row under the mouse never moves. The move happens once the mouse leaves.
  // A call with force always moves the rows, even under the mouse.
  function layout(force) {
    const all = ordered();
    const running = all.filter((t) => t.status === "running");
    const hist = visibleHistory();
    if (hover && !force) {
      // Only the wide rows are counted, one per task.
      const cur = [...list.children].filter((n) => n.classList.contains("t-row")).map((n) => n.dataset.tid);
      const target = [...running, ...hist].map((t) => t.task_id);
      if (cur.indexOf(hover) !== target.indexOf(hover)) {
        pendingLayout = true;
        return;
      }
    }
    pendingLayout = false;
    list.replaceChildren();
    const total = all.length;
    nRun.textContent = "· " + running.length;
    nHist.textContent = "· " + hist.length;
    gRun.style.display = running.length === 0 ? "none" : "";
    gHist.style.display = total === 0 ? "none" : "";
    for (const t of running) {
      const r = rows.get(t.task_id);
      list.appendChild(r.row);
      list.appendChild(r.card);
    }
    list.appendChild(gHist);
    for (const t of hist) {
      const r = rows.get(t.task_id);
      list.appendChild(r.row);
      list.appendChild(r.card);
    }
    if (total === 0) {
      const w = document.createElement("div");
      emptyState(w, "rows", "No runs yet", "This tool has no task records in the last 30 days.", "Back to Tools", "/tools");
      list.appendChild(w);
    } else if (total > 0 && running.length === 0 && hist.length === 0) {
      const w = document.createElement("div");
      emptyState(w, "search", "No tasks match", "Clear filters or adjust search.", "Clear", null);
      list.appendChild(w);
      w.querySelector(".btn").addEventListener("click", clearFilters);
    }
    loadWrap.classList.toggle("hidden", !nextCursor);
    loadBtn.textContent = "Showing most recent " + data.size + " — Load older";
    updateStrip();
  }

  function clearFilters() {
    q = "";
    status = "";
    el.querySelector("#tt-q").value = "";
    el.querySelector("#tt-mq").value = "";
    el.querySelector("#tt-status").value = "";
    for (const c of el.querySelectorAll(".chip")) c.classList.toggle("active", c.dataset.v === "");
    layout(true);
  }

  let tasksDesc = "";

  function updateStrip() {
    const t = s.tools.get(tool);
    // A tool that is not on the tool list can still have runs. Its counts and text
    // come from elsewhere, so a list with rows never says "No runs yet".
    const c = t || s.toolStats[tool] || null;
    const desc = (t && t.description) || tasksDesc || "";
    const mk = (cls, n, label, dot) =>
      '<span class="s ' + cls + '"><span class="s-dot ' + dot + '"></span><span class="n">' + n + "</span> " + label + "</span>";
    const zero = !c || c.running + c.completed + c.failed + c.cancelled === 0;
    const strip = (zero && data.size === 0)
      ? '<span class="none">No runs yet</span>'
      : mk("run", c ? c.running : 0, "running", "running") + mk("done", c ? c.completed : 0, "completed", "completed") +
        mk("fail", c ? c.failed : 0, "failed", "failed") + mk("canc", c ? c.cancelled : 0, "cancelled", "cancelled");
    el.querySelector("#tt-strip").innerHTML = strip;
    el.querySelector("#tt-sub").textContent = desc;
    el.querySelector("#tt-m-head").innerHTML =
      '<span class="tt-m-title">' + esc(tool) + "</span>" +
      '<span class="tt-m-sub">' + esc(desc) + "</span>" +
      '<span class="tt-m-strip">' + strip + "</span>";
  }

  function bindSearch(input) {
    let t = null;
    input.addEventListener("input", () => {
      clearTimeout(t);
      t = setTimeout(() => {
        q = input.value.trim().toLowerCase();
        layout(true);
      }, DEBOUNCE_MS);
    });
  }
  bindSearch(el.querySelector("#tt-q"));
  bindSearch(el.querySelector("#tt-mq"));
  el.querySelector("#tt-status").addEventListener("change", (ev) => {
    status = ev.target.value;
    for (const c of el.querySelectorAll(".chip")) c.classList.toggle("active", c.dataset.v === status);
    layout(true);
  });
  for (const c of el.querySelectorAll(".chip")) {
    c.addEventListener("click", () => {
      const v = c.dataset.v;
      status = (status === v && v !== "") ? "" : v;
      for (const cc of el.querySelectorAll(".chip")) cc.classList.toggle("active", cc.dataset.v === status);
      el.querySelector("#tt-status").value = status;
      layout(true);
    });
  }
  loadBtn.addEventListener("click", async () => {
    loadBtn.disabled = true;
    try {
      const body = await getJSON("/api/tools/" + encodeURIComponent(tool) + "/tasks?" +
        new URLSearchParams({ limit: String(_PAGE), cursor: nextCursor }));
      for (const t of body.tasks) {
        data.set(t.task_id, t);
        upsert(t, false);
        cellUpdate(t);
      }
      nextCursor = body.next_cursor;
      layout(true);
      loadBtn.disabled = false;
    } catch (err) {
      toast("Load older failed: " + (err.message || err), { error: true });
      loadBtn.disabled = false;
    }
  });

  const off = on((kind, payload) => {
    if (!ready) return;
    if (kind === "changes") {
      let touched = false;
      for (const t of payload) {
        if (t.tool !== tool) continue;
        touched = true;
        const isNew = !data.has(t.task_id);
        data.set(t.task_id, t);
        upsert(t, isNew);
        cellUpdate(t);
      }
      if (touched) layout();
    }
    updateStrip();
    setLastPolled(el, s.lastPollLocal);
  });
  const stopTick = startTicker(() => {
    if (!ready) return;
    for (const t of data.values()) if (t.status === "running") cellUpdate(t);
    setLastPolled(el, s.lastPollLocal);
  });

  setLastPolled(el, s.lastPollLocal);
  list.appendChild(gHist);
  skeletonRows(list, 4);
  const t0 = Date.now();
  try {
    const body = await getJSON("/api/tools/" + encodeURIComponent(tool) + "/tasks?" +
      new URLSearchParams({ limit: String(_PAGE) }));
    for (const t of body.tasks) {
      data.set(t.task_id, t);
      upsert(t, false);
      cellUpdate(t);
    }
    tasksDesc = body.description || "";
    nextCursor = body.next_cursor;
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) {
      notFound();
    } else {
      toast("Load tasks failed: " + (err.message || err), { error: true });
      renderLoadError(err);
    }
    return { destroy() { off(); stopTick(); document.body.classList.remove("pushed"); } };
  }
  const rest = SKELETON_MS - (Date.now() - t0);
  if (rest > 0) await new Promise((r) => setTimeout(r, rest));
  ready = true;
  layout();

  return {
    destroy() {
      off();
      stopTick();
      document.body.classList.remove("pushed");
    },
  };
}
