import { getJSON, postJSON, ApiError } from "../api.js";
import { esc, parseUtc, fmtDur, serverNow } from "../format.js";
import { icon } from "../ui.js";
import { toast, copyText } from "../toast.js";
import { getState } from "../state.js";
import { navigate, entryView } from "../router.js";
import { mountCrumb } from "../crumb.js";
import { mountSubtaskList } from "../subtask-list.js";
import { mountResult } from "./result.js";
import { mountLog } from "./log-viewer.js";
import { mountParams, loadToolSchema } from "./params.js";

// How long a remount keeps re-asserting the saved scroll position while the
// tab content still fills in. A user wheel, touch, or key takes over instead.
const _RESTORE_MS = 5000;

// How long the position must stay clamped before the restore gives up.
const _CLAMP_BAIL_MS = 1500;

export async function renderTaskDetail(el, params) {
  el.classList.add("td-screen");
  const taskId = params.task_id;
  // The entry view holds the saved tab and scroll of this visit. A fresh
  // entry has an empty view, so the defaults apply.
  const view = entryView();
  let detail = null;
  let activeTab = view.tab ? view.tab : "results";
  let resultHandle = null;
  let logHandle = null;
  let subtaskHandle = null;
  let crumbHandle = null;
  let agentHandle = null;
  let ticker = null;
  let destroyed = false;
  let paramsStop = null;
  let refetching = false;
  let renderSeq = 0;
  let restoreTop = 0;
  let restoreRaf = null;
  let restoreDeadline = 0;
  let stuckObserver = null;
  let terminalAtMount = false;
  let terminalFired = false;

  function _stopRestore() {
    if (restoreRaf) { cancelAnimationFrame(restoreRaf); restoreRaf = null; }
    restoreTop = 0;
    restoreDeadline = 0;
  }

  // Re-assert the saved scroller position every frame until it holds for a
  // frame, the clamp outlasts the bail window, the deadline passes, or the
  // user takes over.
  function _startRestore(top) {
    _stopRestore();
    if (top <= 0) return;
    restoreTop = top;
    restoreDeadline = performance.now() + _RESTORE_MS;
    let justSet = false;
    let clampStart = 0;
    const step = () => {
      restoreRaf = null;
      if (destroyed || !el.isConnected || !restoreTop) return;
      if (performance.now() > restoreDeadline) { _stopRestore(); return; }
      if (justSet && Math.abs(el.scrollTop - restoreTop) <= 2) {
        restoreTop = 0;
        restoreDeadline = 0;
        return;
      }
      el.scrollTop = restoreTop;
      if (restoreTop - el.scrollTop >= 2) {
        if (!clampStart) clampStart = performance.now();
        else if (performance.now() - clampStart >= _CLAMP_BAIL_MS) {
          // The position stayed clamped for the whole window, so the page is
          // shorter than the saved position.
          restoreTop = 0;
          restoreDeadline = 0;
          return;
        }
      } else {
        clampStart = 0;
      }
      justSet = true;
      restoreRaf = requestAnimationFrame(step);
    };
    restoreRaf = requestAnimationFrame(step);
  }

  function _userTookOver() {
    _stopRestore();
  }

  function _saveScroll() {
    view.scrollTop = el.scrollTop;
  }
  el.addEventListener("scroll", _saveScroll, { passive: true });
  el.addEventListener("wheel", _userTookOver, { passive: true });
  el.addEventListener("touchmove", _userTookOver, { passive: true });
  el.addEventListener("keydown", _userTookOver);

  function _skeleton() {
    el.replaceChildren(
      Object.assign(document.createElement("div"), { className: "well-skeleton" })
    );
  }

  function _notFound(id) {
    _teardown();
    el.replaceChildren(
      Object.assign(document.createElement("div"), {
        className: "feed-state",
        innerHTML:
          '<span class="vp-state-ic">' + icon("alert") + "</span>" +
          '<h1 class="vp-state-title">Task not found</h1>' +
          '<p class="vp-state-sub">No task with id "' + esc(id) + '" (or a unique prefix of it).</p>' +
          '<div class="vp-state-actions"><button class="btn ghost" type="button" data-act="back">Back to Tools</button></div>',
      })
    );
    el.querySelector('[data-act="back"]').addEventListener("click", () => navigate("/tools"));
  }

  function _backTarget(d) {
    if (d.task.parent_task_id) return "/task/" + d.task.parent_task_id;
    if (d.task.tool) return "/tools/" + encodeURIComponent(d.task.tool);
    return "/tools";
  }

  function _rerunLink(d) {
    if (!d.task.rerunnable || !d.task.tool || d.task.status === "running") return null;
    const rerun = document.createElement("a");
    rerun.className = "btn ghost td-rerun-btn";
    rerun.href = "/tools/" + encodeURIComponent(d.task.tool) + "/run?from=" + encodeURIComponent(d.task.task_id);
    rerun.dataset.spa = "";
    rerun.textContent = "Re-run";
    return rerun;
  }

  function _fillActions(actions, d) {
    actions.replaceChildren();
    const rerun = _rerunLink(d);
    if (rerun) actions.appendChild(rerun);
    if (d.task.tool && !d.task.rerunnable) {
      const note = document.createElement("span");
      note.className = "td-inloop-note";
      note.textContent = "Not re-runnable \u2014 this is an internal step.";
      actions.appendChild(note);
    }
    if (!d.task.parent_task_id && d.task.status === "running") {
      const cancel = document.createElement("button");
      cancel.type = "button";
      cancel.className = "btn ghost danger";
      cancel.textContent = "Cancel";
      cancel.addEventListener("click", async () => {
        try {
          await postJSON("/api/tasks/" + encodeURIComponent(d.task.task_id) + "/cancel");
          toast("Cancel requested");
        } catch { toast("Cancel failed", { error: true }); }
      });
      actions.appendChild(cancel);
    }
  }

  function _header(d) {
    const head = document.createElement("div");
    head.className = "td-head";
    const back = document.createElement("button");
    back.type = "button";
    back.className = "td-back";
    back.innerHTML = icon("chevL");
    back.addEventListener("click", () => navigate(_backTarget(d)));
    head.appendChild(back);

    const line = document.createElement("div");
    line.className = "td-title-line";
    const dot = document.createElement("span");
    dot.className = "td-dot td-dot-" + d.task.status;
    if (d.task.status === "running") dot.classList.add("pulse");
    line.appendChild(dot);
    const name = document.createElement("h1");
    name.className = "td-title";
    name.textContent = d.task.tool || d.task.agent_name || d.task.task_id.slice(0, 8);
    if (d.task.agent_name) {
      const row = document.createElement("div");
      row.className = "td-name-row";
      name.title = name.textContent;
      const chip = document.createElement("span");
      chip.className = "st-chip";
      chip.textContent = "AGENT";
      row.append(name, chip);
      line.appendChild(row);
    } else {
      line.appendChild(name);
    }
    const badge = document.createElement("span");
    badge.className = "td-badge td-badge-" + d.task.status;
    badge.id = "d-badge";
    badge.textContent = d.task.status;
    line.appendChild(badge);

    const actions = document.createElement("div");
    actions.className = "td-actions";
    _fillActions(actions, d);
    line.appendChild(actions);
    head.appendChild(line);

    const meta = document.createElement("div");
    meta.className = "td-meta";
    meta.id = "td-meta";
    _metaFill(meta, d);
    head.appendChild(meta);
    return head;
  }

  function _metaCell(label, value) {
    const cell = document.createElement("span");
    cell.className = "tm-cell";
    const k = document.createElement("span");
    k.className = "k";
    k.textContent = label;
    const v = document.createElement("span");
    v.className = "v";
    v.textContent = value;
    cell.append(k, v);
    return cell;
  }

  function _metaFill(meta, d) {
    meta.replaceChildren();
    const idCell = _metaCell("ID", "");
    const idWrap = document.createElement("span");
    idWrap.className = "tm-id-wrap";
    const idText = document.createElement("span");
    idText.className = "tm-id";
    idText.textContent = d.task.task_id;
    idText.title = d.task.task_id;
    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "icon-btn";
    copyBtn.title = "Copy task id";
    copyBtn.innerHTML = icon("copy");
    copyBtn.addEventListener("click", () => copyText(d.task.task_id, "Task id copied"));
    idWrap.append(idText, copyBtn);
    idCell.querySelector(".v").appendChild(idWrap);
    meta.appendChild(idCell);
    meta.appendChild(_metaCell("Started", d.task.started_at_local || ""));
    if (d.task.status === "running") {
      const live = _metaCell("Elapsed", fmtDur(serverNow() - parseUtc(d.task.started_at)));
      const liveVal = live.querySelector(".v");
      liveVal.classList.add("live", "tm-elapsed");
      liveVal.id = "d-elapsed";
      meta.appendChild(live);
    } else {
      meta.appendChild(_metaCell("Ended", d.task.ended_at_local || ""));
      const started = parseUtc(d.task.started_at);
      const ended = parseUtc(d.task.ended_at);
      meta.appendChild(_metaCell("Duration", Number.isFinite(ended) ? fmtDur(ended - started) : ""));
    }
  }

  function _errBanner(d) {
    if (!d.task.error) return null;
    const el = document.createElement("div");
    el.className = "err-banner";
    const msg = document.createElement("div");
    msg.className = "msg";
    msg.textContent = d.task.error;
    el.appendChild(msg);
    return el;
  }

  function _tabs() {
    const bar = document.createElement("div");
    bar.className = "td-tabs";
    const count = detail ? detail.subtasks.length : 0;
    const labels = [
      ["results", "Results"],
      ["log", "Log"],
      ["subtasks", "Subtasks" + (count ? " " + count : "")],
    ];
    for (const [id, label] of labels) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "td-tab" + (activeTab === id ? " active" : "");
      btn.textContent = label;
      btn.dataset.tab = id;
      btn.addEventListener("click", () => {
        el.scrollTop = 0;
        view.scrollTop = 0;
        switchTab(id);
      });
      bar.appendChild(btn);
    }
    const sep = document.createElement("span");
    sep.className = "td-totop-sep";
    bar.appendChild(sep);
    const toTop = document.createElement("button");
    toTop.type = "button";
    toTop.className = "td-totop";
    toTop.title = "Back to top";
    toTop.setAttribute("aria-label", "Back to top");
    toTop.innerHTML = icon("chevUp");
    toTop.addEventListener("click", () => {
      // A click fires no wheel, touch, or key event, and the restore loop only stops on those.
      _stopRestore();
      const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
      el.scrollTo({ top: 0, behavior: reduced ? "auto" : "smooth" });
    });
    bar.appendChild(toTop);
    return bar;
  }

  function _contentWell() {
    const well = document.createElement("div");
    well.className = "td-well";
    return well;
  }

  function _stopHandle(h) {
    if (h && h.destroy) h.destroy();
  }

  // Full teardown, shared by destroy() and the not-found path so a re-fetch
  // that 404s drops every live handle (ticker, crumb, tab mounts, params).
  function _teardown() {
    destroyed = true;
    _stopRestore();
    if (stuckObserver) { stuckObserver.disconnect(); stuckObserver = null; }
    el.removeEventListener("scroll", _saveScroll);
    el.removeEventListener("wheel", _userTookOver);
    el.removeEventListener("touchmove", _userTookOver);
    el.removeEventListener("keydown", _userTookOver);
    if (ticker) { clearInterval(ticker); ticker = null; }
    _stopHandle(resultHandle); resultHandle = null;
    _stopHandle(logHandle); logHandle = null;
    _stopHandle(subtaskHandle); subtaskHandle = null;
    _stopHandle(crumbHandle); crumbHandle = null;
    _stopHandle(agentHandle); agentHandle = null;
    if (paramsStop) { paramsStop(); paramsStop = null; }
  }

  // The tab label follows the live subtask count the list reports.
  function _subCountLabel(n) {
    const btn = el.querySelector('.td-tab[data-tab="subtasks"]');
    if (btn) btn.textContent = "Subtasks" + (n ? " " + n : "");
  }

  function switchTab(id) {
    activeTab = id;
    _stopRestore();
    view.tab = id;
    terminalAtMount = detail ? detail.task.status !== "running" : false;
    terminalFired = false;
    const well = el.querySelector(".td-well");
    if (!well) return;
    _stopHandle(resultHandle); resultHandle = null;
    _stopHandle(logHandle); logHandle = null;
    _stopHandle(subtaskHandle); subtaskHandle = null;
    well.replaceChildren();
    const tabs = el.querySelector(".td-tabs");
    if (tabs) {
      for (const btn of tabs.querySelectorAll(".td-tab")) {
        btn.classList.toggle("active", btn.dataset.tab === id);
      }
    }
    if (id === "results") {
      resultHandle = mountResult(well, detail, "No result yet \u2014 the task is still running.");
    } else if (id === "log") {
      const metaEl = document.createElement("span");
      metaEl.id = "log-count";
      const dlLink = document.createElement("a");
      const clearBtn = document.createElement("button");
      const pauseBtn = document.createElement("button");
      dlLink.className = "icon-btn";
      dlLink.title = "Download log";
      dlLink.target = "_blank";
      dlLink.rel = "noopener";
      dlLink.href = "/api/tasks/" + encodeURIComponent(detail.task.task_id) + "/log/download";
      dlLink.innerHTML = icon("download");
      clearBtn.type = "button";
      clearBtn.className = "icon-btn";
      clearBtn.title = "Clear log view (client-side only)";
      clearBtn.innerHTML = icon("x");
      pauseBtn.type = "button";
      pauseBtn.className = "icon-btn";
      pauseBtn.title = "Pause follow";
      pauseBtn.innerHTML = icon("pause");
      const logBar = document.createElement("div");
      logBar.className = "log-bar";
      logBar.append(metaEl, dlLink, clearBtn, pauseBtn);
      well.appendChild(logBar);
      const logWell = document.createElement("div");
      logWell.id = "log-well";
      well.appendChild(logWell);
      logHandle = mountLog(logWell, detail, metaEl, clearBtn, pauseBtn);
    } else if (id === "subtasks") {
      subtaskHandle = mountSubtaskList(well, detail.subtasks, {
        taskId: detail.task.task_id,
        onRefetch: () => fetchDetail(),
        onCount: (n) => _subCountLabel(n),
      });
    }
  }

  async function fetchDetail() {
    if (destroyed || refetching) return;
    refetching = true;
    let body = null;
    try {
      body = await getJSON("/api/tasks/" + encodeURIComponent(taskId));
    } catch (err) {
      refetching = false;
      if (destroyed) return;
      if (err instanceof ApiError && (err.code === "not_found" || err.status === 404)) {
        _notFound(taskId);
        return;
      }
      toast("Load task failed: " + ((err && err.message) || err), { error: true });
      return;
    }
    refetching = false;
    if (destroyed) return;
    if (detail) {
      _applyDetail(body);
      _refresh();
      return;
    }
    // mountResult and mountLog read detail.id and detail.task, so the payload
    // is augmented with the id they expect.
    detail = { ...body, id: body.task.task_id };
    render();
  }

  // Copies the fresh payload into the object the first fetch created. The
  // object is never reassigned, so the tab mounts' closures observe it.
  function _applyDetail(body) {
    Object.assign(detail, body);
    detail.id = body.task.task_id;
  }

  // A status change refreshes the chrome and settles the active tab mount
  // in place. The screen and the tab well are never rebuilt here, so the
  // user's tab and scroll survive the run finishing.
  function _refresh() {
    if (destroyed || !detail) return;
    const d = detail;
    const dot = el.querySelector(".td-dot");
    if (dot) {
      dot.className = "td-dot td-dot-" + d.task.status;
      if (d.task.status === "running") dot.classList.add("pulse");
    }
    const badge = el.querySelector(".td-badge");
    if (badge) {
      badge.className = "td-badge td-badge-" + d.task.status;
      badge.textContent = d.task.status;
    }
    const meta = el.querySelector("#td-meta");
    if (meta) _metaFill(meta, d);
    const actions = el.querySelector(".td-actions");
    if (actions) _fillActions(actions, d);
    const head = el.querySelector(".td-head");
    const banner = el.querySelector(".err-banner");
    if (d.task.error) {
      if (banner) {
        const msg = banner.querySelector(".msg");
        (msg || banner).textContent = d.task.error;
      } else el.insertBefore(_errBanner(d), head ? head.nextSibling : null);
    } else if (banner) {
      banner.remove();
    }
    const rerun = _rerunLink(d);
    const phone = el.querySelector(".td-rerun");
    if (rerun && !phone) {
      const row = document.createElement("div");
      row.className = "td-rerun";
      row.appendChild(rerun);
      el.appendChild(row);
    }
    if (d.task.agent_name) {
      if (agentHandle && agentHandle.refresh) agentHandle.refresh(d);
      return;
    }
    _subCountLabel(d.subtasks.length);
    const active = activeTab === "results" ? resultHandle : activeTab === "log" ? logHandle : subtaskHandle;
    if (d.task.status !== "running" && !terminalAtMount && !terminalFired && active && active.onTerminal) {
      terminalFired = true;
      active.onTerminal();
    }
    if (activeTab === "subtasks" && subtaskHandle && subtaskHandle.refresh) {
      subtaskHandle.refresh(d.subtasks);
    }
  }

  async function render() {
    // A status change can start a second render while this one is suspended
    // at an await. The stale continuation must not mount into the element.
    const seq = ++renderSeq;
    // Capture the scroller position before the re-render clears it, then
    // restore it after the tab mounts.
    const top = Math.max(el.scrollTop, view.scrollTop || 0);
    _stopHandle(crumbHandle); crumbHandle = null;
    _stopHandle(agentHandle); agentHandle = null;
    if (paramsStop) { paramsStop(); paramsStop = null; }
    el.replaceChildren();
    const d = detail;
    const head = _header(d);
    el.appendChild(head);
    const banner = _errBanner(d);
    if (banner) el.appendChild(banner);
    const phoneRerun = _rerunLink(d);
    if (phoneRerun) {
      const row = document.createElement("div");
      row.className = "td-rerun";
      row.appendChild(phoneRerun);
      el.appendChild(row);
    }

    if (d.ancestors.length) {
      crumbHandle = mountCrumb(head, d.ancestors, d.task);
    } else {
      head.classList.add("no-crumb");
    }
    document.title = (d.task.tool || d.task.agent_name || "Task") + " \u00B7 Agent Hub";
    _startTicker();

    if (d.task.agent_name) {
      const { mountAgent } = await import("./agent-detail.js");
      if (destroyed || !el.isConnected || seq !== renderSeq) return;
      agentHandle = mountAgent(el, detail, view, _stopRestore);
      _startRestore(top);
      return;
    }

    if (d.task.tool) {
      // A tab the tool template does not own falls back to results.
      if (!["results", "log", "subtasks"].includes(activeTab)) activeTab = "results";
      const paramsSection = document.createElement("div");
      paramsSection.className = "td-params";
      const pHead = document.createElement("div");
      pHead.className = "td-params-head";
      const pTitle = document.createElement("span");
      pTitle.className = "td-params-title";
      pTitle.textContent = "Params";
      const pCount = document.createElement("span");
      pCount.className = "p-count";
      pCount.id = "p-count";
      pHead.append(pTitle, pCount);
      const pWell = document.createElement("div");
      pWell.className = "p-well";
      pWell.id = "p-well";
      paramsSection.append(pHead, pWell);
      el.appendChild(paramsSection);
      loadToolSchema(d.task.tool).then((schema) => {
        if (destroyed || !pWell.isConnected || seq !== renderSeq) return;
        paramsStop = mountParams(pWell, pCount, d.task.params, schema);
      });
    }

    const sentinel = document.createElement("div");
    sentinel.className = "td-tabs-sentinel";
    const bar = _tabs();
    el.appendChild(sentinel);
    el.appendChild(bar);
    if (stuckObserver) stuckObserver.disconnect();
    stuckObserver = new IntersectionObserver((entries) => {
      bar.classList.toggle("is-stuck", !entries[0].isIntersecting);
    }, { root: el });
    stuckObserver.observe(sentinel);
    const well = _contentWell();
    el.appendChild(well);
    switchTab(activeTab);
    _startRestore(top);
  }

  function _startTicker() {
    if (ticker) clearInterval(ticker);
    ticker = setInterval(() => {
      if (!detail) return;
      // The changes feed refreshes state.tasks every 2s, so a status change
      // surfaces within a few ticks. The refetch refreshes the screen in
      // place, and the ticker then self-clears on a terminal status.
      const live = getState().tasks.get(detail.task.task_id);
      if (live && live.status !== detail.task.status) { fetchDetail(); return; }
      if (detail.task.status !== "running") { clearInterval(ticker); ticker = null; return; }
      const elapsed = el.querySelector(".tm-elapsed");
      if (elapsed) elapsed.textContent = fmtDur(serverNow() - parseUtc(detail.task.started_at));
    }, 1000);
  }

  _skeleton();
  fetchDetail();

  return {
    destroy() {
      _teardown();
      el.replaceChildren();
    },
  };
}
