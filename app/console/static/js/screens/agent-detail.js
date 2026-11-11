import { icon } from "../ui.js";
import { mountSubtaskList } from "../subtask-list.js";
import { mountLog } from "./log-viewer.js";
import { mountResult } from "./result.js";
import { mountConversation } from "./conversation.js";

const _TAB_IDS = ["results", "conversation", "log", "toolcalls"];

export function mountAgent(mount, detail, saved, stopRestore) {
  // The saved tab reopens when it is present.
  // Without a saved tab the screen opens Results, matching the tool template.
  const wanted = saved && saved.tab;
  let activeTab = wanted && _TAB_IDS.includes(wanted) ? wanted : "results";
  let convHandle = null;
  let resultHandle = null;
  let logHandle = null;
  let subtaskHandle = null;
  let terminalAtMount = false;
  let terminalFired = false;

  function _stopHandle(h) {
    if (h && h.destroy) h.destroy();
  }

  function _contextLine() {
    if (!detail.ancestors.length) return null;
    const parent = detail.ancestors[detail.ancestors.length - 1];
    const line = document.createElement("div");
    line.className = "td-context";
    const name = document.createElement("span");
    name.className = "mono";
    name.textContent = parent.agent_name || parent.tool || parent.task_id.slice(0, 8);
    const id = document.createElement("span");
    id.className = "mono";
    id.textContent = parent.task_id.slice(0, 8);
    line.append("Agent run of ", name, " \u00B7 ", id);
    return line;
  }

  function _tabs() {
    const bar = document.createElement("div");
    const count = detail.subtasks.length;
    const labels = [
      ["results", "Results"],
      ["conversation", "Conversation"],
      ["log", "Log"],
      ["toolcalls", "Tool calls" + (count ? " " + count : "")],
    ];
    bar.className = "td-tabs td-tabs-tight";
    for (const [id, label] of labels) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "td-tab" + (activeTab === id ? " active" : "");
      btn.textContent = label;
      btn.dataset.tab = id;
      btn.addEventListener("click", () => {
        mount.scrollTop = 0;
        saved.scrollTop = 0;
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
      if (stopRestore) stopRestore();
      const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
      mount.scrollTo({ top: 0, behavior: reduced ? "auto" : "smooth" });
    });
    bar.appendChild(toTop);
    return bar;
  }

  function switchTab(id) {
    activeTab = id;
    if (stopRestore) stopRestore();
    saved.tab = id;
    terminalAtMount = detail.task.status !== "running";
    terminalFired = false;
    const well = mount.querySelector(".td-well");
    if (!well) return;
    _stopHandle(convHandle); convHandle = null;
    _stopHandle(resultHandle); resultHandle = null;
    _stopHandle(logHandle); logHandle = null;
    _stopHandle(subtaskHandle); subtaskHandle = null;
    well.replaceChildren();
    const tabs = mount.querySelector(".td-tabs");
    if (tabs) {
      for (const btn of tabs.querySelectorAll(".td-tab")) {
        btn.classList.toggle("active", btn.dataset.tab === id);
      }
    }
    if (id === "conversation") {
      convHandle = mountConversation(well, detail);
    } else if (id === "results") {
      resultHandle = mountResult(well, detail, "No result yet \u2014 the agent is still working.");
    } else if (id === "log") {
      const metaEl = document.createElement("span");
      const clearBtn = document.createElement("button");
      const pauseBtn = document.createElement("button");
      clearBtn.type = "button";
      clearBtn.className = "icon-btn";
      clearBtn.innerHTML = icon("x");
      pauseBtn.type = "button";
      pauseBtn.className = "icon-btn";
      pauseBtn.innerHTML = icon("pause");
      const logBar = document.createElement("div");
      logBar.className = "log-bar";
      logBar.append(metaEl, clearBtn, pauseBtn);
      well.appendChild(logBar);
      const logWell = document.createElement("div");
      well.appendChild(logWell);
      logHandle = mountLog(logWell, detail, metaEl, clearBtn, pauseBtn);
    } else if (id === "toolcalls") {
      subtaskHandle = mountSubtaskList(well, detail.subtasks, {
        taskId: detail.task.task_id,
        onRefetch: () => {},
        onCount: (n) => _callsCountLabel(n),
      });
    }
  }

  // The tab label follows the live subtask count the list reports.
  function _callsCountLabel(n) {
    const callBtn = mount.querySelector('.td-tab[data-tab="toolcalls"]');
    if (callBtn) callBtn.textContent = "Tool calls" + (n ? " " + n : "");
  }

  const ctx = _contextLine();
  if (ctx) mount.appendChild(ctx);
  const tabs = _tabs();
  const sentinel = document.createElement("div");
  sentinel.className = "td-tabs-sentinel";
  mount.appendChild(sentinel);
  mount.appendChild(tabs);
  const stuckObserver = new IntersectionObserver((entries) => {
    tabs.classList.toggle("is-stuck", !entries[0].isIntersecting);
  }, { root: mount });
  stuckObserver.observe(sentinel);
  const well = document.createElement("div");
  well.className = "td-well";
  mount.appendChild(well);
  switchTab(activeTab);

  // Applies the screen's in-place refresh inside the agent subtree.
  // The detail object is the one the screen updated in place.
  // This method never re-fetches.
  function refresh(d) {
    _callsCountLabel(d.subtasks.length);
    const active = activeTab === "conversation" ? convHandle : activeTab === "results" ? resultHandle : activeTab === "log" ? logHandle : subtaskHandle;
    if (d.task.status !== "running" && !terminalAtMount && !terminalFired && active && active.onTerminal) {
      terminalFired = true;
      active.onTerminal();
    }
    if (activeTab === "toolcalls" && subtaskHandle && subtaskHandle.refresh) {
      subtaskHandle.refresh(d.subtasks);
    }
  }

  return {
    refresh,
    destroy() {
      _stopHandle(convHandle);
      _stopHandle(resultHandle);
      _stopHandle(logHandle);
      _stopHandle(subtaskHandle);
      if (ctx) ctx.remove();
      stuckObserver.disconnect();
      sentinel.remove();
      tabs.remove();
      well.remove();
    },
  };
}
