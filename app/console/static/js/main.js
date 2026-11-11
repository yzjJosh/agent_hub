import { initTheme, toggleTheme } from "./theme.js";
import { bootstrap, startPolling, getState, on } from "./state.js";
import { FAIL_BANNER_AT } from "./ui.js";
import { initRouter, addRoute } from "./router.js";
import { renderTools } from "./screens/tools.js";
import { renderToolTasks } from "./screens/tool-tasks.js";
import { renderToolRun } from "./screens/tool-run.js";
import { renderTaskDetail } from "./screens/task-detail.js";
import { renderResources } from "./screens/resources.js";
import { renderPresented } from "./screens/presented.js";
import { renderBrowser } from "./screens/browser.js";

function _sbToggleLabel(collapsed) {
  const b = document.getElementById("sb-toggle");
  if (!b) return;
  const t = collapsed ? "Expand sidebar" : "Collapse sidebar";
  b.title = t;
  b.setAttribute("aria-label", t);
}

export function toggleSidebar() {
  const collapsed = document.documentElement.classList.toggle("sb-collapsed");
  try { localStorage.setItem("agent-console-sidebar", collapsed ? "1" : "0"); } catch { /* saving can fail */ }
  _sbToggleLabel(collapsed);
}

function _initKeyboard() {
  document.addEventListener("keydown", (ev) => {
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    const t = ev.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) return;
    if (ev.key === "t") toggleTheme();
    if (ev.key === "s" && matchMedia("(min-width: 768px)").matches) toggleSidebar();
  });
}

function _sidebarStatus() {
  const s = getState();
  const running = (s.stats && s.stats.running) || 0;
  const total = (s.stats && s.stats.total) || 0;
  const meta = running + " tasks running · " + total + " total (30d)";
  const head = s.healthy ? "Service healthy" : "Service unreachable";
  const dotClass = "dot " + (s.healthy ? "ok" : "bad");
  const sbDot = document.getElementById("sb-dot");
  sbDot.className = dotClass;
  document.getElementById("sb-text").textContent = head;
  document.getElementById("sb-meta").textContent = meta;
  const endpoint = s.davBase ? "v1 · MCP " + s.davBase + "/mcp" : "v1";
  document.getElementById("sb-endpoint").textContent = endpoint;
  document.getElementById("sidebar-status").title = head + " · " + meta + " · " + endpoint;
  const count = document.getElementById("nav-count");
  count.hidden = running === 0;
  count.textContent = String(running);
  document.getElementById("ab-dot").className = dotClass;
  document.getElementById("ab-text").textContent = head;
}

function _setBanner(err) {
  const banner = document.getElementById("banner");
  if (err) {
    document.getElementById("banner-detail").textContent = err.message || "poll failed";
    banner.classList.remove("hidden");
    document.body.classList.add("stale");
  } else {
    banner.classList.add("hidden");
    document.body.classList.remove("stale");
  }
}

async function _boot() {
  _sbToggleLabel(document.documentElement.classList.contains("sb-collapsed"));
  on((kind, payload) => {
    _sidebarStatus();
    if (kind === "fail") {
      if (getState().failStreak >= FAIL_BANNER_AT) _setBanner(payload);
    } else {
      _setBanner(null);
    }
  });
  try {
    await bootstrap();
  } catch {
    getState().healthy = false;
    _sidebarStatus();
  }
  startPolling();
  addRoute({ re: /^\/tools\/?$/, params: [], section: "tools", title: () => "Tools", render: renderTools });
  addRoute({ re: /^\/tools\/([^/]+)\/?$/, params: ["tool"], section: "tools", title: (p) => p.tool, render: renderToolTasks });
  addRoute({ re: /^\/tools\/([^/]+)\/run\/?$/, params: ["tool"], section: "tools", title: (p) => p.tool, render: renderToolRun });
  addRoute({ re: /^\/task\/([^/]+)\/?$/, params: ["task_id"], section: "tools", title: (p) => p.task_id, render: renderTaskDetail });
  addRoute({ re: /^\/resources\/?$/, params: [], section: "resources", title: () => "Resources", render: renderResources });
  addRoute({ re: /^\/presented\/?$/, params: [], section: "presented", title: () => "Presented by Agent", render: renderPresented });
  addRoute({ re: /^\/browser\/?$/, params: [], section: "browser", title: () => "Browser", render: renderBrowser });
  initRouter();
}

initTheme();
document.getElementById("sb-toggle").addEventListener("click", toggleSidebar);
_initKeyboard();
_boot();
