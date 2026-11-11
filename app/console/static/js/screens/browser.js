import { getJSON } from "../api.js";
import { icon } from "../ui.js";

const _PROBE_MS = 2000;
const _SRC = "/vnc/vnc.html?autoconnect=true&resize=scale&path=websockify";
const _RESOLUTION = "1920\u00d71080";
const _HIDE_CSS =
  "#noVNC_status{display:none!important}" +
  ".noVNC_logo{display:none!important}" +
  "#noVNC_connect_dlg{display:none!important}" +
  "html,body{margin:0!important;height:100%!important;overflow:hidden!important}" +
  "#noVNC_container{height:100%!important}";

export function renderBrowser(el, _params) {
  el.innerHTML =
    '<div class="br-toolbar">' +
      '<span class="br-status"><span class="dot" id="br-dot"></span><span id="br-label">Connecting…</span></span>' +
      '<span class="br-res">' + _RESOLUTION + "</span>" +
      '<span class="spacer"></span>' +
      '<button class="icon-btn" id="br-full" title="Fullscreen">' + icon("maximize") + "</button>" +
      '<button class="btn ghost danger hidden" id="br-disconnect">Disconnect</button>' +
      '<button class="btn solid hidden" id="br-connect">Connect</button>' +
    "</div>" +
    '<div class="br-canvas" id="br-canvas">' +
      '<div class="br-connecting" id="br-connecting"><div class="br-spin"></div><span>Connecting to display…</span></div>' +
      '<div class="br-gone hidden" id="br-gone">' + icon("monitorOff") +
        '<span class="t1">Display disconnected</span>' +
        '<span class="t2">The VNC stream was closed or is unreachable.</span>' +
        '<button class="btn solid" id="br-gone-connect">Connect</button>' +
      "</div>" +
      '<iframe class="br-frame hidden" id="br-frame" title="Remote desktop (noVNC)" allow="fullscreen"></iframe>' +
    "</div>";

  const frame = el.querySelector("#br-frame");
  const connecting = el.querySelector("#br-connecting");
  const gone = el.querySelector("#br-gone");
  const dot = el.querySelector("#br-dot");
  const label = el.querySelector("#br-label");
  const connectBtn = el.querySelector("#br-connect");
  const disconnectBtn = el.querySelector("#br-disconnect");
  const canvas = el.querySelector("#br-canvas");
  let mode = "connecting";
  let downStreak = 0;
  let seenVisual = false;
  let connectSeq = 0;
  let timer = null;
  let destroyed = false;

  function setStatus(next) {
    mode = next;
    dot.className = "dot " + (next === "connected" ? "ok" : next === "connecting" ? "amber" : "bad");
    label.textContent = next === "connected" ? "Connected" : next === "connecting" ? "Connecting…" : "Disconnected";
    connecting.classList.toggle("hidden", next !== "connecting");
    connectBtn.classList.toggle("hidden", next !== "disconnected");
    disconnectBtn.classList.toggle("hidden", next !== "connected" && next !== "connecting");
    gone.classList.toggle("hidden", next !== "disconnected");
  }

  function injectChromeHider() {
    try {
      const doc = frame.contentDocument;
      if (!doc || doc.querySelector("style[data-hide]")) return;
      const st = doc.createElement("style");
      st.setAttribute("data-hide", "1");
      st.textContent = _HIDE_CSS;
      doc.head.appendChild(st);
    } catch { /* the frame may block us from reading it */ }
  }

  function unmountFrame() {
    frame.classList.add("hidden");
    frame.removeAttribute("src");
    frame.src = "about:blank";
  }

  async function connect() {
    const seq = ++connectSeq;
    downStreak = 0;
    seenVisual = false;
    setStatus("connecting");
    let reachable = false;
    try {
      reachable = (await getJSON("/api/vnc/status")).reachable;
    } catch { reachable = false; }
    if (destroyed || seq !== connectSeq) return;
    if (!reachable) {
      unmountFrame();
      setStatus("disconnected");
      return;
    }
    frame.classList.remove("hidden");
    if (frame.getAttribute("src") !== _SRC) frame.src = _SRC;
  }

  function disconnect() {
    connectSeq += 1;
    unmountFrame();
    setStatus("disconnected");
  }

  connectBtn.addEventListener("click", connect);
  el.querySelector("#br-gone-connect").addEventListener("click", connect);
  disconnectBtn.addEventListener("click", disconnect);
  frame.addEventListener("load", injectChromeHider);
  el.querySelector("#br-full").addEventListener("click", () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else if (canvas.requestFullscreen) canvas.requestFullscreen();
  });

  async function probe() {
    if (destroyed || mode === "disconnected") return;
    if (frame.getAttribute("src") !== _SRC) return;
    let cls = "down";
    try {
      const html = frame.contentDocument && frame.contentDocument.documentElement;
      if (html) {
        if (html.classList.contains("noVNC_connected")) cls = "connected";
        else if (html.classList.contains("noVNC_connecting") || html.classList.contains("noVNC_reconnecting")) cls = "connecting";
      }
    } catch { cls = "down"; }
    if (cls === "connected") {
      downStreak = 0;
      if (mode !== "connected") setStatus("connected");
    } else if (cls === "connecting") {
      downStreak = 0;
      if (mode !== "connecting") setStatus("connecting");
    } else if (seenVisual) {
      downStreak += 1;
      if (downStreak >= 2) disconnect();
    }
    if (cls !== "down") seenVisual = true;
  }

  timer = setInterval(probe, _PROBE_MS);
  probe();
  connect();

  return {
    destroy() {
      destroyed = true;
      connectSeq += 1;
      clearInterval(timer);
      disconnect();
    },
  };
}
