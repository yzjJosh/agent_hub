import { getJSON, ApiError } from "../api.js";
import { esc, countLines } from "../format.js";
import { icon } from "../ui.js";
import { toast } from "../toast.js";

const _INITIAL_LINES = 200;
const _EARLIER_LINES = 500;
const _FOLLOW_MS = 2000;
const _BAND_PX = 24;
const _WINDOW_BYTES = 1048576;
const _MAX_RESUME = 32;

const _RE_ERROR = /\bError\b|\bException\b|\sfailed:|Timeout\s+.*exceeded/i;
const _RE_WARN = /\bWarning\b|(^|\s)Timeout\s|\bretry\b/i;

export function mountLog(well, detail, metaEl, clearBtn, pauseBtn) {
  // The detail page scrolls as one, and follow drives the screen scroller.
  const screen = well.closest(".td-screen");
  let term = null;
  let scroll = null;
  let linesBox = null;
  let topBar = null;
  let cursor = 0;
  let pending = "";
  let tmpLine = null;
  let rendered = 0;
  let total = 0;
  let earliestOffset = 0;
  let tailOffset = 0;
  let allShown = false;
  let followOn = true;
  let inBlock = false;
  let timer = null;
  let viewP = Promise.resolve();
  let ended = false;
  let destroyed = false;

  function setCount() {
    metaEl.textContent = rendered + (rendered === 1 ? " line" : " lines");
    clearBtn.disabled = rendered === 0;
  }

  function showEmpty() {
    if (!linesBox) return;
    linesBox.replaceChildren();
    const d = document.createElement("div");
    d.className = "term-empty";
    d.innerHTML = icon("terminal") + "<span>No log output</span>";
    linesBox.appendChild(d);
  }

  function buildTerm() {
    well.replaceChildren();
    term = document.createElement("div");
    term.className = "term";
    term.setAttribute("role", "log");
    term.setAttribute("aria-live", "polite");
    term.tabIndex = -1;
    scroll = document.createElement("div");
    scroll.className = "term-scroll";
    topBar = document.createElement("div");
    topBar.className = "term-top hidden";
    topBar.addEventListener("click", showEarlier);
    scroll.appendChild(topBar);
    linesBox = document.createElement("div");
    linesBox.className = "term-lines";
    scroll.appendChild(linesBox);
    term.appendChild(scroll);
    well.appendChild(term);
    renderTopBar();
  }

  function renderTopBar() {
    if (allShown && rendered > 0 && tailOffset > 0) {
      topBar.textContent = "Collapse to tail";
      topBar.classList.remove("hidden");
    } else if (total > rendered) {
      topBar.textContent = "Show earlier " + (total - rendered).toLocaleString("en-US") + " lines";
      topBar.classList.remove("hidden");
    } else {
      topBar.classList.add("hidden");
    }
  }

  function classifyLine(line) {
    if (/^={10,}/.test(line)) {
      inBlock = !inBlock;
      return { cls: "block", html: esc(line) };
    }
    const m = line.match(/^\[([^\]]+)\]\s?(.*)$/);
    const msg = m ? m[2] : line;
    let cls = inBlock ? "block" : "";
    if (_RE_ERROR.test(msg)) cls = cls ? cls + " error" : "error";
    else if (_RE_WARN.test(msg)) cls = cls ? cls + " warn" : "warn";
    const html = m
      ? '<span class="tag">[' + esc(m[1]) + "]</span> " + esc(m[2])
      : esc(line);
    return { cls, html };
  }

  function buildLine(line) {
    const d = document.createElement("div");
    if (line === "") {
      d.className = "term-line blank";
    } else {
      const { cls, html } = classifyLine(line);
      d.className = "term-line" + (cls ? " " + cls : "");
      d.innerHTML = html;
    }
    return d;
  }

  function appendLines(text) {
    if (!linesBox) return;
    const emp = linesBox.querySelector(".term-empty");
    if (emp) emp.remove();
    pending += text;
    const parts = pending.split("\n");
    const endsNewline = pending.endsWith("\n");
    const complete = parts.slice(0, -1);
    pending = endsNewline ? "" : parts[parts.length - 1];
    if (complete.length > 0 && tmpLine) { tmpLine.remove(); tmpLine = null; }
    const wasAtBottom = atBottom();
    let n = 0;
    for (const line of complete) {
      if (line !== "") n += 1;
      rendered += 1;
      linesBox.appendChild(buildLine(line));
    }
    if (pending !== "") {
      const d = document.createElement("div");
      d.className = "term-line";
      d.textContent = pending;
      linesBox.appendChild(d);
      tmpLine = d;
    }
    setCount();
    renderTopBar();
    if (n > 0 && followOn && wasAtBottom) scrollToBottom();
  }

  function atBottom() {
    if (!screen) return true;
    return screen.scrollHeight - screen.scrollTop - screen.clientHeight <= _BAND_PX;
  }

  function scrollToBottom() {
    if (screen) screen.scrollTop = screen.scrollHeight;
  }

  function jumpResume() {
    scrollToBottom();
    followOn = true;
    syncPauseBtn();
  }

  function onScroll() {
    if (ended) return;
    if (atBottom()) {
      if (!followOn) jumpResume();
    } else if (followOn) {
      followOn = false;
      syncPauseBtn();
    }
  }

  function syncPauseBtn() {
    const paused = !followOn;
    pauseBtn.classList.toggle("active", paused);
    pauseBtn.innerHTML = paused ? icon("play") : icon("pause");
    pauseBtn.title = paused ? "Resume follow" : "Pause follow";
  }

  pauseBtn.addEventListener("click", () => {
    followOn = !followOn;
    if (followOn) jumpResume();
    syncPauseBtn();
  });
  if (screen) screen.addEventListener("scroll", onScroll);

  clearBtn.addEventListener("click", () => {
    linesBox.replaceChildren();
    tmpLine = null;
    pending = "";
    rendered = 0;
    inBlock = false;
    setCount();
    renderTopBar();
  });

  async function loadEarlier() {
    if (earliestOffset === 0) return; // the top of the log may already be shown
    try {
      const res = await getJSON(`/api/tasks/${detail.id}/log?lines=${_EARLIER_LINES}&end_offset=${earliestOffset}`);
      const frag = document.createDocumentFragment();
      const wasAtBottom = atBottom();
      // These older lines must not change the block state that new lines use.
      const savedInBlock = inBlock;
      for (const line of res.content.replace(/\n$/, "").split("\n")) {
        frag.appendChild(buildLine(line));
        rendered += 1;
      }
      inBlock = savedInBlock;
      if (linesBox.firstChild) linesBox.insertBefore(frag, linesBox.firstChild);
      else linesBox.appendChild(frag);
      earliestOffset = res.offset;
      allShown = res.head_reached;
      total = res.total_lines;
      setCount();
      renderTopBar();
      if (wasAtBottom) scrollToBottom();
    } catch (err) {
      toast("Load earlier lines failed: " + (err.message || err), { error: true });
    }
  }

  function showEarlier() {
    if (allShown) {
      queuePaint();
      return;
    }
    viewP = viewP.then(() => loadEarlier());
  }

  async function tickOnce() {
    if (destroyed) return;
    let res = await getJSON(`/api/tasks/${detail.id}/log?offset=${cursor}&max_bytes=${_WINDOW_BYTES}`);
    if (res.size < cursor) {
      queuePaint(); // the log file was replaced, so paint it all again
      return;
    }
    if (res.content) appendLines(res.content);
    cursor = res.offset + res.content_bytes;
    let hops = 0;
    while (res.truncated && hops < _MAX_RESUME) {
      res = await getJSON(`/api/tasks/${detail.id}/log?offset=${cursor}&max_bytes=${_WINDOW_BYTES}`);
      if (res.content) appendLines(res.content);
      cursor = res.offset + res.content_bytes;
      hops += 1;
    }
  }

  function tick() {
    const run = viewP.then(() => tickOnce()).catch((err) => {
      if (!(err instanceof ApiError) || (err.code !== "log_not_found" && err.status !== 404)) {
        toast("Log follow failed: " + (err.message || err), { error: true });
      }
    });
    viewP = run;
    return run;
  }

  async function forwardTick() {
    if (ended || !linesBox) return;
    await tick();
  }

  function queuePaint() {
    viewP = viewP.then(() => firstPaint());
    return viewP;
  }

  async function firstPaint() {
    buildTerm();
    rendered = 0;
    total = 0;
    pending = "";
    tailOffset = 0;
    inBlock = false;
    allShown = false;
    try {
      let res = await getJSON(`/api/tasks/${detail.id}/log?lines=${_INITIAL_LINES}&end_offset=0`);
      if (res.content_bytes === 0 && res.size > 0) {
        // One very long line can make the read from the end come back empty.
        res = await getJSON(`/api/tasks/${detail.id}/log?offset=0&max_bytes=${_WINDOW_BYTES}`);
        appendLines(res.content);
        cursor = res.offset + res.content_bytes;
        total = countLines(res.content);
        earliestOffset = 0;
        allShown = true;
      } else {
        appendLines(res.content);
        cursor = res.offset + res.content_bytes;
        earliestOffset = res.offset;
        tailOffset = res.offset;
        total = res.total_lines;
        allShown = res.head_reached;
      }
      setCount();
      renderTopBar();
      if (total === 0 && res.size === 0) showEmpty();
      else scrollToBottom();
    } catch (err) {
      if (err instanceof ApiError && (err.code === "log_not_found" || err.status === 404)) showEmpty();
      else toast("Load log failed: " + (err.message || err), { error: true });
    }
    if (destroyed) return;
    if (ended) { applyEndTick(); return; }
    if (detail.task.status !== "running") finish();
    else if (!timer) timer = setInterval(forwardTick, _FOLLOW_MS);
  }

  function applyEndTick() {
    // Only one line may carry the end mark.
    if (linesBox) {
      for (const old of linesBox.querySelectorAll(".term-line.end-tick")) old.classList.remove("end-tick");
    }
    const lines = linesBox ? linesBox.querySelectorAll(".term-line:not(.blank)") : [];
    const last = lines[lines.length - 1];
    if (last) last.classList.add("end-tick");
    setCount();
    renderTopBar();
  }

  function finish() {
    if (ended) return;
    ended = true;
    if (timer) { clearInterval(timer); timer = null; }
    viewP.then(() => tick()).then(() => applyEndTick());
  }

  viewP = firstPaint();

  return {
    onTerminal() {
      finish();
    },
    destroy() {
      ended = true;
      destroyed = true;
      if (timer) { clearInterval(timer); timer = null; }
      if (screen) screen.removeEventListener("scroll", onScroll);
    },
  };
}
