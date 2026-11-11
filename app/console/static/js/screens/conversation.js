import { on, getState } from "../state.js";
import { parseUtc, fmtClock, fmtDur, serverNow } from "../format.js";
import { icon } from "../ui.js";
import { postRaw, getJSON } from "../api.js";
import { attachFoldable } from "./conv-fold.js";

const _FOLLOW_BAND = 24;
const _TICK_MS = 1000;
const _POLL_MS = 1500;
const _LIVENESS_MS = 10000;
const _WINDOW_BYTES = 524288;
const _DRAIN_CAP = 100;

export function mountConversation(mount, detail) {
  const taskId = detail.task.task_id;
  let running = detail.task.status === "running";
  let destroyed = false;
  let firstPaint = true;
  let nEvents = 0;
  let curAttempt = 0;
  let userSeen = 0;
  let pendingUser = null;
  let activeThink = null;
  let activeContent = null;
  let settledOpen = null;
  const foldCards = new Set();
  const foldOf = new Map();
  const cardByBox = new Map();
  const dirtyContent = new Set();
  const renderRevisions = new WeakMap();
  let tickTimer = null;
  let pinned = true;
  const terminalAtLoad = detail.task.status !== "running";

  const byCallId = new Map();
  const openCalls = new Set();
  const liveCalls = new Set();
  const callSeq = new Map();
  const byChild = new Map();
  const subtasks = new Map();
  for (const s of detail.subtasks || []) subtasks.set(s.task_id, s);

  const conv = document.createElement("div");
  conv.className = "conv";
  const head = document.createElement("div");
  head.className = "conv-head";
  const live = document.createElement("span");
  live.className = "conv-live" + (running ? " show" : "");
  live.textContent = "Live";
  const count = document.createElement("span");
  count.className = "conv-count";
  count.textContent = "0 events";
  head.append(live, count);
  const well = document.createElement("div");
  well.className = "conv-well";
  const empty = document.createElement("div");
  empty.className = "conv-empty";
  empty.textContent = "No conversation yet";
  well.appendChild(empty);
  const follow = document.createElement("button");
  follow.type = "button";
  follow.className = "conv-follow hidden";
  follow.textContent = "Follow";
  conv.append(head, well, follow);
  mount.appendChild(conv);

  // ---- follow ----
  // The detail page scrolls as one, and follow drives its scroller.
  const screen = mount.closest(".td-screen");

  function atBottom() {
    if (!screen) return true;
    return screen.scrollHeight - screen.scrollTop - screen.clientHeight <= _FOLLOW_BAND;
  }

  function scrollToBottom() {
    if (screen) screen.scrollTop = screen.scrollHeight;
  }

  // The button shows only while the run is running and the screen is scrolled up.
  function syncFollow() {
    follow.classList.toggle("hidden", !running || pinned);
  }

  function setPinned(p) {
    if (p === pinned) return;
    pinned = p;
    syncFollow();
    if (p) scrollToBottom();
  }

  function onScreenScroll() {
    setPinned(atBottom());
  }
  if (screen) screen.addEventListener("scroll", onScreenScroll);
  follow.addEventListener("click", () => setPinned(true));
  const layoutObserver = new ResizeObserver(() => {
    if (!destroyed && pinned && !terminalAtLoad) scrollToBottom();
  });
  layoutObserver.observe(well);

  // ---- blocks ----

  function _fold(card, body, labelText) {
    const fold = attachFoldable(card, body, labelText);
    foldCards.add(fold);
    foldOf.set(body, fold);
    return fold;
  }

  // A finalized message text goes through the same render pipeline as the
  // Results tab. The plain text stays until the fragment arrives, then the
  // body is swapped to the rendered document.
  async function _renderBody(body, text) {
    const revision = (renderRevisions.get(body) || 0) + 1;
    renderRevisions.set(body, revision);
    if (!text) return;
    try {
      const res = await postRaw("/api/render/message", new TextEncoder().encode(text));
      if (destroyed || !body.isConnected || renderRevisions.get(body) !== revision) return;
      if (res && res.fragment) {
        const tmp = document.createElement("div");
        tmp.innerHTML = res.fragment;
        body.replaceChildren();
        while (tmp.firstChild) body.appendChild(tmp.firstChild);
        const fold = foldOf.get(body);
        if (fold) fold.onSwapped();
      }
    } catch {
      // The plain text stays as the fallback.
    }
  }

  function userCard(text, system) {
    const card = document.createElement("div");
    card.className = "conv-user" + (system ? " system" : "");
    const body = document.createElement("div");
    body.className = "conv-msg";
    const msg = (text || "").trim();
    body.textContent = msg;
    card.appendChild(body);
    const fold = _fold(card, body, system ? "System" : "User");
    _renderBody(body, msg);
    fold.settle({ animate: running });
    return card;
  }

  function divider(attempt) {
    const div = document.createElement("div");
    div.className = "conv-div";
    div.dataset.attempt = String(attempt);
    div.innerHTML = '<span class="conv-div-rule"></span>' +
      '<span class="conv-div-lbl">Attempt ' + attempt + "</span>" +
      '<span class="conv-div-fail">previous attempt failed</span>' +
      '<span class="conv-div-rule"></span>';
    return div;
  }

  function setOpen(tb, open) {
    tb.open = open;
    tb.box.classList.toggle("open", open);
  }

  function makeThink(attempt) {
    const box = document.createElement("div");
    box.className = "conv-think";
    const line = document.createElement("div");
    line.className = "conv-think-line";
    const chev = document.createElement("span");
    chev.className = "conv-think-chev";
    chev.innerHTML = icon("caret");
    const lbl = document.createElement("span");
    lbl.className = "conv-think-lbl";
    lbl.textContent = "Thinking";
    const prev = document.createElement("span");
    prev.className = "conv-think-prev";
    line.append(chev, lbl, prev);
    const full = document.createElement("div");
    full.className = "conv-think-full";
    box.append(line, full);
    const tb = { box, prev, full, text: "", attempt, toggled: false, open: false };
    line.addEventListener("click", () => {
      tb.toggled = true;
      setOpen(tb, !tb.open);
    });
    return tb;
  }

  function updateThink(tb) {
    const first = tb.text.split("\n", 1)[0];
    tb.prev.textContent = first ? "— " + first : "";
    tb.prev.title = first;
    tb.full.textContent = tb.text;
  }

  function finalizeThink() {
    if (activeThink && !activeThink.toggled) setOpen(activeThink, false);
    activeThink = null;
  }

  function thinkDelta(ev) {
    if (!activeThink || activeThink.attempt !== ev.attempt) {
      finalizeThink();
      activeThink = makeThink(ev.attempt);
      well.appendChild(activeThink.box);
    }
    activeThink.text += ev.text || "";
    updateThink(activeThink);
    // The in-flight block follows the stream; the user's own toggle sticks.
    if (!activeThink.toggled) setOpen(activeThink, running);
  }

  function makeContent(attempt) {
    const box = document.createElement("div");
    box.className = "conv-assist";
    box.dataset.attempt = String(attempt);
    const body = document.createElement("div");
    body.className = "conv-assist-body";
    const pre = document.createElement("pre");
    pre.className = "conv-pre";
    body.appendChild(pre);
    box.appendChild(body);
    const fold = _fold(box, body, "Assistant");
    const rec = { box, body, pre, text: "", attempt, fold };
    cardByBox.set(box, rec);
    return rec;
  }

  function sealContent() {
    if (activeContent) {
      renderAssistant(activeContent);
      activeContent.fold.settle({ animate: running });
    }
    activeContent = null;
  }

  // Post-settle content resolves its card from the well. The last
  // .conv-assist whose attempt matches is the card. Cards are only ever
  // added to the well until the tab is destroyed.
  function _cardForAttempt(attempt) {
    let rec = null;
    for (const el of well.querySelectorAll(".conv-assist")) {
      if (el.dataset.attempt === String(attempt)) rec = cardByBox.get(el);
    }
    return rec;
  }

  function contentDelta(ev) {
    if (activeContent && activeContent.attempt === ev.attempt) {
      activeContent.text += ev.text || "";
      _showContent(activeContent);
      return;
    }
    if (settledOpen === ev.attempt) {
      const rec = _cardForAttempt(ev.attempt);
      if (rec) {
        rec.text += ev.text || "";
        _showContent(rec);
        return;
      }
    }
    activeContent = makeContent(ev.attempt);
    well.appendChild(activeContent.box);
    activeContent.text += ev.text || "";
    _showContent(activeContent);
  }

  function _showContent(block) {
    renderRevisions.set(block.body, (renderRevisions.get(block.body) || 0) + 1);
    block.pre.textContent = block.text.trim();
    if (block.body.firstChild !== block.pre) block.body.replaceChildren(block.pre);
    block.fold.onSwapped();
    dirtyContent.add(block);
  }

  function assistantFinal(ev) {
    const hit = activeContent && activeContent.attempt === ev.attempt ? activeContent : _cardForAttempt(ev.attempt);
    activeContent = null;
    const block = hit || makeContent(ev.attempt);
    if (!hit) well.appendChild(block.box);
    block.text = (ev.text || "").trim();
    _showContent(block);
    if (settledOpen === ev.attempt) settledOpen = null;
    renderAssistant(block);
    block.fold.settle({ animate: running });
  }

  function renderAssistant(block) {
    dirtyContent.delete(block);
    _renderBody(block.body, (block.text || "").trim());
  }

  // ---- tool-call cards ----

  function nextSeq(attempt, name) {
    const key = attempt + ":" + name;
    const n = callSeq.get(key) || 0;
    callSeq.set(key, n + 1);
    return n;
  }

  function makeToolCard(name) {
    const card = document.createElement("a");
    card.className = "conv-tcall";
    const dot = document.createElement("span");
    dot.className = "s-dot running conv-pulse";
    const nm = document.createElement("span");
    nm.className = "conv-tname";
    nm.textContent = name;
    nm.title = name;
    const spacer = document.createElement("span");
    spacer.className = "spacer";
    const when = document.createElement("span");
    when.className = "conv-twhen";
    const dur = document.createElement("span");
    dur.className = "conv-tdur live";
    dur.textContent = "0s";
    card.append(dot, nm, spacer, when, dur);
    const err = document.createElement("div");
    err.className = "conv-terr hidden";
    return { card, dot, when, dur, err };
  }

  function toolCall(ev) {
    const parts = makeToolCard(ev.tool_name);
    const chip = document.createElement("div");
    chip.className = "conv-tchip";
    chip.appendChild(parts.card);
    const wrap = document.createElement("div");
    wrap.className = "conv-twrap";
    wrap.append(chip, parts.err);
    well.appendChild(wrap);
    const e = Object.assign(parts, {
      wrap,
      name: ev.tool_name,
      callId: ev.call_id || null,
      attempt: ev.attempt,
      startMs: parseUtc(ev.ts),
      endMs: NaN,
      taskId: null,
      seq: nextSeq(ev.attempt, ev.tool_name),
      paired: false,
      endedStatus: null,
    });
    if (Number.isFinite(e.startMs)) e.when.textContent = fmtClock(e.startMs);
    if (e.callId) byCallId.set(e.callId, e);
    openCalls.add(e);
    for (const s of subtasks.values()) _matchChild(s);
    syncCard(e);
    if (!running) {
      // A run that is already finished never receives the result.
      e.endMs = parseUtc(detail.task.ended_at);
      settleOpen(detail.task.status);
    }
  }

  function pickOpen(name, attempt) {
    let best = null;
    for (const e of openCalls) {
      if (e.name !== name || e.attempt !== attempt) continue;
      if (!best || e.seq < best.seq) best = e;
    }
    return best;
  }

  function _unlink(e) {
    const old = byChild.get(e.taskId);
    if (!old) return;
    old.delete(e);
    if (old.size === 0) byChild.delete(e.taskId);
  }

  function _link(e, tid) {
    e.taskId = tid;
    const list = byChild.get(tid) || new Set();
    list.add(e);
    byChild.set(tid, list);
  }

  function toolResult(ev) {
    let e = ev.call_id ? byCallId.get(ev.call_id) : null;
    if (!e) e = pickOpen(ev.tool_name, ev.attempt);
    if (!e) return;
    if (ev.call_id) byCallId.delete(ev.call_id);
    openCalls.delete(e);
    e.paired = true;
    e.endMs = parseUtc(ev.ts);
    // A pre-linked child the result does not name goes to the next open call of the same tool.
    const stale = e.taskId && e.taskId !== ev.task_id ? e.taskId : null;
    if (stale) _unlink(e);
    if (ev.task_id) _link(e, ev.task_id);
    else if (stale) e.taskId = null;
    if (stale) _matchChild({ task_id: stale, parent_task_id: taskId, tool: e.name });
    syncCard(e);
  }

  // The changes feed only knows recent tasks, so the detail's subtask list is
  // the fallback for child status and error.
  function childState(taskId) {
    return getState().tasks.get(taskId) || subtasks.get(taskId) || null;
  }

  // The child appears in the changes feed before its tool_result,
  // so a still-running call can link to its detail page. The result event is authoritative.
  function _matchChild(t) {
    if (!t || t.parent_task_id !== taskId || !t.tool || byChild.has(t.task_id)) return;
    let best = null;
    for (const e of openCalls) {
      if (e.name !== t.tool || e.taskId) continue;
      if (!best || e.seq < best.seq) best = e;
    }
    if (!best) return;
    _link(best, t.task_id);
    syncCard(best);
  }

  function syncCard(e) {
    const child = e.taskId ? childState(e.taskId) : null;
    const status = child ? child.status : (e.paired ? "completed" : (e.endedStatus || "running"));
    e.dot.className = "s-dot " + status + (status === "running" ? " conv-pulse" : "");
    if (e.taskId) {
      e.card.setAttribute("data-spa", "");
      e.card.href = "/task/" + e.taskId;
    } else {
      e.card.removeAttribute("data-spa");
      e.card.removeAttribute("href");
    }
    if (status === "failed" && child && child.error) {
      e.err.textContent = child.error;
      e.err.classList.remove("hidden");
    } else {
      e.err.classList.add("hidden");
    }
    if (status === "running") liveCalls.add(e);
    else liveCalls.delete(e);
    if (status === "running") {
      e.dur.classList.add("live");
      e.dur.textContent = fmtDur(serverNow() - e.startMs);
    } else if (Number.isFinite(e.endMs) && Number.isFinite(e.startMs)) {
      e.dur.classList.remove("live");
      e.dur.textContent = fmtDur(e.endMs - e.startMs);
    }
    ensureTicker();
  }

  function ensureTicker() {
    if (!tickTimer && liveCalls.size > 0) tickTimer = setInterval(tickDurs, _TICK_MS);
  }

  function stopTicker() {
    if (tickTimer) {
      clearInterval(tickTimer);
      tickTimer = null;
    }
  }

  function tickDurs() {
    if (liveCalls.size === 0) {
      stopTicker();
      return;
    }
    const now = serverNow();
    for (const e of liveCalls) e.dur.textContent = fmtDur(now - e.startMs);
  }

  // A run that ends leaves its still-open calls settled on the run's status.
  // The call-id map stays, so a result that still arrives pairs as usual.
  function settleOpen(status) {
    for (const e of openCalls) {
      e.endedStatus = status;
      liveCalls.delete(e);
      e.dot.className = "s-dot " + status;
      e.dur.classList.remove("live");
      if (Number.isFinite(e.endMs) && Number.isFinite(e.startMs)) e.dur.textContent = fmtDur(e.endMs - e.startMs);
      else if (Number.isFinite(e.startMs)) e.dur.textContent = fmtDur(serverNow() - e.startMs);
    }
  }

  // A retry-feedback line stashed by onUser sits right after the divider of
  // the attempt it belongs to.
  function _flushPendingUser() {
    if (!pendingUser) return;
    const card = userCard(pendingUser.text, false);
    const divs = well.querySelectorAll(".conv-div");
    let anchor = null;
    for (const d of divs) {
      if (d.dataset.attempt === String(pendingUser.attempt)) anchor = d;
    }
    if (!anchor && divs.length) anchor = divs[divs.length - 1];
    if (anchor) anchor.after(card);
    else well.appendChild(card);
    pendingUser = null;
  }

  // ---- changes feed ----

  const _onChanges = (kind, payload) => {
    if (kind !== "changes") return;
    for (const t of payload) {
      if (t.task_id === taskId) {
        if (running && t.status !== "running") {
          _terminalSettle(t.status);
        }
      }
      if (t.parent_task_id === taskId) _matchChild(t);
      const list = byChild.get(t.task_id);
      if (list) for (const e of list) syncCard(e);
    }
  };
  const unsub = on(_onChanges);

  // ---- delivery ----
  //
  // Live runs stream from offset 0 and follow. A stalled stream or an early
  // 404 drops to 1.5 s windowed polling. Terminal runs drain the file in
  // 512 KiB windows.

  let streamController = null;
  let pollTimer = null;
  let livenessTimer = null;
  let receivedOffset = 0;
  let deliveryDone = false;
  let tail = "";

  const streamUrl = "/api/tasks/" + encodeURIComponent(taskId) + "/conversation/stream";
  const windowUrl = "/api/tasks/" + encodeURIComponent(taskId) + "/conversation";

  function cancelStream() {
    if (streamController) {
      try {
        streamController.abort();
      } catch {
        // The abort raced with the stream closing.
      }
      streamController = null;
    }
  }

  function stopDeliveryTimers() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
    if (livenessTimer) {
      clearInterval(livenessTimer);
      livenessTimer = null;
    }
  }

  // A windowed read can cut a line mid JSON. Complete lines ingest now, the
  // partial last line carries to the next read.
  function feed(content, atEnd) {
    if (!content && !atEnd) return;
    tail += content || "";
    const nl = tail.lastIndexOf("\n");
    if (!atEnd) {
      if (nl === -1) return;
      ingest(tail.slice(0, nl + 1));
      tail = tail.slice(nl + 1);
      return;
    }
    if (tail) ingest(tail);
    tail = "";
  }

  async function drainForward(fromOffset) {
    if (deliveryDone) return;
    stopDeliveryTimers();
    deliveryDone = true;
    // A finished run paints its whole history without following the tail.
    if (terminalAtLoad) pinned = false;
    let offset = fromOffset;
    for (let i = 0; i < _DRAIN_CAP && !destroyed; i++) {
      try {
        const res = await getJSON(windowUrl + "?offset=" + offset + "&max_bytes=" + _WINDOW_BYTES);
        feed(res.content, !res.truncated);
        offset = res.offset + res.content_bytes;
        if (!res.truncated) break;
      } catch {
        break;
      }
    }
    // A retry line stashed for a divider that never arrives flushes here.
    if (!destroyed) _flushPendingUser();
    if (!destroyed && !running) {
      for (const block of dirtyContent) {
        renderAssistant(block);
        block.fold.settle({ animate: false });
      }
    }
  }

  function completeDelivery() {
    if (deliveryDone) return;
    cancelStream();
    drainForward(receivedOffset);
  }

  // One shared terminal settle, guarded on running so first observation wins
  // and the changes feed and the screen's onTerminal can fire in either
  // order. The open content card renders and settles in place before the
  // drain delivers the undelivered tail, and its attempt becomes the
  // settled-open phase so late deltas resume into it.
  function _terminalSettle(status) {
    if (!running) return;
    running = false;
    live.classList.remove("show");
    syncFollow();
    finalizeThink();
    settleOpen(status);
    _flushPendingUser();
    if (activeContent) {
      settledOpen = activeContent.attempt;
      renderAssistant(activeContent);
      activeContent.fold.settle({ animate: true });
      activeContent = null;
    }
    completeDelivery();
  }

  function startPolling() {
    if (destroyed || deliveryDone || pollTimer) return;
    cancelStream();
    const tick = async () => {
      if (destroyed || deliveryDone) return;
      try {
        const res = await getJSON(windowUrl + "?offset=" + receivedOffset + "&max_bytes=" + _WINDOW_BYTES);
        if (destroyed || deliveryDone) return;
        feed(res.content, !res.truncated);
        // The offset comes back adjusted when the start lands mid UTF-8 char.
        receivedOffset = res.offset + res.content_bytes;
        if (!res.truncated && !running) completeDelivery();
      } catch {
        // The file may not exist yet, or the network hiccuped.
      }
    };
    pollTimer = setInterval(tick, _POLL_MS);
    tick();
  }

  function startLiveness() {
    if (livenessTimer) return;
    livenessTimer = setInterval(async () => {
      if (destroyed || deliveryDone || pollTimer) return;
      try {
        const res = await getJSON(windowUrl + "?offset=" + receivedOffset + "&max_bytes=1");
        if (res.size > receivedOffset) startPolling();
      } catch {
        // The file may not exist yet.
      }
    }, _LIVENESS_MS);
  }

  async function startStream() {
    if (destroyed || deliveryDone) return;
    streamController = new AbortController();
    try {
      const res = await fetch(streamUrl, { signal: streamController.signal });
      if (!res.ok || !res.body) {
        startPolling();
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      const enc = new TextEncoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let i = buf.indexOf("\n");
        while (i !== -1) {
          const line = buf.slice(0, i);
          buf = buf.slice(i + 1);
          if (line) {
            receivedOffset += enc.encode(line + "\n").length;
            ingest(line);
          }
          i = buf.indexOf("\n");
        }
      }
      const rem = buf + decoder.decode();
      if (rem) {
        receivedOffset += enc.encode(rem).length;
        ingest(rem);
      }
      if (destroyed || deliveryDone) return;
      completeDelivery();
    } catch {
      if (destroyed || deliveryDone) return;
      startPolling();
    } finally {
      streamController = null;
    }
  }

  if (running) {
    startStream();
    startLiveness();
  } else {
    drainForward(0);
  }

  // ---- ingest ----

  function onUser(ev) {
    userSeen += 1;
    if (userSeen === 1) {
      well.appendChild(userCard(ev.text, false));
    } else {
      // The retry feedback waits for its attempt divider.
      if (pendingUser) well.appendChild(userCard(pendingUser.text, false));
      pendingUser = ev;
    }
  }

  function applyEvent(ev) {
    if (destroyed) return;
    if (firstPaint) {
      firstPaint = false;
      empty.remove();
    }
    nEvents += 1;
    count.textContent = nEvents + (nEvents === 1 ? " event" : " events");
    const sameThink = ev.type === "thinking" && activeThink && activeThink.attempt === ev.attempt;
    if (!sameThink) {
      if (ev.attempt > curAttempt) {
        if (curAttempt > 0) {
          well.appendChild(divider(ev.attempt));
          if (pendingUser) {
            well.appendChild(userCard(pendingUser.text, false));
            pendingUser = null;
          }
        }
        curAttempt = ev.attempt;
      }
      if (ev.type !== "thinking") finalizeThink();
      if (ev.type !== "content" && ev.type !== "assistant") {
        sealContent();
        settledOpen = null;
      }
    }
    if (ev.type === "system") well.appendChild(userCard(ev.text, true));
    else if (ev.type === "user") onUser(ev);
    else if (ev.type === "thinking") thinkDelta(ev);
    else if (ev.type === "content") contentDelta(ev);
    else if (ev.type === "assistant") assistantFinal(ev);
    else if (ev.type === "tool_call") toolCall(ev);
    else if (ev.type === "tool_result") toolResult(ev);
    if (pinned) scrollToBottom();
  }

  function ingest(text) {
    if (destroyed || !text) return;
    for (const line of text.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      let ev;
      try {
        ev = JSON.parse(trimmed);
      } catch {
        continue;
      }
      applyEvent(ev);
    }
  }

  return {
    ingest,
    onTerminal() {
      if (detail.task.status === "running") return;
      _terminalSettle(detail.task.status);
    },
    destroy() {
      if (destroyed) return;
      destroyed = true;
      for (const fold of foldCards) fold.destroy();
      layoutObserver.disconnect();
      cancelStream();
      stopDeliveryTimers();
      stopTicker();
      if (screen) screen.removeEventListener("scroll", onScreenScroll);
      unsub();
      conv.remove();
    },
  };
}
