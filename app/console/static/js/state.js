import { fetchChanges, getJSON } from "./api.js";
import { noteServerTime } from "./format.js";

const _POLL_MS = 2000;

const state = {
  tools: new Map(),
  tasks: new Map(),
  running: [],
  stats: null,
  toolStats: {},
  fingerprint: null,
  generatedAt: null,
  davBase: "",
  lastPollLocal: null,
  healthy: true,
  failStreak: 0,
  _listeners: new Set(),
};

let _timer = null;
let _inFlight = false;

export function getState() {
  return state;
}

export function on(fn) {
  state._listeners.add(fn);
  return () => state._listeners.delete(fn);
}

function _emit(kind, payload) {
  for (const fn of state._listeners) fn(kind, payload);
}

function _sumStats() {
  let total = 0;
  let running = 0;
  for (const t of state.tools.values()) {
    total += t.running + t.completed + t.failed + t.cancelled;
    running += t.running;
  }
  return { total, running };
}

export async function bootstrap() {
  const [tools, health] = await Promise.all([getJSON("/api/tools"), getJSON("/api/health")]);
  noteServerTime(tools.generated_at, tools.generated_at_local);
  for (const t of tools.tools) state.tools.set(t.name, t);
  state.stats = _sumStats();
  state.davBase = health.dav_base_url || location.origin;
  state.lastPollLocal = (tools.generated_at_local || "").slice(11, 19);
  _emit("bootstrap");
}

function _applyBody(body) {
  state.fingerprint = body.fingerprint;
  state.generatedAt = body.generated_at;
  state.lastPollLocal = (body.generated_at_local || "").slice(11, 19);
  noteServerTime(body.generated_at, body.generated_at_local);
  // Counts for every tool, even the ones the console does not list.
  state.toolStats = body.stats.tools;
  const changed = [];
  for (const list of [body.running, body.started, body.finished]) {
    for (const t of list) {
      const prev = state.tasks.get(t.task_id);
      if (!prev || prev.status !== t.status || prev.ended_at !== t.ended_at) {
        state.tasks.set(t.task_id, t);
        changed.push(t);
      }
    }
  }
  state.running = body.running;
  for (const [name, s] of Object.entries(body.stats.tools)) {
    const row = state.tools.get(name);
    if (row) {
      row.running = s.running;
      row.completed = s.completed;
      row.failed = s.failed;
      row.cancelled = s.cancelled;
      row.last_activity = s.last_activity;
      row.last_activity_local = s.last_activity_local;
      row.running_since = s.running_since;
    }
  }
  // The counter adds up only the tools the console lists.
  state.stats = _sumStats();
  return changed;
}

async function _poll() {
  if (_inFlight) return;
  _inFlight = true;
  try {
    const body = await fetchChanges(state.fingerprint, state.generatedAt);
    if (body === null) {
      state.failStreak = 0;
      state.healthy = true;
      _emit("poll-304");
      return;
    }
    state.failStreak = 0;
    state.healthy = true;
    const changed = _applyBody(body);
    _emit("changes", changed);
  } catch (err) {
    state.failStreak += 1;
    state.healthy = false;
    _emit("fail", err);
  } finally {
    _inFlight = false;
  }
}

export function startPolling() {
  if (_timer) return;
  _timer = setInterval(_poll, _POLL_MS);
  _poll();
}

export function stopPolling() {
  if (_timer) {
    clearInterval(_timer);
    _timer = null;
  }
}
