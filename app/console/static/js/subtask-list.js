import { on, getState } from "./state.js";

function _nameOf(t) {
  return t.agent_name || t.tool || t.task_id.slice(0, 8);
}

function _duration(t) {
  if (!t.started_at_local) return "";
  if (t.status === "running") return "";
  if (!t.ended_at_local) return "";
  const s = new Date(t.ended_at) - new Date(t.started_at);
  if (s < 1000) return s + "ms";
  if (s < 60000) return (s / 1000).toFixed(1) + "s";
  return Math.floor(s / 60000) + "m " + Math.round((s % 60000) / 1000) + "s";
}

function _row(t) {
  const row = document.createElement("a");
  row.className = "st-row";
  row.href = "/task/" + t.task_id;
  row.dataset.spa = "";
  row.dataset.taskId = t.task_id;

  const dot = document.createElement("span");
  dot.className = "st-dot st-dot-" + t.status;
  if (t.status === "running") dot.classList.add("pulse");

  const badge = document.createElement("span");
  badge.className = "st-badge st-badge-" + t.status;
  badge.textContent = t.status;

  const name = document.createElement("span");
  name.className = "st-name";
  name.textContent = _nameOf(t);

  const cells = [dot, badge, name];
  if (t.agent_name) {
    const chip = document.createElement("span");
    chip.className = "st-chip";
    chip.textContent = "AGENT";
    cells.push(chip);
  }

  const start = document.createElement("span");
  start.className = "st-start";
  start.textContent = t.started_at_local || "";

  const dur = document.createElement("span");
  dur.className = "st-dur";
  dur.textContent = _duration(t);

  row.append(...cells, start, dur);
  return row;
}

export function mountSubtaskList(mount, subtasks, opts) {
  const list = document.createElement("div");
  list.className = "st-list";
  list.id = "sub-well";

  // The visible list merges the seed snapshot with the children the changes
  // poll has seen for this parent. The state entry wins a task both carry.
  let _seed = [...(subtasks || [])];

  function _merged() {
    const byId = new Map();
    for (const t of _seed) byId.set(t.task_id, t);
    for (const t of getState().tasks.values()) {
      if (t.parent_task_id === opts.taskId) byId.set(t.task_id, t);
    }
    return [...byId.values()];
  }

  let _lastCount = -1;

  function render(tasks) {
    list.replaceChildren();
    if (!tasks.length) {
      const empty = document.createElement("div");
      empty.className = "st-empty";
      empty.textContent = "No subtasks";
      list.appendChild(empty);
    } else {
      for (const t of tasks) list.appendChild(_row(t));
    }
    if (opts.onCount && tasks.length !== _lastCount) {
      _lastCount = tasks.length;
      opts.onCount(tasks.length);
    }
  }

  render(_merged());
  mount.appendChild(list);

  const _onChanges = (kind, payload) => {
    if (kind !== "changes") return;
    let refetch = false;
    for (const task of payload) {
      if (task.parent_task_id !== opts.taskId) continue;
      if (task.status !== "running") refetch = true;
    }
    render(_merged());
    if (refetch && opts.onRefetch) opts.onRefetch();
  };

  const unsub = on(_onChanges);

  return {
    refresh(newTasks) {
      _seed = [...(newTasks || [])];
      render(_merged());
    },
    destroy() {
      if (unsub) unsub();
      list.remove();
    },
  };
}
