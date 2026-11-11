import { icon } from "./ui.js";
import { mountParameterText } from "./parameter-text.js";

const _INITIAL = 6;
const _BATCH = 50;

function _button(text, action, className = "btn ghost") {
  const button = document.createElement("button");
  button.type = "button";
  button.className = className;
  button.textContent = text;
  button.addEventListener("click", action);
  return button;
}

function _grow(input) {
  input.style.height = "auto";
  input.style.height = input.scrollHeight + "px";
}

export function mountParameterList(host, values, {
  editable = false, label = "List", onChange = () => {}, onEditingChange = () => {},
} = {}) {
  let entries = [];
  let limit = _INITIAL;
  let editing = null;
  let latest = null;
  const rows = new Map();
  const box = document.createElement("div");
  box.className = "parameter-list";
  box.setAttribute("role", "group");
  box.setAttribute("aria-label", label);
  const count = document.createElement("div");
  count.className = "parameter-list-count";
  const body = document.createElement("div");
  const continuation = document.createElement("div");
  continuation.className = "parameter-continuation";
  const latestBox = document.createElement("div");
  const composer = document.createElement("div");
  composer.className = "parameter-composer";
  const input = document.createElement("textarea");
  input.className = "f-input";
  input.rows = 1;
  input.setAttribute("aria-label", label + " item");
  input.placeholder = "Add an item";
  const message = document.createElement("div");
  message.setAttribute("role", "alert");
  const add = _button("Add", () => { commit(); input.focus(); });
  add.disabled = true;
  composer.append(input, add);
  box.append(count, body, continuation, latestBox);
  if (editable) box.append(composer, message);
  host.appendChild(box);

  function notify() {
    onChange();
    render();
  }

  function finish(entry) {
    editing = null;
    rows.get(entry).stop();
    rows.get(entry).row.remove();
    rows.delete(entry);
    onEditingChange(false);
    render();
    rows.get(entry)?.edit?.focus();
  }

  function remove(entry) {
    const index = entries.indexOf(entry);
    const next = entries[index + 1] || entries[index - 1];
    entries.splice(index, 1);
    if (editing === entry) {
      editing = null;
      onEditingChange(false);
    }
    notify();
    (rows.get(next)?.edit || input).focus();
  }

  function buildRow(entry) {
    const row = document.createElement("div");
    row.className = "parameter-item";
    const content = document.createElement("div");
    content.className = "parameter-item-content";
    row.appendChild(content);
    if (editing !== entry) {
      const text = mountParameterText(content, entry.value);
      const edit = editable ? _button("", () => {
        if (editing) return;
        editing = entry;
        rows.get(entry).stop();
        rows.get(entry).row.remove();
        rows.delete(entry);
        onEditingChange(true);
        render();
        rows.get(entry).input.focus();
      }, "icon-btn parameter-action") : null;
      if (edit) {
        edit.innerHTML = icon("edit");
        edit.setAttribute("aria-label", "Edit " + entry.value);
        row.appendChild(edit);
      }
      return { row, edit, stop: text.stop };
    }
    const draft = document.createElement("textarea");
    draft.className = "f-input";
    draft.value = entry.value;
    const initialValue = draft.value;
    draft.setAttribute("aria-label", "Edit " + label + " item");
    draft.addEventListener("input", () => _grow(draft));
    const error = document.createElement("div");
    error.setAttribute("role", "alert");
    const actions = document.createElement("div");
    actions.className = "parameter-editor-actions";
    actions.append(_button("Save", () => {
      const value = draft.value === initialValue ? entry.value : draft.value;
      if (!value.trim()) { error.textContent = "Enter an item."; return; }
      if (value !== entry.value && entries.some((item) => item !== entry && item.value === value)) {
        error.textContent = "This item is already in the list.";
        return;
      }
      entry.value = value;
      finish(entry);
      onChange();
    }), _button("Cancel", () => finish(entry)));
    content.append(draft, actions, error);
    const trash = _button("", () => remove(entry), "icon-btn parameter-action parameter-remove");
    trash.innerHTML = icon("trash");
    trash.setAttribute("aria-label", "Remove " + entry.value);
    row.appendChild(trash);
    const frame = requestAnimationFrame(() => _grow(draft));
    return { row, input: draft, stop: () => cancelAnimationFrame(frame) };
  }

  function render() {
    const focused = document.activeElement;
    count.textContent = entries.length ? entries.length + (entries.length === 1 ? " item" : " items") : "No items";
    const visible = entries.slice(0, limit);
    const extra = latest && entries.includes(latest) && !visible.includes(latest) ? latest : null;
    const wanted = new Set(extra ? [...visible, extra] : visible);
    for (const [entry, child] of rows) {
      if (wanted.has(entry)) continue;
      child.stop(); child.row.remove(); rows.delete(entry);
    }
    for (const entry of wanted) if (!rows.has(entry)) rows.set(entry, buildRow(entry));
    visible.forEach((entry, index) => {
      const row = rows.get(entry).row;
      if (body.children[index] !== row) body.insertBefore(row, body.children[index] || null);
    });
    continuation.replaceChildren();
    if (limit < entries.length) continuation.appendChild(_button("Show next " + Math.min(_BATCH, entries.length - limit), () => {
      limit += _BATCH; render(); continuation.firstElementChild?.focus();
    }));
    if (limit > _INITIAL) {
      const collapse = _button("Show first 6", () => { limit = _INITIAL; render(); continuation.firstElementChild?.focus(); });
      collapse.disabled = !!editing && entries.indexOf(editing) >= _INITIAL;
      continuation.appendChild(collapse);
    }
    if (extra) {
      if (!latestBox.firstChild || latestBox.firstChild.nodeType !== Node.TEXT_NODE) latestBox.prepend(document.createTextNode("Latest addition"));
      latestBox.appendChild(rows.get(extra).row);
    } else latestBox.replaceChildren();
    if (editing && focused && box.contains(focused) && document.activeElement !== focused) focused.focus();
  }

  function commit() {
    if (editing) return false;
    if (!input.value.trim()) return true;
    if (entries.some((entry) => entry.value === input.value)) {
      message.textContent = "This item is already in the list.";
      return false;
    }
    latest = { value: input.value };
    entries.push(latest);
    input.value = "";
    add.disabled = true;
    message.textContent = "";
    _grow(input);
    notify();
    return true;
  }

  input.addEventListener("input", () => { add.disabled = !input.value.trim(); message.textContent = ""; _grow(input); });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      if (input.value) { input.value = ""; add.disabled = true; message.textContent = ""; _grow(input); }
      else input.blur();
    } else if (event.key === "Backspace" && !input.value && !editing && entries.length) {
      event.preventDefault(); entries.pop(); notify();
    }
  });
  function set(next) {
    for (const child of rows.values()) { child.stop(); child.row.remove(); }
    rows.clear();
    entries = next.map((value) => ({ value }));
    editing = null; latest = null; limit = _INITIAL;
    onEditingChange(false);
    render();
  }
  set(values);
  return { control: input, read: () => entries.map((entry) => entry.value), set, commit,
    stop: () => { for (const child of rows.values()) child.stop(); } };
}
