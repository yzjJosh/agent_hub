import { navigate } from "./router.js";
import { getState, on } from "./state.js";

function _crumbEl(name, linkId, tid, status) {
  const el = document.createElement("span");
  el.className = "crumb-item";
  el.dataset.tid = tid;
  if (linkId) {
    el.classList.add("crumb-link");
    el.addEventListener("click", (ev) => {
      ev.preventDefault();
      navigate("/task/" + linkId);
    });
  }
  if (status === "running") {
    const dot = document.createElement("span");
    dot.className = "crumb-dot";
    el.appendChild(dot);
  }
  const label = document.createElement("span");
  label.className = "crumb-name";
  label.textContent = name;
  el.appendChild(label);
  return el;
}

function _sep() {
  const el = document.createElement("span");
  el.className = "crumb-sep";
  el.textContent = "\u203A";
  return el;
}

function _nameOf(entry) {
  return entry.agent_name || entry.tool || entry.task_id.slice(0, 8);
}

function _statusOf(entry) {
  const live = getState().tasks.get(entry.task_id);
  if (live) return live.status;
  return entry.status;
}

export function mountCrumb(mount, ancestors, currentTask) {
  const bar = document.createElement("nav");
  bar.className = "crumb-bar";
  bar.setAttribute("aria-label", "Task ancestry");

  const all = [...ancestors, currentTask];
  const visible = all.length <= 5 ? all : [all[0], null, all[all.length - 2], all[all.length - 1]];

  const menuClosers = [];
  for (const entry of visible) {
    if (entry === null) {
      const dots = document.createElement("button");
      dots.type = "button";
      dots.className = "crumb-ellipsis";
      dots.textContent = "\u2026";
      const hidden = all.slice(1, -2);
      dots.title = hidden.length + " hidden ancestors";
      let menu = null;
      let _outside = null;
      let _esc = null;
      function _closeMenu() {
        if (!menu) return;
        menu.remove();
        dots.classList.remove("open");
        document.removeEventListener("mousedown", _outside);
        document.removeEventListener("keydown", _esc);
        menu = null;
        _outside = null;
        _esc = null;
      }
      menuClosers.push(_closeMenu);
      function _openMenu() {
        menu = document.createElement("div");
        menu.className = "crumb-menu";
        const cap = document.createElement("div");
        cap.className = "crumb-menu-cap";
        cap.textContent = "Hidden ancestors";
        menu.appendChild(cap);
        for (const h of hidden) {
          const row = document.createElement("button");
          row.type = "button";
          row.className = "crumb-menu-row";
          row.textContent = _nameOf(h);
          if (_statusOf(h) === "running") {
            const d = document.createElement("span");
            d.className = "crumb-dot";
            row.prepend(d);
          }
          row.addEventListener("click", () => {
            navigate("/task/" + h.task_id);
            _closeMenu();
          });
          menu.appendChild(row);
        }
        _outside = (ev) => { if (!bar.contains(ev.target)) _closeMenu(); };
        _esc = (ev) => { if (ev.key === "Escape") _closeMenu(); };
        document.addEventListener("mousedown", _outside);
        document.addEventListener("keydown", _esc);
        dots.classList.add("open");
        bar.appendChild(menu);
      }
      dots.addEventListener("click", (ev) => {
        ev.stopPropagation();
        if (menu) { _closeMenu(); return; }
        _openMenu();
      });
      bar.appendChild(dots);
      bar.appendChild(_sep());
      continue;
    }
    const isCurrent = entry.task_id === currentTask.task_id;
    // The current crumb never carries the dot, its own status is in the header.
    const crumb = _crumbEl(_nameOf(entry), isCurrent ? null : entry.task_id, entry.task_id, isCurrent ? null : _statusOf(entry));
    if (isCurrent) crumb.classList.add("crumb-current");
    bar.appendChild(crumb);
    if (!isCurrent) bar.appendChild(_sep());
  }

  mount.appendChild(bar);

  // On a phone the bar is a pill strip scrolled to the current pill, with a
  // left-edge fade marking the clipped ancestors.
  let _stripScroll = null;
  if (matchMedia("(max-width:767px)").matches) {
    bar.scrollLeft = bar.scrollWidth;
    _stripScroll = () => bar.classList.toggle("scrolled", bar.scrollLeft > 4);
    bar.addEventListener("scroll", _stripScroll);
    _stripScroll();
  }

  const _onTask = (kind, payload) => {
    if (kind !== "changes") return;
    for (const task of payload) {
      const id = task.task_id;
      const item = bar.querySelector('.crumb-item[data-tid="' + id + '"]');
      if (!item || item.classList.contains("crumb-current")) continue;
      const t = getState().tasks.get(id);
      if (!t) continue;
      const dot = item.querySelector(".crumb-dot");
      if (t.status === "running" && !dot) {
        const d = document.createElement("span");
        d.className = "crumb-dot";
        item.prepend(d);
      } else if (t.status !== "running" && dot) {
        dot.remove();
      }
    }
  };

  const unsub = on(_onTask);
  return {
    destroy() {
      if (unsub) unsub();
      if (_stripScroll) bar.removeEventListener("scroll", _stripScroll);
      for (const close of menuClosers) close();
      bar.remove();
    },
  };
}
