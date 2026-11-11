import { emptyState } from "./ui.js";

const _routes = [];
let _current = null;

// The navigation model mirrors the browser session history. The stack runs
// from the session's first SPA entry to the browser's current entry, the top.
// The forward list holds SPA-owned entries ahead of the current, in browser
// order, closest first, the front.
const _stack = [];
const _forward = [];
let _ids = 0;
const _byId = new Map();

// A jump walk in flight holds the target entry id, and one outstanding
// go(-1) between its issue and its delivery.
let _walk = null;
let _deferred = null;

function _nextId() {
  _ids += 1;
  return _ids;
}

function _stateFor(entry) {
  return Object.assign({ id: entry.id }, entry.state);
}

// The canonical full URL. "/" shows the tools screen. The legacy
// /tools/<t>/<id> form is the /task/<id> route, and the rewrite applies
// only when the segment in the task-id position is not "run".
function _canonical(url) {
  const u = new URL(url, location.origin);
  let path = u.pathname;
  const search = u.search;
  if (path === "/" || path === "") path = "/tools";
  else if (path.length > 1 && path.endsWith("/")) path = path.slice(0, -1);
  const seg = path.split("/");
  if (seg.length === 4 && seg[1] === "tools" && seg[3] !== "run") {
    path = "/task/" + seg[3];
  }
  return path + search;
}

function _entryFor(canonical) {
  const u = new URL(canonical, location.origin);
  const id = _nextId();
  const entry = { id, path: u.pathname, search: u.search, view: {}, state: {} };
  _byId.set(id, entry);
  return entry;
}

function _top() {
  return _stack[_stack.length - 1] || null;
}

export function entryView() {
  const top = _top();
  return top ? top.view : {};
}

export function entryState() {
  const top = _top();
  return top ? Object.assign({}, top.state) : {};
}

// Pushes a new history entry at the current URL carrying stateFields in
// history.state alongside the entry id. The new entry gets a fresh empty
// view, becomes the top, and the forward list is cleared.
export function pushEntry(stateFields) {
  const entry = {
    id: _nextId(),
    path: location.pathname,
    search: location.search,
    view: {},
    state: Object.assign({}, stateFields || {}),
  };
  _byId.set(entry.id, entry);
  history.pushState(_stateFor(entry), "", entry.path + entry.search);
  _stack.push(entry);
  _forward.length = 0;
  return entry;
}

// Rewrites the current entry's screen state fields to exactly stateFields
// and, when url is given, rewrites the entry's URL in the same call. Fields
// not in the argument are dropped, so a caller that preserves the current
// fields passes them back from entryState().
export function replaceEntry(stateFields, url) {
  const top = _top();
  if (!top) return;
  top.state = Object.assign({}, stateFields || {});
  if (url != null) {
    const u = new URL(_canonical(url), location.origin);
    top.path = u.pathname;
    top.search = u.search;
  }
  history.replaceState(_stateFor(top), "", top.path + top.search);
}

export function addRoute(spec) {
  _routes.push(spec);
}

function _match(path) {
  for (const r of _routes) {
    const m = path.match(r.re);
    if (!m) continue;
    const params = {};
    try {
      r.params.forEach((name, i) => { params[name] = decodeURIComponent(m[i + 1]); });
    } catch {
      continue; // A bad path does not match this route.
    }
    return { route: r, params };
  }
  return null;
}

// The topmost stack entry the URL names. A URL with a search matches only
// an entry with the same search. A URL without a search matches any search
// variant of the same pathname. Forward-list entries do not match.
function _matchEntry(canonical) {
  const u = new URL(canonical, location.origin);
  for (let i = _stack.length - 1; i >= 0; i--) {
    const e = _stack[i];
    if (e.path !== u.pathname) continue;
    if (u.search && e.search !== u.search) continue;
    return e;
  }
  return null;
}

// Screens render one at a time, even when clicks come fast.
let _chain = Promise.resolve();

export function navigate(url) {
  _chain = _chain.then(() => _navigate(url)).catch(() => {});
  return _chain;
}

function _navigate(url) {
  if (_walk) {
    if (!_deferred) _deferred = [];
    _deferred.push(url);
    return;
  }
  const canonical = _canonical(url);
  const target = _matchEntry(canonical);
  if (target === _top()) return;
  if (target) {
    _walk = { targetId: target.id, outstanding: 1 };
    history.go(-1);
    return;
  }
  const entry = _entryFor(canonical);
  history.pushState(_stateFor(entry), "", entry.path + entry.search);
  _stack.push(entry);
  _forward.length = 0;
  _dispatchQueue(entry);
}

// The walk settles when its target is reached or a user traversal cancels
// it. Deferred navigations re-resolve against the settled stack. A brief
// flash of the walk's target before a deferred push is the accepted cost.
function _walkSettled() {
  _walk = null;
  if (_deferred && _deferred.length) {
    const urls = _deferred;
    _deferred = null;
    for (const u of urls) navigate(u);
  }
}

export function dispatch() {
  const top = _top();
  if (top) _dispatchQueue(top);
}

function _dispatchQueue(entry) {
  _chain = _chain.then(() => _renderEntry(entry)).catch(() => {});
}

async function _renderEntry(entry) {
  // A dispatch renders only when its entry is the top at run time, so the
  // ambient location equals the entry's URL and the screens' existing
  // location.search reads stay valid.
  if (entry !== _top()) return;
  if (_current && _current.destroy) _current.destroy();
  _current = null;
  const view = document.getElementById("view");
  view.replaceChildren();
  const hit = _match(entry.path);
  if (!hit) {
    _setNav(null);
    document.title = "Agent Hub";
    emptyState(view, "alert", "Not found", "This screen does not exist.", "Back to Tools", "/tools");
    _current = { root: view };
    return;
  }
  _setNav(hit.route.section);
  document.title = (hit.route.title(hit.params) || "Agent Hub") + " — Agent Hub";
  const el = document.createElement("div");
  el.className = "screen";
  view.appendChild(el);
  _current = await hit.route.render(el, hit.params);
}

function _setNav(section) {
  document.querySelectorAll("[data-nav]").forEach((a) => a.classList.toggle("active", a.getAttribute("data-nav") === section));
}

// The init adoption takes the current URL, canonicalizes it, and makes it
// the first entry. The id is retained when the current entry already
// carries one, otherwise a new one is assigned and the sequence starts
// above it. The screen fields of a previous load are dropped, and a fresh
// load starts from defaults.
function _adoptCurrent() {
  const s = history.state || {};
  let id = 0;
  if (s.id != null && Number.isInteger(Number(s.id)) && Number(s.id) > 0) id = Number(s.id);
  if (!id) id = _nextId();
  else _ids = Math.max(_ids, id);
  const u = new URL(_canonical(location.href), location.origin);
  const entry = { id, path: u.pathname, search: u.search, view: {}, state: {} };
  _byId.set(id, entry);
  history.replaceState(_stateFor(entry), "", entry.path + entry.search);
  _stack.push(entry);
}

// The landing adoption handles a popstate on an id this load never stamped.
// A traversal to an entry the current document does not own fires popstate
// without navigating only when the URL is the same, and the SPA's only
// same-URL pair is the resources preview base and preview. The fresh entry
// keeps the landed screen fields, is placed by the id comparison with the
// current top, and is re-stamped with a fresh id. The smaller id sits
// behind only while the two ids are in browser-creation order, which is
// the single-reload case. The double-reload sub-case misclassifies one
// step and the rebuild re-anchors. The rule ends here.
function _adoptLanded() {
  const prevTop = _top();
  const prevId = prevTop ? prevTop.id : 0;
  const s = history.state || {};
  const landedId = s.id != null ? Number(s.id) : 0;
  const state = {};
  for (const key of Object.keys(s)) if (key !== "id") state[key] = s[key];
  const u = new URL(_canonical(location.href), location.origin);
  const id = _nextId();
  const entry = { id, path: u.pathname, search: u.search, view: {}, state };
  _byId.set(id, entry);
  if (prevTop && landedId < prevId) {
    _stack.pop();
    _forward.unshift(prevTop);
  }
  _stack.push(entry);
  history.replaceState(_stateFor(entry), "", entry.path + entry.search);
  _dispatchQueue(entry);
}

// The defensive rebuild re-anchors the model when a tracked entry lands
// that no other rule classified.
function _rebuild(landed) {
  const inStack = _stack.indexOf(landed);
  const inForward = _forward.indexOf(landed);
  if (inStack < 0 && inForward < 0) {
    for (const e of _stack) e.view = {};
    for (const e of _forward) e.view = {};
    _stack.length = 0;
    _forward.length = 0;
    _stack.push(landed);
    _dispatchQueue(landed);
    return;
  }
  const behind = [];
  const ahead = [];
  if (inStack >= 0) {
    behind.push(..._stack.slice(0, inStack));
    ahead.push(..._stack.slice(inStack + 1));
  } else {
    behind.push(..._stack, ..._forward.slice(0, inForward));
    ahead.push(..._forward.slice(inForward + 1));
  }
  _stack.length = 0;
  _forward.length = 0;
  for (const e of behind) _stack.push(e);
  _stack.push(landed);
  for (const e of ahead) {
    e.view = {};
    _forward.push(e);
  }
  _dispatchQueue(landed);
}

// The popstate rule resolves by id first. An unknown id adopts and the
// rule ends. Then it tries, in turn, a walk step, a walk cancellation, a
// forward step, a user back, and the defensive rebuild. onPop is consulted
// only in the user-back branch, since a walk step and a walk completion
// never consult it.
function _onPop() {
  const s = history.state || {};
  const landed = s.id != null ? _byId.get(Number(s.id)) : undefined;
  if (!landed) {
    _adoptLanded();
    return;
  }
  const below = _stack[_stack.length - 2] || null;
  if (_walk && _walk.outstanding === 1 && landed === below) {
    _walk.outstanding = 0;
    const popped = _stack.pop();
    popped.view = {};
    _forward.unshift(popped);
    if (landed.id === _walk.targetId) {
      _dispatchQueue(landed);
      _walkSettled();
    } else {
      _walk.outstanding = 1;
      history.go(-1);
    }
    return;
  }
  if (_walk) _walkSettled();
  if (landed === _forward[0]) {
    _forward.shift();
    _stack.push(landed);
    _dispatchQueue(landed);
    return;
  }
  if (landed === below) {
    const popped = _stack.pop();
    _forward.unshift(popped);
    if (_current && _current.onPop && _current.onPop()) return;
    _dispatchQueue(landed);
    return;
  }
  _rebuild(landed);
}

export function initRouter() {
  window.addEventListener("popstate", _onPop);
  document.addEventListener("click", (ev) => {
    const a = ev.target.closest ? ev.target.closest("a[data-spa]") : null;
    if (!a) return;
    ev.preventDefault();
    navigate(a.getAttribute("href"));
  });
  _adoptCurrent();
  dispatch();
}
