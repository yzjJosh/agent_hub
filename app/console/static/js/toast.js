import { esc } from "./format.js";
import { copyToClipboard } from "./clipboard.js";

const _DISMISS_MS = 2500;
const _ERROR_DISMISS_MS = 4000;
const _MAX_STACK = 3;

const _ICONS = {
  ok: '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m3 8.5 3.5 3.5L13 5"/></svg>',
  alert: '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="8" cy="8" r="6"/><path d="M8 5v4M8 11.2v.3"/></svg>',
  x: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M4 4l8 8M12 4l-8 8"/></svg>',
};

export function toast(msg, opts = {}) {
  const stack = document.getElementById("toast-stack");
  if (!stack) return;
  while (stack.children.length >= _MAX_STACK) stack.firstElementChild.remove();
  const el = document.createElement("div");
  el.className = "toast" + (opts.error ? " toast-error" : "");
  el.innerHTML =
    '<span class="toast-ic">' + (opts.error ? _ICONS.alert : _ICONS.ok) + "</span>" +
    '<span class="toast-msg">' + esc(msg) + "</span>" +
    '<button class="toast-x" title="Dismiss">' + _ICONS.x + "</button>";
  el.querySelector(".toast-x").addEventListener("click", () => el.remove());
  stack.appendChild(el);
  requestAnimationFrame(() => el.classList.add("show"));
  setTimeout(() => {
    el.classList.remove("show");
    setTimeout(() => el.remove(), 200);
  }, opts.error ? _ERROR_DISMISS_MS : _DISMISS_MS);
}

export function copyText(text, msg) {
  copyToClipboard(text).then(
    (ok) => {
      if (ok) toast(msg);
      else toast("Copy failed", { error: true });
    },
    () => toast("Copy failed", { error: true })
  );
}
