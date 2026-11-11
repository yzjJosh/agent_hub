export function mountParameterText(host, text, { href } = {}) {
  const box = document.createElement("div");
  box.className = "parameter-text";
  const value = document.createElement(href ? "a" : "div");
  value.className = "parameter-text-value";
  value.textContent = text;
  if (href) {
    value.href = href;
    value.target = "_blank";
    value.rel = "noopener";
  }
  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "parameter-disclosure";
  toggle.textContent = "Read more";
  toggle.setAttribute("aria-expanded", "false");
  toggle.hidden = true;
  box.append(value, toggle);
  host.appendChild(box);
  let expanded = false;
  let frame = null;
  let stopped = false;
  const measure = () => {
    frame = null;
    if (stopped || !value.getClientRects().length) return;
    if (!expanded) toggle.hidden = value.scrollHeight > value.clientHeight + 1 ? false : true;
  };
  const schedule = () => {
    if (!stopped && frame === null) frame = requestAnimationFrame(measure);
  };
  const observer = new ResizeObserver(schedule);
  observer.observe(value);
  toggle.addEventListener("click", () => {
    expanded = !expanded;
    box.classList.toggle("expanded", expanded);
    toggle.textContent = expanded ? "Read less" : "Read more";
    toggle.setAttribute("aria-expanded", String(expanded));
    schedule();
  });
  document.fonts.ready.then(schedule);
  schedule();
  return { stop: () => {
    stopped = true;
    observer.disconnect();
    if (frame !== null) cancelAnimationFrame(frame);
  } };
}
