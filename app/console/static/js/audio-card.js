// The audio player card for a result.
// This file must not import anything, and must not touch the page when it loads.
const _PLAY = '<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M4 2.5v11l9-5.5z"/></svg>';
const _PAUSE = '<svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><rect x="3.5" y="2.5" width="3.5" height="11" rx="1"/><rect x="9" y="2.5" width="3.5" height="11" rx="1"/></svg>';
const _TICK_MS = 250;

function _pad(n) {
  return String(n).padStart(2, "0");
}

// Turns seconds into a clock like 3:07, or 1:02:07 for an hour or more.
export function fmtMediaClock(seconds) {
  const s = Math.max(0, Math.floor(Number.isFinite(seconds) ? seconds : 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return h > 0 ? h + ":" + _pad(m) + ":" + _pad(sec) : m + ":" + _pad(sec);
}

export function initAudioCard(container, src, filename) {
  // The card is built inside container. Stopping takes container off the page.
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "audio-play";
  btn.innerHTML = _PLAY;
  const name = document.createElement("span");
  name.className = "audio-name";
  name.textContent = filename || "";
  const seek = document.createElement("div");
  seek.className = "audio-seek";
  const fill = document.createElement("div");
  fill.className = "audio-seek-fill";
  const knob = document.createElement("div");
  knob.className = "audio-seek-knob";
  seek.append(fill, knob);
  const clock = document.createElement("span");
  clock.className = "audio-clock";
  const audio = new Audio(src);
  audio.style.display = "none";
  container.append(btn, name, seek, clock, audio);

  let tick = null;

  function sync() {
    const dur = audio.duration || 0;
    clock.textContent = fmtMediaClock(audio.currentTime || 0) + " / " + fmtMediaClock(dur);
    const pct = dur ? (Math.min(1, Math.max(0, audio.currentTime / dur)) * 100) : 0;
    fill.style.width = pct + "%";
    knob.style.left = pct + "%";
  }

  function onPlay() {
    btn.innerHTML = _PAUSE;
    btn.setAttribute("aria-label", "Pause");
    if (!tick) tick = setInterval(sync, _TICK_MS);
  }

  function onPause() {
    btn.innerHTML = _PLAY;
    btn.setAttribute("aria-label", "Play");
    if (tick) { clearInterval(tick); tick = null; }
    sync();
  }

  audio.addEventListener("play", onPlay);
  audio.addEventListener("pause", onPause);
  audio.addEventListener("ended", onPause);
  audio.addEventListener("loadedmetadata", sync);
  audio.addEventListener("timeupdate", sync);

  btn.addEventListener("click", () => {
    if (audio.paused) audio.play().catch(() => {});
    else audio.pause();
  });
  seek.addEventListener("click", (ev) => {
    if (!audio.duration) return;
    const r = seek.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width));
    audio.currentTime = frac * audio.duration;
    sync();
  });

  sync();

  function stop() {
    audio.removeEventListener("play", onPlay);
    audio.removeEventListener("pause", onPause);
    audio.removeEventListener("ended", onPause);
    audio.removeEventListener("loadedmetadata", sync);
    audio.removeEventListener("timeupdate", sync);
    if (tick) clearInterval(tick);
    audio.pause();
    audio.src = "";
    container.remove();
  }
  return { stop };
}
