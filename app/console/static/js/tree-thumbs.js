// Watches one whole list for unsettled thumbnail rows.
// On every tick it asks the server to make the thumbnails in each dir that
// still has one, re-fetches those dirs, and hands each fresh listing back to
// applyDir. It stops on its own when the surface goes away or nothing is
// left to wait for, and the handle stops it when the surface unmounts.
import { getJSON, postJSON } from "./api.js";

const _POLL_MS = 2000;

export function watchThumbs({ active, dirsInScope, applyDir, pollMs = _POLL_MS }) {
  let timer = null;
  let inFlight = false;

  function stop() {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
  }

  async function tick() {
    if (!active()) {
      stop();
      return;
    }
    const dirs = [...dirsInScope()];
    if (dirs.length === 0) {
      stop();
      return;
    }
    if (inFlight) return; // A slow re-fetch is still running. The next tick picks it up.
    inFlight = true;
    for (const d of dirs) {
      postJSON("/api/resources/thumbs?" + new URLSearchParams({ path: d })).catch(() => {});
      let body;
      try {
        body = await getJSON("/api/resources?" + new URLSearchParams({ path: d }));
      } catch {
        continue; // The folder is gone or the fetch failed. The rows keep their last state.
      }
      applyDir(d, body.entries || []);
    }
    inFlight = false;
    if (!active() || [...dirsInScope()].length === 0) stop();
  }

  // Idempotent. It does nothing while already running or while there is
  // nothing unsettled to watch, so a surface can call it after every render.
  function start() {
    if (timer) return;
    if ([...dirsInScope()].length === 0) return;
    timer = setInterval(tick, pollMs);
  }

  return { start, stop };
}
