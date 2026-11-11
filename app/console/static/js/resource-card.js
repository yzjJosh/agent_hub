// The file card on the tool-run form. One file, one reviewed card.
// It renders from a fact or from a bare path, and finishes in place when
// the missing pieces arrive. The fact comes from the parent folder's
// listing and the thumbnail from a 2-second probe while it is being made.
import { getJSON, encodePath } from "./api.js";
import { esc, fmtBytes } from "./format.js";
import { icon, fileGlyph } from "./ui.js";
import { compactSlot } from "./gallery.js";

const _PROBE_MS = 2000;
const _GONE_NOTE = "Removed by 30-day retention · no longer exists";

// One listing fetch per parent folder per session, used by every card in it.
const _listingPromises = new Map();

function _isPhone() {
  return matchMedia("(max-width: 767px)").matches;
}

// The parent folder of one path, "" for the top folder.
function _parent(path) {
  const i = path.lastIndexOf("/");
  return i === -1 ? "" : path.slice(0, i);
}

// The listing of one parent folder, null when the folder is gone.
function _listingOf(parent) {
  if (!_listingPromises.has(parent)) {
    _listingPromises.set(
      parent,
      getJSON("/api/resources?" + new URLSearchParams({ path: parent }))
        .then((body) => body.entries)
        .catch((err) => {
          if (err && err.status === 404) return null;
          _listingPromises.delete(parent);
          throw err;
        }),
    );
  }
  return _listingPromises.get(parent);
}

// The fact of one file from its parent folder's listing.
function _factOf(path) {
  return _listingOf(_parent(path)).then((entries) => (entries ? entries.find((e) => e.path === path) || null : null));
}

export function renderResourceCard(host, source, opts = {}) {
  const phone = _isPhone();
  const path = typeof source === "string" ? source : source.path;
  let fact = typeof source === "string" ? null : source;
  let gone = false;
  let probeTimer = null;
  let stopped = false;

  const card = document.createElement("div");
  card.className = "res-card";
  card.dataset.path = path;
  host.appendChild(card);

  const name = () => (fact ? fact.name : path.split("/").pop());
  const folder = () => {
    const p = _parent(path);
    return p === "" ? "resource" : p.split("/").pop();
  };
  const thumbUrl = () => "/api/resources/thumb?path=" + encodeURIComponent(path);

  function subline() {
    if (gone) return _GONE_NOTE;
    const parts = [folder()];
    if (fact && fact.size != null) parts.push(fmtBytes(fact.size));
    if (fact && fact.kind && !phone) parts.push(fact.kind);
    return parts.join(" · ");
  }

  function slot() {
    if (gone) return '<span class="thumb-1610"><span class="thumb-glyph">' + fileGlyph(name()) + "</span></span>";
    return compactSlot(fact || { name: name() });
  }

  // The card has one trailing square action. It is Remove for the run form,
  // Copy for the L3 multi-element items, and none for the read-only single card.
  // onRemove wins when both are supplied.
  function trailing() {
    if (opts.onRemove) {
      return '<button class="act-x" type="button" title="Remove">' + icon("x") + "</button>";
    }
    if (opts.onCopy) {
      return '<button class="act-copy" type="button" title="Copy path" aria-label="Copy path">' + icon("copy") + "</button>";
    }
    return "";
  }

  // Every draw is a fact change. The caller's onFact sees the live fact,
  // or null once the card has settled gone.
  function emitFact() {
    if (opts.onFact) opts.onFact(gone ? null : fact);
  }

  function draw() {
    const inner =
      slot() +
      '<span class="txt"><span class="nm" title="' + esc(path) + '">' + esc(name()) + '</span><span class="sub">' + esc(subline()) + "</span></span>";
    const acts = trailing();
    const actsHtml = acts ? '<span class="acts">' + acts + "</span>" : "";
    card.className = "res-card" + (gone ? " gone" : "");
    if (gone) {
      card.innerHTML = '<span class="res-body">' + inner + "</span>" + actsHtml;
    } else {
      // The link is the tappable region. It wraps the slot and the text and
      // spans the card interior to the trailing button. Enter opens natively.
      card.innerHTML =
        '<a class="res-link" href="/view/' + esc(encodePath(path)) + '" target="_blank" rel="noopener" title="Open the viewer in a new tab" aria-label="Open ' + esc(name()) + ' in a new tab">' +
        inner +
        "</a>" +
        actsHtml;
      const link = card.querySelector(".res-link");
      link.addEventListener("keydown", (ev) => {
        if (ev.key !== " ") return;
        ev.preventDefault();
        link.click();
      });
    }
    if (opts.onRemove) {
      card.querySelector(".act-x").addEventListener("click", () => opts.onRemove());
    }
    if (opts.onCopy) {
      card.querySelector(".act-copy").addEventListener("click", () => opts.onCopy());
    }
    const img = card.querySelector("img");
    if (img) {
      // A done image that cannot load lost its PNG. Shimmer again and let
      // the probe say whether the file itself is gone.
      img.addEventListener(
        "error",
        () => {
          if (fact && fact.thumb) fact = { ...fact, thumb: { ...fact.thumb, status: "pending" } };
          draw();
          startProbe();
        },
        { once: true },
      );
    }
    emitFact();
  }

  function stopProbe() {
    if (probeTimer) {
      clearInterval(probeTimer);
      probeTimer = null;
    }
  }

  // Every 2 seconds it checks for the PNG. 200 lands it, 404 means the
  // file is gone, anything else keeps shimmering.
  function startProbe() {
    stopProbe();
    if (stopped || gone) return;
    const tick = async () => {
      if (stopped || gone) {
        stopProbe();
        return;
      }
      let res = null;
      try {
        res = await fetch(thumbUrl(), { headers: { Accept: "image/png" } });
      } catch {
        return; // The server is unreachable. Keep shimmering and try again.
      }
      if (stopped || gone) {
        stopProbe();
        return;
      }
      if (res.status === 200) {
        fact = {
          name: name(),
          path,
          size: fact ? fact.size : null,
          kind: fact ? fact.kind : null,
          thumb: { status: "done", url: thumbUrl() },
        };
        stopProbe();
        draw();
      } else if (res.status === 404) {
        gone = true;
        stopProbe();
        draw();
      }
    };
    probeTimer = setInterval(tick, _PROBE_MS);
  }

  function pending() {
    const s = fact && fact.thumb ? fact.thumb.status : "none";
    return s === "pending" || s === "generating";
  }

  draw();
  if (typeof source === "string") {
    // A bare path finishes its subline and slot from the parent listing.
    // A 404 parent or a missing entry means the file is gone. Anything else
    // keeps the path-derived content as is.
    _factOf(path).then(
      (entry) => {
        if (stopped || gone) return;
        if (entry) {
          fact = entry;
          draw();
          if (pending()) startProbe();
        } else {
          gone = true;
          draw();
        }
      },
      (err) => {
        if (stopped || gone || (err && err.status !== 404)) return;
        gone = true;
        draw();
      },
    );
  } else if (pending()) {
    startProbe();
  }

  return {
    el: card,
    stop() {
      stopped = true;
      stopProbe();
    },
  };
}
