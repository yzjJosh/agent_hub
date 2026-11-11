export class ApiError extends Error {
  constructor(status, code, message, raw) {
    super(message);
    this.status = status;
    this.code = code;
    this.raw = raw; // The error body the server sent.
  }
}

async function _parse(res) {
  if (res.status === 304) return null;
  let body = null;
  try { body = await res.json(); } catch { body = null; }
  if (!res.ok) {
    throw new ApiError(res.status, (body && body.code) || "internal", (body && body.error) || res.statusText, body);
  }
  return body;
}

export async function getJSON(url) {
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  return _parse(res);
}

export async function fetchChanges(fingerprint, generatedAt) {
  const q = new URLSearchParams();
  if (fingerprint) q.set("since_fp", fingerprint);
  if (generatedAt) q.set("since_gen", generatedAt);
  const qs = q.toString();
  const res = await fetch("/api/changes" + (qs ? "?" + qs : ""), { headers: { Accept: "application/json" } });
  return _parse(res);
}

export async function postJSON(url) {
  const res = await fetch(url, { method: "POST", headers: { Accept: "application/json" } });
  return _parse(res);
}

export async function postJSONBody(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return _parse(res);
}

export async function postRaw(url, bytes) {
  const res = await fetch(url, { method: "POST", headers: { Accept: "application/json" }, body: bytes });
  return _parse(res);
}

export function encodePath(path) {
  return path.split("/").map(encodeURIComponent).join("/");
}

export function davHref(path) {
  return "/dav/" + encodePath(path);
}

export function dlHref(path) {
  return "/dav/dl?path=" + encodePath(path);
}
