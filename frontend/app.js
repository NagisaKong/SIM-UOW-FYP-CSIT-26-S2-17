// Shared front-end utilities.
// Point this at your running FastAPI host:
const API_BASE = "http://127.0.0.1:8000";

function authHeader() {
  const token = localStorage.getItem("token");
  return token ? {"Authorization": "Bearer " + token} : {};
}

async function api(path, opts = {}) {
  const res = await fetch(API_BASE + path, {
    ...opts,
    headers: {...(opts.headers || {}), ...authHeader()},
  });
  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch { data = {raw: text}; }
  if (!res.ok) {
    const msg = data.detail || data.message || `HTTP ${res.status}`;
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return data;
}

function requireAuth(expectedRole) {
  const token = localStorage.getItem("token");
  const user = JSON.parse(localStorage.getItem("user") || "null");
  if (!token || !user) { location.href = "index.html"; return null; }
  if (expectedRole && user.role !== expectedRole) {
    alert("权限不足: 需要 " + expectedRole);
    location.href = "index.html";
    return null;
  }
  return user;
}

function logout() {
  localStorage.removeItem("token");
  localStorage.removeItem("user");
  location.href = "index.html";
}

function fmt(ts) {
  if (!ts) return "-";
  return new Date(ts).toLocaleString();
}

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of children) {
    if (c == null) continue;
    e.append(c instanceof Node ? c : document.createTextNode(c));
  }
  return e;
}
