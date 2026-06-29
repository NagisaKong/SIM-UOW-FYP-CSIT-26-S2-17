// Shared front-end utilities.
//
// API base resolution (first match wins):
//   1. ?api=http://host:port in the URL — saved to localStorage for later visits.
//   2. localStorage "apiBase" — a previously saved override.
//   3. window.API_BASE — set it via an inline <script> before app.js if you want
//      a hard-coded value.
//   4. Auto: same protocol + hostname as the current page, on the API port below
//      (default 8000, override with window.API_PORT). Falls back to 127.0.0.1
//      when opened directly from disk (file://).
const API_BASE = (() => {
  const fromQuery = new URLSearchParams(location.search).get("api");
  if (fromQuery) localStorage.setItem("apiBase", fromQuery.replace(/\/+$/, ""));

  const override = localStorage.getItem("apiBase") || window.API_BASE;
  if (override) return override.replace(/\/+$/, "");

  const port = window.API_PORT || "8000";
  const host = location.hostname || "127.0.0.1";
  const proto = location.protocol === "file:" ? "http:" : location.protocol;
  return `${proto}//${host}:${port}`;
})();

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
    alert("Insufficient permissions: " + expectedRole + " role required");
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
  return new Date(ts).toLocaleString("en-US");
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

// On phones the horizontal tab bar is cramped; mirror it into a <select>
// dropdown (shown only on narrow screens via CSS). Selecting an option just
// clicks the matching tab button, so each page's existing tab logic runs
// unchanged. Stays in sync when a tab is activated by other means.
function setupMobileTabs() {
  const tabs = document.querySelector(".tabs");
  if (!tabs || tabs.nextElementSibling?.classList.contains("tabs-select")) return;
  const buttons = [...tabs.querySelectorAll(".tab-btn")];
  if (!buttons.length) return;

  const select = el("select", {class: "tabs-select", "aria-label": "Navigation"});
  for (const b of buttons) {
    const opt = el("option", {value: b.dataset.tab}, b.textContent.trim());
    if (b.classList.contains("active")) opt.selected = true;
    select.append(opt);
  }
  select.addEventListener("change", () => {
    const b = buttons.find(x => x.dataset.tab === select.value);
    if (b) b.click();
  });
  buttons.forEach(b => b.addEventListener("click", () => { select.value = b.dataset.tab; }));
  tabs.insertAdjacentElement("afterend", select);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", setupMobileTabs);
} else {
  setupMobileTabs();
}
