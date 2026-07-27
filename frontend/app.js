// Shared front-end utilities.
//
// API base resolution (first match wins):
//   1. ?api=http://host:port — a one-off override, remembered for later visits.
//      Use ?api=default (or ?api=) to forget it again.
//   2. localStorage "apiBase" — a previously saved override. This is how the
//      hosted UI is pointed at a local GPU machine for behaviour analysis.
//   3. Local development: a page opened from localhost / 127.0.0.1 / file://
//      always talks to a backend on the SAME machine. config.js is skipped on
//      purpose — the deployed backend's ALLOWED_ORIGINS does not include
//      localhost, so pointing there would only produce CORS failures.
//   4. window.API_BASE from config.js — the deployment default for the hosted
//      site. Edit that one file; do not hard-code URLs anywhere else.
//   5. Fallback: same protocol + hostname as the page, on window.API_PORT
//      (default 8000).
const API_BASE = (() => {
  const strip = (u) => String(u).trim().replace(/\/+$/, "");

  const params = new URLSearchParams(location.search);
  if (params.has("api")) {
    const q = strip(params.get("api"));
    if (!q || q === "default" || q === "reset") localStorage.removeItem("apiBase");
    else localStorage.setItem("apiBase", q);
  }

  const saved = localStorage.getItem("apiBase");
  if (saved) return strip(saved);

  const host = location.hostname;
  const isLocal =
    location.protocol === "file:" ||
    !host ||
    ["localhost", "127.0.0.1", "0.0.0.0", "[::1]", "::1"].includes(host);

  if (!isLocal && window.API_BASE) return strip(window.API_BASE);

  const port = window.API_PORT || "8000";
  const proto = location.protocol === "file:" ? "http:" : location.protocol;
  return `${proto}//${host || "127.0.0.1"}:${port}`;
})();

function authHeader() {
  const token = localStorage.getItem("token");
  return token ? {"Authorization": "Bearer " + token} : {};
}

async function api(path, opts = {}) {
  let res;
  try {
    res = await fetch(API_BASE + path, {
      ...opts,
      headers: {...(opts.headers || {}), ...authHeader()},
    });
  } catch (networkError) {
    // fetch() only rejects when the request never got a response: the API is
    // down, the URL is wrong, or CORS blocked it. Name the address we tried,
    // otherwise every page just shows an opaque "Failed to fetch".
    throw new Error(
      `Cannot reach the API at ${API_BASE}. Check that the backend is running ` +
      `and that this site's address is listed in its ALLOWED_ORIGINS. ` +
      `(You can override the address with ?api=https://your-backend)`
    );
  }
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
