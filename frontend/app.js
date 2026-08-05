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

// Compact form for <option> labels: no seconds, two-digit year.
// A <select> shows its selected label inside a fixed-width control, and on a
// narrow window that box can fall below 300px — at which point the full
// fmt() string ("8/3/2026, 1:30:00 PM") gets cut off mid-time and two
// sittings of the same course become indistinguishable. Every character
// counts there, so this trims the ~5 that make the difference.
// Tables and reports keep using fmt(): they have the room, and the seconds
// are occasionally useful.
function fmtShort(ts) {
  if (!ts) return "-";
  return new Date(ts).toLocaleString("en-US", {
    year: "2-digit", month: "numeric", day: "numeric",
    hour: "numeric", minute: "2-digit",
  });
}

// Mirror the selected option's text into the control's tooltip, and keep it
// in sync. Shortening the labels buys enough room down to roughly a 330px
// window; below that a <select> still truncates its closed state and there
// is no CSS for it, so hovering has to be able to reveal the rest.
function syncSelectTitle(sel) {
  if (!sel) return;
  const apply = () => {
    const opt = sel.selectedOptions && sel.selectedOptions[0];
    sel.title = opt ? opt.textContent : "";
  };
  apply();
  if (!sel.dataset.titleSynced) {
    sel.addEventListener("change", apply);
    sel.dataset.titleSynced = "1";
  }
}

// Render a student-submitted supporting document (U08 appeals, U28 leave)
// into a detail panel. Used by both the teacher and admin consoles, which
// show the same evidence in the same shape.
//
// The document endpoints require the bearer token, so the bytes cannot be
// pointed at from an <img src> — they are fetched and handed over as an
// object URL (the same approach as the CSV report export). That URL is
// returned so the caller can revoke it when the panel closes; object URLs
// live until revoked, and leaking one per opened record adds up.
async function renderSupportingDocument({url, hasDocument, name, type, ids}) {
  const section = document.getElementById(ids.section);
  const img = document.getElementById(ids.image);
  const link = document.getElementById(ids.link);
  const msg = document.getElementById(ids.msg);
  if (!section) return null;
  img.style.display = "none";
  link.style.display = "none";
  img.removeAttribute("src");
  if (!hasDocument) {
    section.style.display = "none";
    return null;
  }
  section.style.display = "";
  msg.style.color = "";
  msg.textContent = "Loading document…";
  try {
    const res = await fetch(API_BASE + url, {headers: authHeader()});
    if (!res.ok) throw new Error(await res.text() || `HTTP ${res.status}`);
    const blob = await res.blob();
    const objectUrl = URL.createObjectURL(blob);
    const label = name || "document";
    const isImage = (type || blob.type || "").startsWith("image/");
    // Images additionally get an inline preview, but the link is shown for
    // every type: a preview alone left the reviewer with no way to open the
    // evidence full-size or keep a copy for the record.
    if (isImage) {
      img.src = objectUrl;
      img.style.display = "";
      // An image is already on screen, so the link's job is to save a copy.
      link.download = label;
      link.textContent = `Download ${label}`;
    } else {
      // A PDF cannot be previewed here, so the link opens it for reading
      // instead — forcing a download would be the wrong default.
      link.removeAttribute("download");
      link.textContent = `Open ${label} in a new tab`;
    }
    link.href = objectUrl;
    link.style.display = "";
    msg.textContent = isImage ? label : "";
    return objectUrl;
  } catch (ex) {
    msg.style.color = "#c0392b";
    msg.textContent = "Could not load the document: " + ex.message;
    return null;
  }
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

// Browser support notice, rendered into `<div class="browser-notice"
// data-role="…">` at the foot of each dashboard.
//
// The requirement differs by role, so the text does too rather than repeating
// one generic warning everywhere: the admin console touches no camera at all,
// student face registration opens one, and the teacher's classroom scan opens
// several concurrently — which is the case browsers differ most on.
const _BROWSER_NOTE_BY_ROLE = {
  teacher:
    "Starting a classroom scan opens every selected camera at the same time. " +
    "Chrome, Edge and Firefox handle concurrent camera streams reliably; " +
    "Safari has not been verified with this system and is known to be " +
    "restrictive here, so avoid it for scanning.",
  student:
    "Registering or updating your face photo uses your device camera, which " +
    "these browsers support without extra setup.",
  admin:
    "The admin console needs no camera, so any of these browsers is fine. " +
    "Use one of them when demonstrating the teacher scan page as well.",
};

function renderBrowserNotice() {
  const host = document.querySelector(".browser-notice");
  if (!host || host.dataset.rendered) return;
  host.dataset.rendered = "1";

  const role = host.dataset.role || "";
  const bits = [
    el("strong", {}, "Recommended browsers: "),
    document.createTextNode(
      "Google Chrome, Microsoft Edge and Mozilla Firefox (desktop) are the " +
      "recommended browsers for this system. "),
  ];
  if (_BROWSER_NOTE_BY_ROLE[role]) {
    bits.push(document.createTextNode(_BROWSER_NOTE_BY_ROLE[role] + " "));
  }
  host.append(el("p", {class: "small"}, ...bits));

  // Worth stating explicitly: this is the one failure that looks like a
  // broken feature rather than a configuration mistake. Browsers block
  // getUserMedia outside a secure context, so opening the site by LAN IP over
  // plain http gives a camera that simply never starts, with no obvious cause.
  if (role === "teacher" || role === "student") {
    const secure = window.isSecureContext;
    const note = el("p", {class: "small"},
      el("strong", {}, secure ? "Camera access: " : "Camera unavailable here: "),
      document.createTextNode(
        secure
          ? "this page is on a secure origin, so cameras can be used. Allow " +
            "access when the browser asks — the permission is per site."
          : `this page was opened over an insecure origin (${location.protocol}//` +
            `${location.host}). Browsers only grant camera access over HTTPS ` +
            "or on localhost, so scanning and face registration will not " +
            "start here. Use the deployed HTTPS address, or run the frontend " +
            "from localhost."),
    );
    if (!secure) note.classList.add("browser-notice-warn");
    host.append(note);
  }
}

function _initSharedUi() {
  setupMobileTabs();
  renderBrowserNotice();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", _initSharedUi);
} else {
  _initSharedUi();
}
