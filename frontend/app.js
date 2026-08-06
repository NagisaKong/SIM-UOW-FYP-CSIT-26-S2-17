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
  return token ? { "Authorization": "Bearer " + token } : {};
}

async function api(path, opts = {}) {
  let res;
  try {
    res = await fetch(API_BASE + path, {
      ...opts,
      headers: { ...(opts.headers || {}), ...authHeader() },
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
  try { data = text ? JSON.parse(text) : {}; } catch { data = { raw: text }; }
  if (!res.ok) {
    const msg = data.detail || data.message || `HTTP ${res.status}`;
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return data;
}

// Guards a dashboard page: redirects to the login page unless a valid
// session for `expectedRole` is present. `alertOnMismatch` controls whether a
// role mismatch pops an alert — suppressed on the bfcache re-check below,
// since that fires mid-navigation (e.g. while the browser is restoring a
// cached page for Back) and an alert there reads as a random interruption
// rather than feedback on something the user just did.
function _checkAuth(expectedRole, alertOnMismatch) {
  const token = localStorage.getItem("token");
  const user = JSON.parse(localStorage.getItem("user") || "null");
  if (!token || !user) { location.replace("index.html"); return null; }
  if (expectedRole && user.role !== expectedRole) {
    if (alertOnMismatch) {
      alert("Insufficient permissions: " + expectedRole + " role required");
    }
    location.replace("index.html");
    return null;
  }
  return user;
}

function requireAuth(expectedRole) {
  // Browsers restore a page from the back-forward cache (bfcache) on Back /
  // Forward without re-running this script, so the check below only runs
  // once, at the initial load. That is exactly what breaks logout: the
  // dashboard is still sitting in bfcache with its last-rendered DOM, and
  // clicking Back after logout would show it as-is even though localStorage
  // no longer holds a session.
  //
  // `pageshow` fires on both a normal load AND a bfcache restore, and only
  // the latter sets `event.persisted`, so re-running the same check there
  // catches exactly the case that needs it. The listener is attached once,
  // during this normal call, and — because it lives on an object bfcache
  // keeps alive rather than in the script body — survives to fire again when
  // the page is restored, even though the script itself does not re-execute.
  window.addEventListener("pageshow", (e) => {
    if (e.persisted) _checkAuth(expectedRole, false);
  });
  return _checkAuth(expectedRole, true);
}

function logout() {
  localStorage.removeItem("token");
  localStorage.removeItem("user");
  // replace(), not href=, so the dashboard is dropped from history instead
  // of sitting one Forward-click away from a page pageshow will immediately
  // bounce out of anyway — no functional difference, just avoids a redirect
  // the user would visibly see happen if they went Forward.
  location.replace("index.html");
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
async function renderSupportingDocument({ url, hasDocument, name, type, ids }) {
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
    const res = await fetch(API_BASE + url, { headers: authHeader() });
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

  const select = el("select", { class: "tabs-select", "aria-label": "Navigation" });
  for (const b of buttons) {
    const opt = el("option", { value: b.dataset.tab }, b.textContent.trim());
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
// The marks next to each name are each vendor's own published logo (Google
// Chrome, Microsoft Edge, Mozilla Firefox), vendored under assets/browser/ so
// the pages keep working offline. They are used whole and unrecoloured, which
// is what the vendors' brand guidelines ask for — earlier revisions of this
// file approximated the shapes by hand and read as the wrong browser at 15px.
const _BROWSER_MARKS = {
  chrome: "assets/browser/chrome.svg",
  edge: "assets/browser/edge.svg",
  firefox: "assets/browser/firefox.svg",
};

function _browserChip(name, key) {
  const span = el("span", { class: "browser-chip" });
  const mark = el("img", {
    class: "browser-mark", src: _BROWSER_MARKS[key], alt: "",
    "aria-hidden": "true",
  });
  span.append(mark, document.createTextNode(name));
  return span;
}

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
    _browserChip("Google Chrome", "chrome"),
    document.createTextNode(", "),
    _browserChip("Microsoft Edge", "edge"),
    document.createTextNode(" and "),
    _browserChip("Mozilla Firefox", "firefox"),
    document.createTextNode(
      " (desktop) are the recommended browsers for this system. "),
  ];
  if (_BROWSER_NOTE_BY_ROLE[role]) {
    bits.push(document.createTextNode(_BROWSER_NOTE_BY_ROLE[role] + " "));
  }
  host.append(el("p", { class: "small" }, ...bits));

  // Worth stating explicitly: this is the one failure that looks like a
  // broken feature rather than a configuration mistake. Browsers block
  // getUserMedia outside a secure context, so opening the site by LAN IP over
  // plain http gives a camera that simply never starts, with no obvious cause.
  if (role === "teacher" || role === "student") {
    const secure = window.isSecureContext;
    const note = el("p", { class: "small" },
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

// Floating "back to top" control. Injected rather than added to each page so
// there is one implementation, and hidden until there is actually something
// to scroll back from — a button that does nothing on a short page is noise.
function setupBackToTop() {
  if (document.querySelector(".to-top")) return;

  const btn = el("button", {
    type: "button", class: "to-top", "aria-label": "Back to top", title: "Back to top",
  });
  btn.innerHTML =
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" ' +
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<path d="M12 19V5"/><path d="M5 12l7-7 7 7"/></svg>';
  btn.addEventListener("click", () => {
    // `smooth` is ignored when the user has asked for reduced motion, which
    // is the correct behaviour rather than something to work around.
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
  document.body.append(btn);

  const SHOW_AFTER = 320;   // px — roughly one screenful on a laptop
  let shown = false;
  const sync = () => {
    const should = window.scrollY > SHOW_AFTER;
    if (should === shown) return;      // avoid touching the DOM every scroll tick
    shown = should;
    btn.classList.toggle("is-visible", should);
  };
  window.addEventListener("scroll", sync, { passive: true });
  sync();
}

// Small copyright line at the very bottom of the dashboards. Self-injected,
// like the back-to-top button, so there is one implementation instead of the
// same text pasted into every HTML file — and a year update only ever needs
// to happen here.
const _COPYRIGHT_YEAR = 2026;
const _COPYRIGHT_TEAM = "FYP-26-S2-17 Team";

function setupCopyrightFooter() {
  if (document.querySelector(".app-footer")) return;
  // The login page centres its single card in a flex body, so an appended
  // footer lands beside the card rather than under it. It is also the one
  // page with nothing above it to attribute — so it gets no footer at all.
  if (document.body.classList.contains("centered")) return;
  document.body.append(
    el("footer", { class: "app-footer" },
      el("p", { class: "small" },
        `© ${_COPYRIGHT_YEAR} ${_COPYRIGHT_TEAM}. All rights reserved.`),
    ),
  );
}

const PAGE_LEAVE_MS = 200;

function navigateWithFade(url, node) {
  const target = node || document.querySelector(".card");
  const reduce = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (!target || reduce) {
    location.href = url;
    return;
  }

  let done = false;
  const go = () => {
    if (done) return;
    done = true;
    target.removeEventListener("animationend", onEnd);
    location.href = url;
  };
  const onEnd = (e) => {
    if (e.animationName === "page-leave") go();
  };

  target.addEventListener("animationend", onEnd);
  setTimeout(go, PAGE_LEAVE_MS + 60);
  target.classList.add("is-leaving");
}

function _initSharedUi() {
  setupMobileTabs();
  renderBrowserNotice();
  setupBackToTop();
  setupCopyrightFooter();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", _initSharedUi);
} else {
  _initSharedUi();
}
