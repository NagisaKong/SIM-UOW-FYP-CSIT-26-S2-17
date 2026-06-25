const user = requireAuth("teacher");
document.getElementById("who").textContent = `${user.full_name || user.email} (teacher)`;

// ── Tabs ──────────────────────────────────────────────────────
const TABS = ["sessions", "live", "roster", "analytics", "reports", "leave"];
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    TABS.forEach(name => {
      document.getElementById("tab-" + name).style.display =
        name === btn.dataset.tab ? "" : "none";
    });
    if (btn.dataset.tab === "live") loadLiveSessions();
    if (btn.dataset.tab === "roster") loadRoster();
    if (btn.dataset.tab === "analytics") loadAnalytics();
    if (btn.dataset.tab === "leave") loadTeacherLeave();
  });
});

// ── Shared state ─────────────────────────────────────────────
let coursesCache = [];

async function loadCourses() {
  const res = await api("/teacher/courses");
  coursesCache = res.courses || [];
  const selectors = ["sess-course", "roster-course", "ana-course", "report-course"];
  for (const id of selectors) {
    const sel = document.getElementById(id);
    if (!sel) continue;
    const keepBlank = id === "report-course";
    const prev = sel.value;
    sel.innerHTML = keepBlank ? '<option value="">All courses</option>' : "";
    for (const c of coursesCache) {
      const o = document.createElement("option");
      o.value = c.courseid;
      o.textContent = `${c.course_code} — ${c.course_name}`;
      sel.appendChild(o);
    }
    if (prev) sel.value = prev;
  }
}

// ──────────────────────────────────────────────────────────────
// U15 Sessions tab
// ──────────────────────────────────────────────────────────────
async function loadSessions() {
  const courseId = document.getElementById("sess-course").value;
  const status = document.getElementById("sess-status").value;
  const params = new URLSearchParams();
  if (courseId) params.set("course_id", courseId);
  if (status) params.set("status", status);
  const res = await api("/teacher/sessions?" + params.toString());
  const tbody = document.getElementById("sess-body");
  tbody.innerHTML = "";
  for (const s of res.sessions || []) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${s.attendancesessionid}</td>
      <td>${s.course_code} — ${s.course_name}</td>
      <td>${fmt(s.start_time)}</td>
      <td>${fmt(s.end_time)}</td>
      <td><span class="badge">${s.status}</span></td>
      <td></td>
    `;
    const actions = tr.lastElementChild;
    if (s.status === "scheduled") {
      const b = document.createElement("button");
      b.textContent = "Start";
      b.onclick = () => startSession(s.attendancesessionid);
      actions.appendChild(b);
    } else if (s.status === "active") {
      const b = document.createElement("button");
      b.className = "danger";
      b.textContent = "End";
      b.onclick = () => endSession(s.attendancesessionid);
      actions.appendChild(b);
    } else if (s.status === "ended") {
      const b = document.createElement("button");
      b.className = "secondary";
      b.textContent = "Resend emails";
      b.onclick = () => resendNotifications(s.attendancesessionid);
      actions.appendChild(b);
    }
    tbody.appendChild(tr);
  }
  document.getElementById("sess-msg").textContent =
    `${(res.sessions || []).length} sessions`;
}

async function startSession(id) {
  try {
    const res = await api(`/teacher/sessions/${id}/start`, {method: "POST"});
    document.getElementById("sess-msg").textContent = `Session ${res.session_id} started`;
    loadSessions();
  } catch (e) {
    document.getElementById("sess-msg").textContent = "Error: " + e.message;
  }
}

async function endSession(id) {
  if (!confirm("End this session? Unmarked students will be set to absent and emailed.")) return;
  try {
    const res = await api(`/teacher/sessions/${id}/end`, {method: "POST"});
    document.getElementById("sess-msg").textContent =
      `Session ${res.session_id} ended (${res.marked_absent} marked absent, ${res.notifications_queued} notification emails queued)`;
    loadSessions();
  } catch (e) {
    document.getElementById("sess-msg").textContent = "Error: " + e.message;
  }
}

async function resendNotifications(id) {
  try {
    const res = await api(`/teacher/sessions/${id}/notify`, {method: "POST"});
    document.getElementById("sess-msg").textContent =
      `Re-queued ${res.queued} notification emails for session ${id}`;
  } catch (e) {
    document.getElementById("sess-msg").textContent = "Error: " + e.message;
  }
}

document.getElementById("sess-refresh").addEventListener("click", loadSessions);
document.getElementById("sess-course").addEventListener("change", loadSessions);
document.getElementById("sess-status").addEventListener("change", loadSessions);

// ──────────────────────────────────────────────────────────────
// U12 Live tab
// ──────────────────────────────────────────────────────────────

async function loadLiveSessions() {
  const res = await api("/teacher/sessions?status=active");
  const sel = document.getElementById("live-session");
  const prev = sel.value;
  sel.innerHTML = '<option value="">— select a session —</option>';
  for (const s of res.sessions || []) {
    const o = document.createElement("option");
    o.value = s.attendancesessionid;
    o.textContent = `#${s.attendancesessionid} ${s.course_code} (${fmt(s.start_time)})`;
    sel.appendChild(o);
  }
  if (prev) sel.value = prev;
  if (sel.value) refreshLive();
}

async function refreshLive() {
  const id = document.getElementById("live-session").value;
  const tbody = document.getElementById("live-body");
  if (!id) {
    tbody.innerHTML = "";
    document.getElementById("earlyleft-body").innerHTML = "";
    return;
  }
  loadEarlyLeft(id);
  try {
    const res = await api(`/teacher/sessions/${id}/live`);
    tbody.innerHTML = "";
    for (const r of res.roster || []) {
      const status = r.attendance_status || "absent";
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${r.student_id || "-"}</td>
        <td>${r.full_name || "-"}</td>
        <td><span class="status-${status}">${status}</span></td>
        <td>${fmt(r.marked_at)}</td>
      `;
      tbody.appendChild(tr);
    }
    const sm = res.summary || {};
    document.getElementById("live-summary").textContent =
      `Present ${sm.present || 0} · Late ${sm.late || 0} · Left early ${sm.early_left || 0} · ` +
      `Absent ${sm.absent || 0} · No record ${sm.no_record || 0}`;
  } catch (e) {
    document.getElementById("live-summary").textContent = "Error: " + e.message;
  }
}

// U16: early departure summary for the selected session
async function loadEarlyLeft(id) {
  const body = document.getElementById("earlyleft-body");
  const msg = document.getElementById("earlyleft-msg");
  try {
    const res = await api(`/teacher/sessions/${id}/early-left`);
    const rows = res.early_left || [];
    // Build the new content first, then swap it in once — avoids the
    // clear-then-fetch flicker on the 5s auto-refresh.
    msg.style.color = "";
    msg.textContent = rows.length ? "" : "No early departures flagged for this session.";
    body.innerHTML = rows.map(r => `
      <tr>
        <td>${r.student_id || "-"}</td>
        <td>${r.full_name || "-"}</td>
        <td><span class="status-${r.attendance_status}">${r.attendance_status}</span></td>
        <td>${fmt(r.marked_at)}</td>
        <td>${r.last_seen ? fmt(r.last_seen) : "—"}</td>
      </tr>
    `).join("");
  } catch (e) {
    msg.style.color = "#c0392b";
    msg.textContent = e.message;
  }
}

document.getElementById("live-session").addEventListener("change", refreshLive);
document.getElementById("live-refresh").addEventListener("click", refreshLive);

// ──────────────────────────────────────────────────────────────
// U03 Classroom camera scan (teacher-activated detection windows)
// ──────────────────────────────────────────────────────────────
const scanCamsHost = document.getElementById("scan-cams");
const scanCamsEmpty = document.getElementById("scan-cams-empty");
const scanCamList = document.getElementById("scan-cam-list");
const scanCamRefresh = document.getElementById("scan-cam-refresh");
const scanCanvas = document.getElementById("scan-canvas");
const scanStart = document.getElementById("scan-start");
const scanCapture = document.getElementById("scan-capture");
const scanFinalize = document.getElementById("scan-finalize");
const scanStop = document.getElementById("scan-stop");
const scanAuto = document.getElementById("scan-auto");
const scanInterval = document.getElementById("scan-interval");
const scanCountdownToggle = document.getElementById("scan-countdown-toggle");
const scanCountdownEl = document.getElementById("scan-countdown");
const scanMsg = document.getElementById("scan-msg");
// One entry per opened camera: {deviceId, stream, video, label}.
let scanCams = [];
let scanTimer = null;
let scanBusy = false;
let scanCountdown = 0;

// Default the auto-snapshot interval to the admin-configured detection
// interval (U03/U34). Falls back to the input's existing value on error.
async function loadScanConfig() {
  try {
    const cfg = await api("/config/attendance");
    if (cfg && cfg.detection_interval_seconds) {
      scanInterval.value = cfg.detection_interval_seconds;
    }
  } catch { /* keep the default in the input */ }
}
loadScanConfig();

function currentSessionId() {
  return document.getElementById("live-session").value;
}

function stopScanCamera() {
  if (scanTimer) { clearInterval(scanTimer); scanTimer = null; }
  for (const cam of scanCams) cam.stream.getTracks().forEach(t => t.stop());
  scanCams = [];
  scanCamsHost.innerHTML = "";
  scanCamsHost.append(scanCamsEmpty);
  scanCamsEmpty.style.display = "";
  scanStart.disabled = false;
  scanCapture.disabled = true;
  scanFinalize.disabled = true;
  scanStop.disabled = true;
  updateCountdownDisplay();
}

// Cameras whose label looks like a virtual/streaming device — unchecked by
// default so e.g. OBS Virtual Camera isn't used for attendance unless asked.
const VIRTUAL_CAM_RE = /\b(obs|virtual|snap|manycam|xsplit|droidcam|ndi)\b/i;

// Probe for camera permission (so labels are exposed), then render a checkbox
// per connected camera. Returns the number of cameras found.
async function listScanCameras() {
  let probe = null;
  try {
    probe = await navigator.mediaDevices.getUserMedia({video: true});
  } catch (e) {
    scanCamList.innerHTML = "";
    scanCamList.append(el("p", {class: "error small", style: "margin:0"},
      "Camera permission denied or no camera found."));
    return 0;
  }
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cams = devices.filter(d => d.kind === "videoinput");
  probe.getTracks().forEach(t => t.stop());

  scanCamList.innerHTML = "";
  if (!cams.length) {
    scanCamList.append(el("p", {class: "muted small", style: "margin:0"}, "No cameras detected."));
    return 0;
  }
  cams.forEach((cam, i) => {
    const label = cam.label || `Camera ${i + 1}`;
    const isVirtual = VIRTUAL_CAM_RE.test(label);
    const cb = el("input", {type: "checkbox", value: cam.deviceId});
    cb.checked = !isVirtual; // default: real cameras on, virtual ones off
    const row = el("label", {class: "scan-cam-pick"},
      cb, el("span", {}, label),
      isVirtual ? el("span", {class: "tag"}, "virtual") : null);
    scanCamList.append(row);
  });
  return cams.length;
}

// deviceIds of the cameras the teacher has ticked.
function selectedScanCameraIds() {
  return Array.from(scanCamList.querySelectorAll('input[type="checkbox"]:checked'))
    .map(cb => cb.value);
}

// Open only the given cameras and show a live tile for each. Returns the
// number successfully opened.
async function openSelectedScanCameras(deviceIds) {
  scanCamsHost.innerHTML = "";
  scanCams = [];
  for (let i = 0; i < deviceIds.length; i++) {
    const deviceId = deviceIds[i];
    // Recover a friendly label from the picker checkbox, if present.
    const cb = scanCamList.querySelector(`input[value="${CSS.escape(deviceId)}"]`);
    const label = (cb && cb.parentElement.querySelector("span")?.textContent) || `Camera ${i + 1}`;
    try {
      // Request full HD so distant faces keep enough pixels to be detected;
      // `ideal` lets the browser fall back if the camera can't do 1080p.
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {deviceId: {exact: deviceId}, width: {ideal: 1920}, height: {ideal: 1080}},
      });
      const video = el("video", {autoplay: "", playsinline: "", class: "scan-cam-video"});
      video.muted = true;
      video.srcObject = stream;
      const tile = el("div", {class: "scan-cam-tile"},
        video, el("span", {class: "scan-cam-label"}, label));
      scanCamsHost.append(tile);
      scanCams.push({deviceId, stream, video, label});
    } catch (e) {
      // A camera may refuse to open (busy, shared USB bandwidth, etc.) —
      // skip it but keep opening the rest.
      console.warn(`Could not open ${label}:`, e);
    }
  }
  return scanCams.length;
}

scanCamRefresh.addEventListener("click", async () => {
  scanCamRefresh.disabled = true;
  scanMsg.style.color = "#555";
  scanMsg.textContent = "Detecting cameras…";
  try {
    const n = await listScanCameras();
    scanMsg.textContent = n ? `${n} camera${n > 1 ? "s" : ""} found — tick the ones to use.` : "";
  } finally {
    scanCamRefresh.disabled = false;
  }
});

function updateCountdownDisplay() {
  // Only show while auto-scanning is actively running and the toggle is on.
  if (scanCountdownToggle.checked && scanTimer) {
    scanCountdownEl.textContent = `Next snapshot in ${scanCountdown}s`;
  } else {
    scanCountdownEl.textContent = "";
  }
}

scanStart.addEventListener("click", async () => {
  if (!currentSessionId()) {
    scanMsg.style.color = "#c0392b";
    scanMsg.textContent = "Select an active session first.";
    return;
  }
  // If cameras haven't been detected yet, list them and let the teacher pick
  // first (so OBS / virtual cams can be excluded) before actually opening.
  if (!scanCamList.querySelector('input[type="checkbox"]')) {
    scanMsg.style.color = "#555";
    scanMsg.textContent = "Detecting cameras…";
    const n = await listScanCameras();
    if (n) {
      scanMsg.style.color = "#555";
      scanMsg.textContent = "Tick the cameras to use, then press Start Scan again.";
    }
    return;
  }
  const ids = selectedScanCameraIds();
  if (!ids.length) {
    scanMsg.style.color = "#c0392b";
    scanMsg.textContent = "Select at least one camera to use.";
    return;
  }
  scanMsg.style.color = "#555";
  scanMsg.textContent = "Opening cameras…";
  try {
    const opened = await openSelectedScanCameras(ids);
    if (opened === 0) {
      scanMsg.style.color = "#c0392b";
      scanMsg.textContent = "No camera could be opened.";
      stopScanCamera();
      return;
    }
    scanMsg.style.color = "#16a34a";
    scanMsg.textContent = `${opened} camera${opened > 1 ? "s" : ""} ready.`;
    scanStart.disabled = true;
    scanCapture.disabled = false;
    scanFinalize.disabled = false;
    scanStop.disabled = false;
    if (scanAuto.checked) startAutoSnapshots();
  } catch (e) {
    scanMsg.style.color = "#c0392b";
    scanMsg.textContent = "Unable to open camera: " + e.message;
    stopScanCamera();
  }
});

// Grab one frame from a single camera and POST it as a scan snapshot.
// Each camera's snapshot is an independent, additive presence record, so we
// simply send one request per camera and let the backend accumulate them.
async function snapshotOneCamera(cam, sessionId) {
  const v = cam.video;
  const w = v.videoWidth || 640, h = v.videoHeight || 480;
  scanCanvas.width = w; scanCanvas.height = h;
  scanCanvas.getContext("2d").drawImage(v, 0, 0, w, h);
  const blob = await new Promise(res => scanCanvas.toBlob(res, "image/jpeg", 0.92));
  const fd = new FormData();
  fd.append("file", blob, "snapshot.jpg");
  return api(`/teacher/sessions/${sessionId}/scan`, {method: "POST", body: fd});
}

async function captureSnapshot() {
  const id = currentSessionId();
  if (!scanCams.length || !id || scanBusy) return;
  scanBusy = true;
  let ok = 0, fail = 0;
  // Sequential: the capture canvas is shared, and it avoids hammering the
  // backend with N simultaneous uploads.
  for (const cam of scanCams) {
    try {
      await snapshotOneCamera(cam, id);
      ok++;
    } catch (ex) {
      fail++;
      console.warn(`Scan failed for ${cam.label}:`, ex);
    }
  }
  scanMsg.style.color = fail ? "#c0392b" : "#16a34a";
  scanMsg.textContent =
    `Captured ${scanCams.length} camera${scanCams.length > 1 ? "s" : ""}` +
    ` · ${ok} ok${fail ? `, ${fail} failed` : ""}`;
  refreshLive();
  scanBusy = false;
}

scanCapture.addEventListener("click", captureSnapshot);

function startAutoSnapshots() {
  if (scanTimer) clearInterval(scanTimer);
  const secs = Math.max(3, Number(scanInterval.value) || 15);
  scanCountdown = secs;
  updateCountdownDisplay();
  // 1-second tick drives both the countdown and the capture, so the
  // displayed number always matches when the next snapshot fires.
  scanTimer = setInterval(async () => {
    scanCountdown -= 1;
    if (scanCountdown <= 0) {
      scanCountdown = secs;
      await captureSnapshot();
    }
    updateCountdownDisplay();
  }, 1000);
}

scanAuto.addEventListener("change", () => {
  if (scanAuto.checked && scanCams.length) {
    startAutoSnapshots();
  } else if (scanTimer) {
    clearInterval(scanTimer); scanTimer = null;
    updateCountdownDisplay();
  }
});

scanCountdownToggle.addEventListener("change", updateCountdownDisplay);
// Restart the cadence if the interval is changed mid-scan.
scanInterval.addEventListener("change", () => {
  if (scanAuto.checked && scanCams.length) startAutoSnapshots();
});

scanFinalize.addEventListener("click", async () => {
  const id = currentSessionId();
  if (!id) return;
  if (!confirm("End the scan and finalize attendance for this session?")) return;
  try {
    const res = await api(`/teacher/sessions/${id}/end`, {method: "POST"});
    scanMsg.style.color = "#16a34a";
    scanMsg.textContent =
      `Finalized: Present ${res.present}, Late ${res.late}, Left early ${res.early_left}, ` +
      `Absent ${res.marked_absent} (${res.notifications_queued} emails queued)`;
    stopScanCamera();
    loadLiveSessions();
    refreshLive();
  } catch (ex) {
    scanMsg.style.color = "#c0392b";
    scanMsg.textContent = "✗ " + ex.message;
  }
});

scanStop.addEventListener("click", stopScanCamera);
window.addEventListener("pagehide", stopScanCamera);

// ──────────────────────────────────────────────────────────────
// U13 Roster tab
// ──────────────────────────────────────────────────────────────
// Toggle the roster list ↔ per-student history sub-pages (Admin-style).
function showRosterView(view) {
  document.getElementById("roster-list-view").hidden = view !== "list";
  document.getElementById("roster-detail-view").hidden = view !== "detail";
}

async function loadRoster() {
  const cid = document.getElementById("roster-course").value;
  const tbody = document.getElementById("roster-body");
  tbody.innerHTML = "";
  showRosterView("list");
  if (!cid) return;
  const res = await api(`/teacher/courses/${cid}/students`);
  for (const s of res.students || []) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${s.student_id || "-"}</td>
      <td>${s.full_name || "-"}</td>
      <td>${s.email}</td>
      <td>${s.attended} / ${s.sessions_completed}</td>
      <td><button class="secondary">View history</button></td>
    `;
    tr.querySelector("button").onclick = () => viewStudentHistory(s.accountid, s.full_name || s.email, cid);
    tbody.appendChild(tr);
  }
}

async function viewStudentHistory(accountId, name, courseId) {
  const res = await api(`/teacher/students/${accountId}/attendance?course_id=${courseId}`);
  showRosterView("detail");
  document.getElementById("student-history-title").textContent = `Attendance History — ${name}`;
  const sm = res.summary || {};
  document.getElementById("student-history-summary").textContent =
    `Rate ${res.rate}% · Present ${sm.present || 0} · Late ${sm.late || 0} · ` +
    `Left early ${sm.early_left || 0} · Absent ${sm.absent || 0} · Leave ${sm.leave || 0} · Total ${res.total}`;
  const tbody = document.getElementById("student-history-body");
  tbody.innerHTML = "";
  for (const r of res.records || []) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.course_code} — ${r.course_name}</td>
      <td>${fmt(r.start_time)}</td>
      <td><span class="status-${r.status}">${r.status}</span></td>
      <td>${fmt(r.marked_at)}</td>
    `;
    tbody.appendChild(tr);
  }
}

document.getElementById("roster-course").addEventListener("change", loadRoster);
document.getElementById("back-roster").addEventListener("click", () => showRosterView("list"));

// ──────────────────────────────────────────────────────────────
// U30 Analytics tab
// ──────────────────────────────────────────────────────────────
let trendChart = null, breakdownChart = null;

async function loadAnalytics() {
  const cid = document.getElementById("ana-course").value;
  if (!cid) return;
  const params = new URLSearchParams();
  const from = document.getElementById("ana-from").value;
  const to = document.getElementById("ana-to").value;
  if (from) params.set("date_from", from);
  if (to) params.set("date_to", to);
  const res = await api(`/teacher/courses/${cid}/analytics?${params.toString()}`);
  renderTrend(res.trend || []);
  renderBreakdown(res.breakdown || {});
  const b = res.breakdown || {};
  document.getElementById("ana-summary").textContent =
    `Total records ${b.total || 0} · Rate ${b.rate || 0}% · Present ${b.present || 0} · Late ${b.late || 0} · Absent ${b.absent || 0}`;
}

// Format a YYYY-MM-DD date string without timezone shifting (avoids the
// new Date("2026-06-08") UTC-parse off-by-one bug).
function fmtDay(s) {
  const [y, m, d] = String(s).slice(0, 10).split("-").map(Number);
  return new Date(y, m - 1, d).toLocaleDateString("en-US");
}

function renderTrend(trend) {
  const ctx = document.getElementById("ana-trend");
  if (!ctx || typeof Chart === "undefined") return;
  const labels = trend.map(r => fmtDay(r.bucket ?? r.week));
  const data = trend.map(r => r.rate);
  if (trendChart) trendChart.destroy();
  trendChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Attendance rate (%)",
        data,
        borderColor: "#2a4b7f",
        backgroundColor: "rgba(42,75,127,0.15)",
        fill: true,
        tension: 0.3,
        pointRadius: 5,
        pointBackgroundColor: "#2a4b7f",
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {y: {beginAtZero: true, max: 100}},
    },
  });
}

function renderBreakdown(b) {
  const ctx = document.getElementById("ana-breakdown");
  if (!ctx || typeof Chart === "undefined") return;
  if (breakdownChart) breakdownChart.destroy();
  breakdownChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Present", "Late", "Absent"],
      datasets: [{
        data: [b.present || 0, b.late || 0, b.absent || 0],
        backgroundColor: ["#16a34a", "#d97706", "#dc2626"],
      }],
    },
    options: {responsive: true, maintainAspectRatio: false},
  });
}

document.getElementById("ana-refresh").addEventListener("click", loadAnalytics);
document.getElementById("ana-course").addEventListener("change", loadAnalytics);

// ──────────────────────────────────────────────────────────────
// U14 Reports tab
// ──────────────────────────────────────────────────────────────
document.getElementById("report-form").addEventListener("submit", async e => {
  e.preventDefault();
  const params = new URLSearchParams();
  const cid = document.getElementById("report-course").value;
  const from = document.getElementById("report-from").value;
  const to = document.getElementById("report-to").value;
  if (cid) params.set("course_id", cid);
  if (from) params.set("date_from", from);
  if (to) params.set("date_to", to);
  const msg = document.getElementById("report-msg");
  msg.textContent = "Generating…";
  try {
    const res = await fetch(API_BASE + "/teacher/reports/export?" + params.toString(), {
      headers: authHeader(),
    });
    if (!res.ok) throw new Error(await res.text() || `HTTP ${res.status}`);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `attendance_report_${Date.now()}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    msg.textContent = "Downloaded.";
  } catch (ex) {
    msg.textContent = "Error: " + ex.message;
  }
});

// ──────────────────────────────────────────────────────────────
// U31 Leave Applications tab
// ──────────────────────────────────────────────────────────────
async function loadTeacherLeave() {
  const body = document.getElementById("tleave-body");
  const msg = document.getElementById("tleave-msg");
  body.innerHTML = "";
  msg.textContent = "";
  try {
    const res = await api("/teacher/leave-applications");
    const apps = res.applications || [];
    if (!apps.length) {
      body.append(el("tr", {}, el("td", {colspan: 6, class: "muted"}, "No pending leave applications.")));
      return;
    }
    for (const a of apps) {
      const doc = a.supporting_doc_url
        ? el("a", {href: a.supporting_doc_url, target: "_blank"}, "View")
        : "—";
      const approveBtn = el("button", {onclick: () => reviewLeave(a.leaveapplicationid, "approved")}, "Approve");
      const rejectBtn = el("button", {class: "danger", onclick: () => reviewLeave(a.leaveapplicationid, "rejected")}, "Reject");
      body.append(el("tr", {},
        el("td", {}, `${a.full_name || "-"} (${a.student_id || a.accountid})`),
        el("td", {}, `${a.course_code} — ${a.course_name}`),
        el("td", {}, fmt(a.start_time)),
        el("td", {}, a.reason),
        el("td", {}, doc),
        el("td", {}, approveBtn, rejectBtn),
      ));
    }
  } catch (e) {
    msg.style.color = "#c0392b";
    msg.textContent = e.message;
  }
}

async function reviewLeave(leaveId, decision) {
  const comment = prompt(`Optional comment for this ${decision} decision:`, "") || null;
  const msg = document.getElementById("tleave-msg");
  try {
    await api(`/teacher/leave-applications/${leaveId}`, {
      method: "PATCH",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({decision, comment}),
    });
    msg.style.color = "#16a34a";
    msg.textContent = `Application ${decision}.`;
    loadTeacherLeave();
  } catch (ex) {
    msg.style.color = "#c0392b";
    msg.textContent = ex.message;
  }
}

document.getElementById("leave-refresh").addEventListener("click", loadTeacherLeave);

// ── Init ──────────────────────────────────────────────────────
(async () => {
  await loadCourses();
  await loadSessions();
})();
