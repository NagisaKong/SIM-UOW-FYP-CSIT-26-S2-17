const user = requireAuth("teacher");
document.getElementById("who").textContent = `${user.full_name || user.email} (teacher)`;

// ── Tabs ──────────────────────────────────────────────────────
const TABS = ["sessions", "live", "roster", "analytics", "behaviour", "reports", "leave", "appeals"];
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
    if (btn.dataset.tab === "behaviour") loadBehaviourTab();
    if (btn.dataset.tab === "leave") loadTeacherLeave();
    if (btn.dataset.tab === "appeals") loadTeacherAppeals();
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
const scanDiagnostics = document.getElementById("scan-diagnostics");
const scanCountdownEl = document.getElementById("scan-countdown");
const scanMsg = document.getElementById("scan-msg");
// One entry per opened camera: {deviceId, stream, video, label}.
let scanCams = [];
let scanTimer = null;
let scanBusy = false;
let scanCountdown = 0;

// Live box preview (detection-only, no attendance recording).
// The loop is self-throttling: the next round starts PREVIEW_GAP_MS after
// the previous one finishes, so the effective frame rate adapts to how fast
// the backend can run inference (fast GPU → several fps; slow CPU → the
// inference time itself becomes the pace and requests never pile up).
let previewTimer = null;
let previewBusy = false;
const PREVIEW_GAP_MS = 200;
const previewCanvas = document.createElement("canvas"); // off-screen, separate from capture

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
  stopPreviewLoop();
  stopBehaviourLoop();
  for (const cam of scanCams) cam.stream.getTracks().forEach(t => t.stop());
  scanCams = [];
  scanCamsHost.innerHTML = "";
  scanCamsHost.append(scanCamsEmpty);
  scanCamsEmpty.style.display = "";
  const detectedHost = document.getElementById("scan-detected");
  if (detectedHost) detectedHost.innerHTML = "";
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
      video.setAttribute("muted", "");      // belt-and-suspenders for autoplay
      video.srcObject = stream;
      const overlay = el("canvas", {class: "scan-cam-overlay"});
      const stage = el("div", {class: "scan-cam-stage"}, video, overlay);
      const tile = el("div", {class: "scan-cam-tile"},
        stage, el("span", {class: "scan-cam-label"}, label));
      scanCamsHost.append(tile);
      // Autoplay can fail to kick in for srcObject streams; start it explicitly.
      video.play().catch(err => console.warn(`video.play() failed for ${label}:`, err));
      scanCams.push({deviceId, stream, video, overlay, label});
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
    startPreviewLoop(); // real-time boxes
    startBehaviourLoop(); // CR-06 ~1 fps behaviour sampling (self-stops if disabled)
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

// Live behaviour state fed by the ~1 fps behaviour sampler (CR-06): which
// recognised students are currently confirmed drowsy / on their phone.
// Entries expire after BEH_STATE_TTL_MS so a stale flag can't stay red
// forever if the sampler stops (behaviour disabled, service off, error).
const BEH_STATE_TTL_MS = 8000;
const behLive = {drowsy: new Set(), phone: new Set(), ts: 0};

// The live overlay reports a single state — "Not paying attention" — rather
// than naming sleeping or phone use over a student's head in front of the
// class. The distinction is still recorded: behaviour_event stores the
// specific type, and the teacher's Behaviour report breaks it down per
// student (U32). Merging only the public-facing label keeps the accusation
// low-stakes while the underlying data stays precise, which matters given
// these detections are assistive indicators rather than proof.
function behStateFor(accountId) {
  if (accountId == null || Date.now() - behLive.ts > BEH_STATE_TTL_MS) return [];
  if (behLive.drowsy.has(accountId) || behLive.phone.has(accountId)) {
    return ["Not paying attention"];
  }
  return [];
}

// Draw the detected face boxes returned for one camera onto its overlay.
// Box coords are in the uploaded frame's pixels; the canvas is sized to the
// frame so CSS scales it onto the displayed video automatically.
// Colours: green = recognised + attentive, amber = unknown face,
// red = recognised but flagged sleeping / using phone (label says which).
function drawScanBoxes(cam, res) {
  const overlay = cam.overlay;
  if (!overlay) return;
  const fw = res.frame_width || cam.video.videoWidth || 1280;
  const fh = res.frame_height || cam.video.videoHeight || 720;
  overlay.width = fw; overlay.height = fh;
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, fw, fh);
  const fontPx = Math.max(16, Math.round(fw / 45));
  ctx.lineWidth = Math.max(2, Math.round(fw / 350));
  ctx.font = `bold ${fontPx}px sans-serif`;
  ctx.textBaseline = "alphabetic";
  for (const b of (res.boxes || [])) {
    const [x1, y1, x2, y2] = b.bbox;
    let color, text;
    if (!b.recognised) {
      color = "#e3a008";
      text = "Unknown";
    } else {
      const acts = behStateFor(b.account_id);
      if (acts.length) {
        color = "#dc2626";
        text = `${b.label} — ${acts.join(" + ")}`;
      } else {
        color = "#16a34a";
        text = `${b.label} ${b.score}`;
      }
    }
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const tw = ctx.measureText(text).width;
    const ty = y1 > fontPx + 4 ? y1 : y2 + fontPx + 4; // keep label on-screen
    ctx.fillStyle = color;
    ctx.fillRect(x1, ty - fontPx - 2, tw + 10, fontPx + 6);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, x1 + 5, ty);

    // Diagnostic sub-label under an unrecognised face: the nearest gallery
    // entries and the threshold in force. It distinguishes "nobody is close"
    // (raise enrolment quality) from "two people are equally close" (the
    // margin rule refusing to guess) — invisible otherwise.
    if (!b.recognised && scanDiagnostics.checked && (b.candidates || []).length) {
      const hint = describeCandidates(b);
      const smallPx = Math.round(fontPx * 0.62);
      ctx.font = `${smallPx}px sans-serif`;
      const hw = ctx.measureText(hint).width;
      const hy = ty + smallPx + 6;
      ctx.fillStyle = "rgba(0,0,0,0.65)";
      ctx.fillRect(x1, hy - smallPx - 2, hw + 8, smallPx + 6);
      ctx.fillStyle = "#fde68a";
      ctx.fillText(hint, x1 + 4, hy);
      ctx.font = `bold ${fontPx}px sans-serif`;
    }
  }
}

// "≈ zhang jiqian 0.38 / DOMINIC 0.31 (need 0.43)" — the top-2 gallery
// scores plus the active threshold, so the reason for a rejection is
// readable straight off the video.
function describeCandidates(box) {
  const top = (box.candidates || []).slice(0, 2).map(c => {
    const who = c.full_name || c.student_id || `acc#${c.account_id}`;
    return `${who} ${Number(c.score).toFixed(2)}`;
  });
  const need = box.threshold != null ? ` (need ${Number(box.threshold).toFixed(2)})` : "";
  return `≈ ${top.join(" / ")}${need}`;
}

// Render the "Detected N: name1, name2…" summary from the union of all
// cameras' recognised students in the latest snapshot cycle.
function renderScanDetected(map, totalFaces) {
  const host = document.getElementById("scan-detected");
  host.innerHTML = "";
  // Stats line: total faces in frame vs. recognised students.
  host.append(el("div", {class: "scan-stats small"},
    `Faces in frame: ${totalFaces} · Recognised: ${map.size}`));
  if (!map.size) {
    host.append(el("div", {class: "muted small"}, "No students recognised."));
    return;
  }
  const names = el("div", {class: "scan-names"});
  for (const name of map.values()) names.append(el("span", {class: "chip"}, name));
  host.append(names);
}

async function captureSnapshot() {
  const id = currentSessionId();
  if (!scanCams.length || !id || scanBusy) return;
  scanBusy = true;
  let ok = 0, fail = 0, totalFaces = 0;
  const detected = new Map(); // account_id -> display name (union across cameras)
  // Sequential: the capture canvas is shared, and it avoids hammering the
  // backend with N simultaneous uploads.
  for (const cam of scanCams) {
    try {
      const res = await snapshotOneCamera(cam, id);
      drawScanBoxes(cam, res);
      totalFaces += res.faces_in_frame || (res.boxes ? res.boxes.length : 0);
      for (const d of (res.detected || [])) {
        detected.set(d.account_id, d.full_name || d.student_id || `acc#${d.account_id}`);
      }
      ok++;
    } catch (ex) {
      fail++;
      if (cam.overlay) cam.overlay.getContext("2d").clearRect(0, 0, cam.overlay.width, cam.overlay.height);
      console.warn(`Scan failed for ${cam.label}:`, ex);
    }
  }
  renderScanDetected(detected, totalFaces);
  scanMsg.style.color = fail ? "#c0392b" : "#16a34a";
  scanMsg.textContent =
    `Captured ${scanCams.length} camera${scanCams.length > 1 ? "s" : ""}` +
    ` · ${ok} ok${fail ? `, ${fail} failed` : ""}`;
  refreshLive();
  scanBusy = false;
}

scanCapture.addEventListener("click", captureSnapshot);

// ── Live box preview ──────────────────────────────────────────
// Polls a detection-only endpoint per camera and redraws the boxes, giving a
// near-real-time overlay without writing any attendance rows.
async function previewOneCamera(cam) {
  const v = cam.video;
  const w = v.videoWidth || 1280, h = v.videoHeight || 720;
  previewCanvas.width = w; previewCanvas.height = h;
  previewCanvas.getContext("2d").drawImage(v, 0, 0, w, h);
  const blob = await new Promise(r => previewCanvas.toBlob(r, "image/jpeg", 0.7));
  const fd = new FormData();
  fd.append("file", blob, "preview.jpg");
  return api("/teacher/preview-detect", {method: "POST", body: fd});
}

async function previewTick() {
  if (previewBusy || !scanCams.length) return;
  previewBusy = true;
  const detected = new Map(); // label -> name (deduped across cameras)
  let totalFaces = 0;
  for (const cam of scanCams) {
    try {
      const res = await previewOneCamera(cam);
      drawScanBoxes(cam, res);
      totalFaces += res.faces_in_frame || 0;
      for (const b of (res.boxes || [])) {
        if (b.recognised) detected.set(b.label, b.label);
      }
    } catch (e) {
      // Transient errors (busy model, network) — skip this round silently.
    }
  }
  renderScanDetected(detected, totalFaces);
  previewBusy = false;
}

function startPreviewLoop() {
  stopPreviewLoop();
  const loop = async () => {
    await previewTick();
    if (scanCams.length) previewTimer = setTimeout(loop, PREVIEW_GAP_MS);
  };
  previewTimer = setTimeout(loop, 100);
}

function stopPreviewLoop() {
  if (previewTimer) { clearTimeout(previewTimer); previewTimer = null; }
  previewBusy = false;
}

// ── CR-06 behaviour sampling (~1 fps) ─────────────────────────
// While the classroom scan is running, one frame per second is sent to the
// behaviour-scan endpoint (drowsiness / phone / heatmap). With several
// cameras open we rotate round-robin — one camera per tick — so the total
// upload rate stays at ~1 fps regardless of camera count. The loop stops
// itself when the course has behaviour analysis disabled (U35) or the
// backend service is off (AI_BEHAVIOUR=false).
const BEHAVIOUR_INTERVAL_MS = 1000;
const behCanvas = document.createElement("canvas"); // off-screen frame grabber
const behStatusEl = document.getElementById("scan-behaviour");
let behTimer = null;
let behBusy = false;
let behCamIndex = 0;

async function behaviourTick() {
  if (behBusy || !scanCams.length) return;
  const sessionId = currentSessionId();
  if (!sessionId) return;
  behBusy = true;
  try {
    const cam = scanCams[behCamIndex % scanCams.length];
    behCamIndex++;
    const v = cam.video;
    const w = v.videoWidth || 1280, h = v.videoHeight || 720;
    behCanvas.width = w; behCanvas.height = h;
    behCanvas.getContext("2d").drawImage(v, 0, 0, w, h);
    const blob = await new Promise(r => behCanvas.toBlob(r, "image/jpeg", 0.8));
    const fd = new FormData();
    fd.append("file", blob, "behaviour.jpg");
    const res = await api(`/teacher/sessions/${sessionId}/behaviour-scan`, {
      method: "POST", body: fd,
    });
    if (res.enabled === false) {
      // U35: switched off for this course — no point sampling further.
      behStatusEl.textContent = "Behaviour analysis: disabled for this course (U35).";
      stopBehaviourLoop(true);
      return;
    }
    if (res.skipped) return; // backend still busy with the previous frame
    // Feed the live overlay state (red boxes on the camera preview).
    behLive.drowsy = new Set(res.drowsy_active || []);
    behLive.phone = new Set(res.phone_active || []);
    behLive.ts = Date.now();
    const drowsy = (res.drowsy_active || []).length;
    const phone = (res.phone_active || []).length;
    behStatusEl.textContent =
      `Behaviour analysis: sampling ~1 fps · not paying attention now: ` +
      `${new Set([...(res.drowsy_active || []), ...(res.phone_active || [])]).size}` +
      ` (drowsy ${drowsy} / phone ${phone})` +
      (res.events_written ? ` · +${res.events_written} event(s) recorded` : "");
  } catch (e) {
    // 503 = AI_BEHAVIOUR off on this server; anything else transient → keep trying.
    if (/not running|AI_BEHAVIOUR/i.test(e.message)) {
      behStatusEl.textContent = "Behaviour analysis: service not enabled on this server.";
      stopBehaviourLoop(true);
    }
  } finally {
    behBusy = false;
  }
}

function startBehaviourLoop() {
  stopBehaviourLoop();
  behStatusEl.textContent = "Behaviour analysis: starting…";
  const loop = async () => {
    await behaviourTick();
    if (scanCams.length && behTimer !== null) {
      behTimer = setTimeout(loop, BEHAVIOUR_INTERVAL_MS);
    }
  };
  behTimer = setTimeout(loop, 500);
}

function stopBehaviourLoop(keepMessage = false) {
  if (behTimer) { clearTimeout(behTimer); }
  behTimer = null;
  behBusy = false;
  behCamIndex = 0;
  behLive.drowsy.clear();
  behLive.phone.clear();
  behLive.ts = 0;
  if (!keepMessage && behStatusEl) behStatusEl.textContent = "";
}

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
// U32/U33 Behaviour tab (CR-06)
// ──────────────────────────────────────────────────────────────
function fmtDur(seconds) {
  const s = Number(seconds) || 0;
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${s % 60}s`;
}

// Sessions worth reporting on: currently active or already ended.
async function loadBehaviourSessions() {
  const sel = document.getElementById("beh-session");
  const prev = sel.value;
  sel.innerHTML = '<option value="">— select a session —</option>';
  const res = await api("/teacher/sessions");
  for (const s of (res.sessions || []).filter(x => x.status === "active" || x.status === "ended")) {
    const o = document.createElement("option");
    o.value = s.attendancesessionid;
    o.textContent = `#${s.attendancesessionid} ${s.course_code} (${fmt(s.start_time)}) [${s.status}]`;
    sel.appendChild(o);
  }
  if (prev) sel.value = prev;
}

async function loadBehaviourData() {
  const id = document.getElementById("beh-session").value;
  const msg = document.getElementById("beh-msg");
  const summaryBody = document.getElementById("beh-summary-body");
  const timelineBody = document.getElementById("beh-timeline-body");
  summaryBody.innerHTML = "";
  timelineBody.innerHTML = "";
  renderBehaviourHeatmap([]);
  if (!id) { msg.textContent = "Select a session to view its behaviour report."; return; }
  msg.textContent = "Loading…";
  try {
    const [report, heat] = await Promise.all([
      api(`/teacher/sessions/${id}/behaviour`),
      api(`/teacher/sessions/${id}/heatmap${heatmapRangeQuery()}`),
    ]);
    seedHeatmapRange(heat);

    const students = report.students || [];
    if (!students.length) {
      summaryBody.append(el("tr", {},
        el("td", {colspan: 6, class: "muted"}, "No behaviour events recorded for this session.")));
    }
    for (const s of students) {
      // U32: a student the analyser never got usable landmarks for is
      // reported as inconclusive, not as well-behaved.
      const status = s.analysis_status || "not_analysed";
      const coverage = status === "inconclusive"
        ? el("span", {class: "badge", title:
            "Insufficient camera coverage — no usable facial landmarks were captured for this student",
          }, "Inconclusive")
        : status === "analysed"
          ? el("span", {class: "muted small"},
              s.samples_analysed ? `${s.samples_analysed} samples` : "analysed")
          : el("span", {class: "muted small"}, "—");
      summaryBody.append(el("tr", {},
        el("td", {}, s.full_name || `acc#${s.accountid}`),
        el("td", {}, String(s.drowsy || 0)),
        el("td", {}, fmtDur(s.drowsy_seconds)),
        el("td", {}, String(s.phone || 0)),
        el("td", {}, fmtDur(s.phone_seconds)),
        el("td", {}, coverage),
      ));
    }

    const timeline = report.timeline || [];
    if (!timeline.length) {
      timelineBody.append(el("tr", {},
        el("td", {colspan: 5, class: "muted"}, "No events.")));
    }
    for (const ev of timeline) {
      const meta = ev.metadata || {};
      const details = ev.event_type === "drowsiness"
        ? (meta.reasons || []).join(", ")
        : (meta.conf != null ? `conf ${meta.conf}` : "");
      timelineBody.append(el("tr", {},
        el("td", {}, fmt(ev.detected_at)),
        el("td", {}, ev.full_name || `acc#${ev.accountid}`),
        el("td", {}, el("span", {class: "badge"}, ev.event_type)),
        el("td", {}, fmtDur(ev.duration_seconds)),
        el("td", {class: "small muted"}, details),
      ));
    }

    renderBehaviourHeatmap(heat.zones || []);
    const inconclusive = report.inconclusive_count || 0;
    msg.textContent = report.note || heat.note ||
      `${students.length} student(s) · ${timeline.length} event(s)` +
      (inconclusive ? ` · ${inconclusive} inconclusive (insufficient coverage)` : "");
  } catch (e) {
    msg.textContent = "Error: " + e.message;
  }
}

// ── U33 heatmap time-range filter ─────────────────────────────
// Builds the ?time_from/&time_to query from the two pickers. The inputs
// are datetime-local (no timezone), so convert to ISO before sending.
function heatmapRangeQuery() {
  const from = document.getElementById("beh-heat-from").value;
  const to = document.getElementById("beh-heat-to").value;
  const params = new URLSearchParams();
  if (from) params.set("time_from", new Date(from).toISOString());
  if (to) params.set("time_to", new Date(to).toISOString());
  const q = params.toString();
  return q ? "?" + q : "";
}

// Seed empty pickers with the session's full capture window so the teacher
// can narrow from a sensible starting point.
function seedHeatmapRange(heat) {
  const fromEl = document.getElementById("beh-heat-from");
  const toEl = document.getElementById("beh-heat-to");
  const toLocal = (ts) => {
    const d = new Date(ts);
    const pad = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}` +
           `T${pad(d.getHours())}:${pad(d.getMinutes())}`;
  };
  if (heat.captured_from) {
    fromEl.min = toLocal(heat.captured_from);
    if (!fromEl.value) fromEl.placeholder = fromEl.min;
  }
  if (heat.captured_to) {
    toEl.max = toLocal(heat.captured_to);
  }
}

document.getElementById("beh-heat-apply").addEventListener("click", loadBehaviourData);
document.getElementById("beh-heat-reset").addEventListener("click", () => {
  document.getElementById("beh-heat-from").value = "";
  document.getElementById("beh-heat-to").value = "";
  loadBehaviourData();
});

// U33: aggregate all snapshots into one grid (mean intensity per cell),
// then paint cold→warm cells onto the canvas.
function renderBehaviourHeatmap(zones) {
  const canvas = document.getElementById("beh-heatmap");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!zones.length) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "14px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("No heatmap data for this session.", canvas.width / 2, canvas.height / 2);
    return;
  }
  // Grid size = max index + 1 (backend default is 8×6).
  const cols = Math.max(...zones.map(z => z.zone_x)) + 1;
  const rows = Math.max(...zones.map(z => z.zone_y)) + 1;
  const acc = new Map(); // "x,y" -> {sum, n}
  for (const z of zones) {
    const k = `${z.zone_x},${z.zone_y}`;
    const a = acc.get(k) || {sum: 0, n: 0};
    a.sum += Number(z.intensity) || 0;
    a.n += 1;
    acc.set(k, a);
  }
  let peak = 0;
  for (const a of acc.values()) peak = Math.max(peak, a.sum / a.n);
  const cw = canvas.width / cols, ch = canvas.height / rows;
  for (const [k, a] of acc.entries()) {
    const [x, y] = k.split(",").map(Number);
    const v = peak > 0 ? (a.sum / a.n) / peak : 0; // 0..1
    // Cold (hue 220, blue) → warm (hue 0, red).
    ctx.fillStyle = `hsla(${Math.round(220 - 220 * v)}, 85%, 55%, ${0.25 + 0.65 * v})`;
    ctx.fillRect(x * cw + 1, y * ch + 1, cw - 2, ch - 2);
  }
}

// Colour legend strip: same cold→warm formula as the heatmap cells, painted
// over the same dark backdrop, so the legend matches the plot exactly.
// Leftmost = the backdrop itself (no detections at all).
function drawHeatmapLegend() {
  const canvas = document.getElementById("beh-legend");
  if (!canvas || canvas.dataset.drawn) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, w, h);
  for (let x = Math.round(w * 0.06); x < w; x++) {
    const v = x / (w - 1); // 0..1 intensity, mirroring renderBehaviourHeatmap
    ctx.fillStyle = `hsla(${Math.round(220 - 220 * v)}, 85%, 55%, ${0.25 + 0.65 * v})`;
    ctx.fillRect(x, 0, 1, h);
  }
  canvas.dataset.drawn = "1";
}

async function loadBehaviourTab() {
  drawHeatmapLegend();
  await loadBehaviourSessions();
  await loadBehaviourData();
}

document.getElementById("beh-session").addEventListener("change", loadBehaviourData);
document.getElementById("beh-refresh").addEventListener("click", loadBehaviourTab);

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

// ──────────────────────────────────────────────────────────────
// U08 Attendance Appeals tab (teacher review)
// List shows no reason column; the teacher opens a detail view to
// read the reason and decide there (mirrors the Admin appeals UX).
// ──────────────────────────────────────────────────────────────
let currentAppeal = null;

function showAppealsView(view) {
  document.getElementById("tappeals-list-view").style.display = view === "list" ? "" : "none";
  document.getElementById("tappeals-detail-view").style.display = view === "detail" ? "" : "none";
}

async function loadTeacherAppeals() {
  const body = document.getElementById("tappeals-body");
  const msg = document.getElementById("tappeals-msg");
  body.innerHTML = "";
  msg.textContent = "";
  showAppealsView("list");
  try {
    const res = await api("/teacher/appeals");
    const appeals = res.appeals || [];
    if (!appeals.length) {
      body.append(el("tr", {}, el("td", {colspan: 6, class: "muted"}, "No appeals submitted.")));
      return;
    }
    for (const a of appeals) {
      body.append(el("tr", {},
        el("td", {}, `${a.full_name || "-"} (${a.student_id || a.accountid})`),
        el("td", {}, `${a.course_code} — ${fmt(a.start_time)}`),
        el("td", {}, el("span", {class: "badge"}, a.record_status || "-")),
        el("td", {}, fmt(a.created_at)),
        el("td", {}, a.status),
        el("td", {}, el("button", {onclick: () => openTeacherAppeal(a)}, "View")),
      ));
    }
  } catch (e) {
    msg.style.color = "#c0392b";
    msg.textContent = e.message;
  }
}

function openTeacherAppeal(a) {
  currentAppeal = a;
  document.getElementById("tappeal-student").textContent =
    `${a.full_name || "-"} (${a.student_id || a.accountid})`;
  document.getElementById("tappeal-session").textContent =
    `${a.course_code} — ${a.course_name} · ${fmt(a.start_time)}`;
  document.getElementById("tappeal-record").textContent = a.record_status || "-";
  document.getElementById("tappeal-created").textContent = fmt(a.created_at);
  document.getElementById("tappeal-status").textContent = a.status;
  document.getElementById("tappeal-reviewed").textContent =
    a.reviewed_at ? fmt(a.reviewed_at) : "Not reviewed yet";
  document.getElementById("tappeal-reason").textContent = a.reason || "-";
  document.getElementById("tappeal-actions").style.display =
    a.status === "pending" ? "" : "none";
  document.getElementById("tappeal-detail-msg").textContent = "";
  showAppealsView("detail");
}

async function reviewAppeal(status) {
  if (!currentAppeal) return;
  if (!confirm(`Set this appeal to ${status}?` +
    (status === "approved" ? "\nThe attendance record will be corrected to Present." : ""))) return;
  const msg = document.getElementById("tappeal-detail-msg");
  try {
    await api(`/teacher/appeals/${currentAppeal.appealid}`, {
      method: "PATCH",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({status}),
    });
    const listMsg = document.getElementById("tappeals-msg");
    listMsg.style.color = "#16a34a";
    listMsg.textContent = `Appeal ${status}.`;
    currentAppeal = null;
    loadTeacherAppeals();
  } catch (ex) {
    msg.style.color = "#c0392b";
    msg.textContent = ex.message;
  }
}

document.getElementById("appeals-refresh").addEventListener("click", loadTeacherAppeals);
document.getElementById("tappeal-back").addEventListener("click", () => showAppealsView("list"));
document.getElementById("tappeal-approve").addEventListener("click", () => reviewAppeal("approved"));
document.getElementById("tappeal-reject").addEventListener("click", () => reviewAppeal("rejected"));

// ── Init ──────────────────────────────────────────────────────
(async () => {
  await loadCourses();
  await loadSessions();
})();
