const user = requireAuth("student");
document.getElementById("who").textContent = `${user.full_name || user.email} (student)`;

// ── Tabs ──────────────────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    ["attendance", "analytics", "appeals", "leave", "face"].forEach(name => {
      document.getElementById("tab-" + name).style.display =
        name === btn.dataset.tab ? "" : "none";
    });
    if (btn.dataset.tab === "analytics") loadStudentAnalytics();
    if (btn.dataset.tab === "appeals") showAppealsView("list");
    if (btn.dataset.tab === "leave") { showLeaveView("list"); loadLeave(); }
  });
});

// ── Monthly summary (pie chart) ───────────────────────────────
let monthlyChart = null;

function renderMonthlySummary(records) {
  const now = new Date();
  const y = now.getFullYear(), m = now.getMonth();
  const monthName = now.toLocaleString("en-US", {month: "long", year: "numeric"});
  document.getElementById("monthly-title").textContent = `${monthName} Attendance`;
  const first = new Date(y, m, 1);
  const last = new Date(y, m + 1, 0);
  document.getElementById("monthly-range").textContent =
    `${first.toLocaleDateString("en-US")} – ${last.toLocaleDateString("en-US")}`;

  const counts = {present: 0, late: 0, absent: 0};
  for (const r of records) {
    const d = new Date(r.start_time);
    if (d.getFullYear() !== y || d.getMonth() !== m) continue;
    if (counts[r.status] !== undefined) counts[r.status]++;
  }
  const total = counts.present + counts.late + counts.absent;
  const attended = counts.present + counts.late;
  const rate = total ? Math.round((attended / total) * 100) : 0;

  document.getElementById("monthly-rate").textContent = total ? `${rate}%` : "--";
  document.getElementById("monthly-counts").textContent = total
    ? `(${attended}/${total} sessions)`
    : "(no sessions this month)";

  const canvas = document.getElementById("monthly-chart");
  if (!canvas || typeof Chart === "undefined") return;

  const data = {
    labels: ["Present", "Late", "Absent"],
    datasets: [{
      data: [counts.present, counts.late, counts.absent],
      backgroundColor: ["#16a34a", "#d97706", "#c0392b"],
      borderColor: "#ffffff",
      borderWidth: 2,
    }],
  };
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    cutout: "60%",
    plugins: {
      legend: {position: "bottom", labels: {boxWidth: 12, font: {size: 11}}},
      tooltip: {
        callbacks: {
          label: (ctx) => {
            const v = ctx.parsed;
            const pct = total ? Math.round((v / total) * 100) : 0;
            return `${ctx.label}: ${v} (${pct}%)`;
          },
        },
      },
    },
  };

  if (monthlyChart) {
    monthlyChart.data = data;
    monthlyChart.update();
  } else {
    monthlyChart = new Chart(canvas, {type: "doughnut", data, options});
  }
}

// ── Attendance list ───────────────────────────────────────────
let attRecords = [];

async function loadAttendance() {
  const body = document.getElementById("att-body");
  body.innerHTML = "";
  try {
    const res = await api("/student/attendance");
    attRecords = res.records || [];
    fillAnalyticsCourses(attRecords);
    renderMonthlySummary(res.records || []);
    if (document.getElementById("att-calendar-view").style.display !== "none") {
      renderAttCalendar();
    }
    if (!res.records.length) {
      body.append(el("tr", {}, el("td", {colspan: 5, class: "muted"}, "No records yet.")));
      return;
    }
    for (const r of res.records) {
      const tr = el("tr", {},
        el("td", {}, `${r.course_code} — ${r.course_name}`),
        el("td", {}, fmt(r.start_time)),
        el("td", {}, statusCell(r)),
        el("td", {}, r.marked_at ? fmt(r.marked_at) : "—"),
        el("td", {},
          el("button", {
            class: "secondary",
            onclick: () => toggleSessionDetail(tr, r.session_id),
          }, "Details"),
          appealButton(r),
        ),
      );
      body.append(tr);
    }
  } catch (e) {
    body.append(el("tr", {}, el("td", {colspan: 5, class: "error"}, e.message)));
  }
}

// ── Attendance calendar view ──────────────────────────────────
// `calCursor` is the first day of the month currently shown.
let calCursor = new Date(new Date().getFullYear(), new Date().getMonth(), 1);
const DOW = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

// Map an attendance record to a coloured chip class.
function calChipClass(r) {
  if (r.status === "present" || r.status === "late" || r.status === "absent") {
    return r.status;
  }
  return "pending"; // scheduled / active / no record yet
}

// Bucket records by local calendar day (YYYY-M-D key).
function recordsByDay() {
  const map = {};
  for (const r of attRecords) {
    if (!r.start_time) continue;
    const d = new Date(r.start_time);
    const key = `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`;
    (map[key] ||= []).push(r);
  }
  return map;
}

function renderAttCalendar() {
  const host = document.getElementById("att-calendar-view");
  host.innerHTML = "";
  const y = calCursor.getFullYear(), m = calCursor.getMonth();
  const monthName = calCursor.toLocaleString("en-US", {month: "long", year: "numeric"});
  const byDay = recordsByDay();
  const today = new Date();

  // Header with month label + prev/next navigation.
  const head = el("div", {class: "cal-head"},
    el("button", {class: "cal-nav", title: "Previous month",
      onclick: () => { calCursor = new Date(y, m - 1, 1); renderAttCalendar(); }}, "‹"),
    el("h3", {}, monthName),
    el("button", {class: "cal-nav", title: "Next month",
      onclick: () => { calCursor = new Date(y, m + 1, 1); renderAttCalendar(); }}, "›"),
  );

  const grid = el("div", {class: "cal-grid"});
  for (const d of DOW) grid.append(el("div", {class: "cal-dow"}, d));

  const firstDow = new Date(y, m, 1).getDay();
  const daysInMonth = new Date(y, m + 1, 0).getDate();
  // Leading blanks so day 1 lands under the right weekday.
  for (let i = 0; i < firstDow; i++) grid.append(el("div", {class: "cal-cell cal-empty"}));

  for (let day = 1; day <= daysInMonth; day++) {
    const isToday = today.getFullYear() === y && today.getMonth() === m && today.getDate() === day;
    const cell = el("div", {class: "cal-cell" + (isToday ? " cal-today" : "")},
      el("div", {class: "cal-date"}, String(day)));
    const recs = byDay[`${y}-${m}-${day}`] || [];
    if (recs.length) {
      const events = el("div", {class: "cal-events"});
      for (const r of recs) {
        const status = r.status || r.session_status || "scheduled";
        const chip = el("div", {
          class: "cal-chip " + calChipClass(r),
          title: `${r.course_code} — ${r.course_name}\n${fmt(r.start_time)}\nStatus: ${status}\n(click for details)`,
          onclick: () => showSessionModal(r.session_id),
        }, r.course_code || "Class");
        events.append(chip);
      }
      cell.append(events);
    }
    grid.append(cell);
  }

  // Trailing blanks to complete the final week row.
  const trailing = (7 - ((firstDow + daysInMonth) % 7)) % 7;
  for (let i = 0; i < trailing; i++) grid.append(el("div", {class: "cal-cell cal-empty"}));

  host.append(el("div", {class: "cal"}, head, grid));
}

// Session detail popup, opened from a calendar chip. Mirrors the inline
// detail shown in the list view but as a centred modal.
async function showSessionModal(sessionId) {
  const body = el("div", {class: "cal-modal-body"}, "Loading…");
  const close = () => backdrop.remove();
  const modal = el("div", {class: "cal-modal", onclick: (e) => e.stopPropagation()},
    el("div", {class: "cal-modal-head"},
      el("h3", {}, "Session details"),
      el("button", {class: "cal-modal-close", title: "Close", onclick: close}, "×"),
    ),
    body,
  );
  const backdrop = el("div", {class: "cal-modal-backdrop", onclick: close}, modal);
  document.body.append(backdrop);
  document.addEventListener("keydown", function esc(e) {
    if (e.key === "Escape") { close(); document.removeEventListener("keydown", esc); }
  });

  try {
    const res = await api(`/student/sessions/${sessionId}`);
    const s = res.session;
    const checkedIn = s.marked_at && s.attendance_status && s.attendance_status !== "absent";
    const rows = [
      ["Course", `${s.course_code} — ${s.course_name}`],
      ["Date & time", `${fmt(s.start_time)}${s.end_time ? " – " + fmt(s.end_time) : ""}`],
      ["Session status", s.session_status],
      ["Attendance status", s.attendance_status
        ? `<span class="status-${s.attendance_status}">${s.attendance_status}</span>`
        : "—"],
      ["Check-in timestamp", checkedIn ? fmt(s.marked_at)
        : '<span class="muted">No check-in recorded</span>'],
    ];
    body.innerHTML = rows.map(([k, v]) =>
      `<div class="cal-modal-row"><div class="k">${k}</div><div>${v}</div></div>`
    ).join("");
  } catch (e) {
    body.innerHTML = `<span class="error">${e.message}</span>`;
  }
}

// View toggle: List ↔ Calendar.
function setAttView(view) {
  const isCal = view === "calendar";
  document.getElementById("att-list-view").style.display = isCal ? "none" : "";
  document.getElementById("att-calendar-view").style.display = isCal ? "" : "none";
  const listBtn = document.getElementById("att-view-list");
  const calBtn = document.getElementById("att-view-calendar");
  listBtn.classList.toggle("active", !isCal);
  listBtn.classList.toggle("secondary", isCal);
  calBtn.classList.toggle("active", isCal);
  calBtn.classList.toggle("secondary", !isCal);
  if (isCal) renderAttCalendar();
}
document.getElementById("att-view-list").addEventListener("click", () => setAttView("list"));
document.getElementById("att-view-calendar").addEventListener("click", () => setAttView("calendar"));

// ── Attendance row helpers ────────────────────────────────────
// Status cell: show the recorded attendance status, or a muted label for
// sessions that have no record yet (e.g. not-yet-started classes).
function statusCell(r) {
  if (r.status) return el("span", {class: "status-" + r.status}, r.status);
  const labels = {
    scheduled: "Not started",
    active: "In progress",
    cancelled: "Cancelled",
    ended: "—",
  };
  return el("span", {class: "muted"}, labels[r.session_status] || "—");
}

// Appeal is only for finished sessions with a disputable attendance mark.
// Exclude present/leave, and anything not yet started (scheduled / future).
const APPEALABLE_STATUSES = new Set(["absent", "late", "early_left"]);
function isAppealable(r) {
  if (r.record_id == null || !APPEALABLE_STATUSES.has(r.status)) return false;
  if (r.session_status === "scheduled" || r.session_status === "cancelled") return false;
  if (r.start_time && new Date(r.start_time).getTime() > Date.now()) return false;
  return true;
}

function appealButton(r) {
  const appealable = isAppealable(r);
  const attrs = {class: "secondary"};
  if (appealable) {
    attrs.onclick = () => {
      document.querySelector('[data-tab="appeals"]').click();
      document.getElementById("appeal-msg").textContent = "";
      showAppealsView("form");
      loadAppealableSessions(r.record_id);
    };
  } else {
    attrs.disabled = "";
  }
  return el("button", attrs, "Appeal");
}

// ── Session details (U07) — inline slide-out below the clicked row ─────
function closeDetailRow(detailTr) {
  const slide = detailTr.querySelector(".detail-slide");
  if (!slide) { detailTr.remove(); return; }
  slide.classList.remove("open");
  slide.addEventListener("transitionend", () => detailTr.remove(), {once: true});
}

async function toggleSessionDetail(tr, sessionId) {
  const next = tr.nextElementSibling;
  // Clicking again on an already-open row collapses it.
  if (next && next.classList.contains("detail-row")
      && next.dataset.for === String(sessionId)) {
    closeDetailRow(next);
    return;
  }
  // Only one detail open at a time.
  document.querySelectorAll("#att-body .detail-row").forEach(closeDetailRow);

  const content = el("div", {class: "detail-content"}, "Loading…");
  const slide = el("div", {class: "detail-slide"}, content);
  const detailTr = el("tr", {class: "detail-row"}, el("td", {colspan: 5}, slide));
  detailTr.dataset.for = String(sessionId);
  tr.after(detailTr);
  requestAnimationFrame(() => slide.classList.add("open"));

  try {
    const res = await api(`/student/sessions/${sessionId}`);
    const s = res.session;
    const checkedIn = s.marked_at && s.attendance_status && s.attendance_status !== "absent";
    const rows = [
      ["Course", `${s.course_code} — ${s.course_name}`],
      ["Date & time", `${fmt(s.start_time)}${s.end_time ? " – " + fmt(s.end_time) : ""}`],
      ["Session status", s.session_status],
      ["Attendance status", s.attendance_status
        ? `<span class="status-${s.attendance_status}">${s.attendance_status}</span>`
        : "—"],
      ["Check-in timestamp", checkedIn ? fmt(s.marked_at)
        : '<span class="muted">No check-in recorded</span>'],
    ];
    content.innerHTML = rows.map(([k, v]) =>
      `<div style="display:flex;gap:.75rem;padding:.25rem 0;border-bottom:1px solid var(--c-border-2)">
         <div style="min-width:160px;color:var(--c-text-2)">${k}</div><div>${v}</div></div>`
    ).join("");
  } catch (e) {
    content.innerHTML = `<span class="error">${e.message}</span>`;
  }
}

// ── Appeals ───────────────────────────────────────────────────
function showAppealsView(view) {
  document.getElementById("appeals-list").hidden = view !== "list";
  document.getElementById("appeals-form-view").hidden = view !== "form";
}
document.getElementById("show-appeal-form").addEventListener("click", () => {
  document.getElementById("appeal-msg").textContent = "";
  showAppealsView("form");
  loadAppealableSessions();
});
document.getElementById("back-appeals").addEventListener("click", () => showAppealsView("list"));

// Populate the appeal form's session dropdown with finished sessions that
// have a disputable mark (absent / late / early_left). Leave, present, and
// not-yet-started sessions are excluded. If none, disable submission.
async function loadAppealableSessions(selectedRecordId) {
  const sel = document.getElementById("appeal-record");
  const submitBtn = document.querySelector('#appeal-form button[type="submit"]');
  try {
    const res = await api("/student/attendance");
    const records = (res.records || []).filter(isAppealable);
    if (!records.length) {
      sel.innerHTML = '<option value="">No courses available to appeal</option>';
      sel.disabled = true;
      submitBtn.disabled = true;
      return;
    }
    sel.disabled = false;
    submitBtn.disabled = false;
    sel.innerHTML = "";
    for (const r of records) {
      const o = document.createElement("option");
      o.value = r.record_id;
      o.textContent = `${r.course_code} — ${fmt(r.start_time)} (${r.status})`;
      sel.appendChild(o);
    }
    if (selectedRecordId != null) sel.value = String(selectedRecordId);
  } catch (e) {
    sel.innerHTML = `<option value="">${e.message}</option>`;
    sel.disabled = true;
    submitBtn.disabled = true;
  }
}

document.getElementById("appeal-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = document.getElementById("appeal-msg");
  msg.textContent = "";
  const fd = new FormData(e.target);
  try {
    await api("/student/appeals", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        record_id: Number(fd.get("record_id")),
        reason: fd.get("reason"),
      }),
    });
    msg.style.color = "#16a34a";
    msg.textContent = "Appeal submitted.";
    e.target.reset();
    loadAppeals();
    showAppealsView("list");
  } catch (ex) {
    msg.style.color = "#c0392b";
    msg.textContent = ex.message;
  }
});

async function loadAppeals() {
  const body = document.getElementById("appeal-body");
  body.innerHTML = "";
  try {
    const res = await api("/student/appeals");
    if (!res.appeals.length) {
      body.append(el("tr", {}, el("td", {colspan: 5, class: "muted"}, "No appeals.")));
      return;
    }
    for (const a of res.appeals) {
      body.append(el("tr", {},
        el("td", {}, a.appealid),
        el("td", {}, a.attendancerecordid),
        el("td", {}, a.reason),
        el("td", {}, el("span", {class: "badge"}, a.status)),
        el("td", {}, fmt(a.created_at)),
      ));
    }
  } catch (e) {
    body.append(el("tr", {}, el("td", {colspan: 5, class: "error"}, e.message)));
  }
}

// ── Analytics (U27) ───────────────────────────────────────────
let sTrendChart = null, sBreakdownChart = null;

// U27: populate the module filter from the courses the student is enrolled
// in (derived from their own attendance records — no extra endpoint needed).
function fillAnalyticsCourses(records) {
  const sel = document.getElementById("sana-course");
  if (!sel) return;
  const seen = new Map();
  for (const r of records || []) {
    if (r.courseid && !seen.has(r.courseid)) {
      seen.set(r.courseid, `${r.course_code} — ${r.course_name}`);
    }
  }
  const prev = sel.value;
  sel.innerHTML = '<option value="">All modules</option>';
  for (const [id, label] of seen) {
    const o = document.createElement("option");
    o.value = id;
    o.textContent = label;
    sel.appendChild(o);
  }
  if (prev && seen.has(Number(prev))) sel.value = prev;
}

async function loadStudentAnalytics() {
  const params = new URLSearchParams();
  const courseId = document.getElementById("sana-course").value;
  const from = document.getElementById("sana-from").value;
  const to = document.getElementById("sana-to").value;
  if (courseId) params.set("course_id", courseId);
  if (from) params.set("date_from", from);
  if (to) params.set("date_to", to);
  const summaryEl = document.getElementById("sana-summary");
  try {
    const res = await api("/student/analytics?" + params.toString());
    const b = res.breakdown || {};
    summaryEl.style.color = "";
    summaryEl.textContent =
      `Total ${b.total || 0} · Rate ${b.rate || 0}% · Present ${b.present || 0} · ` +
      `Late ${b.late || 0} · Absent ${b.absent || 0}`;
    renderStudentTrend(res.trend || []);
    renderStudentBreakdown(b);
  } catch (e) {
    summaryEl.style.color = "#c0392b";
    summaryEl.textContent = e.message;
  }
}

// Format a YYYY-MM-DD date string without timezone shifting.
function fmtDay(s) {
  const [y, m, d] = String(s).slice(0, 10).split("-").map(Number);
  return new Date(y, m - 1, d).toLocaleDateString("en-US");
}

function renderStudentTrend(trend) {
  const ctx = document.getElementById("sana-trend");
  if (!ctx || typeof Chart === "undefined") return;
  const labels = trend.map(r => fmtDay(r.bucket ?? r.week));
  const data = trend.map(r => r.rate);
  if (sTrendChart) sTrendChart.destroy();
  sTrendChart = new Chart(ctx, {
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
    options: {responsive: true, maintainAspectRatio: false,
      scales: {y: {beginAtZero: true, max: 100}}},
  });
}

function renderStudentBreakdown(b) {
  const ctx = document.getElementById("sana-breakdown");
  if (!ctx || typeof Chart === "undefined") return;
  if (sBreakdownChart) sBreakdownChart.destroy();
  sBreakdownChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Present", "Late", "Absent"],
      datasets: [{
        data: [b.present || 0, b.late || 0, b.absent || 0],
        backgroundColor: ["#16a34a", "#d97706", "#c0392b"],
      }],
    },
    options: {responsive: true, maintainAspectRatio: false,
      plugins: {legend: {position: "bottom", labels: {boxWidth: 12, font: {size: 11}}}}},
  });
}

document.getElementById("sana-refresh").addEventListener("click", loadStudentAnalytics);
document.getElementById("sana-course").addEventListener("change", loadStudentAnalytics);

// ── Leave Application (U28) ───────────────────────────────────
function showLeaveView(view) {
  document.getElementById("leave-list").hidden = view !== "list";
  document.getElementById("leave-form-view").hidden = view !== "form";
}
document.getElementById("show-leave-form").addEventListener("click", () => {
  document.getElementById("leave-msg").textContent = "";
  showLeaveView("form");
  loadLeaveSessions();
});
document.getElementById("back-leave").addEventListener("click", () => showLeaveView("list"));

async function loadLeaveSessions() {
  const sel = document.getElementById("leave-session");
  const msg = document.getElementById("leave-msg");
  try {
    const res = await api("/student/sessions/upcoming");
    const sessions = res.sessions || [];
    sel.innerHTML = sessions.length
      ? '<option value="">— select a session —</option>'
      : '<option value="">No upcoming sessions</option>';
    for (const s of sessions) {
      const o = document.createElement("option");
      o.value = s.session_id;
      o.textContent = `#${s.session_id} ${s.course_code} — ${fmt(s.start_time)}`;
      sel.appendChild(o);
    }
  } catch (e) {
    sel.innerHTML = '<option value="">Unable to load sessions</option>';
    if (msg) {
      msg.style.color = "#c0392b";
      msg.textContent = e.message;
    }
  }
}

async function loadLeave() {
  await loadLeaveSessions();
  loadMyLeave();
}

async function loadMyLeave() {
  const body = document.getElementById("leave-body");
  body.innerHTML = "";
  try {
    const res = await api("/student/leave-applications");
    if (!res.applications.length) {
      body.append(el("tr", {}, el("td", {colspan: 5, class: "muted"}, "No leave applications.")));
      return;
    }
    for (const a of res.applications) {
      body.append(el("tr", {},
        el("td", {}, `${a.course_code} — ${a.course_name}`),
        el("td", {}, fmt(a.start_time)),
        el("td", {}, a.reason),
        el("td", {}, el("span", {class: "badge"}, a.status)),
        el("td", {}, a.reviewer_comment || "—"),
      ));
    }
  } catch (e) {
    body.append(el("tr", {}, el("td", {colspan: 5, class: "error"}, e.message)));
  }
}

document.getElementById("leave-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = document.getElementById("leave-msg");
  msg.textContent = "";
  const fd = new FormData(e.target);
  const doc = fd.get("supporting_doc_url");
  try {
    await api("/student/leave-applications", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        session_id: Number(fd.get("session_id")),
        reason: fd.get("reason"),
        supporting_doc_url: doc ? doc : null,
      }),
    });
    msg.style.color = "#16a34a";
    msg.textContent = "Leave application submitted.";
    e.target.reset();
    loadMyLeave();
    showLeaveView("list");
  } catch (ex) {
    msg.style.color = "#c0392b";
    msg.textContent = ex.message;
  }
});

// ── Face re-register ──────────────────────────────────────────
// ── Face Re-register: shared submit ───────────────────────────
async function submitFaceBlob(blob, filename) {
  const msg = document.getElementById("face-msg");
  msg.style.color = "#555";
  msg.textContent = "Uploading...";
  const fd = new FormData();
  fd.append("account_id", user.account_id);
  fd.append("file", blob, filename);
  try {
    const res = await api("/register", {method: "POST", body: fd});
    msg.style.color = res.success ? "#16a34a" : "#c0392b";
    msg.textContent = res.message;
    return res.success;
  } catch (ex) {
    msg.style.color = "#c0392b";
    msg.textContent = ex.message;
    return false;
  }
}

// Mode switch (camera vs upload)
const faceCamPanel = document.getElementById("face-mode-camera");
const faceUploadForm = document.getElementById("face-upload-form");
document.querySelectorAll(".face-mode-btn").forEach(b => {
  b.addEventListener("click", () => {
    document.querySelectorAll(".face-mode-btn").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    const mode = b.dataset.mode;
    faceCamPanel.style.display    = mode === "camera" ? "" : "none";
    faceUploadForm.style.display  = mode === "upload" ? "" : "none";
    if (mode !== "camera") stopFaceCam();
  });
});

// File-upload handler
faceUploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = e.target.querySelector('input[name="file"]').files[0];
  if (!f) return;
  if (await submitFaceBlob(f, f.name)) e.target.reset();
});

// Camera handler
const faceVideo   = document.getElementById("face-cam");
const faceCanvas  = document.getElementById("face-canvas");
const facePreview = document.getElementById("face-preview");
const faceStartBtn   = document.getElementById("face-cam-start");
const faceCaptureBtn = document.getElementById("face-cam-capture");
const faceRetakeBtn  = document.getElementById("face-cam-retake");
const faceSubmitBtn  = document.getElementById("face-cam-submit");
let faceStream = null;
let faceBlob = null;

function stopFaceCam() {
  if (faceStream) faceStream.getTracks().forEach(t => t.stop());
  faceStream = null;
  faceVideo.srcObject = null;
  faceStartBtn.disabled = false;
  faceCaptureBtn.disabled = true;
}

faceStartBtn.addEventListener("click", async () => {
  document.getElementById("face-msg").textContent = "";
  facePreview.style.display = "none";
  faceVideo.style.display = "";
  faceRetakeBtn.style.display = "none";
  faceBlob = null;
  faceSubmitBtn.disabled = true;
  try {
    faceStream = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 480}});
    faceVideo.srcObject = faceStream;
    faceStartBtn.disabled = true;
    faceCaptureBtn.disabled = false;
  } catch (e) {
    const msg = document.getElementById("face-msg");
    msg.style.color = "#c0392b";
    msg.textContent = "Unable to open camera: " + e.message;
  }
});

faceCaptureBtn.addEventListener("click", () => {
  if (!faceStream) return;
  const w = faceVideo.videoWidth || 640, h = faceVideo.videoHeight || 480;
  faceCanvas.width = w; faceCanvas.height = h;
  faceCanvas.getContext("2d").drawImage(faceVideo, 0, 0, w, h);
  faceCanvas.toBlob((blob) => {
    faceBlob = blob;
    facePreview.src = URL.createObjectURL(blob);
    facePreview.style.display = "";
    faceVideo.style.display = "none";
    faceRetakeBtn.style.display = "";
    faceSubmitBtn.disabled = false;
    stopFaceCam();
  }, "image/jpeg", 0.92);
});

faceRetakeBtn.addEventListener("click", () => {
  faceBlob = null;
  faceSubmitBtn.disabled = true;
  faceStartBtn.click();
});

faceSubmitBtn.addEventListener("click", async () => {
  if (!faceBlob) return;
  faceSubmitBtn.disabled = true;
  const ok = await submitFaceBlob(faceBlob, "face.jpg");
  if (ok) {
    facePreview.style.display = "none";
    faceRetakeBtn.style.display = "none";
    faceBlob = null;
  } else {
    faceSubmitBtn.disabled = false;
  }
});

window.addEventListener("pagehide", stopFaceCam);

loadAttendance();
loadAppeals();
