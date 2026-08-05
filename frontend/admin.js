const user = requireAuth("admin");
document.getElementById("who").textContent = `${user.full_name || user.email} (admin)`;

// ── Tabs ──────────────────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    ["users", "faces", "courses", "attendance", "appeals", "att-config", "ai-config", "ensemble", "behaviour"].forEach(name => {
      document.getElementById("tab-" + name).style.display =
        name === btn.dataset.tab ? "" : "none";
    });
    if (btn.dataset.tab === "users") showUsersView("list");
    if (btn.dataset.tab === "faces") { showFacesView("list"); loadFaces(); }
    if (btn.dataset.tab === "courses") loadCourses();
    if (btn.dataset.tab === "attendance") loadAttendance();
    if (btn.dataset.tab === "appeals") {
      showAppealsView("list");
      loadAppeals();
    }
    if (btn.dataset.tab === "att-config") loadAttConfig();
    if (btn.dataset.tab === "behaviour") loadBehaviourTab();
    if (btn.dataset.tab === "ai-config") {}

    // Reveal the Courses sub-menu (slides out) only on the Courses tab.
    const submenu = document.getElementById("courses-submenu");
    if (btn.dataset.tab === "courses") {
      showCourseSub("manage");
      // Defer so the display flip registers before the slide transition runs.
      requestAnimationFrame(() => submenu.classList.add("open"));
    } else {
      submenu.classList.remove("open");
    }
  });
});

// ── Courses sub-pages ─────────────────────────────────────────
function showCourseSub(name) {
  document.querySelectorAll("#courses-submenu .submenu-btn").forEach(b =>
    b.classList.toggle("active", b.dataset.sub === name));
  ["manage", "enroll", "sessions"].forEach(sub => {
    document.getElementById("sub-" + sub).hidden = sub !== name;
  });
  // Always land on the list view, not the add/schedule form.
  if (name === "manage") showManageView("list");
  if (name === "sessions") showSessionsView("list");
}
document.querySelectorAll("#courses-submenu .submenu-btn").forEach(btn => {
  btn.addEventListener("click", () => showCourseSub(btn.dataset.sub));
});

// ── Users / Faces list ↔ form sub-pages ───────────────────────
function showUsersView(view) {
  document.getElementById("users-list").hidden = view !== "list";
  document.getElementById("users-form-view").hidden = view !== "form";
  document.getElementById("users-detail-view").hidden = view !== "detail";
  document.getElementById("users-edit-view").hidden = view !== "edit";
}
function showFacesView(view) {
  document.getElementById("faces-list").hidden = view !== "list";
  document.getElementById("faces-form-view").hidden = view !== "form";
}
document.getElementById("show-user-form").addEventListener("click", () => showUsersView("form"));
document.getElementById("back-users").addEventListener("click", () => showUsersView("list"));
document.getElementById("show-face-form").addEventListener("click", () => {
  document.getElementById("face-user-search").value = "";
  loadFaceUserOptions();
  showFacesView("form");
});
document.getElementById("back-faces").addEventListener("click", () => showFacesView("list"));

// ── Users ─────────────────────────────────────────────────────
async function loadUsers() {
  const body = document.getElementById("users-body");
  body.innerHTML = "";
  const res = await api("/admin/users");
  for (const u of res.users) {
    body.append(el("tr", {},
      el("td", {}, u.accountid),
      el("td", {}, u.email),
      el("td", {}, u.role),
      el("td", {}, u.full_name || "-"),
      el("td", {}, u.student_id || "-"),
      el("td", {}, u.status),
      el("td", {},
        el("button", {
          onclick: () => openUserDetail(u),
        }, "View"),
        el("button", {
          class: "secondary",
          style: "margin-left:6px",
          onclick: () => openUserEdit(u),
        }, "Edit"),
      ),
    ));
  }
}

// ── User View (detail) / Edit sub-pages ───────────────────────
let currentUser = null;  // the user being viewed/edited

function openUserDetail(u) {
  currentUser = u;
  document.getElementById("detail-id").textContent = u.accountid;
  document.getElementById("detail-email").textContent = u.email || "-";
  document.getElementById("detail-role").textContent = u.role || "-";
  document.getElementById("detail-name").textContent = u.full_name || "-";
  document.getElementById("detail-student").textContent = u.student_id || "-";
  document.getElementById("detail-staff").textContent = u.staff_id || "-";
  document.getElementById("detail-status").textContent = u.status || "-";
  document.getElementById("detail-created").textContent = u.created_at ? fmt(u.created_at) : "-";
  showUsersView("detail");
}

function openUserEdit(u) {
  currentUser = u;
  const f = document.getElementById("user-edit-form");
  f.email.value = u.email || "";
  f.full_name.value = u.full_name || "";
  f.student_id.value = u.student_id || "";
  f.staff_id.value = u.staff_id || "";
  document.getElementById("user-edit-msg").textContent = "";
  // Student-only controls
  document.getElementById("edit-uploadface-btn").style.display =
    u.role === "student" ? "" : "none";
  f.student_id.closest("label").style.display = u.role === "student" ? "" : "none";
  f.staff_id.closest("label").style.display = u.role === "student" ? "none" : "";
  // Deactivate / Activate label reflects current status
  const da = document.getElementById("edit-deactivate-btn");
  const active = u.status === "active";
  da.textContent = active ? "Deactivate" : "Activate";
  da.className = active ? "danger" : "secondary";
  showUsersView("edit");
}

document.getElementById("detail-edit-btn").addEventListener("click", () => {
  if (currentUser) openUserEdit(currentUser);
});
document.getElementById("back-users-detail").addEventListener("click", () => showUsersView("list"));
document.getElementById("back-users-edit").addEventListener("click", () => {
  if (currentUser) openUserDetail(currentUser); else showUsersView("list");
});

// Save edited details
document.getElementById("user-edit-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!currentUser) return;
  const msg = document.getElementById("user-edit-msg");
  msg.textContent = "";
  const fd = Object.fromEntries(new FormData(e.target));
  const payload = {email: fd.email, full_name: fd.full_name};
  if (currentUser.role === "student") payload.student_id = fd.student_id;
  else payload.staff_id = fd.staff_id;
  try {
    await api(`/admin/users/${currentUser.accountid}`, {
      method: "PATCH",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    Object.assign(currentUser, payload);  // keep local copy in sync
    await loadUsers();
    openUserDetail(currentUser);  // return to the (updated) detail view
  } catch (ex) {
    msg.style.color = "#c0392b";
    msg.textContent = ex.message;
  }
});

// Deactivate / Activate from the edit page
document.getElementById("edit-deactivate-btn").addEventListener("click", async () => {
  if (!currentUser) return;
  const target = currentUser.status === "active" ? "inactive" : "active";
  try {
    await api(`/admin/users/${currentUser.accountid}/status`, {
      method: "PATCH",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({status: target}),
    });
    currentUser.status = target;
    await loadUsers();
    openUserEdit(currentUser);  // refresh button label
  } catch (ex) {
    const msg = document.getElementById("user-edit-msg");
    msg.style.color = "#c0392b";
    msg.textContent = ex.message;
  }
});

// Upload Face from the edit page (students only)
document.getElementById("edit-uploadface-btn").addEventListener("click", () => {
  if (currentUser) promptUploadFace(currentUser.accountid);
});

// U19/U21 — admin uploads a facial image for an existing student.
function promptUploadFace(accountId) {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/*";
  input.onchange = async () => {
    if (!input.files[0]) return;
    const fd = new FormData();
    fd.append("account_id", accountId);
    fd.append("file", input.files[0]);
    try {
      const res = await api("/register", {method: "POST", body: fd});
      alert(res.message || (res.success ? "Face registered." : "Face image rejected."));
    } catch (ex) {
      alert("Upload failed: " + ex.message);
    }
  };
  input.click();
}

// ── U23: start a training run and poll its progress ───────────
const fmtPct = v => v == null ? "—" : (v * 100).toFixed(1) + "%";

// Scroll to the U22 form and flash it so the admin sees where step 1 lives.
function highlightTrainingDataForm() {
  const form = document.getElementById("training-data-form");
  if (!form) return;
  form.scrollIntoView({behavior: "smooth", block: "center"});
  form.style.transition = "box-shadow .3s, border-radius .3s";
  form.style.borderRadius = "8px";
  form.style.boxShadow = "0 0 0 3px var(--c-primary-soft), 0 0 0 5px var(--c-primary)";
  setTimeout(() => { form.style.boxShadow = ""; }, 2500);
}

function renderTrainResult(r) {
  const body = document.getElementById("train-result-body");
  body.innerHTML = "";
  body.append(el("tr", {},
    el("td", {}, r.model_name),
    el("td", {}, String(r.new_threshold)),
    el("td", {}, fmtPct(r.accuracy)),
    el("td", {}, fmtPct(r.fpr)),
    el("td", {}, fmtPct(r.fnr)),
    el("td", {}, `${r.genuine_pairs} / ${r.imposter_pairs}`),
  ));
  document.getElementById("train-result").style.display = "";
}

async function pollTraining(statusText, bar) {
  for (;;) {
    await new Promise(r => setTimeout(r, 800));
    const s = await api("/admin/training-status");
    bar.style.width = (s.progress || 0) + "%";
    statusText.textContent = `${s.message || ""} (${s.progress || 0}%)`;
    if (s.status === "done") return s.result;
    if (s.status === "failed") throw new Error(s.error || "Training failed");
  }
}

document.getElementById('recalibrate-btn').addEventListener('click', async () => {
  const btn = document.getElementById('recalibrate-btn');
  const statusText = document.getElementById('recalibrate-status');
  const bar = document.getElementById('train-progress-bar');
  btn.disabled = true;
  btn.textContent = "Training…";
  statusText.style.color = "";
  document.getElementById("train-result").style.display = "none";
  document.getElementById("train-progress-wrap").style.display = "";
  bar.style.width = "0%";
  try {
    await api('/admin/train', {method: 'POST'});
    const result = await pollTraining(statusText, bar);
    renderTrainResult(result);
    let note = `Done in ${result.duration_s}s. New threshold ${result.new_threshold}, ` +
      `accuracy ${fmtPct(result.accuracy)}`;
    if (result.current_accuracy != null) {
      note += ` (current deployment: ${fmtPct(result.current_accuracy)})`;
    }
    if (result.limited_calibration) {
      note += " — calibrated from imposter pairs only (each account has a single embedding).";
    }
    statusText.textContent = note;
    statusText.style.color = "#16a34a";
  } catch (error) {
    document.getElementById("train-progress-wrap").style.display = "none";
    if (/selectTrainingDataSet|dataset/i.test(error.message)) {
      statusText.textContent =
        "Step 1 first: assign the training data above — choose the train split " +
        "and target model, then click “Save Data Assignment”.";
      statusText.style.color = "#c0392b";
      highlightTrainingDataForm();
    } else {
      statusText.textContent = "Error: " + error.message;
      statusText.style.color = "#c0392b";
    }
  } finally {
    btn.disabled = false;
    btn.textContent = "Start Training";
  }
});

// ── U25: deploy the last completed training run ────────────────
document.getElementById("deploy-btn").addEventListener("click", async () => {
  const msg = document.getElementById("deploy-msg");
  msg.style.color = "#333";
  msg.textContent = "Deploying…";
  try {
    let res = await api("/admin/deploy", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({force: false}),
    });
    if (res.warning) {
      if (!confirm(`${res.warning}\nDeploy anyway?`)) {
        msg.style.color = "#c0392b";
        msg.textContent = "Deploy aborted; previous model retained.";
        return;
      }
      res = await api("/admin/deploy", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({force: true}),
      });
    }
    msg.style.color = "#16a34a";
    msg.textContent = `Deployed. Threshold ${res.new_threshold} is now active` +
      (res.applied_live ? " (live pipeline updated)." : " (applies on next restart).");
  } catch (ex) {
    msg.style.color = "#c0392b";
    msg.textContent = ex.message;
  }
});

document.getElementById("user-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = document.getElementById("user-msg");
  msg.textContent = "";
  const formEl = e.target;
  const formData = new FormData(formEl);
  const faceFile = formData.get("face_image");
  formData.delete("face_image");
  const fd = Object.fromEntries(formData);
  if (!fd.student_id) delete fd.student_id;
  try {
    const res = await api("/admin/users", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(fd),
    });
    if (faceFile && faceFile.size && res.account_id) {
      const ffd = new FormData();
      ffd.append("account_id", res.account_id);
      ffd.append("file", faceFile);
      try {
        const fres = await api("/register", {method: "POST", body: ffd});
        if (!fres.success) {
          msg.style.color = "#c0392b";
          msg.textContent = "Account created but face image rejected: " + (fres.message || "no face detected");
          formEl.reset();
          loadUsers();
          return;
        }
      } catch (ex) {
        msg.style.color = "#c0392b";
        msg.textContent = "Account created but face upload failed: " + ex.message;
        formEl.reset();
        loadUsers();
        return;
      }
    }
    msg.style.color = "#16a34a";
    msg.textContent = "Created.";
    formEl.reset();
    loadUsers();
    showUsersView("list");
  } catch (ex) {
    msg.style.color = "#c0392b";
    msg.textContent = ex.message;
  }
});

// Face DB: searchable student picker (find by name or email) ----
let faceUserList = [];

async function loadFaceUserOptions() {
  try {
    const res = await api("/admin/users");
    faceUserList = (res.users || []).filter(u => u.role === "student");
  } catch {
    faceUserList = [];
  }
  renderFaceUserOptions("");
}

function renderFaceUserOptions(query) {
  const sel = document.getElementById("face-account-select");
  if (!sel) return;
  const q = query.trim().toLowerCase();
  const matches = faceUserList.filter(u =>
    !q ||
    (u.full_name || "").toLowerCase().includes(q) ||
    (u.email || "").toLowerCase().includes(q) ||
    (u.student_id || "").toLowerCase().includes(q));
  const prev = sel.value;
  sel.innerHTML = "";
  if (!matches.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = q ? "No matching student" : "No students found";
    opt.disabled = true;
    sel.append(opt);
    return;
  }
  for (const u of matches) {
    const opt = document.createElement("option");
    opt.value = u.accountid;
    const id = u.student_id || ("acc#" + u.accountid);
    opt.textContent = `${u.full_name || u.email} — ${u.email} (${id})`;
    sel.append(opt);
  }
  // Keep the previous selection if it is still in the filtered list.
  if (matches.some(u => String(u.accountid) === prev)) sel.value = prev;
}

document.getElementById("face-user-search").addEventListener("input", (e) => {
  renderFaceUserOptions(e.target.value);
});

// Face DB direct upload form
document.getElementById("face-upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = document.getElementById("face-upload-msg");
  msg.textContent = "";
  const fd = new FormData(e.target);
  try {
    const res = await api("/register", {method: "POST", body: fd});
    msg.style.color = res.success ? "#16a34a" : "#c0392b";
    msg.textContent = res.message || (res.success ? "Face registered." : "Failed.");
    if (res.success) { e.target.reset(); loadFaces(); showFacesView("list"); }
  } catch (ex) {
    msg.style.color = "#c0392b";
    msg.textContent = ex.message;
  }
});

// ── Courses (U26) ─────────────────────────────────────────────
async function loadCourses() {
  const body = document.getElementById("courses-body");
  const select = document.getElementById("session-course-select");
  const enrollSelect = document.getElementById("enroll-course-select");
  body.innerHTML = "";
  select.innerHTML = "";
  if (enrollSelect) enrollSelect.innerHTML = "";
  const res = await api("/admin/courses");
  for (const c of res.courses) {
    body.append(el("tr", {},
      el("td", {}, c.courseid),
      el("td", {}, c.course_code),
      el("td", {}, c.course_name),
      el("td", {}, c.teacher_name || "—"),
      el("td", {}, c.status || "active"),
      el("td", {}, c.active_sessions ?? 0),
      el("td", {}, el("button", {
        class: (c.status === "inactive") ? "secondary" : "danger",
        onclick: async () => {
          const target = c.status === "inactive" ? "active" : "inactive";
          if (target === "inactive" && (c.active_sessions ?? 0) > 0) {
            if (!confirm(`Course ${c.course_code} has ${c.active_sessions} active session(s). Deactivate anyway?`)) return;
          }
          try {
            await api(`/admin/courses/${c.courseid}/status`, {
              method: "PATCH",
              headers: {"Content-Type": "application/json"},
              body: JSON.stringify({status: target}),
            });
            loadCourses();
          } catch (ex) {
            alert(ex.message);
          }
        }
      }, c.status === "inactive" ? "Activate" : "Deactivate")),
      el("td", {}, el("button", {
        class: "danger",
        onclick: async () => {
          if (!confirm(`Permanently delete course ${c.course_code} — ${c.course_name}?`)) return;
          try {
            await api(`/admin/courses/${c.courseid}`, {method: "DELETE"});
            loadCourses();
          } catch (ex) {
            if (/force=true/.test(ex.message)) {
              if (!confirm(`${ex.message}\n\nForce delete (also removes all scheduled sessions)?`)) return;
              try {
                await api(`/admin/courses/${c.courseid}?force=true`, {method: "DELETE"});
                loadCourses();
              } catch (ex2) { alert(ex2.message); }
            } else {
              alert(ex.message);
            }
          }
        }
      }, "Delete")),
    ));
    if ((c.status || "active") === "active") {
      const opt = document.createElement("option");
      opt.value = c.courseid;
      opt.textContent = `${c.course_code} — ${c.course_name}`;
      select.append(opt);
      if (enrollSelect) {
        const opt2 = document.createElement("option");
        opt2.value = c.courseid;
        opt2.textContent = `${c.course_code} — ${c.course_name}`;
        enrollSelect.append(opt2);
      }
    }
  }
  loadSessions();
  loadStudentsForEnrollment();
  loadEnrollments();
}

// Enrolment: searchable student picker (find by name or student ID).
// Mirrors the Face DB picker above — a full class roster is too long to
// scroll, so the list is narrowed by typing rather than by scrolling.
let enrollUserList = [];

async function loadStudentsForEnrollment() {
  try {
    const res = await api("/admin/users");
    enrollUserList = (res.users || []).filter(
      u => u.role === "student" && u.status === "active");
  } catch {
    enrollUserList = [];
  }
  renderEnrollStudentOptions(
    document.getElementById("enroll-student-search")?.value || "");
}

function renderEnrollStudentOptions(query) {
  const sel = document.getElementById("enroll-student-select");
  if (!sel) return;
  const q = query.trim().toLowerCase();
  const matches = enrollUserList.filter(u =>
    !q ||
    (u.full_name || "").toLowerCase().includes(q) ||
    (u.student_id || "").toLowerCase().includes(q) ||
    (u.email || "").toLowerCase().includes(q));
  const prev = sel.value;
  sel.innerHTML = "";
  if (!matches.length) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = q ? "No matching student" : "No students found";
    opt.disabled = true;
    sel.append(opt);
    return;
  }
  for (const u of matches) {
    const opt = document.createElement("option");
    opt.value = u.accountid;
    const id = u.student_id || ("acc#" + u.accountid);
    opt.textContent = `${u.full_name || u.email} — ${id}`;
    sel.append(opt);
  }
  // Keep the previous selection if it is still in the filtered list.
  if (matches.some(u => String(u.accountid) === prev)) sel.value = prev;
}

async function loadEnrollments() {
  const body = document.getElementById("enrollments-body");
  const sel = document.getElementById("enroll-course-select");
  if (!body || !sel) return;
  body.innerHTML = "";
  const courseId = sel.value;
  if (!courseId) return;
  let res;
  try {
    res = await api(`/admin/courses/${courseId}/enrollments`);
  } catch (ex) {
    return;
  }
  for (const e of res.enrollments) {
    body.append(el("tr", {},
      el("td", {}, e.student_id || "-"),
      el("td", {}, e.full_name || "-"),
      el("td", {}, e.email || "-"),
      el("td", {}, e.status),
      el("td", {}, el("button", {
        class: "danger",
        onclick: async () => {
          if (!confirm(`Remove ${e.full_name || e.email} from this course?`)) return;
          try {
            await api(`/admin/courses/${courseId}/enrollments/${e.accountid}`, {method: "DELETE"});
            loadEnrollments();
          } catch (ex) { alert(ex.message); }
        }
      }, "Remove")),
    ));
  }
}

async function loadSessions() {
  const body = document.getElementById("sessions-body");
  body.innerHTML = "";
  const res = await api("/admin/sessions");
  for (const s of res.sessions) {
    body.append(el("tr", {},
      el("td", {}, s.attendancesessionid),
      el("td", {}, `${s.course_code} — ${s.course_name}`),
      el("td", {}, fmt(s.start_time)),
      el("td", {}, s.end_time ? fmt(s.end_time) : "-"),
      el("td", {}, s.status),
      el("td", {},
        // Only a session that has not run yet can be started. "ended" and
        // "cancelled" are terminal: re-activating one would reopen a sitting
        // whose attendance has already been finalised.
        s.status === "scheduled" ? el("button", {
          style: "min-width:64px",
          onclick: () => updateSession(s.attendancesessionid, {status: "active"}),
        }, "Start") : null,
        s.status === "active" ? el("button", {
          class: "secondary",
          style: "min-width:64px",
          onclick: () => updateSession(s.attendancesessionid, {status: "ended"}),
        }, "End") : null,
        el("button", {
          class: "danger",
          style: "margin-left:6px",
          onclick: async () => {
            if (!confirm(`Delete session #${s.attendancesessionid}?`)) return;
            try {
              await api(`/admin/sessions/${s.attendancesessionid}`, {method: "DELETE"});
              loadSessions();
            } catch (ex) { alert(ex.message); }
          }
        }, "Delete"),
      ),
    ));
  }
}

async function updateSession(id, patch) {
  try {
    await api(`/admin/sessions/${id}`, {
      method: "PATCH",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(patch),
    });
    loadSessions();
  } catch (ex) { alert(ex.message); }
}

// ── Class Sessions: list ↔ Schedule Session form sub-pages ────
function showSessionsView(view) {
  document.getElementById("sessions-list").hidden = view !== "list";
  document.getElementById("sessions-form-view").hidden = view !== "form";
}
document.getElementById("show-session-form").addEventListener("click", () => {
  document.getElementById("session-msg").textContent = "";
  showSessionsView("form");
});
document.getElementById("back-sessions").addEventListener("click", () => showSessionsView("list"));

document.getElementById("session-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = document.getElementById("session-msg");
  msg.textContent = "";
  const fd = Object.fromEntries(new FormData(e.target));
  const body = {
    course_id: parseInt(fd.course_id),
    start_time: new Date(fd.start_time).toISOString(),
    end_time: fd.end_time ? new Date(fd.end_time).toISOString() : null,
    status: fd.status || "scheduled",
  };
  try {
    await api("/admin/sessions", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });
    msg.style.color = "#16a34a";
    msg.textContent = "Session scheduled.";
    e.target.reset();
    loadSessions();
    showSessionsView("list");
  } catch (ex) {
    msg.style.color = "#c0392b";
    msg.textContent = ex.message;
  }
});

document.getElementById("enroll-student-search").addEventListener("input", (e) => {
  renderEnrollStudentOptions(e.target.value);
});

document.getElementById("enroll-course-select").addEventListener("change", loadEnrollments);

document.getElementById("enrollment-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = document.getElementById("enrollment-msg");
  msg.textContent = "";
  const fd = Object.fromEntries(new FormData(e.target));
  const courseId = parseInt(fd.course_id);
  const accountId = parseInt(fd.account_id);
  if (!courseId || !accountId) {
    msg.style.color = "#c0392b";
    msg.textContent = "Please select a course and a student.";
    return;
  }
  try {
    await api(`/admin/courses/${courseId}/enrollments`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({account_id: accountId}),
    });
    msg.style.color = "#16a34a";
    msg.textContent = "Student assigned to course.";
    loadEnrollments();
  } catch (ex) {
    msg.style.color = "#c0392b";
    msg.textContent = ex.message;
  }
});

// ── Manage Courses: list ↔ Add Course form sub-pages ──────────
function showManageView(view) {
  document.getElementById("courses-list").hidden = view !== "list";
  document.getElementById("courses-form-view").hidden = view !== "form";
}

async function loadTeacherOptions() {
  const sel = document.getElementById("course-teacher-select");
  if (!sel) return;
  const prev = sel.value;
  sel.innerHTML = '<option value="">— No teacher —</option>';
  try {
    const res = await api("/admin/users");
    for (const u of res.users || []) {
      if (u.role !== "teacher" || u.status !== "active") continue;
      const opt = document.createElement("option");
      opt.value = u.accountid;
      opt.textContent = `${u.full_name || u.email} (${u.email})`;
      sel.append(opt);
    }
  } catch { /* leave just the "no teacher" option */ }
  sel.value = prev;
}

document.getElementById("show-course-form").addEventListener("click", () => {
  document.getElementById("course-msg").textContent = "";
  loadTeacherOptions();
  showManageView("form");
});
document.getElementById("back-courses").addEventListener("click", () => showManageView("list"));

document.getElementById("course-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = document.getElementById("course-msg");
  msg.textContent = "";
  const fd = Object.fromEntries(new FormData(e.target));
  const payload = {course_code: fd.course_code, course_name: fd.course_name};
  if (fd.teacher_id) payload.teacher_id = parseInt(fd.teacher_id);
  try {
    await api("/admin/courses", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    e.target.reset();
    loadCourses();
    showManageView("list");
  } catch (ex) {
    msg.style.color = "#c0392b";
    msg.textContent = ex.message;
  }
});

// ── AI Model Governance (U22-U25) ─────────────────────────────
document.getElementById("training-data-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = document.getElementById("training-data-msg");
  const fd = Object.fromEntries(new FormData(e.target));
  try {
    const res = await api("/admin/training-data", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({train_pct: parseInt(fd.train_pct), model_name: fd.model_name}),
    });
    msg.style.color = "#16a34a";
    msg.textContent = `Saved. Train=${res.train_count}, Test=${res.test_count}.`;
  } catch (ex) {
    msg.style.color = "#c0392b";
    msg.textContent = ex.message;
  }
});

document.getElementById("ensemble-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = document.getElementById("ensemble-msg");
  const form = e.target;
  // Backend expects a list of model names. Two or more => ensemble voting;
  // a single model runs on its own.
  const models = [];
  if (form.use_arcface.checked) models.push("arcface");
  if (form.use_facenet.checked) models.push("facenet");
  if (!models.length) {
    msg.style.color = "#c0392b";
    msg.textContent = "Select at least one model.";
    return;
  }
  try {
    const res = await api("/admin/ensemble", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({models, weighting: form.weighting.value}),
    });
    msg.style.color = "#16a34a";
    msg.textContent = (res.is_ensemble
      ? `Ensemble active (${res.models.join(" + ")}, ${res.weighting} weighting).`
      : `Saved. Single model active: ${res.models.join("")} (no ensemble voting).`)
      + ((res.warnings || []).length ? ` Note: ${res.warnings.join("; ")}` : "");
  } catch (ex) {
    msg.style.color = "#c0392b";
    msg.textContent = ex.message;
  }
});

document.getElementById("retrain-btn").addEventListener("click", async () => {
  const msg = document.getElementById("retrain-msg");
  const btn = document.getElementById("retrain-btn");
  btn.disabled = true;
  msg.style.color = "#333";
  msg.textContent = "Retraining…";
  try {
    let res = await api("/admin/retrain", {method: "POST"});
    if (res.warning) {
      if (!confirm(`${res.warning}\nDeploy anyway?`)) {
        msg.style.color = "#c0392b";
        msg.textContent = "Retrain aborted; previous model retained.";
        return;
      }
      res = await api("/admin/retrain?force=true", {method: "POST"});
    }
    msg.style.color = "#16a34a";
    msg.textContent = `Redeployed ${res.model_name}. New threshold ${res.new_threshold}, ` +
      `accuracy ${(res.accuracy * 100).toFixed(1)}%` +
      (res.applied_live ? " (live)." : " (applies on next restart).");
  } catch (ex) {
    msg.style.color = "#c0392b";
    msg.textContent = ex.message;
  } finally {
    btn.disabled = false;
  }
});

// ── Faces ─────────────────────────────────────────────────────
async function loadFaces() {
  const body = document.getElementById("faces-body");
  body.innerHTML = "";
  const res = await api("/admin/faces");
  for (const f of res.faces) {
    body.append(el("tr", {},
      el("td", {}, f.faceid),
      el("td", {}, f.accountid),
      el("td", {}, f.full_name || "-"),
      el("td", {}, f.student_id || "-"),
      el("td", {}, `${f.model_name} / ${f.model_version}`),
      el("td", {}, f.dimension),
      el("td", {}, f.is_active ? "yes" : "no"),
      el("td", {}, fmt(f.created_at)),
      el("td", {}, f.is_active ? el("button", {
        class: "danger",
        onclick: async () => {
          if (!confirm(`Deactivate faceid ${f.faceid}?`)) return;
          await api(`/admin/faces/${f.faceid}`, {method: "DELETE"});
          loadFaces();
        }
      }, "Delete") : null),
    ));
  }
}

// ── Attendance ────────────────────────────────────────────────
let attAllRecords = [];

async function loadAttendance() {
  const res = await api("/admin/attendance");
  attAllRecords = res.records || [];

  // Populate the status filter with the distinct statuses present.
  const sel = document.getElementById("att-f-status");
  const current = sel.value;
  const statuses = [...new Set(attAllRecords.map(r => r.status).filter(Boolean))].sort();
  sel.innerHTML = '<option value="">All statuses</option>' +
    statuses.map(s => `<option value="${s}">${s}</option>`).join("");
  sel.value = current; // keep selection across reloads

  renderAttendance();
}

// Case-insensitive substring match; empty query always matches.
function attMatch(query, ...fields) {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  return fields.some(f => String(f ?? "").toLowerCase().includes(q));
}

// Local YYYY-MM-DD for a timestamp (so date comparisons are calendar-day based).
function localDateKey(ts) {
  const d = new Date(ts);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

// Inclusive date-range filter. Empty from/to = open-ended; from === to = a
// single day (a point). YYYY-MM-DD strings compare correctly as text.
function attInDateRange(ts, from, to) {
  if (!from && !to) return true;
  if (!ts) return false;
  const ds = localDateKey(ts);
  if (from && ds < from) return false;
  if (to && ds > to) return false;
  return true;
}

function renderAttendance() {
  const body = document.getElementById("att-body");
  body.innerHTML = "";
  const fCourse = document.getElementById("att-f-course").value;
  const fStudent = document.getElementById("att-f-student").value;
  const fDateFrom = document.getElementById("att-f-date-from").value;
  const fDateTo = document.getElementById("att-f-date-to").value;
  const fStatus = document.getElementById("att-f-status").value;

  const rows = attAllRecords.filter(r =>
    attMatch(fCourse, r.course_code, r.course_name) &&
    attMatch(fStudent, r.full_name, r.student_id, r.accountid) &&
    attInDateRange(r.start_time, fDateFrom, fDateTo) &&
    (!fStatus || r.status === fStatus)
  );

  for (const r of rows) {
    body.append(el("tr", {},
      el("td", {}, r.attendancesessionid),
      el("td", {}, `${r.course_code} — ${r.course_name}`),
      el("td", {}, fmt(r.start_time)),
      el("td", {}, `${r.full_name || "-"} (${r.student_id || r.accountid})`),
      el("td", {}, el("span", {class: "status-" + r.status}, r.status)),
      el("td", {}, fmt(r.marked_at)),
    ));
  }
  if (!rows.length) {
    body.append(el("tr", {}, el("td", {colspan: 6, class: "muted"}, "No matching records.")));
  }
  document.getElementById("att-f-count").textContent =
    `${rows.length} / ${attAllRecords.length}`;
}

// Wire up the filter controls (live filtering).
["att-f-course", "att-f-student"].forEach(id =>
  document.getElementById(id).addEventListener("input", renderAttendance));
["att-f-date-from", "att-f-date-to", "att-f-status"].forEach(id =>
  document.getElementById(id).addEventListener("change", renderAttendance));
document.getElementById("att-f-clear").addEventListener("click", () => {
  ["att-f-course", "att-f-student", "att-f-date-from", "att-f-date-to"]
    .forEach(id => document.getElementById(id).value = "");
  document.getElementById("att-f-status").value = "";
  renderAttendance();
});

// ── Appeals ───────────────────────────────────────────────────
async function loadAppeals() {
  try {
    const res = await api("/admin/appeals");
    const appeals = res.appeals || [];
    const body = document.getElementById("appeals-body");
    body.innerHTML = "";

    appeals.forEach(app => {
      const tr = document.createElement("tr");

      tr.appendChild(el("td", {}, app.appealid));
      tr.appendChild(el("td", {}, app.student_id || "-"));
      tr.appendChild(el("td", {}, app.full_name || "-"));
      tr.appendChild(el("td", {}, app.attendancerecordid));
      tr.appendChild(el("td", {}, app.reason || ""));
      tr.appendChild(el("td", {}, app.status));

      // Keep the <td> a real table cell (display:flex on a td detaches it
      // from row-height calculation and misaligns it when other cells wrap);
      // lay the buttons out in an inner flex wrapper instead.
      const actionsTd = document.createElement("td");
      actionsTd.style.whiteSpace = "nowrap";
      const actionsWrap = document.createElement("div");
      actionsWrap.style.display = "flex";
      actionsWrap.style.gap = "8px";
      actionsWrap.style.alignItems = "center";
      actionsTd.appendChild(actionsWrap);

      // U08: appeals are reviewed by the teacher; the admin view is
      // read-only oversight.
      const viewBtn = el("button", { class: "primary small", style: "margin: 0;" }, "View");
      viewBtn.addEventListener("click", () => openAppealDetail(app));
      actionsWrap.appendChild(viewBtn);

      tr.appendChild(actionsTd);
      body.appendChild(tr);
    });
  } catch (e) {
    console.error(e);
  }
}

let adminAppealDocUrl = null;

async function openAppealDetail(appeal) {
  if (adminAppealDocUrl) {
    URL.revokeObjectURL(adminAppealDocUrl);
    adminAppealDocUrl = null;
  }
  document.getElementById("detail-appeal-id").textContent = appeal.appealid || "-";
  document.getElementById("detail-student-id").textContent = appeal.student_id || "-";
  document.getElementById("detail-full-name").textContent = appeal.full_name || "-";
  document.getElementById("detail-record-id").textContent = appeal.attendancerecordid || "-";
  document.getElementById("detail-reason").textContent = appeal.reason || "-";

  document.getElementById("detail-created-at").textContent = fmt(appeal.created_at);
  document.getElementById("detail-reviewed-at").textContent = appeal.reviewed_at ? fmt(appeal.reviewed_at) : "Not reviewed yet";

  // U08 audit trail: who decided this appeal. A reviewer only exists once the
  // appeal leaves 'pending'; the account may also have been deleted since,
  // which leaves reviewed_by set but the joined name NULL.
  const reviewedBy = document.getElementById("detail-reviewed-by");
  const reviewerMeta = document.getElementById("detail-reviewer-meta");
  if (!appeal.reviewed_by) {
    reviewedBy.textContent = "Not reviewed yet";
    reviewerMeta.textContent = "";
  } else {
    reviewedBy.textContent =
      appeal.reviewer_name || `Account #${appeal.reviewed_by} (deleted)`;
    const parts = [];
    if (appeal.reviewer_role) parts.push(appeal.reviewer_role);
    if (appeal.reviewer_staff_id) parts.push(appeal.reviewer_staff_id);
    if (appeal.reviewer_email) parts.push(appeal.reviewer_email);
    reviewerMeta.textContent = parts.join(" · ");
  }

  const statusEl = document.getElementById("detail-status");
  const currentStatus = appeal.status || "pending";
  statusEl.textContent = currentStatus.toUpperCase();
  
  if (currentStatus === "approved") {
    statusEl.style.background = "#e6f4ea";
    statusEl.style.color = "#137333";
    statusEl.style.border = "1px solid #c2e7cd";
  } else if (currentStatus === "rejected") {
    statusEl.style.background = "#fce8e6";
    statusEl.style.color = "#c5221f";
    statusEl.style.border = "1px solid #f9d2cd";
  } else {
    // pending
    statusEl.style.background = "#fef7e0";
    statusEl.style.color = "#b06000";
    statusEl.style.border = "1px solid #fbe4a2";
  }

  showAppealsView("detail");
  adminAppealDocUrl = await renderSupportingDocument({
    url: `/appeals/${appeal.appealid}/document`,
    hasDocument: appeal.has_document,
    name: appeal.supporting_doc_name,
    type: appeal.supporting_doc_type,
    ids: {
      section: "adm-appeal-doc-section",
      image: "adm-appeal-doc-image",
      link: "adm-appeal-doc-link",
      msg: "adm-appeal-doc-msg",
    },
  });
}

function showAppealsView(view) {
  const listView = document.getElementById("appeals-list-view");
  const detailView = document.getElementById("appeals-detail-view");
  if (view === "list") {
    listView.style.display = "";
    detailView.style.display = "none";
    if (adminAppealDocUrl) {
      URL.revokeObjectURL(adminAppealDocUrl);
      adminAppealDocUrl = null;
    }
  } else if (view === "detail") {
    listView.style.display = "none";
    detailView.style.display = "";
  }
}

document.getElementById("btn-back-to-appeals").addEventListener("click", () => {
  showAppealsView("list");
});

// ── Attendance Config (U03 detection interval + U34 thresholds) ───
async function loadAttConfig() {
  const form = document.getElementById("att-config-form");
  const msg = document.getElementById("att-config-msg");
  msg.textContent = "";
  try {
    const cfg = await api("/config/attendance");
    form.detection_interval_seconds.value = cfg.detection_interval_seconds;
    form.late_grace_seconds.value = cfg.late_grace_seconds;
    form.minimum_rate.value = cfg.minimum_attendance_rate;
    form.consecutive_threshold.value = cfg.absence_threshold;
  } catch (e) {
    msg.style.color = "#c0392b";
    msg.textContent = e.message;
  }
}

document.getElementById("att-config-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = document.getElementById("att-config-msg");
  const fd = new FormData(e.target);
  const body = {
    detection_interval_seconds: Number(fd.get("detection_interval_seconds")),
    late_grace_seconds: Number(fd.get("late_grace_seconds")),
    minimum_rate: Number(fd.get("minimum_rate")),
    consecutive_threshold: Number(fd.get("consecutive_threshold")),
  };
  try {
    const res = await api("/admin/config/absence-threshold", {
      method: "PATCH",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });
    msg.style.color = "#16a34a";
    msg.textContent = `Saved. Detection interval ${res.detection_interval_seconds}s · ` +
      `late grace ${res.late_grace_seconds}s · min rate ${res.minimum_attendance_rate}% · ` +
      `reminder after ${res.absence_threshold} sessions.`;
  } catch (ex) {
    msg.style.color = "#c0392b";
    msg.textContent = ex.message;
  }
});

loadUsers();

// ── Behaviour Analysis Settings (U35) ─────────────────────────────
let behaviourCoursesLoaded = false;

async function loadBehaviourTab() {
  const sel = document.getElementById("behaviour-course");
  if (!behaviourCoursesLoaded) {
    const res = await api("/admin/courses");
    sel.innerHTML = "";
    for (const c of res.courses || []) {
      const o = document.createElement("option");
      o.value = c.courseid;
      o.textContent = `${c.course_code} — ${c.course_name}`;
      sel.appendChild(o);
    }
    behaviourCoursesLoaded = true;
  }
  loadBehaviourConfig();
}

async function loadBehaviourConfig() {
  const cid = document.getElementById("behaviour-course").value;
  const msg = document.getElementById("behaviour-msg");
  if (!cid) return;
  msg.textContent = "";
  try {
    const res = await api(`/admin/courses/${cid}/behaviour-analysis`);
    const cfg = res.config || {};
    document.getElementById("behaviour-enable").checked = !!cfg.enabled;
    document.getElementById("behaviour-drowsiness").checked = !!cfg.drowsiness;
    document.getElementById("behaviour-phone").checked = !!cfg.phone_usage;
    document.getElementById("behaviour-heatmap").checked = !!cfg.heatmap;
    // Tuning: null means "no override", shown as an empty field whose
    // placeholder reads "server default".
    document.getElementById("behaviour-adaptive").checked = cfg.adaptive_ear !== false;
    renderBehaviourSliders();
    for (const f of BEHAVIOUR_TUNING_FIELDS) {
      const range = document.getElementById(`beh-range-${f.key}`);
      const overridden = cfg[f.key] != null;
      range.dataset.overridden = String(overridden);
      range.value = overridden ? cfg[f.key] : f.fallback;
      range._sync();
    }
  } catch (e) {
    msg.style.color = "#c0392b";
    msg.textContent = e.message;
  }
}

// Per-course detection thresholds (U35), rendered as sliders.
//
// `sensitive` says which direction makes the detector fire MORE often — it
// is not the same for every field, which is exactly why a bare number box
// was confusing: raising the eye threshold makes the model more trigger
// happy, while raising the phone threshold makes it more conservative.
// `up` / `down` are shown live as the slider moves.
const BEHAVIOUR_TUNING_FIELDS = [
  {
    key: "ear_threshold",
    label: "Eye-closure (EAR) threshold",
    min: 0.05, max: 0.60, step: 0.01, fallback: 0.21, decimals: 2,
    sensitive: "up",
    lowLabel: "0.05 · strict", highLabel: "0.60 · sensitive",
    up: "Eyes count as closed sooner. Catches light drowsiness, but alert students — " +
        "especially anyone with naturally narrow eyes — get flagged more often.",
    down: "Eyes must be more fully closed before it counts. Far fewer false alarms; " +
          "brief or partial drowsiness may go unrecorded.",
    note: "Only used before individual calibration finishes, or when calibration is off.",
  },
  {
    key: "mar_threshold",
    label: "Yawn (MAR) threshold",
    min: 0.20, max: 1.50, step: 0.05, fallback: 0.60, decimals: 2,
    sensitive: "down",
    lowLabel: "0.20 · sensitive", highLabel: "1.50 · strict",
    up: "The mouth must open wider to count as a yawn. Talking and laughing stop " +
        "triggering it; small yawns are missed.",
    down: "Smaller mouth movements count as yawns. Catches more real yawns, but " +
          "speaking students may be flagged.",
  },
  {
    key: "headpose_pitch_deg",
    label: "Head-tilt limit (degrees)",
    min: 5, max: 89, step: 1, fallback: 30, decimals: 0,
    sensitive: "down",
    lowLabel: "5° · sensitive", highLabel: "89° · strict",
    up: "The head must be further down before it counts. Students looking at their " +
        "desk or notes are left alone; genuine head-nodding may be missed.",
    down: "A slight downward tilt already counts. Catches head-nodding early, but " +
          "anyone writing or reading on the desk is flagged.",
  },
  {
    key: "phone_conf",
    label: "Phone detection confidence",
    min: 0.05, max: 0.95, step: 0.05, fallback: 0.45, decimals: 2,
    sensitive: "down",
    lowLabel: "0.05 · sensitive", highLabel: "0.95 · strict",
    up: "The detector must be more certain before calling something a phone. " +
        "Cuts false alarms from pencil cases, calculators and hands; partly hidden " +
        "phones are missed.",
    down: "Weaker detections count too. Finds phones held low or half-covered, at the " +
          "cost of more objects being mistaken for one.",
  },
  {
    key: "drowsy_confirm_seconds",
    label: "Confirm drowsiness after (seconds)",
    min: 1, max: 120, step: 1, fallback: 2, decimals: 0,
    sensitive: "down",
    lowLabel: "1s · sensitive", highLabel: "120s · strict",
    up: "The state must persist longer before it is recorded. Blinks and glances are " +
        "ignored; a short nap may end before it is confirmed.",
    down: "Reacts faster. Short lapses are captured, but ordinary blinking and " +
          "looking down start producing events.",
  },
];

// Build the slider controls once, from the metadata above.
function renderBehaviourSliders() {
  const host = document.getElementById("behaviour-sliders");
  if (!host || host.childElementCount) return;
  for (const f of BEHAVIOUR_TUNING_FIELDS) {
    const value = el("span", {class: "beh-slider__value beh-slider__value--default"});
    const reset = el("button", {
      type: "button", class: "secondary beh-slider__reset",
      title: "Clear this course's override and follow the server default",
    }, "Use default");
    const range = el("input", {
      type: "range", id: `beh-range-${f.key}`,
      min: f.min, max: f.max, step: f.step, value: f.fallback,
    });
    const effect = el("p", {class: "beh-slider__effect"});

    // `overridden` is the difference between "this course sets 0.21" and
    // "this course follows whatever the server default happens to be".
    range.dataset.overridden = "false";
    const sync = () => {
      const overridden = range.dataset.overridden === "true";
      const current = Number(range.value);
      value.textContent = overridden
        ? current.toFixed(f.decimals)
        : `${f.fallback.toFixed(f.decimals)} (server default)`;
      value.classList.toggle("beh-slider__value--default", !overridden);
      reset.disabled = !overridden;
      describeBehaviourEffect(f, current, overridden, effect);
    };
    range.addEventListener("input", () => {
      range.dataset.overridden = "true";
      sync();
    });
    reset.addEventListener("click", () => {
      range.dataset.overridden = "false";
      range.value = f.fallback;
      sync();
    });
    range._sync = sync;  // used by loadBehaviourConfig after fetching values

    host.append(el("div", {class: "beh-slider"},
      el("div", {class: "beh-slider__head"},
        el("span", {class: "beh-slider__name"}, f.label), value, reset),
      range,
      el("div", {class: "beh-slider__scale"},
        el("span", {}, f.lowLabel), el("span", {}, f.highLabel)),
      effect,
      f.note ? el("p", {class: "muted small", style: "margin:.35rem 0 0"}, f.note) : null,
    ));
    sync();
  }
}

// Explain what the current position does, relative to the default.
function describeBehaviourEffect(f, current, overridden, node) {
  node.classList.remove("beh-slider__effect--sensitive", "beh-slider__effect--strict");
  if (!overridden || Math.abs(current - f.fallback) < f.step / 2) {
    node.textContent = "Using the server default for this course.";
    return;
  }
  const raised = current > f.fallback;
  const moreSensitive = (f.sensitive === "up") === raised;
  node.classList.add(moreSensitive ? "beh-slider__effect--sensitive" : "beh-slider__effect--strict");
  node.textContent =
    `${raised ? "Higher" : "Lower"} than the default (${f.fallback.toFixed(f.decimals)}) — ` +
    `${moreSensitive ? "MORE sensitive: " : "MORE conservative: "}` +
    (raised ? f.up : f.down);
}

document.getElementById("behaviour-course").addEventListener("change", loadBehaviourConfig);

document.getElementById("behaviour-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const cid = document.getElementById("behaviour-course").value;
  const msg = document.getElementById("behaviour-msg");
  if (!cid) { msg.textContent = "Select a course."; return; }
  const body = {
    enable: document.getElementById("behaviour-enable").checked,
    drowsiness: document.getElementById("behaviour-drowsiness").checked,
    phone_usage: document.getElementById("behaviour-phone").checked,
    heatmap: document.getElementById("behaviour-heatmap").checked,
    adaptive_ear: document.getElementById("behaviour-adaptive").checked,
  };
  // An empty field sends null, which clears the override server-side and
  // returns the course to the server default — that is how an admin undoes
  // a setting without needing to know the default value.
  for (const f of BEHAVIOUR_TUNING_FIELDS) {
    const range = document.getElementById(`beh-range-${f.key}`);
    body[f.key] = range && range.dataset.overridden === "true" ? Number(range.value) : null;
  }
  try {
    const res = await api(`/admin/courses/${cid}/behaviour-analysis`, {
      method: "PATCH",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });
    const cfg = res.config || {};
    msg.style.color = "#16a34a";
    const overrides = BEHAVIOUR_TUNING_FIELDS.filter(f => cfg[f.key] != null).length;
    msg.style.color = "#16a34a";
    msg.textContent =
      `Saved. Behaviour analysis ${cfg.enabled ? "ENABLED" : "disabled"} for this course` +
      (cfg.enabled ? ` · ${overrides} threshold override(s) · individual eye calibration ${cfg.adaptive_ear === false ? "off" : "on"}.` : ".");
  } catch (ex) {
    msg.style.color = "#c0392b";
    msg.textContent = ex.message;
  }
});