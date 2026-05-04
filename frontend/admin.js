const user = requireAuth("admin");
document.getElementById("who").textContent = `${user.full_name || user.email} (admin)`;

// ── Tabs ──────────────────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    ["users", "faces", "courses", "attendance", "appeals", "ai-config", "live-scanner"].forEach(name => {
      document.getElementById("tab-" + name).style.display =
        name === btn.dataset.tab ? "" : "none";
    });
    if (btn.dataset.tab === "faces") loadFaces();
    if (btn.dataset.tab === "courses") loadCourses();
    if (btn.dataset.tab === "attendance") loadAttendance();
    if (btn.dataset.tab === "appeals") loadAppeals();
    if (btn.dataset.tab === "ai-config") {}
    if (btn.dataset.tab === "live-scanner") {}
  });
});

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
          class: u.status === "active" ? "danger" : "secondary",
          onclick: async () => {
            await api(`/admin/users/${u.accountid}/status`, {
              method: "PATCH",
              headers: {"Content-Type": "application/json"},
              body: JSON.stringify({status: u.status === "active" ? "inactive" : "active"}),
            });
            loadUsers();
          }
        }, u.status === "active" ? "Deactivate" : "Activate"),
        u.role === "student" ? el("button", {
          class: "secondary",
          style: "margin-left:6px",
          onclick: () => promptUploadFace(u.accountid),
        }, "Upload Face") : null,
      ),
    ));
  }
}

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

document.getElementById('recalibrate-btn').addEventListener('click', async () => {
    const btn = document.getElementById('recalibrate-btn');
    const statusText = document.getElementById('recalibrate-status');
    
    // Disable button to prevent double-clicking while the heavy GPU loads
    btn.disabled = true;
    btn.textContent = "Running calibration…";
    statusText.innerText = "Running StyleGAN — this may take a few minutes. Check the backend terminal for progress.";
    statusText.style.color = "";

    try {
        // Use your existing api() wrapper which automatically handles the auth token
        const result = await api('/admin/recalibrate', {
            method: 'POST'
        });

        if (result.status === 'success') {
            statusText.innerText = `✅ Success! Threshold automatically updated to: ${result.new_threshold}`;
            statusText.style.color = "#16a34a"; // Green
        } else {
            statusText.innerText = `❌ Error: Something went wrong.`;
            statusText.style.color = "#c0392b"; // Red
        }
    } catch (error) {
        statusText.innerText = `❌ Connection error: ${error.message}`;
        statusText.style.color = "#c0392b"; // Red
    } finally {
        // Re-enable the button
        btn.disabled = false;
        btn.textContent = "Start Training";
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
  } catch (ex) {
    msg.style.color = "#c0392b";
    msg.textContent = ex.message;
  }
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
    if (res.success) { e.target.reset(); loadFaces(); }
  } catch (ex) {
    msg.style.color = "#c0392b";
    msg.textContent = ex.message;
  }
});

// ── Courses (U26) ─────────────────────────────────────────────
async function loadCourses() {
  const body = document.getElementById("courses-body");
  const select = document.getElementById("session-course-select");
  body.innerHTML = "";
  select.innerHTML = "";
  const res = await api("/admin/courses");
  for (const c of res.courses) {
    body.append(el("tr", {},
      el("td", {}, c.courseid),
      el("td", {}, c.course_code),
      el("td", {}, c.course_name),
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
    ));
    if ((c.status || "active") === "active") {
      const opt = document.createElement("option");
      opt.value = c.courseid;
      opt.textContent = `${c.course_code} — ${c.course_name}`;
      select.append(opt);
    }
  }
  loadSessions();
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
        s.status !== "active" ? el("button", {
          onclick: () => updateSession(s.attendancesessionid, {status: "active"}),
        }, "Start") : null,
        s.status === "active" ? el("button", {
          class: "secondary",
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
  } catch (ex) {
    msg.style.color = "#c0392b";
    msg.textContent = ex.message;
  }
});

document.getElementById("course-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = document.getElementById("course-msg");
  msg.textContent = "";
  const fd = Object.fromEntries(new FormData(e.target));
  try {
    await api("/admin/courses", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(fd),
    });
    msg.style.color = "#16a34a";
    msg.textContent = "Course created.";
    e.target.reset();
    loadCourses();
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
  const body = {
    use_arcface: form.use_arcface.checked,
    use_facenet: form.use_facenet.checked,
    weighting: form.weighting.value,
  };
  if (!body.use_arcface && !body.use_facenet) {
    msg.style.color = "#c0392b";
    msg.textContent = "Select at least one model.";
    return;
  }
  if ([body.use_arcface, body.use_facenet].filter(Boolean).length < 2) {
    msg.style.color = "#c0392b";
    msg.textContent = "Ensemble requires at least two models (per U24).";
    return;
  }
  try {
    await api("/admin/ensemble", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body),
    });
    msg.style.color = "#16a34a";
    msg.textContent = "Ensemble configuration saved.";
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
    const res = await api("/admin/retrain", {method: "POST"});
    if (res.warning) {
      if (!confirm(`${res.warning}\nDeploy anyway?`)) {
        msg.style.color = "#c0392b";
        msg.textContent = "Retrain aborted; previous model retained.";
        return;
      }
      await api("/admin/retrain?force=true", {method: "POST"});
    }
    msg.style.color = "#16a34a";
    msg.textContent = `Redeployed. New threshold: ${res.new_threshold}`;
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
async function loadAttendance() {
  const body = document.getElementById("att-body");
  body.innerHTML = "";
  const res = await api("/admin/attendance");
  for (const r of res.records) {
    body.append(el("tr", {},
      el("td", {}, r.attendancesessionid),
      el("td", {}, `${r.course_code} — ${r.course_name}`),
      el("td", {}, fmt(r.start_time)),
      el("td", {}, `${r.full_name || "-"} (${r.student_id || r.accountid})`),
      el("td", {class: "status-" + r.status}, r.status),
      el("td", {}, fmt(r.marked_at)),
    ));
  }
}

// ── Appeals ───────────────────────────────────────────────────
async function loadAppeals() {
  const body = document.getElementById("appeals-body");
  body.innerHTML = "";
  const res = await api("/admin/appeals");
  for (const a of res.appeals) {
    const canReview = a.status === "pending";
    body.append(el("tr", {},
      el("td", {}, a.appealid),
      el("td", {}, `${a.full_name || "-"} (${a.student_id || a.accountid})`),
      el("td", {}, a.attendancerecordid),
      el("td", {}, a.reason),
      el("td", {}, el("span", {class: "badge"}, a.status)),
      el("td", {}, fmt(a.created_at)),
      el("td", {},
        canReview ? el("button", {
          onclick: () => reviewAppeal(a.appealid, "approved"),
        }, "Approve") : null,
        canReview ? el("button", {
          class: "danger",
          onclick: () => reviewAppeal(a.appealid, "rejected"),
        }, "Reject") : null,
      ),
    ));
  }
}

async function reviewAppeal(id, status) {
  await api(`/admin/appeals/${id}`, {
    method: "PATCH",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({status}),
  });
  loadAppeals();
}

loadUsers();

// ── Live Scanner (Webcam) ─────────────────────────────────────────
document.getElementById("start-webcam-btn").addEventListener("click", async () => {
  const btn = document.getElementById("start-webcam-btn");
  const msg = document.getElementById("scan-status-msg");
  const video = document.getElementById("webcam-feed");
  const canvas = document.getElementById("snapshot-canvas");
  const ctx = canvas.getContext("2d");
  const placeholder = document.getElementById("camera-placeholder");

  const totalScans = parseInt(document.getElementById("total-scans").value);
  const intervalSeconds = parseInt(document.getElementById("scan-interval").value);
  
  btn.disabled = true;
  btn.textContent = "Scanning…";
  msg.textContent = "Requesting webcam access…";
  msg.style.color = "";
  
  try {
    // 1. Turn on the Webcam
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.style.display = "block";
    placeholder.style.display = "none";
    
    // 2. Tell the backend we are starting a new session
    const initRes = await api("/admin/start-webcam-scan", { method: "POST" });
    const trackingId = initRes.tracking_id;
    
    let scansCompleted = 0;
    msg.textContent = `Starting ${totalScans} scans, every ${intervalSeconds} seconds...`;

    // 3. Start the looping interval
    const scanIntervalId = setInterval(async () => {
      scansCompleted++;
      msg.textContent = `Performing scan ${scansCompleted} of ${totalScans}...`;
      
      // Take a snapshot
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const base64Image = canvas.toDataURL("image/jpeg");
      
      // Send the snapshot to FastAPI for recognition
      await api("/admin/process-webcam-frame", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ image: base64Image, tracking_id: trackingId })
      });
      
      // 4. If all scans are done, calculate the 70% rule and shut down
      if (scansCompleted >= totalScans) {
        clearInterval(scanIntervalId);
        
        // Turn off the webcam light
        stream.getTracks().forEach(track => track.stop()); 
        video.style.display = "none";
        placeholder.style.display = "block";
        
        msg.textContent = "Scans complete! Calculating 70% rule logic...";
        
        const finalRes = await api(`/admin/finalize-webcam-scan?tracking_id=${trackingId}&total_scans=${totalScans}`, {
          method: "POST"
        });
        
        msg.style.color = "var(--c-success)";
        msg.textContent = `Finished — ${finalRes.present_count} retained as present, ${finalRes.absent_count} changed to absent.`;
        loadAttendance(); // Refresh table

        btn.disabled = false;
        btn.textContent = "Start Periodic Webcam Scan";
      }
    }, intervalSeconds * 1000); // Convert seconds to milliseconds
    
  } catch (err) {
    msg.style.color = "var(--c-danger)";
    msg.textContent = "Camera error: " + err.message + " (please allow camera permissions).";
    btn.disabled = false;
    btn.textContent = "Start Periodic Webcam Scan";
  }
});