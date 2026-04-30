const user = requireAuth("admin");
document.getElementById("who").textContent = `${user.full_name || user.email} (admin)`;

// ── Tabs ──────────────────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    ["users", "faces", "attendance", "appeals", "ai-config", "live-scanner"].forEach(name => {
      document.getElementById("tab-" + name).style.display =
        name === btn.dataset.tab ? "" : "none";
    });
    if (btn.dataset.tab === "faces") loadFaces();
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
      el("td", {}, el("button", {
        class: u.status === "active" ? "danger" : "secondary",
        onclick: async () => {
          await api(`/admin/users/${u.accountid}/status`, {
            method: "PATCH",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({status: u.status === "active" ? "inactive" : "active"}),
          });
          loadUsers();
        }
      }, u.status === "active" ? "Deactivate" : "Activate")),
    ));
  }
}

document.getElementById('recalibrate-btn').addEventListener('click', async () => {
    const btn = document.getElementById('recalibrate-btn');
    const statusText = document.getElementById('recalibrate-status');
    
    // Disable button to prevent double-clicking while the heavy GPU loads
    btn.disabled = true;
    btn.style.backgroundColor = "gray";
    btn.textContent = "Running Calibration...";
    statusText.innerText = "Running StyleGAN... This may take a few minutes. Check your backend terminal!";
    statusText.style.color = "black";

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
        btn.style.backgroundColor = "#4CAF50";
        btn.textContent = "Run StyleGAN Calibration";
    }
});

document.getElementById("user-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = document.getElementById("user-msg");
  msg.textContent = "";
  const fd = Object.fromEntries(new FormData(e.target));
  if (!fd.student_id) delete fd.student_id;
  try {
    await api("/admin/users", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(fd),
    });
    msg.style.color = "#16a34a";
    msg.textContent = "Created.";
    e.target.reset();
    loadUsers();
  } catch (ex) {
    msg.style.color = "#c0392b";
    msg.textContent = ex.message;
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
  btn.style.backgroundColor = "gray";
  msg.textContent = "Requesting webcam access...";
  msg.style.color = "#333";
  
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
        
        msg.style.color = "#16a34a"; // Green
        msg.textContent = `Finished! ${finalRes.present_count} retained as present, ${finalRes.absent_count} changed to absent.`;
        loadAttendance(); // Refresh table
        
        btn.disabled = false;
        btn.style.backgroundColor = "#2563eb";
        btn.textContent = "Start Periodic Webcam Scan";
      }
    }, intervalSeconds * 1000); // Convert seconds to milliseconds
    
  } catch (err) {
    msg.style.color = "#c0392b";
    msg.textContent = "Camera Error: " + err.message + " (Make sure you allow camera permissions!)";
    btn.disabled = false;
    btn.style.backgroundColor = "#2563eb";
  }
});