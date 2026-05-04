const user = requireAuth("student");
document.getElementById("who").textContent = `${user.full_name || user.email} (student)`;

// ── Tabs ──────────────────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    ["checkin", "attendance", "appeals", "face"].forEach(name => {
      document.getElementById("tab-" + name).style.display =
        name === btn.dataset.tab ? "" : "none";
    });
  });
});

// ── Attendance list ───────────────────────────────────────────
async function loadAttendance() {
  const body = document.getElementById("att-body");
  body.innerHTML = "";
  try {
    const res = await api("/student/attendance");
    if (!res.records.length) {
      body.append(el("tr", {}, el("td", {colspan: 5, class: "muted"}, "No records yet.")));
      return;
    }
    for (const r of res.records) {
      body.append(el("tr", {},
        el("td", {}, `${r.course_code} — ${r.course_name}`),
        el("td", {}, fmt(r.start_time)),
        el("td", {class: "status-" + r.status}, r.status),
        el("td", {}, fmt(r.marked_at)),
        el("td", {}, el("button", {
          class: "secondary",
          onclick: () => {
            document.querySelector('[data-tab="appeals"]').click();
            document.querySelector('#appeal-form [name="record_id"]').value = r.record_id;
          }
        }, "Appeal")),
      ));
    }
  } catch (e) {
    body.append(el("tr", {}, el("td", {colspan: 5, class: "error"}, e.message)));
  }
}

// ── Appeals ───────────────────────────────────────────────────
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
    msg.textContent = "无法打开摄像头: " + e.message;
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

// ── Face Check-in ─────────────────────────────────────────────
const cam = document.getElementById("cam");
const canvas = document.getElementById("cam-canvas");
const btnStart = document.getElementById("btn-start");
const btnCap = document.getElementById("btn-capture");
const btnStop = document.getElementById("btn-stop");
const ciMsg = document.getElementById("checkin-msg");
let camStream = null;

btnStart.addEventListener("click", async () => {
  ciMsg.textContent = "";
  try {
    camStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    cam.srcObject = camStream;
    btnStart.disabled = true;
    btnCap.disabled = false;
    btnStop.disabled = false;
  } catch (e) {
    ciMsg.style.color = "#c0392b";
    ciMsg.textContent = "无法打开摄像头: " + e.message;
  }
});

btnStop.addEventListener("click", stopCamera);

function stopCamera() {
  if (camStream) camStream.getTracks().forEach(t => t.stop());
  camStream = null;
  cam.srcObject = null;
  btnStart.disabled = false;
  btnCap.disabled = true;
  btnStop.disabled = true;
}

btnCap.addEventListener("click", async () => {
  if (!camStream) return;
  ciMsg.style.color = "#555";
  ciMsg.textContent = "识别中，请稍候...";
  btnCap.disabled = true;

  const w = cam.videoWidth || 640, h = cam.videoHeight || 480;
  canvas.width = w; canvas.height = h;
  canvas.getContext("2d").drawImage(cam, 0, 0, w, h);

  canvas.toBlob(async (blob) => {
    const fd = new FormData();
    fd.append("file", blob, "checkin.jpg");
    try {
      const res = await api("/student/checkin", { method: "POST", body: fd });
      if (res.success) {
        ciMsg.style.color = res.already_checked_in ? "#d97706" : "#16a34a";
        ciMsg.textContent = `✓ ${res.message}  (confidence=${res.confidence})`;
        stopCamera();
        loadAttendance();
      } else {
        ciMsg.style.color = "#c0392b";
        ciMsg.textContent = "✗ " + res.message;
        btnCap.disabled = false;
      }
    } catch (ex) {
      ciMsg.style.color = "#c0392b";
      ciMsg.textContent = ex.message;
      btnCap.disabled = false;
    }
  }, "image/jpeg", 0.92);
});

// Auto-stop camera when leaving the page/tab
window.addEventListener("pagehide", stopCamera);

loadAttendance();
loadAppeals();
