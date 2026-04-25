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
document.getElementById("face-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = document.getElementById("face-msg");
  msg.textContent = "Uploading...";
  const fd = new FormData(e.target);
  fd.append("account_id", user.account_id);
  try {
    const res = await api("/register", {method: "POST", body: fd});
    msg.style.color = res.success ? "#16a34a" : "#c0392b";
    msg.textContent = res.message;
  } catch (ex) {
    msg.style.color = "#c0392b";
    msg.textContent = ex.message;
  }
});

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
