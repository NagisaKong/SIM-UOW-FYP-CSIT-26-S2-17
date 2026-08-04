# FYP-26-S2-17 — Face Recognition Attendance System

> An AI-powered student attendance tracking system using deep learning facial recognition, built for SIM Global Education / University of Wollongong.

---

## Overview

This project replaces manual roll calls and QR code check-ins with automated facial recognition of the seated class. Check-in is passive (CR-05): the teacher starts a camera scan for the whole class, and the system opens periodic detection windows, recognises every enrolled student present, and classifies each as **Present**, **Late**, **Early-left** or **Absent**. Students take no individual check-in action.

Built on an ensemble of state-of-the-art deep learning models (SCRFD + ArcFace as primary, MTCNN + FaceNet as secondary), with optional image enhancement for non-ideal lighting conditions. All biometric data stays in a single-tenant, Singapore-region managed database (Supabase) with no third-party sharing, in line with Singapore PDPC guidelines under the Personal Data Protection Act 2012.

The system additionally performs classroom behaviour analysis (CR-06) — drowsiness, mobile-phone use and a spatial activity heatmap — on frames sampled at ~1 fps. Only derived events are persisted; no video is ever written to disk.

---

## Team

> Roles below mirror the Final Technical Documentation §F.7.

| Name | Role |
|------|------|
| YU, ZHANGHAO | Project Leader / Producer · Computer Vision & Backend Developer · Lead AI Engineer |
| WHYE LI HENG, DOMINIC | Lead AI Engineer · Computer Vision & Backend Developer |
| ZHANG, CHENGWEI | Lead AI Engineer · Computer Vision & Backend Developer · UI/UX Designer |
| ZHANG, JIQIAN | Lead AI Engineer · Computer Vision & Backend Developer · QA Engineer & Documentation Lead |
| ZHAO, SHIYIN | UI/UX Designer & Frontend Developer · QA Engineer & Documentation Lead |

---

## Key Features

- **Automated attendance logging** — the teacher starts a whole-class camera scan and every enrolled student in frame is identified; no manual check-in required (CR-05)
- **In-class presence verification** — periodic checks detect students who sign in and leave early
- **Present / Late / Absent classification** — configurable time thresholds determine attendance status
- **Basic anti-spoofing heuristic** — faces below a minimum pixel size are discarded (`AI_ANTISPOOF_MIN_FACE_PX`); a full liveness model is out of scope for this release
- **Ensemble multi-model voting** — SCRFD + ArcFace and MTCNN + FaceNet vote to improve accuracy
- **Low-light enhancement** — CLAHE by default, with an optional face-restoration GAN (GFPGAN / Real-ESRGAN) applied only to frames detected as low-light; StyleGAN data augmentation runs strictly offline (CR-04)
- **Role-based access** — separate dashboards for Students, Teachers, and Administrators
- **Classroom behaviour analysis** — drowsiness (EAR/MAR/head-pose with an adaptive per-student baseline and PERCLOS), mobile-phone use (YOLO, attributed to the nearest recognised student), and a spatial activity heatmap; per-course toggle and tuning
- **ML model management** — admins can split train/test embeddings, calibrate the recognition similarity threshold, review accuracy/FPR/FNR, and deploy it from the web interface (the recognition networks themselves are pre-trained)
- **Accuracy statistics dashboard** — visualises balanced accuracy, FPR and FNR for each calibration run against the currently deployed threshold
- **PDPC compliant** — single-tenant Singapore-region database, no third-party data sharing

---

## System Architecture

```
Classroom Camera(s) (browser, getUserMedia)
          ↓ Individual JPEG frames over HTTPS multipart
          ↓ (no continuous stream, no video file on disk)
    FastAPI Backend
          ↓
      AI Module
      ├── Pre-processing         CLAHE, or an optional face-restoration GAN
      │                          (GFPGAN / Real-ESRGAN) on low-light frames only;
      │                          disabled on CPU hosts via AI_USE_ENHANCER=false
      ├── Face Detection         SCRFD (primary) + MTCNN (ensemble)
      ├── Face Recognition       ArcFace (primary) + FaceNet (ensemble)
      └── Voting Aggregation     final prediction
          ↓
    PostgreSQL Database
          ↑
    Static Web Frontend (HTML5 / CSS3 / Vanilla JS)
    Student / Teacher / Admin dashboards
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3, Vanilla JavaScript (no build step — see CR-01) |
| Backend | Python, FastAPI |
| Database | PostgreSQL |
| AI — Detection | SCRFD (InsightFace), MTCNN |
| AI — Recognition | ArcFace (InsightFace), FaceNet |
| AI — Enhancement | StyleGAN / StarGAN, GAN (Super-Resolution / Image Enhancement) |
| Image Processing | OpenCV |
| Deployment | Docker, Vercel (frontend), Railway (backend) |
| CI | GitHub Actions — ruff lint, schema apply, backend import + pytest (CD deferred, see Branch Strategy) |

---

## Project Structure

```
SIM-UOW-FYP-CSIT-26-S2-17/
│
├── main_api.py                     # Entry point: FastAPI app + register blueprints + run()
│
├── core/
│   ├── __init__.py                 # Aggregate and export 9 blueprints
│   │
│   ├── attendancePipeline.py       # Shared AI/DB layer: AIConfig, detectors,
│   │                               # recognisers, EmbeddingRepo — not a business class
│   │
│   ├── userInformation.py          # Class 1  (U01,U02,U10,U11,U17,U18,U20)
│   │
│   ├── attendanceRecord.py         # Class 2  (U03,U04,U12,U13,U16,U27,U30,U34)
│   │
│   ├── notification.py             # Class 3  (U05,U29)
│   │
│   ├── facialImage.py              # Class 4  (U06,U09,U19,U21)
│   │
│   ├── attendanceSession.py        # Class 5  (U07,U15,U26)
│   │
│   ├── attendanceAppeal.py         # Class 6  (U08,U28,U31)
│   │
│   ├── report.py                   # Class 7  (U14)
│   │
│   ├── behaviourAnalysis.py        # Class 8  (U32,U33,U35 + CR-06 detection service)
│   │
│   ├── trainConfiguration.py       # Class 9  (U22,U23,U24,U25)
│   │
│   └── training/                   # Offline GAN toolchain (CR-04) — StyleGAN
│                                   # synthetic pairs + threshold calibration.
│                                   # Never invoked during live recognition.
│
├── database/
│   ├── schema.sql
│   └── seed_demo.sql
│
├── frontend/
│   ├── admin.html / admin.js
│   ├── teacher.html / teacher.js
│   ├── student.html / student.js
│   ├── index.html / app.js
│   ├── config.js                   # Deployment default for window.API_BASE
│   ├── style.css
│   └── vercel.json
│
├── tests/                          # pytest: conftest.py (stub pipeline),
│                                   # test_system_testing.py (ST-* suite),
│                                   # test_behaviour_analysis.py, test_face_matching.py
│
├── docs/evidence/                  # Generated system-test result tables
├── scripts/prefetch_models.py      # Build-time InsightFace model download
├── .github/workflows/ci.yml
│
├── Dockerfile / .dockerignore / railway.json
├── requirements.txt
├── ruff.toml
└── README.md / DEPLOY.md / LICENSE / CONTRIBUTING.md
```

---
## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Production-ready code. Deployment to Railway/Vercel is triggered manually — there is no CD workflow; CI (`.github/workflows/ci.yml`) gates every PR |
| `develop` | Integration branch for completed features |
| `feature/XXX` | new features, branched from `develop` |
---

## User Roles

| Role | Key Capabilities |
|------|----------------|
| Student | View attendance records and analytics, submit appeals and leave applications, register/update own facial image |
| Teacher | Run classroom scans, real-time attendance dashboard, early-departure summary, behaviour report & heatmap, review appeals and leave applications, export CSV reports |
| Admin | Manage users and courses, facial image database, threshold calibration & deployment, ensemble configuration, reminder and behaviour-analysis settings |

> Attendance statuses are corrected through the appeal (U08) and leave-approval (U31) workflows — there is no direct free-form status-edit endpoint.

---

## References

- Guo, J. et al. (2022). *SCRFD: Sample and Computation Redistribution for Efficient Face Detection*. ICLR 2022. arXiv:2105.04714
- Deng, J. et al. (2019). *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*. CVPR 2019.
- Zhang, K. et al. (2016). *Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks*. IEEE Signal Processing Letters, 23(10).
- Schroff, F. et al. (2015). *FaceNet: A Unified Embedding for Face Recognition and Clustering*. CVPR 2015.
- Personal Data Protection Commission Singapore. (2022). *Guide on Responsible Use of Biometric Data in Security Applications*.

---

## License

This project is developed for academic purposes under CSIT321 at the University of Wollongong (SIM campus). All rights reserved by the project team.
