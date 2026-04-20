# FYP-26-S2-17 — Face Recognition Attendance System

> An AI-powered student attendance tracking system using deep learning facial recognition, built for SIM Global Education / University of Wollongong.

---

## Overview

This project replaces manual roll calls and QR code check-ins with an automated facial recognition system deployed at classroom entry points. The system identifies students in real time, logs attendance with timestamps, performs periodic in-class presence checks to detect early departures, and classifies each student as **Present**, **Late**, or **Absent**.

Built on an ensemble of state-of-the-art deep learning models (SCRFD + ArcFace as primary, MTCNN + FaceNet as secondary), with GAN-based image enhancement for non-ideal lighting conditions. All biometric data is self-hosted in compliance with Singapore PDPC guidelines under the Personal Data Protection Act 2012.

---

## Team

| Name | Role |
|------|------|
| YU, ZHANGHAO | Project Leader / Producer |
| WHYE LI HENG, DOMINIC | Lead AI Engineer |
| ZHANG, CHENGWEI | Lead AI Engineer / UI/UX Designer |
| ZHANG, JIQIAN | Computer Vision & Backend Developer |
| ZHAO, SHIYIN | QA Engineer & Documentation Lead |

---

## Key Features

- **Automated attendance logging** — students are identified at classroom entry via camera, no manual check-in required
- **In-class presence verification** — periodic checks detect students who sign in and leave early
- **Present / Late / Absent classification** — configurable time thresholds determine attendance status
- **Anti-spoofing detection** — prevents fraudulent sign-ins using photographs
- **Ensemble multi-model voting** — SCRFD + ArcFace and MTCNN + FaceNet vote to improve accuracy
- **GAN image enhancement** — improves recognition under poor lighting conditions
- **Role-based access** — separate dashboards for Students, Teachers, and Administrators
- **ML model management** — admins can configure, train, and redeploy models from the web interface
- **Accuracy statistics dashboard** — visualises FP/FN rates, correct classification rates, and inference time
- **PDPC compliant** — fully self-hosted, no third-party data sharing

---

## System Architecture

```
Classroom Camera (Browser Fullscreen)
          ↓ Real-time video stream
    FastAPI Backend
          ↓
      AI Module
      ├── GAN Pre-processing     (image enhancement)
      ├── Face Detection         SCRFD (primary) + MTCNN (ensemble)
      ├── Face Recognition       ArcFace (primary) + FaceNet (ensemble)
      └── Voting Aggregation     final prediction
          ↓
    PostgreSQL Database
          ↑
    React Web Frontend
    Student / Teacher / Admin dashboards
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React.js, Tailwind CSS |
| Backend | Python, FastAPI |
| Database | PostgreSQL |
| AI — Detection | SCRFD (InsightFace), MTCNN |
| AI — Recognition | ArcFace (InsightFace), FaceNet |
| AI — Enhancement | StyleGAN / StarGAN, GAN (Super-Resolution / Image Enhancement) |
| Image Processing | OpenCV |
| Deployment | Docker, Vercel (frontend), Railway (backend) |
| CI/CD | GitHub Actions |

---

## Project Structure

```
FYP-26-S2-17/
├── frontend/          # React web application
├── backend/           # FastAPI REST API
├── ai/                # AI pipeline (detection, recognition, GAN, ensemble)
├── database/          # Schema and migrations
├── .github/workflows/ # CI/CD pipelines
├── docker-compose.yml
└── README.md
```

---
## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Production-ready code, auto-deployed via CD |
| `develop` | Integration branch for completed features |

---

## User Roles

| Role | Key Capabilities |
|------|----------------|
| Student | View attendance records, submit appeals, register facial image |
| Teacher | Real-time attendance dashboard, override records, export reports |
| Admin | Manage users, facial image database, ML model config & retraining, accuracy statistics |

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
