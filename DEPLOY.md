# Deployment Guide

Production stack: **Vercel** (static frontend) + **Railway** (FastAPI backend, CPU) +
**Supabase** (Postgres + pgvector, already live).

> ⚠️ The AI pipeline has no GPU on Railway. SCRFD + ArcFace run on CPU; the heavy
> ensemble (MTCNN / FaceNet / GAN enhancer) is disabled via env to fit memory.
> Recognition is slower than on the local RTX box. Use a paid Railway plan with
> enough RAM (the buffalo_l model + onnxruntime CPU need ~1.5–2 GB).

---

## 1. Backend → Railway

1. Push the deploy files to GitHub (already in the repo: `Dockerfile`,
   `.dockerignore`, `railway.json`).
2. On [railway.app](https://railway.app): **New Project → Deploy from GitHub repo**,
   pick `NagisaKong/SIM-UOW-FYP-CSIT-26-S2-17`. Railway auto-detects `railway.json`
   and builds the Dockerfile.
3. Set **environment variables** (Railway → Variables):

   ```
   APP_ENV=production
   JWT_SECRET=<python -c "import secrets; print(secrets.token_hex(32))">
   ALLOWED_ORIGINS=https://<your-frontend>.vercel.app
   DATABASE_URL=<your Supabase pooler URL>
   APP_TIMEZONE=Asia/Singapore

   # CPU mode — disable GPU + heavy ensemble models
   AI_DEVICE=cpu
   AI_CTX_ID=-1
   AI_USE_MTCNN=false
   AI_USE_FACENET=false
   AI_USE_ENHANCER=false

   # Face detection range (see "Detection range tuning" below)
   AI_DET_SIZE=1280
   # AI_DET_THRESH=0.5
   # AI_ANTISPOOF_MIN_FACE_PX=50

   # SMTP (notifications) — copy from local .env
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USER=...
   SMTP_PASSWORD=...
   SMTP_FROM=...
   SMTP_USE_TLS=1
   ```

   Do **not** set `HOST`/`PORT` — Railway injects `PORT`, and the Dockerfile sets
   `HOST=0.0.0.0`.
4. Deploy. Confirm the healthcheck at `https://<backend>.railway.app/health`
   returns `{"success": true, ...}`. `/docs` is disabled in production (expected).

### Detection range tuning (far / small faces)

The classroom-scan frontend captures at **1080p**, but detection range is still
governed by these knobs. Symptom to watch for: faces beyond ~2–3 m stop being
detected.

| Variable | Default | Effect |
|---|---|---|
| `AI_DET_SIZE` | `1280` | SCRFD input size. Bigger = detects smaller/farther faces. Accepts `1280` (square) or `1920x1080`. **Main lever for range.** Larger = slower + more memory. |
| `AI_DET_THRESH` | `0.5` | Detection confidence cutoff. Lower (e.g. `0.4`) recalls more far/blurry faces, at the cost of more false positives. |
| `AI_ANTISPOOF_MIN_FACE_PX` | `50` | Faces smaller than this (in pixels) are discarded. Lower (e.g. `30`) keeps more distant faces. |

Guidance:
- On **CPU**, `AI_DET_SIZE=1280` is ~3–4× slower per frame than the old `640`.
  If snapshots feel slow, drop to `960`. Since scanning is interval snapshots
  (not live video), this is usually acceptable.
- For long rooms (4 m+), prefer a **GPU** (`AI_DEVICE=cuda`, `AI_CTX_ID=0`) so
  you can run `AI_DET_SIZE=1920` without a big latency hit.

## 2. Frontend → Vercel

1. On [vercel.com](https://vercel.com): **Add New → Project**, import the same repo.
2. Set **Root Directory = `frontend`** (the static files + `vercel.json` live there).
   Framework preset: **Other** (no build step).
3. Deploy. You get `https://<your-frontend>.vercel.app`.
4. Point the frontend at the backend — two options:
   - Visit `https://<your-frontend>.vercel.app/?api=https://<backend>.railway.app`
     once; `app.js` saves it to `localStorage` for subsequent visits, **or**
   - Add an inline `<script>window.API_BASE="https://<backend>.railway.app"</script>`
     before `app.js` in the HTML files for a fixed default.
5. Go back to Railway and make sure `ALLOWED_ORIGINS` matches the Vercel URL exactly
   (scheme + host, no trailing slash), then redeploy the backend.

## 3. Post-launch checklist

- [ ] `/health` green on Railway
- [ ] Login works end-to-end from the Vercel URL (CORS OK, no console errors)
- [ ] Change/disable the demo accounts — `demo123` is publicly known
- [ ] Confirm Supabase automatic backups are enabled
- [ ] (Optional, deferred) rate limiting on login + recognition endpoints
- [ ] (Optional) tighten frontend CSP once camera/Chart.js sources are pinned

## Local verification (optional, before pushing)

Build and run the backend image exactly as Railway will:

```bash
docker build -t fyp-backend .
docker run --rm -p 8000:8000 \
  -e APP_ENV=production \
  -e JWT_SECRET=$(python -c "import secrets; print(secrets.token_hex(32))") \
  -e ALLOWED_ORIGINS=http://localhost:5500 \
  -e DATABASE_URL="<supabase url>" \
  -e AI_USE_MTCNN=false -e AI_USE_FACENET=false -e AI_USE_ENHANCER=false \
  fyp-backend
# then: curl http://localhost:8000/health
```
