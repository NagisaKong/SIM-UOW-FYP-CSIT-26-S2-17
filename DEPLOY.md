# Deployment Guide

## 0. Local setup after pulling the branch

One command does all of it — dependencies, `.env`, and the GPU wheels if the
machine has an NVIDIA card:

```bash
python scripts/setup.py
```

On Windows you can double-click **`setup.bat`** instead; it creates `.venv`
first if there isn't one, so packages never land system-wide by accident.

It is safe to re-run: an existing `.env` is never overwritten, CUDA wheels are
never installed on a machine without a GPU, and `--check` reports what it would
do without changing anything. Then fill in `DATABASE_URL` (ask the project
leader) and:

```bash
python -m pytest tests/ -q
```

Doing it by hand instead, if you prefer:

```bash
cp .env.example .env                     # copy .env.example .env  on Windows
pip install -r requirements.txt
python scripts/check_gpu.py --fix        # only if you have an NVIDIA GPU
```

`.env.example` documents every `AI_*` knob and the measured reason for each
default, so it is worth reading once even if the script does the work.

> Point `DATABASE_URL` at a **local** Postgres for testing. `tests/` exercises
> the real database — it creates accounts, sessions and embeddings — so running
> the suite against the production Supabase instance writes test data into it.

### If you have an NVIDIA GPU

`requirements.txt` gives everyone the CPU runtimes, so a GPU is not used until
you switch the wheels — and that failure is silent, not loud. Run:

```bash
python scripts/check_gpu.py --fix
```

Then set `AI_DEVICE=cuda` and `AI_CTX_ID=0` in `.env`. Both halves are needed:
correct wheels with `AI_CTX_ID=-1` still runs on CPU, and `cuda` with CPU
wheels also runs on CPU.

Confirm what is actually running rather than assuming — `/health` reports the
live configuration, and InsightFace logs the provider per model at startup:

```bash
python -c "from core.attendancePipeline import AIConfig; c=AIConfig(); print(c.log_summary())"
```

### Two ways to tune the models

| | Changed by | Applies |
|---|---|---|
| **Environment (`AI_*`)** | editing `.env`, restart required | server-wide; detector choice, thresholds, device, ensemble |
| **Admin UI** | Model Management (U22–U25), Behaviour Analysis (U35) | at runtime, no restart; recognition similarity threshold and per-course behaviour parameters |

Prefer the Admin UI for anything it covers: the recognition threshold deployed
there is persisted to `MODEL_CONFIGS` and hot-applied, so it survives restarts
and is shared by everyone, whereas `.env` is per-machine. Use `AI_*` for
choices the UI does not expose — which detector, which device, which models
are loaded at all.

If you change a detection default, re-measure rather than guessing:

```bash
python tests/record_behaviour_clips.py --camera 0 --out clips
python tests/run_behaviour_st.py --clips clips --evidence docs/evidence
python tests/run_ml_st.py --clips clips --evidence docs/evidence
```

`clips/` holds real facial images and is gitignored — keep it that way.

---

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
   AI_BEHAVIOUR=false           # CR-06 behaviour analysis — GPU rig only (see below)

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

### Model weights are baked into the image — do not remove that build step

`Dockerfile` runs `scripts/prefetch_models.py` during the build to download the
SCRFD + ArcFace pack (`buffalo_l`, ~280 MB) into `/root/.insightface`.

This is not an optimisation, it is a correctness fix. InsightFace otherwise
downloads the pack lazily when `FaceAnalysis` is first constructed — which
happens inside the FastAPI **lifespan handler**. On a PaaS the container
filesystem is ephemeral, so every cold start re-downloads it, and a single
failed download raises inside lifespan and takes the whole application down
(login, dashboards, reports included). Railway then restarts it, fails again,
and the deployment crash-loops with `RuntimeError: Failed downloading url
.../buffalo_l.zip`. Fetching at build time turns that into a loud build
failure instead, and leaves the running container independent of GitHub.

If a build ever fails at this step, GitHub releases were unreachable from the
builder — re-run the build; the old deployment keeps serving in the meantime.

### Detection range tuning (far / small faces)

The classroom-scan frontend captures at **1080p**, but detection range is still
governed by these knobs. Symptom to watch for: faces beyond ~2–3 m stop being
detected.

| Variable | Default | Effect |
|---|---|---|
| `AI_DETECT_TILES` | `1x1` | **Strongest range lever.** Split each frame into `cols×rows` tiles and detect on each, so distant faces become large enough to detect. `2x2` ≈ **doubles** range (e.g. 3.5 m → ~7 m); `3x3` ≈ triples it. Costs `cols*rows`× the detection compute — keep `1x1` on CPU, use `2x2`+ on GPU. |
| `AI_DETECT_TILE_OVERLAP` | `0.15` | Fractional overlap between tiles so faces on a tile boundary aren't missed. Rarely needs changing. |
| `AI_DET_SIZE` | `1280` | SCRFD input size. Bigger = detects smaller/farther faces. Accepts `1280` (square) or `1920x1080`. Larger = slower + more memory. |
| `AI_DET_THRESH` | `0.5` | Detection confidence cutoff. Lower (e.g. `0.4`) recalls more far/blurry faces, at the cost of more false positives. |
| `AI_ANTISPOOF_MIN_FACE_PX` | `50` | Faces smaller than this (in pixels) are discarded. Lower (e.g. `30`) keeps more distant faces. |

**Why tiling, not just resolution:** range is limited by *pixels on the face*,
not camera megapixels. Without tiling, only `AI_DET_SIZE` (relative to the
lens field-of-view) matters — capturing 4K then shrinking to a 1280 detector
input gains nothing. Tiling detects each tile at near-native scale, which is
the only software lever that genuinely multiplies range with the existing model.

Recommended for a long room on the **GPU rig** (RTX 5080):
```
AI_DEVICE=cuda
AI_CTX_ID=0
AI_DETECT_TILES=2x2          # or 3x3 for very long rooms
AI_ANTISPOOF_MIN_FACE_PX=30
# AI_DET_THRESH=0.4          # if some far faces are still missed
```
1080p capture + `2x2` already roughly doubles range; no 4K needed. Beyond
software, a **narrower field-of-view / telephoto lens** physically puts more
pixels on a distant face and is the most reliable way to reach 7 m+.

Guidance:
- On **CPU** (Railway), keep `AI_DETECT_TILES=1x1` — tiling multiplies compute
  and would make snapshots slow. `AI_DET_SIZE=1280` is already ~3–4× slower per
  frame than the old `640`; drop to `960` if snapshots lag.

### Behaviour analysis (CR-06 — GPU rig only)

The classroom behaviour module (drowsiness via MediaPipe FaceMesh, phone use
via YOLOv8n, activity heatmap) samples ~1 frame/sec from the teacher's scan
page, which is far too heavy for the Railway CPU plan. Run it only on the
local GPU machine:

```
AI_BEHAVIOUR=true            # master switch (default false)
# Optional tuning (defaults shown):
# AI_EAR_THRESHOLD=0.21      # eyes-closed EAR cutoff
# AI_EAR_CONSEC_SECONDS=2.0  # signal must persist this long to count
# AI_MAR_THRESHOLD=0.6       # yawn cutoff
# AI_HEADPOSE_PITCH_DEG=30   # head-tilt cutoff (degrees)
# AI_PHONE_CONF=0.50         # YOLO phone confidence (see model table below)
# AI_PHONE_CONSEC=3          # consecutive ~1s samples to confirm phone use
# AI_PHONE_MODEL=yolov8n.pt  # see the model table below BEFORE changing this
# AI_PHONE_IMGSZ=1280        # YOLO input size; the 640 default shrinks frames
#                            # and loses phones beyond ~3 m. Match your capture
#                            # width on GPU: 1920 for 1080p cameras, 1280 for 720p
# AI_HEADPOSE_SOURCE=insightface  # 'solvepnp' to skip the extra pose model
# AI_HEADPOSE_DELTA_DEG=10   # degrees below personal baseline = head down
# AI_PHONE_STATIC_SAMPLES=45 # samples of an immobile box before it is judged
#                            # a room fixture (wall switch, socket) and ignored
# AI_HEATMAP_GRID=8x6        # heatmap cells (cols x rows)
# AI_HEATMAP_FLUSH_SECONDS=60
```

### Choosing the phone-detection model — bigger is not better

Measured against labelled clips at `AI_PHONE_CONF=0.50`
(`docs/evidence/behaviour_st_analysis.md`), recall / false-alarm rate:

| Model | @1280 | @1920 |
|---|---|---|
| `yolov8n.pt` | 1.00 / **0.000** | 1.00 / **0.000** |
| `yolov8s.pt` | 1.00 / 0.267 | 0.90 / 0.324 |
| `yolov8m.pt` | 1.00 / **0.000** | 1.00 / **0.000** |
| `yolov8l.pt` | — | 1.00 / 0.106 |

> ⚠️ **Do not use `yolov8s.pt`.** It scores a wall-mounted switch panel as a
> phone at up to 0.59 confidence, and because that panel never moves it lands
> in the same student's reach region every sample — one student gets blamed
> all lesson. `yolov8n` and `yolov8m` do not do this at any input size.
>
> Do not try to fix a noisy model by raising `AI_PHONE_CONF`. It suppresses
> the false alarms but costs real recall: at 0.80, `yolov8n` drops from 1.00
> to 0.25 because genuine phones do not always score high either.

`_StaticBoxFilter` is a second line of defence — a detection whose position is
unchanged for `AI_PHONE_STATIC_SAMPLES` samples is treated as a fixture. Note
that temporal debouncing does *not* help here: a permanently present false
positive is exactly what the episode tracker confirms into an event.

### GPU setup

`requirements.txt` installs the CPU runtimes on purpose — it has to work on
Railway, in CI, and on machines without an NVIDIA card. GPU machines switch
afterwards:

```bash
python scripts/check_gpu.py --fix
```

That runs the three steps in the only order that works; `requirements-gpu.txt`
documents them if you would rather do it by hand.

The ordering matters because `onnxruntime` (CPU) and `onnxruntime-gpu` install
into the same `onnxruntime/` package directory. With both present the CPU build
wins and `get_available_providers()` silently omits CUDA — the app then runs on
CPU while appearing configured for GPU. Uninstalling the CPU build is therefore
required, and it deletes files the GPU build shares, which is why the reinstall
that follows uses `--force-reinstall`.

Verify before trusting it. `pip list` is not evidence — it will show
`onnxruntime-gpu` as installed while the CPU build silently shadows it:

```bash
python scripts/check_gpu.py
```

It checks the driver, both package sets, the actual provider list, the torch
build and the `.env` configuration, and prints the exact fix if any of them
disagree. Exit code 1 means the GPU will not be used.

The app also reports this at runtime: `/health` returns the live device, and
both InsightFace and the YOLO phone detector log a warning at startup when a
GPU was requested but they came up on CPU.

Measured on an RTX 4060 Laptop, per behaviour frame: InsightFace detection +
3D pose 17 ms, YOLOv8s@1920 32 ms (vs 377 ms on CPU), MediaPipe 7 ms — about
56 ms total against the 1 fps budget, roughly 17x headroom. The same clips
score **identically** on CPU and GPU: the GPU buys capacity for heavier models
(`AI_DETECT_TILES=2x2`, `yolov8m`), not accuracy by itself.

Notes:
- `pip install mediapipe ultralytics` is required (already in
  requirements.txt); YOLOv8n weights download automatically on first frame.
- The per-course toggle (admin → Behaviour Analysis, U35) must also be ON —
  the frontend sampler stops itself if either switch is off.
- Privacy: frames are analysed in memory and discarded; only derived event
  tuples (`behaviour_event`) and grid intensities (`heatmap_snapshot`) are
  stored. Keep it this way — never add frame persistence to this path.
- Detection range: FaceMesh needs ≥~64 px faces, so drowsiness detection has a
  shorter usable range than attendance detection. Place the behaviour camera
  closer to the students than the long-range attendance camera.

## 2. Frontend → Vercel

1. On [vercel.com](https://vercel.com): **Add New → Project**, import the same repo.
2. Set **Root Directory = `frontend`** (the static files + `vercel.json` live there).
   Framework preset: **Other** (no build step).
3. Deploy. You get `https://<your-frontend>.vercel.app`.
4. **Point the frontend at the backend — edit `frontend/config.js`:**

   ```js
   window.API_BASE = "https://<backend>.up.railway.app";
   ```

   That single file is the deployment default, loaded before `app.js` on every
   page. Without it the pages fall back to `<current-host>:8000`, which does not
   exist on Vercel — visitors would get "Cannot reach the API".
   Commit and redeploy the frontend after changing it.

   Runtime overrides (no rebuild needed) still work and take precedence:
   - `?api=https://other-backend` — use and remember another backend. This is how
     the hosted UI is pointed at a **local GPU machine** so behaviour analysis
     (CR-06) can run; that machine must be reachable from the browser.
   - `?api=default` — forget the override and go back to `config.js`.

5. Go back to Railway and make sure `ALLOWED_ORIGINS` matches the Vercel URL exactly
   (scheme + host, no trailing slash), then redeploy the backend. A mismatch here
   shows up as "Cannot reach the API …" in the UI and a CORS error in the console,
   even though the backend itself is healthy.

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
