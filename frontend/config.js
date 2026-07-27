// Deployment configuration — the ONE place that names the backend API.
//
// Loaded before app.js on every page. Set API_BASE to the deployed FastAPI
// origin (scheme + host, no trailing slash, no path):
//
//     window.API_BASE = "https://your-backend.up.railway.app";
//
// This is the default for the HOSTED site only. Pages opened from localhost /
// 127.0.0.1 / file:// ignore it and always talk to a backend on the same
// machine (http://localhost:8000), so local development keeps working after
// you set this — the deployed backend does not allow localhost origins anyway.
//
// This value is only the DEFAULT. It can still be overridden at runtime:
//   • visit  ?api=http://192.168.1.50:8000   → use that backend and remember it
//     (used to point the hosted UI at a local GPU machine for behaviour analysis)
//   • visit  ?api=default                    → forget the override, use this file
//
// Remember: whatever origin serves these pages must also be listed in the
// backend's ALLOWED_ORIGINS environment variable, or the browser will block
// the requests (CORS).
window.API_BASE = "https://sim-uow-fyp-csit-26-s2-17-production.up.railway.app/";
