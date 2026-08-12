// Deployment configuration — the one place that names the backend API.
// Loaded before app.js on every page.
//
// Applies to the hosted site only: pages opened from localhost / 127.0.0.1 /
// file:// always use http://localhost:8000 instead, so local development keeps
// working. Override at runtime with ?api=<origin>, or ?api=default to clear it.
//
// Whichever origin serves these pages must be listed in the backend's
// ALLOWED_ORIGINS, or the browser blocks the requests (CORS).
window.API_BASE = "https://sim-uow-fyp-csit-26-s2-17-production.up.railway.app/";
