# ============================================================
#  FYP-26-S2-17  Attendance AI — one-click demo launcher (PowerShell)
# ============================================================
#  Usage:   powershell -ExecutionPolicy Bypass -File demo.ps1
#           # or just:  .\demo.ps1 --mode image
# ============================================================

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# Prefer the repo-local venv if it exists
$py = "python"
if (Test-Path ".venv\Scripts\python.exe") {
    $py = ".venv\Scripts\python.exe"
}

if (-not (Test-Path ".env")) {
    Write-Host "[demo] ERROR: .env not found. Copy .env.example to .env and fill DATABASE_URL." -ForegroundColor Red
    exit 2
}

# Install deps on first run
& $py -c "import insightface, cv2, psycopg2" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[demo] Installing AI dependencies (first run only)..." -ForegroundColor Yellow
    & $py -m pip install -r ai\requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[demo] pip install failed." -ForegroundColor Red
        exit 3
    }
}

Write-Host "[demo] Launching one-click demo..." -ForegroundColor Cyan
& $py -m ai.demo @args
exit $LASTEXITCODE
