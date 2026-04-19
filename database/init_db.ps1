# ============================================================
# FYP-26-S2-17: Initialize local PostgreSQL database
# Usage: Run from project root in PowerShell
#   powershell -ExecutionPolicy Bypass -File database\init_db.ps1
# ============================================================

$ErrorActionPreference = "Stop"

# --- Configuration ---
$PG_VERSION     = "17"
$PG_BIN         = "C:\Program Files\PostgreSQL\$PG_VERSION\bin"
$DB_NAME        = "fyp_database"
$DB_USER        = "fyp_user"
$DB_PASSWORD    = "fyp_user"
$SCHEMA_FILE    = Join-Path $PSScriptRoot "schema.sql"

# Check psql exists
if (-not (Test-Path "$PG_BIN\psql.exe")) {
    Write-Host "ERROR: psql.exe not found at $PG_BIN" -ForegroundColor Red
    Write-Host "Install PostgreSQL 17 first: winget install PostgreSQL.PostgreSQL.17"
    exit 1
}

# Prompt for postgres superuser password (set during install)
Write-Host "Enter the 'postgres' superuser password (set during installation):" -ForegroundColor Yellow
$securePwd = Read-Host -AsSecureString
$bstr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePwd)
$env:PGPASSWORD = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)

Write-Host "`n[1/3] Creating role '$DB_USER' ..." -ForegroundColor Cyan
& "$PG_BIN\psql.exe" -U postgres -h localhost -d postgres -v ON_ERROR_STOP=0 -c `
    "DO `$`$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='$DB_USER') THEN CREATE ROLE $DB_USER WITH LOGIN PASSWORD '$DB_PASSWORD'; END IF; END `$`$;"

Write-Host "[2/3] Creating database '$DB_NAME' ..." -ForegroundColor Cyan
$dbExists = & "$PG_BIN\psql.exe" -U postgres -h localhost -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME';"
if ($dbExists -ne "1") {
    & "$PG_BIN\psql.exe" -U postgres -h localhost -d postgres -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;"
} else {
    Write-Host "  Database '$DB_NAME' already exists — skipped." -ForegroundColor DarkGray
}

Write-Host "[3/3] Applying schema from $SCHEMA_FILE ..." -ForegroundColor Cyan
$env:PGPASSWORD = $DB_PASSWORD
& "$PG_BIN\psql.exe" -U $DB_USER -h localhost -d $DB_NAME -v ON_ERROR_STOP=1 -f "$SCHEMA_FILE"

Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue

Write-Host "`nDone. Verify with:" -ForegroundColor Green
Write-Host "  & `"$PG_BIN\psql.exe`" -U $DB_USER -h localhost -d $DB_NAME -c `"\dt`""
