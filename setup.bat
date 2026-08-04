@echo off
REM One-click local setup for Windows. Double-click, or run from a terminal.
REM Prefers the project virtualenv so packages never land system-wide by
REM accident; falls back to whatever `python` is on PATH.
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python was not found on PATH.
        echo Install Python 3.10+ from https://www.python.org/downloads/
        echo and tick "Add python.exe to PATH" during installation.
        pause
        exit /b 1
    )
    set "PY=python"
    echo No .venv found. Creating one so packages stay inside the project...
    python -m venv .venv
    if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
)

"%PY%" scripts\setup.py %*
set "RC=%ERRORLEVEL%"

echo.
REM Keep the window open when launched by double-click, where there is no
REM terminal to read the output in afterwards.
pause
exit /b %RC%
