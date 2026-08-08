@echo off
setlocal enableextensions enabledelayedexpansion

REM ===========================================================================
REM  FYP-26-S2-17  FRAS System Launcher (Windows)
REM
REM  One-stop local entry point: dependency setup, database initialisation,
REM  API / frontend startup, environment diagnostics and the test runs.
REM  Every option delegates to an existing script (scripts\setup.py,
REM  scripts\check_gpu.py, tests\*) rather than duplicating its logic.
REM
REM  Usage:
REM    (double-click)          open the point-and-click GUI launcher
REM    launcher.bat --console  skip the GUI, use the text menu
REM    launcher.bat <action>   run one action directly and stop; <action> is
REM                            one of the tokens listed under :DISPATCH below.
REM                            This is how the GUI buttons call back in, so
REM                            the GUI holds no logic of its own.
REM ===========================================================================

REM Always run from the folder where this .bat lives
cd /d "%~dp0"

set "ACTION=%~1"
set "DIRECT="
if defined ACTION (
    if /i not "%ACTION%"=="--console" if /i not "%ACTION%"=="-c" set "DIRECT=1"
)

call :DETECT_PY
call :DETECT_ENV

REM - Nothing below this point exits before the menu: a machine with no .venv
REM   and no dependencies is exactly the machine that needs option [6]. -

if defined DIRECT goto :DISPATCH
if defined ACTION goto :MAIN_MENU

REM - No arguments: prefer the GUI, fall back to the text menu if it will not
REM   start (no PowerShell, missing .ps1, or it exits non-zero). -
call :TRY_GUI && goto :EXIT_NOW
goto :MAIN_MENU

REM ===========================================================================
REM  Action dispatch - the GUI's only interface to this script
REM ===========================================================================
:DISPATCH
if /i "%ACTION%"=="system"       goto :ACT_FULL
if /i "%ACTION%"=="demo"         goto :ACT_FULL
if /i "%ACTION%"=="stop"         goto :ACT_STOP
if /i "%ACTION%"=="api"          goto :ACT_WEB
if /i "%ACTION%"=="frontend"     goto :ACT_FRONTEND
if /i "%ACTION%"=="health"       goto :ACT_HEALTH
if /i "%ACTION%"=="db"           goto :ACT_DB
if /i "%ACTION%"=="setup"        (set "SETUP_ARGS="        & goto :ACT_SETUP)
if /i "%ACTION%"=="setup-check"  (set "SETUP_ARGS=--check" & goto :ACT_SETUP)
if /i "%ACTION%"=="setup-nogpu"  (set "SETUP_ARGS=--no-gpu"& goto :ACT_SETUP)
if /i "%ACTION%"=="gpu"          goto :ACT_GPU
if /i "%ACTION%"=="gpu-fix"      goto :ACT_GPUFIX
if /i "%ACTION%"=="models"       goto :ACT_MODELS
if /i "%ACTION%"=="aiconfig"     goto :ACT_AICONFIG
if /i "%ACTION%"=="pytest"       (set "TC=1" & goto :ACT_PYTEST)
if /i "%ACTION%"=="ruff"         goto :ACT_RUFF
if /i "%ACTION%"=="test-both"    (set "TC=3" & goto :ACT_PYTEST)
if /i "%ACTION%"=="clips"        goto :ACT_CLIPS
if /i "%ACTION%"=="st-bh"        goto :ACT_STBH
if /i "%ACTION%"=="st-ml"        goto :ACT_STML
echo.
echo  ERROR: unknown action "%ACTION%".
echo  Run without arguments for the GUI, or --console for the text menu.
echo.
pause
goto :EXIT_NOW

REM -
:MAIN_MENU
cls
echo.
echo  +----------------------------------------------------------------+
echo  ^|              FYP-26-S2-17   FRAS System Launcher               ^|
echo  +----------------------------------------------------------------+
echo.
echo    Python: !PY_LABEL!   ^|   deps: !DEPS_LABEL!   ^|   .env: !ENV_LABEL!
if not "!DEPS_OK!"=="1" echo    ^>^> Dependencies are missing - run [6] first.
if not "!ENV_OK!"=="1"  echo    ^>^> .env is missing - run [6], then fill in DATABASE_URL.
echo.
echo  +================================================================+
echo  ^|                                                                ^|
echo  ^|   [1]  START SYSTEM       API + frontend + browser, one step   ^|
echo  ^|   [2]  Stop system        shut the API and frontend down       ^|
echo  ^|                                                                ^|
echo  +================================================================+
echo.
echo  +-- Run individually --------------------------------------------+
echo  ^|  [3]  Start Web API       FastAPI server on 127.0.0.1:8000     ^|
echo  ^|  [4]  Serve frontend      static files on 127.0.0.1:5500       ^|
echo  ^|  [5]  Health check        GET /health on the running API       ^|
echo  ^|                                                                ^|
echo  +-- Setup -------------------------------------------------------+
echo  ^|  [6]  Setup database      schema.sql + demo seed (first run)   ^|
echo  ^|  [7]  Install deps        requirements.txt + .env + GPU wheels ^|
echo  ^|  [8]  Diagnostics         GPU / model weights / live AI config ^|
echo  ^|                                                                ^|
echo  +-- Test --------------------------------------------------------+
echo  ^|  [9]  Test suite          pytest / ruff / ST measurement runs  ^|
echo  ^|                                                                ^|
echo  ^|  [0]  Exit                                                     ^|
echo  ^|                                                                ^|
echo  ^|  Note: the API needs a reachable PostgreSQL (pgvector) at      ^|
echo  ^|        DATABASE_URL. First time? Run [7], [6], then [1].       ^|
echo  +----------------------------------------------------------------+
echo.

set "CHOICE="
REM set /p drops leading spaces from its prompt, so it starts with '>'.
set /p "CHOICE=> Select [0-9]: "

if "!CHOICE!"=="0" goto :END
if "!CHOICE!"=="1" goto :ACT_FULL
if "!CHOICE!"=="2" goto :ACT_STOP
if "!CHOICE!"=="3" goto :ACT_WEB
if "!CHOICE!"=="4" goto :ACT_FRONTEND
if "!CHOICE!"=="5" goto :ACT_HEALTH
if "!CHOICE!"=="6" goto :ACT_DB
if "!CHOICE!"=="7" goto :MODE_SETUP
if "!CHOICE!"=="8" goto :MODE_DIAG
if "!CHOICE!"=="9" goto :MODE_TEST
echo  Invalid choice, try again.
timeout /t 1 >nul
goto :MAIN_MENU

REM ===========================================================================
REM  Run
REM ===========================================================================
:ACT_WEB
call :REQUIRE_READY || goto :DONE
echo.
echo  [run] Launching FastAPI server at http://127.0.0.1:8000 ...
echo  (API docs: http://127.0.0.1:8000/docs  -  Press Ctrl+C to stop)
echo.
"%PY%" -m uvicorn main_api:app --host 127.0.0.1 --port 8000
goto :DONE

:ACT_FRONTEND
call :REQUIRE_PY || goto :DONE
echo.
echo  [run] Serving frontend at http://127.0.0.1:5500 ...
echo  (Make sure the Web API (option 1) is also running in another window)
echo  (Press Ctrl+C to stop)
echo.
"%PY%" -m http.server 5500 --directory "%~dp0frontend"
goto :DONE

:ACT_FULL
call :REQUIRE_READY || goto :DONE
echo.
echo  [run] Starting API in a new window...
start "FYP Web API" cmd /k ""%PY%" -m uvicorn main_api:app --host 127.0.0.1 --port 8000"
echo  [run] Waiting for the API to come up...
timeout /t 5 >nul
echo  [run] Opening browser at http://127.0.0.1:5500 ...
start "" "http://127.0.0.1:5500/index.html"
echo  [run] Serving frontend (Ctrl+C here, or "Stop system", to shut down)...
echo.
"%PY%" -m http.server 5500 --directory "%~dp0frontend"
goto :DONE

REM - Counterpart to ACT_FULL. Ctrl+C only reaches whichever window has focus,
REM   and the API runs in a second one, so stop by listening port instead:
REM   that catches both however they were started.
:ACT_STOP
echo.
echo  [stop] Shutting the FRAS servers down...
echo.
set "STOPPED="
call :KILL_PORT 8000 "Web API"
call :KILL_PORT 5500 "frontend"
REM   Best effort: close the console ACT_FULL opened for the API. Harmless if
REM   the window is already gone or was never opened.
taskkill /F /FI "WINDOWTITLE eq FYP Web API*" >nul 2>&1
echo.
if defined STOPPED (
    echo  [stop] Done. The browser tab stays open - it will just fail to reach
    echo         the API until you start the system again.
) else (
    echo  [stop] Nothing was listening on 127.0.0.1:8000 or 127.0.0.1:5500.
)
goto :DONE

:ACT_HEALTH
call :REQUIRE_PY || goto :DONE
echo.
echo  [run] GET http://127.0.0.1:8000/health ...
"%PY%" -c "import urllib.request,sys; print(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5).read().decode())" 2>nul
if errorlevel 1 (
    echo  Could not reach the API. Start it first with option 1.
)
goto :DONE

REM ===========================================================================
REM  Setup
REM ===========================================================================
:ACT_DB
call :REQUIRE_READY || goto :DONE
echo.
echo  [db] Initialising the database (schema + demo seed)...
echo.

REM - psql is required to apply the SQL (seed uses psql meta-commands) -
where psql >nul 2>&1
if errorlevel 1 (
    echo  ERROR: psql not found on PATH.
    echo  Install the PostgreSQL client tools, or apply the SQL manually:
    echo      psql "DATABASE_URL" -f database\schema.sql
    echo      psql "DATABASE_URL" -v demo_password_hash="HASH" -f database\seed_demo.sql
    goto :DONE
)

call :READ_DBURL
if not defined DBURL (
    echo  ERROR: DATABASE_URL not found in .env
    goto :DONE
)

REM - Destructive guard: seed_demo.sql DELETEs all data before re-seeding.
REM   .env may point at a LIVE/remote database (e.g. Supabase), so confirm.
echo.
echo  +-- WARNING -----------------------------------------------------+
echo  ^|                                                                ^|
echo  ^|  This RESETS the database in your .env: it DELETES all         ^|
echo  ^|  existing data and re-inserts demo data. If DATABASE_URL       ^|
echo  ^|  points at your LIVE Supabase, that data will be wiped.        ^|
echo  ^|                                                                ^|
echo  +----------------------------------------------------------------+
echo.
echo    Target host: !DBHOST!
echo.
set "CONFIRM="
set /p "CONFIRM=> Type YES to proceed (anything else cancels): "
if /i not "!CONFIRM!"=="YES" (
    echo  Cancelled. No changes made.
    goto :DONE
)

REM - Generate the Argon2id hash for the demo password (demo123) -
echo  [db] Generating Argon2id hash for the demo accounts...
"%PY%" -c "from argon2 import PasswordHasher, Type; open(r'%TEMP%\fyp_pw.txt','w').write(PasswordHasher(type=Type.ID, memory_cost=65536, time_cost=3, parallelism=4).hash('demo123'))"
if errorlevel 1 (
    echo  ERROR: could not generate password hash ^(is argon2-cffi installed?^).
    goto :DONE
)
set "PWHASH="
set /p "PWHASH="<"%TEMP%\fyp_pw.txt"
del "%TEMP%\fyp_pw.txt" >nul 2>&1
if not defined PWHASH (
    echo  ERROR: password hash came back empty.
    goto :DONE
)

REM - Ensure pgvector exists (no-op if already enabled, e.g. on Supabase).
REM   Requires the pgvector extension to be installed on the PG server.
echo  [db] Enabling pgvector extension...
psql "!DBURL!" -v ON_ERROR_STOP=1 -c "CREATE EXTENSION IF NOT EXISTS vector;"
if errorlevel 1 (
    echo.
    echo  ERROR: could not enable pgvector. Is the extension installed on the
    echo         PostgreSQL server? ^(Supabase has it built in.^)
    goto :DONE
)

REM - Apply schema, then the demo seed with the injected hash -
echo  [db] Applying database\schema.sql ...
psql "!DBURL!" -v ON_ERROR_STOP=1 -f "%~dp0database\schema.sql"
if errorlevel 1 (
    echo.
    echo  ERROR: schema.sql failed to apply. See the psql output above.
    goto :DONE
)

echo  [db] Applying database\seed_demo.sql ...
psql "!DBURL!" -v ON_ERROR_STOP=1 -v demo_password_hash=!PWHASH! -f "%~dp0database\seed_demo.sql"
if errorlevel 1 (
    echo.
    echo  ERROR: seed_demo.sql failed to apply. See the psql output above.
    goto :DONE
)

echo.
echo  [db] Done. Demo accounts created (password: demo123).
echo       e.g. admin@demo.local / demo123
goto :DONE

REM -
:MODE_SETUP
cls
echo.
echo  +-- Install / update dependencies -------------------------------+
echo  ^|                                                                ^|
echo  ^|  Delegates to scripts\setup.py: installs requirements.txt,     ^|
echo  ^|  creates .env from .env.example if missing (an existing one is ^|
echo  ^|  never overwritten), and switches to the CUDA wheels when an   ^|
echo  ^|  NVIDIA card is found. Safe to re-run.                         ^|
echo  ^|                                                                ^|
echo  ^|  [1]  Full setup          dependencies + .env + GPU wheels     ^|
echo  ^|  [2]  Report only         --check: report, change nothing      ^|
echo  ^|  [3]  Skip GPU            --no-gpu: CPU wheels only            ^|
echo  ^|                                                                ^|
echo  ^|  [0]  Back                                                     ^|
echo  +----------------------------------------------------------------+
echo.
set "SC="
set /p "SC=> Select [0-3]: "
if "!SC!"=="0" goto :MAIN_MENU
set "SETUP_ARGS="
if "!SC!"=="1" set "SETUP_ARGS="
if "!SC!"=="2" set "SETUP_ARGS=--check"
if "!SC!"=="3" set "SETUP_ARGS=--no-gpu"
if not "!SC!"=="1" if not "!SC!"=="2" if not "!SC!"=="3" (
    echo  Invalid choice.
    timeout /t 1 >nul
    goto :MODE_SETUP
)

:ACT_SETUP
REM - Create the venv first so packages never land system-wide by accident.
REM   Same approach as setup.bat. Skipped in --check mode, which must not
REM   change anything.
if not exist "%~dp0.venv\Scripts\python.exe" if not "!SETUP_ARGS!"=="--check" (
    echo.
    echo  [setup] No .venv found. Creating one so packages stay in the project...
    "%PY%" -m venv "%~dp0.venv"
    if exist "%~dp0.venv\Scripts\python.exe" (
        set "PY=%~dp0.venv\Scripts\python.exe"
        echo  [setup] Using !PY!
    ) else (
        echo  WARNING: could not create .venv; installing with !PY! instead.
    )
)

echo.
echo  [setup] Running scripts\setup.py !SETUP_ARGS! ...
echo.
"%PY%" "%~dp0scripts\setup.py" !SETUP_ARGS!
if errorlevel 1 (
    echo.
    echo  ERROR: setup did not complete. See the output above.
)

REM - Refresh the status bar so the menu reflects what just happened -
call :DETECT_PY
call :DETECT_ENV
goto :DONE

REM -
:MODE_DIAG
cls
echo.
echo  +-- Environment diagnostics -------------------------------------+
echo  ^|                                                                ^|
echo  ^|  [1]  Check GPU           driver, wheels, providers, .env      ^|
echo  ^|  [2]  Fix GPU wheels      check_gpu.py --fix (installs CUDA)   ^|
echo  ^|  [3]  Prefetch models     download buffalo_l (~280 MB)         ^|
echo  ^|  [4]  Show AI config      AIConfig().log_summary()             ^|
echo  ^|                                                                ^|
echo  ^|  [0]  Back                                                     ^|
echo  +----------------------------------------------------------------+
echo.
set "DC="
set /p "DC=> Select [0-4]: "
if "!DC!"=="0" goto :MAIN_MENU
if "!DC!"=="1" goto :ACT_GPU
if "!DC!"=="2" goto :ACT_GPUFIX
if "!DC!"=="3" goto :ACT_MODELS
if "!DC!"=="4" goto :ACT_AICONFIG
echo  Invalid choice.
timeout /t 1 >nul
goto :MODE_DIAG

:ACT_GPU
call :REQUIRE_PY || goto :DONE
echo.
"%PY%" "%~dp0scripts\check_gpu.py"
echo.
echo  Exit code 1 above means the GPU will NOT be used.
goto :DONE

:ACT_GPUFIX
call :REQUIRE_PY || goto :DONE
echo.
echo  [diag] Switching to the CUDA wheels. This reinstalls onnxruntime
echo         and torch - it can take several minutes.
echo.
"%PY%" "%~dp0scripts\check_gpu.py" --fix
echo.
echo  Then set AI_DEVICE=cuda and AI_CTX_ID=0 in .env - both are needed.
goto :DONE

:ACT_MODELS
call :REQUIRE_PY || goto :DONE
echo.
"%PY%" "%~dp0scripts\prefetch_models.py"
goto :DONE

:ACT_AICONFIG
call :REQUIRE_PY || goto :DONE
echo.
"%PY%" -c "from core.attendancePipeline import AIConfig; c=AIConfig(); print(c.log_summary())"
goto :DONE

REM ===========================================================================
REM  Test
REM ===========================================================================
:MODE_TEST
call :REQUIRE_READY || goto :DONE
cls
echo.
echo  +-- Test suite --------------------------------------------------+
echo  ^|                                                                ^|
echo  ^|  [1]  pytest              python -m pytest tests\ -q           ^|
echo  ^|  [2]  ruff                lint core, main_api.py, tests (CI)   ^|
echo  ^|  [3]  Both                ruff, then pytest                    ^|
echo  ^|  [4]  ML / behaviour ST   record clips, score ST-BH / ST-ML    ^|
echo  ^|                                                                ^|
echo  ^|  [0]  Back                                                     ^|
echo  +----------------------------------------------------------------+
echo.
set "TC="
set /p "TC=> Select [0-4]: "
if "!TC!"=="0" goto :MAIN_MENU
if "!TC!"=="4" goto :MODE_ST
if not "!TC!"=="1" if not "!TC!"=="2" if not "!TC!"=="3" (
    echo  Invalid choice.
    timeout /t 1 >nul
    goto :MODE_TEST
)

if "!TC!"=="2" goto :ACT_RUFF

:ACT_PYTEST
call :REQUIRE_READY || goto :DONE
REM - tests/ exercises the REAL database: it creates accounts, sessions and
REM   embeddings. Show which database before letting that happen.
call :READ_DBURL
echo.
echo  +-- WARNING -----------------------------------------------------+
echo  ^|                                                                ^|
echo  ^|  The test suite writes to the database in your .env. It        ^|
echo  ^|  creates accounts, sessions and embeddings. Point it at        ^|
echo  ^|  a LOCAL Postgres - not the live Supabase instance.            ^|
echo  ^|                                                                ^|
echo  +----------------------------------------------------------------+
echo.
echo    Target host: !DBHOST!
echo.
set "CONFIRM="
set /p "CONFIRM=> Type YES to run the tests (anything else cancels): "
if /i not "!CONFIRM!"=="YES" (
    echo  Cancelled. Nothing was run.
    goto :DONE
)

if "!TC!"=="3" (
    echo.
    echo  [test] ruff check core main_api.py tests
    "%PY%" -m ruff check core main_api.py tests
    if errorlevel 1 echo  ruff reported issues - see above.
)
echo.
echo  [test] python -m pytest tests\ -q
"%PY%" -m pytest "%~dp0tests" -q
goto :DONE

:ACT_RUFF
call :REQUIRE_READY || goto :DONE
echo.
echo  [test] ruff check core main_api.py tests
"%PY%" -m ruff check core main_api.py tests
goto :DONE

REM -
:MODE_ST
call :REQUIRE_READY || goto :DONE
cls
echo.
echo  +-- ML / behaviour measurement (ST-ML, ST-BH) -------------------+
echo  ^|                                                                ^|
echo  ^|  Scores the real detectors against labelled footage and writes ^|
echo  ^|  the evidence reports under docs\evidence\.                    ^|
echo  ^|                                                                ^|
echo  ^|  [1]  Record clips        guided capture, 1080p @ 1 fps        ^|
echo  ^|  [2]  Behaviour ST        score ST-BH from clips\              ^|
echo  ^|  [3]  ML ST               score ST-ML (detection / TAR / FAR)  ^|
echo  ^|                                                                ^|
echo  ^|  [0]  Back                                                     ^|
echo  ^|                                                                ^|
echo  ^|  Privacy: clips\ holds real facial footage. It is gitignored   ^|
echo  ^|  and must stay that way (CR-09 / PDPC).                        ^|
echo  +----------------------------------------------------------------+
echo.
set "MC="
set /p "MC=> Select [0-3]: "
if "!MC!"=="0" goto :MODE_TEST
if "!MC!"=="1" goto :ACT_CLIPS
if "!MC!"=="2" goto :ACT_STBH
if "!MC!"=="3" goto :ACT_STML
echo  Invalid choice.
timeout /t 1 >nul
goto :MODE_ST

:ACT_CLIPS
call :REQUIRE_READY || goto :DONE
echo.
"%PY%" "%~dp0tests\record_behaviour_clips.py" --camera 0 --out clips
goto :DONE

:ACT_STBH
call :REQUIRE_READY || goto :DONE
echo.
"%PY%" "%~dp0tests\run_behaviour_st.py" --clips clips --evidence docs/evidence
goto :DONE

:ACT_STML
call :REQUIRE_READY || goto :DONE
echo.
"%PY%" "%~dp0tests\run_ml_st.py" --clips clips --evidence docs/evidence
goto :DONE

REM ===========================================================================
REM  Helpers
REM ===========================================================================

REM - Locate Python (prefer the repo-local .venv). Never fatal: option [6]
REM   is what creates the venv, so the menu has to be reachable without one.
:DETECT_PY
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "PY="
set "PY_LABEL=NOT FOUND"
if exist "%VENV_PY%" (
    set "PY=%VENV_PY%"
    set "PY_LABEL=.venv"
) else (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        if not defined PY set "PY=%%P"
    )
    if defined PY set "PY_LABEL=system (no .venv)"
)
if defined PY (
    "%PY%" --version >nul 2>&1
    if errorlevel 1 (
        set "PY="
        set "PY_LABEL=BROKEN"
    )
)
exit /b 0

REM - .env presence and importability of the runtime deps. Both are status
REM   flags, not gates.
:DETECT_ENV
set "ENV_OK=0"
set "ENV_LABEL=MISSING"
if exist "%~dp0.env" (
    set "ENV_OK=1"
    set "ENV_LABEL=OK"
)
set "DEPS_OK=0"
set "DEPS_LABEL=MISSING"
if defined PY (
    "%PY%" -c "import fastapi, uvicorn, cv2, psycopg2, insightface" >nul 2>&1
    if not errorlevel 1 (
        set "DEPS_OK=1"
        set "DEPS_LABEL=OK"
    )
)
exit /b 0

REM - Read DATABASE_URL out of .env (first match wins) and derive a host
REM   label for the confirmation prompts, so nobody types YES blind.
:READ_DBURL
set "DBURL="
set "DBHOST=(unknown)"
if not exist "%~dp0.env" exit /b 0
for /f "usebackq tokens=1,* delims==" %%A in ("%~dp0.env") do (
    if /i "%%A"=="DATABASE_URL" if not defined DBURL set "DBURL=%%B"
)
if not defined DBURL exit /b 0
for /f "tokens=2 delims=@" %%H in ("!DBURL!") do set "DBHOST=%%H"
if "!DBHOST!"=="(unknown)" set "DBHOST=!DBURL!"
exit /b 0

REM - Kill whatever holds a TCP listener on %1. The ":<port> " pattern needs
REM   the colon so :8000 does not also match :18000.
:KILL_PORT
set "PORT=%~1"
set "LABEL=%~2"
for /f "tokens=5" %%P in ('netstat -ano -p tcp ^| findstr /r /c:":!PORT! .*LISTENING"') do (
    if not "%%P"=="0" (
        taskkill /F /PID %%P >nul 2>&1
        if not errorlevel 1 (
            echo   stopped !LABEL! ^(pid %%P^) on port !PORT!
            set "STOPPED=1"
        )
    )
)
exit /b 0

:REQUIRE_PY
if defined PY exit /b 0
echo.
echo  ERROR: no usable Python. Install Python 3.10+ ^(tick "Add python.exe
echo         to PATH"^), then run [6] to create the virtual environment.
exit /b 1

:REQUIRE_READY
call :REQUIRE_PY || exit /b 1
if not "!DEPS_OK!"=="1" (
    echo.
    echo  ERROR: the backend dependencies are not installed. Run [6] first.
    exit /b 1
)
if not "!ENV_OK!"=="1" (
    echo.
    echo  ERROR: .env not found at %~dp0
    echo  Run [6] to create it from .env.example, then set DATABASE_URL, e.g.:
    echo      DATABASE_URL=postgresql://fyp_user:fyp_user@localhost:5432/fyp_database
    exit /b 1
)
exit /b 0

REM - Open the point-and-click GUI. Returns 0 only if it actually ran, so the
REM   caller can fall back to the text menu on any older / locked-down box.
:TRY_GUI
REM   The GUI lives under scripts\ on purpose. A .ps1 has no file association
REM   on a default Windows install, so a copy sitting next to this file would
REM   just be a thing people double-click and get nothing from. This .bat is
REM   the only entry point.
set "GUI=%~dp0scripts\fras_launcher_gui.ps1"
if not exist "%GUI%" exit /b 1
where powershell >nul 2>&1
if errorlevel 1 exit /b 1

REM   This call returns as soon as the .ps1 has re-spawned itself with
REM   CREATE_NO_WINDOW (see the header there); returning 0 makes the caller
REM   exit, which closes this console and leaves the GUI alone on screen with
REM   no terminal behind it. A non-zero code means the GUI never started, so
REM   we fall through to the text menu while this console is still open.
echo.
echo  Opening the GUI launcher...
powershell -NoProfile -ExecutionPolicy Bypass -STA -File "%GUI%"
if errorlevel 1 (
    echo.
    echo  The GUI could not start - falling back to the text menu.
    timeout /t 2 >nul
    exit /b 1
)
exit /b 0

REM -
:DONE
echo.
if "!ERRORLEVEL!"=="0" (echo  Done.) else (echo  Exited with code !ERRORLEVEL!.)
echo.
REM - One action per window when the GUI (or the command line) asked for a
REM   specific one; only the interactive text menu loops. -
if defined DIRECT (
    pause
    goto :EXIT_NOW
)
pause
goto :MAIN_MENU

:END
echo.
echo  Goodbye.
pause
goto :EXIT_NOW

:EXIT_NOW
endlocal
exit /b 0
