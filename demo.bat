@echo off
setlocal enableextensions enabledelayedexpansion

REM Always run from the folder where this .bat lives
cd /d "%~dp0"

REM - Locate Python (absolute path) -
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
set "PY="

if exist "%VENV_PY%" (
    set "PY=%VENV_PY%"
) else (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        if not defined PY set "PY=%%P"
    )
)

if not defined PY (
    echo.
    echo  ERROR: Python not found.
    echo  Make sure Python 3.10+ is installed, or that the .venv folder exists.
    echo  Expected venv path: %VENV_PY%
    echo.
    pause
    exit /b 1
)

echo  [init] Python: %PY%

REM - Verify Python runs -
"%PY%" --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: Python found but cannot execute: %PY%
    echo.
    pause
    exit /b 1
)

REM - .env check -
if not exist "%~dp0.env" (
    echo.
    echo  ERROR: .env not found at %~dp0
    echo  Copy .env.example to .env and fill in DATABASE_URL.
    echo.
    pause
    exit /b 1
)

REM - Dependency check -
"%PY%" -c "import insightface, cv2, psycopg2, fastapi" >nul 2>&1
if errorlevel 1 (
    echo  [setup] Installing dependencies, please wait...
    "%PY%" -m pip install -r "%~dp0requirements.txt"
    if errorlevel 1 (
        echo.
        echo  ERROR: pip install failed.
        echo.
        pause
        exit /b 1
    )
)

REM -
:MAIN_MENU
cls
echo.
echo  +--------------------------------------------------+
echo  ^|  FYP-26-S2-17  Attendance AI   Demo Launcher     ^|
echo  +--------------------------------------------------+
echo.
echo    [1]  Image mode      -- run on sample/test images
echo    [2]  Webcam          -- physical camera (index 0)
echo    [3]  Virtual camera  -- choose camera index
echo    [4]  Video file      -- local file or RTSP stream
echo    [5]  Ping only       -- verify Supabase + models
echo    [6]  List cameras    -- probe indices 0-9
echo    [7]  Start Web API   -- launch FastAPI server on 127.0.0.1:8000
echo    [0]  Exit
echo.

set "CHOICE="
set /p "CHOICE=  Select [0-7]: "

if "!CHOICE!"=="0" goto :END
if "!CHOICE!"=="1" goto :MODE_IMAGE
if "!CHOICE!"=="2" goto :MODE_WEBCAM
if "!CHOICE!"=="3" goto :MODE_VIRTUAL
if "!CHOICE!"=="4" goto :MODE_VIDEO
if "!CHOICE!"=="5" goto :MODE_PING
if "!CHOICE!"=="6" goto :MODE_LIST
if "!CHOICE!"=="7" goto :MODE_WEB
echo  Invalid choice, try again.
timeout /t 1 >nul
goto :MAIN_MENU

REM -
:MODE_IMAGE
echo.
echo  [demo] Image mode...
"%PY%" -m ai.demo --mode image --no-enrol
goto :DONE

:MODE_WEBCAM
echo.
echo  [demo] Opening camera 0...
"%PY%" -m ai.demo --mode webcam --camera 0 --no-enrol
goto :DONE

:MODE_VIRTUAL
echo.
echo  Common virtual camera indices: OBS=1, ManyCam=1, DroidCam=1
echo  (Run option 6 first to find yours)
echo.
set "CAM_IDX="
set /p "CAM_IDX=  Camera index: "
if "!CAM_IDX!"=="" goto :MODE_VIRTUAL

set "VALID=0"
for %%D in (0 1 2 3 4 5 6 7 8 9) do if "!CAM_IDX!"=="%%D" set "VALID=1"
if "!VALID!"=="0" (
    echo  Please enter a digit 0-9.
    timeout /t 1 >nul
    goto :MODE_VIRTUAL
)

set "COURSE_ARG="
set /p "COURSE_ID=  Course ID for attendance (blank to skip): "
if not "!COURSE_ID!"=="" set "COURSE_ARG=--course !COURSE_ID!"

echo.
echo  [demo] Opening virtual camera !CAM_IDX!...
"%PY%" -m ai.demo --mode webcam --camera !CAM_IDX! !COURSE_ARG! --no-enrol
goto :DONE

:MODE_VIDEO
echo.
echo  Examples: test.mp4  /  C:\Videos\lecture.avi  /  rtsp://192.168.1.10/stream
echo.
set "VSRC="
set /p "VSRC=  File path or URL: "
if "!VSRC!"=="" goto :MAIN_MENU

set "COURSE_ARG="
set /p "COURSE_ID=  Course ID for attendance (blank to skip): "
if not "!COURSE_ID!"=="" set "COURSE_ARG=--course !COURSE_ID!"

echo.
echo  [demo] Opening: !VSRC!
"%PY%" -m ai.demo --mode webcam --video "!VSRC!" !COURSE_ARG! --no-enrol
goto :DONE

:MODE_PING
echo.
echo  [demo] Checking Supabase + models...
"%PY%" -m ai.run_prototype --ping
goto :DONE

:MODE_LIST
echo.
echo  [demo] Probing camera indices 0-9...
"%PY%" -m ai.run_prototype --list-cameras
echo.
echo  Start OBS/ManyCam first if your virtual camera is missing.
goto :DONE

:MODE_WEB
echo.
echo  [demo] Launching FastAPI server at http://127.0.0.1:8000 ...
echo  (Press Ctrl+C in this window to stop the server)
echo.
"%PY%" -m uvicorn api.main_api:app --host 127.0.0.1 --port 8000
goto :DONE

REM -
:DONE
echo.
if "!ERRORLEVEL!"=="0" (echo  Done.) else (echo  Exited with code !ERRORLEVEL!.)
echo.
pause
goto :MAIN_MENU

:END
echo.
echo  Goodbye.
pause
endlocal
exit /b 0
