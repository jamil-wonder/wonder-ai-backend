@echo off
echo =======================================
echo    Starting Wonder AI Backend Server
echo =======================================

set "ENV_DIR=.venv"
set "PY_EXE="
set "BACKEND_PORT=%BACKEND_PORT%"
if "%BACKEND_PORT%"=="" set "BACKEND_PORT=8000"
if not exist "%ENV_DIR%" (
    if exist "venv" (
        set "ENV_DIR=venv"
    ) else (
        echo [1/4] Creating Python Virtual Environment ^(.venv^) ...
        python -m venv .venv
    )
)
set "PY_EXE=%ENV_DIR%\Scripts\python.exe"

if not exist "%PY_EXE%" (
    echo [ERROR] Python executable not found in %ENV_DIR%.
    exit /b 1
)

echo [2/4] Using virtual environment: %ENV_DIR%

if not exist "%ENV_DIR%\_deps_installed.flag" (
    echo [3/4] Installing Required Packages...
    "%PY_EXE%" -m pip install --upgrade pip setuptools wheel
    "%PY_EXE%" -m pip install -r requirements.txt
    type nul > "%ENV_DIR%\_deps_installed.flag"
) else (
    echo [3/4] Dependencies already installed. Skipping.
)

if not exist "%ENV_DIR%\_playwright_installed.flag" (
    echo [4/4] Installing Browser Binaries for Data Scraping...
    "%PY_EXE%" -m playwright install
    type nul > "%ENV_DIR%\_playwright_installed.flag"
) else (
    echo [4/4] Playwright browsers already installed. Skipping.
)

echo =======================================
echo    Server starting on port %BACKEND_PORT%
echo =======================================
if /I "%ENABLE_RELOAD%"=="1" (
    echo [MODE] Uvicorn reload enabled
    "%PY_EXE%" -m uvicorn main:app --host 0.0.0.0 --port %BACKEND_PORT% --reload
) else (
    echo [MODE] Uvicorn reload disabled for stability
    "%PY_EXE%" -m uvicorn main:app --host 0.0.0.0 --port %BACKEND_PORT%
)
