@echo off
echo =======================================
echo    Starting Wonder AI Backend Server
echo =======================================

if not exist venv (
    echo [1/4] Creating Python Virtual Environment...
    python -m venv venv
)

echo [2/4] Activating Virtual Environment...
call venv\Scripts\activate.bat

echo [3/4] Installing Required Packages...
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo [4/4] Installing Browser Binaries for Data Scraping...
python -m playwright install

echo =======================================
echo    Server starting on port 8000
echo =======================================
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
