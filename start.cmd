@echo off
echo =======================================
echo    Starting Wonder AI Backend Server
echo =======================================

if not exist venv (
    echo [1/4] Creating Python Virtual Environment...
    "C:\Users\Laptop\AppData\Local\Programs\Python\Python312\python.exe" -m venv venv
)

echo [2/4] Activating Virtual Environment...
call venv\Scripts\activate.bat

echo [3/4] Installing Required Packages...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo [4/4] Installing Browser Binaries for Data Scraping...
playwright install chromium

echo =======================================
echo    Server starting on port 8000
echo =======================================
uvicorn main:app --reload
