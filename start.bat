@echo off
title CropSense - Crop Disease Prediction Server
color 0A

echo.
echo  ====================================================
echo       CropSense - AI Plant Disease Diagnostics
echo  ====================================================
echo.

:: Navigate to the project directory
cd /d "%~dp0"

:: Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo  [!] Virtual environment not found. Creating one...
    python -m venv venv
    echo  [✓] Virtual environment created.
    echo.
    echo  [~] Installing dependencies...
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
    echo  [✓] Dependencies installed.
) else (
    call venv\Scripts\activate.bat
)

echo.
echo  [~] Starting server...
echo  [~] Once started, your browser will open automatically.
echo  [~] Press Ctrl+C in this window to stop the server.
echo.
echo  ====================================================
echo       Server: http://localhost:8000
echo  ====================================================
echo.

:: Open browser after a short delay (in background)
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8000"

:: Start the FastAPI server
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

pause
