@echo off
echo ====================================================
echo Starting Crop Disease Prediction Backend Server...
echo ====================================================
cd /d "%~dp0"

echo Installing any missing requirements...
pip install -r requirements.txt

echo.
echo Starting app.py...
echo Once you see "Uvicorn running on http://0.0.0.0:8000",
echo You can open your browser to http://localhost:8000
echo.
python app.py
pause
