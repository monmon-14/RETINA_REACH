@echo off
title RetinaCam - Fundus Grad-CAM Explainer
cd /d "%~dp0"
echo ===================================================
echo     Launching RetinaCam: Fundus Grad-CAM Explainer
echo ===================================================
echo.
if not exist "venv\Scripts\python.exe" (
    echo Error: Virtual environment not found. Please initialize venv first.
    pause
    exit /b 1
)

echo Starting Streamlit web application...
venv\Scripts\streamlit.exe run app.py
pause
