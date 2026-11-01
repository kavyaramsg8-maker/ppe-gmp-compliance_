@echo off
title GMP PPE - System Launcher
cd /d "%~dp0"
echo Starting GMP PPE Monitoring System...

:: Set your API key
set PPE_API_KEY=this-is-a-much-longer-super-secret-key-for-testing-123456789

:: 1. Start the Cloud Backend
echo Starting Cloud Backend...
start "1. Cloud Backend" cmd /k "call .venv-cloud\Scripts\activate.bat && python -m cloud_backend.serve || pause"

:: 2. Start the AI Worker
echo Starting AI Worker...
start "2. AI Worker" cmd /k "call venv\Scripts\activate.bat && python -m cloud_backend.worker --detector ai_detector:detect || pause"

:: 3. Start the Streamlit Dashboard
echo Starting Dashboard...
start "3. Dashboard" cmd /k "call .venv-cloud\Scripts\activate.bat && streamlit run dashboard/app.py || pause"

:: 4. Start the Live Monitor
echo Starting Live Monitor...
start "4. Live Monitor" cmd /k "call venv\Scripts\activate.bat && python monitor.py || pause"

:: 5. Start ngrok Tunnel
echo Starting Ngrok Tunnel...
start "5. Ngrok Tunnel" cmd /k "%~dp0ngrok.exe http 8000 || pause"

echo =======================================================
echo All launch commands executed. Check individual windows
echo for any specific error messages.
echo =======================================================
pause