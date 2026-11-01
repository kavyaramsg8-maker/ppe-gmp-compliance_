@echo off
cd /d "%~dp0"
echo Starting GMP PPE Monitoring System...

:: Set your API key
set PPE_API_KEY=this-is-a-much-longer-super-secret-key-for-testing-123456789

:: 1. Start the Cloud Backend (cloud venv)
start "1. Cloud Backend" cmd /k "call .venv-cloud\Scripts\activate.bat && python -m cloud_backend.serve"

:: 2. Start the AI Worker (CV venv)
start "2. AI Worker" cmd /k "call venv\Scripts\activate.bat && python -m cloud_backend.worker --detector ai_detector:detect"

:: 3. Start the Streamlit Dashboard (cloud venv)
start "3. Dashboard" cmd /k "call .venv-cloud\Scripts\activate.bat && streamlit run dashboard/app.py"

:: 4. Start the Live Monitor (CV venv)
start "4. Live Monitor" cmd /k "call venv\Scripts\activate.bat && python monitor.py"

:: 5. Start ngrok Tunnel (bundled ngrok.exe)
start "5. Ngrok Tunnel" cmd /k "%~dp0ngrok.exe http 8000"

echo All systems started!
pause