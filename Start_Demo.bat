@echo off
echo Starting GMP PPE Monitoring System...

:: Set your API key
set PPE_API_KEY=this-is-a-much-longer-super-secret-key-for-testing-123456789

:: 1. Start the Cloud Backend (Uses cloud venv)
start "1. Cloud Backend" cmd /k "call .venv-cloud\Scripts\activate.bat && python -m cloud_backend.serve"

:: 2. Start the AI Worker (Uses the Desktop CV venv)
start "2. AI Worker" cmd /k "call C:\Users\cczuj\OneDrive\Desktop\GMP_PPE_Project\venv\Scripts\activate.bat && python -m cloud_backend.worker --detector ai_detector:detect"

:: 3. Start the Streamlit Dashboard (Uses cloud venv)
start "3. Dashboard" cmd /k "call .venv-cloud\Scripts\activate.bat && streamlit run dashboard/app.py"

:: 4. Start the Live Monitor (Uses the Desktop CV venv)
start "4. Live Monitor" cmd /k "call C:\Users\cczuj\OneDrive\Desktop\GMP_PPE_Project\venv\Scripts\activate.bat && python monitor.py"

:: 5. Start ngrok Tunnel
start "5. Ngrok Tunnel" cmd /k "ngrok http 8000"

echo All systems started!
pause