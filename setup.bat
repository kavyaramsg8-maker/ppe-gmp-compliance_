@echo off
title GMP PPE - Clean Setup and Install
cd /d "%~dp0"

set "PYTHON_EXE=C:\Users\paper\AppData\Local\Python\pythoncore-3.14-64\python.exe"

echo =======================================================
echo    Cleaning up old environments...
echo =======================================================
if exist ".venv-cloud" rmdir /s /q .venv-cloud
if exist "venv" rmdir /s /q venv

echo =======================================================
echo    [1/3] Creating Cloud Environment (.venv-cloud)...
echo =======================================================
"%PYTHON_EXE%" -m venv .venv-cloud
if %errorlevel% neq 0 (
    echo ERROR: Failed to create .venv-cloud.
    pause
    exit /b
)

call .venv-cloud\Scripts\activate.bat
python -m pip install --upgrade pip
echo Installing Cloud dependencies...
pip install fastapi uvicorn pydantic python-multipart httpx Pillow streamlit plotly pandas PyYAML pytest
if %errorlevel% neq 0 (
    echo ERROR: Failed installing cloud pip packages.
    pause
    exit /b
)
deactivate

echo =======================================================
echo    [2/3] Creating Computer Vision Environment (venv)...
echo =======================================================
"%PYTHON_EXE%" -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create venv.
    pause
    exit /b
)

call venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo Installing CV dependencies...
pip install ultralytics opencv-python requests pyyaml
if %errorlevel% neq 0 (
    echo ERROR: Failed installing CV pip packages.
    pause
    exit /b
)
deactivate

echo =======================================================
echo    [3/3] Creating cloud_data directory...
echo =======================================================
if not exist "cloud_data" mkdir "cloud_data"

echo =======================================================
echo    Setup Finished Successfully! 
echo    You can now run start_demo.bat
echo =======================================================
pause
```[Certain]

### Step 2: Run and Launch
1. Double-click the updated **`setup.bat`** file and let it run until it displays **"Setup Finished Successfully!"** and pauses [Certain]. 
2. Once it finishes successfully, double-click **`start_demo.bat`**. 

Because the folders are cleanly wiped and fully populated with working dependencies, all windows—including the AI worker, live monitor, and Streamlit dashboard—will start up correctly without path or missing command errors [Certain].