@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
if exist ".venv\Scripts\python.exe" goto check_dependencies
echo Creating the local Python environment...
py -3.10 -m venv .venv
if errorlevel 1 goto failed

:check_dependencies
".venv\Scripts\python.exe" -c "from importlib.metadata import version; import tkinter; [version(n) for n in ('numpy', 'Pillow', 'PyYAML', 'torch', 'torchvision', 'ultralytics', 'onnx', 'onnxruntime', 'opencv-python')]" >nul 2>&1
if not errorlevel 1 goto launch
echo Installing packages. The first launch needs an internet connection.
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -c "from importlib.metadata import version; version('torch'); version('torchvision')" >nul 2>&1
if not errorlevel 1 goto install_packages
".venv\Scripts\python.exe" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed

:install_packages
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed

:launch
".venv\Scripts\python.exe" app.py
if errorlevel 1 goto failed
exit /b 0

:failed
echo.
echo Setup or application failed. See the message above.
echo Install Python 3.10 with Tcl/Tk and the Python launcher, then retry.
echo For NVIDIA GPU training, run setup_gpu.cmd.
pause
exit /b 1
