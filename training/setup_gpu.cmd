@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if exist ".venv\Scripts\python.exe" goto install
echo Creating the local Python environment...
py -3.10 -m venv .venv
if errorlevel 1 goto failed

:install
echo Installing NVIDIA CUDA 12.1 PyTorch. The download is several GB.
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements-gpu.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); raise SystemExit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 goto failed
echo.
echo GPU setup complete. Double-click start.cmd to open the application.
pause
exit /b 0

:failed
echo.
echo GPU setup failed. See the message above and check your NVIDIA driver.
echo The application can also train on CPU through start.cmd.
pause
exit /b 1
