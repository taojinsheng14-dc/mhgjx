@echo off
setlocal
cd /d "%~dp0"

if not exist "toolbox.py" (
  echo toolbox.py not found.
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^pythonw?\.exe$' -and $_.CommandLine -like '*scrcpy_human_automation*toolbox.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>nul

if not exist ".venv\Scripts\pythonw.exe" (
  echo Creating Python venv...
  py -3.13 -m venv .venv
  if errorlevel 1 py -3 -m venv .venv
  if errorlevel 1 (
    echo Failed to create venv. Please install Python first.
    pause
    exit /b 1
  )
  if exist "requirements.txt" (
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  )
)

start "scrcpy human automation toolbox" /D "%~dp0" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0toolbox.py"
exit /b 0
