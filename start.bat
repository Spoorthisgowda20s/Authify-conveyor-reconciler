@echo off
rem One command on Windows (cmd or PowerShell):  start.bat
rem Needs Python 3.9+ on PATH. Creates .venv, installs numpy + opencv-python-headless, starts the server.
cd /d "%~dp0"
set PY=python
where python >nul 2>nul || set PY=py
%PY% --version >nul 2>nul || (echo Python not found. Install it from https://www.python.org/downloads/windows/ and tick "Add python.exe to PATH". & exit /b 1)
if not exist .venv\Scripts\python.exe %PY% -m venv .venv
.venv\Scripts\python.exe -m pip install -q -r requirements.txt || (echo Dependency install failed. & exit /b 1)
echo Starting... open http://127.0.0.1:8000/ in your browser
.venv\Scripts\python.exe run.py %*
