@echo off
cd /d "%~dp0"
set "PATH=%~dp0.tools;%PATH%"
set "DENO_DIR=%~dp0data\models\deno"
if not exist .venv (echo Run setup.bat first. & pause & exit /b 1)
echo Starting Arsu Clip at http://127.0.0.1:8765  (close this window to stop)
start "" http://127.0.0.1:8765
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8765
pause
