@echo off
cd /d "%~dp0"
REM Forcer l'utilisation de l'environnement portable
set PYTHONPATH=%~dp0Lib;%~dp0Lib\site-packages;%PYTHONPATH%
set PATH=%~dp0;%~dp0Scripts;%PATH%
".\python.exe" "risk_simulation_app.py"
pause