@echo off
REM Script wrapper pour isoler l'environnement portable
cd /d "%~dp0"

REM Sauvegarder les variables d'environnement système
set "ORIGINAL_PYTHONPATH=%PYTHONPATH%"
set "ORIGINAL_PATH=%PATH%"

REM Définir l'environnement portable en priorité absolue
set "PYTHONPATH=%~dp0Lib;%~dp0Lib\site-packages;%~dp0;%PYTHONPATH%"
set "PATH=%~dp0;%~dp0Scripts;%PATH%"

REM Désactiver temporairement les installations système
set "PYTHONHOME="
set "PYTHONNOUSERSITE=1"

REM Lancer l'application
python.exe "risk_simulation_app.py"

REM Restaurer les variables d'environnement
set "PYTHONPATH=%ORIGINAL_PYTHONPATH%"
set "PATH=%ORIGINAL_PATH%"

pause