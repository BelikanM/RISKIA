@echo off
REM Lanceur rapide pour Dust3r avec vérifications
echo ========================================
echo 🚀 LANCEMENT RAPIDE DE DUST3R
echo ========================================
echo.

set PYTHON_EXE=C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD\python311\python.exe
set APP_FILE=Dust3r.py

REM Vérifications rapides
if not exist "%PYTHON_EXE%" (
    echo ❌ Python portable non trouvé
    echo Lancez d'abord final_pylance_fix.bat
    pause
    exit /b 1
)

if not exist "%APP_FILE%" (
    echo ❌ Fichier Dust3r.py non trouvé
    pause
    exit /b 1
)

echo ✅ Environnement prêt
echo.

REM Lancement de l'application
echo 🌟 Démarrage de Dust3r...
echo 📱 URL: http://localhost:8501
echo.

"%PYTHON_EXE%" -m streamlit run "%APP_FILE%" --server.port 8501 --server.address 0.0.0.0

echo.
echo ✅ Application arrêtée
pause