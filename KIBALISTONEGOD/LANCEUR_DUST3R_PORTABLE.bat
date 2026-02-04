@echo off
REM Lanceur DUSt3R Portable - Reconstruction 3D Ultra-Réaliste
REM Détection automatique des URLs et adresses IP

echo.
echo ========================================
echo 🚀 DUSt3R - Reconstruction 3D Ultra-Réaliste
echo ========================================
echo.

REM Détection de l'adresse IP
for /f "tokens=2 delims=:" %%i in ('ipconfig ^| findstr /R /C:"IPv4 Address" ^| findstr /V /C:"127.0.0.1"') do (
    set IP=%%i
    goto :found
)

:found
set IP=%IP:~1%

if "%IP%"=="" (
    echo ⚠️ Impossible de détecter l'adresse IP automatiquement
    set IP=192.168.1.XX
)

echo 📡 Adresse IP détectée : %IP%
echo 🌐 URLs d'accès :
echo    Local : http://localhost:8501
echo    Réseau : http://%IP%:8501
echo.
echo 🐍 Python portable : %~dp0python311\python.exe
echo 📄 Script : %~dp0Dust3r.py
echo.
echo ⏳ Démarrage de Streamlit...
echo ========================================

REM Lancement avec Python portable
"%~dp0python311\python.exe" -m streamlit run "%~dp0Dust3r.py"

echo.
echo ✅ Application arrêtée.
pause