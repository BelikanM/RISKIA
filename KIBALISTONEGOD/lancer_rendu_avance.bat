@echo off
REM Lanceur rapide pour le rendu avancé
echo ========================================
echo  🎬 Lancement Rendu Avancé Pro
echo ========================================
echo.

REM Vérifier si Python portable existe
if exist "python311\python.exe" (
    echo Utilisation de Python portable...
    set PYTHON="python311\python.exe"
    set STREAMLIT="python311\Scripts\streamlit.exe"
) else (
    set PYTHON=python
    set STREAMLIT=streamlit
)

REM Vérifier les dépendances
echo Vérification des dépendances...
%PYTHON% -c "import pyrender, trimesh, cv2; print('✅ Dépendances OK')" 2>nul
if errorlevel 1 (
    echo ❌ Dépendances manquantes. Lancez install_rendu_avance.bat d'abord.
    pause
    exit /b 1
)

echo.
echo 🚀 Lancement de la démonstration du rendu avancé...
echo.
echo Cette démo montre les capacités photoréalistes
echo qui surpassent Blender en qualité et performance!
echo.

REM Lancer la démo
%STREAMLIT% run demo_rendu_avance.py --server.port 8502 --server.address 0.0.0.0

pause