@echo off
REM Script de nettoyage complet pour Pylance et VS Code
echo ========================================
echo 🧹 NETTOYAGE COMPLET PYLANCE/VS CODE
echo ========================================
echo.

echo Étape 1: Arrêt des processus Python...
echo.

REM Arrêter tous les processus Python
taskkill /f /im python.exe 2>nul
taskkill /f /im python3.exe 2>nul
taskkill /f /im pythonw.exe 2>nul

echo.
echo Étape 2: Nettoyage du cache VS Code...
echo.

REM Nettoyer le cache de VS Code pour ce workspace
if exist "%APPDATA%\Code" (
    echo Nettoyage du cache utilisateur...

    REM Supprimer le cache de l'extension Python
    rd /s /q "%APPDATA%\Code\User\globalStorage\ms-python.python" 2>nul

    REM Supprimer les caches de workspace
    for /d %%i in ("%APPDATA%\Code\User\workspaceStorage\*") do (
        rd /s /q "%%i" 2>nul
    )

    REM Supprimer le cache Pylance
    rd /s /q "%APPDATA%\Code\User\globalStorage\ms-python.pylance" 2>nul

    REM Supprimer le cache Jedi
    rd /s /q "%APPDATA%\Code\User\globalStorage\ms-python.jedi" 2>nul

    echo ✅ Cache VS Code nettoyé
)

echo.
echo Étape 3: Nettoyage des caches locaux...
echo.

REM Nettoyer les caches Python locaux
if exist "__pycache__" rd /s /q "__pycache__" 2>nul
if exist "*.pyc" del /q "*.pyc" 2>nul
if exist ".pytest_cache" rd /s /q ".pytest_cache" 2>nul
if exist ".mypy_cache" rd /s /q ".mypy_cache" 2>nul

REM Nettoyer les caches spécifiques au projet
if exist "python311\__pycache__" rd /s /q "python311\__pycache__" 2>nul
if exist "python311\*.pyc" del /q "python311\*.pyc" 2>nul

echo ✅ Caches locaux nettoyés

echo.
echo Étape 4: Vérification de l'environnement...
echo.

if exist "python311\python.exe" (
    echo ✅ Environnement Python portable trouvé
    python311\python.exe -c "import sys; print('Version:', sys.version.split()[0]); print('Chemin:', sys.executable)"
) else (
    echo ❌ Environnement Python portable manquant
)

echo.
echo Étape 5: Test des imports critiques...
echo.

if exist "python311\python.exe" (
    python311\python.exe -c "
try:
    import streamlit, torch, numpy, PIL, plotly, open3d
    print('✅ Imports de base réussis')
except ImportError as e:
    print('❌ Import error:', e)
"
)

echo.
echo ========================================
echo 🎯 INSTRUCTIONS DE REDÉMARRAGE
echo ========================================
echo.
echo NETTOYAGE TERMINÉ! Suivez ces étapes:
echo.
echo 1️⃣ Fermez COMPLETEMENT VS Code (Menu Fichier → Quitter)
echo 2️⃣ Attendez 10 secondes
echo 3️⃣ Redémarrez VS Code
echo 4️⃣ Ouvrez le dossier KIBALISTONEGOD
echo 5️⃣ Ouvrez Dust3r.py
echo.
echo ✅ Les erreurs Pylance devraient avoir disparu!
echo.
echo 🔧 Si des erreurs persistent:
echo - Lancez: install_all_dependencies.bat
echo - Puis relancez ce script
echo - Redémarrez votre PC
echo.

pause