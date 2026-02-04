@echo off
REM Script pour corriger les erreurs Pylance
echo ========================================
echo 🔧 Correction des erreurs Pylance
echo ========================================
echo.

echo Étape 1: Redémarrage des services Python...
echo.

REM Forcer l'arrêt des processus Python
taskkill /f /im python.exe 2>nul
taskkill /f /im python3.exe 2>nul

echo.
echo Étape 2: Nettoyage du cache Pylance...
echo.

REM Supprimer le cache Pylance/VS Code
if exist "%APPDATA%\Code\User\workspaceStorage" (
    for /d %%i in ("%APPDATA%\Code\User\workspaceStorage\*") do (
        if exist "%%i\workspace.json" (
            findstr /c:"KIBALISTONEGOD" "%%i\workspace.json" >nul 2>&1
            if !errorlevel! equ 0 (
                rd /s /q "%%i" 2>nul
            )
        )
    )
)

REM Supprimer les caches Python
if exist "__pycache__" rd /s /q "__pycache__"
if exist "*.pyc" del /q "*.pyc"

echo.
echo Étape 3: Vérification de l'environnement Python...
echo.

if exist "python311\python.exe" (
    echo ✅ Environnement Python portable trouvé
    python311\python.exe -c "import sys; print('Python:', sys.version); print('Executable:', sys.executable)"
) else (
    echo ❌ Environnement Python portable non trouvé
)

echo.
echo Étape 4: Test des imports principaux...
echo.

if exist "python311\python.exe" (
    python311\python.exe -c "
try:
    import streamlit, torch, numpy, PIL
    print('✅ Imports de base OK')
except ImportError as e:
    print('❌ Import error:', e)
"
)

echo.
echo ========================================
echo ✅ Correction terminée!
echo ========================================
echo.
echo Instructions:
echo 1. Fermez complètement VS Code
echo 2. Redémarrez VS Code
echo 3. Ouvrez le workspace KIBALISTONEGOD
echo 4. Les erreurs Pylance devraient avoir disparu
echo.
echo Si les erreurs persistent:
echo - Vérifiez que python311\python.exe existe
echo - Lancez install_rendu_avance.bat si nécessaire
echo.

pause