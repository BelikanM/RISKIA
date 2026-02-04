@echo off
echo ============================================
echo   LANCEMENT VS CODE AVEC PYTHON PORTABLE
echo ============================================
echo.

cd /d "C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD"

echo Verification de l'existence de VS Code...
where code >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ VS Code n'est pas installe ou n'est pas dans le PATH
    echo Veuillez installer VS Code et l'ajouter au PATH systeme
    pause
    exit /b 1
)

echo.
echo Lancement de VS Code avec le dossier KIBALISTONEGOD...
echo Le Python portable sera automatiquement utilise
echo.

code "C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD" --new-window

echo.
echo ✅ VS Code lance avec la configuration Python portable
echo Les erreurs Pylance devraient disparaitre automatiquement
echo.

pause