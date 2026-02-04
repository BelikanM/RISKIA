@echo off
REM ============================================
REM DUST3R PORTABLE LAUNCHER
REM Fonctionne sur n'importe quel disque (C:, D:, E:, carte SD, USB, etc.)
REM ============================================

REM Detecte automatiquement le dossier du script
cd /d "%~dp0"

echo ========================================
echo DUST3R - Application Portable
echo ========================================
echo.
echo Repertoire actuel: %CD%
echo.

REM Verifie que venv existe
if not exist "venv\Scripts\python.exe" (
    echo [ERREUR] Environnement virtuel non trouve!
    echo Assurez-vous que le dossier 'venv' est present.
    echo.
    pause
    exit /b 1
)

REM Verifie que Dust3r.py existe
if not exist "Dust3r.py" (
    echo [ERREUR] Dust3r.py non trouve!
    echo.
    pause
    exit /b 1
)

echo [OK] Environnement virtuel detecte
echo [OK] Dust3r.py detecte
echo.

REM ========================================
REM DETECTION AUTOMATIQUE DE PORT LIBRE
REM ========================================
echo Recherche d'un port disponible...
echo.

set "PORT=8501"
set "PORT_FOUND=0"

:CHECK_PORT
netstat -ano | findstr ":%PORT% " >nul 2>&1
if %errorlevel% equ 0 (
    echo [OCCUPE] Port %PORT% deja utilise, test suivant...
    set /a PORT+=1
    if %PORT% gtr 8599 (
        echo [ERREUR] Aucun port libre trouve entre 8501-8599
        pause
        exit /b 1
    )
    goto CHECK_PORT
) else (
    echo [OK] Port %PORT% disponible !
    set "PORT_FOUND=1"
)

echo.
echo Demarrage de l'application...
echo.
echo L'application s'ouvrira dans votre navigateur.
echo URL locale:    http://localhost:%PORT%
echo URL reseau:    http://%COMPUTERNAME%:%PORT%
echo.

REM Active l'environnement virtuel pour s'assurer que le bon Python est utilisé
call "%~dp0venv\Scripts\activate.bat"

REM Lance Streamlit avec le Python du venv
"%~dp0venv\Scripts\python.exe" -m streamlit run "%~dp0Dust3r.py" --server.port %PORT% --server.headless true

REM Si erreur
if %errorlevel% neq 0 (
    echo.
    echo [ERREUR] L'application a rencontre une erreur.
    echo Code d'erreur: %errorlevel%
    echo.
)

pause
