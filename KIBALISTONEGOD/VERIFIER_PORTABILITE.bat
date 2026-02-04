@echo off
REM ============================================
REM VERIFICATEUR DE PORTABILITE DUST3R
REM ============================================

cd /d "%~dp0"

echo ========================================
echo VERIFICATION DE PORTABILITE DUST3R
echo ========================================
echo.

set "ERREURS=0"

echo [CHECK 1/7] Detection du dossier...
echo Dossier actuel: %CD%
echo Lecteur: %~d0
echo.

echo [CHECK 2/7] Verification venv/...
if exist "venv\Scripts\python.exe" (
    echo [OK] Python trouve: venv\Scripts\python.exe
) else (
    echo [ERREUR] Python manquant dans venv\Scripts\
    set /a ERREURS+=1
)
echo.

echo [CHECK 3/7] Verification Streamlit...
if exist "venv\Scripts\streamlit.exe" (
    echo [OK] Streamlit trouve: venv\Scripts\streamlit.exe
) else (
    echo [ERREUR] Streamlit manquant
    set /a ERREURS+=1
)
echo.

echo [CHECK 4/7] Verification Dust3r.py...
if exist "Dust3r.py" (
    echo [OK] Application principale trouvee
) else (
    echo [ERREUR] Dust3r.py manquant
    set /a ERREURS+=1
)
echo.

echo [CHECK 5/7] Verification module dust3r/...
if exist "dust3r\__init__.py" (
    echo [OK] Module dust3r detecte
) else (
    echo [ATTENTION] Module dust3r absent (peut causer erreurs)
)
echo.

echo [CHECK 6/7] Verification modeles IA...
if exist "models--naver--DUSt3R_ViTLarge_BaseDecoder_512_dpt\" (
    echo [OK] Modeles IA detectes
) else (
    echo [ATTENTION] Modeles IA absents (seront telecharges au 1er lancement)
)
echo.

echo [CHECK 7/7] Test version Python...
"%~dp0venv\Scripts\python.exe" --version 2>nul
if %errorlevel% equ 0 (
    echo [OK] Python fonctionnel
) else (
    echo [ERREUR] Python ne repond pas
    set /a ERREURS+=1
)
echo.

echo ========================================
echo RESULTAT FINAL
echo ========================================

if %ERREURS% equ 0 (
    echo.
    echo   ████████████████████████████████
    echo   █                              █
    echo   █   ✓ APPLICATION PORTABLE !   █
    echo   █                              █
    echo   ████████████████████████████████
    echo.
    echo Vous pouvez copier le dossier A3E sur:
    echo   - Carte SD
    echo   - SSD externe
    echo   - Cle USB
    echo   - Disque dur externe
    echo.
    echo Lancez avec: DUST3R_PORTABLE.bat
    echo.
) else (
    echo.
    echo   ████████████████████████████████
    echo   █                              █
    echo   █   ✗ ERREURS DETECTEES: %ERREURS%    █
    echo   █                              █
    echo   ████████████████████████████████
    echo.
    echo Corrigez les erreurs ci-dessus avant de copier.
    echo.
)

echo ========================================
echo INFORMATIONS SYSTEME
echo ========================================
echo.

REM Calcul taille approximative
echo Calcul de la taille totale...
for /f "tokens=3" %%a in ('dir /s /-c "%~dp0" 2^>nul ^| find "octets"') do set "SIZE=%%a"
echo Taille totale: %SIZE% octets
echo.

REM Info disque
for /f "tokens=3" %%a in ('dir /-c "%~d0\" 2^>nul ^| find "disponible"') do set "FREE=%%a"
echo Espace libre sur %~d0\: %FREE% octets
echo.

echo ========================================

pause
