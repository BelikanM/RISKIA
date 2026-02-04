@echo off
echo ============================================
echo   LANCEMENT DUST3R - PHOTOGRAMMETRIE IA
echo ============================================
echo.

cd /d "C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD"

echo Verification des dependances...
C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD\python311\python.exe -c "import streamlit, torch, PIL, numpy, plotly, open3d, transformers, pynvml, faiss, pandas, sklearn, psutil; print('✓ Toutes les dependances sont presentes')"

if %errorlevel% neq 0 (
    echo ❌ Erreur: Dependances manquantes
    pause
    exit /b 1
)

echo.
echo Detection de l'adresse IP locale...
for /f "tokens=2 delims=:" %%i in ('ipconfig ^| findstr /R /C:"IPv4 Address"') do (
    for /f "tokens=*" %%a in ("%%i") do (
        set IP=%%a
        goto :found_ip
    )
)

:found_ip
if defined IP (
    echo Adresse IP detectee: %IP%
    echo.
    echo Lancement de l'application Streamlit...
    echo L'application sera accessible sur:
    echo   - http://localhost:8501
    echo   - http://%IP%:8501
    echo.
) else (
    echo Adresse IP non detectee, utilisation de localhost uniquement
    echo.
    echo Lancement de l'application Streamlit...
    echo L'application sera accessible sur http://localhost:8501
    echo.
)

C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD\python311\python.exe -m streamlit run Dust3r.py --server.port 8501 --server.address 0.0.0.0

pause