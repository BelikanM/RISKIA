@echo off
echo Activation de l'environnement virtuel...
call "%~dp0venv\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo Erreur lors de l'activation de l'environnement virtuel
    pause
    exit /b 1
)

echo Lancement de l'application Dust3r...
"%~dp0venv\Scripts\streamlit.exe" run Dust3r.py --server.port 8501 --server.headless true %*
pause