@echo off
REM Script pour forcer la reconnaissance de l'environnement Python par Pylance
echo ========================================
echo 🔧 FORÇAGE DE LA RECONNAISSANCE PYLANCE
echo ========================================
echo.

echo Étape 1: Nettoyage complet du cache VS Code...
echo.

REM Supprimer tous les caches VS Code pour ce workspace
if exist "%APPDATA%\Code\User\workspaceStorage" (
    rd /s /q "%APPDATA%\Code\User\workspaceStorage" 2>nul
    echo ✅ Cache workspace nettoyé
)

if exist "%APPDATA%\Code\User\globalStorage\ms-python.python" (
    rd /s /q "%APPDATA%\Code\User\globalStorage\ms-python.python" 2>nul
    echo ✅ Cache Python nettoyé
)

if exist "%APPDATA%\Code\User\globalStorage\ms-python.pylance" (
    rd /s /q "%APPDATA%\Code\User\globalStorage\ms-python.pylance" 2>nul
    echo ✅ Cache Pylance nettoyé
)

REM Supprimer les caches locaux
if exist "__pycache__" rd /s /q "__pycache__" 2>nul
if exist "*.pyc" del /q "*.pyc" 2>nul

echo.
echo Étape 2: Test de l'environnement Python...
echo.

set PYTHON_EXE=C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD\python311\python.exe

"%PYTHON_EXE%" -c "
import sys
print('✅ Python trouvé:', sys.executable)
print('✅ Version:', sys.version.split()[0])

# Test rapide des imports
imports = ['streamlit', 'torch', 'numpy', 'PIL', 'plotly', 'open3d']
for mod in imports:
    try:
        __import__(mod)
        print(f'✅ {mod}')
    except ImportError as e:
        print(f'❌ {mod}: {e}')
"

echo.
echo Étape 3: Création d'un fichier de test pour Pylance...
echo.

echo # Test file for Pylance > test_pylance.py
echo import streamlit as st >> test_pylance.py
echo import torch >> test_pylance.py
echo import numpy as np >> test_pylance.py
echo print("All imports work!") >> test_pylance.py

echo.
echo ========================================
echo 🎯 INSTRUCTIONS SUIVANTES:
echo ========================================
echo.
echo 1️⃣ Fermez COMPLETEMENT VS Code (Ctrl+Shift+W)
echo 2️⃣ Attendez 10 secondes minimum
echo 3️⃣ Redémarrez VS Code
echo 4️⃣ Ouvrez le workspace KIBALISTONEGOD
echo 5️⃣ Ouvrez Dust3r.py
echo 6️⃣ Si les erreurs persistent, cliquez sur l'interpréteur Python
echo    en bas à droite et sélectionnez l'environnement python311
echo.
echo 🔍 Vérifiez que l'interpréteur affiché est:
echo    'C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD\python311\python.exe'
echo.
echo ✅ Les erreurs Pylance devraient disparaître!
echo.

pause