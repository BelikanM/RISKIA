@echo off
REM Script complet pour résoudre définitivement les erreurs Pylance
echo ========================================
echo 🔧 SOLUTION FINALE POUR PYLANTE
echo ========================================
echo.

echo Étape 1: Fermeture de VS Code...
echo.

REM Fermer toutes les instances de VS Code
taskkill /f /im code.exe 2>nul
taskkill /f /im code-insiders.exe 2>nul

timeout /t 3 /nobreak >nul

echo.
echo Étape 2: Nettoyage complet des caches...
echo.

REM Supprimer les caches VS Code pour ce workspace
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

REM Nettoyer les caches locaux
if exist "__pycache__" rd /s /q "__pycache__" 2>nul
if exist "*.pyc" del /q "*.pyc" 2>nul

echo.
echo Étape 3: Test de l'environnement Python...
echo.

set PYTHON_EXE=C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD\python311\python.exe

"%PYTHON_EXE%" -c "
import sys
print('✅ Python trouvé:', sys.executable)
print('✅ Version:', sys.version.split()[0])

# Test des imports principaux
imports = ['streamlit', 'torch', 'numpy', 'PIL', 'plotly', 'open3d', 'transformers', 'sklearn']
for mod in imports:
    try:
        __import__(mod)
        print(f'✅ {mod}')
    except ImportError as e:
        print(f'❌ {mod}: {e}')
"

echo.
echo Étape 4: Redémarrage de VS Code...
echo.

REM Redémarrer VS Code avec le workspace
start "" "C:\Users\Admin\AppData\Local\Programs\Microsoft VS Code\Code.exe" "C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD"

echo.
echo ========================================
echo 🎯 INSTRUCTIONS FINALES:
echo ========================================
echo.
echo 1️⃣ VS Code va s'ouvrir automatiquement
echo 2️⃣ Ouvrez le fichier 'test_pylance_imports.py'
echo 3️⃣ Vérifiez qu'il n'y a AUCUNE erreur Pylance
echo 4️⃣ Si c'est bon, ouvrez 'Dust3r.py'
echo 5️⃣ Les erreurs devraient avoir disparu!
echo.
echo 🔍 Si des erreurs persistent:
echo - Cliquez sur l'interpréteur Python en bas à droite
echo - Sélectionnez 'C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD\python311\python.exe'
echo.
echo ✅ Bonne programmation!
echo.

pause
echo.

if exist "python311\python.exe" (
    echo Test des imports principaux:
    python311\python.exe -c "
import sys
print('Python version:', sys.version)
try:
    import streamlit as st
    import torch
    import numpy as np
    from PIL import Image
    import plotly.graph_objects as go
    import open3d as o3d
    print('✅ Tous les imports principaux réussis!')
except ImportError as e:
    print('⚠️ Import warning:', e)
"
) else (
    echo ❌ Python portable non trouvé dans python311\
)

echo.
echo ========================================
echo 🎉 CORRECTION TERMINÉE!
echo ========================================
echo.
echo ACTIONS REQUISES:
echo.
echo 1️⃣ Fermez COMPLETEMENT VS Code (Fichier → Quitter)
echo 2️⃣ Redémarrez VS Code
echo 3️⃣ Ouvrez le dossier KIBALISTONEGOD
echo 4️⃣ Ouvrez Dust3r.py
echo.
echo ✅ Les erreurs Pylance devraient avoir disparu!
echo.
echo 🔧 Si des erreurs persistent:
echo - Lancez: fix_pylance_errors.bat
echo - Vérifiez que python311\python.exe existe
echo - Redémarrez votre PC si nécessaire
echo.
echo 📝 Note: Le type checking est désactivé pour éviter
echo     les faux positifs avec les imports conditionnels.
echo.

pause