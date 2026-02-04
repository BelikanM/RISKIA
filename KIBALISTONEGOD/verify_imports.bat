@echo off
REM Script de vérification des imports Python
echo 🔍 VÉRIFICATION DES IMPORTS PYTHON
echo ========================================
echo.

set PYTHON_EXE=C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD\python311\python.exe

if not exist "%PYTHON_EXE%" (
    echo ❌ Python 3.11 introuvable dans python311
    echo Vérifiez le chemin: %PYTHON_EXE%
    pause
    exit /b 1
)

echo ✅ Python trouvé: %PYTHON_EXE%
echo.

echo 📦 Test des imports principaux...
echo.

"%PYTHON_EXE%" -c "
import sys
print('Python version:', sys.version)
print('Python executable:', sys.executable)
print()

# Test des imports critiques
imports_to_test = [
    'streamlit',
    'torch',
    'PIL',
    'numpy',
    'plotly',
    'open3d',
    'pyrender',
    'trimesh',
    'cv2',
    'transformers',
    'lightly',
    'sklearn',
    'scipy',
    'matplotlib',
    'pandas'
]

success_count = 0
failed_imports = []

for module in imports_to_test:
    try:
        __import__(module)
        print(f'✅ {module}')
        success_count += 1
    except ImportError as e:
        print(f'❌ {module}: {e}')
        failed_imports.append(module)

print()
print(f'📊 RÉSULTATS: {success_count}/{len(imports_to_test)} modules importés avec succès')

if failed_imports:
    print(f'❌ Modules manquants: {', '.join(failed_imports)}')
    print('Lancez install_all_dependencies.bat pour les installer')
else:
    print('🎉 TOUS les modules sont disponibles!')
"

echo.
echo ========================================
echo 🔧 PROCHAINES ÉTAPES:
echo.

if errorlevel 1 (
    echo ❌ Certains imports ont échoué
    echo Lancez install_all_dependencies.bat
) else (
    echo ✅ Tous les imports fonctionnent!
    echo Vous pouvez maintenant utiliser Dust3r.py sans erreurs Pylance
)

echo.
pause