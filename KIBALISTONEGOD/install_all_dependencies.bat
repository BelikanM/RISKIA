@echo off
REM Script complet pour installer TOUTES les dépendances dans python311
echo ========================================
echo  🚀 INSTALLATION COMPLÈTE DES DÉPENDANCES
echo ========================================
echo.

set PYTHON_DIR=C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD\python311
set PYTHON_EXE=%PYTHON_DIR%\python.exe
set PIP_EXE=%PYTHON_DIR%\Scripts\pip.exe

echo Vérification de l'environnement Python...
echo.

if not exist "%PYTHON_EXE%" (
    echo ❌ ERREUR: Python portable non trouvé dans %PYTHON_DIR%
    echo Téléchargez et installez Python 3.11 dans ce dossier.
    pause
    exit /b 1
)

echo ✅ Python trouvé: %PYTHON_EXE%
echo.

"%PYTHON_EXE%" -c "import sys; print('Version Python:', sys.version); print('Executable:', sys.executable)"

echo.
echo ========================================
echo 📦 MISE À JOUR DE PIP
echo ========================================
echo.

"%PIP_EXE%" install --upgrade pip

echo.
echo ========================================
echo 🔧 INSTALLATION DES DÉPENDANCES DE BASE
echo ========================================
echo.

REM Dépendances essentielles
"%PIP_EXE%" install --no-cache-dir numpy
"%PIP_EXE%" install --no-cache-dir pillow
"%PIP_EXE%" install --no-cache-dir matplotlib
"%PIP_EXE%" install --no-cache-dir pandas
"%PIP_EXE%" install --no-cache-dir scikit-learn
"%PIP_EXE%" install --no-cache-dir psutil
"%PIP_EXE%" install --no-cache-dir pynvml
"%PIP_EXE%" install --no-cache-dir faiss-cpu

echo.
echo ========================================
echo 🎨 INSTALLATION DES LIBRAIRIES GRAPHIQUES
echo ========================================
echo.

REM Streamlit et Plotly
"%PIP_EXE%" install --no-cache-dir streamlit
"%PIP_EXE%" install --no-cache-dir plotly

REM Open3D
"%PIP_EXE%" install --no-cache-dir open3d

echo.
echo ========================================
echo 🤖 INSTALLATION DES LIBRAIRIES ML/AI
echo ========================================
echo.

REM PyTorch NIGHTLY avec CUDA 12.3 (cu130)
"%PIP_EXE%" install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130 --pre

REM Lightly pour apprentissage auto-supervisé
"%PIP_EXE%" install --no-cache-dir lightly

REM Transformers et autres ML
"%PIP_EXE%" install --no-cache-dir transformers
"%PIP_EXE%" install --no-cache-dir accelerate

echo.
echo ========================================
echo 🎬 INSTALLATION DES LIBRAIRIES DE RENDU AVANCÉ
echo ========================================
echo.

REM PyRender et Trimesh pour rendu 3D avancé
"%PIP_EXE%" install --no-cache-dir pyrender
"%PIP_EXE%" install --no-cache-dir trimesh

REM OpenCV pour traitement d'images
"%PIP_EXE%" install --no-cache-dir opencv-python

REM Scikit-image pour effets avancés
"%PIP_EXE%" install --no-cache-dir scikit-image

REM SciPy pour calculs avancés
"%PIP_EXE%" install --no-cache-dir scipy

echo.
echo ========================================
echo 🔍 VÉRIFICATION DES INSTALLATIONS
echo ========================================
echo.

echo Test des imports principaux...
echo.

"%PYTHON_EXE%" -c "
import sys
print('=== VÉRIFICATION DES IMPORTS ===')
print('Python executable:', sys.executable)
print()

imports_to_test = [
    ('streamlit', 'st'),
    ('torch', 'torch'),
    ('PIL', 'PIL'),
    ('numpy', 'np'),
    ('plotly.graph_objects', 'go'),
    ('plotly.express', 'px'),
    ('plotly.subplots', 'make_subplots'),
    ('open3d', 'o3d'),
    ('pandas', 'pd'),
    ('sklearn.cluster', 'KMeans'),
    ('sklearn.neighbors', 'NearestNeighbors'),
    ('transformers', 'transformers'),
    ('psutil', 'psutil'),
    ('pynvml', 'pynvml'),
    ('faiss', 'faiss'),
    ('pyrender', 'pyrender'),
    ('trimesh', 'trimesh'),
    ('cv2', 'cv2'),
    ('scipy', 'scipy'),
    ('skimage', 'skimage')
]

success_count = 0
for module_name, import_name in imports_to_test:
    try:
        if '.' in module_name:
            # Import avec sous-module
            exec(f'import {module_name}')
        else:
            # Import simple
            exec(f'import {module_name}')
        print(f'✅ {module_name}')
        success_count += 1
    except ImportError as e:
        print(f'❌ {module_name}: {e}')

print()
print(f'RÉSULTAT: {success_count}/{len(imports_to_test)} modules importés avec succès')
print()

if success_count == len(imports_to_test):
    print('🎉 TOUTES LES DÉPENDANCES SONT CORRECTEMENT INSTALLÉES!')
else:
    print('⚠️ Certaines dépendances sont manquantes.')
"

echo.
echo ========================================
echo 📋 CRÉATION DU RAPPORT D'INSTALLATION
echo ========================================
echo.

echo Création du rapport d'installation...
echo.

echo INSTALLATION TERMINÉE > installation_report.txt
echo Date: %DATE% %TIME% >> installation_report.txt
echo. >> installation_report.txt
echo Python executable: %PYTHON_EXE% >> installation_report.txt
echo. >> installation_report.txt

"%PYTHON_EXE%" -c "
import sys
print('Version Python:', sys.version)
print('Chemin d exécution:', sys.executable)
print()
print('Chemins d import (5 premiers):')
for i, path in enumerate(sys.path[:5]):
    print(f'  {i+1}. {path}')
" >> installation_report.txt

echo. >> installation_report.txt
echo === LISTE DES PACKAGES INSTALLÉS === >> installation_report.txt
"%PIP_EXE%" list >> installation_report.txt

echo.
echo ✅ Rapport d'installation créé: installation_report.txt
echo.

echo ========================================
echo 🎯 PROCHAINES ÉTAPES
echo ========================================
echo.
echo 1. Fermez complètement VS Code
echo 2. Redémarrez VS Code
echo 3. Ouvrez le workspace KIBALISTONEGOD
echo 4. Les erreurs Pylance devraient avoir disparu
echo.
echo Si des erreurs persistent:
echo - Vérifiez que python311\python.exe existe
echo - Lancez: final_pylance_fix.bat
echo - Redémarrez votre PC si nécessaire
echo.

pause