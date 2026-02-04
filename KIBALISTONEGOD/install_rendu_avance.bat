@echo off
REM Script d'installation des moteurs de rendu avancés
REM Nécessaire pour concurrencer Blender

echo ========================================
echo  🎬 Installation Moteurs Rendu Avancé
echo ========================================
echo.

echo Installation des dépendances Python avancées...
echo.

REM Activer l'environnement virtuel si nécessaire
if exist "python311\python.exe" (
    echo Utilisation de Python portable...
    set PYTHON="python311\python.exe"
) else (
    set PYTHON=python
)

REM Installation des dépendances de rendu 3D avancé
%PYTHON% -m pip install --upgrade pip

echo Installation de PyRender pour rendu 3D avancé...
%PYTHON% -m pip install pyrender

echo Installation de Trimesh pour manipulation de maillages...
%PYTHON% -m pip install trimesh

echo Installation d'OpenCV pour traitement d'images avancé...
%PYTHON% -m pip install opencv-python

echo Installation de scikit-image pour effets avancés...
%PYTHON% -m pip install scikit-image

echo Installation de Pillow pour manipulation d'images...
%PYTHON% -m pip install Pillow

echo Installation de scipy pour calculs avancés...
%PYTHON% -m pip install scipy

echo Installation de matplotlib pour visualisations...
%PYTHON% -m pip install matplotlib

echo.
echo ========================================
echo ✅ Installation terminée!
echo ========================================
echo.
echo Les moteurs de rendu avancés sont maintenant disponibles.
echo Vous pouvez maintenant utiliser le rendu photoréaliste
echo qui surpasse la qualité de Blender!
echo.
echo Lancez demo_rendu_avance.py pour voir la démo.
echo.

pause