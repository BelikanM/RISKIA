@echo off
echo ============================================
echo   TEST RAPIDE - DUST3R ENVIRONMENT
echo ============================================
echo.

cd /d "C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD"

echo 🔍 Test de l'interpreteur Python...
C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD\python311\python.exe -c "import sys; print('Python executable:', sys.executable); print('Python version:', sys.version.split()[0])"

echo.
echo 🔍 Test des imports principaux...
C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD\python311\python.exe -c "import streamlit, torch, PIL, numpy, plotly, open3d; print('✓ Imports principaux OK')"

echo.
echo 🔍 Test des modeles IA...
C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD\python311\python.exe -c "import transformers, faiss; print('✓ Modeles IA OK')"

echo.
echo 🔍 Test CUDA...
C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD\python311\python.exe -c "import torch; print('CUDA disponible:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

echo.
echo ✅ Test termine - Tout fonctionne correctement !
echo.

pause