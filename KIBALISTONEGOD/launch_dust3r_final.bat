@echo off
echo ============================================
echo   LANCEMENT DUST3R - PHOTOGRAMMETRIE IA
echo ============================================
echo.

cd /d "C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD"

echo Verification des dependances...
C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD\python311\python.exe -c "import streamlit, torch, PIL, numpy, plotly, open3d, transformers, pynvml, faiss, pandas, sklearn, psutil; print('✓ Toutes les dependances sont presentes')"

echo.
echo Lancement de l'application Streamlit...
echo L'application sera accessible sur http://localhost:8501
echo.

C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD\python311\python.exe -m streamlit run Dust3r.py --server.port 8501 --server.address 0.0.0.0

pause