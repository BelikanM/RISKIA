#!/usr/bin/env python3
"""
Script de test pour vérifier que l'environnement portable fonctionne correctement
"""
import sys
import os

print("=== Test de l'environnement portable riskIA ===")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
print()

# Test des imports critiques
tests = [
    ("torch", "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"),
    ("transformers", "from transformers import CLIPProcessor, CLIPModel; print('CLIP OK')"),
    ("cv2", "import cv2; print('OpenCV OK')"),
    ("PIL", "from PIL import Image; print('PIL OK')"),
    ("numpy", "import numpy as np; print('NumPy OK')"),
    ("matplotlib", "import matplotlib; print('Matplotlib OK')"),
    ("danger_study", "from danger_study import DangerStudy; print('DangerStudy OK')"),
]

for module_name, test_code in tests:
    try:
        exec(test_code)
        print(f"✅ {module_name}: OK")
    except Exception as e:
        print(f"❌ {module_name}: ERREUR - {e}")
    print()

print("=== Test terminé ===")