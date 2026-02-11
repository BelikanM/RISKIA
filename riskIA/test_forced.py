#!/usr/bin/env python3
"""
Script de test forcé pour l'analyse de textures avec environnement complètement isolé
"""
import sys
import os
import subprocess

# Chemin du script
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(script_dir, 'Lib')
site_packages_dir = os.path.join(lib_dir, 'site-packages')
python_exe = os.path.join(script_dir, 'python.exe')

print("=== Test forcé de l'environnement portable ===")
print(f"Script dir: {script_dir}")
print(f"Python exe: {python_exe}")
print(f"Lib dir: {lib_dir}")
print(f"Site packages: {site_packages_dir}")
print()

# Test avec subprocess pour isoler complètement
env = os.environ.copy()
env['PYTHONPATH'] = f"{lib_dir};{site_packages_dir};{script_dir}"
env['PYTHONHOME'] = script_dir
env['PYTHONNOUSERSITE'] = '1'
env['HF_HOME'] = os.path.join(script_dir, 'models')
env['TRANSFORMERS_CACHE'] = os.path.join(script_dir, 'models')

test_code = '''
import sys
print("=== Environnement de test ===")
print(f"Python executable: {sys.executable}")
print(f"sys.path[0]: {sys.path[0] if sys.path else None}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
except Exception as e:
    print(f"PyTorch error: {e}")

try:
    from transformers import CLIPProcessor, CLIPModel
    print("CLIP imports: OK")
except Exception as e:
    print(f"CLIP error: {e}")

# Test du modèle CLIP
try:
    import os
    model_path = os.path.join(os.path.dirname(__file__), "models", "hub", "models--openai--clip-vit-base-patch32")
    print(f"Model path: {model_path}")
    print(f"Model exists: {os.path.exists(model_path)}")

    if os.path.exists(model_path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP on {device}...")

        clip_model = CLIPModel.from_pretrained(model_path).to(device)
        clip_processor = CLIPProcessor.from_pretrained(model_path)
        print("CLIP loaded successfully!")

        # Test rapide
        import numpy as np
        from PIL import Image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(dummy_image.astype("uint8"), "RGB")

        inputs = clip_processor(images=pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
        print("CLIP test successful!")

    else:
        print("Model path does not exist!")

except Exception as e:
    print(f"CLIP test error: {e}")
    import traceback
    traceback.print_exc()
'''

try:
    result = subprocess.run([python_exe, '-c', test_code],
                          cwd=script_dir,
                          env=env,
                          capture_output=True,
                          text=True,
                          timeout=60)

    print("=== STDOUT ===")
    print(result.stdout)
    if result.stderr:
        print("=== STDERR ===")
        print(result.stderr)
    print(f"=== Return code: {result.returncode} ===")

except Exception as e:
    print(f"Subprocess error: {e}")