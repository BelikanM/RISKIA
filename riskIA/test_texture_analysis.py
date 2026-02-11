#!/usr/bin/env python3
"""
Script de test pour l'analyse de textures
"""
import sys
import os

# Forcer l'utilisation de l'environnement portable
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(script_dir, 'Lib')
site_packages_dir = os.path.join(lib_dir, 'site-packages')

sys.path.insert(0, lib_dir)
sys.path.insert(0, site_packages_dir)
sys.path.insert(0, script_dir)

os.environ['PYTHONPATH'] = f"{lib_dir};{site_packages_dir};{script_dir};{os.environ.get('PYTHONPATH', '')}"

# Set local cache for models
models_dir = os.path.join(script_dir, 'models')
os.environ['HF_HOME'] = models_dir
os.environ['TRANSFORMERS_CACHE'] = models_dir

print("=== Test de l'analyse de textures ===")
print(f"Python executable: {sys.executable}")
print(f"Models directory: {models_dir}")
print(f"CLIP model path: {os.path.join(models_dir, 'hub', 'models--openai--clip-vit-base-patch32')}")
print()

try:
    import torch
    print(f"✅ PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")

    from transformers import CLIPProcessor, CLIPModel
    print("✅ CLIP imports OK")

    from PIL import Image
    import numpy as np
    print("✅ PIL and NumPy OK")

    # Test du chargement du modèle CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_path = os.path.join(script_dir, "models", "hub", "models--openai--clip-vit-base-patch32")
    print(f"Loading CLIP from: {model_path}")

    if os.path.exists(model_path):
        print("✅ CLIP model directory exists")

        clip_model = CLIPModel.from_pretrained(model_path).to(device)
        clip_processor = CLIPProcessor.from_pretrained(model_path)
        print("✅ CLIP model loaded successfully")

        # Test avec une image factice
        print("Testing with dummy image...")
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(dummy_image.astype('uint8'), 'RGB')

        inputs = clip_processor(images=pil_image, return_tensors="pt").to(device)
        print("✅ Image processing OK")

        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
            print("✅ Image features extracted")

        # Test avec des labels
        texture_labels = ["corroded metal surface", "rusted steel structure"]
        text_inputs = clip_processor(text=texture_labels, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            text_features = clip_model.get_text_features(**text_inputs)
            print("✅ Text features extracted")

        # Calcul de similarité
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        probs = similarity[0].cpu().numpy()
        results = [(texture_labels[i], float(probs[i])) for i in range(len(texture_labels))]
        results.sort(key=lambda x: x[1], reverse=True)

        print("✅ Texture analysis test successful!")
        print(f"Top result: {results[0][0]} ({results[0][1]:.3f})")

    else:
        print(f"❌ CLIP model directory not found: {model_path}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("=== Test terminé ===")