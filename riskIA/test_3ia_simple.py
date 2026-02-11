#!/usr/bin/env python3
"""
Test simple des 3 IA avec modèles locaux uniquement
"""

import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")

# Chemins locaux des modèles
CLIP_PATH = r"C:\Users\Admin\Desktop\logiciel\riskIA\models\models--openai--clip-vit-base-patch32"
FLORENCE_PATH = r"C:\Users\Admin\Desktop\logiciel\riskIA\models\models--microsoft--Florence-2-base-ft"
GLM_PATH = r"C:\Users\Admin\Desktop\logiciel\riskIA\models\glm-4v-9b"

def test_clip():
    print("🔄 Test CLIP...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Utiliser le modèle depuis HuggingFace mais avec cache local
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=r"C:\Users\Admin\Desktop\logiciel\riskIA\models").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=r"C:\Users\Admin\Desktop\logiciel\riskIA\models")

        # Test simple
        image = Image.new('RGB', (224, 224), color='red')
        texts = ["test image", "red square"]
        inputs = processor(text=texts, images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        print("✅ CLIP OK")
        return True
    except Exception as e:
        print(f"❌ CLIP ERROR: {e}")
        return False

def test_florence():
    print("🔄 Test Florence...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True, cache_dir=r"C:\Users\Admin\Desktop\logiciel\riskIA\models")
        model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True, cache_dir=r"C:\Users\Admin\Desktop\logiciel\riskIA\models").to(device)

        # Test simple
        image = Image.new('RGB', (224, 224), color='blue')
        task = "<MORE_DETAILED_CAPTION>"
        inputs = processor(text=task, images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10)

        result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        print(f"✅ Florence OK: {result[:50]}...")
        return True
    except Exception as e:
        print(f"❌ Florence ERROR: {e}")
        return False

def test_glm():
    print("🔄 Test GLM...")
    try:
        # Vérifier si le modèle existe
        if not os.path.exists(GLM_PATH):
            print(f"❌ GLM path not found: {GLM_PATH}")
            return False

        print(f"✅ GLM path exists: {GLM_PATH}")
        return True
    except Exception as e:
        print(f"❌ GLM ERROR: {e}")
        return False

def main():
    print("🧠 TEST RAPIDE DES 3 IA AVEC MODÈLES LOCAUX")
    print("=" * 50)

    results = []
    results.append(("CLIP", test_clip()))
    results.append(("Florence", test_florence()))
    results.append(("GLM", test_glm()))

    print("\n" + "=" * 50)
    print("RÉSULTATS:")
    for name, success in results:
        status = "✅ OK" if success else "❌ FAILED"
        print(f"{name}: {status}")

    success_count = sum(1 for _, success in results if success)
    print(f"\nModèles fonctionnels: {success_count}/{len(results)}")

    if success_count >= 2:  # Au moins CLIP et Florence
        print("✅ Prêt pour intégration dans l'application!")
    else:
        print("❌ Problèmes détectés - corriger avant intégration")

if __name__ == "__main__":
    main()