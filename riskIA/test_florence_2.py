#!/usr/bin/env python3
"""
Script de test pour Florence-2
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

os.environ['PYTHONPATH'] = f"{lib_dir};{site_packages_dir};{script_dir}"

print("=== Test Florence-2 ===")
print(f"Script dir: {script_dir}")

try:
    import torch
    print(f"✅ PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")

    from transformers import AutoModelForCausalLM, AutoProcessor
    print("✅ Transformers imports OK")

    # Tester le chargement de Florence-2
    florence_path = os.path.join(script_dir, "models", "models--microsoft--Florence-2-base-ft")
    print(f"Florence path: {florence_path}")

    if os.path.exists(florence_path):
        print("✅ Florence directory exists")

        # Test rapide du modèle Florence-2
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(florence_path, trust_remote_code=True).to(device)
        processor = AutoProcessor.from_pretrained(florence_path, trust_remote_code=True)

        print("✅ Florence-2 model loaded")
        print("✅ Florence-2 processor loaded")

        # Test avec une tâche simple
        from PIL import Image
        import numpy as np

        # Créer une image de test simple
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

        # Test de détection d'objets
        task_prompt = "<OD>"
        inputs = processor(text=task_prompt, images=test_image, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(224, 224))

        print("✅ Florence-2 object detection test passed")
        print(f"📊 Result: {parsed_answer}")

        print("🎉 Florence-2 prêt pour l'analyse de textures!")

    else:
        print(f"❌ Florence directory not found: {florence_path}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("=== Test terminé ===")