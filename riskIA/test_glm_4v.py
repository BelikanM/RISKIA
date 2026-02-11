#!/usr/bin/env python3
"""
Script de test pour GLM-4V-9B
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

print("=== Test GLM-4V-9B ===")
print(f"Script dir: {script_dir}")

try:
    import torch
    print(f"✅ PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
    print("✅ Transformers imports OK")

    # Tester le chargement de GLM-4V-9B
    glm_path = os.path.join(script_dir, "models", "glm-4v-9b")
    print(f"GLM path: {glm_path}")

    if os.path.exists(glm_path):
        print("✅ GLM directory exists")

        # Test rapide du tokenizer avec approche alternative
        try:
            tokenizer = AutoTokenizer.from_pretrained(glm_path, trust_remote_code=True, use_fast=False)
            print("✅ GLM tokenizer loaded (slow tokenizer)")
        except Exception as e:
            print(f"⚠️  Tokenizer loading failed: {e}")
            print("🔄 Tentative de chargement alternatif...")
            try:
                # Essayer de charger directement depuis les fichiers
                from transformers import ChatGLMTokenizer
                tokenizer = ChatGLMTokenizer.from_pretrained(glm_path, trust_remote_code=True)
                print("✅ GLM tokenizer loaded (ChatGLMTokenizer)")
            except Exception as e2:
                print(f"❌ Alternative tokenizer failed: {e2}")
                tokenizer = None

        # Test rapide du processor
        processor = AutoProcessor.from_pretrained(glm_path, trust_remote_code=True)
        print("✅ GLM processor loaded")

        print("🎉 GLM-4V-9B prêt pour l'analyse de textures!")

    else:
        print(f"❌ GLM directory not found: {glm_path}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("=== Test terminé ===")