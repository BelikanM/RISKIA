#!/usr/bin/env python3
"""
Test final de l'analyse de textures avec toutes les corrections
"""
import sys
import os
import subprocess

# Configuration de l'environnement portable
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(script_dir, 'Lib')
site_packages_dir = os.path.join(lib_dir, 'site-packages')
python_exe = os.path.join(script_dir, 'python.exe')

print("=== Test final de l'analyse de textures ===")
print(f"Environnement portable: {script_dir}")
print()

# Test avec subprocess isolé
env = os.environ.copy()
env['PYTHONPATH'] = f"{lib_dir};{site_packages_dir};{script_dir}"
env['PYTHONHOME'] = script_dir
env['PYTHONNOUSERSITE'] = '1'
env['HF_HOME'] = os.path.join(script_dir, 'models')
env['TRANSFORMERS_CACHE'] = os.path.join(script_dir, 'models')

test_code = '''
import sys
import os

# Simuler les variables définies dans l'application
script_dir = r"''' + script_dir.replace('\\', '\\\\') + '''"
models_dir = os.path.join(script_dir, 'models')
os.environ['HF_HOME'] = models_dir
os.environ['TRANSFORMERS_CACHE'] = models_dir

print("=== Simulation de run_texture_analysis ===")

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    import numpy as np

    print("✅ Imports réussis")

    # Simuler le chargement des modèles comme dans l'application
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(script_dir, "models", "hub", "models--openai--clip-vit-base-patch32")
    kibali_path = os.path.join(script_dir, "models", "kibali-final-merged")

    print(f"Device: {device}")
    print(f"CLIP path: {model_path}")
    print(f"Kibali path: {kibali_path}")
    print(f"CLIP exists: {os.path.exists(model_path)}")
    print(f"Kibali exists: {os.path.exists(kibali_path)}")

    # Charger CLIP
    clip_model = CLIPModel.from_pretrained(model_path).to(device)
    clip_processor = CLIPProcessor.from_pretrained(model_path)
    print("✅ CLIP chargé")

    # Simuler une image comme dans l'application
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    pil_image = Image.fromarray(dummy_image.astype('uint8'), 'RGB')
    print("✅ Image PIL créée")

    # Traiter l'image (comme corrigé dans l'application)
    inputs = clip_processor(images=pil_image, return_tensors="pt").to(device)
    print("✅ Image traitée par CLIP")

    # Labels de test
    texture_labels = ["corroded metal surface", "rusted steel structure", "burnt vegetation"]

    # Encoder les labels
    text_inputs = clip_processor(text=texture_labels, return_tensors="pt", padding=True).to(device)
    print("✅ Labels encodés")

    # Calculer les similarités
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
        text_features = clip_model.get_text_features(**text_inputs)

        # Normaliser
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Calculer les similarités
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # Résultats
    probs = similarity[0].cpu().numpy()
    detected_textures = [(texture_labels[i], float(probs[i])) for i in range(len(texture_labels))]
    detected_textures.sort(key=lambda x: x[1], reverse=True)

    print("✅ Analyse de textures réussie!")
    print(f"Résultats: {detected_textures[:3]}")

    # Test Kibali si disponible
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        kibali_model = AutoModelForCausalLM.from_pretrained(kibali_path)
        kibali_tokenizer = AutoTokenizer.from_pretrained(kibali_path)
        print("✅ Kibali disponible")

        # Test rapide de génération
        prompt = "Test Kibali"
        inputs = kibali_tokenizer(prompt, return_tensors="pt", max_length=50, truncation=True)
        with torch.no_grad():
            outputs = kibali_model.generate(**inputs, max_new_tokens=10, do_sample=False)
        response = kibali_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Kibali génération: {response[:50]}...")

    except Exception as e:
        print(f"⚠️ Kibali non disponible: {e}")

except Exception as e:
    print(f"❌ Erreur: {e}")
    import traceback
    traceback.print_exc()
'''

try:
    result = subprocess.run([python_exe, '-c', test_code],
                          cwd=script_dir,
                          env=env,
                          capture_output=True,
                          text=True,
                          timeout=120)

    print("=== RÉSULTATS DU TEST ===")
    print(result.stdout)
    if result.stderr:
        print("=== ERREURS ===")
        print(result.stderr)
    print(f"=== CODE RETOUR: {result.returncode} ===")

    if result.returncode == 0 and "Analyse de textures réussie!" in result.stdout:
        print("\n🎉 SUCCÈS ! L'analyse de textures fonctionne maintenant !")
    else:
        print("\n❌ ÉCHEC ! Il y a encore des problèmes.")

except Exception as e:
    print(f"Erreur subprocess: {e}")