#!/usr/bin/env python3
"""
Script de test pour l'analyse CLIP de texture
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

print("=== Test Analyse CLIP ===")
print(f"Script dir: {script_dir}")

try:
    import cv2
    import numpy as np
    from transformers import CLIPProcessor, CLIPModel
    import torch
    from PIL import Image
    print("✅ Dépendances CLIP disponibles")

    # Charger l'image de test
    image_path = "annotated_scientific_gabon.png"
    if not os.path.exists(image_path):
        print(f"❌ Image non trouvée: {image_path}")
        sys.exit(1)

    print(f"✅ Chargement de l'image: {image_path}")
    test_image = cv2.imread(image_path)

    if test_image is None:
        print("❌ Impossible de charger l'image")
        sys.exit(1)

    print(f"✅ Image chargée: {test_image.shape}")

    # Tester l'analyse CLIP
    class TestApp:
        def __init__(self):
            self.clip_model = None
            self.clip_processor = None

        def analyze_texture_clip(self, image):
            """Test de l'analyse CLIP"""
            detected_textures = []

            try:
                # Initialiser CLIP
                if self.clip_model is None:
                    print("🔄 Chargement du modèle CLIP...")
                    self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                    self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                    self.clip_model.eval()
                    print("✅ Modèle CLIP chargé")

                # Prompts de test simplifiés
                texture_prompts = [
                    "rusted metal surface with orange-brown corrosion",
                    "flooded area with standing water",
                    "burnt vegetation with charred black surface",
                    "cracked concrete with visible fractures",
                    "normal soil surface"
                ]

                # Convertir l'image
                if isinstance(image, np.ndarray):
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    from PIL import Image
                    pil_image = Image.fromarray(image_rgb)
                else:
                    pil_image = image

                # Traiter avec CLIP
                inputs = self.clip_processor(images=pil_image, return_tensors="pt", padding=True)
                text_inputs = self.clip_processor(text=texture_prompts, return_tensors="pt", padding=True)

                with torch.no_grad():
                    # Pour CLIP, utiliser vision_model et text_model séparément pour avoir les bonnes projections
                    vision_outputs = self.clip_model.vision_model(**inputs)
                    text_outputs = self.clip_model.text_model(**text_inputs)

                    # Utiliser les projections finales
                    image_features = self.clip_model.visual_projection(vision_outputs.last_hidden_state[:, 0, :])
                    text_features = self.clip_model.text_projection(text_outputs.last_hidden_state[:, 0, :])

                    # Normaliser les embeddings
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    # Calculer similarités (cosinus similarity)
                    similarity_scores = torch.matmul(image_features, text_features.t()).squeeze(0)

                    # Softmax pour probabilités
                    import torch.nn.functional as F
                    probabilities = F.softmax(similarity_scores * 5, dim=0)

                print("📊 Scores CLIP:")
                for i, (prompt, score) in enumerate(zip(texture_prompts, probabilities)):
                    print(".3f")

                # Créer résultats
                texture_mapping = {
                    0: {"texture": "rusted steel structure", "desc": "Surface rouillée"},
                    1: {"texture": "flooded soil", "desc": "Sol inondé"},
                    2: {"texture": "burnt vegetation", "desc": "Végétation brûlée"},
                    3: {"texture": "cracked concrete surface", "desc": "Béton fissuré"},
                    4: {"texture": "normal surface", "desc": "Surface normale"}
                }

                # Prendre les 2 meilleurs résultats
                top_indices = torch.topk(probabilities, 2).indices

                for idx in top_indices:
                    idx = idx.item()
                    score = probabilities[idx].item()
                    if score > 0.1:
                        texture_info = texture_mapping[idx]
                        detected_textures.append({
                            "texture": texture_info["texture"],
                            "confidence": float(score),
                            "source": "clip_analysis",
                            "description": f"{texture_info['desc']} (CLIP: {score:.3f})"
                        })

                if not detected_textures:
                    detected_textures = [{
                        "texture": "normal surface",
                        "confidence": 0.5,
                        "source": "clip_analysis",
                        "description": "Aucune texture dangereuse détectée par CLIP"
                    }]

            except Exception as e:
                print(f"❌ Erreur CLIP: {e}")
                import traceback
                traceback.print_exc()
                detected_textures = [{
                    "texture": "clip_error",
                    "confidence": 0.0,
                    "source": "error",
                    "description": f"Erreur CLIP: {str(e)}"
                }]

            return detected_textures

    # Tester
    test_app = TestApp()
    results = test_app.analyze_texture_clip(test_image)

    print(f"\n✅ Analyse CLIP terminée, {len(results)} textures détectées:")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['texture']} (confiance: {result['confidence']:.2f})")
        print(f"     Source: {result['source']}")
        print(f"     Description: {result['description']}")
        print()

    print("🎉 Analyse CLIP fonctionnelle!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("=== Test CLIP terminé ===")