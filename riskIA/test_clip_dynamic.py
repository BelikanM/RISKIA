#!/usr/bin/env python3
"""
Test dynamique d'analyse CLIP sur une image utilisateur
- Analyse naturelle et adaptative
- Fonctionne sur tout type d'image
- Classification granulaire avec Top 10
"""

import sys
import os

# Configuration de l'environnement (comme les autres scripts)
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(script_dir, 'Lib')
site_packages_dir = os.path.join(lib_dir, 'site-packages')

sys.path.insert(0, lib_dir)
sys.path.insert(0, site_packages_dir)
sys.path.insert(0, script_dir)

os.environ['PYTHONPATH'] = f"{lib_dir};{site_packages_dir};{script_dir}"

print("🔄 Configuration environnement CLIP...")

try:
    import torch
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    from transformers import CLIPProcessor, CLIPModel
    import warnings
    warnings.filterwarnings('ignore')
    print("✅ Dépendances CLIP chargées")
except Exception as e:
    print(f"❌ Erreur dépendances: {e}")
    sys.exit(1)

class DynamicCLIPAnalyzer:
    """Analyseur CLIP dynamique pour tout type d'image"""

    def __init__(self):
        print("🔄 Chargement du modèle CLIP...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Textures de risque adaptatives (50+ types sans répétition)
        self.risk_textures = [
            # Corrosion métallique
            "rusted pitted metal surface with orange-brown oxidation",
            "flaking corroded steel layers peeling off",
            "oxidized metal spots with rust formation",
            "degraded rusted pipeline with holes and decay",
            "galvanic corrosion patterns with different metal reactions",
            "acid-etched corrosion with chemically engraved surfaces",
            "atmospheric rust formation on exposed metal",
            "localized crevice corrosion in hidden areas",

            # Dommages hydriques
            "standing water surface with reflective puddles",
            "waterlogged saturated soil with muddy consistency",
            "flooded areas with water accumulation",
            "moist soil with water retention",

            # Dommages structurels
            "cracked concrete with visible fissures",
            "deteriorated building materials",
            "structural damage with material breakdown",
            "erosion patterns on surfaces",

            # Contamination
            "oil contaminated soil with dark staining",
            "chemical spills with discoloration",
            "toxic waste residues on ground",
            "industrial pollution marks",

            # Végétation et environnement
            "dead vegetation with wilting plants",
            "soil erosion with exposed roots",
            "deforested areas with bare soil",
            "overgrown vegetation blocking access",

            # Infrastructures
            "damaged electrical equipment",
            "corroded metal structures",
            "deteriorated wooden elements",
            "exposed rebar in concrete",

            # Conditions météorologiques
            "wind-damaged structures",
            "hail impact marks on surfaces",
            "lightning strike evidence",
            "frost damage patterns",

            # Risques géologiques
            "landslide scars on terrain",
            "earthquake cracks in ground",
            "sinkhole formations",
            "soil subsidence areas",

            # Risques biologiques
            "mold growth on surfaces",
            "fungus infected materials",
            "biological contamination signs",
            "pest damage evidence",

            # Conditions atmosphériques
            "air pollution residue",
            "acid rain damage",
            "ozone layer effects",
            "UV degradation marks"
        ]

        print(f"✅ Modèle CLIP chargé sur {self.device}")
        print(f"📚 {len(self.risk_textures)} textures de risque configurées")

    def analyze_image(self, image_path):
        """Analyse dynamique d'une image avec CLIP"""
        print(f"\n🔍 Analyse de l'image: {image_path}")

        # Charger et prétraiter l'image
        try:
            image = Image.open(image_path).convert('RGB')
            print(f"📏 Dimensions: {image.size}")
        except Exception as e:
            print(f"❌ Erreur chargement image: {e}")
            return None

        # Prétraiter l'image et les textes
        inputs = self.processor(
            text=self.risk_textures,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Calculer les similarités
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        # Obtenir les résultats
        probabilities = probs[0].cpu().numpy()

        # Créer les résultats détaillés
        results = []
        for i, (texture, prob) in enumerate(zip(self.risk_textures, probabilities)):
            results.append({
                'rank': i + 1,
                'texture': texture.replace('_', ' ').title(),
                'score': float(prob),
                'description': self._get_texture_description(texture)
            })

        # Trier par score décroissant
        results.sort(key=lambda x: x['score'], reverse=True)

        # Réassigner les rangs
        for i, result in enumerate(results):
            result['rank'] = i + 1

        return results[:10], image  # Top 10 seulement

    def _get_texture_description(self, texture):
        """Génère une description naturelle pour chaque texture"""
        descriptions = {
            "rusted pitted metal surface with orange-brown oxidation": "Surface métallique rouillée avec oxydation orange-brun piquetée",
            "flaking corroded steel layers peeling off": "Acier corrodé avec couches qui s'effritent et se détachent",
            "oxidized metal spots with rust formation": "Métal oxydé avec formation de taches de rouille",
            "degraded rusted pipeline with holes and decay": "Pipeline rouillé dégradé avec trous et signes de décomposition",
            "galvanic corrosion patterns with different metal reactions": "Motifs de corrosion galvanique avec réactions métalliques différentes",
            "acid-etched corrosion with chemically engraved surfaces": "Corrosion chimique avec surfaces gravées par acide",
            "atmospheric rust formation on exposed metal": "Formation de rouille atmosphérique sur métal exposé",
            "localized crevice corrosion in hidden areas": "Corrosion de fissure localisée dans les zones cachées",
            "standing water surface with reflective puddles": "Surface d'eau stagnante avec flaques réfléchissantes",
            "waterlogged saturated soil with muddy consistency": "Sol saturé d'eau avec consistance boueuse"
        }

        # Description par défaut si non trouvée
        if texture in descriptions:
            return descriptions[texture]
        else:
            return f"Texture de risque: {texture.replace('_', ' ')}"

    def display_results(self, results, image):
        """Affiche les résultats de manière visuelle"""
        print("\n" + "="*80)
        print("🎯 ANALYSE CLIP DYNAMIQUE - TOP 10 TEXTURES DÉTECTÉES")
        print("="*80)

        for result in results:
            print(f"{result['rank']}. {result['texture']}")
            print(f"   🎯 Score: {result['score']:.4f}")
            print(f"   📝 {result['description']}")
            print()

        # Afficher l'image avec les résultats
        plt.figure(figsize=(15, 10))

        # Image originale
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Image Analysée', fontsize=14, fontweight='bold')
        plt.axis('off')

        # Graphique des résultats
        plt.subplot(1, 2, 2)
        textures = [r['texture'][:30] + "..." if len(r['texture']) > 30 else r['texture'] for r in results]
        scores = [r['score'] for r in results]

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
                 '#DDA0DD', '#98FB98', '#F0E68C', '#FFA07A', '#87CEFA']

        bars = plt.barh(textures, scores, color=colors, alpha=0.8)
        plt.title('Top 10 Textures de Risque Détectées', fontsize=14, fontweight='bold')
        plt.xlabel('Score de Similarité', fontsize=12)

        # Ajouter les valeurs
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                    '.3f', ha='left', va='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.show()

def main():
    """Fonction principale de test"""
    analyzer = DynamicCLIPAnalyzer()

    # Tester avec l'image cap.png
    image_path = r"C:\Users\Admin\Desktop\logiciel\riskIA\cap.png"

    if not os.path.exists(image_path):
        print(f"❌ Image non trouvée: {image_path}")
        return

    # Analyser l'image
    results, image = analyzer.analyze_image(image_path)

    if results:
        analyzer.display_results(results, image)
        print("✅ Analyse CLIP dynamique terminée avec succès!")
    else:
        print("❌ Échec de l'analyse")

if __name__ == "__main__":
    main()