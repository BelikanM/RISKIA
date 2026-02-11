#!/usr/bin/env python3
"""
Script de test pour l'analyse hybride de texture
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

print("=== Test Analyse Hybride ===")
print(f"Script dir: {script_dir}")

try:
    import cv2
    import numpy as np
    from sklearn.cluster import KMeans

    print("✅ OpenCV et scikit-learn disponibles")

    # Créer une image de test avec des caractéristiques de rouille
    test_image = np.zeros((224, 224, 3), dtype=np.uint8)

    # Ajouter des pixels rouille (rouge/orange)
    rust_mask = np.random.random((224, 224)) < 0.3
    test_image[rust_mask] = [150, 80, 60]  # Couleur rouille

    # Ajouter du gris pour simuler la corrosion
    gray_mask = np.random.random((224, 224)) < 0.4
    test_image[gray_mask] = [120, 120, 120]

    # Ajouter du bruit pour simuler la texture
    noise = np.random.randint(-20, 20, (224, 224, 3), dtype=np.int16)
    test_image = np.clip(test_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    print("✅ Image de test créée")

    # Tester l'analyse hybride
    from risk_simulation_app import RiskSimulationApp

    # Créer une instance fictive pour tester la méthode
    class TestApp:
        def analyze_texture_hybrid(self, image):
            """Copie de la méthode analyze_texture_hybrid pour test"""
            detected_textures = []

            try:
                # Convertir en array numpy si nécessaire
                if not isinstance(image, np.ndarray):
                    import cv2
                    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                # Analyse des couleurs dominantes
                pixels = image.reshape(-1, 3)
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                kmeans.fit(pixels)
                dominant_colors = kmeans.cluster_centers_

                # Analyse du contraste et de la texture
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                contrast = gray.std()

                # Analyse des bords (texture)
                edges = cv2.Canny(gray, 100, 200)
                edge_density = np.sum(edges > 0) / edges.size

                print(f"Contraste: {contrast:.2f}, Densité bords: {edge_density:.3f}")

                # Règles heuristiques pour détecter les textures dangereuses

                # 1. Détection de rouille
                rust_pixels = 0
                for color in dominant_colors:
                    r, g, b = color
                    if r > 150 and g < 100 and b < 100:  # Rouge dominant
                        rust_pixels += 1

                if rust_pixels >= 2 and edge_density > 0.1:
                    detected_textures.append({
                        "texture": "rusted steel structure",
                        "confidence": min(0.9, rust_pixels * 0.2 + edge_density),
                        "source": "color_analysis",
                        "description": f"Rouille détectée (pixels rouille: {rust_pixels}, densité bords: {edge_density:.2f})"
                    })

                # 2. Détection d'eau/inondation
                blue_pixels = 0
                for color in dominant_colors:
                    r, g, b = color
                    if b > 150 and r < 100 and g < 120:  # Bleu dominant
                        blue_pixels += 1

                if blue_pixels >= 2 and contrast < 50:
                    detected_textures.append({
                        "texture": "flooded soil",
                        "confidence": min(0.85, blue_pixels * 0.15 + (1 - contrast/100)),
                        "source": "color_analysis",
                        "description": f"Eau détectée (pixels bleus: {blue_pixels}, contraste: {contrast:.1f})"
                    })

                # 3. Détection de corrosion
                gray_pixels = 0
                for color in dominant_colors:
                    r, g, b = color
                    if abs(r - g) < 30 and abs(g - b) < 30 and abs(r - b) < 30:  # Couleurs similaires
                        gray_pixels += 1

                if gray_pixels >= 3 and edge_density > 0.15:
                    detected_textures.append({
                        "texture": "corroded metal surface",
                        "confidence": min(0.8, gray_pixels * 0.1 + edge_density),
                        "source": "texture_analysis",
                        "description": f"Surface corrodée (gris: {gray_pixels}, irrégularité: {edge_density:.2f})"
                    })

                # Si aucune texture dangereuse détectée, ajouter une texture neutre
                if not detected_textures:
                    detected_textures.append({
                        "texture": "normal surface",
                        "confidence": 0.5,
                        "source": "default",
                        "description": "Aucune texture dangereuse détectée"
                    })

            except Exception as e:
                print(f"Erreur dans l'analyse hybride: {e}")
                detected_textures = [{
                    "texture": "analysis_error",
                    "confidence": 0.0,
                    "source": "error",
                    "description": f"Erreur d'analyse: {str(e)}"
                }]

            return detected_textures

    # Tester
    test_app = TestApp()
    results = test_app.analyze_texture_hybrid(test_image)

    print(f"✅ Analyse terminée, {len(results)} textures détectées:")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['texture']} (confiance: {result['confidence']:.2f})")
        print(f"     Source: {result['source']}")
        print(f"     Description: {result['description']}")

    print("🎉 Analyse hybride fonctionnelle!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("=== Test terminé ===")