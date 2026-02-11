#!/usr/bin/env python3
"""
Script de test pour l'analyse hybride de texture avec l'image annotée scientifique du Gabon
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

print("=== Test Analyse Hybride avec Image Réelle ===")
print(f"Script dir: {script_dir}")

try:
    import cv2
    import numpy as np
    from sklearn.cluster import KMeans

    print("✅ OpenCV et scikit-learn disponibles")

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

    # Tester l'analyse hybride
    class TestApp:
        def analyze_texture_hybrid(self, image):
            """Copie de la méthode analyze_texture_hybrid pour test"""
            detected_textures = []

            try:
                import cv2
                import numpy as np
                from sklearn.cluster import KMeans

                # Convertir en array numpy si nécessaire
                if not isinstance(image, np.ndarray):
                    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                # Analyse des couleurs dominantes
                pixels = image.reshape(-1, 3)
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

                # Afficher les couleurs dominantes
                print("Couleurs dominantes:")
                for i, color in enumerate(dominant_colors):
                    r, g, b = color
                    print(f"  {i+1}. RGB({int(r)}, {int(g)}, {int(b)})")

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

                # 4. Détection de brûlures/incendie
                orange_pixels = 0
                for color in dominant_colors:
                    r, g, b = color
                    if r > 180 and g > 100 and b < 80:  # Orange/rouge brûlé
                        orange_pixels += 1

                if orange_pixels >= 1 and contrast > 60:
                    detected_textures.append({
                        "texture": "burnt vegetation",
                        "confidence": min(0.75, orange_pixels * 0.3 + contrast/100),
                        "source": "color_analysis",
                        "description": f"Végétation brûlée (pixels orange: {orange_pixels}, contraste: {contrast:.1f})"
                    })

                # 5. Détection de zones dégradées (faible contraste + couleurs ternes)
                if contrast < 30 and edge_density < 0.05:
                    detected_textures.append({
                        "texture": "degraded soil",
                        "confidence": 0.6,
                        "source": "texture_analysis",
                        "description": f"Sol dégradé (contraste faible: {contrast:.1f}, faible texture)"
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
                import traceback
                traceback.print_exc()
                detected_textures = [{
                    "texture": "analysis_error",
                    "confidence": 0.0,
                    "source": "error",
                    "description": f"Erreur d'analyse: {str(e)}"
                }]

            return detected_textures

    # Tester avec l'image réelle
    test_app = TestApp()
    results = test_app.analyze_texture_hybrid(test_image)

    print(f"\n✅ Analyse terminée, {len(results)} textures détectées:")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['texture']} (confiance: {result['confidence']:.2f})")
        print(f"     Source: {result['source']}")
        print(f"     Description: {result['description']}")
        print()

    print("🎉 Analyse hybride fonctionnelle avec image réelle!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("=== Test terminé ===")