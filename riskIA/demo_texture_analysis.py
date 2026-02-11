#!/usr/bin/env python3
"""
Script de démonstration de l'analyse de texture hybride
Permet de tester l'analyse sur plusieurs images
"""
import sys
import os
import glob

# Forcer l'utilisation de l'environnement portable
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(script_dir, 'Lib')
site_packages_dir = os.path.join(lib_dir, 'site-packages')

sys.path.insert(0, lib_dir)
sys.path.insert(0, site_packages_dir)
sys.path.insert(0, script_dir)

os.environ['PYTHONPATH'] = f"{lib_dir};{site_packages_dir};{script_dir}"

print("=== Démonstration Analyse Hybride de Texture ===")

try:
    import cv2
    import numpy as np
    from sklearn.cluster import KMeans

    print("✅ Dépendances chargées")

    class TextureAnalyzer:
        def analyze_texture_hybrid(self, image):
            """Méthode d'analyse hybride de texture"""
            detected_textures = []

            try:
                import cv2
                import numpy as np
                from sklearn.cluster import KMeans

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

                print(f"📊 Analyse: Contraste={contrast:.1f}, Bords={edge_density:.3f}")

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
                print(f"❌ Erreur dans l'analyse: {e}")
                detected_textures = [{
                    "texture": "analysis_error",
                    "confidence": 0.0,
                    "source": "error",
                    "description": f"Erreur d'analyse: {str(e)}"
                }]

            return detected_textures

    # Liste des images à tester
    test_images = [
        "annotated_scientific_gabon.png",
        "demo_100_points.dxf",  # Peut-être une image générée
        "demo_1000_points.dxf",
        "demo_5000_points.dxf"
    ]

    analyzer = TextureAnalyzer()

    for image_name in test_images:
        if os.path.exists(image_name):
            print(f"\n🔍 Analyse de: {image_name}")
            try:
                image = cv2.imread(image_name)
                if image is not None:
                    results = analyzer.analyze_texture_hybrid(image)
                    print(f"📋 Résultats ({len(results)} détections):")
                    for i, result in enumerate(results):
                        print(f"  {i+1}. {result['texture']} ({result['confidence']:.2f})")
                        print(f"     {result['description']}")
                else:
                    print("❌ Impossible de charger l'image")
            except Exception as e:
                print(f"❌ Erreur avec {image_name}: {e}")
        else:
            print(f"⚠️ Image non trouvée: {image_name}")

    # Recherche d'autres images PNG/JPG dans le répertoire
    print("\n🔍 Recherche d'autres images...")
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    found_images = []
    for ext in image_extensions:
        found_images.extend(glob.glob(ext))

    if found_images:
        print(f"📸 Images trouvées: {len(found_images)}")
        for img in found_images[:5]:  # Limiter à 5 images
            print(f"   - {img}")
    else:
        print("📭 Aucune autre image trouvée")

    print("\n🎯 L'analyse hybride est maintenant opérationnelle!")
    print("💡 Elle peut détecter: rouille, corrosion, inondation, brûlures, sols dégradés")

except Exception as e:
    print(f"❌ Erreur générale: {e}")
    import traceback
    traceback.print_exc()

print("=== Démonstration terminée ===")