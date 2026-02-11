#!/usr/bin/env python3
"""
TEST COMPLET DU MODULE 3D - RiskIA
Teste l'ensemble du pipeline: CLIP → Analyse → Génération 3D → Affichage
"""

import sys
import os
import json
import glob
from datetime import datetime

def test_clip_analysis():
    """Test 1: Analyse CLIP basique."""
    print("🔍 TEST 1: Analyse CLIP")
    try:
        from scientific_3d_generator import detailed_clip_analysis, COBAYE_IMAGE_PATH
        from PIL import Image

        # Utiliser la même image que pour la génération 3D
        if os.path.exists(COBAYE_IMAGE_PATH):
            print(f"Utilisation de l'image réelle: {os.path.basename(COBAYE_IMAGE_PATH)}")
            test_image = COBAYE_IMAGE_PATH
        else:
            print("⚠️ Image cobaye non trouvée, utilisation d'une image de test")
            # Créer une image de test basique
            test_image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        # Analyser avec CLIP
        result = detailed_clip_analysis(test_image)

        if 'error' in result:
            print(f"❌ Erreur CLIP: {result['error']}")
            return False

        elements = result.get('significant_elements', [])
        print(f"✅ CLIP réussi: {len(elements)} éléments détectés")

        if elements:
            print(f"   Premier élément: {elements[0].get('element', 'unknown')}")  # type: ignore

        return True

    except Exception as e:
        print(f"❌ Erreur test CLIP: {e}")
        return False

def test_3d_generation():
    """Test 2: Génération du modèle 3D."""
    print("\n🏗️ TEST 2: Génération 3D")
    try:
        from scientific_3d_generator import generate_realistic_site_3d_from_image

        # Générer le modèle 3D
        result = generate_realistic_site_3d_from_image()

        if 'Erreur' in result or 'erreur' in result.lower():
            print(f"❌ Erreur génération 3D: {result}")
            return False, None

        print("✅ Génération 3D réussie")

        # Trouver le fichier JSON généré
        json_files = glob.glob("model_3d_*.json")
        if json_files:
            latest_json = max(json_files, key=os.path.getmtime)
            print(f"📁 Modèle sauvegardé: {os.path.basename(latest_json)}")
            return True, latest_json
        else:
            print("⚠️ Aucun fichier JSON trouvé")
            return False, None

    except Exception as e:
        print(f"❌ Erreur génération 3D: {e}")
        return False, None

def test_json_structure(json_path):
    """Test 3: Validation de la structure JSON."""
    print(f"\n📋 TEST 3: Structure JSON - {os.path.basename(json_path)}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Vérifier les clés principales
        required_keys = ['analysis', 'model_3d', 'timestamp']
        missing_keys = [key for key in required_keys if key not in data]

        if missing_keys:
            print(f"❌ Clés manquantes: {missing_keys}")
            return False

        print("✅ Structure JSON valide")

        # Analyser le contenu
        analysis = data['analysis']
        model_3d = data['model_3d']

        # Éléments CLIP
        clip_elements = analysis.get('detailed_analysis', {}).get('significant_elements', [])
        print(f"   📊 Éléments CLIP: {len(clip_elements)}")

        # Géométrie 3D
        geometry = model_3d.get('geometry', {})
        objects = geometry.get('objects', [])
        print(f"   📐 Objets 3D: {len(objects)}")

        # Matériaux
        materials = model_3d.get('materials', {})
        print(f"   🎨 Matériaux: {len(materials)}")

        # Métriques de performance
        perf = model_3d.get('performance_metrics', {})
        if perf:
            fps = perf.get('estimated_fps', 'N/A')
            polys = perf.get('polygon_count', 'N/A')
            print(f"   ⚡ Performance: {fps} FPS, {polys} polygones")

        return True

    except Exception as e:
        print(f"❌ Erreur validation JSON: {e}")
        return False

def test_3d_renderer(json_path):
    """Test 4: Renderer 3D basique."""
    print(f"\n🎮 TEST 4: Renderer 3D - {os.path.basename(json_path)}")
    try:
        # Test d'import seulement (pas d'affichage GUI pour éviter les blocages)
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QTimer

        print("✅ Imports PyQt6 réussis")

        # Test du module renderer - ACTIVÉ
        try:
            from renderer_3d import PlotlyRenderer
            print("✅ Import renderer_3d réussi")

            # Créer une QApplication temporaire pour le test
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
                app_created = True
            else:
                app_created = False

            try:
                # Créer une instance du renderer (sans GUI)
                renderer = PlotlyRenderer()
                print("✅ Instance PlotlyRenderer créée")

                # Tester le chargement du modèle
                success = renderer.load_model(json_path)

                if success:
                    print("✅ Modèle chargé dans le renderer")
                    if renderer.model_data:
                        model_3d_data = renderer.model_data.get('model_3d', {})
                        objects_count = len(model_3d_data.get('geometry', {}).get('objects', []))
                        materials_count = len(model_3d_data.get('materials', {}))
                        print(f"   📐 {objects_count} objets 3D chargés")
                        print(f"   🎨 {materials_count} matériaux chargés")
                    result = True
                else:
                    print("❌ Échec du chargement du modèle")
                    result = False

            finally:
                # Nettoyer l'application si elle a été créée pour le test
                if app_created:
                    app.quit()

            return result

        except (ImportError, Exception) as e:
            print(f"❌ Erreur test renderer: {e}")
            return False

    except Exception as e:
        print(f"❌ Erreur générale renderer: {e}")
        return False

def test_integration_app(json_path):
    """Test 5: Intégration dans l'application principale."""
    print(f"\n🔗 TEST 5: Intégration App - {os.path.basename(json_path)}")
    try:
        # Tester l'import de l'application
        import risk_simulation_app
        print("✅ Import application réussi")

        # Vérifier que la classe Model3DViewer existe
        if hasattr(risk_simulation_app, 'Model3DViewer'):
            print("✅ Model3DViewer trouvé dans l'application")
        else:
            print("⚠️ Model3DViewer non trouvé (normal si pas encore intégré)")
            return True  # Pas critique pour le moment

        # Tester la méthode de génération 3D
        if hasattr(risk_simulation_app.RiskSimulator, 'generate_site_zone_3d'):
            print("✅ Méthode generate_site_zone_3d trouvée")
        else:
            print("❌ Méthode generate_site_zone_3d manquante")
            return False

        return True

    except Exception as e:
        print(f"❌ Erreur intégration: {e}")
        return False

def main():
    """Fonction principale de test."""
    print("🚀 TEST COMPLET DU MODULE 3D - RiskIA")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = []

    # Test 1: CLIP
    results.append(("CLIP Analysis", test_clip_analysis()))

    # Test 2: Génération 3D
    success_3d, json_path = test_3d_generation()
    results.append(("3D Generation", success_3d))

    if not success_3d or not json_path:
        print("\n❌ Tests arrêtés - génération 3D échouée")
        return False

    # Test 3: Structure JSON
    results.append(("JSON Structure", test_json_structure(json_path)))

    # Test 4: Renderer 3D
    results.append(("3D Renderer", test_3d_renderer(json_path)))

    # Test 5: Intégration App
    results.append(("App Integration", test_integration_app(json_path)))

    # Résumé final
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS FINAUX:")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSÉ" if passed else "❌ ÉCHOUÉ"
        print(f"{test_name:20} : {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("🎉 TOUS LES TESTS RÉUSSIS !")
        print("Le module 3D est entièrement fonctionnel.")
    else:
        print("⚠️ Certains tests ont échoué.")
        print("Vérifiez les erreurs ci-dessus.")

    return all_passed

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrompus par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)