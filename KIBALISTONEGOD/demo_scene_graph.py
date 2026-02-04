#!/usr/bin/env python3
"""
Démonstration des Graphes de Scènes 3D pour l'Analyse Spatiale Avancée
Basé sur les concepts de l'article Medium "Build 3D Scene Graphs for Spatial AI LLMs"
"""

import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import json

# Import du module d'analyse de graphes de scènes
from scene_graph_analyzer import (  # type: ignore
    SceneGraphBuilder, SceneObject, SpatialRelation, ObjectCategory,
    create_scene_objects_from_point_cloud
)

def create_demo_scene():
    """Crée une scène de démonstration avec différents objets"""
    objects = []

    # Objet 1: Table (au centre)
    table = SceneObject(
        id="table_1",
        category=ObjectCategory.FURNITURE,
        position=np.array([0.0, 0.0, 0.75]),  # Centre de la table à 75cm du sol
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # Pas de rotation
        dimensions=np.array([1.5, 0.8, 0.05]),  # Largeur, profondeur, hauteur
        confidence=0.95,
        semantic_label="Table en bois",
        point_indices=[],  # Sera rempli automatiquement
        bounding_box=o3d.geometry.OrientedBoundingBox(
            center=np.array([0.0, 0.0, 0.75]),
            R=np.eye(3),
            extent=np.array([1.5, 0.8, 0.05])
        ),
        features={'material': 'wood', 'color': 'brown'}
    )
    objects.append(table)

    # Objet 2: Chaise (devant la table)
    chair = SceneObject(
        id="chair_1",
        category=ObjectCategory.FURNITURE,
        position=np.array([0.0, -1.2, 0.45]),  # Devant la table
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        dimensions=np.array([0.5, 0.5, 0.9]),
        confidence=0.90,
        semantic_label="Chaise",
        point_indices=[],
        bounding_box=o3d.geometry.OrientedBoundingBox(
            center=np.array([0.0, -1.2, 0.45]),
            R=np.eye(3),
            extent=np.array([0.5, 0.5, 0.9])
        ),
        features={'material': 'plastic', 'color': 'black'}
    )
    objects.append(chair)

    # Objet 3: Porte (à droite)
    door = SceneObject(
        id="door_1",
        category=ObjectCategory.DOOR,
        position=np.array([2.5, 0.0, 1.0]),  # À droite de la scène
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        dimensions=np.array([0.1, 0.9, 2.0]),
        confidence=0.85,
        semantic_label="Porte d'entrée",
        point_indices=[],
        bounding_box=o3d.geometry.OrientedBoundingBox(
            center=np.array([2.5, 0.0, 1.0]),
            R=np.eye(3),
            extent=np.array([0.1, 0.9, 2.0])
        ),
        features={'material': 'wood', 'state': 'closed'}
    )
    objects.append(door)

    # Objet 4: Fenêtre (en face)
    window = SceneObject(
        id="window_1",
        category=ObjectCategory.WINDOW,
        position=np.array([0.0, 3.0, 1.5]),  # En face de la table
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        dimensions=np.array([1.2, 0.1, 1.0]),
        confidence=0.80,
        semantic_label="Fenêtre",
        point_indices=[],
        bounding_box=o3d.geometry.OrientedBoundingBox(
            center=np.array([0.0, 3.0, 1.5]),
            R=np.eye(3),
            extent=np.array([1.2, 0.1, 1.0])
        ),
        features={'material': 'glass', 'state': 'open'}
    )
    objects.append(window)

    return objects

def demo_scene_graph_analysis():
    """Démonstration complète de l'analyse de graphes de scènes"""
    print("🚀 Démonstration des Graphes de Scènes 3D")
    print("=" * 50)

    # Vérifier la disponibilité du module
    try:
        from scene_graph_analyzer import SceneGraphBuilder  # type: ignore
        scene_graph_available = True
    except ImportError:
        print("❌ Module scene_graph_analyzer non disponible. Veuillez vérifier l'installation.")
        return

    # Le reste du code utilise maintenant les vraies classes

    # Création de la scène de démonstration
    print("📦 Création de la scène de démonstration...")
    scene_objects = create_demo_scene()

    # Construction du graphe
    print("🧠 Construction du graphe de scènes...")
    scene_graph = SceneGraphBuilder(spatial_threshold=3.0, angle_threshold=30)

    # Ajout des objets
    for obj in scene_objects:
        scene_graph.add_object(obj)
        print(f"  ✅ Ajouté: {obj.semantic_label} (ID: {obj.id})")

    # Construction des relations spatiales
    print("🔗 Analyse des relations spatiales...")
    scene_graph.build_spatial_relations()

    # Statistiques
    stats = scene_graph.get_scene_statistics()
    print(f"\n📊 Statistiques du graphe:")
    print(f"  • Objets: {stats['total_objects']}")
    print(f"  • Relations: {stats['total_relations']}")
    print(f"  • Volume de la scène: {stats['spatial_coverage']['volume']:.1f} m³")
    print(f"  • Surface de la scène: {stats['spatial_coverage']['area']:.1f} m²")

    # Affichage des relations
    print(f"\n🔗 Relations spatiales détectées:")
    for u, v, data in scene_graph.graph.edges(data=True):
        relation = data.get('relation', 'unknown')
        confidence = data.get('confidence', 0.0)
        distance = data.get('distance', 0.0)
        print(f"  • {u} ⟷ {v} ({relation}, confiance: {confidence:.2f})")

    # Requêtes en langage naturel
    print(f"\n💬 Test des requêtes en langage naturel:")
    queries = [
        "Quels objets sont près de la porte ?",
        "Qu'est-ce qui supporte la chaise ?",
        "Quels meubles sont dans la pièce ?"
    ]

    for query in queries:
        print(f"\nQuestion: '{query}'")
        results = scene_graph.query_scene(query)

        if results['objects']:
            print("  Objets trouvés:")
            for obj_id in results['objects']:
                obj = scene_graph.objects[obj_id]
                print(f"    • {obj.semantic_label}")

        if results['relations']:
            print("  Relations trouvées:")
            for source, target, data in results['relations']:
                relation = data.get('relation', 'unknown')
                print(f"    • {source} ⟷ {target} ({relation})")

        if not results['objects'] and not results['relations']:
            print("    Aucun résultat trouvé.")

    # Export OpenUSD
    print(f"\n💾 Export au format OpenUSD...")
    usd_data = scene_graph.export_to_openusd_format()

    with open("demo_scene_graph.json", "w", encoding="utf-8") as f:
        f.write(usd_data)

    print("  ✅ Fichier 'demo_scene_graph.json' créé !")

    # Visualisation
    print(f"\n📈 Génération de la visualisation...")
    try:
        fig = scene_graph.visualize_scene_graph()
        fig.write_html("demo_scene_graph.html")
        print("  ✅ Visualisation sauvegardée dans 'demo_scene_graph.html'")
    except Exception as e:
        print(f"  ❌ Erreur de visualisation: {e}")

    print(f"\n🎉 Démonstration terminée !")
    print("Fichiers générés:")
    print("  • demo_scene_graph.json (format OpenUSD)")
    print("  • demo_scene_graph.html (visualisation interactive)")

if __name__ == "__main__":
    demo_scene_graph_analysis()