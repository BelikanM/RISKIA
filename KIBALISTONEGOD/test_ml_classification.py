#!/usr/bin/env python3
"""
Test script pour la classification ML PointNet-inspired
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import open3d as o3d
import time

def create_synthetic_scene():
    """Crée une scène synthétique avec différents types d'objets"""
    points_list = []
    colors_list = []

    # 1. Terrain (bas, plat, vert)
    terrain_points = np.random.uniform([-10, -10, -1], [10, 10, -0.5], (5000, 3))
    terrain_colors = np.tile([0.4, 0.8, 0.4], (5000, 1))  # Vert
    points_list.append(terrain_points)
    colors_list.append(terrain_colors)

    # 2. Bâtiments (vertical, régulier, rouge)
    building_points = []
    building_colors = []
    for i in range(3):  # 3 bâtiments
        # Base du bâtiment
        base_x = np.random.uniform(-5, 5)
        base_y = np.random.uniform(-5, 5)
        height = np.random.uniform(2, 5)

        # Points du bâtiment (parois verticales)
        building = np.random.uniform([base_x-1, base_y-1, 0], [base_x+1, base_y+1, height], (1000, 3))
        building[:, 2] = np.random.uniform(0, height, 1000)  # Hauteur variable
        building_colors_building = np.tile([0.8, 0.4, 0.4], (1000, 1))  # Rouge

        building_points.append(building)
        building_colors.append(building_colors_building)

    points_list.extend(building_points)
    colors_list.extend(building_colors)

    # 3. Végétation (irrégulier, haut, bleu)
    vegetation_points = np.random.uniform([-8, -8, 0.5], [8, 8, 3], (3000, 3))
    # Ajouter de l'irrégularité
    vegetation_points[:, 2] += np.random.normal(0, 0.5, 3000)
    vegetation_colors = np.tile([0.4, 0.4, 0.8], (3000, 1))  # Bleu
    points_list.append(vegetation_points)
    colors_list.append(vegetation_colors)

    # 4. Véhicules (petit, isolé, jaune)
    vehicle_points = np.random.uniform([-3, -3, 0], [3, 3, 1], (500, 3))
    vehicle_colors = np.tile([0.8, 0.8, 0.4], (500, 1))  # Jaune
    points_list.append(vehicle_points)
    colors_list.append(vehicle_colors)

    # Concaténation
    all_points = np.vstack(points_list)
    all_colors = np.vstack(colors_list)

    # Création du point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    return pcd

def test_ml_classification():
    """Test de la classification ML PointNet-inspired"""
    print("🧠 Test de la Classification ML PointNet-inspired")
    print("=" * 60)

    # Import de la fonction
    try:
        from Dust3r import apply_pointnet_classification
        print("✅ Fonction importée avec succès")
    except ImportError as e:
        print(f"❌ Erreur d'import : {e}")
        return

    # Création d'une scène synthétique
    print("\n🏗️ Création d'une scène synthétique avec différents objets...")
    pcd = create_synthetic_scene()
    original_points = len(np.asarray(pcd.points))
    print(f"✅ Scène créée : {original_points:,} points")

    # Test de classification avec différents seuils
    thresholds = [0.5, 0.7, 0.9]

    for threshold in thresholds:
        print(f"\n🎯 Test classification avec seuil {threshold}")
        print("-" * 40)

        start_time = time.time()
        try:
            pcd_classified, stats = apply_pointnet_classification(
                pcd,
                confidence_threshold=threshold
            )

            processing_time = time.time() - start_time

            print("✅ Classification terminée !")
            print(f"   Points classifiés : {stats['classified_objects']:,}/{stats['total_points']:,}")
            print(f"   Confiance moyenne : {stats['avg_confidence']:.2f}")
            print(f"   Temps : {processing_time:.1f}ms")
            
            # Distribution des classes
            class_names = ["Terrain", "Bâtiments", "Végétation", "Véhicules", "Autres"]
            class_dist = stats['class_distribution']

            print("   📊 Distribution :")
            for class_id, count in class_dist.items():
                if count > 0:
                    percentage = (count / stats['total_points']) * 100
                    print(f"     • {class_names[class_id]} : {count:,} points ({percentage:.1f}%)")

        except Exception as e:
            print(f"❌ Erreur : {e}")

    print("\n🎉 Test terminé !")

if __name__ == "__main__":
    test_ml_classification()