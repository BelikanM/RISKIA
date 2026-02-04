#!/usr/bin/env python3
"""
Test script pour le pipeline de downsampling temps réel
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import open3d as o3d
import time

def create_test_point_cloud(n_points=1000000):
    """Crée un nuage de points de test"""
    # Distribution gaussienne 3D
    points = np.random.normal(0, 1, (n_points, 3))

    # Ajout de couleurs aléatoires
    colors = np.random.random((n_points, 3))

    # Création du point cloud Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def test_realtime_downsampling():
    """Test du pipeline de downsampling temps réel"""
    print("🚀 Test du Pipeline de Downsampling Temps Réel Ultra-Rapide")
    print("=" * 60)

    # Import de la fonction
    try:
        from Dust3r import apply_realtime_downsampling_pipeline
        print("✅ Fonction importée avec succès")
    except ImportError as e:
        print(f"❌ Erreur d'import : {e}")
        return

    # Création d'un nuage de test massif
    print("\n📊 Création d'un nuage de points de test (1M points)...")
    pcd = create_test_point_cloud(1000000)
    original_points = len(np.asarray(pcd.points))
    print(f"✅ Nuage créé : {original_points:,} points")

    # Test des différentes stratégies
    strategies = ['auto', 'speed', 'quality', 'balanced']
    target_points = 100000

    for strategy in strategies:
        print(f"\n🎯 Test stratégie '{strategy}' → {target_points:,} points cibles")
        print("-" * 40)

        start_time = time.time()
        try:
            downsampled_pcd = apply_realtime_downsampling_pipeline(
                pcd,
                target_points=target_points,
                strategy=strategy,
                preserve_colors=True,
                preserve_normals=False
            )

            processing_time = time.time() - start_time
            final_points = len(np.asarray(downsampled_pcd.points))
            compression_ratio = original_points / final_points

            print("✅ Downsampling réussi !")
            print(f"   Points finaux : {final_points:,}")
            print(f"   Temps : {processing_time:.1f}ms")
            print(f"   Compression : {compression_ratio:.1f}x")
            
            # Vérification de la qualité
            if abs(final_points - target_points) / target_points < 0.1:  # Tolérance 10%
                print("   🎯 Objectif atteint !")
            else:
                print("   ⚠️ Objectif partiellement atteint")

        except Exception as e:
            print(f"❌ Erreur : {e}")

    print("\n🎉 Test terminé !")

if __name__ == "__main__":
    test_realtime_downsampling()