#!/usr/bin/env python3
"""
Script de test pour vérifier l'intégration du viewer 3D
sans importer les modules lourds comme torch.
"""

import sys
import os

# Ajouter le répertoire courant au path
sys.path.insert(0, os.path.dirname(__file__))

try:
    # Test des imports de base
    from PyQt6.QtWidgets import QApplication, QWidget
    from PyQt6.QtCore import Qt
    print("✓ PyQt6 importé avec succès")

    # Test du renderer 3D
    from renderer_3d import Model3DViewer, PlotlyRenderer
    print("✓ Renderer 3D importé avec succès")

    # Test de création d'une application Qt
    app = QApplication(sys.argv)
    print("✓ Application Qt créée")

    # Test de création du viewer 3D
    viewer = Model3DViewer()
    print("✓ Model3DViewer créé")

    # Test des propriétés
    print(f"✓ show_wireframe: {viewer.show_wireframe}")
    print(f"✓ show_bounding_boxes: {viewer.show_bounding_boxes}")
    print(f"✓ show_collision: {viewer.show_collision}")

    # Test des méthodes
    viewer.show_wireframe = True
    viewer.show_bounding_boxes = True
    viewer.show_collision = True
    print("✓ Propriétés définies avec succès")

    print("\n🎉 Tous les tests d'intégration 3D réussis !")
    print("L'application RiskIA avec viewer 3D est prête.")

except Exception as e:
    print(f"❌ Erreur lors du test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)