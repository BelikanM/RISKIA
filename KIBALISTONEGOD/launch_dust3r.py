#!/usr/bin/env python3
"""
Lanceur automatique pour Dust3r.py avec vérification des dépendances
Utilise le Python portable KIBALISTONEGOD
"""

import sys
import os
import subprocess

# Configuration des chemins
PYTHON_PORTABLE = r"C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD\python311\python.exe"
DUST3R_SCRIPT = r"C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD\Dust3r.py"
WORKING_DIR = r"C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD"

def check_dependencies():
    """Vérifie que toutes les dépendances sont installées"""
    required_modules = [
        'streamlit', 'torch', 'PIL', 'numpy', 'plotly',
        'open3d', 'transformers', 'pynvml', 'faiss',
        'pandas', 'sklearn', 'psutil'
    ]

    print("🔍 Vérification des dépendances...")
    missing = []

    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            missing.append(module)
            print(f"✗ {module}")

    if missing:
        print(f"\n❌ Modules manquants: {', '.join(missing)}")
        return False

    print("\n✅ Toutes les dépendances sont présentes!")
    return True

def launch_application():
    """Lance l'application Streamlit"""
    print("\n🚀 Lancement de Dust3r.py...")
    print("L'application sera accessible sur: http://localhost:8501")

    # Commande pour lancer Streamlit
    cmd = [
        PYTHON_PORTABLE, "-m", "streamlit", "run", DUST3R_SCRIPT,
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ]

    try:
        subprocess.run(cmd, cwd=WORKING_DIR, check=True)
    except KeyboardInterrupt:
        print("\n👋 Application arrêtée par l'utilisateur")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erreur lors du lancement: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("   LANCEUR DUST3R - PHOTOGRAMMÉTRIE IA")
    print("=" * 50)

    # Vérification des dépendances
    if not check_dependencies():
        print("\n❌ Veuillez installer les dépendances manquantes")
        sys.exit(1)

    # Lancement de l'application
    launch_application()