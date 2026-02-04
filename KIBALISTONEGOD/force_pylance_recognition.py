#!/usr/bin/env python3
"""
Script pour forcer la reconnaissance de l'environnement Python par Pylance
"""
import sys
import os
import shutil

print("🔧 FORÇAGE DE LA RECONNAISSANCE PYLANCE")
print("=" * 50)

# Test de l'environnement Python
print("\n✅ Python trouvé:", sys.executable)
print("✅ Version:", sys.version.split()[0])

# Test rapide des imports
print("\n📦 Test des imports principaux...")
imports = ['streamlit', 'torch', 'numpy', 'PIL', 'plotly', 'open3d', 'transformers', 'sklearn']
for mod in imports:
    try:
        __import__(mod)
        print(f'✅ {mod}')
    except ImportError as e:
        print(f'❌ {mod}: {e}')

print("\n🎯 INSTRUCTIONS SUIVANTES:")
print("=" * 30)
print("1️⃣ Fermez COMPLETEMENT VS Code (Ctrl+Shift+W)")
print("2️⃣ Attendez 10 secondes minimum")
print("3️⃣ Redémarrez VS Code")
print("4️⃣ Ouvrez le workspace KIBALISTONEGOD")
print("5️⃣ Ouvrez Dust3r.py")
print("6️⃣ Si les erreurs persistent, cliquez sur l'interpréteur Python")
print("   en bas à droite et sélectionnez l'environnement python311")
print()
print("🔍 Vérifiez que l'interpréteur affiché est:")
print("   'C:\\Users\\Admin\\Desktop\\logiciel\\KIBALISTONEGOD\\python311\\python.exe'")
print()
print("✅ Les erreurs Pylance devraient disparaître!")

input("\nAppuyez sur Entrée pour continuer...")