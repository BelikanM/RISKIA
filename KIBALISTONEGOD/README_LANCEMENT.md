# 🚀 Dust3r - Photogrammétrie IA Ultra-Puissante

Application de reconstruction 3D à partir d'images utilisant l'IA avancée (DUSt3R, CLIP, Phi-1.5).

## ✅ Installation Terminée

Toutes les dépendances sont installées dans l'environnement Python portable.

## 🔧 Résolution des Erreurs Pylance

Les erreurs "Import could not be resolved" dans VS Code sont normales car Pylance utilise l'interpréteur Python système par défaut.

### ✅ Solutions Appliquées :

1. **Configuration VS Code** :
   - `.vscode/settings.json` : Configure l'interpréteur Python portable
   - `pyrightconfig.json` : Masque les erreurs d'import et configure les chemins
   - `.python.env` : Définit les variables d'environnement Python

2. **Redémarrage VS Code** :
   - Fermez complètement VS Code
   - Ouvrez le dossier `KIBALISTONEGOD`
   - Les erreurs devraient disparaître automatiquement

3. **Vérification de l'Interpréteur** :
   - `Ctrl+Shift+P` → "Python: Select Interpreter"
   - Vérifiez que c'est bien : `C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD\python311\python.exe`

## 🚀 Lancement de l'Application

### Méthode 1 : Script Batch (Recommandé)
```bash
Double-cliquez sur LANCER_DUST3R.bat
```

### Méthode 2 : Script PowerShell
```powershell
.\LANCER_DUST3R.ps1
```

### Méthode 3 : Ligne de commande
```bash
python311\python.exe -m streamlit run Dust3r.py --server.port 8501
```

## 🔧 Résolution DÉFINITIVE des Erreurs Pylance

### ✅ Méthode Garantie (Redémarrage Complet)

1. **Fermez complètement VS Code** :
   ```bash
   # Dans le terminal :
   taskkill /f /im Code.exe /t
   ```

2. **Utilisez le lanceur dédié** :
   ```bash
   Double-cliquez sur LANCER_VSCODE.bat
   ```
   Ou :
   ```powershell
   .\LANCER_VSCODE.ps1
   ```

3. **Vérification** :
   - Ouvrez un terminal intégré dans VS Code (`Ctrl+Shift+ÿ`)
   - Tapez : `python --version`
   - Vous devriez voir : `Python 3.11.x`

### 🔍 Vérifications Supplémentaires

Si les erreurs persistent :

1. **Vérifiez l'interpréteur actif** :
   - `Ctrl+Shift+P` → "Python: Select Interpreter"
   - Sélectionnez : `C:\Users\Admin\Desktop\logiciel\KIBALISTONEGOD\python311\python.exe`

2. **Rechargez la fenêtre VS Code** :
   - `Ctrl+Shift+P` → "Developer: Reload Window"

3. **Vérifiez les configurations** :
   - Les fichiers `.vscode/settings.json` et `pyrightconfig.json` sont présents
   - Le fichier `.python.env` définit les bonnes variables

## 🎯 Fonctionnalités

- **Reconstruction 3D** : À partir de photos multiples
- **Textures PBR** : Injection intelligente de matériaux
- **UV Mapping Avancé** : Optimisation automatique des coutures
- **IA Multi-Modèles** :
  - DUSt3R : Photogrammétrie
  - CLIP : Analyse d'images
  - Phi-1.5 : Traitement du langage

## 📁 Structure du Projet

```
KIBALISTONEGOD/
├── Dust3r.py                 # Application principale
├── python311/               # Python portable avec toutes les dépendances
├── dust3r/                  # Bibliothèque DUSt3R
├── models--naver--DUSt3R_ViTLarge_BaseDecoder_512_dpt/
├── models--openai--clip-vit-base-patch32/
├── phi-1_5/                 # Modèle Phi-1.5
├── kibali-final-merged/     # Modèle Kibali
├── texture_pbr_analyzer.py  # Analyseur de textures PBR
├── intelligent_vfx_engine.py # Moteur VFX IA
├── auto_pbr_mapper.py       # Mappeur PBR automatique
├── texture_download_manager.py # Gestionnaire de téléchargements
├── LANCER_DUST3R.bat       # Script de lancement (Batch)
├── LANCER_DUST3R.ps1       # Script de lancement (PowerShell)
├── LANCER_VSCODE.bat       # Lanceur VS Code avec Python portable
├── LANCER_VSCODE.ps1       # Lanceur VS Code (PowerShell)
├── TEST_ENVIRONMENT.bat    # Test rapide de l'environnement
├── .vscode/                # Configuration VS Code
├── pyrightconfig.json      # Configuration Pylance
└── .python.env            # Variables d'environnement
```

## 🌐 Utilisation

1. Lancez `LANCER_DUST3R.bat`
2. Ouvrez http://localhost:8501 dans votre navigateur
3. Importez vos images
4. Lancez la reconstruction 3D

## ⚡ Performance

- **GPU** : CUDA 13.0 supporté
- **CPU** : Optimisé pour multi-threading
- **Mémoire** : Gestion intelligente des ressources

## 🔧 Dépannage

### Erreurs Pylance Persistantes
```bash
# Forcer l'utilisation du bon interpréteur
python311\python.exe -c "import sys; print(sys.executable)"
```

### Problèmes de Lancement
```bash
# Vérifier les dépendances
python311\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## 📞 Support

L'application est entièrement portable et ne nécessite aucune installation système.