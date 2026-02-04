# 📦 DUST3R - Application Portable

## ✅ Votre application est maintenant PORTABLE !

### 📂 Structure du dossier portable :
```
A3E/
├── venv/                    # Environnement Python (inclus)
├── Dust3r.py               # Application principale
├── dust3r/                 # Module DUSt3R
├── DUST3R_PORTABLE.bat     # 🚀 LANCEUR PORTABLE
├── launch_dust3r.bat       # Lanceur alternatif
├── launch_dust3r.py        # Lanceur Python
└── models--naver--DUSt3R_ViTLarge_BaseDecoder_512_dpt/  # Modèles IA
```

---

## 🚀 Comment utiliser sur différents appareils ?

### 1️⃣ **Sur Windows (n'importe quel disque)**
Double-cliquez simplement sur :
```
DUST3R_PORTABLE.bat
```

### 2️⃣ **Sur carte SD / SSD externe / Clé USB**
1. Copiez tout le dossier `A3E/` sur votre support
2. Double-cliquez sur `DUST3R_PORTABLE.bat`
3. ✅ Ça fonctionne !

### 3️⃣ **Sur un autre PC Windows**
1. Branchez votre carte SD/SSD
2. Naviguez vers `[Lettre_disque]:\...\A3E\`
3. Double-cliquez sur `DUST3R_PORTABLE.bat`

---

## ⚙️ Compatibilité

### ✅ Fonctionne sur :
- Windows 10/11 (x64)
- Carte SD
- SSD externe
- Clé USB 3.0
- Disque dur externe
- NVMe externe

### ⚠️ Prérequis :
- **Windows 64 bits**
- **4 GB RAM minimum** (8 GB recommandé)
- **GPU NVIDIA** (optionnel, pour accélération)
- **~5-10 GB d'espace disque** (pour l'environnement complet)

---

## 🔧 Dépannage

### ❌ "Environnement virtuel non trouvé"
**Solution :** Vérifiez que le dossier `venv/` est présent et n'a pas été supprimé.

### ❌ "Python.exe manquant"
**Solution :** Copiez l'intégralité du dossier `A3E/`, pas seulement certains fichiers.

### ❌ "Port 8501 déjà utilisé"
**Solution :** Fermez toutes les autres instances de Streamlit ou modifiez le port dans `DUST3R_PORTABLE.bat` :
```bat
--server.port 8502
```

### ❌ "Erreur CUDA"
**Solution :** 
1. Lancez l'application
2. Dans la barre latérale, **décochez** "Utiliser GPU"
3. L'application basculera sur CPU

---

## 📊 Taille de l'installation complète

| Composant | Taille approximative |
|-----------|---------------------|
| venv/ (Python + libs) | ~4 GB |
| Modèles IA | ~2 GB |
| Application | ~50 MB |
| **TOTAL** | **~6-7 GB** |

**Conseil :** Utilisez une carte SD/SSD d'au moins **16 GB** pour confort.

---

## 🌍 Utilisation multi-machine

### Scénario 1 : Travail terrain
```
1. Copiez A3E/ sur une carte SD de 32 GB
2. Branchez sur votre PC terrain
3. Lancez DUST3R_PORTABLE.bat
4. Traitez vos photos
5. Sauvegardez les résultats
```

### Scénario 2 : Démonstration client
```
1. Copiez A3E/ sur un SSD USB-C rapide
2. Branchez sur le PC du client
3. Démonstration en temps réel
4. Pas d'installation nécessaire
```

### Scénario 3 : Backup de sécurité
```
1. Dupliquez A3E/ sur 2 supports (SD + SSD)
2. Gardez une version de secours
3. Environnement prêt à l'emploi partout
```

---

## 🔐 Sécurité des données

### Données locales uniquement
- ✅ Aucune connexion Internet requise (après téléchargement modèles)
- ✅ Vos photos restent sur votre machine
- ✅ Pas de télémétrie
- ✅ Traitement 100% local

### Sauvegarde recommandée
Copiez régulièrement :
- `reports/` : Rapports générés
- `database.db*` : Base de données projets
- Vos exports 3D (.ply, .obj)

---

## 🚀 Performance selon le support

| Support | Vitesse lecture | Recommandé pour |
|---------|----------------|-----------------|
| NVMe interne | ⭐⭐⭐⭐⭐ | Production intensive |
| SSD USB 3.1 | ⭐⭐⭐⭐ | Travail quotidien |
| SSD USB 3.0 | ⭐⭐⭐ | Démonstrations |
| Carte SD UHS-II | ⭐⭐⭐ | Terrain, backup |
| Disque dur USB | ⭐⭐ | Archive, backup |
| Clé USB 3.0 | ⭐ | Urgence uniquement |

**Conseil :** Pour meilleures performances, utilisez un **SSD externe USB 3.1** ou supérieur.

---

## 📝 Notes techniques

### Chemins relatifs
L'application utilise `%~dp0` (batch) qui détecte automatiquement :
- La lettre du disque actuel
- Le chemin complet vers A3E/
- Pas besoin de configuration manuelle

### Environnement isolé
Le `venv/` contient :
- Python 3.11.8
- PyTorch 2.10.0+cu130
- Toutes les dépendances
- **Aucun conflit** avec d'autres installations Python

---

## 🆘 Support

### Problème non résolu ?
1. Vérifiez que **tout le dossier A3E/** a été copié
2. Testez sur le disque C: d'abord
3. Vérifiez les droits d'écriture sur le support externe
4. Consultez les logs dans la fenêtre de commande

### Contact
**Développé par :** NYUNDU FRANCIS ARNAUD  
**Pour :** SETRAF GABON

---

## ✨ Fonctionnalités

- ✅ **100% portable** - Aucune installation Windows requise
- ✅ **Détection automatique** du disque/dossier
- ✅ **Environnement isolé** - Pas de conflit de versions
- ✅ **Support GPU/CPU** - Bascule automatique
- ✅ **Interface web moderne** - Streamlit
- ✅ **Export multi-formats** - PLY, OBJ, FBX
- ✅ **Base de données incluse** - Historique projets

---

**Version :** 2.0 Portable  
**Date :** Février 2026  
**Licence :** Usage SETRAF GABON
