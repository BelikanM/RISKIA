# 🧪 Application d'Analyse Géotechnique CPT/CPTU

Application Streamlit complète et robuste pour l'analyse de données de pénétrométrie (Cone Penetration Test) avec plus de **10 visualisations 3D avancées** et une architecture modulaire.

## 🚀 Démarrage Rapide

### Installation
```bash
pip install -r requirements.txt
streamlit run main.py
```

### Création d'Exécutable Windows
```bash
# Build simplifié (recommandé)
python build_simple.py

# OU double-cliquez sur build_simple.bat
```

**Résultat** : `dist/CPT_Analysis_Simple.exe` + dossier `models/`

📖 **Documentation complète** : [BUILD_README.md](BUILD_README.md)

## ✨ Fonctionnalités Principales

### 📁 Upload et Chargement
- **Fichiers .cal** : Chargement automatique des données CPT binaires
- **Validation** : Vérification automatique du format et des données
- **Preprocessing** : Imputation automatique des valeurs manquantes

### 🔍 Analyse Géotechnique Avancée
- **Classification des sols** : Algorithme basé sur Robertson (1990)
- **Calcul CRR** : Cyclic Resistance Ratio pour l'analyse de liquéfaction
- **Clustering automatique** : K-means avec preprocessing (StandardScaler + PCA)
- **Détection d'anomalies** : Identification des points problématiques

### 📊 Visualisations Interactives (Plus de 10 types)

#### Graphiques 2D Améliorés
- **qc vs fs** : Avec classification par type de sol et lignes de référence
- **Profils de profondeur** : Triple vue (qc, fs, combiné) avec remplissage
- **Dashboard combiné** : 6 graphiques en subplot (Plotly)
- **Coupes géologiques** : Sections verticales avec couches colorées
- **Graphiques radar** : Profils normalisés des paramètres
- **Violin plots** : Distributions statistiques avancées
- **Heatmaps** : Corrélations et distributions 2D

#### Visualisations 3D Avancées (10+ types)
- **3D Scatter** : Nuages de points avec couleurs par paramètres
- **3D Surface** : Surfaces interpolées avec contours
- **3D Wireframe** : Structure filaire avec données
- **3D Contours** : Coupes à différentes profondeurs
- **3D Streamlines** : Flux du gradient de qc
- **3D Isosurface** : Surfaces d'égale valeur
- **3D Voxels** : Visualisation volumétrique
- **3D Point Clouds** : Export PLY pour visualisation externe

### 📋 Rapports Détaillés et Export
- **Résumé exécutif** : 5 métriques clés en temps réel
- **Statistiques complètes** : Analyse descriptive détaillée
- **Analyse de risque** : Évaluation quantitative de la liquéfaction
- **Rapport multi-onglets** : Statistiques, sols, risques, export
- **Export multiple** : CSV, TXT, PNG (graphiques)

## 🏗️ Architecture Modulaire

```
logiciel/
├── core/              # Logique principale
├── analysis/          # Analyses géotechniques
├── visualization/     # 10+ types de graphiques
├── utils/            # Utilitaires et session
├── models/           # Algorithmes ML
├── data/             # Traitement des données
├── app.py            # Application principale
├── launch.py         # Lanceur automatique
└── requirements.txt  # Dépendances (sans IA)
```

## 🚀 Installation

### Prérequis
- Python 3.8+
- pip

### Installation automatique
```bash
# Installation des dépendances (version modulaire sans IA)
pip install -r requirements.txt
```

### Lancement
```bash
# Lancement automatique avec détection IP
python launch.py
```

## 🎨 Nouvelles Visualisations

### Dashboard Combiné
- 6 graphiques synchronisés en temps réel
- qc vs fs, profils, distributions, barres

### Coupes Géologiques
- Représentation verticale des couches
- Colorisation automatique par type de sol
- Échelle de profondeur inversée

### Visualisations 3D Avancées
- **Surface** : Interpolation lissée des valeurs
- **Wireframe** : Structure 3D avec points de données
- **Contours** : Coupes horizontales colorées
- **Streamlines** : Visualisation des gradients
- **Isosurface** : Surfaces d'égale résistance
- **Voxels** : Discrétisation volumétrique

### Graphiques Statistiques
- **Radar** : Comparaison normalisée des paramètres
- **Violin** : Distributions avec densité
- **Heatmaps** : Corrélations et patterns temporels

## 📊 Améliorations du Design

- **Interface responsive** : Adaptation automatique à l'écran
- **Métriques en temps réel** : 5 indicateurs clés mis à jour
- **Navigation intuitive** : 4 onglets principaux
- **Exports multiples** : CSV, TXT, images haute résolution
- **Gestion d'erreurs** : Messages informatifs et récupération

## 🔧 Technologies Utilisées

- **Streamlit** : Interface web interactive
- **Pandas/NumPy** : Traitement des données
- **Matplotlib/Seaborn** : Graphiques 2D
- **Plotly** : Graphiques 3D interactifs
- **Scikit-learn** : Machine Learning
- **Open3D/PyVista** : Visualisation 3D avancée

## 📈 Performances

- **Traitement rapide** : Analyse complète en < 30 secondes
- **Visualisations optimisées** : Rendu 3D fluide
- **Mémoire efficace** : Gestion optimisée des gros datasets
- **Export haute qualité** : Images 300 DPI

## 🎯 Cas d'Usage

- **Études géotechniques** : Analyse de sondages CPT
- **Évaluation de risques** : Calcul de liquéfaction
- **Classification de sols** : Automatisation de l'expertise
- **Visualisation 3D** : Présentation de résultats
- **Rapports techniques** : Génération automatique

---

**Développé avec ❤️ pour la géotechnique moderne**
pip install streamlit pandas numpy matplotlib seaborn scikit-learn plotly
pip install PyMuPDF sentence-transformers transformers torch
pip install langchain langchain-community langchain-huggingface
pip install open3d trimesh pyvista xgboost shap faiss-cpu
```

## 📖 Utilisation

### Lancement de l'application
```bash
streamlit run app.py
```

L'application sera accessible sur `http://localhost:8501`

### Workflow d'analyse

1. **📁 Upload des données**
   - Sélectionnez un fichier `.cal` contenant les données CPT
   - Optionnellement, ajoutez un fichier PDF de contexte
   - Cliquez sur "🚀 Traiter les fichiers"

2. **🔍 Analyse et traitement**
   - **Classification des sols** : Identifie automatiquement les types de sol
   - **Calcul CRR** : Évalue le risque de liquéfaction
   - **Clustering** : Regroupe les données similaires

3. **📊 Visualisations**
   - Explorez différents types de graphiques
   - Générez des visualisations 3D
   - Téléchargez les fichiers PLY

4. **🤖 Assistant RAG**
   - Posez des questions sur vos données
   - Obtenez des analyses et recommandations

5. **📋 Rapport**
   - Consultez le rapport complet
   - Téléchargez les données analysées

## 📄 Format des fichiers .cal

Les fichiers `.cal` doivent être au format binaire CPT standard :
- **CPT** : 3 colonnes (Depth, qc, fs)
- **CPTU** : 4 colonnes (Depth, qc, fs, u2)

### Structure attendue
```
Depth (m)    qc (MPa)    fs (MPa)    [u2 (kPa)]
0.1          1.2         0.05        [10.5]
0.2          1.5         0.08        [12.1]
...
```

## 🔧 Configuration

### Variables d'environnement
```bash
# Pour l'API Hugging Face (optionnel)
export HF_TOKEN="votre_token_huggingface"
```

### Personnalisation
- **Nombre de clusters** : Ajustable dans l'interface (2-10)
- **Modèles LLM** : Sélection dans la barre latérale
- **Paramètres de visualisation** : Interactifs

## 🏗️ Architecture

```
app.py                 # Application principale Streamlit
├── Upload & Validation
├── Analyse géotechnique
├── Visualisations
├── Assistant RAG
└── Rapport

requirements.txt       # Dépendances Python
README.md             # Documentation
```

## 🐛 Dépannage

### Erreurs communes

**Erreur de chargement .cal**
- Vérifiez que le fichier est au format binaire CPT
- Assurez-vous que les données ne sont pas corrompues

**Problèmes de mémoire**
- Pour les gros fichiers, utilisez Dask pour le preprocessing
- Réduisez le nombre de clusters

**Erreurs RAG**
- Vérifiez la connexion internet pour les modèles Hugging Face
- Configurez le token HF_TOKEN si nécessaire

### Logs et debug
```bash
# Mode debug
streamlit run app.py --logger.level=debug
```

## 🤝 Contribution

Pour contribuer :
1. Fork le projet
2. Créez une branche feature
3. Committez vos changements
4. Pushez vers la branche
5. Ouvrez une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## 📞 Support

Pour des questions ou support :
- Ouvrez une issue sur GitHub
- Consultez la documentation
- Vérifiez les exemples d'utilisation

---

**Développé avec ❤️ pour la communauté géotechnique**</content>
<parameter name="filePath">C:\Users\Admin\Desktop\logiciel\README.md