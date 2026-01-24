# ⚡ CPT/CPTU PRO ANALYZER - Logiciel Desktop

Application desktop professionnelle pour l'analyse géotechnique de données CPT/CPTU avec interface moderne PySide6.

## 🚀 Fonctionnalités

### 📊 Analyse de Données
- **Chargement automatique** de fichiers Excel (.xlsx/.xls) et texte (.txt/.csv)
- **Mapping intelligent** des colonnes selon normes ISO 22476-1
- **Adaptation automatique** aux formats de fichiers réels avec profondeurs précises
- **Validation des données** et création automatique des colonnes manquantes

### 🔬 Analyse Géotechnique Avancée
- **Classification Robertson** (SBT - Soil Behavior Type)
- **Analyse de liquéfaction** selon normes internationales
- **Clustering automatique** avec K-Means et PCA
- **Calculs géotechniques** (Ic, Fr, Qt, capacité portante)
- **Lissage Savitzky-Golay** pour réduction du bruit

### 📈 Visualisations Puissantes
- **Profils qc/fs** avec lissage automatique
- **Classification Robertson** (Qt vs Fr)
- **Distribution des types de sol** (camembert)
- **Risque de liquéfaction** avec échelle de couleurs
- **Corrélations** qc-fs par type de sol
- **Indice Ic** avec zones SBT
- **Clusters PCA** en 2D
- **Capacité portante** par tranches

### 🎯 Visualisations 3D
- **Nuage de points 3D** avec coloration par qc
- **Couches géologiques 3D** par type de sol
- **Modèle triangulé** du sol

### 🤖 Intelligence Artificielle
- **Explications en temps réel** des graphiques
- **Chat géotechnique** intelligent
- **Recommandations** personnalisées
- **Analyse automatique** des corrélations

## 🛠️ Installation

### Prérequis
- **Python 3.8+**
- **Windows 10/11** (optimisé pour Windows)

### Installation automatique
```bash
# Cloner ou télécharger le projet
cd votre-dossier

# Installer toutes les dépendances
pip install -r requirements.txt
```

### Installation manuelle
```bash
pip install PySide6 pandas numpy matplotlib scikit-learn seaborn pyqtgraph openpyxl scipy
```

## 🚀 Lancement

### Méthode 1: Lanceur automatique
```bash
python launch_desktop.py
```

### Méthode 2: Lancement direct
```bash
python cpt_analyzer_desktop.py
```

## 📖 Utilisation

### 1. Chargement des Données
1. Cliquez sur **"📂 Charger fichier CPT"**
2. Sélectionnez votre fichier (.xlsx, .xls, .txt, .csv)
3. L'application détecte automatiquement le format et mappe les colonnes

### 2. Analyse des Données
1. Ajustez les paramètres d'analyse si nécessaire
2. Cliquez sur **"🚀 Lancer l'analyse"**
3. Attendez la fin du traitement (barre de progression)

### 3. Exploration des Résultats
- **Onglet Données**: Visualisation du tableau et statistiques
- **Onglet Analyse**: Résultats détaillés de l'analyse
- **Onglet Visualisations**: Graphiques interactifs 2D
- **Onglet 3D**: Visualisations tridimensionnelles
- **Onglet IA**: Explications et recommandations intelligentes

### 4. Export des Résultats
- Cliquez sur **"💾 Exporter"** pour sauvegarder en Excel ou CSV

## 🎨 Interface Utilisateur

### Design Moderne
- **Thème professionnel** avec palette moderne
- **Interface intuitive** avec onglets organisés
- **Boutons stylisés** et indicateurs visuels
- **Barre de progression** pour les opérations longues
- **Messages d'état** informatifs

### Navigation
- **Barre d'outils** principale pour actions rapides
- **Menu contextuel** pour options avancées
- **Raccourcis clavier** pour utilisateurs expérimentés

## 📋 Formats de Fichiers Supportés

### Excel (.xlsx, .xls)
- **En-têtes dans la 2ème ligne** (format standard CPTU)
- **Colonnes multiples** avec données réelles
- **Profondeurs précises** de 0m à la profondeur maximale

### Texte (.txt, .csv)
- **Séparateurs automatiques** (tabulation, point-virgule, virgule)
- **Adaptation ISO** des noms de colonnes
- **Création automatique** des colonnes manquantes

### Mapping Automatique des Colonnes
```
Depth → profondeur, depth, prof, z
qc → pression, pointe, qc, cone, q_c
fs → fs, friction, f_s, frottement
```

## 🔧 Fonctionnalités Avancées

### Analyse de Clustering
- **K-Means** avec nombre de clusters configurable
- **Visualisation PCA** en 2D
- **Classification automatique** des groupes

### Calculs Géotechniques
- **Indice Ic** (Soil Behavior Type Index)
- **Ratio de frottement Fr**
- **Résistance normalisée Qt**
- **Capacité portante q_adm**

### Analyse de Risque
- **CRR (Cyclic Resistance Ratio)**
- **FS (Factor of Safety)** pour liquéfaction
- **Seuils de risque** colorés

## 🤖 Intelligence Artificielle

### Explications Automatiques
- **Analyse de graphiques** en temps réel
- **Explications contextuelles** des résultats
- **Recommandations** personnalisées

### Chat Géotechnique
- **Questions naturelles** sur les données
- **Réponses expertes** basées sur l'analyse
- **Suggestions** d'analyses complémentaires

## 📊 Conformité Normes

- **ISO 22476-1** - Investigation géotechnique - Essais au pénétromètre statique
- **ASTM D5778** - Electronic Friction Cone and Piezocone Penetration Testing
- **Classification Robertson** (1986, 1990)
- **Normes européennes** pour l'analyse de liquéfaction

## 🐛 Dépannage

### Problèmes Courants

**Erreur d'import PySide6**
```bash
pip install PySide6
```

**Fichier non reconnu**
- Vérifiez le format du fichier
- Assurez-vous que les colonnes essentielles sont présentes
- L'application crée automatiquement les colonnes manquantes

**IA non disponible**
- Vérifiez la connexion internet pour le modèle IA
- L'application fonctionne sans IA si nécessaire

### Logs et Debug
- Les erreurs sont affichées dans la console
- Messages d'état dans la barre inférieure
- Détails des analyses dans l'onglet "Analyse"

## 📈 Performance

### Optimisations
- **Traitement en arrière-plan** pour les gros fichiers
- **Mise en cache** des calculs coûteux
- **Lazy loading** des visualisations 3D
- **Multithreading** pour l'IA

### Configurations Recommandées
- **RAM**: 4GB minimum, 8GB recommandé
- **Disque**: 500MB pour l'application + espace données
- **CPU**: Dual-core minimum, quad-core recommandé

## 🔄 Migration depuis Streamlit

Cette version desktop remplace complètement l'ancienne application Streamlit avec :

- **Interface native Windows** plus rapide
- **Graphiques matplotlib** plus précis
- **Fonctionnalités hors ligne** complètes
- **Installation standalone** sans navigateur
- **Performance améliorée** pour gros volumes de données

## 📞 Support

Pour support technique :
1. Vérifiez cette documentation
2. Consultez les logs d'erreur
3. Contactez l'équipe de développement

---

**⚡ CPT/CPTU PRO ANALYZER** - Logiciel professionnel d'analyse géotechnique
Conforme ISO 22476-1 • Développé avec PySide6 • Optimisé Windows