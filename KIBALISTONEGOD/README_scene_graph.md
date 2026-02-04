# Graphes de Scènes 3D pour l'Analyse Spatiale Avancée

## Vue d'ensemble

Cette fonctionnalité ajoute des capacités d'analyse spatiale intelligente à l'application DUST3R, inspirée des concepts présentés dans l'article Medium "Build 3D Scene Graphs for Spatial AI LLMs from Point Cloud Python Tutorial".

## Fonctionnalités

### 🧠 Construction de Graphes de Scènes
- **Transformation automatique** de nuages de points en graphes de scènes intelligents
- **Classification sémantique** des objets (meubles, structures, portes, fenêtres, etc.)
- **Analyse des relations spatiales** : contient, supporte, adjacent, proche, etc.

### 🔗 Relations Spatiales
Le système détecte automatiquement les relations suivantes :
- `CONTAINS` : Un objet en contient un autre
- `SUPPORTS` : Un objet supporte un autre objet
- `TOUCHES` : Deux objets se touchent
- `NEAR` : Objets proches les uns des autres
- `ABOVE/BELOW` : Relations verticales
- `LEFT_OF/RIGHT_OF` : Relations latérales
- `FRONT_OF/BEHIND_OF` : Relations de profondeur
- `ADJACENT` : Objets adjacents
- `CONNECTED` : Objets connectés

### 💬 Requêtes en Langage Naturel
Interrogez votre scène 3D avec du texte naturel :
- "Quels objets sont près de la porte ?"
- "Qu'est-ce qui supporte la table ?"
- "Quels meubles sont dans la pièce ?"

### 📊 Analyse Statistique
- Statistiques complètes de la scène
- Métriques de couverture spatiale (volume, surface)
- Distribution des catégories d'objets
- Fréquences des types de relations

### 📈 Visualisation Interactive
- Graphes interactifs avec Plotly
- Couleurs par catégories d'objets
- Arêtes colorées selon les types de relations
- Exploration interactive de la topologie spatiale

### 💾 Export OpenUSD
- Export au format JSON compatible OpenUSD
- Métadonnées complètes des objets et relations
- Prêt pour l'intégration dans les moteurs 3D (Unreal, Unity, Blender)

## Architecture Technique

### Classes Principales

#### `SceneObject`
Représente un objet dans la scène 3D :
```python
@dataclass
class SceneObject:
    id: str                           # Identifiant unique
    category: ObjectCategory         # Catégorie sémantique
    position: np.ndarray             # Position [x, y, z]
    orientation: np.ndarray          # Quaternion [w, x, y, z]
    dimensions: np.ndarray           # Dimensions [l, p, h]
    confidence: float                # Confiance de détection
    semantic_label: str             # Étiquette sémantique
    point_indices: List[int]        # Indices des points appartenant à l'objet
    bounding_box: o3d.geometry.OrientedBoundingBox
    features: Dict[str, Any]        # Caractéristiques supplémentaires
```

#### `SceneGraphBuilder`
Constructeur principal du graphe de scènes :
```python
class SceneGraphBuilder:
    def __init__(self, spatial_threshold: float = 2.0, angle_threshold: float = 30.0):
        # spatial_threshold: Distance max pour relations spatiales
        # angle_threshold: Seuil d'angle pour relations directionnelles

    def add_object(self, obj: SceneObject) -> None:
        # Ajoute un objet à la scène

    def build_spatial_relations(self) -> None:
        # Construit automatiquement toutes les relations spatiales

    def query_scene(self, query: str) -> Dict[str, Any]:
        # Interroge la scène en langage naturel

    def export_to_openusd_format(self) -> str:
        # Exporte en JSON OpenUSD

    def visualize_scene_graph(self) -> go.Figure:
        # Crée une visualisation Plotly interactive
```

### Dépendances

- **NetworkX** : Construction et analyse de graphes
- **Open3D** : Géométrie 3D et boîtes englobantes
- **NumPy** : Calculs vectoriels
- **Plotly** : Visualisations interactives
- **SciPy** : Recherche spatiale (KDTree)

## Utilisation dans DUST3R

### Activation
1. Dans l'interface Streamlit, allez à la section "🧠 Graphe de Scènes 3D Intelligent"
2. Cochez "🧠 Activer l'analyse de graphes de scènes"
3. Configurez les paramètres :
   - Distance spatiale maximale
   - Seuil d'angle directionnel
   - Utilisation d'étiquettes sémantiques
   - Activation des requêtes naturelles

### Workflow
1. **Reconstruction 3D** : Effectuez d'abord une reconstruction avec DUST3R
2. **Construction du graphe** : Cliquez sur "🔨 Construire le Graphe de Scènes"
3. **Analyse** : Explorez les statistiques et visualisations
4. **Interrogation** : Posez des questions en langage naturel
5. **Export** : Téléchargez les résultats au format OpenUSD

## Exemple de Code

```python
from scene_graph_analyzer import SceneGraphBuilder, SceneObject, ObjectCategory
import numpy as np
import open3d as o3d

# Création d'objets de scène
table = SceneObject(
    id="table_1",
    category=ObjectCategory.FURNITURE,
    position=np.array([0.0, 0.0, 0.75]),
    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    dimensions=np.array([1.5, 0.8, 0.05]),
    confidence=0.95,
    semantic_label="Table en bois",
    point_indices=[],
    bounding_box=o3d.geometry.OrientedBoundingBox(...),
    features={'material': 'wood'}
)

# Construction du graphe
scene_graph = SceneGraphBuilder(spatial_threshold=2.0)
scene_graph.add_object(table)
scene_graph.build_spatial_relations()

# Interrogation
results = scene_graph.query_scene("Quels objets sont près de la table ?")
print(results)

# Export
usd_data = scene_graph.export_to_openusd_format()
```

## Démonstration

Lancez la démonstration complète :
```bash
python demo_scene_graph.py
```

Cela génère :
- `demo_scene_graph.json` : Export OpenUSD
- `demo_scene_graph.html` : Visualisation interactive

## Intégration LLM (Futur)

Le système est conçu pour s'intégrer facilement avec des LLMs :
- **Descriptions contextuelles** : Génération automatique de descriptions
- **Requêtes complexes** : "Montre-moi tous les objets rouges près des fenêtres"
- **Analyse sémantique** : Compréhension du contexte spatial
- **Génération de scènes** : Création de nouvelles scènes à partir de descriptions

## Performances

- **Complexité** : O(n²) pour l'analyse des relations (n = nombre d'objets)
- **Optimisation** : Utilise KDTree pour les recherches spatiales efficaces
- **Mémoire** : Stockage efficace des graphes avec NetworkX
- **Évolutivité** : Adapté pour scènes de taille moyenne (10-100 objets)

## Extensions Futures

- **Classification sémantique avancée** avec modèles de deep learning
- **Intégration OpenUSD native** pour compatibilité universelle
- **Physique réaliste** : simulation de gravité et contraintes physiques
- **Multi-scènes** : gestion de scènes hiérarchiques
- **Streaming temps réel** : analyse de scènes dynamiques

## Références

- Article Medium : "Build 3D Scene Graphs for Spatial AI LLMs from Point Cloud Python Tutorial"
- OpenUSD : Universal Scene Description
- NetworkX : Bibliothèque d'analyse de graphes Python
- DUST3R : Reconstruction 3D à partir d'images