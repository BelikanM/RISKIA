"""
Module de graphes de scènes 3D pour l'analyse spatiale avancée
Basé sur les concepts de l'article Medium "Build 3D Scene Graphs for Spatial AI LLMs"
Intègre NetworkX pour la création de graphes et l'analyse de relations spatiales
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import open3d as o3d
from scipy.spatial import KDTree
from dataclasses import dataclass
from enum import Enum
import json
import plotly.graph_objects as go

class SpatialRelation(Enum):
    """Types de relations spatiales entre objets"""
    CONTAINS = "contains"
    SUPPORTS = "supports"
    TOUCHES = "touches"
    NEAR = "near"
    ABOVE = "above"
    BELOW = "below"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    FRONT_OF = "front_of"
    BEHIND = "behind_of"
    INSIDE = "inside"
    OUTSIDE = "outside"
    ADJACENT = "adjacent"
    CONNECTED = "connected"

class ObjectCategory(Enum):
    """Catégories d'objets dans la scène"""
    STRUCTURE = "structure"
    FURNITURE = "furniture"
    VEHICLE = "vehicle"
    PERSON = "person"
    ANIMAL = "animal"
    PLANT = "plant"
    GROUND = "ground"
    WALL = "wall"
    CEILING = "ceiling"
    DOOR = "door"
    WINDOW = "window"
    STAIRS = "stairs"
    UNKNOWN = "unknown"

@dataclass
class SceneObject:
    """Représentation d'un objet dans la scène 3D"""
    id: str
    category: ObjectCategory
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # Quaternion [w, x, y, z]
    dimensions: np.ndarray  # [width, height, depth]
    confidence: float
    semantic_label: str
    point_indices: List[int]  # Indices des points appartenant à cet objet
    bounding_box: o3d.geometry.OrientedBoundingBox
    features: Dict[str, Any]  # Caractéristiques supplémentaires

@dataclass
class SpatialEdge:
    """Arête représentant une relation spatiale"""
    source_id: str
    target_id: str
    relation: SpatialRelation
    confidence: float
    distance: float
    metadata: Dict[str, Any]

class SceneGraphBuilder:
    """
    Constructeur de graphes de scènes 3D pour l'analyse spatiale
    Inspiré des concepts de l'article Medium sur les graphes de scènes OpenUSD
    """

    def __init__(self, spatial_threshold: float = 2.0, angle_threshold: float = 30.0):
        """
        Args:
            spatial_threshold: Distance maximale pour considérer deux objets comme liés
            angle_threshold: Seuil d'angle pour les relations directionnelles (degrés)
        """
        self.spatial_threshold = spatial_threshold
        self.angle_threshold = angle_threshold
        self.graph = nx.DiGraph()
        self.objects: Dict[str, SceneObject] = {}
        self.spatial_index: Optional[KDTree] = None

    def add_object(self, obj: SceneObject) -> None:
        """Ajoute un objet à la scène"""
        self.objects[obj.id] = obj
        self.graph.add_node(obj.id,
                          category=obj.category.value,
                          position=obj.position.tolist(),
                          dimensions=obj.dimensions.tolist(),
                          confidence=obj.confidence,
                          semantic_label=obj.semantic_label)

    def build_spatial_relations(self) -> None:
        """Construit les relations spatiales entre tous les objets"""
        if len(self.objects) < 2:
            return

        # Créer l'index spatial pour les recherches efficaces
        positions = np.array([obj.position for obj in self.objects.values()])
        self.spatial_index = KDTree(positions)

        object_ids = list(self.objects.keys())

        for i, obj_id in enumerate(object_ids):
            obj = self.objects[obj_id]

            # Trouver les objets voisins
            neighbors = self._find_spatial_neighbors(obj, spatial_threshold=self.spatial_threshold)

            for neighbor_id, distance in neighbors:
                if neighbor_id == obj_id:
                    continue

                neighbor_obj = self.objects[neighbor_id]

                # Déterminer les relations spatiales
                relations = self._analyze_spatial_relations(obj, neighbor_obj)

                for relation, confidence in relations.items():
                    if confidence > 0.3:  # Seuil de confiance minimal
                        edge = SpatialEdge(
                            source_id=obj_id,
                            target_id=neighbor_id,
                            relation=relation,
                            confidence=confidence,
                            distance=distance,
                            metadata={
                                'source_category': obj.category.value,
                                'target_category': neighbor_obj.category.value,
                                'spatial_distance': distance
                            }
                        )

                        self.graph.add_edge(obj_id, neighbor_id,
                                          relation=relation.value,
                                          confidence=confidence,
                                          distance=distance,
                                          **edge.metadata)

    def _find_spatial_neighbors(self, obj: SceneObject, spatial_threshold: float) -> List[Tuple[str, float]]:
        """Trouve les voisins spatiaux d'un objet"""
        if self.spatial_index is None:
            return []

        # Recherche des voisins dans le rayon spécifié
        query_result = self.spatial_index.query(
            obj.position.reshape(1, -1),
            k=len(self.objects),
            distance_upper_bound=spatial_threshold
        )

        # KDTree.query peut retourner différents formats selon les paramètres
        if len(query_result) == 2:
            distances, indices = query_result
        else:
            # Cas où un seul résultat
            distances, indices = query_result[0], query_result[1]

        neighbors = []
        object_ids = list(self.objects.keys())

        # Convertir en listes pour itération sécurisée
        try:
            # Convertir en numpy arrays pour manipulation sécurisée
            indices_array = np.asarray(indices).flatten()
            distances_array = np.asarray(distances).flatten()

            # Itérer sur les résultats
            for i in range(len(indices_array)):
                idx = indices_array[i]
                dist = distances_array[i]

                idx_int = int(idx)
                if not np.isinf(dist) and 0 <= idx_int < len(object_ids):
                    neighbor_id = object_ids[idx_int]
                    neighbors.append((neighbor_id, float(dist)))
        except (TypeError, ValueError, IndexError) as e:
            # Fallback en cas d'erreur de format
            print(f"Warning: Error parsing KDTree results: {e}")
            pass

        return neighbors

        return neighbors

    def _analyze_spatial_relations(self, obj1: SceneObject, obj2: SceneObject) -> Dict[SpatialRelation, float]:
        """Analyse les relations spatiales entre deux objets"""
        relations = {}

        # Vecteur entre les centres des objets
        vector = obj2.position - obj1.position
        distance = np.linalg.norm(vector)

        if distance > self.spatial_threshold:
            return relations

        # Normaliser le vecteur
        if distance > 1e-6:
            direction = vector / distance
        else:
            direction = np.array([0, 0, 1])  # Vecteur par défaut

        # Analyse des relations directionnelles
        # Projection sur les axes X, Y, Z
        x_proj, y_proj, z_proj = direction

        # Relation "above/below" basée sur Z
        if abs(z_proj) > np.cos(np.radians(self.angle_threshold)):
            if z_proj > 0:
                relations[SpatialRelation.ABOVE] = min(1.0, abs(z_proj) * 2)
            else:
                relations[SpatialRelation.BELOW] = min(1.0, abs(z_proj) * 2)

        # Relation "left/right" basée sur X
        if abs(x_proj) > np.cos(np.radians(self.angle_threshold)):
            if x_proj > 0:
                relations[SpatialRelation.RIGHT_OF] = min(1.0, abs(x_proj) * 2)
            else:
                relations[SpatialRelation.LEFT_OF] = min(1.0, abs(x_proj) * 2)

        # Relation "front/behind" basée sur Y
        if abs(y_proj) > np.cos(np.radians(self.angle_threshold)):
            if y_proj > 0:
                relations[SpatialRelation.FRONT_OF] = min(1.0, abs(y_proj) * 2)
            else:
                relations[SpatialRelation.BEHIND] = min(1.0, abs(y_proj) * 2)

        # Relation de proximité
        proximity_confidence = max(0, 1.0 - (distance / self.spatial_threshold))
        if proximity_confidence > 0.5:
            relations[SpatialRelation.NEAR] = proximity_confidence

        # Analyse des relations structurelles
        structural_relations = self._analyze_structural_relations(obj1, obj2)
        relations.update(structural_relations)

        return relations

    def _analyze_structural_relations(self, obj1: SceneObject, obj2: SceneObject) -> Dict[SpatialRelation, float]:
        """Analyse les relations structurelles (contient, supporte, etc.)"""
        relations = {}

        # Calcul des boîtes englobantes
        box1 = obj1.bounding_box
        box2 = obj2.bounding_box

        # Vérifier si un objet contient l'autre
        if self._box_contains(box1, box2):
            relations[SpatialRelation.CONTAINS] = 0.9
        elif self._box_contains(box2, box1):
            relations[SpatialRelation.INSIDE] = 0.9

        # Vérifier les relations de support
        if self._is_supporting(obj1, obj2):
            relations[SpatialRelation.SUPPORTS] = 0.8

        # Vérifier l'adjacence
        if self._boxes_adjacent(box1, box2):
            relations[SpatialRelation.ADJACENT] = 0.7

        # Vérifier le contact
        if self._boxes_touch(box1, box2):
            relations[SpatialRelation.TOUCHES] = 0.6

        return relations

    def _box_contains(self, box1: o3d.geometry.OrientedBoundingBox,
                     box2: o3d.geometry.OrientedBoundingBox) -> bool:
        """Vérifie si box1 contient box2"""
        # Approximation simple : vérifier si le centre de box2 est à l'intérieur de box1
        center2 = box2.center
        # Cette implémentation simplifiée peut être améliorée avec une vérification plus précise
        return np.linalg.norm(center2 - box1.center) < (np.min(box1.extent) / 2)

    def _is_supporting(self, obj1: SceneObject, obj2: SceneObject) -> bool:
        """Détermine si obj1 supporte obj2 (relation verticale)"""
        # Vérifier l'alignement vertical et la proximité
        vertical_distance = abs(obj2.position[2] - (obj1.position[2] + obj1.dimensions[2] / 2))
        horizontal_distance = np.linalg.norm(obj2.position[:2] - obj1.position[:2])

        return vertical_distance < 0.5 and horizontal_distance < max(obj1.dimensions[:2]) / 2

    def _boxes_adjacent(self, box1: o3d.geometry.OrientedBoundingBox,
                       box2: o3d.geometry.OrientedBoundingBox) -> bool:
        """Vérifie si deux boîtes sont adjacentes"""
        distance = np.linalg.norm(box2.center - box1.center)
        avg_extent = (np.mean(box1.extent) + np.mean(box2.extent)) / 2
        return distance <= avg_extent * 1.5

    def _boxes_touch(self, box1: o3d.geometry.OrientedBoundingBox,
                    box2: o3d.geometry.OrientedBoundingBox) -> bool:
        """Vérifie si deux boîtes se touchent"""
        distance = np.linalg.norm(box2.center - box1.center)
        min_distance = (np.min(box1.extent) + np.min(box2.extent)) / 2
        return distance <= min_distance * 1.1

    def query_scene(self, query: str) -> Dict[str, Any]:
        """
        Interroge la scène en langage naturel (inspiré de l'intégration LLM)
        Exemple: "Quels objets sont près de la porte ?" ou "Qu'est-ce qui supporte la table ?"
        """
        # Analyse simple de la requête (peut être étendue avec un vrai LLM)
        query_lower = query.lower()

        results = {
            'objects': [],
            'relations': [],
            'spatial_analysis': {}
        }

        # Recherche d'objets par catégorie
        if 'porte' in query_lower or 'door' in query_lower:
            door_objects = [obj for obj in self.objects.values() if obj.category == ObjectCategory.DOOR]
            results['objects'].extend([obj.id for obj in door_objects])

        if 'fenêtre' in query_lower or 'window' in query_lower:
            window_objects = [obj for obj in self.objects.values() if obj.category == ObjectCategory.WINDOW]
            results['objects'].extend([obj.id for obj in window_objects])

        # Recherche de relations spatiales
        if 'près' in query_lower or 'near' in query_lower:
            near_relations = [(u, v, d) for u, v, d in self.graph.edges(data=True)
                            if d.get('relation') == SpatialRelation.NEAR.value]
            results['relations'].extend(near_relations)

        if 'supporte' in query_lower or 'supports' in query_lower:
            support_relations = [(u, v, d) for u, v, d in self.graph.edges(data=True)
                               if d.get('relation') == SpatialRelation.SUPPORTS.value]
            results['relations'].extend(support_relations)

        return results

    def export_to_openusd_format(self) -> str:
        """Exporte le graphe de scène au format OpenUSD (JSON simplifié)"""
        usd_data = {
            "scene_graph": {
                "metadata": {
                    "version": "1.0",
                    "created_by": "DUST3R Scene Graph Builder",
                    "spatial_threshold": self.spatial_threshold
                },
                "objects": {},
                "relations": []
            }
        }

        # Exporter les objets
        for obj_id, obj in self.objects.items():
            usd_data["scene_graph"]["objects"][obj_id] = {
                "category": obj.category.value,
                "position": obj.position.tolist(),
                "dimensions": obj.dimensions.tolist(),
                "orientation": obj.orientation.tolist(),
                "confidence": obj.confidence,
                "semantic_label": obj.semantic_label
            }

        # Exporter les relations
        for u, v, data in self.graph.edges(data=True):
            relation_data = {
                "source": u,
                "target": v,
                "relation": data.get("relation"),
                "confidence": data.get("confidence", 0.0),
                "distance": data.get("distance", 0.0)
            }
            usd_data["scene_graph"]["relations"].append(relation_data)

        return json.dumps(usd_data, indent=2, ensure_ascii=False)

    def visualize_scene_graph(self) -> go.Figure:
        """Crée une visualisation interactive du graphe de scène avec Plotly"""
        # Positions des nœuds (utiliser les positions 3D projetées en 2D)
        pos = {}
        for node in self.graph.nodes():
            obj = self.objects[node]
            # Projection simple : utiliser X et Y, ignorer Z pour la visualisation 2D
            pos[node] = (obj.position[0], obj.position[1])

        # Créer les arêtes avec couleurs selon le type de relation
        edge_x = []
        edge_y = []
        edge_colors = []

        for u, v, data in self.graph.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            # Couleur selon le type de relation
            relation = data.get('relation', '')
            if relation == SpatialRelation.SUPPORTS.value:
                edge_colors.append('red')
            elif relation == SpatialRelation.CONTAINS.value:
                edge_colors.append('blue')
            elif relation == SpatialRelation.NEAR.value:
                edge_colors.append('green')
            else:
                edge_colors.append('gray')

        # Créer la figure
        fig = go.Figure()

        # Ajouter les arêtes
        for i in range(0, len(edge_x), 3):
            fig.add_trace(go.Scatter(
                x=edge_x[i:i+3],
                y=edge_y[i:i+3],
                mode='lines',
                line=dict(color=edge_colors[i//3], width=2),
                showlegend=False
            ))

        # Ajouter les nœuds
        node_x = [pos[node][0] for node in self.graph.nodes()]
        node_y = [pos[node][1] for node in self.graph.nodes()]
        node_text = [f"{node}<br>{self.objects[node].semantic_label}" for node in self.graph.nodes()]
        node_colors = [self._get_category_color(self.objects[node].category) for node in self.graph.nodes()]

        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            marker=dict(
                size=20,
                color=node_colors,
                line=dict(width=2, color='white')
            ),
            name='Objets'
        ))

        # Mise à jour du layout
        fig.update_layout(
            title="Graphe de Scène 3D - Relations Spatiales",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )

        return fig

    def _get_category_color(self, category: ObjectCategory) -> str:
        """Retourne une couleur pour chaque catégorie d'objet"""
        color_map = {
            ObjectCategory.STRUCTURE: 'red',
            ObjectCategory.FURNITURE: 'blue',
            ObjectCategory.VEHICLE: 'green',
            ObjectCategory.PERSON: 'orange',
            ObjectCategory.ANIMAL: 'purple',
            ObjectCategory.PLANT: 'brown',
            ObjectCategory.GROUND: 'gray',
            ObjectCategory.WALL: 'black',
            ObjectCategory.CEILING: 'lightgray',
            ObjectCategory.DOOR: 'yellow',
            ObjectCategory.WINDOW: 'cyan',
            ObjectCategory.STAIRS: 'pink',
            ObjectCategory.UNKNOWN: 'white'
        }
        return color_map.get(category, 'white')

    def get_scene_statistics(self) -> Dict[str, Any]:
        """Retourne des statistiques sur la scène"""
        stats = {
            'total_objects': len(self.objects),
            'total_relations': len(self.graph.edges()),
            'categories': {},
            'relation_types': {},
            'spatial_coverage': self._calculate_spatial_coverage()
        }

        # Statistiques par catégorie
        for obj in self.objects.values():
            cat = obj.category.value
            stats['categories'][cat] = stats['categories'].get(cat, 0) + 1

        # Statistiques par type de relation
        for u, v, data in self.graph.edges(data=True):
            rel = data.get('relation', 'unknown')
            stats['relation_types'][rel] = stats['relation_types'].get(rel, 0) + 1

        return stats

    def _calculate_spatial_coverage(self) -> Dict[str, float]:
        """Calcule la couverture spatiale de la scène"""
        if not self.objects:
            return {'volume': 0.0, 'area': 0.0}

        positions = np.array([obj.position for obj in self.objects.values()])
        min_bounds = np.min(positions, axis=0)
        max_bounds = np.max(positions, axis=0)

        # Volume approximatif (boîte englobante)
        volume = np.prod(max_bounds - min_bounds)

        # Surface approximative
        area = 2 * (max_bounds[0] - min_bounds[0]) * (max_bounds[1] - min_bounds[1]) + \
               2 * (max_bounds[1] - min_bounds[1]) * (max_bounds[2] - min_bounds[2]) + \
               2 * (max_bounds[2] - min_bounds[2]) * (max_bounds[0] - min_bounds[0])

        return {'volume': volume, 'area': area}

def create_scene_objects_from_point_cloud(pcd: o3d.geometry.PointCloud,
                                        semantic_labels: Optional[np.ndarray] = None) -> List[SceneObject]:
    """
    Crée des objets de scène à partir d'un nuage de points avec étiquettes sémantiques
    Cette fonction peut être étendue avec un vrai modèle de segmentation sémantique
    """
    objects = []

    if semantic_labels is None:
        # Création d'un objet unique pour tout le nuage de points
        bbox = pcd.get_axis_aligned_bounding_box()
        obj = SceneObject(
            id="scene_main",
            category=ObjectCategory.UNKNOWN,
            position=bbox.center,
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # Quaternion identité
            dimensions=bbox.extent,
            confidence=1.0,
            semantic_label="Scène principale",
            point_indices=list(range(len(pcd.points))),
            bounding_box=bbox,
            features={}
        )
        objects.append(obj)
    else:
        # Grouper les points par étiquette sémantique (implémentation simplifiée)
        unique_labels = np.unique(semantic_labels)

        for label_idx, label in enumerate(unique_labels):
            mask = semantic_labels == label
            if np.sum(mask) < 10:  # Ignorer les groupes trop petits
                continue

            # Extraire les points de cette classe
            points_subset = np.asarray(pcd.points)[mask]
            subset_pcd = o3d.geometry.PointCloud()
            subset_pcd.points = o3d.utility.Vector3dVector(points_subset)

            # Calculer la boîte englobante
            bbox = subset_pcd.get_axis_aligned_bounding_box()

            # Déterminer la catégorie basée sur l'étiquette (mapping simplifié)
            category = _map_semantic_label_to_category(label)

            obj = SceneObject(
                id=f"object_{label_idx}",
                category=category,
                position=bbox.center,
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                dimensions=bbox.extent,
                confidence=0.8,  # Confiance par défaut
                semantic_label=f"Objet {label}",
                point_indices=np.where(mask)[0].tolist(),
                bounding_box=bbox,
                features={'semantic_class': int(label)}
            )
            objects.append(obj)

    return objects

def _map_semantic_label_to_category(label: int) -> ObjectCategory:
    """Mappe une étiquette sémantique à une catégorie d'objet (mapping simplifié)"""
    # Ce mapping devrait être adapté selon le dataset utilisé
    label_mapping = {
        0: ObjectCategory.GROUND,
        1: ObjectCategory.WALL,
        2: ObjectCategory.CEILING,
        3: ObjectCategory.DOOR,
        4: ObjectCategory.WINDOW,
        5: ObjectCategory.FURNITURE,
        6: ObjectCategory.STRUCTURE,
        7: ObjectCategory.PLANT,
        8: ObjectCategory.VEHICLE,
        9: ObjectCategory.PERSON,
        10: ObjectCategory.ANIMAL
    }

    return label_mapping.get(label, ObjectCategory.UNKNOWN)