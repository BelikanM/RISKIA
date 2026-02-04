"""
Module de Mapping Automatique PBR + Pipeline Temps Réel
Amélioration du système d'analyse pour précision supérieure et génération automatique
Développé par NYUNDU FRANCIS ARNAUD pour SETRAF GABON
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from PIL import Image
import json


def convert_numpy_to_python(obj):
    """Convertit récursivement les types numpy en types Python natifs pour sérialisation JSON"""
    if isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

try:
    from transformers import CLIPModel, CLIPProcessor
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    CLIPModel = None  # type: ignore
    CLIPProcessor = None  # type: ignore


class AutoPBRMapper:
    """
    Système de mapping automatique PBR sur mesh avec génération intelligente
    """
    
    # Maps PBR templates par type de matériau
    PBR_TEMPLATES = {
        "concrete": {
            "base_color": [0.5, 0.5, 0.5],
            "roughness": 0.8,
            "metallic": 0.0,
            "specular": 0.3,
            "normal_strength": 0.5,
            "ao_strength": 0.7,
            "displacement_scale": 0.02
        },
        "metal": {
            "base_color": [0.7, 0.7, 0.7],
            "roughness": 0.3,
            "metallic": 1.0,
            "specular": 0.9,
            "normal_strength": 0.3,
            "ao_strength": 0.5,
            "displacement_scale": 0.005
        },
        "wood": {
            "base_color": [0.4, 0.25, 0.15],
            "roughness": 0.7,
            "metallic": 0.0,
            "specular": 0.2,
            "normal_strength": 0.6,
            "ao_strength": 0.8,
            "displacement_scale": 0.01
        },
        "stone": {
            "base_color": [0.4, 0.4, 0.4],
            "roughness": 0.9,
            "metallic": 0.0,
            "specular": 0.2,
            "normal_strength": 0.7,
            "ao_strength": 0.9,
            "displacement_scale": 0.03
        },
        "asphalt": {
            "base_color": [0.15, 0.15, 0.15],
            "roughness": 0.85,
            "metallic": 0.0,
            "specular": 0.1,
            "normal_strength": 0.4,
            "ao_strength": 0.6,
            "displacement_scale": 0.005
        },
        "grass": {
            "base_color": [0.2, 0.4, 0.15],
            "roughness": 0.9,
            "metallic": 0.0,
            "specular": 0.1,
            "normal_strength": 0.8,
            "ao_strength": 0.7,
            "displacement_scale": 0.02
        }
    }
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialise le mapper PBR automatique"""
        self.device = device
        print(f"🎨 Initialisation AutoPBRMapper sur {device.upper()}")
    
    def generate_pbr_from_analysis(self, material_scores: Dict[str, float],
                                   scene_type: str) -> Dict[str, any]:
        """
        Génère des paramètres PBR optimaux à partir de l'analyse
        
        Args:
            material_scores: Scores de détection des matériaux
            scene_type: Type de scène détecté
            
        Returns:
            Dict avec paramètres PBR complets
        """
        # Identifier le matériau dominant
        dominant_material = max(material_scores.items(), key=lambda x: x[1])[0]
        
        # Mapper vers template
        material_key = self._map_to_template(dominant_material)
        base_pbr = self.PBR_TEMPLATES.get(material_key, self.PBR_TEMPLATES["concrete"])
        
        # Ajustements contextuels selon scène
        adjusted_pbr = self._adjust_for_scene(base_pbr.copy(), scene_type)
        
        # Informations enrichies
        result = {
            "dominant_material": dominant_material,
            "material_confidence": material_scores[dominant_material],
            "material_template": material_key,
            "pbr_parameters": adjusted_pbr,
            "texture_recommendations": self._generate_texture_names(material_key),
            "shader_hints": self._generate_shader_hints(material_key, scene_type)
        }
        
        return result
    
    def _map_to_template(self, material_label: str) -> str:
        """Mappe un label CLIP vers un template PBR"""
        label_lower = material_label.lower()
        
        if any(word in label_lower for word in ["concrete", "cement"]):
            return "concrete"
        elif any(word in label_lower for word in ["metal", "steel", "aluminum", "iron"]):
            return "metal"
        elif any(word in label_lower for word in ["wood", "wooden", "plank"]):
            return "wood"
        elif any(word in label_lower for word in ["stone", "rock", "granite", "marble"]):
            return "stone"
        elif any(word in label_lower for word in ["asphalt", "road", "pavement"]):
            return "asphalt"
        elif any(word in label_lower for word in ["grass", "vegetation"]):
            return "grass"
        else:
            return "concrete"  # Défaut
    
    def _adjust_for_scene(self, pbr: Dict, scene_type: str) -> Dict:
        """Ajuste les paramètres PBR selon le contexte de scène"""
        scene_lower = scene_type.lower()
        
        # Environnements extérieurs = plus de rugosité
        if "outdoor" in scene_lower or "exterior" in scene_lower:
            pbr["roughness"] = min(1.0, pbr["roughness"] * 1.2)
            pbr["ao_strength"] = min(1.0, pbr["ao_strength"] * 1.1)
        
        # Bâtiments abandonnés = plus d'usure
        if "abandoned" in scene_lower or "old" in scene_lower or "weathered" in scene_lower:
            pbr["roughness"] = min(1.0, pbr["roughness"] * 1.3)
            pbr["normal_strength"] = min(1.0, pbr["normal_strength"] * 1.2)
        
        # Environnements industriels = plus métallique
        if "industrial" in scene_lower or "warehouse" in scene_lower:
            if pbr["metallic"] > 0.5:
                pbr["roughness"] *= 1.1  # Métal industriel plus rugueux
        
        # Environnements modernes = plus lisse
        if "modern" in scene_lower or "new" in scene_lower or "clean" in scene_lower:
            pbr["roughness"] *= 0.8
            pbr["specular"] = min(1.0, pbr["specular"] * 1.2)
        
        return pbr
    
    def _generate_texture_names(self, material_key: str) -> List[str]:
        """Génère les noms de textures PBR recommandées"""
        base_name = material_key
        
        return [
            f"{base_name}_albedo_4k.jpg",
            f"{base_name}_normal_4k.jpg",
            f"{base_name}_roughness_4k.jpg",
            f"{base_name}_metallic_4k.jpg",
            f"{base_name}_ao_4k.jpg",
            f"{base_name}_displacement_4k.jpg",
            f"{base_name}_specular_4k.jpg"
        ]
    
    def _generate_shader_hints(self, material_key: str, scene_type: str) -> Dict:
        """Génère des conseils pour shaders/rendu temps réel"""
        hints = {
            "recommended_shader": "PBR Standard",
            "lighting_model": "GGX",
            "use_parallax_mapping": False,
            "use_tessellation": False,
            "lod_bias": 0.0,
            "anisotropic_filtering": 8
        }
        
        # Ajustements selon matériau
        if material_key in ["stone", "concrete"]:
            hints["use_parallax_mapping"] = True  # Parallax pour relief
            hints["use_tessellation"] = True
        
        if material_key == "metal":
            hints["use_anisotropic_reflection"] = True
            hints["clearcoat_layer"] = True
        
        # Ajustements selon scène
        if "close" in scene_type.lower() or "detail" in scene_type.lower():
            hints["lod_bias"] = -1.0  # Plus de détails
            hints["anisotropic_filtering"] = 16
        
        return hints
    
    def generate_uv_unwrap_strategy(self, vertices: np.ndarray,
                                    normals: np.ndarray) -> Dict:
        """
        Génère une stratégie de UV unwrapping automatique
        
        Args:
            vertices: Positions des vertices (N, 3)
            normals: Normales (N, 3)
            
        Returns:
            Stratégie d'unwrap avec paramètres
        """
        # Analyse de la géométrie
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        dimensions = bbox_max - bbox_min
        
        # Déterminer la stratégie d'unwrap
        aspect_ratio = dimensions.max() / (dimensions.min() + 1e-6)
        
        if aspect_ratio > 3.0:
            # Géométrie allongée → projection cylindrique
            strategy = "cylindrical"
            axis = np.argmax(dimensions)
        elif np.std(normals, axis=0).mean() < 0.3:
            # Normales uniformes → projection planaire
            strategy = "planar"
            axis = np.argmax(np.abs(normals.mean(axis=0)))
        else:
            # Géométrie complexe → smart UV project
            strategy = "smart_uv"
            axis = None
        
        result = {
            "strategy": strategy,
            "projection_axis": int(axis) if axis is not None else None,
            "scale_factor": float(dimensions.max()),
            "rotation": 0.0,
            "margin": 0.02,
            "island_margin": 0.01
        }
        
        return convert_numpy_to_python(result)
    
    def estimate_texture_resolution(self, num_vertices: int,
                                    scene_importance: str = "medium") -> int:
        """
        Estime la résolution de texture optimale
        
        Args:
            num_vertices: Nombre de vertices du mesh
            scene_importance: Importance de la scène (low/medium/high)
            
        Returns:
            Résolution recommandée (512, 1024, 2048, 4096)
        """
        importance_multiplier = {
            "low": 0.5,
            "medium": 1.0,
            "high": 2.0
        }.get(scene_importance, 1.0)
        
        base_resolution = 1024
        
        if num_vertices < 10000:
            base_resolution = 512
        elif num_vertices < 50000:
            base_resolution = 1024
        elif num_vertices < 200000:
            base_resolution = 2048
        else:
            base_resolution = 4096
        
        # Ajuster selon importance
        final_resolution = int(base_resolution * importance_multiplier)
        
        # Clamp aux résolutions standards
        if final_resolution <= 512:
            resolution = 512
        elif final_resolution <= 1024:
            resolution = 1024
        elif final_resolution <= 2048:
            resolution = 2048
        else:
            resolution = 4096
        
        return int(resolution)
    
    def generate_realtime_pipeline_config(self, analysis_result: Dict) -> Dict:
        """
        Génère une configuration complète pour pipeline temps réel
        
        Args:
            analysis_result: Résultat de l'analyse PBR
            
        Returns:
            Configuration du pipeline temps réel
        """
        config = {
            "rendering": {
                "engine": "OpenGL 4.5 / Vulkan",
                "shading_model": "PBR (Physically Based)",
                "lighting": "Image Based Lighting (IBL)",
                "shadows": "PCF Soft Shadows",
                "ambient_occlusion": "SSAO (Screen Space)",
                "anti_aliasing": "FXAA / TAA"
            },
            "textures": {
                "format": "BC7 (DX11+) / ASTC (Mobile)",
                "mipmap_generation": "Auto with custom filter",
                "streaming": True,
                "compression": "High Quality"
            },
            "geometry": {
                "lod_levels": 3,
                "culling": "Frustum + Occlusion",
                "instancing": True if analysis_result.get("num_objects", 1) > 10 else False
            },
            "performance": {
                "target_fps": 60,
                "dynamic_resolution": True,
                "adaptive_quality": True
            },
            "pbr_workflow": {
                "metallic_workflow": True,
                "normal_map_format": "OpenGL (Y+)",
                "height_to_normal": True,
                "ao_as_lightmap": False
            }
        }
        
        return config


# Fonction helper pour génération complète
def generate_complete_pbr_pipeline(material_scores: Dict[str, float],
                                   scene_type: str,
                                   vertices: np.ndarray,
                                   normals: np.ndarray,
                                   device='cuda') -> Dict:
    """
    Génère un pipeline PBR complet automatiquement
    
    Args:
        material_scores: Scores des matériaux détectés
        scene_type: Type de scène
        vertices: Vertices du mesh
        normals: Normales
        device: 'cuda' ou 'cpu'
        
    Returns:
        Configuration complète du pipeline PBR
    """
    mapper = AutoPBRMapper(device=device)
    
    # Génération PBR
    pbr_config = mapper.generate_pbr_from_analysis(material_scores, scene_type)
    
    # Stratégie UV
    uv_strategy = mapper.generate_uv_unwrap_strategy(vertices, normals)
    
    # Résolution de texture
    texture_resolution = mapper.estimate_texture_resolution(len(vertices))
    
    # Pipeline temps réel
    realtime_config = mapper.generate_realtime_pipeline_config({
        "num_objects": 1,
        "complexity": "medium"
    })
    
    # Assemblage complet
    complete_pipeline = {
        "pbr_configuration": pbr_config,
        "uv_unwrap_strategy": uv_strategy,
        "texture_resolution": texture_resolution,
        "realtime_rendering": realtime_config,
        "export_formats": [".gltf", ".fbx", ".obj", ".usd"],
        "optimization_hints": {
            "combine_meshes": len(vertices) < 100000,
            "bake_lighting": False,  # Temps réel = dynamic lighting
            "compress_textures": True,
            "generate_lods": True
        }
    }
    
    # Conversion complète pour sérialisation JSON
    return convert_numpy_to_python(complete_pipeline)


if __name__ == "__main__":
    print("🎨 Module AutoPBR Mapper - Test")
    print("✅ Module prêt à l'emploi")
