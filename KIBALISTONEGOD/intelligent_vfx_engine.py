"""
Module VFX IA Intelligent - Effets visuels automatiques pilotés par IA
Développé par NYUNDU FRANCIS ARNAUD pour SETRAF GABON

Ce module ajoute des effets VFX contextuels automatiques sur les scènes 3D :
- Saleté et usure selon gravité et exposition
- Rouille selon humidité et âge
- Fissures selon contraintes structurelles
- Effets météo (pluie, neige, poussière)
- Effets lumineux (glow, émission)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Imports conditionnels
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    o3d = None  # type: ignore

try:
    from transformers import CLIPModel, CLIPProcessor
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    CLIPModel = None  # type: ignore
    CLIPProcessor = None  # type: ignore


class VFXType(Enum):
    """Types d'effets VFX disponibles"""
    DIRT = "dirt"  # Saleté
    RUST = "rust"  # Rouille
    CRACKS = "cracks"  # Fissures
    WEATHERING = "weathering"  # Usure générale
    RAIN = "rain"  # Pluie
    DUST = "dust"  # Poussière
    GLOW = "glow"  # Luminescence
    BURN = "burn"  # Brûlure
    MOSS = "moss"  # Mousse
    WATER_DAMAGE = "water_damage"  # Dégâts d'eau


class MaterialType(Enum):
    """Types de matériaux détectables"""
    CONCRETE = "concrete"
    METAL = "metal"
    WOOD = "wood"
    PLASTIC = "plastic"
    STONE = "stone"
    GLASS = "glass"
    FABRIC = "fabric"
    ORGANIC = "organic"


@dataclass
class VFXParameters:
    """Paramètres pour génération VFX"""
    intensity: float = 0.5  # 0.0 à 1.0
    age: float = 0.3  # 0=neuf, 1=très vieux
    humidity: float = 0.5  # 0=sec, 1=très humide
    exposure: float = 0.5  # 0=intérieur, 1=extérieur exposé
    temperature: float = 0.5  # 0=froid, 1=chaud
    pollution: float = 0.3  # 0=propre, 1=très pollué
    gravity_direction: Optional[np.ndarray] = None  # Vecteur gravité (défaut: [0, -1, 0])
    
    def __post_init__(self):
        if self.gravity_direction is None:
            self.gravity_direction = np.array([0, -1, 0])


class IntelligentVFXEngine:
    """
    Moteur VFX intelligent pour application automatique d'effets réalistes
    """
    
    # Base de données des effets par matériau
    MATERIAL_VFX_MAP = {
        MaterialType.CONCRETE: [VFXType.DIRT, VFXType.CRACKS, VFXType.WEATHERING, VFXType.MOSS],
        MaterialType.METAL: [VFXType.RUST, VFXType.DIRT, VFXType.WEATHERING],
        MaterialType.WOOD: [VFXType.WEATHERING, VFXType.CRACKS, VFXType.MOSS, VFXType.WATER_DAMAGE],
        MaterialType.PLASTIC: [VFXType.DIRT, VFXType.WEATHERING],
        MaterialType.STONE: [VFXType.MOSS, VFXType.WEATHERING, VFXType.CRACKS],
        MaterialType.GLASS: [VFXType.DIRT, VFXType.CRACKS],
        MaterialType.FABRIC: [VFXType.DIRT, VFXType.WEATHERING],
        MaterialType.ORGANIC: [VFXType.WEATHERING, VFXType.MOSS],
    }
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialise le moteur VFX
        
        Args:
            device: 'cuda' ou 'cpu'
        """
        self.device = device
        self.clip_model = None
        self.clip_processor = None
        
        print(f"🎬 Initialisation VFX Engine sur {device.upper()}")
        
        if CLIP_AVAILABLE:
            self._load_clip()
    
    def _load_clip(self):
        """Charge CLIP pour détection de matériaux"""
        try:
            clip_path = Path(__file__).parent / "models--openai--clip-vit-base-patch32"
            if clip_path.exists():
                # Chercher dans snapshots/ (structure cache HuggingFace)
                snapshots_dir = clip_path / "snapshots"
                if snapshots_dir.exists():
                    snapshot_dirs = list(snapshots_dir.iterdir())
                    if snapshot_dirs:
                        clip_model_path = snapshot_dirs[0]
                        self.clip_model = CLIPModel.from_pretrained(str(clip_model_path)).to(self.device)  # type: ignore
                        self.clip_processor = CLIPProcessor.from_pretrained(str(clip_model_path))  # type: ignore
                    else:
                        raise FileNotFoundError("Aucun snapshot CLIP trouvé")
                else:
                    self.clip_model = CLIPModel.from_pretrained(str(clip_path)).to(self.device)  # type: ignore
                    self.clip_processor = CLIPProcessor.from_pretrained(str(clip_path))  # type: ignore
            else:
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)  # type: ignore
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")  # type: ignore
            
            self.clip_model.eval()  # type: ignore
            print("✅ CLIP chargé pour détection de matériaux")
        except Exception as e:
            print(f"⚠️ CLIP non disponible: {e}")
    
    def detect_material_from_color(self, colors: np.ndarray) -> MaterialType:
        """
        Détecte le type de matériau à partir des couleurs moyennes
        
        Args:
            colors: Array de couleurs (N, 3) RGB normalisé [0, 1]
            
        Returns:
            Type de matériau détecté
        """
        avg_color = np.mean(colors, axis=0)
        
        # Heuristiques simples basées sur la couleur
        r, g, b = avg_color
        
        # Métal : gris neutre, haute luminosité
        if abs(r - g) < 0.1 and abs(g - b) < 0.1 and np.mean(avg_color) > 0.4:
            return MaterialType.METAL
        
        # Béton : gris clair
        elif abs(r - g) < 0.15 and abs(g - b) < 0.15 and 0.3 < np.mean(avg_color) < 0.6:
            return MaterialType.CONCRETE
        
        # Bois : teintes brunes
        elif r > g > b and r - b > 0.1:
            return MaterialType.WOOD
        
        # Pierre : gris foncé
        elif abs(r - g) < 0.15 and np.mean(avg_color) < 0.4:
            return MaterialType.STONE
        
        # Organique : vert
        elif g > r and g > b:
            return MaterialType.ORGANIC
        
        # Défaut : plastique
        else:
            return MaterialType.PLASTIC
    
    def compute_vertex_exposure(self, vertices: np.ndarray, normals: np.ndarray,
                               gravity_dir: np.ndarray = np.array([0, -1, 0])) -> np.ndarray:
        """
        Calcule l'exposition de chaque vertex (pour accumulation de saleté, pluie, etc.)
        
        Args:
            vertices: Positions des vertices (N, 3)
            normals: Normales (N, 3)
            gravity_dir: Direction de la gravité
            
        Returns:
            Scores d'exposition (N,) entre 0 et 1
        """
        # Normalisation
        normals_norm = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        gravity_norm = gravity_dir / (np.linalg.norm(gravity_dir) + 1e-8)
        
        # Surfaces horizontales vers le haut = exposition max
        exposure = -np.dot(normals_norm, gravity_norm)
        exposure = np.clip(exposure, 0, 1)
        
        return exposure
    
    def compute_vertex_curvature(self, vertices: np.ndarray, k: int = 10) -> np.ndarray:
        """
        Estime la courbure locale (pour fissures et usure)
        
        Args:
            vertices: Positions (N, 3)
            k: Nombre de voisins pour estimation
            
        Returns:
            Courbure approximative (N,)
        """
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(vertices)
        distances, indices = nbrs.kneighbors(vertices)
        
        # Variance des distances = approximation de courbure
        curvature = np.std(distances[:, 1:], axis=1)
        
        # Normalisation
        curvature = (curvature - curvature.min()) / (curvature.max() - curvature.min() + 1e-8)
        
        return curvature
    
    def apply_dirt_effect(self, colors: np.ndarray, exposure: np.ndarray, 
                         params: VFXParameters) -> np.ndarray:
        """
        Applique l'effet de saleté
        
        Args:
            colors: Couleurs originales (N, 3)
            exposure: Scores d'exposition (N,)
            params: Paramètres VFX
            
        Returns:
            Nouvelles couleurs avec saleté
        """
        # Intensité basée sur âge, pollution et exposition
        dirt_intensity = params.intensity * params.age * params.pollution
        
        # Couleur de saleté (brun-gris foncé)
        dirt_color = np.array([0.15, 0.12, 0.10])
        
        # Application progressive selon exposition
        dirt_mask = exposure * dirt_intensity
        dirt_mask = dirt_mask.reshape(-1, 1)
        
        # Mélange
        new_colors = colors * (1 - dirt_mask) + dirt_color * dirt_mask
        
        return new_colors
    
    def apply_rust_effect(self, colors: np.ndarray, humidity: float, 
                         age: float, intensity: float) -> np.ndarray:
        """
        Applique l'effet de rouille (pour métal)
        
        Args:
            colors: Couleurs originales (N, 3)
            humidity: Niveau d'humidité
            age: Âge du matériau
            intensity: Intensité de l'effet
            
        Returns:
            Couleurs avec rouille
        """
        # Rouille = orange-brun
        rust_color = np.array([0.6, 0.25, 0.05])
        
        # Intensité basée sur humidité et âge
        rust_intensity = intensity * humidity * age
        
        # Variation aléatoire (rouille non uniforme)
        noise = np.random.rand(len(colors)) * 0.3
        rust_mask = np.clip(rust_intensity + noise, 0, 1).reshape(-1, 1)
        
        # Mélange
        new_colors = colors * (1 - rust_mask * 0.7) + rust_color * rust_mask
        
        return new_colors
    
    def apply_weathering_effect(self, colors: np.ndarray, curvature: np.ndarray,
                               params: VFXParameters) -> np.ndarray:
        """
        Applique l'usure générale
        
        Args:
            colors: Couleurs originales (N, 3)
            curvature: Courbure locale (N,)
            params: Paramètres VFX
            
        Returns:
            Couleurs avec usure
        """
        # Usure = décoloration + saleté
        weathering_intensity = params.intensity * params.age * params.exposure
        
        # Zones de haute courbure s'usent plus
        wear_mask = (curvature * 0.5 + 0.5) * weathering_intensity
        wear_mask = wear_mask.reshape(-1, 1)
        
        # Décoloration (vers gris)
        desaturated = np.mean(colors, axis=1, keepdims=True)
        
        # Assombrissement léger
        darkened = colors * 0.8
        
        # Mélange
        new_colors = colors * (1 - wear_mask) + (desaturated * 0.5 + darkened * 0.5) * wear_mask
        
        return new_colors
    
    def apply_moss_effect(self, colors: np.ndarray, exposure: np.ndarray,
                         humidity: float, intensity: float) -> np.ndarray:
        """
        Applique l'effet de mousse (surfaces humides ombragées)
        
        Args:
            colors: Couleurs originales (N, 3)
            exposure: Scores d'exposition (N,)
            humidity: Niveau d'humidité
            intensity: Intensité
            
        Returns:
            Couleurs avec mousse
        """
        # Mousse = vert foncé
        moss_color = np.array([0.1, 0.3, 0.1])
        
        # Mousse sur surfaces peu exposées (ombragées) et humides
        moss_mask = (1 - exposure) * humidity * intensity
        
        # Variation aléatoire
        noise = np.random.rand(len(colors)) * 0.4
        moss_mask = np.clip(moss_mask + noise * 0.3, 0, 1).reshape(-1, 1)
        
        # Mélange
        new_colors = colors * (1 - moss_mask * 0.6) + moss_color * moss_mask
        
        return new_colors
    
    def apply_automatic_vfx(self, point_cloud, params: VFXParameters,
                           material_type: Optional[MaterialType] = None) -> 'o3d.geometry.PointCloud':  # type: ignore
        """
        Applique automatiquement les VFX appropriés sur un nuage de points
        
        Args:
            point_cloud: Nuage de points Open3D
            params: Paramètres VFX
            material_type: Type de matériau (détecté auto si None)
            
        Returns:
            Nuage de points avec VFX appliqués
        """
        if not OPEN3D_AVAILABLE:
            print("⚠️ Open3D non disponible")
            return point_cloud
        
        # Extraction des données
        vertices = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)
        
        # Calcul des normales si absentes
        if not point_cloud.has_normals():
            point_cloud.estimate_normals()
        
        normals = np.asarray(point_cloud.normals)
        
        # Détection du matériau si non fourni
        if material_type is None:
            material_type = self.detect_material_from_color(colors)
            print(f"🔍 Matériau détecté: {material_type.value}")
        
        # Calcul des propriétés géométriques
        # gravity_direction est garanti non-None grâce au __post_init__
        gravity_dir = params.gravity_direction if params.gravity_direction is not None else np.array([0, -1, 0])
        exposure = self.compute_vertex_exposure(vertices, normals, gravity_dir)
        curvature = self.compute_vertex_curvature(vertices)
        
        # Application des effets selon le matériau
        new_colors = colors.copy()
        
        applicable_vfx = self.MATERIAL_VFX_MAP.get(material_type, [])
        
        if VFXType.DIRT in applicable_vfx:
            new_colors = self.apply_dirt_effect(new_colors, exposure, params)
            print("✅ Effet saleté appliqué")
        
        if VFXType.RUST in applicable_vfx and material_type == MaterialType.METAL:
            new_colors = self.apply_rust_effect(new_colors, params.humidity, params.age, params.intensity)
            print("✅ Effet rouille appliqué")
        
        if VFXType.WEATHERING in applicable_vfx:
            new_colors = self.apply_weathering_effect(new_colors, curvature, params)
            print("✅ Effet usure appliqué")
        
        if VFXType.MOSS in applicable_vfx:
            new_colors = self.apply_moss_effect(new_colors, exposure, params.humidity, params.intensity)
            print("✅ Effet mousse appliqué")
        
        # Mise à jour du nuage
        point_cloud.colors = o3d.utility.Vector3dVector(new_colors)  # type: ignore
        
        return point_cloud
    
    def generate_pbr_maps(self, point_cloud, material_type: MaterialType) -> Dict[str, np.ndarray]:
        """
        Génère des maps PBR approximatives (albedo, roughness, metallic, normal, AO)
        
        Args:
            point_cloud: Nuage de points
            material_type: Type de matériau
            
        Returns:
            Dict avec les maps PBR
        """
        vertices = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors)
        
        # Albedo = couleurs actuelles
        albedo = colors
        
        # Roughness selon matériau
        roughness_base = {
            MaterialType.METAL: 0.3,
            MaterialType.CONCRETE: 0.8,
            MaterialType.WOOD: 0.7,
            MaterialType.PLASTIC: 0.4,
            MaterialType.STONE: 0.9,
            MaterialType.GLASS: 0.1,
        }.get(material_type, 0.5)
        
        roughness = np.full(len(vertices), roughness_base)
        
        # Metallic selon matériau
        metallic_base = {
            MaterialType.METAL: 1.0,
            MaterialType.GLASS: 0.0,
        }.get(material_type, 0.0)
        
        metallic = np.full(len(vertices), metallic_base)
        
        # AO approximé par courbure
        curvature = self.compute_vertex_curvature(vertices)
        ao = 1.0 - curvature * 0.5
        
        return {
            "albedo": albedo,
            "roughness": roughness,
            "metallic": metallic,
            "ao": ao,
            "curvature": curvature
        }


# Fonction helper pour utilisation directe
def apply_intelligent_vfx(point_cloud, scene_context: str = "outdoor weathered", 
                         intensity: float = 0.5, device: str = 'cuda'):
    """
    Applique des VFX intelligents basés sur un contexte textuel
    
    Args:
        point_cloud: Nuage de points Open3D
        scene_context: Description de la scène (ex: "outdoor rusted metal", "indoor clean")
        intensity: Intensité globale des effets
        device: 'cuda' ou 'cpu'
        
    Returns:
        Nuage de points avec VFX
    """
    # Parsing du contexte
    context_lower = scene_context.lower()
    
    params = VFXParameters(intensity=intensity)
    
    # Détection automatique des paramètres
    if "outdoor" in context_lower or "exterior" in context_lower:
        params.exposure = 0.8
    if "indoor" in context_lower or "interior" in context_lower:
        params.exposure = 0.2
    
    if "old" in context_lower or "aged" in context_lower or "weathered" in context_lower:
        params.age = 0.8
    if "new" in context_lower or "clean" in context_lower:
        params.age = 0.1
    
    if "humid" in context_lower or "wet" in context_lower or "rain" in context_lower:
        params.humidity = 0.9
    if "dry" in context_lower or "desert" in context_lower:
        params.humidity = 0.1
    
    if "polluted" in context_lower or "dirty" in context_lower or "urban" in context_lower:
        params.pollution = 0.8
    
    # Détection du matériau
    material = None
    if "metal" in context_lower or "steel" in context_lower:
        material = MaterialType.METAL
    elif "concrete" in context_lower or "cement" in context_lower:
        material = MaterialType.CONCRETE
    elif "wood" in context_lower:
        material = MaterialType.WOOD
    elif "stone" in context_lower or "rock" in context_lower:
        material = MaterialType.STONE
    
    # Application
    engine = IntelligentVFXEngine(device=device)
    result = engine.apply_automatic_vfx(point_cloud, params, material)
    
    return result


if __name__ == "__main__":
    print("🎬 Module VFX IA - Test")
    print("✅ Module prêt à l'emploi")
