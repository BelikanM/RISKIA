"""
Module d'analyse intelligente de scènes 3D pour suggestion de textures PBR
Utilise CLIP (vision) + Phi-1.5 (langage) pour identifier les matériaux nécessaires
Développé par NYUNDU FRANCIS ARNAUD pour SETRAF GABON
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import json
from typing import List, Dict, Tuple
from collections import Counter

# Imports conditionnels avec types par défaut
try:
    from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    CLIPProcessor = None  # type: ignore
    CLIPModel = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    print("⚠️ transformers non disponible - Installation requise: pip install transformers")

class TexturePBRAnalyzer:
    """
    Analyseur intelligent de scènes 3D pour recommandations de textures PBR
    """
    
    # Base de données de matériaux PBR courants
    PBR_MATERIALS_DATABASE = {
        "construction": [
            "concrete", "cement", "brick", "stone", "gravel", "sand",
            "asphalt", "pavement", "tiles", "marble", "granite"
        ],
        "nature": [
            "grass", "soil", "dirt", "mud", "rock", "moss", "leaves",
            "bark", "wood", "water", "snow", "ice"
        ],
        "metal": [
            "steel", "iron", "aluminum", "copper", "rust", "metal",
            "chrome", "brushed_metal", "corrugated_metal"
        ],
        "organic": [
            "wood", "leather", "fabric", "cloth", "carpet", "rubber",
            "plastic", "paint", "paper"
        ],
        "infrastructure": [
            "road", "highway", "railway", "fence", "gate", "wall",
            "roof", "gutter", "pipe", "cable"
        ]
    }
    
    # Labels CLIP pour classification de scène (améliorés pour plus de précision)
    SCENE_LABELS = [
        "outdoor construction site with concrete and steel",
        "indoor building interior with walls and floor",
        "urban street with asphalt and pavement",
        "natural landscape with grass and trees",
        "industrial facility with metal structures",
        "residential area with houses and gardens",
        "commercial building with glass and concrete",
        "infrastructure project with roads and bridges",
        "terrain with rocks and soil",
        "architectural detail close-up",
        "concrete structure and walls",
        "metal framework and beams",
        "wooden construction and planks",
        "stone building and masonry",
        "modern architecture with glass",
        "old weathered abandoned building",
        "clean new construction site",
        "industrial warehouse interior",
        "parking lot with asphalt surface",
        "bridge with steel and concrete"
    ]
    
    # Labels détaillés pour matériaux (enrichis pour meilleure détection)
    MATERIAL_LABELS = [
        "rough concrete wall texture",
        "red brick surface pattern",
        "shiny metal beam structure",
        "natural wood plank grain",
        "transparent glass window panel",
        "grey stone pavement blocks",
        "black asphalt road surface",
        "green grass field ground",
        "brown soil dirt ground",
        "rocky terrain surface",
        "rusty metal corroded surface",
        "painted wall smooth surface",
        "ceramic roof tiles pattern",
        "gravel path small stones",
        "polished marble floor smooth",
        "weathered rusty metal texture",
        "smooth concrete finished surface",
        "rough natural stone texture",
        "polished wood glossy finish",
        "aged weathered material patina",
        "steel industrial metal structure",
        "aluminum modern metal finish",
        "copper metal surface",
        "granite stone texture",
        "limestone wall texture"
    ]
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialise l'analyseur avec les modèles CLIP et Phi-1.5
        
        Args:
            device: 'cuda' ou 'cpu'
        """
        self.device = device
        self.clip_model = None
        self.clip_processor = None
        self.phi_model = None
        self.phi_tokenizer = None
        
        print(f"🔧 Initialisation TexturePBRAnalyzer sur {device.upper()}")
        
        if TRANSFORMERS_AVAILABLE:
            self._load_models()
        else:
            print("❌ Transformers non disponible - Fonctionnalités limitées")
    
    def _load_models(self):
        """Charge les modèles CLIP et Phi-1.5"""
        try:
            # Chargement CLIP (structure HuggingFace cache avec snapshots)
            clip_path = Path(__file__).parent / "models--openai--clip-vit-base-patch32"
            if clip_path.exists():
                # Chercher dans snapshots/ (structure cache HuggingFace)
                snapshots_dir = clip_path / "snapshots"
                if snapshots_dir.exists():
                    # Prendre le premier snapshot disponible
                    snapshot_dirs = list(snapshots_dir.iterdir())
                    if snapshot_dirs:
                        clip_model_path = snapshot_dirs[0]
                        print(f"📦 Chargement CLIP depuis {clip_model_path}")
                        self.clip_model = CLIPModel.from_pretrained(str(clip_model_path)).to(self.device)  # type: ignore
                        self.clip_processor = CLIPProcessor.from_pretrained(str(clip_model_path))  # type: ignore
                        self.clip_model.eval()
                        print("✅ CLIP chargé avec succès")
                    else:
                        raise FileNotFoundError("Aucun snapshot CLIP trouvé")
                else:
                    # Essayer le chemin direct
                    print(f"📦 Chargement CLIP depuis {clip_path}")
                    self.clip_model = CLIPModel.from_pretrained(str(clip_path)).to(self.device)  # type: ignore
                    self.clip_processor = CLIPProcessor.from_pretrained(str(clip_path))  # type: ignore
                    self.clip_model.eval()
                    print("✅ CLIP chargé avec succès")
            else:
                print("⚠️ CLIP local non trouvé - Téléchargement depuis HuggingFace...")
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)  # type: ignore
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")  # type: ignore
                self.clip_model.eval()
            
            # Chargement Phi-1.5 (petit modèle de langage)
            phi_path = Path(__file__).parent / "phi-1_5"
            if phi_path.exists():
                print(f"📦 Chargement Phi-1.5 depuis {phi_path}")
                self.phi_tokenizer = AutoTokenizer.from_pretrained(str(phi_path), trust_remote_code=True)  # type: ignore
                self.phi_model = AutoModelForCausalLM.from_pretrained(  # type: ignore
                    str(phi_path),
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    trust_remote_code=True
                ).to(self.device)  # type: ignore
                self.phi_model.eval()
                print("✅ Phi-1.5 chargé avec succès")
            else:
                print("⚠️ Phi-1.5 non trouvé - Fonctionnalité de génération désactivée")
                
        except Exception as e:
            print(f"❌ Erreur chargement modèles: {e}")
    
    def analyze_image_clip(self, image: Image.Image) -> Dict[str, float]:
        """
        Analyse une image avec CLIP pour classifier la scène
        
        Args:
            image: Image PIL à analyser
            
        Returns:
            Dict avec scores de classification pour chaque type de scène
        """
        if self.clip_model is None or self.clip_processor is None:
            return {}
        
        try:
            with torch.no_grad():
                # Préparation de l'image et des labels
                inputs = self.clip_processor(  # type: ignore
                    text=self.SCENE_LABELS,
                    images=image,
                    return_tensors="pt",  # type: ignore
                    padding=True  # type: ignore
                ).to(self.device)
                
                # Prédiction
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
                
                # Création du dictionnaire de résultats
                results = {label: float(prob) for label, prob in zip(self.SCENE_LABELS, probs)}
                
                return results
                
        except Exception as e:
            print(f"❌ Erreur analyse CLIP: {e}")
            return {}
    
    def analyze_materials_clip(self, image: Image.Image) -> Dict[str, float]:
        """
        Analyse les matériaux présents dans l'image avec CLIP
        
        Args:
            image: Image PIL à analyser
            
        Returns:
            Dict avec scores pour chaque type de matériau
        """
        if self.clip_model is None or self.clip_processor is None:
            return {}
        
        try:
            with torch.no_grad():
                inputs = self.clip_processor(  # type: ignore
                    text=self.MATERIAL_LABELS,
                    images=image,
                    return_tensors="pt",  # type: ignore
                    padding=True  # type: ignore
                ).to(self.device)
                
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
                
                results = {label: float(prob) for label, prob in zip(self.MATERIAL_LABELS, probs)}
                
                return results
                
        except Exception as e:
            print(f"❌ Erreur analyse matériaux: {e}")
            return {}
    
    def generate_texture_recommendations(self, scene_type: str, materials: List[str]) -> Dict:
        """
        Génère des recommandations de textures PBR avec Phi-1.5
        
        Args:
            scene_type: Type de scène détecté
            materials: Liste de matériaux détectés
            
        Returns:
            Dict avec recommandations de textures
        """
        if self.phi_model is None or self.phi_tokenizer is None:
            # Fallback: recommandations basiques sans IA
            return self._generate_basic_recommendations(scene_type, materials)
        
        try:
            # Prompt pour Phi-1.5
            prompt = f"""You are a 3D rendering expert. Based on this scene analysis:
Scene Type: {scene_type}
Detected Materials: {', '.join(materials)}

List the essential PBR textures needed (albedo, normal, roughness, metallic, AO) for realistic rendering.
Format: material_name: [texture types]

Textures:"""
            
            inputs = self.phi_tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.phi_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
            
            response = self.phi_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse la réponse
            recommendations = self._parse_phi_response(response)
            
            return recommendations
            
        except Exception as e:
            print(f"⚠️ Erreur génération Phi-1.5: {e} - Utilisation fallback")
            return self._generate_basic_recommendations(scene_type, materials)
    
    def _generate_basic_recommendations(self, scene_type: str, materials: List[str]) -> Dict:
        """
        Génère des recommandations basiques sans IA (fallback)
        """
        recommendations = {
            "scene_type": scene_type,
            "detected_materials": materials,
            "pbr_textures_needed": []
        }
        
        # Mapping simple basé sur les matériaux détectés
        texture_map = {
            "concrete": ["concrete_albedo", "concrete_normal", "concrete_roughness", "concrete_ao"],
            "brick": ["brick_albedo", "brick_normal", "brick_roughness", "brick_ao"],
            "metal": ["metal_albedo", "metal_normal", "metal_metallic", "metal_roughness"],
            "wood": ["wood_albedo", "wood_normal", "wood_roughness", "wood_ao"],
            "stone": ["stone_albedo", "stone_normal", "stone_roughness", "stone_ao"],
            "asphalt": ["asphalt_albedo", "asphalt_normal", "asphalt_roughness"],
            "grass": ["grass_albedo", "grass_normal", "grass_ao"],
            "soil": ["soil_albedo", "soil_normal", "soil_roughness", "soil_ao"]
        }
        
        for material in materials:
            material_lower = material.lower()
            for key, textures in texture_map.items():
                if key in material_lower:
                    recommendations["pbr_textures_needed"].extend(textures)
        
        # Déduplique
        recommendations["pbr_textures_needed"] = list(set(recommendations["pbr_textures_needed"]))
        
        return recommendations
    
    def _parse_phi_response(self, response: str) -> Dict:
        """Parse la réponse de Phi-1.5"""
        # Implémentation simple - peut être améliorée
        lines = response.split('\n')
        textures = []
        
        for line in lines:
            if ':' in line and any(keyword in line.lower() for keyword in ['albedo', 'normal', 'roughness', 'metallic', 'ao']):
                textures.append(line.strip())
        
        return {
            "ai_generated": True,
            "recommended_textures": textures,
            "raw_response": response
        }
    
    def analyze_scene_batch(self, images: List[Image.Image]) -> Dict:
        """
        Analyse un lot d'images pour générer un rapport complet
        
        Args:
            images: Liste d'images PIL
            
        Returns:
            Rapport complet avec recommandations de textures
        """
        print(f"🔍 Analyse de {len(images)} images...")
        
        all_scene_scores = []
        all_material_scores = []
        
        for i, img in enumerate(images):
            print(f"  📸 Image {i+1}/{len(images)}")
            
            # Analyse scène
            scene_scores = self.analyze_image_clip(img)
            all_scene_scores.append(scene_scores)
            
            # Analyse matériaux
            material_scores = self.analyze_materials_clip(img)
            all_material_scores.append(material_scores)
        
        # Agrégation des résultats
        avg_scene_scores = self._average_scores(all_scene_scores)
        avg_material_scores = self._average_scores(all_material_scores)
        
        # Identification du type de scène dominant
        dominant_scene = max(avg_scene_scores.items(), key=lambda x: x[1])[0]
        
        # Top 5 matériaux détectés
        top_materials = sorted(avg_material_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        material_names = [mat[0] for mat in top_materials]
        
        # Génération des recommandations
        recommendations = self.generate_texture_recommendations(dominant_scene, material_names)
        
        report = {
            "num_images_analyzed": len(images),
            "dominant_scene_type": dominant_scene,
            "scene_confidence": float(avg_scene_scores[dominant_scene]),
            "top_materials": [
                {"material": mat, "confidence": float(score)}
                for mat, score in top_materials
            ],
            "texture_recommendations": recommendations,
            "download_links": self._generate_download_links(recommendations)
        }
        
        return report
    
    def _average_scores(self, scores_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Calcule la moyenne des scores"""
        if not scores_list:
            return {}
        
        all_keys = scores_list[0].keys()
        averaged = {}
        
        for key in all_keys:
            values = [scores[key] for scores in scores_list if key in scores]
            averaged[key] = np.mean(values) if values else 0.0
        
        return averaged
    
    def _generate_download_links(self, recommendations: Dict) -> List[Dict]:
        """
        Génère des liens vers des bibliothèques de textures PBR gratuites
        """
        links = []
        
        # Sources gratuites de textures PBR
        sources = [
            {
                "name": "Poly Haven",
                "url": "https://polyhaven.com/textures",
                "description": "Textures PBR 100% gratuites en haute résolution",
                "license": "CC0 (Domaine public)"
            },
            {
                "name": "ambientCG",
                "url": "https://ambientcg.com/",
                "description": "Grande collection de textures PBR gratuites",
                "license": "CC0 (Domaine public)"
            },
            {
                "name": "3D Textures",
                "url": "https://3dtextures.me/",
                "description": "Textures PBR gratuites pour usage commercial",
                "license": "CC0 (Domaine public)"
            },
            {
                "name": "FreePBR",
                "url": "https://freepbr.com/",
                "description": "Textures PBR 100% gratuites",
                "license": "CC0 (Domaine public)"
            }
        ]
        
        # Ajoute des recommandations de recherche
        textures_needed = recommendations.get("pbr_textures_needed", [])
        
        for source in sources:
            link_info = source.copy()
            keywords = list(set([
                tex.split('_')[0] for tex in textures_needed if '_' in tex
            ]))
            link_info["search_keywords"] = keywords  # type: ignore
            links.append(link_info)
        
        return links


# Fonction utilitaire pour utilisation directe
def analyze_images_for_pbr(image_paths: List[Path], device='cuda') -> Dict:
    """
    Fonction helper pour analyser des images et obtenir des recommandations PBR
    
    Args:
        image_paths: Liste de chemins vers les images
        device: 'cuda' ou 'cpu'
        
    Returns:
        Rapport d'analyse complet
    """
    analyzer = TexturePBRAnalyzer(device=device)
    
    # Chargement des images
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"⚠️ Impossible de charger {path}: {e}")
    
    if not images:
        return {"error": "Aucune image valide fournie"}
    
    # Analyse
    report = analyzer.analyze_scene_batch(images)
    
    return report


if __name__ == "__main__":
    # Test du module
    print("🧪 Test du module TexturePBRAnalyzer")
    
    analyzer = TexturePBRAnalyzer()
    
    if analyzer.clip_model:
        print("✅ Module prêt à l'emploi")
    else:
        print("⚠️ CLIP non disponible - Installation transformers requise")
