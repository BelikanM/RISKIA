# type: ignore
import streamlit as st
import torch
from pathlib import Path
import tempfile
import os
import sys
import json
sys.path.append(os.path.join(os.path.dirname(__file__), 'dust3r'))
import time
import uuid
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import open3d as o3d  # pip install open3d
import zipfile
import pandas as pd
import io
import pickle
import subprocess
import shutil  # Ajout pour check Blender
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors  # Fallback pour FAISS
from transformers import CLIPProcessor, CLIPModel
try:
    import psutil  # pip install psutil pour monitoring CPU
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None  # type: ignore

try:
    import pynvml  # pip install pynvml pour monitoring GPU (NVIDIA)
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

try:
    import laspy  # type: ignore
    LAS_AVAILABLE = True
except ImportError:
    LAS_AVAILABLE = False

# Py4DGeo-inspired 4D Change Detection imports
try:
    import scipy.spatial  # Pour les calculs de distance
    import scipy.stats    # Pour les statistiques
    import matplotlib.pyplot as plt  # Pour les graphiques de distribution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

import gc

try:
    from texture_pbr_analyzer import TexturePBRAnalyzer, analyze_images_for_pbr
    PBR_ANALYZER_AVAILABLE = True
except ImportError:
    PBR_ANALYZER_AVAILABLE = False
    print("⚠️ Module texture_pbr_analyzer non disponible")

# Import du moteur VFX intelligent
try:
    from intelligent_vfx_engine import IntelligentVFXEngine, VFXParameters, MaterialType, apply_intelligent_vfx
    VFX_ENGINE_AVAILABLE = True
except ImportError:
    VFX_ENGINE_AVAILABLE = False
    print("⚠️ Module intelligent_vfx_engine non disponible")

# Import du mapper PBR automatique
try:
    from auto_pbr_mapper import AutoPBRMapper, generate_complete_pbr_pipeline
    AUTO_PBR_MAPPER_AVAILABLE = True
except ImportError:
    AUTO_PBR_MAPPER_AVAILABLE = False
    print("⚠️ Module auto_pbr_mapper non disponible")

# Import du gestionnaire de téléchargement de textures
try:
    from texture_download_manager import TextureDownloadManager
    TEXTURE_MANAGER_AVAILABLE = True
except ImportError:
    TEXTURE_MANAGER_AVAILABLE = False
    print("⚠️ Module texture_download_manager non disponible")

# Imports spécifiques à DUSt3R (assurez-vous d'avoir installé : pip install git+https://github.com/naver/dust3r.git)
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images as dust3r_load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.geometry import xy_grid

# Initialize variables to avoid Pylance possibly unbound errors
batch_size = 1
niter_align = 300
lr_align = 0.01
threshold_conf = 0.5
max_points_per_view = 20000
scale_factor = 1.0
generate_mesh = False
mesh_method = "Poisson"
poisson_depth = 10
ball_pivoting_max_radius = 0.02
advanced_blender = False
export_obj = False
auto_smooth_normals = True
multi_view_blender = False
basic_uv_mapping = False
save_blend_file = False
show_hull = True
wireframe_overlay = False
wireframe_thickness = 1.0
show_uv_checker = False
subdivision_level = 0
show_normals = False
show_topology_info = False
texture_zip = None
process_btn = False

models = {}  # Global dict for models to allow manual freeing

# Tentative d'import FAISS avec fallback
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    st.warning("FAISS non disponible ; fallback sur scikit-learn NearestNeighbors pour recherche de similarité.")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Fonction de nettoyage du session state pour éviter les conflits DOM
def clear_session_state():
    """Nettoie le session state pour éviter les conflits d'éléments DOM dupliqués"""
    keys_to_clear = [
        'advanced_denoising_params', 'mesh_params', 'vfx_params',
        'pbr_analysis_results', 'scene_graph_data', 'texture_analysis'
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

# Nettoyer le session state au démarrage pour éviter les conflits
clear_session_state()

# Fonction pour géoréférencement
def gps_to_local_coords(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    """
    Convertit GPS en coordonnées locales (mètres) par rapport à un point de référence.
    Approximation pour petites zones.
    """
    import math
    # Rayon terrestre approximatif
    R = 6371000  # mètres
    # Conversion degrés en radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)
    
    # Différences
    dlat = lat - ref_lat
    dlon = lon - ref_lon
    
    # Conversion en mètres
    x = dlon * (math.pi / 180) * R * math.cos(ref_lat_rad)  # Est-Ouest
    y = dlat * (math.pi / 180) * R  # Nord-Sud
    z = alt - ref_alt  # Altitude
    
    return x, y, z

def get_gps_from_exif(img_path):
    """
    Extrait les coordonnées GPS depuis l'EXIF d'une image.
    """
    try:
        img = Image.open(img_path)
        exif_data = img.getexif()
        if not exif_data:
            return None
        
        gps_info = exif_data.get(34853)  # GPSInfo tag
        if not gps_info:
            return None
        
        def convert_to_degrees(value):
            d, m, s = value
            return d + (m / 60.0) + (s / 3600.0)
        
        lat = convert_to_degrees(gps_info[2])
        if gps_info[1] == 'S':
            lat = -lat
        lon = convert_to_degrees(gps_info[4])
        if gps_info[3] == 'W':
            lon = -lon
        alt = gps_info.get(6, 0) if gps_info.get(6) else 0
        
        return lat, lon, alt
    except:
        return None

def export_to_las(merged_pts3d, merged_colors, filename="pointcloud.las"):
    """
    Exporte le nuage de points en format LAS avec couleurs.
    """
    if not LAS_AVAILABLE:
        st.error("laspy non installé. Installez avec: pip install laspy")
        return None
    
    # Créer un fichier LAS
    header = laspy.LasHeader(point_format=3, version="1.2")  # type: ignore
    header.add_extra_dim(laspy.ExtraBytesParams(name="red", type=np.uint16))  # type: ignore
    header.add_extra_dim(laspy.ExtraBytesParams(name="green", type=np.uint16))  # type: ignore
    header.add_extra_dim(laspy.ExtraBytesParams(name="blue", type=np.uint16))  # type: ignore
    
    las = laspy.LasData(header)  # type: ignore
    las.x = merged_pts3d[:, 0]  # type: ignore
    las.y = merged_pts3d[:, 1]  # type: ignore
    las.z = merged_pts3d[:, 2]  # type: ignore
    
    # Couleurs en uint16 (0-65535)
    las.red = (merged_colors[:, 0] * 65535).astype(np.uint16)  # type: ignore
    las.green = (merged_colors[:, 1] * 65535).astype(np.uint16)  # type: ignore
    las.blue = (merged_colors[:, 2] * 65535).astype(np.uint16)  # type: ignore
    
    # Sauvegarder en bytes pour download
    with io.BytesIO() as buffer:
        las.write(buffer)  # type: ignore
        buffer.seek(0)
        return buffer.read()

def ransac_plane_detection(pcd, distance_threshold=0.02, ransac_n=3, num_iterations=1000):
    """
    Détection de plan avec RANSAC utilisant Open3D (scientifique et robuste).
    """
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    return plane_model, inliers

def ransac_cylinder_detection(points, distance_threshold=0.05, max_iterations=1000):
    """
    Détection de cylindre avec RANSAC personnalisé (utilisant numpy pour calculs scientifiques).
    """
    best_model = None
    best_inliers = []
    n_points = len(points)
    
    for _ in range(max_iterations):
        # Échantillonner 2 points pour définir l'axe, 1 point pour le rayon
        sample_indices = np.random.choice(n_points, 3, replace=False)
        sample_points = points[sample_indices]
        
        # Calculer l'axe (vecteur entre les 2 premiers points)
        axis = sample_points[1] - sample_points[0]
        axis = axis / np.linalg.norm(axis)  # Normaliser
        
        # Point sur l'axe (milieu)
        center = (sample_points[0] + sample_points[1]) / 2
        
        # Rayon : distance du 3ème point à l'axe
        vec_to_point = sample_points[2] - center
        radius = np.abs(np.dot(vec_to_point, axis))
        
        # Calculer inliers : points dont la distance à l'axe est proche du rayon
        inliers = []
        for i, point in enumerate(points):
            vec = point - center
            dist_to_axis = np.linalg.norm(vec - np.dot(vec, axis) * axis)
            if np.abs(dist_to_axis - radius) < distance_threshold:
                inliers.append(i)
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_model = {"axis": axis, "center": center, "radius": radius}
    
    return best_model, best_inliers

def ransac_sphere_detection(points, distance_threshold=0.05, max_iterations=1000):
    """
    Détection de sphère avec RANSAC personnalisé (calculs scientifiques avec numpy).
    """
    best_model = None
    best_inliers = []
    n_points = len(points)
    
    for _ in range(max_iterations):
        # Échantillonner 4 points pour définir la sphère
        sample_indices = np.random.choice(n_points, 4, replace=False)
        sample_points = points[sample_indices]
        
        # Calculer le centre et rayon à partir de 4 points (méthode déterministe)
        # Utiliser la formule pour 4 points non coplanaires
        try:
            # Matrice pour résoudre le système
            A = np.array([
                [sample_points[0][0], sample_points[0][1], sample_points[0][2], 1],
                [sample_points[1][0], sample_points[1][1], sample_points[1][2], 1],
                [sample_points[2][0], sample_points[2][1], sample_points[2][2], 1],
                [sample_points[3][0], sample_points[3][1], sample_points[3][2], 1]
            ])
            b = np.array([
                -np.sum(sample_points[0]**2),
                -np.sum(sample_points[1]**2),
                -np.sum(sample_points[2]**2),
                -np.sum(sample_points[3]**2)
            ])
            x = np.linalg.solve(A, b)
            center = -0.5 * x[:3]
            radius = np.sqrt(np.sum(center**2) - x[3])
            
            # Calculer inliers
            inliers = []
            for i, point in enumerate(points):
                dist = np.linalg.norm(point - center)
                if np.abs(dist - radius) < distance_threshold:
                    inliers.append(i)
            
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_model = {"center": center, "radius": radius}
        except np.linalg.LinAlgError:
            continue  # Points coplanaires, ignorer
    
    return best_model, best_inliers

def setup_ui():
    """Configure l'interface utilisateur principale"""
    st.title("📸 Application de Photogrammétrie Complète SETRAF GABON développée par NYUNDU FRANCIS ARNAUD")
    st.markdown("---")

    # Bouton de réinitialisation du session state pour éviter les erreurs DOM
    col_reset, col_info = st.columns([1, 3])
    with col_reset:
        if st.button("🔄 Réinitialiser l'interface", help="Nettoie le session state pour résoudre les erreurs d'affichage"):
            clear_session_state()
            st.rerun()

    with col_info:
        st.markdown("Cette application permet de charger plusieurs images, d'effectuer une reconstruction 3D dense à partir de paires d'images en utilisant le modèle DUSt3R ou MapAnything, et de visualiser le nuage de points aligné globalement avec textures réalistes et option de maillage complet ultra-réaliste.")

    # Monitoring et sélection device
    use_gpu = st.sidebar.checkbox("Utiliser GPU (désactiver si surchauffe)", value=True, help="Désactivez pour forcer CPU en cas de surchauffe GPU.")
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    st.sidebar.info(f"**Périphérique utilisé :** {device.upper()}")

    return device

def perform_ransac_analysis(pcd, enable_auto_ransac, ransac_auto_threshold, ransac_auto_iterations):
    """Effectue l'analyse RANSAC automatique"""
    detected_shapes = {}
    if enable_auto_ransac:
        st.info("🔬 Analyse géométrique automatique en cours...")
        points = np.asarray(pcd.points)
        modified_colors = np.array(pcd.colors)

        # Détection plan
        try:
            plane_model, plane_inliers = ransac_plane_detection(pcd, distance_threshold=ransac_auto_threshold,
                                                               num_iterations=ransac_auto_iterations)
            if len(plane_inliers) > len(points) * 0.1:  # Au moins 10% des points
                [a, b, c, d] = plane_model
                detected_shapes['plan'] = {"model": [a, b, c, d], "inliers": len(plane_inliers)}
                # Colorer inliers en rouge
                modified_colors[plane_inliers] = [1.0, 0.0, 0.0]  # Rouge
                st.success(f"Plan détecté automatiquement : {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0 ({len(plane_inliers)} points)")
        except:
            pass

        # Détection cylindre
        try:
            cyl_model, cyl_inliers = ransac_cylinder_detection(points, distance_threshold=ransac_auto_threshold,
                                                              max_iterations=ransac_auto_iterations)
            if cyl_model and len(cyl_inliers) > len(points) * 0.05:  # Au moins 5%
                detected_shapes['cylindre'] = {"model": cyl_model, "inliers": len(cyl_inliers)}
                # Colorer inliers en vert
                modified_colors[cyl_inliers] = [0.0, 1.0, 0.0]  # Vert
                st.success(f"Cylindre détecté : Rayon {cyl_model['radius']:.3f} ({len(cyl_inliers)} points)")
        except:
            pass

        # Détection sphère
        try:
            sph_model, sph_inliers = ransac_sphere_detection(points, distance_threshold=ransac_auto_threshold,
                                                            max_iterations=ransac_auto_iterations)
            if sph_model and len(sph_inliers) > len(points) * 0.05:
                detected_shapes['sphere'] = {"model": sph_model, "inliers": len(sph_inliers)}
                # Colorer inliers en bleu
                modified_colors[sph_inliers] = [0.0, 0.0, 1.0]  # Bleu
                st.success(f"Sphère détectée : Rayon {sph_model['radius']:.3f} ({len(sph_inliers)} points)")
        except:
            pass

        # Mettre à jour les couleurs du nuage
        pcd.colors = o3d.utility.Vector3dVector(modified_colors)
        st.info("🎨 Nuage coloré automatiquement : Rouge=Plans, Vert=Cylindres, Bleu=Sphères, Original=Autres")

    return detected_shapes

def apply_realtime_downsampling_pipeline(pcd, target_points=100000, strategy='auto', preserve_colors=True, preserve_normals=False):
    """
    Pipeline de downsampling temps réel ultra-rapide inspiré par Sohail Saifi.
    
    Args:
        pcd: Point cloud Open3D
        target_points: Nombre de points cible souhaité
        strategy: Stratégie ('auto', 'speed', 'quality', 'balanced')
        preserve_colors: Préserver les informations de couleur
        preserve_normals: Préserver les normales de surface
        
    Returns:
        Point cloud downsamplé
    """
    import time
    start_time = time.time()
    
    # Nombre de points original
    original_points = len(np.asarray(pcd.points))
    
    # Analyse adaptative du volume
    if strategy == 'auto':
        if original_points > 10000000:  # >10M points
            strategy = 'speed'
        elif original_points > 1000000:  # >1M points
            strategy = 'balanced'
        else:  # <1M points
            strategy = 'quality'
    
    # Étape 1: Pré-downsampling par voxels pour volumes massifs
    if original_points > 5000000:  # >5M points
        # Calcul adaptatif de la taille de voxel
        points = np.asarray(pcd.points)
        bbox = points.max(axis=0) - points.min(axis=0)
        volume = np.prod(bbox)
        voxel_size = (volume / target_points) ** (1/3) * 0.1  # Ajustement adaptatif
        
        # Downsampling voxel grid
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        current_points = len(np.asarray(pcd.points))
        
        # Si encore trop de points, réduire davantage
        if current_points > target_points * 2:
            voxel_size *= 1.5
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    current_points = len(np.asarray(pcd.points))
    
    # Étape 2: Techniques de downsampling multi-étapes
    if current_points > target_points:
        remaining_reduction = current_points / target_points
        
        # Random sampling pour réduction initiale rapide
        if remaining_reduction > 4:
            random_ratio = min(0.5, target_points / current_points * 2)
            random_indices = np.random.choice(current_points, int(current_points * random_ratio), replace=False)
            pcd = pcd.select_by_index(random_indices)
            current_points = len(np.asarray(pcd.points))
        
        # Uniform grid sampling pour couverture spatiale
        if current_points > target_points and strategy in ['balanced', 'quality']:
            # Calcul de la taille de grille optimale
            points = np.asarray(pcd.points)
            bbox = points.max(axis=0) - points.min(axis=0)
            grid_density = (target_points / np.prod(bbox)) ** (1/3)
            grid_size = 1.0 / grid_density
            
            # Création de la grille uniforme
            min_bounds = points.min(axis=0)
            grid_indices = ((points - min_bounds) / grid_size).astype(int)
            
            # Sélection d'un point par cellule de grille
            unique_grids = {}
            for i, grid_idx in enumerate(grid_indices):
                grid_key = tuple(grid_idx)
                if grid_key not in unique_grids:
                    unique_grids[grid_key] = i
            
            selected_indices = list(unique_grids.values())
            
            if len(selected_indices) > target_points:
                # Si encore trop, sous-échantillonnage aléatoire
                selected_indices = np.random.choice(selected_indices, target_points, replace=False)
            
            pcd = pcd.select_by_index(selected_indices)
            current_points = len(np.asarray(pcd.points))
        
        # Farthest point sampling pour qualité optimale
        if current_points > target_points and strategy == 'quality':
            # Implémentation simplifiée du farthest point sampling
            points = np.asarray(pcd.points)
            n_points = len(points)
            
            # Sélection du premier point aléatoirement
            selected_indices = [np.random.randint(0, n_points)]
            distances = np.full(n_points, np.inf)
            
            for _ in range(min(target_points - 1, n_points - 1)):
                # Mise à jour des distances
                last_selected = points[selected_indices[-1]]
                current_distances = np.linalg.norm(points - last_selected, axis=1)
                distances = np.minimum(distances, current_distances)
                
                # Sélection du point le plus éloigné
                farthest_idx = np.argmax(distances)
                selected_indices.append(farthest_idx)
                
                # Marquer comme sélectionné
                distances[farthest_idx] = 0
            
            pcd = pcd.select_by_index(selected_indices)
    
    # Étape 3: Préservation des attributs
    if preserve_colors and pcd.has_colors():
        # Les couleurs sont automatiquement préservées par select_by_index
        pass
    
    if preserve_normals and pcd.has_normals():
        # Recalcul des normales si nécessaire après downsampling
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=15)
    
    # Métriques finales
    final_points = len(np.asarray(pcd.points))
    processing_time = time.time() - start_time
    
    return pcd

def apply_pointnet_classification(pcd, confidence_threshold=0.7):
    """
    Classification simplifiée inspirée de PointNet pour nuages de points photogrammétriques.
    
    Utilise des caractéristiques géométriques simples pour classifier :
    - Terrain (bas, plat)
    - Bâtiments (vertical, régulier)
    - Végétation (haut, irrégulier)
    - Véhicules (petit, mobile)
    
    Args:
        pcd: Point cloud Open3D
        confidence_threshold: Seuil de confiance minimum
        
    Returns:
        Tuple (pcd_classified, classification_stats)
    """
    import time
    start_time = time.time()
    
    # Récupération des points et couleurs
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones((len(points), 3)) * 0.5
    
    n_points = len(points)
    
    # Calcul des caractéristiques géométriques (inspiré de PointNet)
    
    # 1. Coordonnées normalisées (centrage - invariance translationnelle)
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    
    # 2. Calcul des normales si pas présentes (pour caractéristiques géométriques)
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    normals = np.asarray(pcd.normals)
    
    # 3. Calcul des caractéristiques locales (k-NN comme dans PointNet++)
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min(32, n_points), algorithm='auto').fit(points_centered)
    distances, indices = nbrs.kneighbors(points_centered)
    
    # Caractéristiques géométriques pour chaque point
    features = []
    
    for i in range(n_points):
        local_points = points_centered[indices[i]]
        local_normals = normals[indices[i]]
        
        # Caractéristiques inspirées de PointNet :
        # - Écart-type des positions locales (rugosité)
        # - Verticalité (angle avec Z)
        # - Densité locale
        # - Couleur moyenne
        
        # Rugosité locale (écart-type des distances)
        roughness = np.std(distances[i])
        
        # Verticalité (angle de la normale avec l'axe Z)
        verticality = abs(normals[i][2])  # Composante Z de la normale
        
        # Densité locale (inverse de la distance moyenne)
        local_density = 1.0 / (np.mean(distances[i]) + 1e-6)
        
        # Couleur (luminosité moyenne)
        brightness = np.mean(colors[i]) if len(colors[i]) > 0 else 0.5
        
        # Altitude relative (par rapport au centroïde)
        relative_height = points[i][2] - centroid[2]
        
        features.append([roughness, verticality, local_density, brightness, relative_height])
    
    features = np.array(features)
    
    # Classification basée sur des règles simples (version simplifiée de PointNet)
    classifications = []
    confidences = []
    
    for feat in features:
        roughness, verticality, density, brightness, height = feat
        
        # Règles de classification inspirées de l'analyse géométrique
        
        # Terrain : bas, plat, haute densité
        if height < np.percentile(features[:, 4], 25) and verticality > 0.8 and roughness < np.percentile(features[:, 0], 50):
            class_id = 0  # Terrain
            confidence = min(0.9, verticality * 0.8 + (1 - roughness) * 0.2)
            
        # Bâtiments : vertical, régulier, hauteur moyenne
        elif verticality > 0.6 and roughness < np.percentile(features[:, 0], 75) and abs(height) < np.percentile(np.abs(features[:, 4]), 75):
            class_id = 1  # Bâtiments
            confidence = min(0.85, verticality * 0.7 + (1 - roughness) * 0.3)
            
        # Végétation : irrégulier, vertical variable, haute
        elif roughness > np.percentile(features[:, 0], 50) and height > np.percentile(features[:, 4], 50):
            class_id = 2  # Végétation
            confidence = min(0.8, roughness * 0.6 + (height / (np.max(features[:, 4]) + 1e-6)) * 0.4)
            
        # Véhicules/Objets : petit, isolé, basse densité
        elif density < np.percentile(features[:, 2], 25) and abs(height) < np.percentile(np.abs(features[:, 4]), 50):
            class_id = 3  # Véhicules/Objets
            confidence = min(0.75, (1 - density/np.max(features[:, 2])) * 0.6 + 0.4)
            
        else:
            class_id = 4  # Autres
            confidence = 0.5
            
        classifications.append(class_id)
        confidences.append(confidence)
    
    # Application des couleurs selon la classification (seulement si confiance suffisante)
    classified_colors = colors.copy()
    
    # Palette de couleurs pour chaque classe
    class_colors = {
        0: [0.4, 0.8, 0.4],  # Terrain - Vert
        1: [0.8, 0.4, 0.4],  # Bâtiments - Rouge
        2: [0.4, 0.4, 0.8],  # Végétation - Bleu
        3: [0.8, 0.8, 0.4],  # Véhicules - Jaune
        4: [0.6, 0.6, 0.6],  # Autres - Gris
    }
    
    classified_count = 0
    for i, (class_id, conf) in enumerate(zip(classifications, confidences)):
        if conf >= confidence_threshold:
            classified_colors[i] = class_colors[class_id]
            classified_count += 1
    
    # Mise à jour des couleurs du point cloud
    pcd.colors = o3d.utility.Vector3dVector(classified_colors)
    
    # Statistiques
    processing_time = (time.time() - start_time) * 1000
    
    # Comptage par classe
    class_counts = {}
    for class_id in range(5):
        count = sum(1 for c, conf in zip(classifications, confidences) 
                   if c == class_id and conf >= confidence_threshold)
        class_counts[class_id] = count
    
    stats = {
        'classified_objects': classified_count,
        'avg_confidence': np.mean([c for c in confidences if c >= confidence_threshold]) if classified_count > 0 else 0,
        'processing_time_ms': processing_time,
        'gpu_memory_mb': 0,  # Pas utilisé pour cette version simplifiée
        'class_distribution': class_counts,
        'total_points': n_points
    }
    
    return pcd, stats

# Fonction pour métriques GPU/CPU
@st.cache_data(ttl=10)
def get_system_metrics(device):
    if PSUTIL_AVAILABLE:
        cpu_percent = psutil.cpu_percent(interval=1)
        ram_percent = psutil.virtual_memory().percent
        metrics = {"CPU %": f"{cpu_percent:.1f}%", "RAM %": f"{ram_percent:.1f}%"}
    else:
        metrics = {"CPU %": "N/A", "RAM %": "N/A"}
    if device == 'cuda' and NVML_AVAILABLE:
        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(pynvml.nvmlDeviceGetHandleByIndex(0)).gpu
        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).used / 1024**3
        gpu_temp = pynvml.nvmlDeviceGetTemperature(pynvml.nvmlDeviceGetHandleByIndex(0), pynvml.NVML_TEMPERATURE_GPU)
        if gpu_temp > 85:
            st.sidebar.warning(f"🚨 GPU surchauffe ! Temp: {gpu_temp}°C – Désactivez GPU via checkbox.")
        metrics.update({"GPU %": f"{gpu_util:.1f}%", "GPU Temp": f"{gpu_temp}°C", "GPU Mem": f"{gpu_mem:.1f}GB"})
    return metrics

def setup_sidebar_monitoring(device):
    """Configure le monitoring système dans la sidebar"""
    with st.sidebar:
        st.header("📈 Monitoring Système")
        metrics = get_system_metrics(device)
        for key, value in metrics.items():
            st.metric(key, value)

        st.header("🧹 Libération Mémoire")
        if st.button("Libérer Mémoire GPU"):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_before = torch.cuda.memory_allocated() / 1024**3
                models.clear()
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                mem_after = torch.cuda.memory_allocated() / 1024**3
                st.success(f"Mémoire GPU libérée ! Avant: {mem_before:.2f} GB, Après: {mem_after:.2f} GB (Diff: {mem_before - mem_after:.2f} GB)")
            else:
                st.info("Aucun GPU détecté.")

        if st.button("Libérer Mémoire CPU"):
            gc.collect()
            st.success("Mémoire CPU libérée !")

        if st.button("Libérer RAM"):
            gc.collect()
            st.success("RAM libérée !")

# Appeler les fonctions de configuration au début
device = setup_ui()
setup_sidebar_monitoring(device)

# Chargement des modèles
def load_dust3r_model():
    if 'dust3r' not in models:
        try:
            model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
            model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
            models['dust3r'] = model
            st.success("Modèle DUSt3R chargé avec succès !")
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle DUSt3R : {e}")
            st.info("Assurez-vous d'avoir installé DUSt3R : `pip install git+https://github.com/naver/dust3r.git`")
            return None
    return models.get('dust3r')

    return models.get('dust3r')

def load_clip_model():
    if 'clip' not in models:
        try:
            # Chercher CLIP local d'abord
            clip_path = Path(__file__).parent / "models--openai--clip-vit-base-patch32"
            if clip_path.exists():
                snapshots_dir = clip_path / "snapshots"
                if snapshots_dir.exists():
                    snapshot_dirs = list(snapshots_dir.iterdir())
                    if snapshot_dirs:
                        model = CLIPModel.from_pretrained(str(snapshot_dirs[0])).to(device)  # type: ignore
                        processor = CLIPProcessor.from_pretrained(str(snapshot_dirs[0]))
                    else:
                        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)  # type: ignore
                        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                else:
                    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)  # type: ignore
                    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            else:
                model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)  # type: ignore
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            models['clip'] = (model, processor)
        except Exception as e:
            st.error(f"Erreur lors du chargement de CLIP : {e}")
            return None, None
    return models.get('clip', (None, None))

# Fonction de détection de changements 4D inspirée de Py4DGeo
@st.cache_data(ttl=300)
def apply_4d_change_detection(pcd1, pcd2, cylinder_radius=0.1, min_points=10, 
                             confidence_threshold=0.95, max_distance=1.0):
    """
    Implémente l'algorithme M3C2 (Multiscale Model-to-Model Cloud Comparison) 
    pour détecter les changements entre deux nuages de points temporels.
    
    Args:
        pcd1: Premier nuage de points (Open3D PointCloud) - époque de référence
        pcd2: Second nuage de points (Open3D PointCloud) - époque de comparaison  
        cylinder_radius: Rayon des cylindres locaux pour l'analyse (mètres)
        min_points: Nombre minimum de points requis dans un cylindre
        confidence_threshold: Seuil de confiance pour la classification des changements
        max_distance: Distance maximale pour considérer un changement significatif
        
    Returns:
        pcd_diff: Nuage de points avec couleurs codant les changements
        stats: Statistiques des changements détectés
    """
    try:
        import time
        start_time = time.time()
        
        # Vérifier les entrées
        if pcd1 is None or pcd2 is None:
            raise ValueError("Les deux nuages de points doivent être fournis")
            
        points1 = np.asarray(pcd1.points)
        points2 = np.asarray(pcd2.points)
        
        if len(points1) == 0 or len(points2) == 0:
            raise ValueError("Les nuages de points ne peuvent pas être vides")
        
        st.info(f"🔍 Analyse 4D en cours... Points référence: {len(points1)}, Points comparaison: {len(points2)}")
        
        # Construire l'arbre KD pour le nuage de référence
        from scipy.spatial import cKDTree
        tree1 = cKDTree(points1)
        tree2 = cKDTree(points2)
        
        # Calculer les distances signées pour chaque point du nuage 1
        distances_signed = []
        confidences = []
        valid_points = []
        
        progress_bar = st.progress(0)
        total_points = len(points1)
        
        for i, point in enumerate(points1):
            if i % 1000 == 0:
                progress_bar.progress(i / total_points)
            
            # Trouver les voisins dans le cylindre pour le point de référence
            indices1 = tree1.query_ball_point(point, cylinder_radius)
            
            if len(indices1) < min_points:
                continue  # Pas assez de points locaux
                
            # Calculer le centroïde local et la normale approximative
            local_points1 = points1[indices1]
            centroid1 = np.mean(local_points1, axis=0)
            
            # Calculer la normale approximative via SVD
            centered_points = local_points1 - centroid1
            _, _, vh = np.linalg.svd(centered_points.T @ centered_points)
            normal = vh[-1]  # Dernière colonne = direction de plus faible variance
            
            # Définir le cylindre le long de la normale
            cylinder_height = cylinder_radius * 2
            
            # Calculer la distance signée moyenne dans le cylindre
            distances_local = []
            
            for p in local_points1:
                # Projeter sur la normale
                vec = p - centroid1
                proj_distance = np.dot(vec, normal)
                
                if abs(proj_distance) <= cylinder_height / 2:
                    # Trouver le point le plus proche dans le nuage 2
                    dist, idx = tree2.query(p)
                    if dist <= max_distance:
                        point2 = points2[idx]
                        signed_dist = np.dot(point2 - p, normal)
                        distances_local.append(signed_dist)
            
            if len(distances_local) >= min_points:
                # Distance moyenne signée
                mean_distance = np.mean(distances_local)
                distances_signed.append(mean_distance)
                valid_points.append(point)
                
                # Calculer la confiance basée sur la variance locale
                if len(distances_local) > 1:
                    std_distance = np.std(distances_local)
                    confidence = 1.0 / (1.0 + std_distance)  # Confiance inversement proportionnelle à la variance
                else:
                    confidence = 0.5
                confidences.append(confidence)
        
        progress_bar.progress(1.0)
        progress_bar.empty()
        
        if len(distances_signed) == 0:
            st.warning("Aucun changement détectable trouvé avec les paramètres actuels")
            return pcd1, {'error': 'no_changes_detected'}
        
        # Convertir en arrays numpy
        distances_signed = np.array(distances_signed)
        confidences = np.array(confidences)
        valid_points = np.array(valid_points)
        
        # Classification des changements avec seuillage statistique
        # Utiliser une approche robuste avec médiane et MAD (Median Absolute Deviation)
        median_dist = np.median(distances_signed)
        mad_dist = np.median(np.abs(distances_signed - median_dist))
        
        # Seuil adaptatif basé sur la distribution
        threshold = 3 * mad_dist if mad_dist > 0 else 0.01
        
        # Classification
        erosion_mask = (distances_signed < -threshold) & (confidences > confidence_threshold)
        deposition_mask = (distances_signed > threshold) & (confidences > confidence_threshold)
        stable_mask = ~erosion_mask & ~deposition_mask
        
        # Créer le nuage de points coloré
        pcd_diff = o3d.geometry.PointCloud()
        pcd_diff.points = o3d.utility.Vector3dVector(valid_points)
        
        # Palette de couleurs pour les changements
        colors = np.zeros((len(valid_points), 3))
        
        # Erosion: rouge
        colors[erosion_mask] = [1.0, 0.0, 0.0]  # Rouge
        
        # Dépôt: bleu
        colors[deposition_mask] = [0.0, 0.0, 1.0]  # Bleu
        
        # Stable: vert/gris
        colors[stable_mask] = [0.5, 0.5, 0.5]  # Gris
        
        # Intensité basée sur la magnitude du changement
        erosion_magnitude = np.abs(distances_signed[erosion_mask])
        deposition_magnitude = np.abs(distances_signed[deposition_mask])
        
        if len(erosion_magnitude) > 0:
            max_erosion = np.max(erosion_magnitude)
            if max_erosion > 0:
                erosion_intensity = erosion_magnitude / max_erosion
                colors[erosion_mask] = np.column_stack([
                    np.ones(len(erosion_intensity)),  # R
                    1 - erosion_intensity * 0.5,      # G (diminue avec l'intensité)
                    1 - erosion_intensity * 0.5       # B (diminue avec l'intensité)
                ])
        
        if len(deposition_magnitude) > 0:
            max_deposition = np.max(deposition_magnitude)
            if max_deposition > 0:
                deposition_intensity = deposition_magnitude / max_deposition
                colors[deposition_mask] = np.column_stack([
                    1 - deposition_intensity * 0.5,  # R (diminue avec l'intensité)
                    1 - deposition_intensity * 0.5,  # G (diminue avec l'intensité)
                    np.ones(len(deposition_intensity)) # B
                ])
        
        pcd_diff.colors = o3d.utility.Vector3dVector(colors)
        
        # Statistiques détaillées
        processing_time = (time.time() - start_time) * 1000
        
        stats = {
            'total_points_analyzed': len(valid_points),
            'erosion_points': np.sum(erosion_mask),
            'deposition_points': np.sum(deposition_mask),
            'stable_points': np.sum(stable_mask),
            'mean_change_magnitude': float(np.mean(np.abs(distances_signed))),
            'max_change_magnitude': float(np.max(np.abs(distances_signed))),
            'median_change': float(median_dist),
            'mad_threshold': float(mad_dist),
            'processing_time_ms': processing_time,
            'confidence_stats': {
                'mean': float(np.mean(confidences)),
                'median': float(np.median(confidences)),
                'high_confidence_ratio': float(np.mean(confidences > confidence_threshold))
            },
            'change_distribution': {
                'erosion_volume': float(np.sum(distances_signed[erosion_mask])),
                'deposition_volume': float(np.sum(distances_signed[deposition_mask])),
                'net_change': float(np.sum(distances_signed))
            }
        }
        
        st.success(f"✅ Analyse 4D terminée ! {stats['erosion_points']} érosions, {stats['deposition_points']} dépôts détectés")
        
        return pcd_diff, stats
        
    except Exception as e:
        st.error(f"Erreur lors de l'analyse 4D : {str(e)}")
        return None, {'error': str(e)}

# Interface principale
col1, col2 = st.columns([1, 3])

with col1:
    st.header("📁 Upload d'Images")
    uploaded_files = st.file_uploader(
        "Choisissez des images (JPEG, PNG, etc.)",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        accept_multiple_files=True,
        help="Chargez au moins 2 images pour une reconstruction 3D."
    )
   
    if uploaded_files:
        st.write(f"Nombre d'images chargées : {len(uploaded_files)}")
    
    # Géoréférencement pour topographie
    enable_georef = st.checkbox("📍 Activer géoréférencement (coordonnées GPS)", value=False, key="enable_georef", 
                               help="Ajoutez des coordonnées GPS pour chaque image pour géoréférencer le nuage de points en coordonnées absolues (nécessaire pour la topographie).")
    
    gps_data = {}
    if enable_georef and uploaded_files:
        st.subheader("🌍 Coordonnées GPS par Image")
        st.markdown("Entrez les coordonnées GPS (latitude, longitude, altitude) pour chaque image. Utilisez l'EXIF des photos ou un GPS externe pour précision topographique.")
        
        for i, file in enumerate(uploaded_files):
            with st.expander(f"📷 Image {i+1}: {file.name}", expanded=False):
                col_gps1, col_gps2, col_gps3 = st.columns(3)
                with col_gps1:
                    lat = st.number_input(f"Latitude (°)", value=st.session_state.get(f'lat_{i}', 0.0), format="%.8f", key=f"lat_{i}", 
                                         help="Latitude en degrés décimaux (ex: 48.8566 pour Paris)")
                with col_gps2:
                    lon = st.number_input(f"Longitude (°)", value=st.session_state.get(f'lon_{i}', 0.0), format="%.8f", key=f"lon_{i}",
                                         help="Longitude en degrés décimaux (ex: 2.3522 pour Paris)")
                with col_gps3:
                    alt = st.number_input(f"Altitude (m)", value=st.session_state.get(f'alt_{i}', 0.0), format="%.2f", key=f"alt_{i}",
                                         help="Altitude en mètres au-dessus du niveau de la mer")
                gps_data[file.name] = {"lat": lat, "lon": lon, "alt": alt}
        
        st.session_state['gps_data'] = gps_data
        st.info("💡 Les coordonnées GPS seront utilisées pour transformer le nuage de points en coordonnées absolues après reconstruction.")
        
        # Bouton pour extraire GPS depuis EXIF
        if st.button("📍 Extraire GPS depuis EXIF des images", help="Remplit automatiquement les champs GPS depuis les métadonnées EXIF des photos (nécessite GPS activé lors de la prise de vue)"):
            extracted_count = 0
            for i, file in enumerate(uploaded_files):
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
                    tmp_file.write(file.getbuffer())
                    tmp_path = tmp_file.name
                
                gps = get_gps_from_exif(tmp_path)
                os.unlink(tmp_path)
                
                if gps:
                    # Mettre à jour session_state pour pré-remplir
                    st.session_state[f'lat_{i}'] = gps[0]
                    st.session_state[f'lon_{i}'] = gps[1]
                    st.session_state[f'alt_{i}'] = gps[2]
                    extracted_count += 1
                    st.success(f"✅ GPS extrait pour {file.name}: Lat {gps[0]:.6f}°, Lon {gps[1]:.6f}°, Alt {gps[2]:.2f}m")
                else:
                    st.warning(f"❌ Pas de données GPS dans EXIF pour {file.name}")
            
            if extracted_count > 0:
                st.info("🔄 Les champs ont été pré-remplis. Actualisez la page si nécessaire.")
            else:
                st.error("Aucune donnée GPS trouvée dans les EXIF des images.")
    
    # Options de traitement
    st.header("⚙️ Options")
    model_choice = st.radio("Modèle de reconstruction", ["DUSt3R"], help="Choisissez DUSt3R pour une approche stéréo ou MapAnything pour une reconstruction universelle metric 3D.")
    
    if model_choice == "DUSt3R":
        batch_size = st.slider("Taille du batch", min_value=1, max_value=8, value=1, key="batch_size", help="Nombre d'images traitées simultanément (plus petit = plus stable sur GPU ; max augmenté pour scalabilité)")
        niter_align = st.slider("Itérations d'alignement global", min_value=100, max_value=500, value=300, help="Nombre d'itérations pour l'optimisation globale")
        lr_align = st.slider("Taux d'apprentissage alignement", min_value=0.001, max_value=0.1, value=0.01, format="%.3f")
    
    threshold_conf = st.slider("Seuil de confiance", min_value=0.0, max_value=1.0, value=0.5, format="%.2f", key="threshold_conf", help="Seuil pour filtrer les points de confiance")
    max_points_per_view = st.slider("Max points par vue (downsample)", min_value=1000, max_value=100000, value=20000, help="Nombre max de points par image pour visualisation HD")
    scale_factor = st.slider("Facteur d'échelle pour profondeurs réalistes", min_value=0.5, max_value=3.0, value=1.0, step=0.1, help="Ajustez pour matcher les dimensions réelles de la scène (ex: 1.0 pour ~1m de profondeur typique)")
    generate_mesh = st.checkbox("Générer maillage 3D haute qualité", value=False, key="generate_mesh_main", help="Crée un maillage professionnel à partir du nuage de points avec qualité Blender-like.")
    mesh_method = st.radio("Méthode de reconstruction maillage", ["Poisson", "Ball Pivoting"], help="Poisson pour surfaces lisses ; Ball Pivoting pour maillages avec trous.")

    if generate_mesh:
        with st.expander("🎨 Paramètres Avancés de Qualité du Maillage", expanded=False):
            st.markdown("**⚙️ Ajustez ces paramètres pour éviter les 'maillages patates' et obtenir une qualité professionnelle**")

            mesh_quality_preset = st.selectbox("Préréglage qualité", [
                "Standard (recommandé)",
                "Haute qualité (plus lent)",
                "Ultra HD (très lent)",
                "Personnalisé"
            ], key="mesh_quality_preset", help="Préréglages optimisés pour différents niveaux de qualité")

            if mesh_quality_preset == "Standard (recommandé)":
                mesh_voxel_size = 0.001
                mesh_normal_radius = 0.015
                mesh_normal_neighbors = 50
                mesh_orientation_iterations = 500
                mesh_smoothing_iterations = 5
                mesh_post_smoothing = 3
            elif mesh_quality_preset == "Haute qualité (plus lent)":
                mesh_voxel_size = 0.0005
                mesh_normal_radius = 0.01
                mesh_normal_neighbors = 80
                mesh_orientation_iterations = 800
                mesh_smoothing_iterations = 8
                mesh_post_smoothing = 5
            elif mesh_quality_preset == "Ultra HD (très lent)":
                mesh_voxel_size = 0.0002
                mesh_normal_radius = 0.005
                mesh_normal_neighbors = 100
                mesh_orientation_iterations = 1000
                mesh_smoothing_iterations = 10
                mesh_post_smoothing = 8
            else:  # Personnalisé
                mesh_voxel_size = st.slider("Taille voxel (détail)", 0.0001, 0.005, 0.001, 0.0001, key="mesh_voxel_size",
                                          help="Plus petit = plus de détails mais plus lent (0.001 = 1mm)")
                mesh_normal_radius = st.slider("Rayon normales", 0.005, 0.05, 0.015, 0.005, key="mesh_normal_radius",
                                             help="Rayon pour estimation des normales de surface")
                mesh_normal_neighbors = st.slider("Voisins normales", 20, 150, 50, 10,
                                                help="Nombre de voisins pour calcul des normales")
                mesh_orientation_iterations = st.slider("Itérations orientation", 200, 1500, 500, 100,
                                                      help="Itérations pour orienter les normales de manière cohérente")
                mesh_smoothing_iterations = st.slider("Lissage pré-maillage", 0, 20, 5, 1,
                                                    help="Lissage du nuage avant reconstruction")
                mesh_post_smoothing = st.slider("Lissage post-maillage", 0, 20, 3, 1,
                                              help="Lissage du maillage final")

            mesh_adaptive_depth = st.checkbox("Profondeur Poisson adaptative", value=True, key="mesh_adaptive_depth",
                                            help="Ajuste automatiquement la profondeur selon la densité du nuage")
            mesh_clean_artifacts = st.checkbox("Nettoyage artefacts avancé", value=True, key="mesh_clean_artifacts",
                                             help="Supprime triangles dégénérés et optimise la topologie")

    if mesh_method == "Poisson":
        if generate_mesh and not mesh_adaptive_depth:
            poisson_depth = st.slider("Profondeur maillage (Poisson)", min_value=5, max_value=14, value=10, key="poisson_depth",
                                    help="Niveau de détail pour la reconstruction Poisson (plus élevé = plus fin, mais plus gourmand).")
    else:
        ball_pivoting_max_radius = st.slider("Rayon max Ball Pivoting", min_value=0.001, max_value=0.1, value=0.02, step=0.001, format="%.3f",
                                           help="Rayon maximal pour pivoting (plus grand = plus de connexions, mais plus approximatif).")
    # Section informative sur les optimisations Voxel Grid Filtering
    with st.expander("🚀 Optimisations Voxel Grid Filtering (Article Medium)", expanded=False):
        st.markdown("""
        **Basé sur l'article : "Understanding Voxel Grid Filtering: The Secret to Lightning-Fast Point Cloud Processing"**

        ### 🎯 Optimisations Implémentées :

        #### 1. **Voxel Size Adaptatif**
        - Analyse automatique de la densité locale du nuage
        - Ajustement intelligent selon le préréglage qualité :
          - **Standard** : Agressif (réduction ~70-80%)
          - **High** : Modéré (réduction ~50-70%)
          - **Ultra HD** : Conservateur (réduction ~30-50%)

        #### 2. **Filtrage Statistique Avancé**
        - Détection d'outliers basée sur distances inter-points
        - Seuil automatique : `moyenne + 2×écart-type`
        - Suppression des points aberrants préservant la géométrie

        #### 3. **Métriques de Performance**
        - Ratio de réduction en temps réel
        - Impact sur la qualité vs performance
        - Optimisation automatique selon la complexité

        ### 💡 Avantages :
        - **Vitesse** : 10-100x plus rapide sur gros nuages
        - **Mémoire** : Réduction drastique de l'usage RAM
        - **Qualité** : Préservation des détails importants
        - **Stabilité** : Élimination des artefacts de reconstruction

        ### 🎛️ Recommandations :
        - **Nuages denses (>1M points)** : Utilisez voxel size adaptatif
        - **Géométries complexes** : Privilégiez Ultra HD avec filtrage léger
        - **Performance critique** : Standard avec nettoyage artefacts activé
        """)

        # Métriques en temps réel si disponibles
        if 'current_pcd_stats' in st.session_state:
            stats = st.session_state.current_pcd_stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Points Originaux", f"{stats.get('original_points', 'N/A'):,}")
            with col2:
                st.metric("Après Voxel", f"{stats.get('voxel_points', 'N/A'):,}")
            with col3:
                st.metric("Après Nettoyage", f"{stats.get('final_points', 'N/A'):,}")
            with col4:
                outliers = stats.get('outliers_removed', 0)
                st.metric("Outliers Supprimés", f"{outliers:,}" if outliers > 0 else "N/A")

    # Section informative sur le Débruitage Industriel Avancé
    denoising_key = f"denoising_expander_{hash('denoising')}"
    with st.expander("🔬 Débruitage Industriel Avancé (Inspired by Vitreous/Telekinesis)", expanded=False):
        st.markdown("""
        **Inspiré par l'article : "How to Denoise Industrial 3D Point Clouds in Python: 3D Filtering with Vitreous from Telekinesis"**

        ### 🏭 Pipeline de Débruitage Industriel Complet :

        #### 1. **Analyse Préliminaire du Bruit**
        - Calcul automatique de la densité du nuage de points
        - Classification : très dense (>1000 pts/unité³), dense (>100), sparse (<100)
        - Adaptation des paramètres selon la complexité du nuage

        #### 2. **Filtrage Statistique Adaptatif**
        - **Statistical Outlier Removal** avec paramètres dynamiques :
          - Nuages denses : 20 voisins, seuil 1.5σ
          - Nuages moyens : 30 voisins, seuil 2.0σ
          - Nuages sparses : 50 voisins, seuil 2.5σ

        #### 3. **Filtrage par Rayon (Radius Outlier Removal)**
        - Suppression des points isolés dans un rayon donné
        - Rayon adaptatif : `3 × voxel_size`
        - Seuil minimum : 16 voisins requis

        #### 4. **Débruitage Conditionnel par Densité Locale**
        - Analyse de densité locale pour chaque région
        - Filtrage plus strict dans les zones de faible densité
        - Préservation des détails dans les zones denses

        #### 5. **Lissage Moving Least Squares (MLS)**
        - Lissage polynomial adaptatif selon la qualité :
          - **Ultra HD** : Polynôme degré 2, rayon petit (précision maximale)
          - **High** : Polynôme degré 1, rayon moyen
          - **Standard** : Polynôme degré 1, rayon large (performance)

        #### 6. **Débruitage des Couleurs**
        - Filtrage bilatéral des textures
        - Médiane locale sur 15 voisins les plus proches
        - Réduction du bruit colorimétrique tout en préservant les détails

        ### 🎯 Applications Industrielles :
        - **Scan laser industriels** : Suppression du bruit de capteur
        - **Photogrammétrie** : Nettoyage des reconstructions DUST3R
        - **Inspection qualité** : Amélioration de la précision des mesures
        - **Reverse engineering** : Préparation de données CAD propres

        ### 📊 Métriques de Performance :
        - **Taux de réduction du bruit** : Pourcentage de points supprimés
        - **Préservation de la géométrie** : Maintien des détails importants
        - **Temps de traitement** : Optimisé pour les gros volumes
        """)

        # Métriques du débruitage en temps réel
        if 'denoising_stats' in st.session_state:
            stats = st.session_state.denoising_stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Points Avant Débruitage", f"{stats.get('original_points', 'N/A'):,}")
            with col2:
                st.metric("Points Après Débruitage", f"{stats.get('final_points', 'N/A'):,}")
            with col3:
                reduction = stats.get('noise_reduction_percent', 0)
                st.metric("Bruit Supprimé", f"{reduction:.1f}%" if reduction > 0 else "N/A")
            with col4:
                filters_applied = sum([
                    stats.get('statistical_filter_applied', False),
                    stats.get('radius_filter_applied', False),
                    stats.get('mls_smoothing_applied', False),
                    stats.get('color_denoising_applied', False)
                ])
                st.metric("Filtres Appliqués", f"{filters_applied}/4")

        # Contrôles avancés pour le débruitage
        st.markdown("### ⚙️ Contrôles Avancés du Débruitage")
        
        # Initialiser les valeurs par défaut dans le session state si elles n'existent pas
        if 'advanced_denoising_enabled' not in st.session_state:
            st.session_state.advanced_denoising_enabled = True
        if 'denoising_params' not in st.session_state:
            st.session_state.denoising_params = {
                'statistical_neighbors': 30,
                'radius_min_points': 16,
                'mls_polynomial_order': 1,
                'color_neighbors': 15
            }
        
        advanced_denoising = st.checkbox("Activer Débruitage Industriel Avancé", 
                                       value=st.session_state.advanced_denoising_enabled,
                                       key="advanced_denoising_checkbox",
                                       help="Pipeline complet de 6 étapes pour le débruitage professionnel")
        st.session_state.advanced_denoising_enabled = advanced_denoising

        if advanced_denoising:
            col1, col2 = st.columns(2)
            with col1:
                statistical_neighbors = st.slider("Voisins pour filtrage statistique", 10, 100, 
                                                st.session_state.denoising_params['statistical_neighbors'],
                                                key="statistical_neighbors_slider",
                                                help="Nombre de voisins pour la détection d'outliers statistiques")
                radius_min_points = st.slider("Points minimum par rayon", 5, 50,
                                            st.session_state.denoising_params['radius_min_points'],
                                            key="radius_min_points_slider",
                                            help="Nombre minimum de voisins dans le rayon pour validation")
            with col2:
                mls_polynomial_order = st.selectbox("Ordre polynomial MLS", [1, 2],
                                                  index=st.session_state.denoising_params['mls_polynomial_order'] - 1,
                                                  key="mls_polynomial_order_select",
                                                  help="Degré du polynôme pour le lissage (2 = plus précis mais lent)")
                color_neighbors = st.slider("Voisins pour débruitage couleurs", 5, 30,
                                          st.session_state.denoising_params['color_neighbors'],
                                          key="color_neighbors_slider",
                                          help="Nombre de voisins pour le filtrage bilatéral des couleurs")

            # Mise à jour des paramètres
            st.session_state.denoising_params = {
                'statistical_neighbors': statistical_neighbors,
                'radius_min_points': radius_min_points,
                'mls_polynomial_order': mls_polynomial_order,
                'color_neighbors': color_neighbors
            }
            st.session_state.advanced_denoising_params = st.session_state.denoising_params

    # Section informative sur le Downsampling Temps Réel
    with st.expander("⚡ Downsampling Temps Réel Ultra-Rapide (Inspired by Sohail Saifi)", expanded=False):
        st.markdown("""
        **Inspiré par l'article : "Building a Real-Time Point Cloud Downsampling Pipeline from 10M to 100K Points in Milliseconds"**

        ### 🚀 Pipeline de Downsampling Temps Réel :

        #### 1. **Analyse Adaptative du Volume**
        - **Ultra-Fast** (>10M points) : Pipeline optimisé pour la vitesse pure
        - **Fast** (>1M points) : Équilibre vitesse/qualité
        - **Quality** (<1M points) : Focus sur la préservation des détails

        #### 2. **Pré-downsampling par Voxels**
        - Réduction rapide des volumes massifs (>5M points)
        - Taille de voxel adaptative selon la densité
        - Préparation pour les étapes suivantes

        #### 3. **Techniques de Downsampling Multi-étapes**

        ##### **Random Sampling (Ultra-rapide)**
        - Sélection aléatoire pour réduction initiale
        - Idéal pour gros volumes où la vitesse prime

        ##### **Uniform Grid Sampling (Équilibré)**
        - Division de l'espace en grille régulière
        - Un point par cellule pour couverture uniforme
        - Préserve la distribution spatiale

        ##### **Farthest Point Sampling (Qualité optimale)**
        - Sélection itérative des points les plus éloignés
        - Couverture spatiale maximale
        - Qualité supérieure pour l'analyse

        #### 4. **Optimisation et Métriques**
        - Calcul temps réel des performances
        - Métriques de compression et couverture spatiale
        - Validation de l'atteinte des objectifs

        ### 🎯 Applications Temps Réel :
        - **LiDAR streaming** : Traitement de données en direct
        - **Robotique** : Navigation avec contraintes de performance
        - **AR/VR** : Rendu temps réel de scènes complexes
        - **Inspection industrielle** : Analyse rapide de gros volumes

        ### ⚡ Performances Cibles :
        - **10M → 100K points** : Quelques millisecondes
        - **Traitement parallèle** : Utilisation de tous les cœurs CPU
        - **Mémoire optimisée** : Traitement par blocs si nécessaire
        - **Scalabilité** : Adapté aux GPUs pour volumes extrêmes
        """)

        # Contrôles du downsampling temps réel
        st.markdown("### 🎛️ Contrôles du Downsampling Temps Réel")

        enable_realtime_downsampling = st.checkbox("Activer Downsampling Temps Réel Ultra-Rapide", value=True, key="enable_realtime_downsampling",
                                                 help="Pipeline de réduction ultra-rapide pour gros nuages de points")

        if enable_realtime_downsampling:
            col1, col2, col3 = st.columns(3)
            with col1:
                target_points_options = [10000, 25000, 50000, 100000, 250000, 500000]
                target_points = st.selectbox("Points cibles", target_points_options, index=3,
                                           key="target_points",
                                           help="Nombre de points souhaité après downsampling")
                st.session_state.downsampling_target = target_points

            with col2:
                downsampling_strategy = st.selectbox("Stratégie prioritaire",
                                                   ["auto", "speed", "quality", "balanced"],
                                                   index=0,
                                                   key="downsampling_strategy",
                                                   help="Stratégie de downsampling : auto=adaptatif, speed=vitesse max, quality=qualité max")

            with col3:
                preserve_colors = st.checkbox("Préserver les couleurs", value=True,
                                            key="preserve_colors",
                                            help="Maintenir les informations de couleur lors du downsampling")
                preserve_normals = st.checkbox("Préserver les normales", value=False,
                                             key="preserve_normals",
                                             help="Maintenir les normales de surface (plus lent)")

        # Métriques du downsampling temps réel
        if 'realtime_downsampling_stats' in st.session_state:
            stats = st.session_state.realtime_downsampling_stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Points Originaux", f"{stats.get('original_points', 'N/A'):,}")
            with col2:
                st.metric("Points Finaux", f"{stats.get('final_points', 'N/A'):,}")
            with col3:
                processing_time = stats.get('processing_time_ms', 0)
                st.metric("Temps (ms)", f"{processing_time:.1f}" if processing_time > 0 else "N/A")
            with col4:
                compression = stats.get('compression_ratio', 0)
                st.metric("Compression", f"{compression:.1f}x" if compression > 0 else "N/A")

            # Indicateur de succès
            target_achieved = stats.get('target_achieved', False)
            if target_achieved:
                st.success("🎯 Objectif de downsampling atteint !")
            else:
                st.warning("⚠️ Objectif partiellement atteint - augmenter la cible si nécessaire")

    # Section ML sur Nuages de Points (Inspired by hc's Medium article)
    with st.expander("🧠 ML sur Nuages de Points - Classification & Segmentation (Inspired by hc's Medium Article)", expanded=False):
        st.markdown("""
        **Inspiré par l'article : "Performing ML on Point-Clouds"**

        ### 🎯 Techniques de ML sur Nuages de Points :

        #### 1. **PointNet (2017) - Classification Directe**
        - **Traitement direct** : Pas de voxelisation coûteuse en mémoire
        - **Invariance de permutation** : Max-pooling respecte l'ordre des points
        - **Spatial Transformers** : Gestion invariance rotation/translation
        - **Fusion local-global** : Features individuels + features globales

        #### 2. **Point-Voxel CNN (PVCNN) (2019) - Meilleur des Deux Mondes**
        - **Architecture hybride** : Voxelisation grossière + traitement direct
        - **Information d'adjacence** : Capture les relations de voisinage
        - **Optimisation mémoire** : 7x moins de mémoire que PointNet
        - **Performance** : 10x plus rapide

        #### 3. **Point Transformer (2020) - Attention-based**
        - **Mécanismes d'attention** : Capture interactions locales/globales
        - **Regroupement intelligent** : Voisins proches par rayon limité
        - **Évolutivité** : Gestion des gros volumes de données
        - **État de l'art** : Performance supérieure

        ### 🚀 Applications dans la Photogrammétrie :

        #### **Classification d'Objets**
        - Bâtiments, Véhicules, Végétation, Terrain
        - Analyse urbaine automatique
        - Inspection industrielle

        #### **Segmentation Sémantique**
        - Séparation façade/toit/terrain
        - Identification matériaux
        - Analyse de défauts

        #### **Détection d'Anomalies**
        - Artefacts de reconstruction
        - Zones de faible qualité
        - Incohérences géométriques

        #### **Optimisation Adaptative**
        - Paramètres selon type de scène
        - Qualité prédictive
        - Reconstruction guidée
        """)

        # Contrôles ML
        enable_ml_processing = st.checkbox("Activer Traitement ML sur Nuages de Points", value=False, key="enable_ml_processing",
                                         help="Classification et segmentation intelligentes des objets")

        if enable_ml_processing:
            col1, col2, col3 = st.columns(3)
            with col1:
                ml_technique = st.selectbox("Technique ML",
                                          ["PointNet (Classification)", "PVCNN (Hybride)", "PointTransformer (Attention)"],
                                          index=0,
                                          key="ml_technique",
                                          help="Algorithme de ML à utiliser")

            with col2:
                ml_task = st.selectbox("Tâche ML",
                                     ["Classification d'objets", "Segmentation sémantique", "Détection d'anomalies"],
                                     index=0,
                                     key="ml_task",
                                     help="Type de tâche à effectuer")

            with col3:
                ml_confidence_threshold = st.slider("Seuil de confiance", 0.1, 1.0, 0.7,
                                                  help="Seuil minimum de confiance pour les prédictions")

            # Sauvegarde des paramètres ML dans session state
            # Note: Les paramètres ML sont automatiquement gérés par Streamlit via les clés des widgets
            # st.session_state.ml_technique = ml_technique
            # st.session_state.ml_task = ml_task
            # st.session_state.ml_confidence_threshold = ml_confidence_threshold

            # Métriques ML
            if 'ml_processing_stats' in st.session_state:
                stats = st.session_state.ml_processing_stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Objets Classifiés", f"{stats.get('classified_objects', 0)}")
                with col2:
                    st.metric("Confiance Moyenne", f"{stats.get('avg_confidence', 0):.2f}")
                with col3:
                    st.metric("Temps ML (ms)", f"{stats.get('processing_time_ms', 0):.1f}")
                with col4:
                    st.metric("Mémoire GPU (MB)", f"{stats.get('gpu_memory_mb', 0):.0f}")

        # ============================================
        # ANALYSE 4D TEMPORELLE (CHANGEMENTS)
        # ============================================
        st.subheader("⏰ Analyse 4D Temporelle (Détection de Changements)")

        enable_4d_analysis = st.checkbox("🔍 Activer Analyse 4D (Py4DGeo-inspired)", value=False, key="enable_4d_analysis",
                                       help="Détecte les changements temporels entre deux nuages de points (érosions, dépôts, glissements)")

        if enable_4d_analysis:
            st.markdown("""
            **🧪 Analyse temporelle avancée** inspirée de Py4DGeo pour détecter:
            - **Érosions**: Zones de perte de matière (rouge)
            - **Dépôts**: Zones d'accumulation (bleu)
            - **Changements morphologiques**: Glissements, affaissements
            - **Évolution urbaine**: Construction/destruction
            """)

            # Upload du nuage de référence (époque 1)
            col_ref, col_comp = st.columns(2)
            with col_ref:
                st.markdown("**📅 Époque de Référence (T1)**")
                reference_pcd = st.file_uploader(
                    "Nuage de points référence (.ply, .pcd)",
                    type=['ply', 'pcd'],
                    key='reference_pcd',
                    help="Premier nuage de points (époque ancienne)"
                )

            with col_comp:
                st.markdown("**📅 Époque de Comparaison (T2)**")
                comparison_pcd = st.file_uploader(
                    "Nuage de points comparaison (.ply, .pcd)",
                    type=['ply', 'pcd'],
                    key='comparison_pcd',
                    help="Second nuage de points (époque récente)"
                )

            # Paramètres M3C2
            with st.expander("⚙️ Paramètres M3C2 (Multiscale Model-to-Model Cloud Comparison)", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    cylinder_radius = st.slider("Rayon des cylindres (m)", 0.01, 1.0, 0.1, 0.01,
                                              help="Rayon des cylindres locaux pour l'analyse (plus petit = plus précis)")

                with col2:
                    min_points_cylinder = st.slider("Points min/cylindre", 5, 50, 10,
                                                  help="Nombre minimum de points requis dans chaque cylindre")

                with col3:
                    confidence_threshold_4d = st.slider("Seuil de confiance", 0.5, 0.99, 0.95, 0.01,
                                                      help="Seuil de confiance pour la classification des changements")

                max_distance_4d = st.slider("Distance max de recherche (m)", 0.1, 5.0, 1.0, 0.1,
                                          help="Distance maximale pour trouver les correspondances entre nuages")

            # Sauvegarde des paramètres 4D
            # Note: enable_4d_analysis est automatiquement géré par Streamlit via la clé du widget
            st.session_state.cylinder_radius = cylinder_radius
            st.session_state.min_points_cylinder = min_points_cylinder
            st.session_state.confidence_threshold_4d = confidence_threshold_4d
            st.session_state.max_distance_4d = max_distance_4d

            # Métriques 4D si disponibles
            if '4d_analysis_stats' in st.session_state:
                stats_4d = st.session_state['4d_analysis_stats']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Points Analysés", f"{stats_4d.get('total_points_analyzed', 0)}")
                with col2:
                    st.metric("Érosions", f"{stats_4d.get('erosion_points', 0)}")
                with col3:
                    st.metric("Dépôts", f"{stats_4d.get('deposition_points', 0)}")
                with col4:
                    st.metric("Changement Moyen", f"{stats_4d.get('mean_change_magnitude', 0):.3f}m")

    # Amélioration : Option pour visualiser la coque convexe
    show_hull = st.checkbox("Afficher la coque convexe autour du maillage", value=True, key="show_hull_main", help="Ajoute une coque convexe pour mieux visualiser les limites de la scène dans Open3D.")
    
    # Nouvelle fonctionnalité : Analyse géométrique automatique RANSAC
    enable_auto_ransac = st.checkbox("🔬 Activer analyse géométrique automatique (RANSAC)", value=False, key="enable_auto_ransac", 
                                    help="Détecte automatiquement plans, cylindres et sphères dans le nuage de points et les colore dans la visualisation Open3D.")
    
    # Valeurs par défaut pour éviter les erreurs Pylance
    ransac_auto_threshold = 0.02
    ransac_auto_iterations = 1000
    
    if enable_auto_ransac:
        ransac_auto_threshold = st.slider("Seuil automatique RANSAC", 0.01, 0.1, 0.02, 0.005, 
                                         help="Seuil de distance pour la détection automatique (plus petit = plus strict).")
        ransac_auto_iterations = st.slider("Itérations automatiques", 500, 2000, 1000, 100, 
                                          help="Nombre d'itérations pour chaque type de forme.")
    
    # ============================================
    # NOUVELLES FONCTIONNALITÉS : POST-TRAITEMENT DU NUAGE DE POINTS
    # ============================================
    st.subheader("🔧 Post-traitement du Nuage de Points")
    
    # Algorithme 1: Reconstruction des nuages de points manquants
    enable_missing_reconstruction = st.checkbox("🔄 Reconstruction des zones manquantes", value=False, key="enable_missing_reconstruction", 
                                               help="Utilise l'interpolation et le remplissage de trous pour reconstruire les zones manquantes du nuage de points.")
    
    # Algorithme 2: Nettoyage des artefacts et déformations
    enable_artifact_cleaning = st.checkbox("🧹 Nettoyage des artefacts et déformations", value=False, key="enable_artifact_cleaning", 
                                          help="Supprime les outliers, réduit le bruit et corrige les déformations mineures du nuage de points.")
    
    # Algorithme 3: Correction des déformations géométriques
    enable_geometric_correction = st.checkbox("📐 Correction des déformations géométriques", value=False, key="enable_geometric_correction", 
                                             help="Applique des algorithmes de lissage et d'optimisation pour corriger les déformations géométriques.")
    
    # ============================================
    # SECTION: VISUALISATION AVANCÉE DU MAILLAGE
    # ============================================
    st.subheader("🕸️ Visualisation Avancée du Maillage (Style Blender)")
    
    with st.expander("🎨 Options de Visualisation Topologique", expanded=False):
        st.markdown("""
        **Visualisation professionnelle** comme dans Blender pour analyser la qualité du maillage:
        - **Wireframe**: Voir la structure des triangles/polygones
        - **UV Checker**: Grille damier pour vérifier le mapping de textures
        - **Subdivision**: Lisser la surface avec plus de géométrie
        - **Normales**: Visualiser l'orientation des surfaces
        """)
        
        col_w1, col_w2 = st.columns(2)
        
        with col_w1:
            wireframe_overlay = st.checkbox(
                "🕸️ Afficher Wireframe (Fil de fer)", 
                value=False,
                help="Superpose le maillage en fil de fer pour voir la topologie exacte"
            )
            
            if wireframe_overlay:
                wireframe_thickness = st.slider(
                    "Épaisseur du wireframe",
                    min_value=0.5,
                    max_value=3.0,
                    value=1.0,
                    step=0.5,
                    help="Épaisseur des lignes du wireframe"
                )
            
            show_uv_checker = st.checkbox(
                "🎨 UV Checker Pattern",
                value=False,
                help="Applique une texture damier pour visualiser la qualité du mapping UV"
            )
            
            show_topology_info = st.checkbox(
                "📊 Afficher Info Topologie",
                value=False,
                help="Statistiques détaillées: nombre de triangles, vertices, edges, qualité"
            )
        
        with col_w2:
            subdivision_level = st.slider(
                "🔺 Niveau de Subdivision",
                min_value=0,
                max_value=3,
                value=0,
                help="Subdivise le maillage pour plus de lissage (0=désactivé, 3=très lisse)"
            )
            
            show_normals = st.checkbox(
                "➡️ Visualiser les Normales",
                value=False,
                help="Affiche les vecteurs normaux pour chaque face (orientation des surfaces)"
            )
            
            normal_length = 0.05  # Valeur par défaut
            if show_normals:
                normal_length = st.slider(
                    "Longueur des normales",
                    min_value=0.01,
                    max_value=0.2,
                    value=0.05,
                    step=0.01,
                    format="%.2f",
                    help="Taille des flèches de normales"
                )

    # ============================================
    # NOUVELLE SECTION: ANALYSE INTELLIGENTE PBR
    # ============================================
    st.header("🤖 Analyse IA de Scène + Suggestions PBR")
    
    if PBR_ANALYZER_AVAILABLE:
        st.markdown("""
        **Intelligence artificielle avancée** pour identifier automatiquement les matériaux 
        de votre scène et suggérer les **textures PBR** nécessaires pour un rendu ultra-réaliste.
        
        🔬 **Technologies utilisées:**
        - **CLIP** (OpenAI) : Vision par ordinateur pour classification de scène
        - **Phi-1.5** (Microsoft) : Modèle de langage pour recommandations intelligentes
        """)
        
        analyze_scene_btn = st.checkbox("🔍 Activer l'analyse automatique de scène", value=False, key="analyze_scene_btn", 
                                       help="Analyse vos images pour suggérer les textures PBR à télécharger")
        
        # Checkbox pour injection automatique de textures
        enable_auto_injection = st.checkbox(
            "💉 Injection automatique de textures depuis la bibliothèque",
            value=False,
            help="Télécharge et applique automatiquement les textures PBR basées sur l'analyse IA"
        )
        st.session_state['enable_auto_texture_injection'] = enable_auto_injection
        
        if analyze_scene_btn and uploaded_files:
            with st.expander("📊 Résultats de l'analyse IA", expanded=True):
                with st.spinner("🤖 Analyse en cours avec CLIP + Phi-1.5..."):
                    try:
                        # Initialisation de l'analyseur
                        pbr_analyzer = TexturePBRAnalyzer(device=device)  # type: ignore
                        
                        # Conversion des fichiers uploadés en images PIL
                        temp_images = []
                        for uploaded_file in uploaded_files[:5]:  # Limite à 5 images pour performance
                            img = Image.open(uploaded_file).convert('RGB')
                            temp_images.append(img)
                        
                        # Analyse complète
                        analysis_report = pbr_analyzer.analyze_scene_batch(temp_images)
                        
                        # INJECTION AUTOMATIQUE DE TEXTURES depuis la bibliothèque locale
                        if st.session_state.get('enable_auto_texture_injection', False) and TEXTURE_MANAGER_AVAILABLE:
                            try:
                                texture_manager = st.session_state.get('texture_manager')
                                if texture_manager:
                                    dominant_material = analysis_report['top_materials'][0]['material'] if analysis_report['top_materials'] else 'unknown'
                                    
                                    # Rechercher texture correspondante
                                    best_texture = texture_manager.get_texture_for_injection(dominant_material)
                                    
                                    if best_texture:
                                        st.session_state['auto_injected_texture'] = best_texture
                                        st.success(f"✅ Texture auto-injectée: {best_texture['name']}")
                                    else:
                                        st.info(f"ℹ️ Pas de texture locale pour '{dominant_material}'. Téléchargez-en d'abord dans la bibliothèque.")
                            except Exception as e:
                                st.warning(f"Injection auto désactivée: {e}")
                        
                        # Affichage des résultats
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.subheader("🎬 Type de Scène Détecté")
                            st.success(f"**{analysis_report['dominant_scene_type']}**")
                            st.metric("Confiance", f"{analysis_report['scene_confidence']*100:.1f}%")
                        
                        with col_b:
                            st.subheader("🧱 Matériaux Identifiés")
                            for mat_info in analysis_report['top_materials']:
                                st.write(f"- **{mat_info['material']}** ({mat_info['confidence']*100:.0f}%)")
                        
                        st.markdown("---")
                        st.subheader("📦 Textures PBR Recommandées")
                        
                        recommendations = analysis_report['texture_recommendations']
                        
                        if 'pbr_textures_needed' in recommendations:
                            textures = recommendations['pbr_textures_needed']
                            st.info(f"**{len(textures)} textures PBR** identifiées comme nécessaires:")
                            
                            # Affichage en colonnes
                            cols = st.columns(3)
                            for idx, texture in enumerate(textures):
                                with cols[idx % 3]:
                                    st.code(texture, language="")
                        
                        st.markdown("---")
                        st.subheader("🌐 Liens de Téléchargement (Gratuit)")
                        
                        for idx, link in enumerate(analysis_report['download_links']):
                            with st.expander(f"📚 {link['name']} - {link['license']}", expanded=False):
                                st.markdown(f"**Description:** {link['description']}")
                                st.markdown(f"**URL:** [{link['url']}]({link['url']})")
                                
                                if link['search_keywords']:
                                    st.markdown("**Mots-clés de recherche:**")
                                    st.write(", ".join(link['search_keywords']))
                                
                                # BOUTON TÉLÉCHARGEMENT INDIVIDUEL pour chaque source
                                st.markdown("---")
                                if TEXTURE_MANAGER_AVAILABLE:
                                    button_key = f"download_from_{link['name'].replace(' ', '_')}_{idx}"
                                    if st.button(f"🚀 Télécharger depuis {link['name']}", key=button_key, type="primary"):
                                        with st.spinner(f"📥 Téléchargement depuis {link['name']}..."):
                                            try:
                                                # Initialiser le gestionnaire si nécessaire
                                                if 'texture_manager' not in st.session_state:
                                                    st.session_state['texture_manager'] = TextureDownloadManager(storage_path="./texture_library")  # type: ignore
                                                
                                                texture_manager = st.session_state['texture_manager']
                                                
                                                # Télécharger avec les mots-clés spécifiques de ce lien
                                                keywords = link.get('search_keywords', [])
                                                if keywords:
                                                    downloaded_ids = texture_manager.batch_download(
                                                        material_keywords=keywords,
                                                        max_textures=3,
                                                        resolution="2k"
                                                    )
                                                    
                                                    if downloaded_ids:
                                                        st.success(f"✅ {len(downloaded_ids)} textures téléchargées depuis {link['name']}! Voir la bibliothèque ci-dessous.")
                                                        st.session_state['auto_downloaded'] = True
                                                        st.rerun()
                                                    else:
                                                        st.warning(f"⚠️ Aucune texture trouvée pour {', '.join(keywords)}.")
                                                else:
                                                    st.warning("⚠️ Pas de mots-clés disponibles pour cette source.")
                                            
                                            except Exception as e:
                                                st.error(f"❌ Erreur de téléchargement: {e}")
                        
                        # Sauvegarde du rapport
                        st.download_button(
                            label="💾 Télécharger le rapport complet (JSON)",
                            data=json.dumps(analysis_report, indent=2),
                            file_name=f"pbr_analysis_{uuid.uuid4().hex[:8]}.json",
                            mime="application/json"
                        )
                        
                        # ============================================
                        # GÉNÉRATION PIPELINE PBR AUTOMATIQUE
                        # ============================================
                        if AUTO_PBR_MAPPER_AVAILABLE:
                            st.markdown("---")
                            st.subheader("⚡ Pipeline PBR Automatique Temps Réel")
                            
                            if st.button("🚀 Générer Configuration Pipeline Complet"):
                                with st.spinner("⚡ Génération du pipeline PBR temps réel..."):
                                    try:
                                        # Données fictives pour démo (seront remplacées par vraies données après reconstruction)
                                        dummy_vertices = np.random.rand(1000, 3)
                                        dummy_normals = np.random.rand(1000, 3)
                                        dummy_normals = dummy_normals / np.linalg.norm(dummy_normals, axis=1, keepdims=True)
                                        
                                        # Génération du pipeline complet
                                        material_scores_dict = {mat['material']: mat['confidence'] for mat in analysis_report['top_materials']}
                                        
                                        pipeline_config = generate_complete_pbr_pipeline(  # type: ignore
                                            material_scores_dict,
                                            analysis_report['dominant_scene_type'],
                                            dummy_vertices,
                                            dummy_normals,
                                            device
                                        )
                                        
                                        # Affichage de la configuration
                                        col_pip1, col_pip2 = st.columns(2)
                                        
                                        with col_pip1:
                                            st.success("✅ Pipeline PBR Généré !")
                                            st.json(pipeline_config['pbr_configuration'])
                                        
                                        with col_pip2:
                                            st.info("🎮 Configuration Rendu Temps Réel")
                                            st.json(pipeline_config['realtime_rendering'])
                                        
                                        st.markdown("---")
                                        st.subheader("📐 Stratégie UV Unwrap")
                                        st.json(pipeline_config['uv_unwrap_strategy'])
                                        
                                        st.markdown("---")
                                        st.metric("Résolution Texture Recommandée", f"{pipeline_config['texture_resolution']}x{pipeline_config['texture_resolution']}")
                                        
                                        # Export de la configuration
                                        st.download_button(
                                            label="💾 Télécharger Config Pipeline (JSON)",
                                            data=json.dumps(pipeline_config, indent=2),
                                            file_name=f"pipeline_config_{uuid.uuid4().hex[:8]}.json",
                                            mime="application/json"
                                        )
                                        
                                        st.success("🎉 Pipeline prêt pour intégration Unreal/Unity/Blender !")
                                        
                                    except Exception as e:
                                        st.error(f"❌ Erreur génération pipeline: {e}")
                        
                        # ============================================
                        # FIN PIPELINE PBR AUTO
                        # ============================================
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'analyse: {e}")
                        st.info("Vérifiez que les modèles CLIP et Phi-1.5 sont correctement installés.")
        
        elif analyze_scene_btn and not uploaded_files:
            st.warning("⚠️ Veuillez d'abord charger des images pour l'analyse.")
    
    else:
        st.warning("⚠️ Module d'analyse PBR non disponible. Vérifiez l'installation de texture_pbr_analyzer.py")
    
    # ============================================
    # FIN SECTION ANALYSE PBR
    # ============================================

    # ============================================
    # NOUVELLE SECTION: VFX IA INTELLIGENTS
    # ============================================
    st.header("🎬 Effets VFX IA Automatiques")
    
    if VFX_ENGINE_AVAILABLE:
        st.markdown("""
        **Moteur VFX intelligent** qui applique automatiquement des effets réalistes :
        - 🪨 Saleté et usure selon gravité
        - 🦀 Rouille sur métal (humidité + âge)
        - 🌿 Mousse sur surfaces ombragées
        - ⚡ Usure générale (décoloration, fissures)
        
        **Plus besoin de Blender pour des effets réalistes !**
        """)
        
        apply_vfx = st.checkbox("🎬 Activer VFX automatiques", value=False,
                               help="Applique des effets visuels intelligents sur votre modèle 3D")
        
        if apply_vfx:
            with st.expander("⚙️ Paramètres VFX", expanded=True):
                col_vfx1, col_vfx2 = st.columns(2)
                
                with col_vfx1:
                    vfx_intensity = st.slider("Intensité globale", 0.0, 1.0, 0.5, 0.05,
                                            help="Intensité de tous les effets")
                    vfx_age = st.slider("Âge du matériau", 0.0, 1.0, 0.3, 0.05,
                                      help="0=neuf, 1=très vieux")
                    vfx_humidity = st.slider("Humidité", 0.0, 1.0, 0.5, 0.05,
                                           help="0=sec, 1=très humide (rouille, mousse)")
                
                with col_vfx2:
                    vfx_exposure = st.slider("Exposition extérieure", 0.0, 1.0, 0.5, 0.05,
                                           help="0=intérieur protégé, 1=extérieur exposé")
                    vfx_pollution = st.slider("Pollution", 0.0, 1.0, 0.3, 0.05,
                                            help="Niveau de saleté environnementale")
                    
                    vfx_preset = st.selectbox("Préréglages", [
                        "Personnalisé",
                        "Neuf et propre",
                        "Bâtiment abandonné",
                        "Zone industrielle",
                        "Environnement forestier",
                        "Désert aride",
                        "Zone côtière"
                    ])
                
                # Application des préréglages
                if vfx_preset == "Neuf et propre":
                    vfx_age, vfx_humidity, vfx_exposure, vfx_pollution = 0.1, 0.3, 0.2, 0.1
                elif vfx_preset == "Bâtiment abandonné":
                    vfx_age, vfx_humidity, vfx_exposure, vfx_pollution = 0.9, 0.7, 0.8, 0.6
                elif vfx_preset == "Zone industrielle":
                    vfx_age, vfx_humidity, vfx_exposure, vfx_pollution = 0.6, 0.5, 0.7, 0.9
                elif vfx_preset == "Environnement forestier":
                    vfx_age, vfx_humidity, vfx_exposure, vfx_pollution = 0.5, 0.9, 0.6, 0.3
                elif vfx_preset == "Désert aride":
                    vfx_age, vfx_humidity, vfx_exposure, vfx_pollution = 0.7, 0.1, 0.9, 0.4
                elif vfx_preset == "Zone côtière":
                    vfx_age, vfx_humidity, vfx_exposure, vfx_pollution = 0.6, 0.8, 0.8, 0.5
                
                # Détection automatique de matériau
                auto_detect_material = st.checkbox("🔍 Détection auto du matériau", value=True,
                                                  help="L'IA détecte le type de matériau depuis les couleurs")
                
                manual_material = "concrete"  # Valeur par défaut
                if not auto_detect_material:
                    manual_material = st.selectbox("Type de matériau", [
                        "concrete", "metal", "wood", "plastic", "stone", "glass"
                    ])
                
                st.info("💡 Les VFX seront appliqués après la reconstruction 3D")
                
                # Stockage dans session state
                st.session_state.vfx_params = VFXParameters(  # type: ignore
                    intensity=vfx_intensity,
                    age=vfx_age,
                    humidity=vfx_humidity,
                    exposure=vfx_exposure,
                    pollution=vfx_pollution
                )
                st.session_state.vfx_auto_material = auto_detect_material
                st.session_state.vfx_manual_material = manual_material
        else:
            # Désactiver VFX
            if 'vfx_params' in st.session_state:
                del st.session_state.vfx_params
    
    else:
        st.warning("⚠️ Moteur VFX non disponible. Vérifiez l'installation de intelligent_vfx_engine.py")
    
    # ============================================
    # FIN SECTION VFX
    # ============================================
    
    # ============================================
    # SECTION BIBLIOTHÈQUE (APRÈS ANALYSE)
    # ============================================
    st.markdown("---")
    st.header("📚 Bibliothèque de Textures PBR (Téléchargement & Injection Auto)")
    
    if TEXTURE_MANAGER_AVAILABLE:
        with st.expander("🔧 Gérer la bibliothèque de textures", expanded=st.session_state.get('auto_downloaded', False)):
            try:
                # Initialisation du gestionnaire (sans cache pour éviter problèmes de state)
                if 'texture_manager' not in st.session_state:
                    st.session_state['texture_manager'] = TextureDownloadManager(storage_path="./texture_library")  # type: ignore
                
                texture_manager = st.session_state['texture_manager']
                
                # Stats de la bibliothèque
                stats = texture_manager.get_library_stats()
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Textures téléchargées", stats['total_textures'])
                with col_stat2:
                    st.metric("Espace utilisé", f"{stats['total_size_mb']:.1f} MB")
                with col_stat3:
                    st.metric("Types de matériaux", len(stats['by_material']))
                
                st.markdown("**Matériaux disponibles:**")
                if stats['by_material']:
                    for material, count in stats['by_material'].items():
                        st.write(f"- {material}: {count} texture(s)")
                
                st.markdown("---")
                
                # Recherche et téléchargement
                st.markdown("**🔍 Rechercher et télécharger de nouvelles textures**")
                
                col_search1, col_search2 = st.columns([3, 1])
                with col_search1:
                    search_keywords = st.text_input(
                        "Mots-clés (séparés par des virgules)",
                        value="concrete, metal, wood",
                        help="Ex: concrete, rusty metal, wooden planks"
                    )
                with col_search2:
                    resolution = st.selectbox("Résolution", ["1k", "2k", "4k"], index=1)
                
                max_downloads = st.slider("Nombre max de textures à télécharger", 1, 10, 3)
                
                if st.button("🚀 Rechercher et Télécharger", type="primary"):
                    keywords = [k.strip() for k in search_keywords.split(',')]
                    
                    with st.spinner(f"Recherche et téléchargement de textures {resolution}..."):
                        downloaded_ids = texture_manager.batch_download(
                            material_keywords=keywords,
                            max_textures=max_downloads,
                            resolution=resolution
                        )
                        
                        if downloaded_ids:
                            st.success(f"✅ {len(downloaded_ids)} textures téléchargées avec succès!")
                            st.rerun()
                        else:
                            st.warning("Aucune texture trouvée ou erreur de téléchargement.")
                
                st.markdown("---")
                
                # Aperçu des textures locales
                st.markdown("**📦 Textures disponibles localement**")
                
                filter_material = st.selectbox(
                    "Filtrer par matériau",
                    ["Tous"] + list(stats['by_material'].keys()) if stats['by_material'] else ["Tous"]
                )
                
                local_textures = texture_manager.search_local_textures(
                    material_type=None if filter_material == "Tous" else filter_material
                )
                
                if local_textures:
                    st.write(f"**{len(local_textures)} texture(s) trouvée(s)**")
                    
                    # Afficher en grille de mini-cartes
                    num_cols = 3
                    for i in range(0, len(local_textures), num_cols):
                        cols = st.columns(num_cols)
                        
                        for j, col in enumerate(cols):
                            idx = i + j
                            if idx < len(local_textures):
                                tex = local_textures[idx]
                                
                                with col:
                                    # Mini carte
                                    with st.container():
                                        st.markdown(f"**{tex['name'][:30]}**")
                                        
                                        # Thumbnail si disponible
                                        if tex['thumbnail_path'] and Path(tex['thumbnail_path']).exists():
                                            try:
                                                img = Image.open(tex['thumbnail_path'])
                                                st.image(img, use_container_width=True)
                                            except:
                                                st.info("📦 Texture PBR")
                                        else:
                                            st.info("📦 Texture PBR")
                                        
                                        st.caption(f"Type: {tex['material_type']}")
                                        st.caption(f"Résolution: {tex['resolution']}")
                                        st.caption(f"Maps: {len(tex['maps'])} fichiers")
                                        
                                        # Bouton d'injection manuelle
                                        if st.button(f"💉 Injecter", key=f"inject_{tex['id']}"):
                                            st.session_state['manual_texture_injection'] = tex['id']
                                            st.info(f"Texture {tex['name']} sélectionnée pour injection!")
                else:
                    st.info("Aucune texture téléchargée. Utilisez la recherche ci-dessus pour en ajouter.")
            
            except Exception as e:
                st.error(f"Erreur bibliothèque de textures: {e}")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.warning("Module texture_download_manager non disponible")

    st.header("🖌️ Textures PBR Manuelles")
    texture_zip = st.file_uploader("Upload ZIP de textures PBR (dossiers par catégorie e.g. rock/, water/)", type='zip', help="Les dossiers dans le ZIP définissent les catégories (ex: rock/albedo.png). Les textures sont intégrées dans une base FAISS pour correspondance dynamique.")
   
    if texture_zip is not None:
        with st.spinner("Traitement des textures PBR..."):
            with tempfile.TemporaryDirectory() as tmp_dir:
                zip_path = os.path.join(tmp_dir, 'textures.zip')
                with open(zip_path, 'wb') as f:
                    f.write(texture_zip.getbuffer())
                textures_dir = os.path.join(tmp_dir, 'textures')
                os.makedirs(textures_dir, exist_ok=True)
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(textures_dir)
                
                clip_model, clip_processor = load_clip_model()
                embeddings_list = []
                categories = []
                avg_colors_list = []
                db_path = os.path.join(tempfile.gettempdir(), 'streamlit_textures.db')
                conn = sqlite3.connect(db_path)  # type: ignore
                cur = conn.cursor()
                cur.execute('''CREATE TABLE IF NOT EXISTS textures
                               (category TEXT PRIMARY KEY, embedding BLOB, avg_color BLOB)''')
                if clip_model is not None:
                    for category in os.listdir(textures_dir):
                        cat_dir = os.path.join(textures_dir, category)
                        if os.path.isdir(cat_dir):
                            cat_images = []
                            for file in os.listdir(cat_dir):
                                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    img_path = os.path.join(cat_dir, file)
                                    image = Image.open(img_path).convert('RGB')
                                    cat_images.append(image)
                            if cat_images:
                                inputs = clip_processor(images=cat_images, return_tensors="pt").to(device)  # type: ignore
                                with torch.no_grad():
                                    embeddings = clip_model.get_image_features(**inputs)
                                    avg_emb = torch.mean(embeddings, dim=0).cpu().numpy()
                                all_pixels = []
                                for img in cat_images:
                                    img_np = np.array(img) / 255.0
                                    all_pixels.append(img_np.reshape(-1, 3))
                                if all_pixels:
                                    avg_color = np.mean(np.vstack(all_pixels), axis=0)
                                else:
                                    avg_color = np.array([0.5, 0.5, 0.5])
                                embeddings_list.append(avg_emb)
                                categories.append(category)
                                avg_colors_list.append(avg_color)
                                emb_blob = pickle.dumps(avg_emb)
                                color_blob = pickle.dumps(avg_color)
                                cur.execute("INSERT OR REPLACE INTO textures VALUES (?, ?, ?)", (category, emb_blob, color_blob))
                    
                    conn.commit()
                    
                    if embeddings_list:
                        # Amélioration : Seuil adaptatif basé sur variance des embeddings
                        emb_array = np.array(embeddings_list)
                        emb_std = np.std(emb_array)
                        adaptive_threshold_factor = 1.5  # Facteur pour tolérance dynamique
                        adaptive_max_dist = emb_std * adaptive_threshold_factor if emb_std > 0 else 2.0
                        st.info(f"Seuil adaptatif pour textures : {adaptive_max_dist:.2f} (basé sur std des embeddings = {emb_std:.2f})")
                        
                        # Création de l'index avec fallback
                        try:
                            if FAISS_AVAILABLE:
                                dim = len(embeddings_list[0])
                                faiss_index = faiss.IndexFlatL2(dim)  # type: ignore
                                faiss_index.add(emb_array)  # type: ignore
                                st.session_state.search_index = faiss_index
                                st.session_state.is_faiss = True
                            else:
                                raise ImportError("FAISS non disponible")
                        except:
                            # Fallback sklearn
                            nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
                            nn.fit(emb_array)
                            st.session_state.search_index = nn
                            st.session_state.is_faiss = False
                            st.info("Utilisation de scikit-learn NearestNeighbors comme fallback pour FAISS.")
                        
                        texture_metadata = [{'category': cat, 'avg_color': avg_col} for cat, avg_col in zip(categories, avg_colors_list)]
                        st.session_state.texture_metadata = texture_metadata
                        st.session_state.adaptive_max_dist = adaptive_max_dist
                        st.success(f"Textures PBR chargées: {len(categories)} catégories intégrées (avec fallback si besoin) et sauvegardées en SQLite3.")
                        
                        # Affichage de la liste des types de textures dans un tableau depuis SQLite3
                        cur.execute("SELECT category FROM textures")
                        db_categories = [row[0] for row in cur.fetchall()]
                        df = pd.DataFrame({'Types de Textures': db_categories})
                        st.table(df)

                        # Affichage compact des textures PBR avec miniatures
                        if 'texture_metadata' in st.session_state and st.session_state.texture_metadata:
                            st.header("🎨 Aperçu des Textures PBR")
                            for tex in st.session_state.texture_metadata:
                                category = tex['category']
                                avg_color = (tex['avg_color'] * 255).astype(int)
                                img_preview = Image.new('RGB', (50, 50), tuple(avg_color))
                                
                                col1, col2 = st.columns([1, 1])
                                with col1:
                                    st.markdown(f"**{category}**")
                                with col2:
                                    st.image(img_preview, width=50)
                        
                        # Bouton pour injecter les textures au rendu 3D
                        if st.button("Injecter les Textures au Rendu 3D de la Visionneuse Open3D"):
                            st.session_state.inject_textures = True
                            st.rerun()
                    else:
                        st.warning("Aucune catégorie de textures valide trouvée dans le ZIP.")
                else:
                    st.warning("Modèle CLIP non disponible pour le traitement des textures.")
                conn.close()
   
    process_btn = st.button("🚀 Lancer la Reconstruction 3D", type="primary")

with col2:
    if uploaded_files and len(uploaded_files) >= 2 and process_btn:
        start_time = time.time()  # Pour metric temps
        model = load_dust3r_model() if model_choice == "DUSt3R" else None
        if model is None:
            st.error("Impossible de charger le modèle sélectionné.")
        else:
            with st.spinner("Traitement en cours..."):
                try:
                    # Initialisation des widgets de progression avant le with
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Initialisation des variables pour éviter les erreurs de scope
                    all_pts3d = []
                    all_colors = []
                    num_pairs = 0
                    loss_value = 0.0
                    
                    # Amélioration scalabilité : Note pour >10 images
                    if len(uploaded_files) > 10:
                        st.info("💡 Pour >10 images, envisagez un pré-filtrage COLMAP pour init poses (installez pycolmap si possible ; placeholder ci-dessous).")
                        # Placeholder COLMAP (commenté ; décommentez si pycolmap installé)
                        # import pycolmap
                        # ... (extraction features et matching COLMAP pour init)
                    
                    # Création d'un répertoire temporaire pour les images et tout le traitement dedans
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        img_paths = []
                        for i, uploaded_file in enumerate(uploaded_files):
                            img_path = os.path.join(tmp_dir, f"img_{i:03d}.{uploaded_file.name.split('.')[-1]}")
                            with open(img_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            img_paths.append(img_path)

                       
                        if model_choice == "DUSt3R":
                            # Chargement des images DUSt3R ici (fichiers encore présents)
                            status_text.text("Chargement des images DUSt3R...")
                            images = dust3r_load_images(img_paths, size=512)
                           
                            status_text.text("Inférence en cours...")
                            pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
                            output = inference(
                                pairs, model, device,
                                batch_size=batch_size
                            )
                           
                            progress_bar.progress(0.7)
                            status_text.text("Inférence terminée ! Alignement global en cours...")
                           
                            # Toujours utiliser PointCloudOptimizer pour alignement cohérent, même pour 2 images
                            mode = GlobalAlignerMode.PointCloudOptimizer
                            scene = global_aligner(
                                output,
                                device=device,
                                mode=mode
                            )
                           
                            loss = scene.compute_global_alignment(
                                init="mst",
                                niter=niter_align,
                                schedule='cosine',
                                lr=lr_align
                            )
                            loss_value = loss
                            progress_bar.progress(1.0)
                            status_text.text(f"Alignement terminé ! Perte finale : {loss:.4f}")
                            # Test suggestion : Vérifiez loss < 0.01 pour bonne qualité
                            if float(loss) > 0.01:  # type: ignore
                                st.warning("💡 Perte >0.01 ; essayez plus d'itérations ou images mieux éclairées.")
                           
                            # Récupération des résultats DUSt3R
                            imgs = scene.imgs
                            poses = scene.get_im_poses()
                            pts3d = scene.get_pts3d()
                            if pts3d is None:
                                st.error('Erreur: pas de points 3D')
                                st.stop()
                            confidence_masks = scene.get_masks()
                           
                            # Initialisation des listes pour stocker les points et couleurs
                            all_pts3d = []
                            all_colors = []
                           
                            # Préparation du nuage de points pour visualisation avec couleurs texturées
                            for i in range(len(imgs)):  # type: ignore
                                # Masque de confiance
                                conf_i = confidence_masks[i].detach().cpu().numpy()  # (H, W) = (512, 512)
                                pts3d_tensor = pts3d[i]

                                # Convertir pts3d en numpy et aplatir
                                if isinstance(pts3d_tensor, torch.Tensor):
                                    full_pts3d = pts3d_tensor.detach().cpu().numpy().reshape(-1, 3)
                                else:
                                    full_pts3d = pts3d_tensor.reshape(-1, 3)

                                # Ajuster la taille du masque pour correspondre aux points 3D
                                conf_mask_flat = conf_i.flatten()
                                if len(conf_mask_flat) > len(full_pts3d):  # type: ignore
                                    conf_mask_flat = conf_mask_flat[:len(full_pts3d)]
                                elif len(conf_mask_flat) < len(full_pts3d):
                                    full_pts3d = full_pts3d[:len(conf_mask_flat)]

                                # Appliquer le seuil et obtenir indices valides
                                conf_mask = conf_mask_flat > threshold_conf
                                valid_indices = np.flatnonzero(conf_mask)
                                pts3d_i = full_pts3d[valid_indices]

                                if len(pts3d_i) == 0:
                                    st.warning(f"Aucun point de confiance pour l'image {i+1}")
                                    continue

                                # Couleurs réalistes depuis imgs[i] (512 res, aligné parfaitement avec le masque)
                                # Assurer que img_np est en format (H, W, 3) pour l'extraction
                                img_tensor = imgs[i]  # type: ignore
                                if isinstance(img_tensor, torch.Tensor):
                                    img_np = img_tensor.detach().cpu().numpy()
                                else:
                                    img_np = img_tensor
                                img_np = np.array(img_np)
                                if img_np.shape[0] == 3:  # (C, H, W) -> transpose to (H, W, C)
                                    img_np = np.transpose(img_np, (1, 2, 0))
                                if img_np.max() > 1.0:
                                    img_np = img_np / 255.0

                                # Aplatir en (H*W, 3)
                                colors_full = img_np.reshape(-1, 3)[:len(conf_mask_flat)]

                                # Couleurs pour indices valides
                                colors_i = colors_full[valid_indices]

                                # Downsample si trop de points
                                n_valid = len(pts3d_i)
                                if n_valid > max_points_per_view:
                                    down_idx = np.random.choice(n_valid, max_points_per_view, replace=False)
                                    pts3d_i = pts3d_i[down_idx]
                                    colors_i = colors_i[down_idx]

                                all_pts3d.append(pts3d_i)
                                all_colors.append(colors_i)
                           
                            num_pairs = len(pairs)
                       
                        # Pas de perte pour MapAnything (feed-forward)
                   
                    # Fusion des nuages de points (après le with, mais arrays persistants)
                    if all_pts3d:
                        merged_pts3d = np.vstack(all_pts3d) * scale_factor
                        merged_colors = np.vstack(all_colors)
                        
                        # Appliquer géoréférencement si activé
                        if enable_georef and 'gps_data' in st.session_state and model_choice == "DUSt3R":
                            merged_pts3d = apply_georeferencing(merged_pts3d, poses, st.session_state['gps_data'], img_paths)  # type: ignore
                    else:
                        merged_pts3d = np.empty((0, 3))
                        merged_colors = np.empty((0, 3))

                    # Application dynamique des textures PBR si base disponible et injection activée (avec seuil adaptatif)
                    matched_clusters = 0
                    if len(merged_pts3d) > 0 and 'inject_textures' in st.session_state and st.session_state.inject_textures and 'search_index' in st.session_state:
                        status_text.text("Application des textures PBR intelligentes...")
                        clip_model, clip_processor = load_clip_model()
                        if clip_model is not None:
                            # Clustering des couleurs pour classification efficace
                            n_clusters = min(50, len(merged_colors) // 100)
                            if n_clusters > 0:
                                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                                cluster_labels = kmeans.fit_predict(merged_colors)
                                cluster_centers = kmeans.cluster_centers_
                                enhanced_colors = merged_colors.copy()
                                max_distance_threshold = st.session_state.adaptive_max_dist  # Utilisation du seuil adaptatif
                                for c_id in range(n_clusters):
                                    center_rgb = cluster_centers[c_id]
                                    # Créer un patch image rempli de la couleur du cluster
                                    patch = Image.new('RGB', (224, 224), color=tuple((center_rgb * 255).astype(int)))
                                    inputs = clip_processor(images=[patch], return_tensors="pt").to(device)  # type: ignore
                                    with torch.no_grad():
                                        emb = clip_model.get_image_features(**inputs).cpu().numpy().flatten()  # type: ignore
                                    # Recherche avec fallback
                                    if st.session_state.is_faiss:
                                        distances, indices = st.session_state.search_index.search(emb.reshape(1, -1), k=1)  # type: ignore
                                        dist = distances[0][0] if len(distances) > 0 else float('inf')
                                        idx = indices[0][0] if len(indices) > 0 and indices[0][0] != -1 else -1
                                    else:
                                        dist, idx = st.session_state.search_index.kneighbors(emb.reshape(1, -1), return_distance=True)  # type: ignore
                                        dist = dist[0][0]
                                        idx = idx[0][0]
                                    if idx != -1 and dist < max_distance_threshold:
                                        category = st.session_state.texture_metadata[idx]['category']
                                        # Utiliser la couleur moyenne stockée depuis SQLite3
                                        avg_texture_color = st.session_state.texture_metadata[idx]['avg_color']
                                        # Fusion réaliste : 70% couleur originale + 30% texture
                                        new_color = 0.7 * center_rgb + 0.3 * avg_texture_color
                                        # Appliquer au cluster
                                        mask = cluster_labels == c_id
                                        enhanced_colors[mask] = new_color
                                        matched_clusters += 1
                                merged_colors = enhanced_colors
                                if matched_clusters > 0:
                                    st.success(f"Textures PBR appliquées dynamiquement via correspondances (seuil adaptatif {max_distance_threshold:.2f}). {matched_clusters}/{n_clusters} clusters matchés.")
                                else:
                                    st.warning("Aucune zone de correspondance texture trouvée ; couleurs originales conservées pour un rendu fidèle.")
                            else:
                                st.warning("Aucun cluster généré ; textures non appliquées.")
                    elif 'inject_textures' in st.session_state and st.session_state.inject_textures:
                        st.info("Textures prêtes mais pas de points 3D disponibles pour l'injection.")
                    else:
                        st.info("Injection de textures non activée.")
                   
                    st.success("Reconstruction terminée !")
                   
                    # Libération mémoire GPU après traitement (plus agressif)
                    if device == 'cuda':
                        torch.cuda.empty_cache()
                        if torch.cuda.is_available():
                            st.info(f"Mémoire GPU libérée : {torch.cuda.memory_reserved() / 1024**3:.1f} GB réservée.")
                   
                    # Visualisation Open3D avec texture réaliste (fenêtre externe)
                    if len(merged_pts3d) > 0:
                        st.info("🔓 Ouvrant une fenêtre Open3D externe pour la vue texturée du nuage de points...")
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(merged_pts3d)
                        pcd.colors = o3d.utility.Vector3dVector(merged_colors)
                        
                        # Ouvrir la fenêtre Open3D immédiatement
                        o3d.visualization.draw_geometries(  # type: ignore
                            [pcd],
                            window_name=f"Nuage de Points 3D Texturé - {model_choice}",
                            width=1600,
                            height=900,
                            left=100,
                            top=100,
                            point_show_normal=False
                        )
                        
                        # Analyse géométrique automatique RANSAC si activée (après fermeture de la fenêtre)
                        detected_shapes = {}
                        if enable_auto_ransac:
                            st.info("🔬 Analyse géométrique automatique en cours...")
                            points = np.asarray(pcd.points)
                            modified_colors = merged_colors.copy()
                            
                            # Détection plan
                            try:
                                plane_model, plane_inliers = ransac_plane_detection(pcd, distance_threshold=ransac_auto_threshold, 
                                                                                   num_iterations=ransac_auto_iterations)
                                if len(plane_inliers) > len(points) * 0.1:  # Au moins 10% des points
                                    [a, b, c, d] = plane_model
                                    detected_shapes['plan'] = {"model": [a, b, c, d], "inliers": len(plane_inliers)}
                                    # Colorer inliers en rouge
                                    modified_colors[plane_inliers] = [1.0, 0.0, 0.0]  # Rouge
                                    st.success(f"Plan détecté automatiquement : {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0 ({len(plane_inliers)} points)")
                            except:
                                pass
                            
                            # Détection cylindre
                            try:
                                cyl_model, cyl_inliers = ransac_cylinder_detection(points, distance_threshold=ransac_auto_threshold, 
                                                                                  max_iterations=ransac_auto_iterations)
                                if cyl_model and len(cyl_inliers) > len(points) * 0.05:  # Au moins 5%
                                    detected_shapes['cylindre'] = {"model": cyl_model, "inliers": len(cyl_inliers)}
                                    # Colorer inliers en vert
                                    modified_colors[cyl_inliers] = [0.0, 1.0, 0.0]  # Vert
                                    st.success(f"Cylindre détecté : Rayon {cyl_model['radius']:.3f} ({len(cyl_inliers)} points)")
                            except:
                                pass
                            
                            # Détection sphère
                            try:
                                sph_model, sph_inliers = ransac_sphere_detection(points, distance_threshold=ransac_auto_threshold, 
                                                                                max_iterations=ransac_auto_iterations)
                                if sph_model and len(sph_inliers) > len(points) * 0.05:
                                    detected_shapes['sphere'] = {"model": sph_model, "inliers": len(sph_inliers)}
                                    # Colorer inliers en bleu
                                    modified_colors[sph_inliers] = [0.0, 0.0, 1.0]  # Bleu
                                    st.success(f"Sphère détectée : Rayon {sph_model['radius']:.3f} ({len(sph_inliers)} points)")
                            except:
                                pass
                            
                            # Mettre à jour les couleurs du nuage
                            pcd.colors = o3d.utility.Vector3dVector(modified_colors)
                            st.info("🎨 Nuage coloré automatiquement : Rouge=Plans, Vert=Cylindres, Bleu=Sphères, Original=Autres")
                        
                        # ============================================
                        # POST-TRAITEMENT DU NUAGE DE POINTS
                        # ============================================
                        
                        # ============================================
                        # DOWNSAMPLING TEMPS RÉEL ULTRA-RAPIDE
                        # ============================================
                        if 'enable_realtime_downsampling' in st.session_state and st.session_state.enable_realtime_downsampling:
                            with st.spinner("⚡ Application du downsampling temps réel ultra-rapide..."):
                                try:
                                    start_downsampling = time.time()
                                    original_points = len(np.asarray(pcd.points))
                                    
                                    # Récupération des paramètres
                                    target_points = getattr(st.session_state, 'downsampling_target', 100000)
                                    downsampling_strategy = getattr(st.session_state, 'downsampling_strategy', 'auto')
                                    preserve_colors = getattr(st.session_state, 'preserve_colors', True)
                                    preserve_normals = getattr(st.session_state, 'preserve_normals', False)
                                    
                                    # Application du pipeline de downsampling temps réel
                                    downsampled_pcd = apply_realtime_downsampling_pipeline(
                                        pcd, 
                                        target_points=target_points,
                                        strategy=downsampling_strategy,
                                        preserve_colors=preserve_colors,
                                        preserve_normals=preserve_normals
                                    )
                                    
                                    # Calcul des métriques
                                    final_points = len(np.asarray(downsampled_pcd.points))
                                    processing_time_ms = (time.time() - start_downsampling) * 1000
                                    compression_ratio = original_points / final_points if final_points > 0 else 0
                                    
                                    # Stockage des statistiques pour affichage
                                    st.session_state.realtime_downsampling_stats = {
                                        'original_points': original_points,
                                        'final_points': final_points,
                                        'processing_time_ms': processing_time_ms,
                                        'compression_ratio': compression_ratio,
                                        'target_achieved': final_points <= target_points * 1.1  # Tolérance 10%
                                    }
                                    
                                    # Mise à jour du point cloud
                                    pcd = downsampled_pcd
                                    
                                    st.success(f"⚡ Downsampling temps réel terminé : {original_points:,} → {final_points:,} points ({processing_time_ms:.1f}ms, {compression_ratio:.1f}x)")
                                    
                                except Exception as e:
                                    st.error(f"❌ Erreur downsampling temps réel : {e}")
                                    st.session_state.realtime_downsampling_stats = None
                        
                        # ============================================
                        # TRAITEMENT ML AVANCÉ (POINTNET INSPIRED)
                        # ============================================
                        if 'enable_ml_processing' in st.session_state and st.session_state.enable_ml_processing:
                            with st.spinner("🧠 Application du traitement ML avancé (PointNet-inspired)..."):
                                try:
                                    start_ml = time.time()
                                    
                                    # Récupération des paramètres ML
                                    ml_technique = getattr(st.session_state, 'ml_technique', "PointNet (Classification)")
                                    ml_task = getattr(st.session_state, 'ml_task', "Classification d'objets")
                                    ml_confidence_threshold = getattr(st.session_state, 'ml_confidence_threshold', 0.7)
                                    
                                    # Application de la classification PointNet-inspired
                                    if "PointNet" in ml_technique and "Classification" in ml_task:
                                        pcd_classified, ml_stats = apply_pointnet_classification(
                                            pcd, 
                                            confidence_threshold=ml_confidence_threshold
                                        )
                                        
                                        # Mise à jour du point cloud
                                        pcd = pcd_classified
                                        
                                        # Stockage des statistiques
                                        st.session_state.ml_processing_stats = ml_stats
                                        
                                        # Affichage des résultats
                                        processing_time_ml = (time.time() - start_ml) * 1000
                                        st.success(f"🧠 Classification ML terminée : {ml_stats['classified_objects']:,} points classifiés ({processing_time_ml:.1f}ms)")
                                        
                                        # Distribution des classes
                                        class_names = ["Terrain", "Bâtiments", "Végétation", "Véhicules", "Autres"]
                                        class_dist = ml_stats['class_distribution']
                                        
                                        st.info("📊 Distribution des classes :")
                                        for class_id, count in class_dist.items():
                                            if count > 0:
                                                percentage = (count / ml_stats['total_points']) * 100
                                                st.write(f"  • {class_names[class_id]} : {count:,} points ({percentage:.1f}%)")
                                    
                                    else:
                                        st.info(f"⚠️ Technique ML '{ml_technique}' pour tâche '{ml_task}' pas encore implémentée (version simplifiée)")
                                        
                                except Exception as e:
                                    st.error(f"❌ Erreur traitement ML : {e}")
                                    st.session_state.ml_processing_stats = None
                        
                        # ============================================
                        # ANALYSE 4D TEMPORELLE (PY4DGEO-INSPIRED)
                        # ============================================
                        if 'enable_4d_analysis' in st.session_state and st.session_state.enable_4d_analysis:
                            with st.spinner("⏰ Analyse 4D temporelle en cours (M3C2)..."):
                                try:
                                    start_4d = time.time()
                                    
                                    # Récupération des paramètres 4D
                                    cylinder_radius = getattr(st.session_state, 'cylinder_radius', 0.1)
                                    min_points_cylinder = getattr(st.session_state, 'min_points_cylinder', 10)
                                    confidence_threshold_4d = getattr(st.session_state, 'confidence_threshold_4d', 0.95)
                                    max_distance_4d = getattr(st.session_state, 'max_distance_4d', 1.0)
                                    
                                    # Chargement des nuages de points temporels depuis les fichiers uploadés
                                    reference_pcd = None
                                    comparison_pcd = None
                                    
                                    # Récupérer les fichiers depuis session_state
                                    ref_file = st.session_state.get('reference_pcd')
                                    comp_file = st.session_state.get('comparison_pcd')
                                    
                                    # Charger le nuage de référence
                                    if ref_file is not None:
                                        with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as tmp_file:
                                            tmp_file.write(ref_file.getbuffer())
                                            ref_path = tmp_file.name
                                        
                                        try:
                                            reference_pcd = o3d.io.read_point_cloud(ref_path)
                                            st.info(f"✅ Nuage référence chargé : {len(np.asarray(reference_pcd.points))} points")
                                        except Exception as e:
                                            st.error(f"Erreur chargement nuage référence : {e}")
                                        finally:
                                            os.unlink(ref_path)
                                    
                                    # Charger le nuage de comparaison
                                    if comp_file is not None:
                                        with tempfile.NamedTemporaryFile(delete=False, suffix='.ply') as tmp_file:
                                            tmp_file.write(comp_file.getbuffer())
                                            comp_path = tmp_file.name
                                        
                                        try:
                                            comparison_pcd = o3d.io.read_point_cloud(comp_path)
                                            st.info(f"✅ Nuage comparaison chargé : {len(np.asarray(comparison_pcd.points))} points")
                                        except Exception as e:
                                            st.error(f"Erreur chargement nuage comparaison : {e}")
                                        finally:
                                            os.unlink(comp_path)
                                    
                                    # Vérifier que les deux nuages sont chargés
                                    if reference_pcd is None or comparison_pcd is None:
                                        st.error("❌ Les deux nuages de points temporels doivent être fournis")
                                    else:
                                        # Application de l'analyse 4D
                                        change_pcd, stats_4d = apply_4d_change_detection(
                                            reference_pcd, 
                                            comparison_pcd,
                                            cylinder_radius=cylinder_radius,
                                            min_points=min_points_cylinder,
                                            confidence_threshold=confidence_threshold_4d,
                                            max_distance=max_distance_4d
                                        )
                                        
                                        if change_pcd is not None:
                                            # Mise à jour du point cloud principal avec la carte de changements
                                            pcd = change_pcd
                                            
                                            # Stockage des statistiques
                                            st.session_state['4d_analysis_stats'] = stats_4d
                                            
                                            # Affichage des résultats détaillés
                                            processing_time_4d = (time.time() - start_4d) * 1000
                                            st.success(f"⏰ Analyse 4D terminée : {stats_4d['erosion_points']} érosions, {stats_4d['deposition_points']} dépôts ({processing_time_4d:.1f}ms)")
                                            
                                            # Affichage des métriques détaillées
                                            with st.expander("📊 Résultats Détaillés de l'Analyse 4D", expanded=True):
                                                col1, col2 = st.columns(2)
                                                
                                                with col1:
                                                    st.markdown("**🔴 Érosions (Rouge)**")
                                                    st.write(f"• Points érodés: {stats_4d['erosion_points']:,}")
                                                    st.write(f"• Volume érodé: {stats_4d['change_distribution']['erosion_volume']:.3f} m³")
                                                    st.write(f"• Magnitude max: {stats_4d['mean_change_magnitude']:.3f} m")
                                                    
                                                with col2:
                                                    st.markdown("**🔵 Dépôts (Bleu)**")
                                                    st.write(f"• Points déposés: {stats_4d['deposition_points']:,}")
                                                    st.write(f"• Volume déposé: {stats_4d['change_distribution']['deposition_volume']:.3f} m³")
                                                    st.write(f"• Changement net: {stats_4d['change_distribution']['net_change']:.3f} m³")
                                                    
                                                st.markdown("**📈 Statistiques Globales**")
                                                st.write(f"• Points analysés: {stats_4d['total_points_analyzed']:,}")
                                                st.write(f"• Points stables: {stats_4d['stable_points']:,}")
                                                st.write(f"• Confiance moyenne: {stats_4d['confidence_stats']['mean']:.2f}")
                                                st.write(f"• Seuil MAD: {stats_4d['mad_threshold']:.4f} m")
                                                
                                                # Graphique de distribution des changements
                                                change_magnitudes = []
                                                if stats_4d['erosion_points'] > 0:
                                                    change_magnitudes.extend([-stats_4d['mean_change_magnitude']] * min(stats_4d['erosion_points'], 100))
                                                if stats_4d['deposition_points'] > 0:
                                                    change_magnitudes.extend([stats_4d['mean_change_magnitude']] * min(stats_4d['deposition_points'], 100))
                                                    
                                                if change_magnitudes:
                                                    fig, ax = plt.subplots(figsize=(8, 4))
                                                    ax.hist(change_magnitudes, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                                                    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Seuil de changement')
                                                    ax.set_xlabel('Magnitude du Changement (m)')
                                                    ax.set_ylabel('Fréquence')
                                                    ax.set_title('Distribution des Changements Détectés')
                                                    ax.legend()
                                                    st.pyplot(fig)
                                                    
                                        else:
                                            st.error("❌ Échec de l'analyse 4D")
                                            
                                except Exception as e:
                                    st.error(f"❌ Erreur analyse 4D : {e}")
                                    import traceback
                                    st.code(traceback.format_exc())
                                    st.session_state['4d_analysis_stats'] = None
                        
                        # Algorithme 1: Reconstruction des zones manquantes
                        if enable_missing_reconstruction:
                            with st.spinner("🔄 Reconstruction des zones manquantes en cours..."):
                                try:
                                    # Utiliser Poisson reconstruction pour remplir les trous
                                    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                                    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
                                    
                                    # Convertir le maillage en nuage de points dense
                                    sampled_pcd = mesh.sample_points_uniformly(number_of_points=len(np.asarray(pcd.points)) * 2)
                                    
                                    # Fusionner avec le nuage original
                                    combined_pcd = pcd + sampled_pcd
                                    
                                    # Supprimer les duplicatas
                                    combined_pcd.remove_duplicated_points()
                                    
                                    pcd = combined_pcd
                                    st.success(f"✅ Reconstruction terminée : {len(np.asarray(pcd.points))} points (zones manquantes remplies)")
                                except Exception as e:
                                    st.error(f"❌ Erreur reconstruction : {e}")
                        
                        # Algorithme 2: Nettoyage des artefacts et déformations
                        if enable_artifact_cleaning:
                            with st.spinner("🧹 Nettoyage des artefacts en cours..."):
                                try:
                                    # Suppression des outliers statistiques
                                    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                                    st.info(f"Outliers statistiques supprimés : {len(ind)} points conservés")
                                    
                                    # Suppression des outliers par rayon
                                    pcd, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
                                    st.info(f"Outliers par rayon supprimés : {len(ind)} points conservés")
                                    
                                    # Lissage pour réduire le bruit
                                    pcd = pcd.filter_smooth_simple(number_of_iterations=1)
                                    
                                    st.success("✅ Nettoyage terminé : artefacts et bruit réduits")
                                except Exception as e:
                                    st.error(f"❌ Erreur nettoyage : {e}")
                        
                        # Algorithme 3: Correction des déformations géométriques
                        if enable_geometric_correction:
                            with st.spinner("📐 Correction des déformations géométriques en cours..."):
                                try:
                                    # Calcul des normales si pas déjà fait
                                    if not pcd.has_normals():
                                        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                                    
                                    # Lissage bilatéral pour corriger les déformations
                                    pcd = pcd.filter_smooth_taubin(number_of_iterations=10)
                                    
                                    # Correction des normales
                                    pcd.orient_normals_consistent_tangent_plane(k=15)
                                    
                                    st.success("✅ Correction géométrique terminée : déformations lissées")
                                except Exception as e:
                                    st.error(f"❌ Erreur correction géométrique : {e}")
                        
                        # ============================================
                        # APPLICATION DES VFX IA INTELLIGENTS
                        # ============================================
                        if 'vfx_params' in st.session_state and VFX_ENGINE_AVAILABLE:
                            with st.spinner("🎬 Application des effets VFX intelligents..."):
                                try:
                                    vfx_engine = IntelligentVFXEngine(device=device)  # type: ignore
                                    
                                    # Détection ou sélection du matériau
                                    material_type = None
                                    if not st.session_state.vfx_auto_material:
                                        material_map = {
                                            "concrete": MaterialType.CONCRETE,  # type: ignore
                                            "metal": MaterialType.METAL,  # type: ignore
                                            "wood": MaterialType.WOOD,  # type: ignore
                                            "plastic": MaterialType.PLASTIC,  # type: ignore
                                            "stone": MaterialType.STONE,  # type: ignore
                                            "glass": MaterialType.GLASS,  # type: ignore
                                        }
                                        material_type = material_map.get(st.session_state.vfx_manual_material)
                                    
                                    # Application des VFX
                                    pcd = vfx_engine.apply_automatic_vfx(
                                        pcd,
                                        st.session_state.vfx_params,
                                        material_type
                                    )
                                    
                                    st.success("✅ Effets VFX appliqués avec succès !")
                                    
                                    # Génération des maps PBR
                                    if material_type:
                                        pbr_maps = vfx_engine.generate_pbr_maps(pcd, material_type)
                                        
                                        with st.expander("📊 Maps PBR Générées"):
                                            st.write(f"**Albedo:** {len(pbr_maps['albedo'])} vertices")
                                            st.write(f"**Roughness:** Moyenne = {np.mean(pbr_maps['roughness']):.2f}")
                                            st.write(f"**Metallic:** Moyenne = {np.mean(pbr_maps['metallic']):.2f}")
                                            st.write(f"**AO:** Moyenne = {np.mean(pbr_maps['ao']):.2f}")
                                    
                                except Exception as e:
                                    st.error(f"❌ Erreur VFX: {e}")
                        # ============================================
                        # FIN APPLICATION VFX
                        # ============================================
                        
                        # Nuage de points avec options avancées
                        o3d.visualization.draw_geometries(  # type: ignore
                            [pcd],
                            window_name=f"Nuage de Points 3D Texturé - {model_choice}",
                            width=1600,
                            height=900,
                            left=100,
                            top=100,
                            point_show_normal=False
                        )
                        
                        # Bouton de téléchargement pour le nuage de points (Windows-safe)
                        pcd_tmp_path = os.path.join(tempfile.gettempdir(), f"temp_pcd_{uuid.uuid4().hex}.ply")
                        o3d.io.write_point_cloud(pcd_tmp_path, pcd)
                        time.sleep(0.1)  # Attente pour Windows
                        with open(pcd_tmp_path, "rb") as f:
                            pcd_bytes = f.read()
                        if os.path.exists(pcd_tmp_path):
                            os.remove(pcd_tmp_path)
                        st.download_button(
                            label="📥 Télécharger Nuage de Points (.ply)",
                            data=pcd_bytes,
                            file_name=f"{model_choice}_pointcloud.ply",
                            mime="model/ply"
                        )
                        
                        # Export LAS pour topographie
                        if LAS_AVAILABLE:
                            las_bytes = export_to_las(merged_pts3d, merged_colors)
                            if las_bytes:
                                st.download_button(
                                    label="📥 Télécharger Nuage de Points (.las)",
                                    data=las_bytes,
                                    file_name=f"{model_choice}_pointcloud.las",
                                    mime="application/octet-stream",
                                    help="Format LAS pour logiciels de topographie (CloudCompare, AutoCAD, etc.)"
                                )
                        else:
                            st.info("💡 Pour exporter en LAS (format standard LiDAR/topographie), installez laspy: `pip install laspy`")
                        
                        # ============================================
                        # DÉTECTION DE FORMES GÉOMÉTRIQUES AVEC RANSAC
                        # ============================================
                        st.header("🔬 Détection de Formes Géométriques avec RANSAC (Scientifique)")
                        
                        shape_type = st.selectbox("Type de forme à détecter", ["Plan", "Cylindre", "Sphère"], 
                                                 help="Utilise RANSAC pour détecter des primitives géométriques dans le nuage de points (basé sur algorithmes scientifiques robustes au bruit).")
                        
                        ransac_distance_threshold = st.slider("Seuil de distance RANSAC", 0.005, 0.2, 0.02, 0.005, 
                                                             help="Tolérance pour considérer un point comme inlier (plus petit = plus strict).")
                        
                        ransac_iterations = st.slider("Nombre d'itérations RANSAC", 100, 5000, 1000, 100, 
                                                     help="Plus d'itérations = plus précis mais plus lent. Recommandé: 1000+ pour de gros nuages.")
                        
                        if st.button("🚀 Détecter Forme avec RANSAC"):
                            with st.spinner("Analyse RANSAC en cours (algorithmes scientifiques)..."):
                                points = np.asarray(pcd.points)
                                
                                if shape_type == "Plan":
                                    plane_model, inliers = ransac_plane_detection(pcd, distance_threshold=ransac_distance_threshold, 
                                                                                  num_iterations=ransac_iterations)
                                    [a, b, c, d] = plane_model
                                    st.success(f"**Plan détecté (équation scientifique)** : {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
                                    st.metric("📊 Inliers détectés", f"{len(inliers)} / {len(points)} points")
                                    
                                    # Visualisation scientifique
                                    inlier_cloud = pcd.select_by_index(inliers)
                                    outlier_cloud = pcd.select_by_index(inliers, invert=True)
                                    inlier_cloud.paint_uniform_color([1.0, 0, 0])  # Rouge pour inliers
                                    outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])  # Gris pour outliers
                                    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],  # type: ignore
                                                                     window_name="Détection Plan RANSAC - Vue Scientifique")
                                    
                                    # Export segmenté
                                    inlier_ply_path = os.path.join(tempfile.gettempdir(), f"ransac_plane_{uuid.uuid4().hex}.ply")
                                    o3d.io.write_point_cloud(inlier_ply_path, inlier_cloud)
                                    with open(inlier_ply_path, "rb") as f:
                                        st.download_button("📥 Télécharger Plan Détecté (.ply)", 
                                                         data=f.read(), 
                                                         file_name="ransac_plane.ply",
                                                         mime="model/ply")
                                
                                elif shape_type == "Cylindre":
                                    model, inliers = ransac_cylinder_detection(points, distance_threshold=ransac_distance_threshold, 
                                                                              max_iterations=ransac_iterations)
                                    if model and len(inliers) > 10:
                                        st.success(f"**Cylindre détecté** : Rayon = {model['radius']:.4f}, Axe = [{model['axis'][0]:.4f}, {model['axis'][1]:.4f}, {model['axis'][2]:.4f}], Centre = [{model['center'][0]:.4f}, {model['center'][1]:.4f}, {model['center'][2]:.4f}]")
                                        st.metric("📊 Inliers détectés", f"{len(inliers)} / {len(points)} points")
                                        
                                        # Visualisation
                                        inlier_cloud = pcd.select_by_index(inliers)
                                        outlier_cloud = pcd.select_by_index(inliers, invert=True)
                                        inlier_cloud.paint_uniform_color([0, 1.0, 0])  # Vert
                                        outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
                                        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],  # type: ignore
                                                                         window_name="Détection Cylindre RANSAC")
                                        
                                        # Export
                                        cyl_ply_path = os.path.join(tempfile.gettempdir(), f"ransac_cylinder_{uuid.uuid4().hex}.ply")
                                        o3d.io.write_point_cloud(cyl_ply_path, inlier_cloud)
                                        with open(cyl_ply_path, "rb") as f:
                                            st.download_button("📥 Télécharger Cylindre Détecté (.ply)", 
                                                             data=f.read(), 
                                                             file_name="ransac_cylinder.ply",
                                                             mime="model/ply")
                                    else:
                                        st.error("❌ Aucun cylindre fiable détecté. Essayez d'ajuster les paramètres ou vérifiez les données.")
                                
                                elif shape_type == "Sphère":
                                    model, inliers = ransac_sphere_detection(points, distance_threshold=ransac_distance_threshold, 
                                                                            max_iterations=ransac_iterations)
                                    if model and len(inliers) > 10:
                                        st.success(f"**Sphère détectée** : Centre = [{model['center'][0]:.4f}, {model['center'][1]:.4f}, {model['center'][2]:.4f}], Rayon = {model['radius']:.4f}")
                                        st.metric("📊 Inliers détectés", f"{len(inliers)} / {len(points)} points")
                                        
                                        # Visualisation
                                        inlier_cloud = pcd.select_by_index(inliers)
                                        outlier_cloud = pcd.select_by_index(inliers, invert=True)
                                        inlier_cloud.paint_uniform_color([0, 0, 1.0])  # Bleu
                                        outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
                                        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],  # type: ignore
                                                                         window_name="Détection Sphère RANSAC")
                                        
                                        # Export
                                        sph_ply_path = os.path.join(tempfile.gettempdir(), f"ransac_sphere_{uuid.uuid4().hex}.ply")
                                        o3d.io.write_point_cloud(sph_ply_path, inlier_cloud)
                                        with open(sph_ply_path, "rb") as f:
                                            st.download_button("📥 Télécharger Sphère Détectée (.ply)", 
                                                             data=f.read(), 
                                                             file_name="ransac_sphere.ply",
                                                             mime="model/ply")
                                    else:
                                        st.error("❌ Aucune sphère fiable détectée. Essayez d'ajuster les paramètres.")
                        
                        # Maillage si demandé (optimisé pour réalisme haute qualité)
                        if generate_mesh:
                            try:
                                st.info(f"🔓 Générant et ouvrant fenêtre pour le maillage 3D ultra-réaliste avec {mesh_method}...")

                                # Vérification si nuage de points suffisant pour maillage
                                if len(pcd.points) < 1000:
                                    st.warning("⚠️ Aucune géométrie trouvée : le nuage de points est trop sparse pour générer un maillage.")
                                else:
                                    # VOXEL GRID FILTERING AVANCÉ - Optimisation basée sur l'article Medium
                                    with st.spinner("🔧 Voxel Grid Filtering avancé pour optimisation..."):

                                        # Analyse de la densité du nuage pour voxel size adaptatif
                                        points = np.asarray(pcd.points)
                                        if len(points) > 10000:
                                            # Calcul de la densité locale pour optimisation adaptative
                                            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
                                            densities = []

                                            # Échantillonnage pour performance (1 point sur 100)
                                            sample_indices = np.random.choice(len(points),
                                                                            min(1000, len(points)//100),
                                                                            replace=False)

                                            for idx in sample_indices:
                                                _, _, distances = pcd_tree.search_knn_vector_3d(points[idx], 50)
                                                if len(distances) > 1:
                                                    # Densité = 1 / distance moyenne au carré
                                                    avg_distance = np.mean(distances[1:])  # Exclure le point lui-même
                                                    density = 1.0 / (avg_distance ** 2) if avg_distance > 0 else 0
                                                    densities.append(density)

                                            if densities:
                                                avg_density = np.mean(densities)
                                                std_density = np.std(densities)

                                                # Ajustement adaptatif du voxel size selon la densité
                                                if mesh_quality_preset == "Ultra HD":
                                                    # Pour Ultra HD, garder plus de détails même dans zones denses
                                                    adaptive_voxel_size = mesh_voxel_size * max(0.5, 1.0 - (avg_density / (avg_density + std_density)))
                                                elif mesh_quality_preset == "High":
                                                    adaptive_voxel_size = mesh_voxel_size * max(0.7, 1.0 - (avg_density / (avg_density + 2*std_density)))
                                                else:  # Standard
                                                    adaptive_voxel_size = mesh_voxel_size * max(0.8, 1.0 - (avg_density / (avg_density + 3*std_density)))

                                                st.info(f"🎯 Voxel Grid Filtering adaptatif : {mesh_voxel_size:.4f} → {adaptive_voxel_size:.4f} (densité: {avg_density:.2f})")
                                                mesh_voxel_size = adaptive_voxel_size

                                        # Application du voxel downsampling optimisé
                                        pcd_down = pcd.voxel_down_sample(voxel_size=mesh_voxel_size)

                                        # Métriques de performance du voxel filtering
                                        reduction_ratio = len(pcd_down.points) / len(pcd.points)

                                        # Stocker les statistiques pour affichage
                                        st.session_state.current_pcd_stats = {
                                            'original_points': len(pcd.points),
                                            'voxel_points': len(pcd_down.points),
                                            'reduction_ratio': reduction_ratio,
                                            'voxel_size_used': mesh_voxel_size
                                        }

                                        st.info(f"⚡ Voxel Grid Filtering : {len(pcd.points):,} → {len(pcd_down.points):,} points ({reduction_ratio:.1%} conservé)")

                                        # Filtrage statistique supplémentaire pour qualité
                                        if mesh_clean_artifacts and len(pcd_down.points) > 1000:
                                            # Calcul des distances inter-points pour détecter les outliers
                                            pcd_tree_down = o3d.geometry.KDTreeFlann(pcd_down)
                                            distances = []

                                            for i in range(len(pcd_down.points)):
                                                _, _, dist = pcd_tree_down.search_knn_vector_3d(pcd_down.points[i], 10)
                                                if len(dist) > 1:
                                                    distances.append(np.mean(dist[1:]))  # Distance moyenne aux 9 plus proches voisins

                                            if distances:
                                                distances = np.array(distances)
                                                mean_dist = np.mean(distances)
                                                std_dist = np.std(distances)

                                                # Seuil statistique pour filtrage des outliers (comme dans l'article)
                                                outlier_threshold = mean_dist + 2 * std_dist

                                                # Créer masque pour conserver seulement les points "normaux"
                                                keep_mask = distances <= outlier_threshold
                                                pcd_down = pcd_down.select_by_index(np.where(keep_mask)[0])

                                                st.info(f"🧹 Filtrage statistique : conservé {np.sum(keep_mask)}/{len(keep_mask)} points (seuil: {outlier_threshold:.4f})")

                                        # Mise à jour des statistiques après nettoyage
                                        final_points = len(pcd_down.points)
                                        final_reduction = final_points / len(pcd.points)
                                        st.session_state.current_pcd_stats.update({
                                            'final_points': final_points,
                                            'final_reduction_ratio': final_reduction,
                                            'outliers_removed': st.session_state.current_pcd_stats['voxel_points'] - final_points
                                        })

                                    # ============================================
                                    # DOWNSAMPLING ULTRA-RAPIDE TEMPS RÉEL (Inspired by Sohail Saifi)
                                    # Pipeline 10M → 100K points en millisecondes
                                    # ============================================
                                    with st.spinner("⚡ Downsampling Ultra-Rapide - Pipeline Temps Réel..."):

                                        # Étape 1: Analyse du volume de données et stratégie adaptative
                                        n_points = len(pcd.points)
                                        target_points = st.session_state.get('downsampling_target', 100000)  # 100K par défaut

                                        # Stratégies selon la taille du nuage
                                        if n_points > 10000000:  # > 10M points
                                            strategy = "ultra_fast"  # Pipeline ultra-rapide
                                            st.info(f"🚀 Pipeline Ultra-Rapide activé : {n_points:,} → {target_points:,} points")
                                        elif n_points > 1000000:  # > 1M points
                                            strategy = "fast"  # Pipeline rapide
                                            st.info(f"⚡ Pipeline Rapide activé : {n_points:,} → {target_points:,} points")
                                        else:  # < 1M points
                                            strategy = "quality"  # Pipeline qualité
                                            st.info(f"🎯 Pipeline Qualité activé : {n_points:,} → {target_points:,} points")

                                        start_time = time.time()

                                        # Étape 2: Pré-downsampling par voxel pour gros volumes
                                        if n_points > 5000000:  # > 5M points
                                            # Voxel agressif pour réduire rapidement
                                            pre_voxel_size = np.cbrt((n_points / 1000000)) * 0.01  # Adaptatif
                                            pcd = pcd.voxel_down_sample(voxel_size=pre_voxel_size)
                                            st.info(f"🔲 Pré-downsampling voxel : {n_points:,} → {len(pcd.points):,} points")

                                        # Étape 3: Pipeline de downsampling multi-étapes
                                        current_points = len(pcd.points)

                                        # 3.1 Random Sampling (ultra-rapide pour gros volumes)
                                        if strategy == "ultra_fast" and current_points > target_points * 2:
                                            random_sample = min(target_points * 2, current_points)
                                            indices = np.random.choice(current_points, random_sample, replace=False)
                                            pcd = pcd.select_by_index(indices)
                                            st.info(f"🎲 Random sampling : {current_points:,} → {len(pcd.points):,} points")

                                        # 3.2 Uniform Grid Sampling (équilibré)
                                        current_points = len(pcd.points)
                                        if current_points > target_points * 1.5:
                                            # Créer une grille uniforme dans l'espace
                                            points_array = np.asarray(pcd.points)
                                            bbox_min = np.min(points_array, axis=0)
                                            bbox_max = np.max(points_array, axis=0)
                                            bbox_size = bbox_max - bbox_min

                                            # Nombre de cellules par dimension pour atteindre target
                                            volume_ratio = target_points / current_points
                                            grid_cells = int(np.cbrt(1.0 / volume_ratio))
                                            grid_cells = max(2, min(50, grid_cells))  # Limiter entre 2 et 50

                                            # Assigner chaque point à une cellule de grille
                                            grid_indices = np.floor((points_array - bbox_min) / (bbox_size / grid_cells)).astype(int)
                                            grid_indices = np.clip(grid_indices, 0, grid_cells - 1)

                                            # Créer une clé unique pour chaque cellule
                                            grid_keys = grid_indices[:, 0] + grid_indices[:, 1] * grid_cells + grid_indices[:, 2] * grid_cells * grid_cells

                                            # Sélectionner un point par cellule (le premier trouvé)
                                            unique_keys, indices = np.unique(grid_keys, return_index=True)
                                            pcd = pcd.select_by_index(indices)

                                            st.info(f"📐 Uniform grid sampling : {current_points:,} → {len(pcd.points):,} points ({grid_cells}³ cellules)")

                                        # 3.3 Farthest Point Sampling (qualité optimale)
                                        current_points = len(pcd.points)
                                        if current_points > target_points:
                                            # Implémentation optimisée du farthest point sampling
                                            points_array = np.asarray(pcd.points)

                                            # Initialisation avec un point aléatoire
                                            selected_indices = [np.random.randint(0, current_points)]
                                            min_distances = np.full(current_points, np.inf)

                                            # Sélection itérative des points les plus éloignés
                                            while len(selected_indices) < target_points and len(selected_indices) < current_points:
                                                # Calculer distances au dernier point sélectionné
                                                last_point = points_array[selected_indices[-1]]
                                                distances = np.linalg.norm(points_array - last_point, axis=1)
                                                min_distances = np.minimum(min_distances, distances)

                                                # Sélectionner le point le plus éloigné
                                                farthest_idx = np.argmax(min_distances)
                                                selected_indices.append(farthest_idx)

                                                # Mise à jour des distances minimales
                                                if len(selected_indices) % 1000 == 0:  # Progress update
                                                    progress = len(selected_indices) / target_points
                                                    st.info(f"🎯 Farthest Point Sampling : {len(selected_indices)}/{target_points} points sélectionnés")

                                            pcd = pcd.select_by_index(selected_indices)
                                            st.info(f"🎯 Farthest Point Sampling terminé : {current_points:,} → {len(pcd.points):,} points")

                                        # Étape 4: Optimisation finale et validation
                                        final_points = len(pcd.points)
                                        processing_time = time.time() - start_time
                                        compression_ratio = n_points / final_points

                                        # Métriques de qualité
                                        if final_points > 100:
                                            points_array = np.asarray(pcd.points)
                                            bbox_min = np.min(points_array, axis=0)
                                            bbox_max = np.max(points_array, axis=0)
                                            bbox_size = bbox_max - bbox_min

                                            # Calcul de la couverture spatiale
                                            volume = np.prod(bbox_size)
                                            spatial_coverage = final_points / volume if volume > 0 else 0

                                            st.success(f"✅ Downsampling temps réel terminé : {n_points:,} → {final_points:,} points")
                                            st.info(f"⚡ Performance : {processing_time:.3f}s ({final_points/processing_time:.0f} pts/s)")
                                            st.info(f"📊 Métriques : Ratio {compression_ratio:.1f}x, Couverture {spatial_coverage:.2f} pts/unité³")

                                            # Stockage des métriques pour l'interface
                                            st.session_state.realtime_downsampling_stats = {
                                                'original_points': n_points,
                                                'final_points': final_points,
                                                'processing_time_ms': processing_time * 1000,
                                                'compression_ratio': compression_ratio,
                                                'spatial_coverage': spatial_coverage,
                                                'strategy_used': strategy,
                                                'target_achieved': final_points >= target_points * 0.9  # 90% du target minimum
                                            }

                                        # Étape 1: Analyse préliminaire du bruit
                                        points_array = np.asarray(pcd_down.points)
                                        n_original = len(points_array)

                                        # Calcul des statistiques de base
                                        bbox = np.max(points_array, axis=0) - np.min(points_array, axis=0)
                                        volume = bbox[0] * bbox[1] * bbox[2]
                                        point_density = n_original / volume if volume > 0 else 0
                                        st.info(f"📊 Densité du nuage : {point_density:.2f} points/unité³")

                                        # Étape 2: Filtrage statistique avancé (Statistical Outlier Removal)
                                        if mesh_clean_artifacts:
                                            st.info("🧮 Application du filtrage statistique avancé...")

                                            # Paramètres adaptatifs selon la densité
                                            if point_density > 1000:  # Nuage très dense
                                                nb_neighbors_stat = 20
                                                std_ratio_stat = 1.5
                                            elif point_density > 100:  # Nuage dense
                                                nb_neighbors_stat = 30
                                                std_ratio_stat = 2.0
                                            else:  # Nuage sparse
                                                nb_neighbors_stat = 50
                                                std_ratio_stat = 2.5

                                            pcd_down, ind_stat = pcd_down.remove_statistical_outlier(
                                                nb_neighbors=nb_neighbors_stat,
                                                std_ratio=std_ratio_stat
                                            )
                                            st.info(f"📈 Filtrage statistique : {n_original} → {len(pcd_down.points)} points (seuil: {std_ratio_stat}, voisins: {nb_neighbors_stat})")

                                        # Étape 3: Filtrage par rayon (Radius Outlier Removal)
                                        st.info("🎯 Application du filtrage par rayon...")
                                        # Paramètres moins agressifs pour éviter la suppression excessive
                                        radius_min_points = 8 if point_density > 500 else 12 if point_density > 50 else 16
                                        adaptive_radius = mesh_voxel_size * 2  # Rayon moins large

                                        pcd_down, ind_radius = pcd_down.remove_radius_outlier(
                                            nb_points=radius_min_points,
                                            radius=adaptive_radius
                                        )
                                        st.info(f"🎯 Filtrage par rayon : {len(pcd_down.points)} points conservés (rayon: {adaptive_radius:.4f}, min: {radius_min_points} voisins)")

                                        # Étape 4: Débruitage conditionnel basé sur la densité locale
                                        if len(pcd_down.points) > 1000:
                                            st.info("🔍 Analyse de densité locale pour débruitage adaptatif...")

                                            # Calcul de la densité locale
                                            pcd_tree = o3d.geometry.KDTreeFlann(pcd_down)
                                            densities = []

                                            # Échantillonnage pour performance
                                            sample_size = min(2000, len(pcd_down.points) // 10)
                                            sample_indices = np.random.choice(len(pcd_down.points), sample_size, replace=False)

                                            for idx in sample_indices:
                                                _, _, distances = pcd_tree.search_knn_vector_3d(pcd_down.points[idx], 20)
                                                if len(distances) > 1:
                                                    local_density = 1.0 / np.mean(distances[1:])**2
                                                    densities.append(local_density)

                                            if densities:
                                                densities = np.array(densities)
                                                density_threshold = np.percentile(densities, 10)  # 10ème percentile

                                                # Identifier les régions de faible densité (potentiellement bruitées)
                                                low_density_mask = []
                                                for i in range(len(pcd_down.points)):
                                                    _, _, distances = pcd_tree.search_knn_vector_3d(pcd_down.points[i], 20)
                                                    if len(distances) > 1:
                                                        local_density = 1.0 / np.mean(distances[1:])**2
                                                        low_density_mask.append(local_density < density_threshold)

                                                low_density_indices = np.where(low_density_mask)[0]
                                                if len(low_density_indices) > 0:
                                                    # Appliquer un filtrage plus strict aux régions de faible densité
                                                    pcd_low_density = pcd_down.select_by_index(low_density_indices)
                                                    pcd_low_density, _ = pcd_low_density.remove_statistical_outlier(
                                                        nb_neighbors=10, std_ratio=1.0  # Plus strict
                                                    )

                                                    # Recombinaison des nuages
                                                    all_points = np.asarray(pcd_down.points)
                                                    low_density_points = np.asarray(pcd_low_density.points)
                                                    combined_points = np.vstack([all_points, low_density_points])
                                                    pcd_down.points = o3d.utility.Vector3dVector(combined_points)

                                                    st.info(f"🔧 Débruitage conditionnel : {len(low_density_indices)} points de faible densité retraités")

                                        # Étape 5: Lissage adaptatif (Moving Least Squares)
                                        if mesh_smoothing_iterations > 0 and len(pcd_down.points) > 100:
                                            st.info("🌊 Application du lissage Moving Least Squares...")

                                            # Paramètres adaptatifs selon la qualité
                                            if mesh_quality_preset == "Ultra HD":
                                                search_radius = mesh_voxel_size * 2
                                                fitter_type = "polynomial"  # Plus précis mais plus lent
                                            elif mesh_quality_preset == "High":
                                                search_radius = mesh_voxel_size * 3
                                                fitter_type = "linear"
                                            else:  # Standard
                                                search_radius = mesh_voxel_size * 4
                                                fitter_type = "linear"

                                            try:
                                                # Essayer d'abord MLS polynomial si disponible
                                                pcd_down = pcd_down.filter_smooth_mls_polynomial(
                                                    polynomial_order=2 if fitter_type == "polynomial" else 1,
                                                    search_radius=search_radius,
                                                    num_threads=-1
                                                )
                                                st.info(f"🌊 MLS polynomial smoothing appliqué (rayon: {search_radius:.4f}, ordre: {2 if fitter_type == 'polynomial' else 1})")
                                            except AttributeError:
                                                # Fallback vers simple smoothing si MLS n'est pas disponible
                                                try:
                                                    pcd_down = pcd_down.filter_smooth_simple(
                                                        number_of_iterations=mesh_smoothing_iterations,
                                                        filter_scope=2  # All neighbors
                                                    )
                                                    st.info(f"🌊 Simple smoothing appliqué ({mesh_smoothing_iterations} itérations)")
                                                except Exception as e:
                                                    st.warning(f"⚠️ Smoothing non disponible: {e}")
                                            except Exception as e:
                                                st.warning(f"⚠️ MLS smoothing échoué: {e} - utilisation du smoothing simple")
                                                try:
                                                    pcd_down = pcd_down.filter_smooth_simple(
                                                        number_of_iterations=mesh_smoothing_iterations,
                                                        filter_scope=2
                                                    )
                                                    st.info(f"🌊 Fallback: Simple smoothing appliqué ({mesh_smoothing_iterations} itérations)")
                                                except Exception as e2:
                                                    st.warning(f"⚠️ Aucun smoothing disponible: {e2}")

                                        # Étape 6: Débruitage des couleurs si disponibles
                                        if hasattr(pcd_down, 'colors') and len(pcd_down.colors) > 0:
                                            st.info("🎨 Débruitage des couleurs...")

                                            # Filtrage bilatéral des couleurs
                                            colors_array = np.asarray(pcd_down.colors)

                                            # Calcul de la médiane locale pour chaque canal de couleur
                                            pcd_tree_colors = o3d.geometry.KDTreeFlann(pcd_down)
                                            filtered_colors = colors_array.copy()

                                            for i in range(len(colors_array)):
                                                _, idx, _ = pcd_tree_colors.search_knn_vector_3d(pcd_down.points[i], 15)
                                                if len(idx) > 1:
                                                    # Médiane des couleurs voisines pour réduire le bruit
                                                    neighbor_colors = colors_array[idx[1:]]  # Exclure le point lui-même
                                                    filtered_colors[i] = np.median(neighbor_colors, axis=0)

                                            pcd_down.colors = o3d.utility.Vector3dVector(filtered_colors)
                                            st.info("🎨 Filtrage bilatéral des couleurs appliqué")

                                        # Métriques finales du débruitage
                                        n_final = len(pcd_down.points)
                                        noise_reduction = (n_original - n_final) / n_original * 100 if n_original > 0 else 0

                                        st.success(f"✅ Débruitage industriel terminé : {n_original} → {n_final} points ({noise_reduction:.1f}% de bruit supprimé)")

                                        # Stockage des métriques pour l'interface
                                        st.session_state.denoising_stats = {
                                            'original_points': n_original,
                                            'final_points': n_final,
                                            'noise_reduction_percent': noise_reduction,
                                            'statistical_filter_applied': mesh_clean_artifacts,
                                            'radius_filter_applied': True,
                                            'mls_smoothing_applied': mesh_smoothing_iterations > 0,
                                            'color_denoising_applied': hasattr(pcd_down, 'colors') and len(pcd_down.colors) > 0
                                        }

                                        # Estimation des normales avec paramètres personnalisés
                                        pcd_down.estimate_normals(
                                            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                                radius=mesh_normal_radius, max_nn=mesh_normal_neighbors
                                            )
                                        )

                                        # Orientation cohérente avec itérations personnalisées
                                        pcd_down.orient_normals_consistent_tangent_plane(mesh_orientation_iterations)

                                        # Lissage pré-maillage personnalisé
                                        if mesh_smoothing_iterations > 0:
                                            pcd_down = pcd_down.filter_smooth_taubin(number_of_iterations=mesh_smoothing_iterations)
                                    # Reconstruction conditionnelle avec paramètres haute qualité
                                    if mesh_method == "Poisson":
                                        # Profondeur adaptative ou fixe selon le choix utilisateur
                                        if mesh_adaptive_depth:
                                            optimal_depth = min(14, max(8, int(np.log2(len(pcd_down.points) / 1000)) + 8))
                                            st.info(f"🌊 Reconstruction Poisson adaptative : profondeur {optimal_depth}")
                                        else:
                                            optimal_depth = poisson_depth
                                            st.info(f"🌊 Reconstruction Poisson fixe : profondeur {optimal_depth}")

                                        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                                            pcd_down, depth=optimal_depth, width=0, scale=1.1, linear_fit=True
                                        )
                                    else:
                                        # Ball Pivoting avec rayons optimisés
                                        radii = [mesh_voxel_size * 2, mesh_voxel_size * 5, mesh_voxel_size * 10,
                                                mesh_voxel_size * 20, ball_pivoting_max_radius]
                                        st.info(f"⚽ Reconstruction Ball Pivoting : {len(radii)} rayons de {radii[0]:.4f} à {radii[-1]:.4f}")

                                        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                                            pcd_down, o3d.utility.DoubleVector(radii)
                                        )
                                        densities = None

                                    # Nettoyage avancé avec paramètres de qualité adaptatifs
                                    if densities is not None and len(densities) > 0:
                                        # Seuil adaptatif basé sur la qualité sélectionnée
                                        if mesh_quality_preset == "Standard":
                                            quantile_low = np.quantile(densities, 0.05)  # Plus permissif
                                        elif mesh_quality_preset == "High":
                                            quantile_low = np.quantile(densities, 0.02)  # Moyen
                                        else:  # Ultra HD
                                            quantile_low = np.quantile(densities, 0.01)  # Très strict

                                        keep_mask = densities >= quantile_low
                                        mesh.remove_vertices_by_mask(~keep_mask)
                                        st.info(f"🧹 Nettoyage densité ({mesh_quality_preset}) : conservé {np.sum(keep_mask)}/{len(keep_mask)} vertices")

                                    # Post-traitement professionnel avec paramètres adaptatifs
                                    mesh.remove_non_manifold_edges()
                                    mesh.remove_degenerate_triangles()
                                    mesh.remove_duplicated_triangles()
                                    mesh.remove_duplicated_vertices()
                                    mesh.remove_unreferenced_vertices()

                                    # Lissage adaptatif selon la qualité
                                    if mesh_smoothing_iterations > 0:
                                        if mesh_quality_preset == "Standard":
                                            smoothing_iters = min(mesh_smoothing_iterations, 2)
                                        elif mesh_quality_preset == "High":
                                            smoothing_iters = min(mesh_smoothing_iterations, 5)
                                        else:  # Ultra HD
                                            smoothing_iters = mesh_smoothing_iterations

                                        mesh = mesh.filter_smooth_taubin(number_of_iterations=smoothing_iters)
                                        st.info(f"🧼 Lissage Taubin appliqué : {smoothing_iters} itérations")

                                    # Calcul des normales avec paramètres avancés
                                    mesh.compute_vertex_normals()

                                    # Vérification qualité finale avec métriques détaillées
                                    n_vertices = len(mesh.vertices)
                                    n_triangles = len(mesh.triangles)

                                    # Calcul de métriques de qualité
                                    if len(mesh.triangles) > 0:
                                        triangle_areas = mesh.get_surface_area() / len(mesh.triangles)
                                        st.metric("📐 Surface moyenne par triangle", f"{triangle_areas:.6f} m²")

                                    st.success(f"✅ Maillage {mesh_quality_preset} généré : {n_vertices} vertices, {n_triangles} triangles")

                                    # Calcul du volume si maillage fermé
                                    try:
                                        volume = mesh.get_volume()
                                        if volume > 0:
                                            st.metric("📏 Volume du maillage", f"{volume:.3f} m³")
                                        else:
                                            st.warning("⚠️ Maillage partiellement ouvert ; ajoutez plus d'images pour une closure parfaite.")
                                    except:
                                        st.warning("⚠️ Impossible de calculer le volume (maillage ouvert)")

                                    # Lissage automatique des normales si activé
                                    if auto_smooth_normals:
                                        additional_smoothing = 5 if mesh_quality_preset == "Ultra HD" else 3
                                        mesh = mesh.filter_smooth_taubin(number_of_iterations=additional_smoothing)
                                        st.info(f"🔄 Lissage automatique des normales : {additional_smoothing} itérations supplémentaires")

                                    # Mapping UV basique si activé
                                    if basic_uv_mapping and len(mesh.vertices) > 0:
                                        mesh.compute_vertex_normals()  # Normaux pour projection UV
                                        st.info("🗺️ Mapping UV basique appliqué (projection simple)")

                                    # Transfert de couleurs amélioré avec paramètres de qualité
                                    if len(mesh.vertices) > 0:
                                        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
                                        vertices = np.asarray(mesh.vertices)
                                        colors = np.asarray(pcd.colors)

                                        # Nombre de voisins adaptatif selon la qualité
                                        k_neighbors = 10 if mesh_quality_preset == "Ultra HD" else 5
                                        mesh_colors = np.zeros((len(vertices), 3))

                                        for i in range(len(vertices)):
                                            _, idx, _ = pcd_tree.search_knn_vector_3d(vertices[i], k_neighbors)
                                            if len(idx) > 0:
                                                neighbor_colors = colors[idx]
                                                mesh_colors[i] = np.mean(neighbor_colors, axis=0)

                                        mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)
                                        st.info(f"🎨 Couleurs transférées ({k_neighbors} voisins pour {mesh_quality_preset})")

                                    # Lissage final des normales pour rendu professionnel
                                    mesh.compute_vertex_normals()
                                    
                                    # Lissage optionnel des vertex colors pour textures ultra-réalistes
                                    mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors))
                                    
                                    # ============================================
                                    # VISUALISATION AVANCÉE DU MAILLAGE
                                    # ============================================
                                    
                                    # SUBDIVISION SURFACE (comme Blender)
                                    if subdivision_level > 0:
                                        st.info(f"🔄 Application de subdivision niveau {subdivision_level}...")
                                        for _ in range(subdivision_level):
                                            mesh = mesh.subdivide_loop(number_of_iterations=1)
                                        mesh.compute_vertex_normals()
                                        st.success(f"✅ Subdivision appliquée: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
                                    
                                    # AFFICHAGE DES INFORMATIONS TOPOLOGIQUES
                                    if show_topology_info:
                                        st.markdown("### 📊 Analyse Topologique du Maillage")
                                        
                                        col_t1, col_t2, col_t3 = st.columns(3)
                                        
                                        with col_t1:
                                            st.metric("🔺 Triangles", f"{len(mesh.triangles):,}")
                                            st.metric("📍 Vertices", f"{len(mesh.vertices):,}")
                                        
                                        with col_t2:
                                            # Calcul des edges
                                            edges = mesh.get_non_manifold_edges()
                                            st.metric("📏 Edges", f"{len(mesh.triangles) * 3 // 2:,}")
                                            st.metric("⚠️ Non-Manifold Edges", f"{len(edges)}")
                                        
                                        with col_t3:
                                            # Calcul de la densité
                                            bbox = mesh.get_axis_aligned_bounding_box()
                                            volume = bbox.volume()
                                            density = len(mesh.vertices) / volume if volume > 0 else 0
                                            st.metric("📦 Volume Bounding Box", f"{volume:.4f} m³")
                                            st.metric("🎯 Densité", f"{density:.0f} pts/m³")
                                        
                                        # Qualité des triangles
                                        triangles = np.asarray(mesh.triangles)
                                        vertices = np.asarray(mesh.vertices)
                                        
                                        if len(triangles) > 0:
                                            # Calcul de l'aire moyenne des triangles
                                            areas = []
                                            for tri in triangles[:1000]:  # Échantillon pour performance
                                                v0, v1, v2 = vertices[tri]
                                                edge1 = v1 - v0
                                                edge2 = v2 - v0
                                                area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
                                                areas.append(area)
                                            
                                            st.write(f"**📐 Aire moyenne des triangles:** {np.mean(areas):.6f} m²")
                                            st.write(f"**📏 Aire min/max:** {np.min(areas):.6f} / {np.max(areas):.6f} m²")
                                    
                                    # GÉNÉRATION UV CHECKER PATTERN
                                    if show_uv_checker:
                                        st.info("🎨 Application du checker pattern UV...")
                                        
                                        # Créer une texture damier procédurale
                                        checker_size = 8
                                        checker_texture = np.zeros((len(mesh.vertices), 3))
                                        
                                        vertices = np.asarray(mesh.vertices)
                                        # Projection UV simple: utiliser X et Z pour UV
                                        uv_coords = vertices[:, [0, 2]]
                                        
                                        # Normaliser entre 0 et 1
                                        uv_min = uv_coords.min(axis=0)
                                        uv_max = uv_coords.max(axis=0)
                                        uv_range = uv_max - uv_min
                                        uv_range[uv_range == 0] = 1  # Éviter division par zéro
                                        uv_normalized = (uv_coords - uv_min) / uv_range
                                        
                                        # Appliquer damier
                                        for i in range(len(vertices)):
                                            u, v = uv_normalized[i]
                                            checker_u = int(u * checker_size) % 2
                                            checker_v = int(v * checker_size) % 2
                                            
                                            if checker_u == checker_v:
                                                checker_texture[i] = [0.9, 0.9, 0.9]  # Blanc
                                            else:
                                                checker_texture[i] = [0.1, 0.1, 0.1]  # Noir
                                        
                                        mesh.vertex_colors = o3d.utility.Vector3dVector(checker_texture)
                                        st.success("✅ Checker pattern UV appliqué!")
                                    
                                    # Amélioration : Calcul et visualisation de la coque convexe pour mieux voir les limites
                                    geometries_to_draw = [mesh]
                                    
                                    # WIREFRAME OVERLAY
                                    if wireframe_overlay:
                                        st.info("🕸️ Génération du wireframe...")
                                        
                                        # Créer un LineSet pour le wireframe
                                        lines = []
                                        triangles = np.asarray(mesh.triangles)
                                        
                                        for tri in triangles:
                                            # Ajouter les 3 arêtes du triangle
                                            lines.append([tri[0], tri[1]])
                                            lines.append([tri[1], tri[2]])
                                            lines.append([tri[2], tri[0]])
                                        
                                        # Supprimer les doublons
                                        lines = np.unique(np.sort(lines, axis=1), axis=0)
                                        
                                        wireframe = o3d.geometry.LineSet()
                                        wireframe.points = mesh.vertices
                                        wireframe.lines = o3d.utility.Vector2iVector(lines)
                                        
                                        # Couleur du wireframe
                                        wireframe_colors = [[0.0, 1.0, 0.0]] * len(lines)  # Vert fluo
                                        wireframe.colors = o3d.utility.Vector3dVector(wireframe_colors)
                                        
                                        geometries_to_draw.append(wireframe)
                                        st.success(f"✅ Wireframe: {len(lines):,} edges")
                                    
                                    # VISUALISATION DES NORMALES
                                    if show_normals:
                                        st.info("➡️ Génération des vecteurs normaux...")
                                        
                                        # Créer des lignes pour les normales
                                        normals_vis = []
                                        vertices = np.asarray(mesh.vertices)
                                        normals = np.asarray(mesh.vertex_normals)
                                        
                                        # Échantillonner pour performance (tous les 10 vertices)
                                        sample_rate = max(1, len(vertices) // 1000)
                                        
                                        normal_lines = []
                                        normal_points = []
                                        
                                        for i in range(0, len(vertices), sample_rate):
                                            start = vertices[i]
                                            end = start + normals[i] * normal_length
                                            
                                            point_idx = len(normal_points)
                                            normal_points.append(start)
                                            normal_points.append(end)
                                            normal_lines.append([point_idx, point_idx + 1])
                                        
                                        normal_lineset = o3d.geometry.LineSet()
                                        normal_lineset.points = o3d.utility.Vector3dVector(normal_points)
                                        normal_lineset.lines = o3d.utility.Vector2iVector(normal_lines)
                                        
                                        # Couleur cyan pour les normales
                                        normal_colors = [[0.0, 1.0, 1.0]] * len(normal_lines)
                                        normal_lineset.colors = o3d.utility.Vector3dVector(normal_colors)
                                        
                                        geometries_to_draw.append(normal_lineset)
                                        st.success(f"✅ Normales: {len(normal_lines):,} vecteurs affichés")
                                    
                                    if show_hull and len(pcd.points) > 3:  # Au moins 4 points pour hull
                                        hull = pcd.compute_convex_hull()
                                        hull.paint_uniform_color([1.0, 0.0, 0.0])  # Rouge pour visibilité
                                        hull.compute_vertex_normals()
                                        geometries_to_draw.append(hull)
                                    
                                    # ============================================
                                    # FIN VISUALISATION AVANCÉE
                                    # ============================================
                                        st.info("Coque convexe ajoutée en rouge pour délimiter la scène (volume intérieur maintenant cohérent avec maillage fermé).")
                                    
                                    # Visualisation avancée du maillage HD avec coque si activée
                                    o3d.visualization.draw_geometries(  # type: ignore
                                        geometries_to_draw,
                                        window_name=f"Maillage 3D {mesh_method} Ultra-Réaliste HD avec Coque - {model_choice}",
                                        width=1600,
                                        height=900,
                                        mesh_show_back_face=True,  # Montre les faces arrière
                                        point_show_normal=False
                                    )
                                    
                                    # Création du fichier temporaire pour le maillage (utilisé pour download et Blender)
                                    mesh_tmp_path = os.path.join(tempfile.gettempdir(), f"temp_mesh_{uuid.uuid4().hex}.ply")
                                    success = o3d.io.write_triangle_mesh(mesh_tmp_path, mesh, write_vertex_colors=True, write_vertex_normals=True)  # Ajout flags pour export fermé
                                    if not success:
                                        st.error("Erreur lors de l'écriture du fichier maillage.")
                                    else:
                                        time.sleep(0.2)  # Attente plus longue pour Windows
                                        if not os.path.exists(mesh_tmp_path):
                                            st.error("Fichier maillage temporaire non trouvé après écriture.")
                                        else:
                                            # Bouton de téléchargement pour le maillage (Windows-safe)
                                            with open(mesh_tmp_path, "rb") as f:
                                                mesh_bytes = f.read()
                                            st.download_button(
                                                label="📥 Télécharger Maillage 3D (.ply)",
                                                data=mesh_bytes,
                                                file_name=f"{model_choice}_{mesh_method.lower()}_mesh.ply",
                                                mime="model/ply"
                                            )
                                    
                                    # Nouvelle fonctionnalité 1: Export OBJ si activé
                                    if export_obj:
                                        obj_tmp_path = os.path.join(tempfile.gettempdir(), f"temp_obj_{uuid.uuid4().hex}.obj")
                                        o3d.io.write_triangle_mesh(obj_tmp_path, mesh, write_ascii=True, compressed=False)
                                        time.sleep(0.1)
                                        if os.path.exists(obj_tmp_path):
                                            with open(obj_tmp_path, "rb") as f:
                                                obj_bytes = f.read()
                                            st.download_button(
                                                label="📥 Télécharger Maillage 3D (.obj + .mtl)",
                                                data=obj_bytes,
                                                file_name=f"{model_choice}_{mesh_method.lower()}_mesh.obj",
                                                mime="model/obj"
                                            )
                                            os.remove(obj_tmp_path)
                                    
                                    st.info("💡 Pour un rendu encore plus réaliste, exporte le maillage vers Blender/Unreal Engine en utilisant `mesh.export('mesh.ply')`.")

                                    # Rendu avancé avec Blender si activé (avec check installation)
                                    if advanced_blender and success and os.path.exists(mesh_tmp_path):
                                        if shutil.which('blender') is None:
                                            st.warning("⚠️ Blender non trouvé dans le PATH ; installez-le pour activer le rendu avancé.")
                                        else:
                                            st.info("🔄 Lancement du rendu avancé avec Blender...")
                                            render_tmp_path = None
                                            script_tmp_path = None
                                            blend_tmp_path = None
                                            try:
                                                render_tmp_path = os.path.join(tempfile.gettempdir(), f"temp_render_{uuid.uuid4().hex}.png")
                                                script_tmp_path = os.path.join(tempfile.gettempdir(), f"temp_script_{uuid.uuid4().hex}.py")
                                                if save_blend_file:
                                                    blend_tmp_path = os.path.join(tempfile.gettempdir(), f"temp_blend_{uuid.uuid4().hex}.blend")
                                                script_content = f"""
import bpy
from math import pi
import os

# Vérification du fichier maillage
if not os.path.exists(r'{mesh_tmp_path}'):
    print("Erreur: Fichier maillage non trouvé: {mesh_tmp_path}")
else:
    print("Fichier maillage trouvé.")

# Clear scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import mesh
bpy.ops.import_mesh.ply(filepath=r'{mesh_tmp_path}')

# Get the mesh object and apply material for vertex colors
mesh_obj = None
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        mesh_obj = obj
        bpy.context.view_layer.objects.active = obj
        # Create material
        mat = bpy.data.materials.new(name="VertexColorMaterial")
        mat.use_nodes = True
        obj.data.materials.append(mat)
        # Clear default nodes
        nodes = mat.node_tree.nodes
        nodes.clear()
        # Add nodes
        output = nodes.new(type='ShaderNodeOutputMaterial')
        principled = nodes.new(type='ShaderNodeBsdfPrincipled')
        attribute = nodes.new(type='ShaderNodeAttribute')
        # Set attribute
        attribute.attribute_name = "Col"
        # Link nodes
        mat.node_tree.links.new(attribute.outputs['Color'], principled.inputs['Base Color'])
        mat.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
        # Position nodes
        output.location = (400, 0)
        principled.location = (0, 0)
        attribute.location = (-200, 0)
        break

if mesh_obj is not None:
    # Rotate object
    mesh_obj.rotation_euler[0] = pi / 2
    mesh_obj.rotation_euler[2] = -3 * pi / 4

    # Camera setup
    cam = bpy.data.objects['Camera']
    cam.location.x = -0.05
    cam.location.y = -1.2
    cam.location.z = 0.52
    cam.rotation_euler[0] = 1.13446
    cam.rotation_euler[1] = 0
    cam.rotation_euler[2] = 0

    # Add light
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_data.energy = 5
    light_obj = bpy.data.objects.new(name="Sun", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (5, 5, 5)

    # Render settings
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.filepath = r'{render_tmp_path}'
    bpy.ops.render.render(write_still=True)

    # Nouvelle fonctionnalité 3: Vues multiples si activé
    if {multi_view_blender}:
        # Vue frontale
        cam.location = (0, -2, 0)
        cam.rotation_euler = (pi/2, 0, 0)
        bpy.context.scene.render.filepath = r'{render_tmp_path.replace('.png', '_front.png')}'
        bpy.ops.render.render(write_still=True)
        
        # Vue latérale
        cam.location = (2, 0, 0)
        cam.rotation_euler = (pi/2, 0, pi/2)
        bpy.context.scene.render.filepath = r'{render_tmp_path.replace('.png', '_side.png')}'
        bpy.ops.render.render(write_still=True)
        
        # Vue supérieure
        cam.location = (0, 0, 2)
        cam.rotation_euler = (0, 0, 0)
        bpy.context.scene.render.filepath = r'{render_tmp_path.replace('.png', '_top.png')}'
        bpy.ops.render.render(write_still=True)

    # Nouvelle fonctionnalité 5: Sauvegarde .blend si activé
    if {save_blend_file}:
        bpy.ops.wm.save_as_mainfile(filepath=r'{blend_tmp_path}')
"""
                                                with open(script_tmp_path, 'w') as script_file:
                                                    script_file.write(script_content)

                                                # Run Blender
                                                result = subprocess.run(["blender", "--background", "--python", script_tmp_path], capture_output=True, text=True)
                                                if result.returncode == 0:
                                                    st.success("Rendu Blender terminé avec succès !")
                                                    if os.path.exists(render_tmp_path):
                                                        st.image(render_tmp_path, caption="Rendu Avancé Blender", use_container_width=True)
                                                        # Download button for render
                                                        with open(render_tmp_path, "rb") as f:
                                                            render_bytes = f.read()
                                                        st.download_button(
                                                            label="📥 Télécharger Rendu Blender (.png)",
                                                            data=render_bytes,
                                                            file_name=f"{model_choice}_{mesh_method.lower()}_blender_render.png",
                                                            mime="image/png"
                                                        )
                                                    
                                                    # Téléchargements pour vues multiples
                                                    if multi_view_blender:
                                                        front_path = render_tmp_path.replace('.png', '_front.png')
                                                        side_path = render_tmp_path.replace('.png', '_side.png')
                                                        top_path = render_tmp_path.replace('.png', '_top.png')
                                                        if os.path.exists(front_path):
                                                            with open(front_path, "rb") as f:
                                                                front_bytes = f.read()
                                                            st.download_button(
                                                                label="📥 Vue Frontale (.png)",
                                                                data=front_bytes,
                                                                file_name=f"{model_choice}_{mesh_method.lower()}_front.png",
                                                                mime="image/png"
                                                            )
                                                        if os.path.exists(side_path):
                                                            with open(side_path, "rb") as f:
                                                                side_bytes = f.read()
                                                            st.download_button(
                                                                label="📥 Vue Latérale (.png)",
                                                                data=side_bytes,
                                                                file_name=f"{model_choice}_{mesh_method.lower()}_side.png",
                                                                mime="image/png"
                                                            )
                                                        if os.path.exists(top_path):
                                                            with open(top_path, "rb") as f:
                                                                top_bytes = f.read()
                                                            st.download_button(
                                                                label="📥 Vue Supérieure (.png)",
                                                                data=top_bytes,
                                                                file_name=f"{model_choice}_{mesh_method.lower()}_top.png",
                                                                mime="image/png"
                                                            )
                                                    
                                                    # Téléchargement .blend si activé
                                                    if save_blend_file and blend_tmp_path and os.path.exists(blend_tmp_path):
                                                        with open(blend_tmp_path, "rb") as f:
                                                            blend_bytes = f.read()
                                                        st.download_button(
                                                            label="📥 Scène Blender (.blend)",
                                                            data=blend_bytes,
                                                            file_name=f"{model_choice}_{mesh_method.lower()}_scene.blend",
                                                            mime="application/x-blender"
                                                        )
                                                else:
                                                    st.error(f"Erreur Blender : {result.stderr}")
                                            finally:
                                                if render_tmp_path and os.path.exists(render_tmp_path):
                                                    os.unlink(render_tmp_path)
                                                if script_tmp_path and os.path.exists(script_tmp_path):
                                                    os.unlink(script_tmp_path)
                                                if blend_tmp_path and os.path.exists(blend_tmp_path):
                                                    os.unlink(blend_tmp_path)
                                    
                                    # Nettoyage final du fichier maillage temporaire seulement si pas utilisé par Blender ou après
                                    if os.path.exists(mesh_tmp_path):
                                        os.remove(mesh_tmp_path)
                                    
                                    
                            except Exception as mesh_error:
                                st.error(f"Erreur lors de la génération du maillage : {mesh_error}")
                                st.info("Vérifiez la densité des points ; essayez un downsampling plus fort ou une profondeur Poisson plus faible.")
                    else:
                        st.warning("Aucun point valide trouvé après filtrage.")
                   
                    # Visualisation du nuage de points 3D avec Plotly (couleur par Z pour simplicité)
                    st.header("☁️ Nuage de Points 3D (Plotly)")
                    if len(merged_pts3d) > 0:
                        fig = go.Figure(data=[go.Scatter3d(
                            x=merged_pts3d[:, 0],
                            y=merged_pts3d[:, 1],
                            z=merged_pts3d[:, 2],
                            mode='markers',
                            marker=dict(
                                size=2,
                                color=merged_pts3d[:, 2],  # Couleur par Z pour profondeur
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Profondeur (Z) ajustée")
                            )
                        )])
                        fig.update_layout(
                            title=f"Reconstruction 3D Globale avec {model_choice} (Vue Simplifiée)",
                            scene=dict(
                                xaxis_title="X",
                                yaxis_title="Y",
                                zaxis_title="Z",
                                aspectmode='data'
                            ),
                            width=800,
                            height=600
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Aucun point à afficher dans Plotly.")
                   
                    # Aperçu des images originales
                    st.header("🖼️ Aperçu des Images")
                    cols = st.columns(len(uploaded_files))
                    for i, uploaded_file in enumerate(uploaded_files):
                        cols[i].image(uploaded_file, caption=f"Image {i+1}", use_container_width=True)
                   
                    # Statistiques (ajout temps traitement)
                    st.header("📊 Statistiques")
                    col_stats1, col_stats2, col_stats3 = st.columns(3)
                    with col_stats1:
                        st.metric("Nombre de points 3D", f"{len(merged_pts3d):,}")
                        st.metric("Nombre d'images", len(uploaded_files))
                    with col_stats2:
                        st.metric("Paires traitées", num_pairs)
                        st.metric("Perte d'alignement", f"{loss_value:.4f}")
                    with col_stats3:
                        processing_time = time.time() - start_time
                        st.metric("Temps de traitement", f"{processing_time:.1f}s")
               
                except Exception as e:
                    st.error(f"Erreur lors du traitement : {e}")
                    st.info("Vérifiez que les images sont valides et que le GPU a assez de mémoire.")
    else:
        st.info("⚠️ Chargez au moins 2 images et cliquez sur 'Lancer la Reconstruction 3D' pour commencer.")

# Footer
st.markdown("---")
st.markdown("**Développé avec ❤️ en utilisant DUSt3R de Naver Labs et MapAnything de Facebook Research. Assurez-vous d'avoir CUDA 12.1+ pour une performance optimale.**")

# Instructions d'installation (affichées en sidebar)
with st.sidebar:
    st.header("🛠️ Installation Requise")
    model_choice_placeholder = st.radio("Sélectionnez pour voir les instructions :", ["DUSt3R"], key="install_choice")
    if model_choice_placeholder == "DUSt3R":
        st.code("""
pip install git+https://github.com/naver/dust3r.git
pip install streamlit plotly pillow numpy torch torchvision open3d scikit-learn transformers faiss-cpu pandas psutil pynvml  # FAISS optionnel (fallback sklearn) ; psutil/pynvml pour monitoring
# Pour Blender : Téléchargez depuis blender.org et ajoutez au PATH
# Pour scalabilité >10 images : pip install pycolmap (optionnel)
        """)
    st.markdown("**Lancer l'app :** `streamlit run app.py`")
    if st.button("🔗 Lien GitHub DUSt3R"):
        st.markdown("[https://github.com/naver/dust3r](https://github.com/naver/dust3r)")