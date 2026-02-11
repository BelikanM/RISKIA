"""
Script pour faire une VRAIE analyse d'image sans inventions
Utilise uniquement les détections réelles de Florence-2, CLIP et OpenCV
"""

import sys
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import CLIPProcessor, CLIPModel

def analyser_image_reelle(image_path):
    """Analyse RÉELLE d'une image - aucune donnée inventée"""
    
    print("\n" + "="*70)
    print("🔍 ANALYSE RÉELLE DE L'IMAGE (ZÉRO INVENTION)")
    print("="*70)
    
    # Charger l'image
    image = Image.open(image_path).convert('RGB')
    print(f"✅ Image chargée: {image.size[0]}x{image.size[1]} pixels")
    
    # === 1. FLORENCE-2 - DÉTECTION D'OBJETS RÉELS ===
    print("\n📸 Étape 1: Florence-2 détecte les objets RÉELS...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    florence_model_path = "C:\\Users\\Admin\\Desktop\\logiciel\\florence2_model"
    
    try:
        florence_processor = AutoProcessor.from_pretrained(
            florence_model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        florence_model = AutoModelForCausalLM.from_pretrained(
            florence_model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            attn_implementation="eager"
        )
        florence_model = florence_model.to(device)  # type: ignore
        florence_model.eval()
        
        # Description détaillée
        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = florence_processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=150,
                num_beams=1,
                use_cache=False
            )
        
        description = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        description_clean = description.replace(prompt, "").strip()
        
        print(f"\n📝 DESCRIPTION FLORENCE-2 (ce qu'il voit réellement):")
        print(f"   {description_clean}")
        print(f"   Longueur: {len(description_clean.split())} mots")
        
        # Détection d'objets
        prompt_od = "<OD>"
        inputs_od = florence_processor(text=prompt_od, images=image, return_tensors="pt")
        inputs_od = {k: v.to(device) for k, v in inputs_od.items()}
        
        with torch.no_grad():
            generated_ids_od = florence_model.generate(
                input_ids=inputs_od["input_ids"],
                pixel_values=inputs_od["pixel_values"],
                max_new_tokens=200,
                num_beams=1,
                use_cache=False
            )
        
        result_od = florence_processor.batch_decode(generated_ids_od, skip_special_tokens=False)[0]
        
        # Parser les objets détectés
        florence_objects = []
        if "<OD>" in result_od:
            parsed = florence_processor.post_process_generation(
                result_od, 
                task="<OD>", 
                image_size=(image.width, image.height)
            )
            if parsed and '<OD>' in parsed:
                od_result = parsed['<OD>']
                if 'bboxes' in od_result and 'labels' in od_result:
                    for bbox, label in zip(od_result['bboxes'], od_result['labels']):
                        florence_objects.append({
                            'label': label,
                            'bbox': [int(b) for b in bbox],
                            'confidence': 1.0
                        })
        
        print(f"\n🎯 OBJETS DÉTECTÉS PAR FLORENCE-2: {len(florence_objects)}")
        for i, obj in enumerate(florence_objects, 1):
            print(f"   {i}. {obj['label']} - position {obj['bbox']}")
        
    except Exception as e:
        print(f"⚠️ Erreur Florence-2: {e}")
        description_clean = "Analyse Florence-2 non disponible"
        florence_objects = []
    
    # === 2. CLIP - CLASSIFICATION PRÉCISE ===
    print("\n🤖 Étape 2: CLIP identifie le contenu RÉEL...")
    
    try:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = clip_model.to(device)  # type: ignore
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Labels spécifiques pour détecter la végétation, sols, etc.
        labels = [
            # Végétation
            "dense green vegetation", "forest with many trees", "tropical plants", 
            "grass and bushes", "palm trees", "thick jungle", "green foliage",
            
            # Sols et terrains
            "bare soil ground", "dirt road", "sandy terrain", "rocky ground",
            "muddy surface", "gravel path", "paved area",
            
            # Éléments industriels
            "industrial equipment", "metal storage tank", "electrical transformer",
            "industrial piping", "machinery", "warehouse building", "factory structure",
            
            # Eau et ciel
            "water body", "river or stream", "pond or lake", "cloudy sky",
            "clear blue sky", "overcast weather",
            
            # Éléments construits
            "concrete building", "metal fence", "industrial facility", 
            "construction equipment", "vehicles", "containers"
        ]
        
        inputs = clip_processor(text=labels, images=image, return_tensors="pt", padding=True)  # type: ignore
        inputs = {k: v.to(device) for k, v in inputs.items()}  # type: ignore
        
        with torch.no_grad():
            outputs = clip_model(**inputs)
        
        probs = outputs.logits_per_image.softmax(dim=1)[0]
        
        # Seuil bas pour détecter même les éléments subtils
        detected = [(label, score.item()) for label, score in zip(labels, probs) if score > 0.02]
        detected.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n✅ CLIP A DÉTECTÉ {len(detected)} ÉLÉMENTS RÉELS:")
        for label, score in detected[:15]:
            print(f"   • {label}: {score:.2%} de confiance")
        
        # Compter par catégorie
        vegetation_items = [d for d in detected if any(kw in d[0].lower() for kw in ['vegetation', 'tree', 'forest', 'plant', 'grass', 'bush', 'jungle', 'foliage'])]
        industrial_items = [d for d in detected if any(kw in d[0].lower() for kw in ['industrial', 'tank', 'transformer', 'machinery', 'equipment', 'factory', 'warehouse'])]
        soil_items = [d for d in detected if any(kw in d[0].lower() for kw in ['soil', 'ground', 'dirt', 'terrain', 'sandy', 'rocky', 'gravel'])]
        water_items = [d for d in detected if any(kw in d[0].lower() for kw in ['water', 'river', 'stream', 'pond', 'lake'])]
        
        print(f"\n📊 STATISTIQUES RÉELLES PAR CATÉGORIE:")
        print(f"   🌿 Végétation: {len(vegetation_items)} éléments")
        print(f"   🏭 Industriel: {len(industrial_items)} éléments")
        print(f"   🏜️  Sols/Terrains: {len(soil_items)} éléments")
        print(f"   💧 Eau: {len(water_items)} éléments")
        
    except Exception as e:
        print(f"⚠️ Erreur CLIP: {e}")
        detected = []
        vegetation_items = []
        industrial_items = []
        soil_items = []
        water_items = []
    
    # === 3. OPENCV - ANALYSE TEXTURES ET FEATURES ===
    print("\n🔬 Étape 3: OpenCV analyse les TEXTURES RÉELLES...")
    
    try:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        img_hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        
        total_pixels = img_hsv.shape[0] * img_hsv.shape[1]
        
        # Détecter végétation (vert) avec seuils élargis
        lower_green = np.array([20, 15, 15])  # Seuils très bas
        upper_green = np.array([100, 255, 255])
        mask_vegetation = cv2.inRange(img_hsv, lower_green, upper_green)
        vegetation_pixels = cv2.countNonZero(mask_vegetation)
        vegetation_pct = (vegetation_pixels / total_pixels) * 100
        
        # Détecter sols (marron, beige, ocre)
        lower_soil = np.array([5, 10, 20])
        upper_soil = np.array([35, 180, 180])
        mask_soil = cv2.inRange(img_hsv, lower_soil, upper_soil)
        soil_pixels = cv2.countNonZero(mask_soil)
        soil_pct = (soil_pixels / total_pixels) * 100
        
        # Détecter surfaces métalliques (gris)
        lower_metal = np.array([0, 0, 80])
        upper_metal = np.array([180, 60, 220])
        mask_metal = cv2.inRange(img_hsv, lower_metal, upper_metal)
        metal_pixels = cv2.countNonZero(mask_metal)
        metal_pct = (metal_pixels / total_pixels) * 100
        
        # Détecter eau (bleu)
        lower_water = np.array([85, 40, 40])
        upper_water = np.array([135, 255, 255])
        mask_water = cv2.inRange(img_hsv, lower_water, upper_water)
        water_pixels = cv2.countNonZero(mask_water)
        water_pct = (water_pixels / total_pixels) * 100
        
        print(f"\n🎨 COMPOSITION RÉELLE DE L'IMAGE:")
        print(f"   🌿 Végétation: {vegetation_pct:.1f}% ({vegetation_pixels:,} pixels)")
        print(f"   🏜️  Sol/Terrain: {soil_pct:.1f}% ({soil_pixels:,} pixels)")
        print(f"   ⚙️  Métal/Structure: {metal_pct:.1f}% ({metal_pixels:,} pixels)")
        print(f"   💧 Eau: {water_pct:.1f}% ({water_pixels:,} pixels)")
        
        # Contours
        edges = cv2.Canny(img_gray, 30, 100)  # Seuils réduits
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_significant = [c for c in contours if cv2.contourArea(c) > 50]
        
        # Cercles
        circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=15,
                                   param1=40, param2=25, minRadius=5, maxRadius=150)
        num_circles = len(circles[0]) if circles is not None else 0
        
        # Lignes
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=10)
        num_lines = len(lines) if lines is not None else 0
        
        print(f"\n📐 STRUCTURES GÉOMÉTRIQUES DÉTECTÉES:")
        print(f"   📦 Contours: {len(contours_significant)}")
        print(f"   ⭕ Cercles: {num_circles}")
        print(f"   📏 Lignes: {num_lines}")
        
    except Exception as e:
        print(f"⚠️ Erreur OpenCV: {e}")
        vegetation_pct = 0
        soil_pct = 0
        metal_pct = 0
        water_pct = 0
        contours_significant = []
        num_circles = 0
        num_lines = 0
    
    # === RÉSUMÉ FINAL ===
    print("\n" + "="*70)
    print("📋 RÉSUMÉ DE L'ANALYSE RÉELLE")
    print("="*70)
    print(f"Description Florence-2: {len(description_clean.split())} mots")
    print(f"Objets Florence-2: {len(florence_objects)}")
    print(f"Éléments CLIP détectés: {len(detected)}")
    print(f"  - Végétation: {len(vegetation_items)} éléments")
    print(f"  - Industriel: {len(industrial_items)} éléments")
    print(f"  - Sols: {len(soil_items)} éléments")
    print(f"  - Eau: {len(water_items)} éléments")
    print(f"Composition OpenCV:")
    print(f"  - Végétation: {vegetation_pct:.1f}%")
    print(f"  - Sol: {soil_pct:.1f}%")
    print(f"  - Métal: {metal_pct:.1f}%")
    print(f"  - Eau: {water_pct:.1f}%")
    print(f"Structures: {len(contours_significant)} contours, {num_circles} cercles, {num_lines} lignes")
    print("="*70)
    
    return {
        'description': description_clean,
        'florence_objects': florence_objects,
        'clip_detections': detected,
        'vegetation_elements': len(vegetation_items),
        'industrial_elements': len(industrial_items),
        'soil_elements': len(soil_items),
        'water_elements': len(water_items),
        'vegetation_pct': vegetation_pct,
        'soil_pct': soil_pct,
        'metal_pct': metal_pct,
        'water_pct': water_pct,
        'opencv_contours': len(contours_significant),
        'opencv_circles': num_circles,
        'opencv_lines': num_lines
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyse_reelle.py <chemin_image>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    resultats = analyser_image_reelle(image_path)
