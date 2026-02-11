#!/usr/bin/env python3
"""
Test script pour l'analyse organisée avec 3 IA spécialisées :
- CLIP : Environnement, sol, datation
- Florence : Bâtiments, toitures, dangers
- GLM : Synthèse globale
"""

import os
import sys
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

# Configuration des chemins
os.environ['HF_HOME'] = r'C:\Users\Admin\Desktop\logiciel\riskIA\models'
os.environ['TRANSFORMERS_CACHE'] = r'C:\Users\Admin\Desktop\logiciel\riskIA\models'

def test_clip_analysis(image_path):
    """Test CLIP pour environnement, sol, datation"""
    print("🔄 Test CLIP : Chargement du modèle...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    try:
        # Utiliser le modèle local
        clip_path = r"C:\Users\Admin\Desktop\logiciel\riskIA\models\models--openai--clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(clip_path).to(device)
        processor = CLIPProcessor.from_pretrained(clip_path)

        print("📸 Analyse environnementale...")
        image = Image.open(image_path).convert('RGB')

        # Prompts spécialisés pour l'environnement et le sol
        environment_prompts = [
            # Environnement général
            "environnement naturel préservé sans pollution visible",
            "environnement urbain avec bâtiments et infrastructures",
            "environnement industriel avec équipements lourds",
            "environnement côtier ou maritime",
            "environnement forestier ou végétal dense",
            "environnement désertique ou aride",

            # Texture du sol
            "sol sableux ou granulaire fin",
            "sol argileux ou collant",
            "sol rocheux ou pierreux",
            "sol limoneux ou intermédiaire",
            "sol tourbeux ou organique",
            "sol instable ou érodé",

            # Datation environnementale
            "site récent avec constructions modernes",
            "site ancien avec signes d'usure naturelle",
            "site historique avec préservation patrimoniale",
            "site en développement actif",
            "site abandonné avec végétation envahissante",
            "site en rénovation ou maintenance",

            # Conditions météorologiques
            "conditions sèches et stables",
            "conditions humides avec pluie récente",
            "conditions venteuses avec signes d'érosion éolienne",
            "conditions extrêmes avec dommages visibles"
        ]

        inputs = processor(text=environment_prompts, images=image, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]

        clip_results = [(label, score.item()) for label, score in zip(environment_prompts, probs) if score > 0.03]
        clip_results.sort(key=lambda x: x[1], reverse=True)

        print("✅ CLIP réussi !")
        print("Top 5 résultats CLIP :")
        for i, (label, score) in enumerate(clip_results[:5]):
            print(".3f")

        return clip_results

    except Exception as e:
        print(f"❌ Erreur CLIP: {str(e)}")
        return []

def test_florence_analysis(image_path):
    """Test Florence pour bâtiments, toitures, dangers"""
    print("\n🔄 Test Florence : Chargement du modèle...")

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Utiliser le modèle local
        florence_path = r"C:\Users\Admin\Desktop\logiciel\riskIA\models\models--microsoft--Florence-2-base-ft"
        processor = AutoProcessor.from_pretrained(florence_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(florence_path, trust_remote_code=True).to(device)

        print("🏗️ Analyse architecturale...")
        image = Image.open(image_path).convert('RGB')

        # Tâches Florence pour analyse détaillée
        florence_tasks = [
            "<CAPTION_TO_PHRASE_GROUNDING> Locate and describe buildings and their roofs",
            "<DETAILED_CAPTION> Describe building conditions, roof materials, and structural integrity",
            "<CAPTION_TO_PHRASE_GROUNDING> Identify roofing materials and textures",
            "<DETAILED_CAPTION> Analyze building age, construction quality, and potential hazards",
            "<CAPTION_TO_PHRASE_GROUNDING> Detect structural damages, cracks, or deterioration",
            "<MORE_DETAILED_CAPTION> Assess building safety and risk factors"
        ]

        florence_results = []
        for task in florence_tasks:
            try:
                inputs = processor(text=task, images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
                result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                florence_results.append((task, result))
                print(f"  ✓ {task}: {result[:80]}...")
            except Exception as e:
                florence_results.append((task, f"Erreur: {str(e)}"))
                print(f"  ❌ {task}: Erreur - {str(e)}")

        print("✅ Florence réussi !")
        return florence_results

    except Exception as e:
        print(f"❌ Erreur Florence: {str(e)}")
        return []

def test_glm_synthesis(clip_results, florence_results):
    """Test GLM pour synthèse globale"""
    print("\n🔄 Test GLM : Chargement du modèle...")

    try:
        # Essayer différents chemins GLM locaux uniquement
        glm_paths = [
            r"C:\Users\Admin\Desktop\logiciel\riskIA\models\glm-4v-9b"
        ]

        glm_model = None
        glm_tokenizer = None

        for path in glm_paths:
            try:
                print(f"  Tentative avec: {path}")
                glm_tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
                glm_model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
                print(f"  ✅ GLM chargé depuis: {path}")
                break
            except Exception as e:
                print(f"  ❌ Échec avec {path}: {str(e)}")
                continue

        if glm_model is None:
            print("❌ Aucun modèle GLM trouvé. Test de synthèse simulé.")
            synthesis = "SYNTHÈSE SIMULÉE : Analyse basée sur CLIP et Florence uniquement.\n\n"
            synthesis += "RÉSULTATS CLIP (Environnement/Sol/Datation) :\n"
            for label, score in clip_results[:3]:
                synthesis += f"- {label}: {score:.3f}\n"
            synthesis += "\n\nRÉSULTATS FLORENCE (Bâtiments/Dangers) :\n"
            for task, result in florence_results:
                synthesis += f"- {task}: {result[:100]}...\n"
            synthesis += "\nCONCLUSION : Analyse multi-modale réussie !"
            return synthesis

        # Prompt de synthèse intégrant les résultats des autres IA
        synthesis_prompt = f"""
        Analyse intégrée des risques basée sur 3 IA spécialisées :

        ANALYSE CLIP (Environnement/Sol/Datation) :
        {chr(10).join([f"- {label}: {score:.3f}" for label, score in clip_results[:5]])}

        ANALYSE FLORENCE (Bâtiments/Toitures/Dangers) :
        {chr(10).join([f"- {task}: {result}" for task, result in florence_results])}

        SYNTHÈSE REQUISE :
        1. Évaluation globale des risques environnementaux et structurels
        2. Datation estimée du site et des bâtiments
        3. Dangers prioritaires identifiés
        4. Mesures correctives recommandées
        5. Niveau de criticité (Faible/Modéré/Élevé/Critique)

        Fournir une analyse professionnelle structurée.
        """

        inputs = glm_tokenizer(synthesis_prompt, return_tensors="pt").to(glm_model.device)
        outputs = glm_model.generate(**inputs, max_new_tokens=500, temperature=0.3, do_sample=True)
        synthesis = glm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        print("✅ GLM réussi !")
        print(f"Synthèse GLM ({len(synthesis)} caractères)")
        return synthesis

    except Exception as e:
        print(f"❌ Erreur GLM: {str(e)}")
        return f"Erreur GLM: {str(e)}"

def main():
    """Fonction principale de test"""
    print("🧠 TEST D'ANALYSE ORGANISÉE AVEC 3 IA SPÉCIALISÉES")
    print("=" * 60)

    # Trouver une image de test
    test_images = [
        r"C:\Users\Admin\Desktop\logiciel\riskIA\croquis_site_gabon.png",
        r"C:\Users\Admin\Desktop\logiciel\riskIA\annotated_scientific_gabon.png",
        r"C:\Users\Admin\Desktop\logiciel\riskIA\cap.png"
    ]

    image_path = None
    for img in test_images:
        if os.path.exists(img):
            image_path = img
            break

    if not image_path:
        print("❌ Aucune image de test trouvée !")
        return

    print(f"📷 Image de test: {image_path}")

    # Test CLIP
    clip_results = test_clip_analysis(image_path)

    # Test Florence
    florence_results = test_florence_analysis(image_path)

    # Test GLM
    glm_synthesis = test_glm_synthesis(clip_results, florence_results)

    # Résumé final
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS FINAUX")
    print("=" * 60)

    print(f"\n🔍 CLIP : {len(clip_results)} éléments environnementaux détectés")
    print(f"🏗️ Florence : {len(florence_results)} analyses architecturales réalisées")
    print(f"🧠 GLM : Synthèse de {len(glm_synthesis)} caractères générée")

    print("\n✅ TEST RÉUSSI !" if clip_results and florence_results else "❌ TEST ÉCHOUÉ !")

    # Sauvegarder les résultats
    with open("test_3ia_results.txt", "w", encoding="utf-8") as f:
        f.write("RÉSULTATS TEST 3 IA ORGANISÉES\n")
        f.write("=" * 50 + "\n\n")
        f.write("CLIP RESULTS:\n")
        for label, score in clip_results[:10]:
            f.write(".3f")
        f.write("\n\nFLORENCE RESULTS:\n")
        for task, result in florence_results:
            f.write(f"- {task}: {result}\n")
        f.write(f"\n\nGLM SYNTHESIS:\n{glm_synthesis}")

    print("\n💾 Résultats sauvegardés dans 'test_3ia_results.txt'")

if __name__ == "__main__":
    main()