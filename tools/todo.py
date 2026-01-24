import time
from typing import List, Optional
import re
import os
from dotenv import load_dotenv

load_dotenv()

def analyze_query_type(prompt: str) -> dict:
    """Analyse le type de requête pour adapter la stratégie de réflexion"""
    prompt_lower = prompt.lower()

    analysis = {
        "type": "general",
        "needs_web": False,
        "needs_memory": False,
        "needs_docs": False,
        "complexity": "simple",
        "temporal": False,
        "geographical": False,
        "subjects": [],  # Liste des sujets extraits
        "intent": "general",
        "sentiment": "neutral",
        "needs_cpt_data": False,
        "needs_calculation": False
    }

    # Mots-clés pour CPT et géotechnique - TOUJOURS besoin de RAG
    cpt_keywords = ["cpt", "cp tu", "qc", "fs", "ic", "résistance", "géotechnique", "sol", "profondeur",
                   "sable", "argile", "limon", "classification", "robertson", "liquefaction", "pieu",
                   "fondation", "portance", "statistiques", "moyenne", "écart", "correlation",
                   "construction", "bâtiment", "viable", "civil", "ouvrage", "stabilité", "sécurité",
                   "construire", "bâtir", "édifier", "ériger", "implanter", "implantation"]

    # Mots-clés contextuels pour la construction
    construction_context = ["terrain", "site", "emplacement", "zone", "secteur", "parcelle", "lot",
                           "constructible", "constructibilité", "urbanisme", "permis", "autorisation"]

    # Détecter si c'est une question de construction/terrain
    is_construction_question = any(kw in prompt_lower for kw in cpt_keywords) or \
                              (any(kw in prompt_lower for kw in construction_context) and \
                               any(kw in prompt_lower for kw in ["construire", "construction", "bâtir", "viable", "possible", "faisable"]))

    if is_construction_question:
        analysis["needs_docs"] = True  # Toujours chercher dans les données CPT pour les questions de construction
        analysis["needs_cpt_data"] = True
        analysis["type"] = "technical_cpt"
        analysis["intent"] = "question"
        
        # Extraire les sujets pertinents
        found_subjects = []
        for kw in cpt_keywords + construction_context:
            if kw in prompt_lower:
                found_subjects.append(kw)
        analysis["subjects"] = found_subjects[:5]  # Limiter à 5 sujets
        
        if any(kw in prompt_lower for kw in ["statistiques", "moyenne", "écart", "correlation", "analyse"]):
            analysis["complexity"] = "complex"  # Analyses statistiques sont complexes

    temporal_keywords = ["aujourd'hui", "maintenant", "récent", "actuel", "dernier", "2024", "2025"]
    if any(kw in prompt_lower for kw in temporal_keywords):
        analysis["temporal"] = True
        analysis["needs_web"] = True

    geo_keywords = ["gabon", "libreville", "port-gentil", "franceville", "oyem", "où", "localisation"]
    if any(kw in prompt_lower for kw in geo_keywords):
        analysis["geographical"] = True

    doc_keywords = ["selon le document", "d'après le pdf", "dans le fichier", "uploadé"]
    if any(kw in prompt_lower for kw in doc_keywords):
        analysis["needs_docs"] = True
        analysis["type"] = "document_query"

    continuation_keywords = ["ils", "elles", "lui", "leur", "donc", "alors", "ensuite", "aussi", "également"]
    if any(kw in prompt_lower for kw in continuation_keywords) or len(prompt.split()) < 5:
        analysis["needs_memory"] = True
        analysis["type"] = "continuation"

    if len(prompt.split()) > 15 or (prompt.count("?") > 1):
        analysis["complexity"] = "complex"
    elif any(kw in prompt_lower for kw in ["pourquoi", "comment", "expliquer"]):
        analysis["complexity"] = "medium"

    web_keywords = ["actualité", "news", "prix", "cours", "météo", "horaire"]
    if any(kw in prompt_lower for kw in web_keywords):
        analysis["needs_web"] = True
        analysis["type"] = "real_time"
        analysis["subjects"] = [kw for kw in web_keywords if kw in prompt_lower]

    # Pour les questions techniques sur CPT, activer aussi le web si c'est statistique général
    if analysis["type"] == "technical_cpt":
        analysis["needs_web"] = True  # Toujours chercher des références pour CPT

    return analysis

def detect_subject_shift(prompt: str, current_subject: str, subject_keywords: List[str]) -> dict:
    """Détecte un changement de sujet et évalue la force du changement"""
    if not current_subject or not subject_keywords:
        return {"shift_detected": False, "shift_strength": 0.0, "new_subject_detected": True, "reason": "Init"}
    
    prompt_lower = prompt.lower()
    prompt_words = set(re.findall(r'\b\w{4,}\b', prompt_lower))
    keyword_overlap = len(prompt_words.intersection(set(subject_keywords)))
    overlap_ratio = keyword_overlap / max(len(subject_keywords), 1)
    
    shift_markers = ["maintenant", "sinon", "autre chose", "parlons de", "passons à", "nouveau sujet"]
    has_shift_marker = any(marker in prompt_lower for marker in shift_markers)
    
    shift_strength = 0.0
    if overlap_ratio < 0.2: shift_strength += 0.5
    if has_shift_marker: shift_strength += 0.3
    
    return {
        "shift_detected": shift_strength > 0.4,
        "shift_strength": shift_strength,
        "new_subject_detected": shift_strength > 0.6,
        "reason": f"Overlap: {overlap_ratio:.1%}"
    }

def generate_search_strategy(analysis: dict, subject_keywords: List[str], geo_info: dict) -> dict:
    """Génère une stratégie de recherche optimisée basée sur l'analyse d'intention"""
    strategy = {
        "use_rag": analysis.get("needs_cpt_data", False),
        "use_memory": analysis.get("needs_memory", False),
        "use_web": analysis.get("needs_web", False),
        "memory_k": 5, "rag_k": 3,
        "web_enhanced": False, "search_query_suffix": ""
    }
    
    # Forcer RAG pour les sujets CPT
    if any(kw in ["cpt", "géotechnique", "sol"] for kw in subject_keywords):
        strategy["use_rag"] = True
        strategy["rag_k"] = 5  # Augmenter le nombre de documents
        strategy["use_web"] = True  # Forcer aussi le web pour les sujets CPT
    
    if analysis.get("complexity") == "complex" or analysis.get("intent") in ["analyse", "comparaison"]:
        strategy.update({"memory_k": 8, "rag_k": 5})
    
    if strategy["use_web"]:
        strategy["web_enhanced"] = True
        suffix = " ".join(subject_keywords[:3]) if subject_keywords else ""
        strategy["search_query_suffix"] = f"{suffix} {geo_info.get('city', 'Gabon')}"
    
    return strategy

def analyze_query_intent(prompt: str, model=None) -> dict:
    """Analyse l'intention et le sentiment de la requête en utilisant l'IA"""
    if not model:
        return analyze_query_type(prompt)  # Fallback
    
    intent_prompt = f"""Analysez cette requête utilisateur et déterminez:
1. L'intention principale (analyse, question, calcul, comparaison, statistiques, etc.)
2. Le sentiment (neutre, positif, négatif, incertain)
3. Les sujets/notions précis à rechercher
4. Si besoin de données CPT, web, calculs

Requête: {prompt}

Répondez en JSON avec les clés: intent, sentiment, subjects, needs_cpt_data, needs_web, needs_calculation"""

    try:
        # Utiliser le modèle pour générer la réponse
        response = model(intent_prompt, max_new_tokens=200, temperature=0.1, do_sample=False)
        generated_text = response[0]['generated_text'].strip()
        
        # Extraire le JSON de la réponse
        import json
        import re
        json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "intent": result.get("intent", "general"),
                "sentiment": result.get("sentiment", "neutral"),
                "subjects": result.get("subjects", []),
                "needs_cpt_data": result.get("needs_cpt_data", True),
                "needs_web": result.get("needs_web", False),
                "needs_calculation": result.get("needs_calculation", False)
            }
    except Exception as e:
        print(f"Erreur analyse intention: {e}")
    
    return analyze_query_type(prompt)  # Fallback

def execute_reflection_plan(
    prompt: str, 
    geo_info: Optional[dict] = None, 
    messages: Optional[List] = None,
    current_subject: Optional[str] = None,
    subject_keywords: Optional[List[str]] = None,
    model=None
):
    """Phase de réflexion structurée avec analyse d'intention IA"""
    geo_info = geo_info or {}
    subject_keywords = subject_keywords or []
    
    # TEMPORAIREMENT DÉSACTIVÉ : Utiliser l'analyse par règles pour éviter les hallucinations
    # if model:
    #     query_analysis = analyze_query_intent(prompt, model)
    # else:
    #     query_analysis = analyze_query_type(prompt)
    
    query_analysis = analyze_query_type(prompt)  # Utiliser l'analyse par règles (plus fiable)
    
    subject_shift = detect_subject_shift(prompt, current_subject, subject_keywords)
    search_strategy = generate_search_strategy(query_analysis, subject_keywords, geo_info)
    
    # Logs internes
    print(f"🧠 [REFLECTION] Intent: {query_analysis.get('intent', 'unknown')} | Sentiment: {query_analysis.get('sentiment', 'neutral')} | Subjects: {query_analysis.get('subjects', [])}")
    
    return {
        "analysis": query_analysis,
        "subject_shift": subject_shift,
        "strategy": search_strategy,
        "execution_plan_ready": True
    }