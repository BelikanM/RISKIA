#!/usr/bin/env python3
"""
Script de test complet pour la génération 3D basée sur l'image cobaye.
Teste l'analyse CLIP, la génération 3D et la simulation de risques.
"""

import sys
import os
import json
from datetime import datetime

# Ajouter le répertoire courant au path pour les imports locaux
sys.path.append(os.path.dirname(__file__))

from site_3d_generator import (
    analyze_image_with_clip,
    generate_3d_description_from_clip,
    generate_realistic_site_3d_from_image,
    COBAYE_IMAGE_PATH
)

def simulate_risk_analysis(site_3d_description):
    """
    Simule une analyse de risques basée sur la description 3D.

    Args:
        site_3d_description (str): Description 3D du site.

    Returns:
        dict: Résultats de l'analyse de risques.
    """
    # Extraire les informations clés de la description
    risks = {
        "explosion_risk": 0.0,
        "fire_risk": 0.0,
        "leak_risk": 0.0,
        "environmental_risk": 0.0,
        "human_risk": 0.0
    }

    # Analyser le texte pour les risques
    description_lower = site_3d_description.lower()

    if "réservoir" in description_lower or "storage tank" in description_lower:
        risks["explosion_risk"] += 0.8
        risks["fire_risk"] += 0.6
        risks["leak_risk"] += 0.7
        risks["environmental_risk"] += 0.9

    if "plateforme" in description_lower or "drilling platform" in description_lower:
        risks["explosion_risk"] += 0.9
        risks["fire_risk"] += 0.8
        risks["human_risk"] += 0.7

    if "pipeline" in description_lower:
        risks["leak_risk"] += 0.8
        risks["environmental_risk"] += 0.8
        risks["fire_risk"] += 0.5

    # Normaliser les risques
    for key in risks:
        risks[key] = min(1.0, risks[key])

    # Calculer le risque global
    global_risk = sum(risks.values()) / len(risks)

    return {
        "risks": risks,
        "global_risk_level": global_risk,
        "risk_category": "CRITIQUE" if global_risk > 0.8 else "ÉLEVÉ" if global_risk > 0.6 else "MOYEN" if global_risk > 0.4 else "FAIBLE",
        "recommendations": generate_safety_recommendations(risks)
    }

def generate_safety_recommendations(risks):
    """
    Génère des recommandations de sécurité basées sur les risques identifiés.

    Args:
        risks (dict): Dictionnaire des risques.

    Returns:
        list: Liste des recommandations.
    """
    recommendations = []

    if risks["explosion_risk"] > 0.7:
        recommendations.extend([
            "Installer des systèmes de détection d'explosion avancés",
            "Mettre en place des procédures d'évacuation d'urgence",
            "Augmenter la fréquence des inspections de sécurité"
        ])

    if risks["fire_risk"] > 0.6:
        recommendations.extend([
            "Renforcer les systèmes d'extinction automatique",
            "Installer des caméras thermiques de surveillance",
            "Former le personnel aux procédures anti-incendie"
        ])

    if risks["leak_risk"] > 0.7:
        recommendations.extend([
            "Mettre en place un système de confinement secondaire",
            "Installer des capteurs de détection de fuite en continu",
            "Créer des procédures de réponse aux déversements"
        ])

    if risks["environmental_risk"] > 0.8:
        recommendations.extend([
            "Développer un plan de protection environnementale",
            "Installer des barrières de confinement",
            "Mettre en place un système de surveillance environnementale"
        ])

    if risks["human_risk"] > 0.6:
        recommendations.extend([
            "Augmenter les formations de sécurité du personnel",
            "Fournir un équipement de protection individuelle adapté",
            "Mettre en place des exercices d'urgence réguliers"
        ])

    return recommendations[:5]  # Limiter à 5 recommandations principales

def run_complete_3d_simulation():
    """
    Exécute la simulation 3D complète : analyse CLIP -> génération 3D -> analyse de risques.
    """
    print("=" * 80)
    print("TEST COMPLET DE SIMULATION 3D BASÉE SUR IMAGE")
    print("=" * 80)
    print(f"Date et heure: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Image analysée: {os.path.basename(COBAYE_IMAGE_PATH)}")
    print()

    # Étape 1: Analyse CLIP
    print("ÉTAPE 1: ANALYSE CLIP DE L'IMAGE")
    print("-" * 40)
    clip_results = analyze_image_with_clip(COBAYE_IMAGE_PATH)

    if 'error' in clip_results:
        print(f"❌ ERREUR: {clip_results['error']}")
        return

    print("✅ Analyse CLIP réussie")
    print(f"Élément principal: {clip_results['primary_element']}")
    print(".3f")
    print()
    print("Probabilités détaillées:")
    for element, prob in clip_results.items():
        if element not in ['primary_element', 'confidence']:
            print(".3f")
    print()

    # Étape 2: Génération de la description 3D
    print("ÉTAPE 2: GÉNÉRATION DE LA DESCRIPTION 3D")
    print("-" * 40)
    site_3d_description = generate_3d_description_from_clip(clip_results)
    print("✅ Description 3D générée")
    print("Contenu de la description:")
    print(site_3d_description)
    print()

    # Étape 3: Simulation des risques
    print("ÉTAPE 3: SIMULATION DES RISQUES")
    print("-" * 40)
    risk_analysis = simulate_risk_analysis(site_3d_description)
    print("✅ Analyse de risques effectuée")
    print(f"Niveau de risque global: {risk_analysis['global_risk_level']:.3f}")
    print(f"Catégorie de risque: {risk_analysis['risk_category']}")
    print()
    print("Détail des risques:")
    for risk_type, level in risk_analysis['risks'].items():
        risk_name = risk_type.replace('_', ' ').title()
        print(".3f")
    print()
    print("Recommandations de sécurité:")
    for i, rec in enumerate(risk_analysis['recommendations'], 1):
        print(f"{i}. {rec}")
    print()

    # Étape 4: Validation finale
    print("ÉTAPE 4: VALIDATION FINALE")
    print("-" * 40)

    # Tests de cohérence
    tests_passed = 0
    total_tests = 4

    # Test 1: L'analyse CLIP a identifié un élément
    if clip_results['primary_element']:
        print("✅ Test 1: Élément principal identifié")
        tests_passed += 1
    else:
        print("❌ Test 1: Aucun élément principal identifié")

    # Test 2: La description 3D contient des informations pertinentes
    if len(site_3d_description) > 500 and "ZONE" in site_3d_description:
        print("✅ Test 2: Description 3D complète générée")
        tests_passed += 1
    else:
        print("❌ Test 2: Description 3D incomplète")

    # Test 3: L'analyse de risques a calculé tous les risques
    if all(isinstance(level, (int, float)) and 0 <= level <= 1
           for level in risk_analysis['risks'].values()):
        print("✅ Test 3: Analyse de risques complète")
        tests_passed += 1
    else:
        print("❌ Test 3: Analyse de risques incomplète")

    # Test 4: Des recommandations ont été générées
    if len(risk_analysis['recommendations']) > 0:
        print("✅ Test 4: Recommandations générées")
        tests_passed += 1
    else:
        print("❌ Test 4: Aucune recommandation générée")

    print()
    print(f"RÉSULTAT FINAL: {tests_passed}/{total_tests} tests réussis")

    if tests_passed == total_tests:
        print("🎉 SUCCÈS: Simulation 3D complète et fonctionnelle!")
        print("Prêt pour l'intégration dans l'application RiskIA.")
    else:
        print("⚠️  ATTENTION: Certains tests ont échoué.")
        print("Vérifiez les composants avant l'intégration.")

    print("=" * 80)

    # Sauvegarder les résultats
    results = {
        "timestamp": datetime.now().isoformat(),
        "image_analyzed": os.path.basename(COBAYE_IMAGE_PATH),
        "clip_analysis": clip_results,
        "site_3d_description": site_3d_description,
        "risk_analysis": risk_analysis,
        "tests_passed": tests_passed,
        "total_tests": total_tests,
        "success": tests_passed == total_tests
    }

    # Sauvegarder en JSON
    output_file = "simulation_3d_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"📄 Résultats sauvegardés dans: {output_file}")

if __name__ == "__main__":
    run_complete_3d_simulation()