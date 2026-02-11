#!/usr/bin/env python3
"""
Test script pour vérifier la génération automatique du PDF
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from risk_simulation_app import RiskSimulationApp
import numpy as np

def test_pdf_generation():
    """Test de la génération automatique du PDF"""
    print("🧪 Test de génération automatique du PDF...")

    # Créer une instance de l'application
    app = RiskSimulationApp()

    # Simuler des résultats CLIP
    clip_results = [
        {
            'rank': 1,
            'texture': 'Corrosion avancée avec dépôts calcaires',
            'score': 0.892,
            'description': 'Corrosion sévère avec accumulation de dépôts minéraux'
        },
        {
            'rank': 2,
            'texture': 'Érosion hydrique progressive',
            'score': 0.756,
            'description': 'Usure progressive due à l\'eau avec formation de sillons'
        },
        {
            'rank': 3,
            'texture': 'Fissures structurales profondes',
            'score': 0.643,
            'description': 'Fissures importantes affectant la structure porteuse'
        }
    ]

    # Simuler des résultats ŒIL DE DIEU
    app.god_eye_results = {
        'micro_cracks': {'confidence': 0.85, 'detected': True},
        'soil_defects': {'confidence': 0.72, 'detected': True},
        'hidden_objects': {'confidence': 0.34, 'detected': False}
    }

    # Simuler des résultats solaires
    app.solar_results = {
        'azimuth': 135.5,
        'elevation': 45.2,
        'estimated_time': '14:30'
    }

    app.weather_results = {
        'cloud_cover': 'partiellement nuageux',
        'precipitation_risk': 'faible'
    }

    app.climate_results = {
        'season': 'été'
    }

    app.impact_results = {
        'recommended_actions': ['Inspection immédiate', 'Réparation urgente', 'Monitoring continu']
    }

    # Tester la génération PDF
    try:
        app._generate_automatic_pdf_report(clip_results)
        print("✅ Test de génération PDF réussi!")
        return True
    except Exception as e:
        print(f"❌ Erreur lors du test PDF: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pdf_generation()
    sys.exit(0 if success else 1)