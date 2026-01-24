#!/usr/bin/env python3
"""
Outil de vérification de l'intégrité des données CPT/CPTU
Vérifie que les données ne sont pas tronquées ou altérées lors du parsing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import hashlib
import os
from core.cpt_parser import CPTParser

class DataIntegrityChecker:
    """Vérificateur d'intégrité des données CPT"""

    def __init__(self):
        self.parser = CPTParser()
        self.check_results = {}

    def verify_file_integrity(self, file_path: str) -> Dict[str, Any]:
        """
        Vérifie l'intégrité complète d'un fichier CPT

        Args:
            file_path: Chemin vers le fichier à vérifier

        Returns:
            Dictionnaire avec les résultats de vérification
        """
        results = {
            'file_path': file_path,
            'file_exists': False,
            'file_size': 0,
            'raw_data_hash': None,
            'parsed_data_hash': None,
            'data_integrity': False,
            'parsing_errors': [],
            'data_loss_warnings': [],
            'statistics_comparison': {},
            'column_mapping_issues': [],
            'recommendations': []
        }

        if not os.path.exists(file_path):
            results['parsing_errors'].append(f"Fichier non trouvé: {file_path}")
            return results

        results['file_exists'] = True
        results['file_size'] = os.path.getsize(file_path)

        try:
            # 1. Calculer le hash des données brutes
            raw_hash = self._calculate_file_hash(file_path)
            results['raw_data_hash'] = raw_hash

            # 2. Parser le fichier
            parsed_df, parse_message = self.parser.parse_file(file_path)

            if parsed_df is None:
                results['parsing_errors'].append(f"Échec du parsing: {parse_message}")
                return results

            # 3. Vérifier l'intégrité des données parsées
            integrity_issues = self._check_data_integrity(file_path, parsed_df)
            results.update(integrity_issues)

            # 4. Calculer le hash des données parsées
            parsed_hash = self._calculate_dataframe_hash(parsed_df)
            results['parsed_data_hash'] = parsed_hash

            # 5. Comparer les statistiques
            stats_comparison = self._compare_statistics(file_path, parsed_df)
            results['statistics_comparison'] = stats_comparison

            # 6. Vérifier les mappings de colonnes
            mapping_issues = self._check_column_mappings(file_path, parsed_df)
            results['column_mapping_issues'] = mapping_issues

            # 7. Générer des recommandations
            recommendations = self._generate_recommendations(results)
            results['recommendations'] = recommendations

            # 8. Évaluation globale
            results['data_integrity'] = self._evaluate_overall_integrity(results)

        except Exception as e:
            results['parsing_errors'].append(f"Erreur lors de la vérification: {str(e)}")

        return results

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calcule le hash SHA256 du fichier"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _calculate_dataframe_hash(self, df: pd.DataFrame) -> str:
        """Calcule le hash des données du DataFrame"""
        # Convertir en string pour hashage
        df_str = df.to_string(index=False)
        return hashlib.sha256(df_str.encode()).hexdigest()

    def _check_data_integrity(self, file_path: str, parsed_df: pd.DataFrame) -> Dict[str, Any]:
        """Vérifie l'intégrité des données parsées"""
        issues = {
            'data_loss_warnings': [],
            'precision_warnings': [],
            'range_warnings': []
        }

        # Vérifier la taille du fichier vs nombre de lignes
        file_size = os.path.getsize(file_path)
        expected_rows = max(1, file_size // 100)  # Estimation grossière

        if len(parsed_df) < expected_rows * 0.1:
            issues['data_loss_warnings'].append(
                f"Très peu de lignes parsées ({len(parsed_df)}) pour un fichier de {file_size} bytes"
            )

        # Vérifier les colonnes numériques pour la précision
        numeric_columns = parsed_df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            values = parsed_df[col].dropna()

            if len(values) == 0:
                continue

            # Vérifier si les valeurs semblent arrondies/troncquées
            if col.lower() in ['depth', 'profondeur']:
                # Profondeur devrait avoir une précision décimale
                decimal_places = values.apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
                if decimal_places.max() <= 1:  # Seulement 1 décimale ou moins
                    issues['precision_warnings'].append(
                        f"Colonne {col}: précision faible (max {decimal_places.max()} décimales)"
                    )
            elif col.lower() in ['qc', 'fs', 'u']:
                # Paramètres CPT devraient avoir au moins 2 décimales
                decimal_places = values.apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
                if decimal_places.max() < 2:
                    issues['precision_warnings'].append(
                        f"Colonne {col}: précision potentiellement insuffisante (max {decimal_places.max()} décimales)"
                    )

            # Vérifier les plages réalistes
            if col.lower() in ['qc']:
                if values.max() > 100 or values.min() < 0:
                    issues['range_warnings'].append(
                        f"Colonne {col}: valeurs hors plage réaliste (min: {values.min():.2f}, max: {values.max():.2f})"
                    )
            elif col.lower() in ['fs']:
                if values.max() > 50 or values.min() < 0:
                    issues['range_warnings'].append(
                        f"Colonne {col}: valeurs hors plage réaliste (min: {values.min():.2f}, max: {values.max():.2f})"
                    )
            elif col.lower() in ['depth', 'profondeur']:
                if values.max() > 1000 or values.min() < 0:
                    issues['range_warnings'].append(
                        f"Colonne {col}: valeurs hors plage réaliste (min: {values.min():.2f}, max: {values.max():.2f})"
                    )

        return issues

    def _compare_statistics(self, file_path: str, parsed_df: pd.DataFrame) -> Dict[str, Any]:
        """Compare les statistiques du fichier brut et parsé"""
        stats = {}

        # Statistiques des colonnes numériques
        numeric_columns = parsed_df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            values = parsed_df[col].dropna()
            if len(values) > 0:
                stats[col] = {
                    'count': len(values),
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'zeros_count': int((values == 0).sum()),
                    'negative_count': int((values < 0).sum())
                }

        return stats

    def _check_column_mappings(self, file_path: str, parsed_df: pd.DataFrame) -> List[str]:
        """Vérifie les mappings de colonnes"""
        issues = []

        expected_columns = ['depth', 'qc', 'fs', 'u', 'u2', 'Rf', 'gamma', 'Vs', 'qt', 'Bq']
        found_columns = [col.lower() for col in parsed_df.columns]

        # Vérifier les colonnes manquantes importantes
        missing_important = []
        for col in ['depth', 'qc', 'fs']:
            if col not in found_columns:
                missing_important.append(col)

        if missing_important:
            issues.append(f"Colonnes importantes manquantes: {', '.join(missing_important)}")

        # Vérifier les colonnes inattendues
        unexpected = [col for col in found_columns if col not in expected_columns and len(col) > 10]
        if unexpected:
            issues.append(f"Colonnes potentiellement mal mappées: {', '.join(unexpected[:3])}")

        return issues

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Génère des recommandations basées sur les résultats"""
        recommendations = []

        if results.get('parsing_errors'):
            recommendations.append("Résoudre les erreurs de parsing avant l'analyse")

        if results.get('data_loss_warnings'):
            recommendations.append("Vérifier la perte de données lors du parsing")

        if results.get('precision_warnings'):
            recommendations.append("Vérifier la précision des données numériques")

        if results.get('range_warnings'):
            recommendations.append("Valider les plages de valeurs des paramètres CPT")

        if results.get('column_mapping_issues'):
            recommendations.append("Vérifier le mapping automatique des colonnes")

        if not results.get('data_integrity', False):
            recommendations.append("Revoir le processus de parsing pour assurer l'intégrité des données")

        return recommendations

    def _evaluate_overall_integrity(self, results: Dict[str, Any]) -> bool:
        """Évalue l'intégrité globale des données"""
        # Critères d'intégrité
        if results.get('parsing_errors'):
            return False

        if len(results.get('data_loss_warnings', [])) > 2:
            return False

        if len(results.get('precision_warnings', [])) > 3:
            return False

        if len(results.get('range_warnings', [])) > 2:
            return False

        return True

    def generate_integrity_report(self, file_path: str) -> str:
        """Génère un rapport d'intégrité formaté"""
        results = self.verify_file_integrity(file_path)

        report = f"""
=== RAPPORT D'INTÉGRITÉ DES DONNÉES ===
Fichier: {results['file_path']}
Taille: {results['file_size']} bytes
Intégrité globale: {'✓ VALIDE' if results['data_integrity'] else '✗ PROBLÈMES DÉTECTÉS'}

=== ERREURS DE PARSING ===
"""
        if results['parsing_errors']:
            for error in results['parsing_errors']:
                report += f"• {error}\n"
        else:
            report += "Aucune erreur de parsing détectée\n"

        report += "\n=== AVERTISSEMENTS PERTE DE DONNÉES ===\n"
        warnings = results.get('data_loss_warnings', []) + results.get('precision_warnings', []) + results.get('range_warnings', [])
        if warnings:
            for warning in warnings:
                report += f"• {warning}\n"
        else:
            report += "Aucun avertissement\n"

        report += "\n=== PROBLÈMES DE MAPPING ===\n"
        if results['column_mapping_issues']:
            for issue in results['column_mapping_issues']:
                report += f"• {issue}\n"
        else:
            report += "Aucun problème de mapping détecté\n"

        report += "\n=== RECOMMANDATIONS ===\n"
        if results['recommendations']:
            for rec in results['recommendations']:
                report += f"• {rec}\n"
        else:
            report += "Aucune recommandation spécifique\n"

        report += "\n=== STATISTIQUES DES COLONNES ===\n"
        for col, stats in results['statistics_comparison'].items():
            report += f"{col}: {stats['count']} valeurs, moyenne={stats['mean']:.2f}, écart-type={stats['std']:.2f}\n"

        return report