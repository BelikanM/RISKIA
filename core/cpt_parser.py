#!/usr/bin/env python3
"""
Module de parsing pour les fichiers CPT/CPTU
Supporte les formats .txt, .xlsx, .csv, .xls, .cal
"""

import pandas as pd
import numpy as np
import os
import charset_normalizer
from typing import Optional, Tuple

class CPTParser:
    """Parser spécialisé pour les fichiers CPT/CPTU"""

    def __init__(self):
        self.supported_formats = ['.txt', '.xlsx', '.csv', '.xls', '.cal']
        self.cpt_columns = ['depth', 'qc', 'fs', 'u', 'u2', 'Rf', 'gamma', 'Vs', 'qt', 'Bq']

    def parse_file(self, file_path: str) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Parse un fichier CPT et retourne les données et un message de statut

        Args:
            file_path: Chemin vers le fichier à parser

        Returns:
            Tuple (DataFrame or None, message)
        """
        if not os.path.exists(file_path):
            return None, f"Fichier non trouvé: {file_path}"

        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext not in self.supported_formats:
            return None, f"Format non supporté: {file_ext}"

        try:
            if file_ext in ['.xlsx', '.xls']:
                return self._parse_excel(file_path)
            elif file_ext == '.csv':
                return self._parse_csv(file_path)
            elif file_ext == '.cal':
                return self._parse_cal(file_path)
            else:  # .txt
                return self._parse_text(file_path)
        except Exception as e:
            return None, f"Erreur lors du parsing: {str(e)}"

    def _parse_excel(self, file_path: str) -> Tuple[Optional[pd.DataFrame], str]:
        """Parse un fichier Excel"""
        try:
            df = pd.read_excel(file_path)
            df = self._clean_and_validate(df)
            return df, "Fichier Excel parsé avec succès"
        except Exception as e:
            return None, f"Erreur Excel: {str(e)}"

    def _parse_csv(self, file_path: str) -> Tuple[Optional[pd.DataFrame], str]:
        """Parse un fichier CSV"""
        encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']

        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(file_path, encoding=encoding, sep=None, engine='python')
                df = self._clean_and_validate(df)
                return df, f"Fichier CSV parsé avec succès (encoding: {encoding})"
            except Exception:
                continue

        return None, "Impossible de parser le fichier CSV avec les encodings testés"

    def _parse_text(self, file_path: str) -> Tuple[Optional[pd.DataFrame], str]:
        """Parse un fichier texte avec détection d'encoding et de séparateur"""
        try:
            # Détection automatique de l'encoding
            with open(file_path, 'rb') as f:
                detected = charset_normalizer.detect(f.read())
                encoding = detected.get('encoding', 'utf-8')

            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()

            # Nettoyer les lignes vides au début et à la fin
            lines = [line.strip() for line in lines if line.strip()]

            if not lines:
                return None, "Fichier vide"

            # Détecter si la première ligne est un en-tête
            header_line = lines[0]

            # Essayer différents séparateurs (priorité aux tabulations)
            separators = ['\t', ';', ',', ' ', '|']
            best_df = None
            best_score = 0
            best_sep = None

            for sep in separators:
                try:
                    # Créer un DataFrame de test avec les premières lignes
                    test_content = '\n'.join(lines[:min(10, len(lines))])
                    df_test = pd.read_csv(pd.io.common.StringIO(test_content),
                                        sep=sep, engine='python', header=0)

                    # Vérifier que le DataFrame n'est pas vide
                    if df_test.empty or len(df_test.columns) < 2:
                        continue

                    # Calculer un score basé sur plusieurs critères
                    score = 0

                    # Nombre de colonnes numériques
                    numeric_cols = 0
                    for col in df_test.columns:
                        try:
                            numeric_series = pd.to_numeric(df_test[col], errors='coerce')
                            if not numeric_series.isna().all():
                                numeric_cols += 1
                        except:
                            pass

                    # Ratio de colonnes numériques
                    numeric_ratio = numeric_cols / len(df_test.columns) if df_test.columns.size > 0 else 0

                    # Bonus pour les séparateurs qui donnent plus de colonnes numériques
                    score += numeric_cols * 2

                    # Bonus si la plupart des colonnes sont numériques (sauf la première qui peut être depth)
                    if numeric_ratio >= 0.6:
                        score += 10

                    # Bonus pour les tabulations (format CPT standard)
                    if sep == '\t':
                        score += 5

                    # Pénalité pour les espaces (peuvent créer trop de colonnes)
                    if sep == ' ' and len(df_test.columns) > 10:
                        score -= 10

                    # Vérifier que les données semblent cohérentes (pas que des NaN)
                    if not df_test.dropna(how='all').empty:
                        score += 5

                    if score > best_score:
                        best_score = score
                        best_df = df_test
                        best_sep = sep

                except Exception:
                    continue

            if best_df is None or best_sep is None:
                return None, "Impossible de déterminer le séparateur approprié"

            # Parser le fichier complet avec le meilleur séparateur
            content = '\n'.join(lines)
            df = pd.read_csv(pd.io.common.StringIO(content), sep=best_sep, engine='python', header=None)

            # Si pas d'en-tête détecté (colonnes nommées par défaut), ajouter des noms
            if df.columns.tolist() == list(range(len(df.columns))):
                # Essayer de déterminer si la première ligne contient des en-têtes ou des données
                first_row_is_header = False
                if len(lines) > 1:
                    try:
                        # Vérifier si la première ligne contient des nombres (données) ou du texte (en-têtes)
                        first_row_values = lines[0].strip().split(best_sep)
                        numeric_count = 0
                        for val in first_row_values[:3]:  # Vérifier les 3 premières valeurs
                            try:
                                float(val.replace(',', '.'))
                                numeric_count += 1
                            except:
                                pass

                        # Si la première ligne contient majoritairement des nombres, c'est des données
                        if numeric_count >= len(first_row_values) * 0.5:
                            # Ajouter des noms de colonnes par défaut
                            default_columns = ['depth', 'qc', 'fs', 'u', 'u2', 'Rf', 'gamma', 'Vs', 'qt', 'Bq']
                            df.columns = default_columns[:len(df.columns)]
                        else:
                            # La première ligne est un en-tête, relire avec header=0
                            df = pd.read_csv(pd.io.common.StringIO(content), sep=best_sep, engine='python', header=0)
                    except:
                        # En cas d'erreur, utiliser des noms par défaut
                        default_columns = ['depth', 'qc', 'fs', 'u', 'u2', 'Rf', 'gamma', 'Vs', 'qt', 'Bq']
                        df.columns = default_columns[:len(df.columns)]
                else:
                    # Fichier avec une seule ligne, noms par défaut
                    default_columns = ['depth', 'qc', 'fs', 'u', 'u2', 'Rf', 'gamma', 'Vs', 'qt', 'Bq']
                    df.columns = default_columns[:len(df.columns)]

            df = self._clean_and_validate(df)
            return df, f"Fichier texte parsé avec succès (encoding: {encoding}, séparateur: '{best_sep}')"

        except Exception as e:
            return None, f"Erreur texte: {str(e)}"

    def _parse_cal(self, file_path: str) -> Tuple[Optional[pd.DataFrame], str]:
        """Parse un fichier .cal (format binaire ou texte)"""
        try:
            # Pour l'instant, traiter comme texte
            # TODO: Implémenter parsing binaire si nécessaire
            return self._parse_text(file_path)
        except Exception as e:
            return None, f"Erreur CAL: {str(e)}"

    def _clean_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie et valide les données CPT"""
        # Supprimer les colonnes complètement vides
        df = df.dropna(axis=1, how='all')

        # Convertir les colonnes numériques et nettoyer les noms
        for col in df.columns:
            # Nettoyer le nom de la colonne (supprimer espaces multiples, normaliser)
            clean_col_name = ' '.join(str(col).strip().split())
            df = df.rename(columns={col: clean_col_name})

            # Essayer de convertir en numérique
            try:
                # Remplacer les virgules par des points pour les nombres décimaux
                if df[clean_col_name].dtype == 'object':
                    df[clean_col_name] = df[clean_col_name].astype(str).str.replace(',', '.', regex=False)
                # Convertir explicitement en float64 pour éviter les entiers
                df[clean_col_name] = pd.to_numeric(df[clean_col_name], errors='coerce').astype('float64')
            except:
                pass

        # Mapper les colonnes aux noms standard
        df = self._map_columns(df)

        # Supprimer les colonnes qui ne sont pas des paramètres CPT standard
        cpt_columns = ['depth', 'qc', 'fs', 'u', 'u2', 'Rf', 'gamma', 'Vs', 'qt', 'Bq']
        columns_to_keep = []
        for col in df.columns:
            col_lower = str(col).lower()
            if col_lower in cpt_columns or any(cpt_col in col_lower for cpt_col in cpt_columns):
                columns_to_keep.append(col)

        if columns_to_keep:
            df = df[columns_to_keep]

        # Supprimer les lignes avec toutes les valeurs NaN
        df = df.dropna(how='all')

        # Trier par profondeur si disponible
        depth_cols = [col for col in df.columns if 'depth' in str(col).lower()]
        if depth_cols:
            df = df.sort_values(depth_cols[0]).reset_index(drop=True)

        return df

    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mappe les colonnes aux noms standard CPT"""
        
        column_mappings = {
            # Profondeur
            'profondeur': 'depth', 'depth': 'depth', 'z': 'depth', 'depth (m)': 'depth',
            'depth_m': 'depth', 'depth[m]': 'depth', 'prof': 'depth', 'prof.': 'depth',

            # Résistance de pointe
            'qc': 'qc', 'cone resistance': 'qc', 'qc (mpa)': 'qc', 'qc_mpa': 'qc',
            'qc[mpa]': 'qc', 'résistance': 'qc', 'résistance qc': 'qc', 'pression': 'qc',
            'qc (kn/m²)': 'qc', 'qc_kn/m2': 'qc', 'qc[kn/m2]': 'qc',
            'pression a la pointe': 'qc', 'pression a la pointe (mpa)': 'qc',
            'résistance de pointe': 'qc', 'cone tip resistance': 'qc',

            # Frottement latéral
            'fs': 'fs', 'friction': 'fs', 'sleeve friction': 'fs', 'fs (kpa)': 'fs',
            'fs_kpa': 'fs', 'fs[kpa]': 'fs', 'frottement': 'fs', 'frottement fs': 'fs',
            'fs (kn/m²)': 'fs', 'fs_kn/m2': 'fs', 'fs[kn/m2]': 'fs',
            'frottement latéral': 'fs', 'résistance latérale': 'fs',

            # Pression interstitielle
            'u': 'u', 'pore pressure': 'u', 'u (kpa)': 'u', 'u_kpa': 'u', 'u[kpa]': 'u',
            'pore_pressure': 'u', 'pression pore': 'u', 'pression interstitielle': 'u',
            'u (kn/m²)': 'u', 'u_kn/m2': 'u', 'u[kn/m2]': 'u',

            # Pression interstitielle u2
            'u2': 'u2', 'u2 (kpa)': 'u2', 'u2_kpa': 'u2', 'u2[kpa]': 'u2',
            'u2 (kn/m²)': 'u2', 'u2_kn/m2': 'u2', 'u2[kn/m2]': 'u2',

            # Rapport de frottement
            'rf': 'Rf', 'friction ratio': 'Rf', 'rf (%)': 'Rf', 'rf_percent': 'Rf',
            'rf[%]': 'Rf', 'rapport frottement': 'Rf', 'friction_ratio': 'Rf',

            # Poids volumique
            'gamma': 'gamma', 'unit weight': 'gamma', 'gamma (kn/m³)': 'gamma',
            'gamma_kn/m3': 'gamma', 'gamma[kn/m3]': 'gamma', 'poids volumique': 'gamma',
            'densité': 'gamma', 'unit_weight': 'gamma',

            # Vitesse des ondes de cisaillement
            'vs': 'Vs', 'shear wave': 'Vs', 'vs (m/s)': 'Vs', 'vs_ms': 'Vs', 'vs[m/s]': 'Vs',
            'vitesse cisaillement': 'Vs', 'shear_wave_velocity': 'Vs',

            # Résistance corrigée
            'qt': 'qt', 'qt (mpa)': 'qt', 'qt_mpa': 'qt', 'qt[mpa]': 'qt',
            'résistance corrigée': 'qt',

            # Paramètre de Robertson
            'bq': 'Bq', 'bq': 'Bq', 'paramètre bq': 'Bq', 'robertson bq': 'Bq'
        }

        new_columns = {}
        for col in df.columns:
            col_lower = str(col).lower().strip()
            # Supprimer les espaces multiples et normaliser
            col_lower = ' '.join(col_lower.split())

            if col_lower in column_mappings:
                new_columns[col] = column_mappings[col_lower]
            else:
                new_columns[col] = col

        df = df.rename(columns=new_columns)
        return df