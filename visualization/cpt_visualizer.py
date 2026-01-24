#!/usr/bin/env python3
"""
Module de visualisation pour les données CPT/CPTU
Fournit des classes et fonctions pour créer des visualisations avancées
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional, Any
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

class CPTVisualizer:
    """Classe principale pour la visualisation des données CPT/CPTU"""

    def __init__(self):
        self.default_figsize = (12, 8)
        self.default_dpi = 100

    def create_qc_fs_plot(self, df: pd.DataFrame) -> plt.Figure:
        """Crée un graphique qc vs fs avec classification des sols"""
        fig, ax = plt.subplots(figsize=self.default_figsize)

        # Couleur par type de sol si disponible
        if 'Soil_Type' in df.columns:
            soil_types = df['Soil_Type'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(soil_types)))

            for soil_type, color in zip(soil_types, colors):
                mask = df['Soil_Type'] == soil_type
                ax.scatter(df[mask]['qc'], df[mask]['fs'],
                          c=[color], label=soil_type, alpha=0.8, s=60,
                          edgecolors='black', linewidth=0.5)
        else:
            ax.scatter(df['qc'], df['fs'], alpha=0.8, s=60,
                      edgecolors='black', linewidth=0.5)

        ax.set_xlabel('Résistance de pointe qc (MPa)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Résistance de frottement fs (MPa)', fontsize=12, fontweight='bold')
        ax.set_title('Graphique qc vs fs - Classification des sols',
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        # Lignes de référence Robertson
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Ligne de référence')
        ax.axvline(x=5, color='blue', linestyle='--', alpha=0.7)

        return fig

    def create_depth_profile(self, df: pd.DataFrame, parameter: str) -> plt.Figure:
        """Crée un profil de profondeur pour un paramètre donné"""
        fig, ax = plt.subplots(figsize=self.default_figsize)

        if 'depth' not in df.columns or parameter not in df.columns:
            ax.text(0.5, 0.5, f"Colonnes 'depth' ou '{parameter}' manquantes",
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        ax.plot(df[parameter], df['depth'], 'b-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel(f'{parameter} (unité)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Profondeur (m)', fontsize=12, fontweight='bold')
        ax.set_title(f'Profil de {parameter} en fonction de la profondeur',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Profondeur croissante vers le bas

        return fig

    def create_3d_visualization(self, df: pd.DataFrame) -> plt.Figure:
        """Crée une visualisation 3D des paramètres CPT"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        if 'depth' not in df.columns or 'qc' not in df.columns:
            ax.text(0.5, 0.5, 0.5, "Données insuffisantes pour 3D",
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        # Utiliser qc, fs et depth pour 3D
        x = df.get('qc', np.zeros(len(df)))
        y = df.get('fs', np.zeros(len(df)))
        z = df['depth']

        scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=50, alpha=0.8)

        ax.set_xlabel('qc (MPa)')
        ax.set_ylabel('fs (MPa)')
        ax.set_zlabel('Profondeur (m)')
        ax.set_title('Visualisation 3D des paramètres CPT', fontsize=14, fontweight='bold')

        # Inverser l'axe Z pour profondeur
        ax.invert_zaxis()

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Profondeur (m)')

        return fig

    def create_combined_dashboard(self, df: pd.DataFrame) -> go.Figure:
        """Crée un dashboard combiné avec Plotly"""
        if len(df.columns) < 3:
            fig = go.Figure()
            fig.add_annotation(text="Données insuffisantes pour le dashboard",
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('qc vs Profondeur', 'fs vs Profondeur',
                          'u vs Profondeur', 'qc vs fs', 'Distributions', '3D View'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'histogram'}, {'type': 'scatter3d'}]]
        )

        # Profils de profondeur
        if 'depth' in df.columns:
            if 'qc' in df.columns:
                fig.add_trace(go.Scatter(x=df['qc'], y=df['depth'], mode='lines+markers',
                                       name='qc'), row=1, col=1)
            if 'fs' in df.columns:
                fig.add_trace(go.Scatter(x=df['fs'], y=df['depth'], mode='lines+markers',
                                       name='fs'), row=1, col=2)
            if 'u' in df.columns:
                fig.add_trace(go.Scatter(x=df['u'], y=df['depth'], mode='lines+markers',
                                       name='u'), row=1, col=3)

        # qc vs fs
        if 'qc' in df.columns and 'fs' in df.columns:
            fig.add_trace(go.Scatter(x=df['qc'], y=df['fs'], mode='markers',
                                   name='qc vs fs'), row=2, col=1)

        # Distributions
        if 'qc' in df.columns:
            fig.add_trace(go.Histogram(x=df['qc'], name='qc'), row=2, col=2)

        # 3D
        if 'depth' in df.columns and 'qc' in df.columns and 'fs' in df.columns:
            fig.add_trace(go.Scatter3d(x=df['qc'], y=df['fs'], z=df['depth'],
                                     mode='markers', name='3D'), row=2, col=3)

        fig.update_layout(height=800, title_text="Dashboard d'Analyse Combinée CPTU",
                         showlegend=True)

        return fig

    def create_soil_classification_plot(self, df: pd.DataFrame) -> plt.Figure:
        """Crée un graphique de classification des sols selon Robertson"""
        fig, ax = plt.subplots(figsize=self.default_figsize)

        if 'qc' not in df.columns or 'fs' not in df.columns:
            ax.text(0.5, 0.5, "qc et fs requis pour la classification",
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        # Calculer l'indice Ic de Robertson
        qc = df['qc'].values
        fs = df['fs'].values

        # Éviter division par zéro
        qc_safe = np.where(qc > 0, qc, 1e-6)
        fr = (fs / qc_safe) * 100  # Friction ratio

        # Indice de comportement des sols
        Ic = np.sqrt((3.47 - np.log10(qc_safe))**2 + (np.log10(fr + 1e-6) + 1.22)**2)

        # Scatter plot avec couleur par Ic
        scatter = ax.scatter(qc, fs, c=Ic, cmap='RdYlBu_r', alpha=0.8, s=60,
                           edgecolors='black', linewidth=0.5)

        ax.set_xlabel('qc (MPa)', fontsize=12, fontweight='bold')
        ax.set_ylabel('fs (MPa)', fontsize=12, fontweight='bold')
        ax.set_title('Classification des sols - Indice Ic de Robertson',
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Indice Ic')

        # Zones de classification (simplifiées)
        zones = [
            {'x': [0, 5], 'y': [0, 0.5], 'label': 'Sensible'},
            {'x': [5, 20], 'y': [0.5, 2], 'label': 'Argile'},
            {'x': [5, 50], 'y': [0, 0.5], 'label': 'Sable'},
        ]

        for zone in zones:
            rect = patches.Rectangle((zone['x'][0], zone['y'][0]),
                                   zone['x'][1]-zone['x'][0], zone['y'][1]-zone['y'][0],
                                   linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            ax.text(zone['x'][0] + (zone['x'][1]-zone['x'][0])/2,
                   zone['y'][0] + (zone['y'][1]-zone['y'][0])/2,
                   zone['label'], ha='center', va='center', fontsize=10, color='red')

        return fig