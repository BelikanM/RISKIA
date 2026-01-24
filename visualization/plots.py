#!/usr/bin/env python3
"""
Fonctions de visualisation avancées pour l'analyse CPT/CPTU
Plus de 10 types de visualisations 3D et combinées
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import matplotlib.patches as patches

def create_qc_fs_plot(df):
    """Crée un graphique qc vs fs avec améliorations"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Couleur par type de sol
    if 'Soil_Type' in df.columns:
        soil_types = df['Soil_Type'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(soil_types)))

        for soil_type, color in zip(soil_types, colors):
            mask = df['Soil_Type'] == soil_type
            ax.scatter(df[mask]['qc'], df[mask]['fs'],
                      c=[color], label=soil_type, alpha=0.8, s=60, edgecolors='black', linewidth=0.5)
    else:
        ax.scatter(df['qc'], df['fs'], alpha=0.8, s=60, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('Résistance de pointe qc (MPa)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Résistance de frottement fs (MPa)', fontsize=12, fontweight='bold')
    ax.set_title('Graphique qc vs fs - Classification des sols', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Ajouter des lignes de référence pour classification Robertson
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Ligne de référence')
    ax.axvline(x=5, color='blue', linestyle='--', alpha=0.7)

    return fig

def create_combined_analysis_dashboard(df):
    """Crée un dashboard combiné avec multiples graphiques"""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('qc vs fs', 'Profils qc/fs', 'Distribution qc', 'Distribution fs', 'Soil Types', 'Clusters'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'histogram'}],
               [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'bar'}]]
    )

    # qc vs fs
    fig.add_trace(go.Scatter(x=df['qc'], y=df['fs'], mode='markers',
                            marker=dict(size=6, color=df.get('Cluster', 0), colorscale='viridis')),
                  row=1, col=1)

    # Profils de profondeur
    fig.add_trace(go.Scatter(x=df['qc'], y=df['Depth'], mode='lines', name='qc',
                            line=dict(color='blue', width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=df['fs'], y=df['Depth'], mode='lines', name='fs',
                            line=dict(color='red', width=2)), row=1, col=2)

    # Distribution qc
    fig.add_trace(go.Histogram(x=df['qc'], nbinsx=30, marker_color='blue'), row=1, col=3)

    # Distribution fs
    fig.add_trace(go.Histogram(x=df['fs'], nbinsx=30, marker_color='red'), row=2, col=1)

    # Types de sol
    if 'Soil_Type' in df.columns:
        soil_counts = df['Soil_Type'].value_counts()
        fig.add_trace(go.Bar(x=soil_counts.index, y=soil_counts.values,
                            marker_color='green'), row=2, col=2)

    # Clusters
    if 'Cluster' in df.columns:
        cluster_counts = df['Cluster'].value_counts().sort_index()
        fig.add_trace(go.Bar(x=cluster_counts.index.astype(str), y=cluster_counts.values,
                            marker_color='orange'), row=2, col=3)

    fig.update_layout(height=800, title_text="Dashboard d'Analyse Combinée CPTU")
    fig.update_yaxes(autorange="reversed", row=1, col=2)  # Inverser profondeur

    return fig

def create_geological_cross_section(df):
    """Crée une coupe géologique verticale"""
    fig, ax = plt.subplots(figsize=(16, 10))

    # Trier par profondeur
    df_sorted = df.sort_values('Depth')

    # Créer des couches géologiques basées sur les types de sol
    if 'Soil_Type' in df.columns:
        soil_types = df_sorted['Soil_Type'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(soil_types)))
        soil_color_map = dict(zip(soil_types, colors))

        # Tracer les couches
        prev_depth = 0
        for i, row in df_sorted.iterrows():
            depth = row['Depth']
            soil_type = row['Soil_Type']
            color = soil_color_map[soil_type]

            # Rectangle pour représenter la couche
            rect = Rectangle((0, prev_depth), 1, depth - prev_depth,
                           facecolor=color, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)

            # Ajouter des points de données
            ax.scatter([0.5], [depth], c=[color], s=30, edgecolors='black', zorder=3)

            prev_depth = depth

        # Légende
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, label=soil_type)
                          for soil_type, color in soil_color_map.items()]
        ax.legend(handles=legend_elements, loc='upper right')

    # Ajouter les valeurs qc et fs sur les côtés
    ax2 = ax.twinx()
    ax2.plot(df_sorted['qc'], df_sorted['Depth'], 'b-', linewidth=2, alpha=0.7, label='qc')
    ax2.plot(df_sorted['fs'], df_sorted['Depth'], 'r-', linewidth=2, alpha=0.7, label='fs')
    ax2.set_ylabel('Résistance (MPa)', color='black')
    ax2.legend(loc='upper left')

    ax.set_xlabel('Position horizontale')
    ax.set_ylabel('Profondeur (m)')
    ax.set_title('Coupe Géologique Verticale', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    return fig

def create_3d_surface_plot(df):
    """Visualisation 3D en surface des paramètres CPTU"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Créer une grille
    x = df.get('x', df.index)
    y = df.get('y', [0] * len(df))
    z = df['Depth']

    # Scatter plot avec couleur basée sur qc
    scatter = ax.scatter(x, y, z, c=df['qc'], cmap='viridis', s=50, alpha=0.8)

    # Ajouter une surface interpolée
    from scipy.interpolate import griddata

    # Créer une grille régulière
    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    XI, YI = np.meshgrid(xi, yi)

    # Interpoler qc sur la grille
    ZI = griddata((x, y), df['qc'], (XI, YI), method='linear')

    # Tracer la surface
    surf = ax.plot_surface(XI, YI, np.full_like(ZI, z.min()), facecolors=plt.cm.viridis(ZI/ZI.max()),
                          alpha=0.3, shade=True)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Profondeur (m)')
    ax.set_title('Surface 3D des valeurs qc')
    ax.invert_zaxis()

    # Colorbar
    plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label='qc (MPa)')

    return fig

def create_3d_wireframe_plot(df):
    """Visualisation 3D en fil de fer"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    x = df.get('x', df.index)
    y = df.get('y', [0] * len(df))
    z = df['Depth']

    # Créer wireframe
    from scipy.interpolate import griddata

    xi = np.linspace(x.min(), x.max(), 20)
    yi = np.linspace(y.min(), y.max(), 20)
    XI, YI = np.meshgrid(xi, yi)

    ZI = griddata((x, y), df['qc'], (XI, YI), method='cubic')

    # Wireframe
    wire = ax.plot_wireframe(XI, YI, ZI, color='blue', alpha=0.6, linewidth=0.5)

    # Points de données
    scatter = ax.scatter(x, y, z, c=df['qc'], cmap='plasma', s=30, alpha=0.8)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Profondeur (m)')
    ax.set_title('Wireframe 3D avec points de données')
    ax.invert_zaxis()

    plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label='qc (MPa)')

    return fig

def create_3d_contour_plot(df):
    """Visualisation 3D avec contours"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    x = df.get('x', df.index)
    y = df.get('y', [0] * len(df))

    # Créer contours à différentes profondeurs
    depths = np.linspace(df['Depth'].min(), df['Depth'].max(), 10)

    for i, depth in enumerate(depths):
        mask = (df['Depth'] >= depth - 0.5) & (df['Depth'] <= depth + 0.5)
        subset = df[mask]

        if len(subset) > 3:  # Assez de points pour interpolation
            from scipy.interpolate import griddata

            xi = np.linspace(x.min(), x.max(), 30)
            yi = np.linspace(y.min(), y.max(), 30)
            XI, YI = np.meshgrid(xi, yi)

            ZI = griddata((subset['x'] if 'x' in subset.columns else subset.index,
                          subset['y'] if 'y' in subset.columns else [0]*len(subset)),
                         subset['qc'], (XI, YI), method='linear')

            # Contour
            cs = ax.contour(XI, YI, ZI, levels=5, zdir='z', offset=depth,
                           cmap='coolwarm', alpha=0.7)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Profondeur (m)')
    ax.set_title('Contours 3D par profondeur')
    ax.invert_zaxis()

    return fig

def create_radar_chart(df):
    """Graphique radar pour les paramètres normalisés"""
    if len(df) < 3:
        return None

    # Normaliser les données
    features = ['qc', 'fs', 'Depth']
    if 'u2' in df.columns:
        features.append('u2')

    df_norm = df[features].copy()
    for col in features:
        df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    # Calculer les moyennes par cluster ou type de sol
    if 'Cluster' in df.columns:
        grouped = df_norm.groupby(df['Cluster']).mean()
        categories = [f'Cluster {i}' for i in grouped.index]
    elif 'Soil_Type' in df.columns:
        grouped = df_norm.groupby(df['Soil_Type']).mean()
        categories = grouped.index.tolist()
    else:
        # Moyenne générale
        grouped = df_norm.mean().to_frame().T
        categories = ['Moyenne']

    # Créer le radar chart
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]  # Fermer le cercle

    for i, (_, row) in enumerate(grouped.iterrows()):
        values = row.values.tolist()
        values += values[:1]  # Fermer le cercle

        ax.plot(angles, values, 'o-', linewidth=2, label=categories[i] if len(categories) > 1 else categories[0])
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_title('Profil Radar des Paramètres CPTU', size=16, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.grid(True)

    return fig

def create_violin_plots(df):
    """Violin plots pour distribution des paramètres"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Distributions des Paramètres CPTU', fontsize=16, fontweight='bold')

    # qc distribution
    sns.violinplot(data=df, y='qc', ax=axes[0,0], color='skyblue')
    axes[0,0].set_title('Distribution de qc')
    axes[0,0].set_ylabel('qc (MPa)')

    # fs distribution
    sns.violinplot(data=df, y='fs', ax=axes[0,1], color='lightcoral')
    axes[0,1].set_title('Distribution de fs')
    axes[0,1].set_ylabel('fs (MPa)')

    # Depth distribution
    sns.violinplot(data=df, y='Depth', ax=axes[1,0], color='lightgreen')
    axes[1,0].set_title('Distribution de profondeur')
    axes[1,0].set_ylabel('Profondeur (m)')

    # qc vs fs scatter avec violins marginaux
    if 'Soil_Type' in df.columns:
        sns.scatterplot(data=df, x='qc', y='fs', hue='Soil_Type', ax=axes[1,1], alpha=0.7)
    else:
        sns.scatterplot(data=df, x='qc', y='fs', ax=axes[1,1], alpha=0.7)
    axes[1,1].set_title('qc vs fs')
    axes[1,1].set_xlabel('qc (MPa)')
    axes[1,1].set_ylabel('fs (MPa)')

    plt.tight_layout()
    return fig

def create_3d_streamlines(df):
    """Visualisation 3D avec streamlines"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    x = df.get('x', df.index).values
    y = df.get('y', [0] * len(df)).values
    z = df['Depth'].values

    # Créer des streamlines basées sur le gradient de qc
    qc = df['qc'].values

    # Calculer les gradients
    dx = np.gradient(x)
    dy = np.gradient(y)
    dz = np.gradient(z)
    dqc = np.gradient(qc)

    # Normaliser
    magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
    magnitude[magnitude == 0] = 1  # Éviter division par zéro

    # Streamlines
    ax.quiver(x, y, z, dx/magnitude, dy/magnitude, dz/magnitude,
              length=0.1, normalize=True, color='blue', alpha=0.6)

    # Points colorés par qc
    scatter = ax.scatter(x, y, z, c=qc, cmap='plasma', s=50, alpha=0.8)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Profondeur (m)')
    ax.set_title('Streamlines 3D du gradient de qc')
    ax.invert_zaxis()

    plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label='qc (MPa)')

    return fig

def create_3d_isosurface(df):
    """Visualisation 3D avec isosurfaces"""
    try:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        x = df.get('x', df.index).values
        y = df.get('y', [0] * len(df)).values
        z = df['Depth'].values
        qc = df['qc'].values

        # Créer une grille 3D
        from scipy.interpolate import RegularGridInterpolator

        # Définir la grille
        x_grid = np.linspace(x.min(), x.max(), 20)
        y_grid = np.linspace(y.min(), y.max(), 20)
        z_grid = np.linspace(z.min(), z.max(), 20)

        X, Y, Z = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')

        # Interpoler qc sur la grille 3D
        points = np.column_stack((x, y, z))
        interp = RegularGridInterpolator((x_grid, y_grid, z_grid), qc,
                                       bounds_error=False, fill_value=np.mean(qc))

        QC_interp = interp((X, Y, Z))

        # Isosurface
        isosurface = ax.contour3D(X[:, :, 10], Y[:, :, 10], Z[:, :, 10],
                                 QC_interp[:, :, 10], levels=5, cmap='viridis', alpha=0.7)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Profondeur (m)')
        ax.set_title('Isosurface 3D de qc')
        ax.invert_zaxis()

        plt.colorbar(isosurface, ax=ax, shrink=0.5, aspect=5, label='qc (MPa)')

        return fig

    except Exception as e:
        st.warning(f"Impossible de créer l'isosurface 3D: {str(e)}")
        return create_3d_visualization(df)

def create_heatmaps_combined(df):
    """Heatmaps combinées pour corrélation et distribution"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Heatmaps et Corrélations', fontsize=16, fontweight='bold')

    # Matrice de corrélation
    numeric_cols = ['qc', 'fs', 'Depth']
    if 'u2' in df.columns:
        numeric_cols.append('u2')
    if 'CRR' in df.columns:
        numeric_cols.append('CRR')

    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                ax=axes[0,0], square=True, cbar_kws={'shrink': 0.8})
    axes[0,0].set_title('Matrice de Corrélation')

    # Heatmap qc vs profondeur
    pivot_qc = df.pivot_table(values='qc', index='Depth', aggfunc='mean')
    sns.heatmap(pivot_qc, cmap='viridis', ax=axes[0,1], cbar_kws={'label': 'qc (MPa)'})
    axes[0,1].set_title('Heatmap qc vs Profondeur')
    axes[0,1].invert_yaxis()

    # Heatmap fs vs profondeur
    pivot_fs = df.pivot_table(values='fs', index='Depth', aggfunc='mean')
    sns.heatmap(pivot_fs, cmap='plasma', ax=axes[1,0], cbar_kws={'label': 'fs (MPa)'})
    axes[1,0].set_title('Heatmap fs vs Profondeur')
    axes[1,0].invert_yaxis()

    # Scatter plot avec densité
    sns.kdeplot(data=df, x='qc', y='fs', fill=True, ax=axes[1,1], cmap='Blues', alpha=0.6)
    sns.scatterplot(data=df, x='qc', y='fs', ax=axes[1,1], alpha=0.6, s=30)
    axes[1,1].set_title('Densité qc vs fs')

    plt.tight_layout()
    return fig

def create_3d_voxels(df):
    """Visualisation 3D en voxels"""
    try:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        x = df.get('x', df.index).values
        y = df.get('y', [0] * len(df)).values
        z = df['Depth'].values

        # Discrétiser l'espace
        x_bins = np.linspace(x.min(), x.max(), 10)
        y_bins = np.linspace(y.min(), y.max(), 10)
        z_bins = np.linspace(z.min(), z.max(), 10)

        # Calculer la moyenne de qc dans chaque voxel
        qc_voxels = np.zeros((9, 9, 9))

        for i in range(9):
            for j in range(9):
                for k in range(9):
                    mask = ((x >= x_bins[i]) & (x < x_bins[i+1]) &
                           (y >= y_bins[j]) & (y < y_bins[j+1]) &
                           (z >= z_bins[k]) & (z < z_bins[k+1]))
                    if mask.any():
                        qc_voxels[i, j, k] = df[mask]['qc'].mean()
                    else:
                        qc_voxels[i, j, k] = np.nan

        # Créer les voxels
        X, Y, Z = np.meshgrid(x_bins[:-1], y_bins[:-1], z_bins[:-1])

        # Filtrer les voxels non vides
        mask = ~np.isnan(qc_voxels)
        if mask.any():
            voxels = ax.voxels(X, Y, Z, mask, facecolors=plt.cm.viridis(qc_voxels[mask]/qc_voxels[mask].max()),
                              edgecolor='k', alpha=0.7)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Profondeur (m)')
        ax.set_title('Visualisation 3D en Voxels')
        ax.invert_zaxis()

        return fig

    except Exception as e:
        st.warning(f"Impossible de créer la visualisation voxels: {str(e)}")
        return create_3d_visualization(df)

def create_depth_profile(df):
    """Crée un profil de profondeur amélioré avec qc et fs"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 8))

    # Profil qc
    ax1.plot(df['qc'], df['Depth'], 'b-', linewidth=3, label='qc', alpha=0.8)
    ax1.fill_betweenx(df['Depth'], df['qc'], alpha=0.3, color='blue')
    ax1.set_xlabel('Résistance de pointe qc (MPa)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Profondeur (m)', fontsize=12, fontweight='bold')
    ax1.set_title('Profil de qc', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()

    # Profil fs
    ax2.plot(df['fs'], df['Depth'], 'r-', linewidth=3, label='fs', alpha=0.8)
    ax2.fill_betweenx(df['Depth'], df['fs'], alpha=0.3, color='red')
    ax2.set_xlabel('Résistance de frottement fs (MPa)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Profondeur (m)', fontsize=12, fontweight='bold')
    ax2.set_title('Profil de fs', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()

    # Profil combiné
    ax3.plot(df['qc'], df['Depth'], 'b-', linewidth=2, label='qc', alpha=0.8)
    ax3.plot(df['fs'], df['Depth'], 'r-', linewidth=2, label='fs', alpha=0.8)
    ax3.fill_betweenx(df['Depth'], df['qc'], alpha=0.2, color='blue')
    ax3.fill_betweenx(df['Depth'], df['fs'], alpha=0.2, color='red')
    ax3.set_xlabel('Résistance (MPa)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Profondeur (m)', fontsize=12, fontweight='bold')
    ax3.set_title('Profils combinés', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.invert_yaxis()

    plt.tight_layout()
    return fig

def create_3d_visualization(df):
    """Crée une visualisation 3D améliorée des données CPTU"""
    try:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Couleur par type de sol ou cluster
        if 'Soil_Type' in df.columns:
            soil_types = df['Soil_Type'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(soil_types)))

            for soil_type, color in zip(soil_types, colors):
                mask = df['Soil_Type'] == soil_type
                subset = df[mask]

                x = subset.get('x', subset.index)
                y = subset.get('y', [0] * len(subset))
                z = subset['Depth']

                ax.scatter(x, y, z, c=[color], label=soil_type, alpha=0.8, s=60, edgecolors='black', linewidth=0.5)

        elif 'Cluster' in df.columns:
            n_clusters = df['Cluster'].max() + 1
            colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))

            for cluster in range(n_clusters):
                mask = df['Cluster'] == cluster
                subset = df[mask]

                x = subset.get('x', subset.index)
                y = subset.get('y', [0] * len(subset))
                z = subset['Depth']

                ax.scatter(x, y, z, c=[colors[cluster]], label=f'Cluster {cluster}', alpha=0.8, s=60, edgecolors='black', linewidth=0.5)
        else:
            x = df.get('x', df.index)
            y = df.get('y', [0] * len(df))
            z = df['Depth']
            scatter = ax.scatter(x, y, z, c=df['qc'], cmap='viridis', alpha=0.8, s=60, edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, label='qc (MPa)')

        ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Profondeur (m)', fontsize=12, fontweight='bold')
        ax.set_title('Visualisation 3D des données CPTU', fontsize=14, fontweight='bold')
        ax.invert_zaxis()
        ax.legend()

        return fig

    except Exception as e:
        st.error(f"Erreur lors de la création de la visualisation 3D: {str(e)}")
        return None


class PlotManager:
    """Gestionnaire de visualisation pour l'analyse géotechnique"""

    def __init__(self):
        """Initialise le gestionnaire de visualisation"""
        self.themes = {
            'dark': {
                'background': '#1e1e1e',
                'text': '#ffffff',
                'grid': '#333333',
                'colors': ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#f0932b']
            },
            'light': {
                'background': '#ffffff',
                'text': '#333333',
                'grid': '#cccccc',
                'colors': ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
            }
        }
        self.current_theme = 'dark'

    def set_theme(self, theme_name):
        """Définit le thème de visualisation"""
        if theme_name in self.themes:
            self.current_theme = theme_name

    def create_advanced_plot(self, plot_type, data, **kwargs):
        """Crée un graphique avancé selon le type spécifié"""
        try:
            if plot_type == 'qc_vs_fs':
                return self._create_qc_fs_plot(data, **kwargs)
            elif plot_type == 'depth_profile':
                return self._create_depth_profile(data, **kwargs)
            elif plot_type == 'soil_classification':
                return self._create_soil_classification_plot(data, **kwargs)
            elif plot_type == 'correlation_matrix':
                return self._create_correlation_matrix(data, **kwargs)
            elif plot_type == '3d_visualization':
                return self._create_3d_visualization(data, **kwargs)
            elif plot_type == 'dashboard':
                return self._create_combined_dashboard(data, **kwargs)
            else:
                st.error(f"Type de graphique non supporté: {plot_type}")
                return None
        except Exception as e:
            st.error(f"Erreur lors de la création du graphique {plot_type}: {str(e)}")
            return None

    def _create_qc_fs_plot(self, df, **kwargs):
        """Crée un graphique qc vs fs avec thème"""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Appliquer le thème
        theme = self.themes[self.current_theme]
        fig.patch.set_facecolor(theme['background'])
        ax.set_facecolor(theme['background'])

        # Couleur par type de sol
        if 'Soil_Type' in df.columns:
            soil_types = df['Soil_Type'].unique()
            colors = theme['colors'][:len(soil_types)]

            for soil_type, color in zip(soil_types, colors):
                mask = df['Soil_Type'] == soil_type
                ax.scatter(df[mask]['qc'], df[mask]['fs'],
                          c=color, label=soil_type, alpha=0.8, s=60,
                          edgecolors='white', linewidth=0.5)
        else:
            ax.scatter(df['qc'], df['fs'], alpha=0.8, s=60,
                      edgecolors='white', linewidth=0.5, c=theme['colors'][0])

        ax.set_xlabel('Résistance de pointe qc (MPa)', fontsize=12, fontweight='bold',
                     color=theme['text'])
        ax.set_ylabel('Résistance de frottement fs (MPa)', fontsize=12, fontweight='bold',
                     color=theme['text'])
        ax.set_title('Graphique qc vs fs - Classification des sols', fontsize=14,
                    fontweight='bold', pad=20, color=theme['text'])
        ax.grid(True, alpha=0.3, color=theme['grid'])
        ax.tick_params(colors=theme['text'])

        # Lignes de référence
        ax.axhline(y=0.5, color=theme['colors'][1], linestyle='--', alpha=0.7)
        ax.axvline(x=5, color=theme['colors'][2], linestyle='--', alpha=0.7)

        legend = ax.legend(fontsize=10)
        plt.setp(legend.get_texts(), color=theme['text'])

        return fig

    def _create_depth_profile(self, df, **kwargs):
        """Crée un profil de profondeur"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        theme = self.themes[self.current_theme]
        fig.patch.set_facecolor(theme['background'])

        for ax in [ax1, ax2, ax3]:
            ax.set_facecolor(theme['background'])
            ax.tick_params(colors=theme['text'])
            ax.grid(True, alpha=0.3, color=theme['grid'])

        # qc vs profondeur
        ax1.plot(df['qc'], df['Depth'], 'o-', color=theme['colors'][0], linewidth=2)
        ax1.set_xlabel('qc (MPa)', color=theme['text'])
        ax1.set_ylabel('Profondeur (m)', color=theme['text'])
        ax1.set_title('Résistance de pointe', color=theme['text'])
        ax1.invert_yaxis()

        # fs vs profondeur
        ax2.plot(df['fs'], df['Depth'], 's-', color=theme['colors'][1], linewidth=2)
        ax2.set_xlabel('fs (MPa)', color=theme['text'])
        ax2.set_title('Résistance de frottement', color=theme['text'])
        ax2.invert_yaxis()

        # Rf vs profondeur
        if 'Rf' in df.columns:
            ax3.plot(df['Rf'], df['Depth'], '^-', color=theme['colors'][2], linewidth=2)
            ax3.set_xlabel('Rf (%)', color=theme['text'])
            ax3.set_title('Coefficient de frottement', color=theme['text'])
            ax3.invert_yaxis()

        fig.suptitle('Profils géotechniques', fontsize=16, fontweight='bold',
                    color=theme['text'])

        return fig

    def _create_soil_classification_plot(self, df, **kwargs):
        """Crée un graphique de classification des sols"""
        if 'Soil_Type' not in df.columns:
            st.warning("Colonne 'Soil_Type' manquante pour la classification")
            return None

        fig, ax = plt.subplots(figsize=(10, 8))

        theme = self.themes[self.current_theme]
        fig.patch.set_facecolor(theme['background'])
        ax.set_facecolor(theme['background'])

        soil_counts = df['Soil_Type'].value_counts()
        colors = theme['colors'][:len(soil_counts)]

        bars = ax.bar(range(len(soil_counts)), soil_counts.values,
                     color=colors, alpha=0.8, edgecolor='white', linewidth=1)

        ax.set_xlabel('Type de sol', fontsize=12, fontweight='bold', color=theme['text'])
        ax.set_ylabel('Nombre d\'échantillons', fontsize=12, fontweight='bold',
                     color=theme['text'])
        ax.set_title('Distribution des types de sol', fontsize=14, fontweight='bold',
                    color=theme['text'])
        ax.set_xticks(range(len(soil_counts)))
        ax.set_xticklabels(soil_counts.index, rotation=45, ha='right',
                          color=theme['text'])
        ax.tick_params(colors=theme['text'])
        ax.grid(True, alpha=0.3, color=theme['grid'], axis='y')

        # Ajouter les valeurs sur les barres
        for bar, count in zip(bars, soil_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{count}', ha='center', va='bottom', fontweight='bold',
                   color=theme['text'])

        return fig

    def _create_correlation_matrix(self, df, **kwargs):
        """Crée une matrice de corrélation"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            st.warning("Pas assez de colonnes numériques pour la matrice de corrélation")
            return None

        corr_matrix = df[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 8))

        theme = self.themes[self.current_theme]
        fig.patch.set_facecolor(theme['background'])
        ax.set_facecolor(theme['background'])

        im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha='right', color=theme['text'])
        ax.set_yticklabels(numeric_cols, color=theme['text'])

        ax.set_title('Matrice de corrélation', fontsize=14, fontweight='bold',
                    color=theme['text'])

        # Ajouter les coefficients
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha='center', va='center', color='white', fontweight='bold')

        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(colors=theme['text'])

        return fig

    def _create_3d_visualization(self, df, **kwargs):
        """Crée une visualisation 3D"""
        if not all(col in df.columns for col in ['qc', 'fs', 'Depth']):
            st.warning("Colonnes requises manquantes pour la visualisation 3D")
            return None

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        theme = self.themes[self.current_theme]
        fig.patch.set_facecolor(theme['background'])
        ax.set_facecolor(theme['background'])

        scatter = ax.scatter(df['qc'], df['fs'], df['Depth'],
                           c=df['Depth'], cmap='viridis', alpha=0.8, s=50)

        ax.set_xlabel('qc (MPa)', fontsize=12, fontweight='bold', color=theme['text'])
        ax.set_ylabel('fs (MPa)', fontsize=12, fontweight='bold', color=theme['text'])
        ax.set_zlabel('Profondeur (m)', fontsize=12, fontweight='bold', color=theme['text'])
        ax.set_title('Visualisation 3D des données CPTU', fontsize=14, fontweight='bold',
                    color=theme['text'])
        ax.invert_zaxis()
        ax.tick_params(colors=theme['text'])

        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.ax.tick_params(colors=theme['text'])
        cbar.set_label('Profondeur (m)', color=theme['text'])

        return fig

    def _create_combined_dashboard(self, df, **kwargs):
        """Crée un dashboard combiné"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('qc vs fs', 'Profil qc', 'Profil fs',
                          'Classification', 'Corrélation', 'Distribution qc'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'heatmap'}, {'type': 'histogram'}]]
        )

        theme = self.themes[self.current_theme]

        # qc vs fs
        fig.add_trace(
            go.Scatter(x=df['qc'], y=df['fs'], mode='markers',
                      marker=dict(color=theme['colors'][0], size=8)),
            row=1, col=1
        )

        # Profil qc
        fig.add_trace(
            go.Scatter(x=df['qc'], y=df['Depth'], mode='lines+markers',
                      line=dict(color=theme['colors'][1])),
            row=1, col=2
        )

        # Profil fs
        fig.add_trace(
            go.Scatter(x=df['fs'], y=df['Depth'], mode='lines+markers',
                      line=dict(color=theme['colors'][2])),
            row=1, col=3
        )

        # Classification des sols
        if 'Soil_Type' in df.columns:
            soil_counts = df['Soil_Type'].value_counts()
            fig.add_trace(
                go.Bar(x=soil_counts.index, y=soil_counts.values,
                      marker_color=theme['colors'][3]),
                row=2, col=1
            )

        # Matrice de corrélation
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]  # Limiter à 5 colonnes
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr().values
            fig.add_trace(
                go.Heatmap(z=corr_matrix, x=numeric_cols, y=numeric_cols,
                          colorscale='RdBu', zmin=-1, zmax=1),
                row=2, col=2
            )

        # Distribution qc
        fig.add_trace(
            go.Histogram(x=df['qc'], nbinsx=30,
                        marker_color=theme['colors'][4]),
            row=2, col=3
        )

        # Mise à jour du layout
        fig.update_layout(
            height=800,
            title_text="Dashboard d'analyse géotechnique",
            title_font_size=16,
            paper_bgcolor=theme['background'],
            plot_bgcolor=theme['background'],
            font_color=theme['text']
        )

        # Inverser l'axe Y pour les profils
        fig.update_yaxes(autorange="reversed", row=1, col=2)
        fig.update_yaxes(autorange="reversed", row=1, col=3)

        return fig