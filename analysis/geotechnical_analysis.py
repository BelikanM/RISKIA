#!/usr/bin/env python3
"""
Fonctions d'analyse géotechnique avancées pour CPT/CPTU
Analyse 3D des couches, classification détaillée des sols, géolocalisation
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

# Import GeoPandas pour la géolocalisation
try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    st.warning("⚠️ GeoPandas non installé. Géolocalisation limitée.")

def estimate_soil_type(df):
    """Classification détaillée des sols basée sur Robertson (1990) et Schmertmann (1978)"""
    def classify_soil_detailed(qc, fs, depth):
        if pd.isna(qc) or pd.isna(fs) or qc <= 0 or fs < 0:
            return 'Inconnu', 'Unknown', 0, 0, 0

        # Calcul des indices de Robertson
        fr = (fs / qc) * 100  # Friction ratio (%)
        Ic = ((3.47 - np.log10(qc))**2 + (np.log10(fr) + 1.22)**2)**0.5  # Soil Behavior Type Index

        # Classification détaillée selon Robertson (1986)
        if Ic < 1.31:
            if fr < 0.5:
                soil_type = 'Sable graveleux très dense'
                soil_class = 'Gravel'
                color = '#8B4513'  # Brown
            else:
                soil_type = 'Sable silteux dense'
                soil_class = 'Sand'
                color = '#DAA520'  # Goldenrod
        elif Ic < 2.05:
            if fr < 1:
                soil_type = 'Sable dense à très dense'
                soil_class = 'Sand'
                color = '#FFD700'  # Gold
            else:
                soil_type = 'Sable silteux'
                soil_class = 'Sandy Silt'
                color = '#F0E68C'  # Khaki
        elif Ic < 2.60:
            if fr < 2:
                soil_type = 'Sable lâche à compact'
                soil_class = 'Sand'
                color = '#FFE4B5'  # Moccasin
            else:
                soil_type = 'Silt sableux'
                soil_class = 'Silty Sand'
                color = '#DEB887'  # Burlywood
        elif Ic < 2.95:
            if fr < 4:
                soil_type = 'Silt argileux'
                soil_class = 'Clayey Silt'
                color = '#D2B48C'  # Tan
            else:
                soil_type = 'Argile silteuse'
                soil_class = 'Silty Clay'
                color = '#BC8F8F'  # Rosy Brown
        elif Ic < 3.60:
            soil_type = 'Argile'
            soil_class = 'Clay'
            color = '#CD853F'  # Peru
        else:
            soil_type = 'Argile organique/molte'
            soil_class = 'Organic Clay'
            color = '#A0522D'  # Sienna

        # Ajustements basés sur la profondeur (diagenèse)
        if depth > 20:  # Sols profonds plus consolidés
            if 'Sable' in soil_type and qc > 15:
                soil_type += ' (cimenté)'
            elif 'Argile' in soil_type and qc > 8:
                soil_type += ' (consolidée)'

        return soil_type, soil_class, Ic, fr, color

    df_copy = df.copy()
    results = df_copy.apply(lambda row: classify_soil_detailed(row['qc'], row['fs'], row['Depth']), axis=1)

    df_copy['Soil_Type_Detailed'] = [r[0] for r in results]
    df_copy['Soil_Class'] = [r[1] for r in results]
    df_copy['Ic'] = [r[2] for r in results]
    df_copy['Fr'] = [r[3] for r in results]
    df_copy['Soil_Color'] = [r[4] for r in results]

    # Classification simplifiée pour compatibilité
    df_copy['Soil_Type'] = df_copy['Soil_Type_Detailed'].map({
        'Sable graveleux très dense': 'Sable dense',
        'Sable silteux dense': 'Sable dense',
        'Sable dense à très dense': 'Sable dense',
        'Sable silteux': 'Sable',
        'Sable lâche à compact': 'Sable',
        'Silt sableux': 'Sable',
        'Silt argileux': 'Limon',
        'Argile silteuse': 'Argile',
        'Argile': 'Argile',
        'Argile organique/molte': 'Argile molle'
    }).fillna('Inconnu')

    return df_copy

def calculate_crr(df):
    """Calcule le Cyclic Resistance Ratio (CRR) avec analyse avancée"""
    df_copy = df.copy()

    # Estimation de la contrainte verticale effective
    gamma = 18  # Poids volumique moyen (kN/m³)
    sigma_v = df_copy['Depth'] * gamma  # Contrainte verticale totale
    sigma_vo = sigma_v * 0.5  # Estimation simplifiée de la contrainte effective

    # Normalisation qc1N selon Robertson & Wride (1998)
    df_copy['qc1N'] = df_copy['qc'] * (100 / sigma_vo)**0.5

    # CRR selon Idriss & Boulanger (2008) pour magnitude 7.5
    df_copy['CRR'] = np.exp(df_copy['qc1N']/113 + (df_copy['qc1N']/1000)**2 - 3.5) / 2.36

    # Facteur de sécurité contre la liquéfaction
    df_copy['FS_Liquefaction'] = df_copy['CRR'] / 0.3  # CSR = 0.3 pour M=7.5

    # Classification du risque de liquéfaction
    df_copy['Liquefaction_Risk'] = pd.cut(df_copy['FS_Liquefaction'],
                                         bins=[0, 1, 1.2, 1.5, np.inf],
                                         labels=['Très élevé', 'Élevé', 'Modéré', 'Faible'])

    return df_copy

def identify_soil_layers_3d(df, min_thickness=0.5):
    """Identifie les couches géologiques en 3D avec épaisseurs et transitions"""
    layers = []
    current_layer_start = df['Depth'].min()
    current_soil_type = df.iloc[0]['Soil_Type_Detailed']
    current_color = df.iloc[0]['Soil_Color']

    for i in range(1, len(df)):
        if df.iloc[i]['Soil_Type_Detailed'] != current_soil_type:
            # Fin de couche détectée
            thickness = df.iloc[i-1]['Depth'] - current_layer_start
            if thickness >= min_thickness:
                layers.append({
                    'start_depth': current_layer_start,
                    'end_depth': df.iloc[i-1]['Depth'],
                    'thickness': thickness,
                    'soil_type': current_soil_type,
                    'soil_class': df.iloc[i-1]['Soil_Class'],
                    'color': current_color,
                    'avg_qc': df.iloc[i-1]['qc'],
                    'avg_fs': df.iloc[i-1]['fs'],
                    'avg_Ic': df.iloc[i-1]['Ic']
                })
            current_layer_start = df.iloc[i]['Depth']
            current_soil_type = df.iloc[i]['Soil_Type_Detailed']
            current_color = df.iloc[i]['Soil_Color']

    # Dernière couche
    thickness = df.iloc[-1]['Depth'] - current_layer_start
    if thickness >= min_thickness:
        layers.append({
            'start_depth': current_layer_start,
            'end_depth': df.iloc[-1]['Depth'],
            'thickness': thickness,
            'soil_type': current_soil_type,
            'soil_class': df.iloc[-1]['Soil_Class'],
            'color': current_color,
            'avg_qc': df.iloc[-1]['qc'],
            'avg_fs': df.iloc[-1]['fs'],
            'avg_Ic': df.iloc[-1]['Ic']
        })

    return pd.DataFrame(layers)

def create_geospatial_analysis(df, lat=48.8566, lon=2.3522):
    """Crée une analyse géospatiale avec GeoPandas"""
    if not GEOPANDAS_AVAILABLE:
        st.warning("GeoPandas non disponible. Analyse géospatiale limitée.")
        return None

    try:
        # Créer des points géographiques (simulation autour d'un point central)
        np.random.seed(42)
        n_points = len(df)

        # Distribution gaussienne autour du point central
        lats = np.random.normal(lat, 0.01, n_points)  # ~1km d'écart
        lons = np.random.normal(lon, 0.01, n_points)

        # Créer GeoDataFrame
        geometry = [Point(xy) for xy in zip(lons, lats)]
        gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs='EPSG:4326')

        # Ajouter des attributs géographiques
        gdf['latitude'] = lats
        gdf['longitude'] = lons
        gdf['elevation'] = 50 + np.random.normal(0, 5, n_points)  # Élévation simulée

        return gdf

    except Exception as e:
        st.error(f"Erreur lors de l'analyse géospatiale: {e}")
        return None

def create_advanced_visualizations(df, layers_df=None, gdf=None):
    """Crée 10+ graphiques et tableaux avancés pour l'analyse géotechnique"""

    visualizations = {}

    # 1. Profil 3D des couches géologiques
    if layers_df is not None:
        fig_3d_layers = go.Figure()

        for _, layer in layers_df.iterrows():
            fig_3d_layers.add_trace(go.Scatter3d(
                x=[0, 1, 1, 0, 0],
                y=[layer['start_depth'], layer['start_depth'], layer['end_depth'], layer['end_depth'], layer['start_depth']],
                z=[0, 0, 0, 0, 0],
                mode='lines',
                line=dict(color=layer['color'], width=10),
                name=f"{layer['soil_type']} ({layer['thickness']:.1f}m)",
                showlegend=True
            ))

        fig_3d_layers.update_layout(
            title="Profil 3D des Couches Géologiques",
            scene=dict(
                xaxis_title="Position X",
                yaxis_title="Profondeur (m)",
                zaxis_title="Position Z"
            )
        )
        visualizations['3d_layers'] = fig_3d_layers

    # 2. Carte de chaleur Ic vs Profondeur
    fig_ic_heatmap = go.Figure(data=go.Heatmap(
        z=df['Ic'],
        x=df['Depth'],
        y=df['qc'],
        colorscale='Viridis',
        name='Indice Ic'
    ))
    fig_ic_heatmap.update_layout(
        title="Carte de Chaleur - Indice Ic vs Profondeur",
        xaxis_title="Profondeur (m)",
        yaxis_title="qc (MPa)"
    )
    visualizations['ic_heatmap'] = fig_ic_heatmap

    # 3. Histogramme des types de sols avec distribution
    soil_counts = df['Soil_Type_Detailed'].value_counts()
    fig_soil_dist = px.bar(
        x=soil_counts.index,
        y=soil_counts.values,
        color=soil_counts.index,
        title="Distribution des Types de Sols Détaillés"
    )
    fig_soil_dist.update_layout(xaxis_title="Type de Sol", yaxis_title="Nombre d'échantillons")
    visualizations['soil_distribution'] = fig_soil_dist

    # 4. Graphique radar des propriétés moyennes par couche
    if layers_df is not None:
        categories = ['Épaisseur', 'qc moyen', 'fs moyen', 'Ic moyen']

        fig_radar = go.Figure()

        for _, layer in layers_df.iterrows():
            values = [
                layer['thickness'],
                layer['avg_qc'],
                layer['avg_fs'],
                layer['avg_Ic']
            ]
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=f"{layer['soil_type'][:20]}..."
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            title="Propriétés Moyennes par Couche Géologique"
        )
        visualizations['radar_properties'] = fig_radar

    # 5. Analyse de tendance avec lissage
    window_size = min(21, len(df) // 2 * 2 + 1)  # Taille impaire
    df_smooth = df.copy()
    df_smooth['qc_smooth'] = savgol_filter(df['qc'], window_size, 3)
    df_smooth['fs_smooth'] = savgol_filter(df['fs'], window_size, 3)

    fig_trends = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              subplot_titles=['Résistance de Pointe (qc)', 'Frottement de Manche (fs)'])

    fig_trends.add_trace(go.Scatter(x=df['Depth'], y=df['qc'], mode='markers', name='qc brut',
                                   marker=dict(size=3, color='lightblue')), row=1, col=1)
    fig_trends.add_trace(go.Scatter(x=df_smooth['Depth'], y=df_smooth['qc_smooth'],
                                   mode='lines', name='qc lissé', line=dict(color='blue', width=2)), row=1, col=1)

    fig_trends.add_trace(go.Scatter(x=df['Depth'], y=df['fs'], mode='markers', name='fs brut',
                                   marker=dict(size=3, color='lightcoral')), row=2, col=1)
    fig_trends.add_trace(go.Scatter(x=df_smooth['Depth'], y=df_smooth['fs_smooth'],
                                   mode='lines', name='fs lissé', line=dict(color='red', width=2)), row=2, col=1)

    fig_trends.update_layout(title="Analyse de Tendance avec Lissage Savitzky-Golay")
    visualizations['trend_analysis'] = fig_trends

    # 6. Diagramme de dispersion qc/fs coloré par type de sol
    fig_scatter = px.scatter(df, x='qc', y='fs', color='Soil_Type_Detailed',
                            title="Corrélation qc/fs par Type de Sol",
                            labels={'qc': 'Résistance de Pointe (MPa)', 'fs': 'Frottement de Manche (MPa)'})
    visualizations['correlation_scatter'] = fig_scatter

    # 7. Profil de risque de liquéfaction
    if 'FS_Liquefaction' in df.columns:
        fig_liq = make_subplots(rows=1, cols=2,
                               subplot_titles=['Facteur de Sécurité', 'Risque de Liquéfaction'])

        fig_liq.add_trace(go.Scatter(x=df['FS_Liquefaction'], y=df['Depth'], mode='lines+markers',
                                    name='FS', line=dict(color='red')), row=1, col=1)

        risk_colors = {'Très élevé': 'darkred', 'Élevé': 'red', 'Modéré': 'orange', 'Faible': 'green'}
        for risk in df['Liquefaction_Risk'].unique():
            mask = df['Liquefaction_Risk'] == risk
            fig_liq.add_trace(go.Scatter(
                x=df[mask]['Depth'],
                y=[1] * mask.sum(),
                mode='markers',
                marker=dict(color=risk_colors.get(risk, 'gray'), size=10),
                name=risk
            ), row=1, col=2)

        fig_liq.update_layout(title="Analyse du Risque de Liquéfaction")
        visualizations['liquefaction_profile'] = fig_liq

    # 8. Statistiques descriptives par couche
    if layers_df is not None:
        stats_data = []
        for _, layer in layers_df.iterrows():
            mask = (df['Depth'] >= layer['start_depth']) & (df['Depth'] <= layer['end_depth'])
            layer_data = df[mask]

            stats_data.append({
                'Couche': layer['soil_type'][:30],
                'Épaisseur (m)': f"{layer['thickness']:.1f}",
                'Profondeur (m)': f"{layer['start_depth']:.1f}-{layer['end_depth']:.1f}",
                'qc moyen (MPa)': f"{layer_data['qc'].mean():.1f}",
                'qc min-max (MPa)': f"{layer_data['qc'].min():.1f}-{layer_data['qc'].max():.1f}",
                'fs moyen (MPa)': f"{layer_data['fs'].mean():.1f}",
                'Ic moyen': f"{layer_data['Ic'].mean():.2f}",
                'Échantillons': len(layer_data)
            })

        stats_df = pd.DataFrame(stats_data)
        visualizations['layer_statistics'] = stats_df

    # 9. Analyse fréquentielle (FFT) des variations
    if len(df) > 32:  # Minimum pour FFT
        qc_fft = np.fft.fft(df['qc'].values)
        freqs = np.fft.fftfreq(len(df), d=(df['Depth'].diff().mean()))

        fig_fft = make_subplots(rows=1, cols=2,
                               subplot_titles=['Spectre de Fréquence qc', 'Périodogramme'])

        fig_fft.add_trace(go.Scatter(x=freqs[:len(freqs)//2], y=np.abs(qc_fft)[:len(qc_fft)//2],
                                    mode='lines', name='Amplitude'), row=1, col=1)

        fig_fft.add_trace(go.Scatter(x=df['Depth'], y=df['qc'], mode='lines', name='Signal original'), row=1, col=2)
        fig_fft.add_trace(go.Scatter(x=df['Depth'], y=savgol_filter(df['qc'], 11, 3),
                                    mode='lines', name='Tendance', line=dict(dash='dash')), row=1, col=2)

        fig_fft.update_layout(title="Analyse Fréquentielle des Variations de qc")
        visualizations['frequency_analysis'] = fig_fft

    # 10. Remplacement de la carte géographique par analyse de zones CPTU
    try:
        # Créer des coordonnées simulées pour les zones CPTU
        np.random.seed(42)
        n_zones = min(10, len(df))  # Maximum 10 zones
        zone_centers = []

        # Créer des centres de zones distribués
        for i in range(n_zones):
            angle = 2 * np.pi * i / n_zones
            radius = 50 + np.random.uniform(-20, 20)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            zone_centers.append((x, y))

        # Assigner chaque point à une zone
        df_zones = df.copy()
        df_zones['zone_id'] = np.random.randint(0, n_zones, len(df))
        df_zones['zone_x'] = df_zones['zone_id'].map(lambda i: zone_centers[i][0])
        df_zones['zone_y'] = df_zones['zone_id'].map(lambda i: zone_centers[i][1])

        # Graphique 3D des zones CPTU
        fig_zones_3d = go.Figure()

        for zone_id in range(n_zones):
            zone_data = df_zones[df_zones['zone_id'] == zone_id]
            if not zone_data.empty:
                fig_zones_3d.add_trace(go.Scatter3d(
                    x=zone_data['zone_x'],
                    y=zone_data['zone_y'],
                    z=zone_data['Depth'],
                    mode='markers',
                    name=f'Zone {zone_id + 1}',
                    marker=dict(
                        size=6,
                        color=zone_data['qc'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="qc (MPa)")
                    ),
                    text=[f"Zone {zone_id + 1}<br>Profondeur: {d:.1f}m<br>qc: {q:.1f}MPa"
                          for d, q in zip(zone_data['Depth'], zone_data['qc'])]
                ))

        fig_zones_3d.update_layout(
            title="Zones CPTU 3D avec Distribution Spatiale",
            scene=dict(
                xaxis_title="Position X (m)",
                yaxis_title="Position Y (m)",
                zaxis_title="Profondeur (m)",
                zaxis=dict(autorange="reversed")
            )
        )
        visualizations['cptu_zones_3d'] = fig_zones_3d

    except Exception as e:
        st.warning(f"Erreur lors de la création des zones CPTU: {e}")

    # === 10 NOUVEAUX GRAPHIQUES 3D AVEC TRIANGULATION ===

    # 11. Surface triangulée 3D des types de sol
    try:
        from scipy.spatial import Delaunay

        # Créer une grille de points pour la triangulation
        x_grid = np.linspace(df['Depth'].min(), df['Depth'].max(), 20)
        y_grid = np.linspace(0, 100, 20)  # Position latérale simulée
        X, Y = np.meshgrid(x_grid, y_grid)
        X = X.flatten()
        Y = Y.flatten()

        # Interpoler les valeurs de qc sur la grille
        from scipy.interpolate import griddata
        qc_interp = griddata(
            (df['Depth'], np.random.uniform(0, 100, len(df))),
            df['qc'],
            (X, Y),
            method='linear'
        )

        # Triangulation
        points = np.column_stack([X, Y])
        tri = Delaunay(points)

        # Créer la surface 3D triangulée
        fig_triangulated = go.Figure()

        fig_triangulated.add_trace(go.Mesh3d(
            x=X,
            y=Y,
            z=qc_interp,
            i=tri.simplices[:, 0],
            j=tri.simplices[:, 1],
            k=tri.simplices[:, 2],
            opacity=0.8,
            color='lightblue',
            name='Surface qc'
        ))

        # Ajouter les points de données réels
        fig_triangulated.add_trace(go.Scatter3d(
            x=df['Depth'],
            y=np.random.uniform(0, 100, len(df)),
            z=df['qc'],
            mode='markers',
            marker=dict(size=4, color='red', opacity=0.7),
            name='Points réels'
        ))

        fig_triangulated.update_layout(
            title="Surface Triangulée 3D - Résistance de Pointe (qc)",
            scene=dict(
                xaxis_title="Profondeur (m)",
                yaxis_title="Position Latérale (m)",
                zaxis_title="qc (MPa)"
            )
        )
        visualizations['triangulated_surface_qc'] = fig_triangulated

    except Exception as e:
        st.warning(f"Erreur lors de la triangulation qc: {e}")

    # 12. Surface triangulée Ic vs Profondeur
    try:
        ic_interp = griddata(
            (df['Depth'], np.random.uniform(0, 100, len(df))),
            df['Ic'],
            (X, Y),
            method='linear'
        )

        fig_triangulated_ic = go.Figure()

        fig_triangulated_ic.add_trace(go.Mesh3d(
            x=X,
            y=Y,
            z=ic_interp,
            i=tri.simplices[:, 0],
            j=tri.simplices[:, 1],
            k=tri.simplices[:, 2],
            opacity=0.8,
            colorscale='Viridis',
            intensity=ic_interp,
            name='Surface Ic'
        ))

        fig_triangulated_ic.add_trace(go.Scatter3d(
            x=df['Depth'],
            y=np.random.uniform(0, 100, len(df)),
            z=df['Ic'],
            mode='markers',
            marker=dict(size=4, color='red', opacity=0.7),
            name='Points réels'
        ))

        fig_triangulated_ic.update_layout(
            title="Surface Triangulée 3D - Indice Ic (Soil Behavior Type)",
            scene=dict(
                xaxis_title="Profondeur (m)",
                yaxis_title="Position Latérale (m)",
                zaxis_title="Indice Ic"
            )
        )
        visualizations['triangulated_surface_ic'] = fig_triangulated_ic

    except Exception as e:
        st.warning(f"Erreur lors de la triangulation Ic: {e}")

    # 13. Volume 3D des couches géologiques avec triangulation
    try:
        if layers_df is not None and not layers_df.empty:
            fig_layers_volume = go.Figure()

            colors = ['#8B4513', '#DAA520', '#F4A460', '#DEB887', '#D2B48C', '#BC8F8F']

            for idx, layer in layers_df.iterrows():
                # Créer une surface pour chaque couche
                layer_mask = (df['Depth'] >= layer['start_depth']) & (df['Depth'] <= layer['end_depth'])
                layer_data = df[layer_mask]

                if not layer_data.empty:
                    # Points pour la couche
                    x_layer = np.random.uniform(0, 100, len(layer_data))
                    y_layer = layer_data['Depth']
                    z_layer = np.random.uniform(0, 50, len(layer_data))  # Épaisseur simulée

                    # Triangulation pour la couche
                    if len(layer_data) >= 3:
                        points_layer = np.column_stack([x_layer, y_layer])
                        tri_layer = Delaunay(points_layer)

                        fig_layers_volume.add_trace(go.Mesh3d(
                            x=x_layer,
                            y=y_layer,
                            z=z_layer,
                            i=tri_layer.simplices[:, 0],
                            j=tri_layer.simplices[:, 1],
                            k=tri_layer.simplices[:, 2],
                            opacity=0.7,
                            color=colors[idx % len(colors)],
                            name=f"{layer['soil_type'][:20]}..."
                        ))

            fig_layers_volume.update_layout(
                title="Volume 3D Triangulé des Couches Géologiques",
                scene=dict(
                    xaxis_title="Position X (m)",
                    yaxis_title="Profondeur (m)",
                    zaxis_title="Épaisseur (m)"
                )
            )
            visualizations['layers_volume_3d'] = fig_layers_volume

    except Exception as e:
        st.warning(f"Erreur lors de la création du volume 3D: {e}")

    # 14. Surface 3D des risques de liquéfaction
    try:
        if 'FS_Liquefaction' in df.columns:
            fs_interp = griddata(
                (df['Depth'], np.random.uniform(0, 100, len(df))),
                df['FS_Liquefaction'],
                (X, Y),
                method='linear'
            )

            fig_liquefaction_3d = go.Figure()

            fig_liquefaction_3d.add_trace(go.Mesh3d(
                x=X,
                y=Y,
                z=fs_interp,
                i=tri.simplices[:, 0],
                j=tri.simplices[:, 1],
                k=tri.simplices[:, 2],
                opacity=0.8,
                colorscale='RdYlGn',
                intensity=fs_interp,
                name='FS Liquefaction'
            ))

            # Colorer selon le risque
            risk_colors = []
            for fs in df['FS_Liquefaction']:
                if fs < 1.2:
                    risk_colors.append('red')
                elif fs < 1.5:
                    risk_colors.append('orange')
                else:
                    risk_colors.append('green')

            fig_liquefaction_3d.add_trace(go.Scatter3d(
                x=df['Depth'],
                y=np.random.uniform(0, 100, len(df)),
                z=df['FS_Liquefaction'],
                mode='markers',
                marker=dict(size=6, color=risk_colors, opacity=0.8),
                name='Points de risque'
            ))

            fig_liquefaction_3d.update_layout(
                title="Surface 3D Triangulée - Risque de Liquéfaction",
                scene=dict(
                    xaxis_title="Profondeur (m)",
                    yaxis_title="Position Latérale (m)",
                    zaxis_title="FS Liquefaction"
                )
            )
            visualizations['liquefaction_surface_3d'] = fig_liquefaction_3d

    except Exception as e:
        st.warning(f"Erreur lors de la surface de liquéfaction: {e}")

    # 15. Topographie 3D des clusters
    try:
        if 'Cluster' in df.columns:
            n_clusters = df['Cluster'].max() + 1
            fig_clusters_3d = go.Figure()

            for cluster_id in range(n_clusters):
                cluster_data = df[df['Cluster'] == cluster_id]

                if not cluster_data.empty:
                    # Triangulation par cluster
                    x_cluster = np.random.uniform(0, 100, len(cluster_data))
                    y_cluster = cluster_data['Depth']
                    z_cluster = cluster_data['qc']

                    if len(cluster_data) >= 3:
                        points_cluster = np.column_stack([x_cluster, y_cluster])
                        tri_cluster = Delaunay(points_cluster)

                        fig_clusters_3d.add_trace(go.Mesh3d(
                            x=x_cluster,
                            y=y_cluster,
                            z=z_cluster,
                            i=tri_cluster.simplices[:, 0],
                            j=tri_cluster.simplices[:, 1],
                            k=tri_cluster.simplices[:, 2],
                            opacity=0.6,
                            name=f'Cluster {cluster_id}'
                        ))

            fig_clusters_3d.update_layout(
                title="Topographie 3D Triangulée par Clusters",
                scene=dict(
                    xaxis_title="Position X (m)",
                    yaxis_title="Profondeur (m)",
                    zaxis_title="qc (MPa)"
                )
            )
            visualizations['clusters_topography_3d'] = fig_clusters_3d

    except Exception as e:
        st.warning(f"Erreur lors de la topographie des clusters: {e}")

    # 16. Structure 3D des types de sol détaillés
    try:
        fig_soil_structure = go.Figure()

        soil_types = df['Soil_Type_Detailed'].unique()
        colors_soil = px.colors.qualitative.Set3

        for idx, soil_type in enumerate(soil_types):
            soil_data = df[df['Soil_Type_Detailed'] == soil_type]

            if not soil_data.empty and len(soil_data) >= 3:
                x_soil = np.random.uniform(0, 100, len(soil_data))
                y_soil = soil_data['Depth']
                z_soil = soil_data['qc']

                points_soil = np.column_stack([x_soil, y_soil])
                tri_soil = Delaunay(points_soil)

                fig_soil_structure.add_trace(go.Mesh3d(
                    x=x_soil,
                    y=y_soil,
                    z=z_soil,
                    i=tri_soil.simplices[:, 0],
                    j=tri_soil.simplices[:, 1],
                    k=tri_soil.simplices[:, 2],
                    opacity=0.7,
                    color=colors_soil[idx % len(colors_soil)],
                    name=f"{soil_type[:15]}..."
                ))

        fig_soil_structure.update_layout(
            title="Structure 3D Triangulée des Types de Sol Détaillés",
            scene=dict(
                xaxis_title="Position X (m)",
                yaxis_title="Profondeur (m)",
                zaxis_title="qc (MPa)"
            )
        )
        visualizations['soil_structure_3d'] = fig_soil_structure

    except Exception as e:
        st.warning(f"Erreur lors de la structure des sols: {e}")

    # 17. Gradient 3D de propriétés mécaniques
    try:
        fig_gradient_3d = go.Figure()

        # Calculer le gradient de qc
        qc_gradient = np.gradient(df['qc'].values, df['Depth'].values)

        gradient_interp = griddata(
            (df['Depth'], np.random.uniform(0, 100, len(df))),
            qc_gradient,
            (X, Y),
            method='linear'
        )

        fig_gradient_3d.add_trace(go.Mesh3d(
            x=X,
            y=Y,
            z=gradient_interp,
            i=tri.simplices[:, 0],
            j=tri.simplices[:, 1],
            k=tri.simplices[:, 2],
            opacity=0.8,
            colorscale='RdBu',
            intensity=gradient_interp,
            name='Gradient qc'
        ))

        fig_gradient_3d.update_layout(
            title="Gradient 3D Triangulé des Propriétés Mécaniques",
            scene=dict(
                xaxis_title="Profondeur (m)",
                yaxis_title="Position Latérale (m)",
                zaxis_title="Gradient qc (MPa/m)"
            )
        )
        visualizations['gradient_3d'] = fig_gradient_3d

    except Exception as e:
        st.warning(f"Erreur lors du gradient 3D: {e}")

    # 18. Iso-surfaces 3D des paramètres géotechniques
    try:
        fig_isosurface = go.Figure()

        # Créer des isosurfaces pour différentes valeurs de qc
        qc_values = np.linspace(df['qc'].min(), df['qc'].max(), 5)

        for qc_val in qc_values:
            mask = df['qc'] >= qc_val
            if mask.sum() >= 4:  # Assez de points pour triangulation
                iso_data = df[mask]
                x_iso = np.random.uniform(0, 100, len(iso_data))
                y_iso = iso_data['Depth']
                z_iso = np.full(len(iso_data), qc_val)

                if len(iso_data) >= 3:
                    points_iso = np.column_stack([x_iso, y_iso])
                    tri_iso = Delaunay(points_iso)

                    fig_isosurface.add_trace(go.Mesh3d(
                        x=x_iso,
                        y=y_iso,
                        z=z_iso,
                        i=tri_iso.simplices[:, 0],
                        j=tri_iso.simplices[:, 1],
                        k=tri_iso.simplices[:, 2],
                        opacity=0.3,
                        name=f'qc ≥ {qc_val:.1f} MPa'
                    ))

        fig_isosurface.update_layout(
            title="Iso-surfaces 3D Triangulées des Paramètres Géotechniques",
            scene=dict(
                xaxis_title="Position X (m)",
                yaxis_title="Profondeur (m)",
                zaxis_title="qc (MPa)"
            )
        )
        visualizations['isosurface_3d'] = fig_isosurface

    except Exception as e:
        st.warning(f"Erreur lors des isosurfaces: {e}")

    # 19. Réseau 3D interconnecté des zones
    try:
        fig_network_3d = go.Figure()

        # Créer des connexions entre zones proches
        for i in range(len(zone_centers)):
            for j in range(i+1, len(zone_centers)):
                dist = np.sqrt((zone_centers[i][0] - zone_centers[j][0])**2 +
                              (zone_centers[i][1] - zone_centers[j][1])**2)
                if dist < 80:  # Distance maximale pour connexion
                    fig_network_3d.add_trace(go.Scatter3d(
                        x=[zone_centers[i][0], zone_centers[j][0]],
                        y=[zone_centers[i][1], zone_centers[j][1]],
                        z=[0, 0],  # À la surface
                        mode='lines',
                        line=dict(color='gray', width=2),
                        name=f'Connexion {i+1}-{j+1}'
                    ))

        # Ajouter les zones comme points
        for idx, (x, y) in enumerate(zone_centers):
            zone_data = df_zones[df_zones['zone_id'] == idx]
            avg_qc = zone_data['qc'].mean() if not zone_data.empty else 0

            fig_network_3d.add_trace(go.Scatter3d(
                x=[x],
                y=[y],
                z=[0],
                mode='markers+text',
                marker=dict(size=15, color=avg_qc, colorscale='Viridis', showscale=True),
                text=[f'Zone {idx+1}'],
                textposition="top center",
                name=f'Zone {idx+1}'
            ))

        fig_network_3d.update_layout(
            title="Réseau 3D Interconnecté des Zones CPTU",
            scene=dict(
                xaxis_title="Position X (m)",
                yaxis_title="Position Y (m)",
                zaxis_title="Surface"
            )
        )
        visualizations['network_zones_3d'] = fig_network_3d

    except Exception as e:
        st.warning(f"Erreur lors du réseau 3D: {e}")

    # 20. Toiles d'araignée (Spider plots) pour les propriétés par zone
    try:
        fig_spider_zones = go.Figure()

        # Propriétés à analyser
        properties = ['qc', 'fs', 'Ic', 'Fr']
        if 'FS_Liquefaction' in df.columns:
            properties.append('FS_Liquefaction')

        # Normaliser les valeurs pour chaque propriété
        normalized_data = {}
        for prop in properties:
            if prop in df.columns:
                values = df[prop].values
                normalized_data[prop] = (values - values.min()) / (values.max() - values.min())

        # Créer une toile par zone
        for zone_id in range(min(5, n_zones)):  # Maximum 5 toiles pour lisibilité
            zone_data = df_zones[df_zones['zone_id'] == zone_id]

            if not zone_data.empty:
                r_values = []
                for prop in properties:
                    if prop in normalized_data:
                        zone_values = normalized_data[prop][zone_data.index]
                        r_values.append(zone_values.mean())
                    else:
                        r_values.append(0)

                # Ajouter les valeurs de début et fin pour fermer la toile
                r_values.append(r_values[0])
                theta_values = properties + [properties[0]]

                fig_spider_zones.add_trace(go.Scatterpolar(
                    r=r_values,
                    theta=theta_values,
                    fill='toself',
                    name=f'Zone {zone_id + 1}',
                    opacity=0.7
                ))

        fig_spider_zones.update_layout(
            title="Toiles d'Araignée des Propriétés Géotechniques par Zone",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True
        )
        visualizations['spider_zones'] = fig_spider_zones

    except Exception as e:
        st.warning(f"Erreur lors des toiles d'araignée: {e}")

    return visualizations


class GeotechnicalAnalyzer:
    """Classe pour effectuer l'analyse géotechnique complète"""

    def __init__(self):
        self.analysis_methods = {
            'soil_classification': ['Robertson (1986)', 'Schmertmann (1978)'],
            'liquefaction': ['NCEER (1997)', 'Cetin et al. (2004)']
        }

    def analyze_cpt_data(self, df, groundwater_level=2.0, soil_classification_method='Robertson (1986)',
                        liquefaction_method='NCEER (1997)'):
        """Effectue une analyse complète des données CPT"""
        try:
            results = {}

            # Classification des sols
            df_soil = estimate_soil_type(df.copy())
            results['soil_classification'] = df_soil

            # Analyse de liquéfaction
            df_crr = calculate_crr(df_soil.copy())
            results['liquefaction_analysis'] = df_crr

            # Clustering (sans composants Streamlit)
            from models.clustering import perform_clustering
            df_clustered, kmeans, scaler, pca = perform_clustering(df_crr, n_clusters=3)

            # Si le clustering échoue, utiliser les données CRR
            if df_clustered is None:
                df_clustered = df_crr
                models = None
            else:
                models = {
                    'kmeans': kmeans,
                    'scaler': scaler,
                    'pca': pca
                }

            # Métriques générales
            results['dominant_soil_type'] = df_clustered['Soil_Type'].mode().iloc[0] if 'Soil_Type' in df_clustered.columns else 'Unknown'
            results['liquefaction_risk'] = self._assess_liquefaction_risk(df_clustered)
            results['critical_depth'] = df_clustered['Depth'].max()
            results['safety_factor'] = self._calculate_safety_factor(df_clustered)

            # Stocker les données analysées et les modèles
            results['analyzed_data'] = df_clustered
            results['models'] = models

            return results

        except Exception as e:
            raise ValueError(f"Erreur lors de l'analyse géotechnique: {str(e)}")

    def _assess_liquefaction_risk(self, df):
        """Évalue le risque de liquéfaction global"""
        if 'CRR' not in df.columns:
            return 'Non évalué'

        crr_values = df['CRR'].dropna()
        if len(crr_values) == 0:
            return 'Non évalué'

        avg_crr = crr_values.mean()
        if avg_crr < 0.1:
            return 'Élevé'
        elif avg_crr < 0.3:
            return 'Modéré'
        else:
            return 'Faible'

    def _calculate_safety_factor(self, df):
        """Calcule le facteur de sécurité moyen"""
        if 'CRR' not in df.columns:
            return 0.0

        crr_values = df['CRR'].dropna()
        if len(crr_values) == 0:
            return 0.0

        return crr_values.mean()

def create_correlation_matrix(df):
    """Crée un tableau de corrélation complet entre toutes les propriétés géotechniques"""
    try:
        # Sélectionner les colonnes numériques pertinentes pour la corrélation
        numeric_columns = []
        correlation_columns = []

        # Colonnes de base
        base_columns = ['Depth', 'qc', 'fs', 'Ic']
        for col in base_columns:
            if col in df.columns:
                numeric_columns.append(col)
                correlation_columns.append(col)

        # Colonnes dérivées de l'analyse
        derived_columns = ['CRR', 'FS_Liquefaction', 'qc_smooth', 'fs_smooth', 'Ic_smooth',
                          'Friction_Ratio', 'Soil_Density', 'Young_Modulus', 'qc_gradient',
                          'cluster_distance', 'pca_1', 'pca_2']

        for col in derived_columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                numeric_columns.append(col)
                correlation_columns.append(col)

        if len(numeric_columns) < 2:
            st.warning("Pas assez de colonnes numériques pour calculer la corrélation")
            return None

        # Calculer la matrice de corrélation
        correlation_matrix = df[numeric_columns].corr()

        # Créer une figure Plotly pour la matrice de corrélation
        fig_correlation = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(correlation_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig_correlation.update_layout(
            title="Matrice de Corrélation Complète des Propriétés Géotechniques",
            xaxis_title="Propriétés",
            yaxis_title="Propriétés",
            width=800,
            height=800,
            xaxis=dict(tickangle=-45),
            yaxis=dict(tickangle=0)
        )

        # Créer aussi un tableau stylisé avec les valeurs numériques
        correlation_table = correlation_matrix.round(3)

        # Ajouter des annotations pour interpréter les corrélations
        annotations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                if i != j:  # Ne pas annoter la diagonale
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        strength = "Forte"
                        color = "red" if corr_value > 0 else "blue"
                    elif abs(corr_value) > 0.5:
                        strength = "Modérée"
                        color = "orange" if corr_value > 0 else "cyan"
                    elif abs(corr_value) > 0.3:
                        strength = "Faible"
                        color = "yellow" if corr_value > 0 else "lightblue"
                    else:
                        strength = "Très faible"
                        color = "white"

                    annotations.append({
                        'x': correlation_matrix.columns[j],
                        'y': correlation_matrix.columns[i],
                        'text': f"{strength}<br>({corr_value:.2f})",
                        'showarrow': False,
                        'font': {'size': 8, 'color': 'black'},
                        'bgcolor': color,
                        'opacity': 0.8
                    })

        # Créer une version annotée de la heatmap
        fig_correlation_annotated = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(correlation_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))

        # Ajouter les annotations
        for annotation in annotations:
            fig_correlation_annotated.add_annotation(
                x=annotation['x'],
                y=annotation['y'],
                text=annotation['text'],
                showarrow=annotation['showarrow'],
                font=annotation['font'],
                bgcolor=annotation['bgcolor'],
                opacity=annotation['opacity']
            )

        fig_correlation_annotated.update_layout(
            title="Matrice de Corrélation Annotée des Propriétés Géotechniques",
            xaxis_title="Propriétés",
            yaxis_title="Propriétés",
            width=900,
            height=900,
            xaxis=dict(tickangle=-45),
            yaxis=dict(tickangle=0)
        )

        # Statistiques descriptives des corrélations
        corr_stats = {
            'max_correlation': correlation_matrix.max().max(),
            'min_correlation': correlation_matrix.min().min(),
            'strong_positive_corr': len(correlation_matrix[(correlation_matrix > 0.7) & (correlation_matrix < 1.0)].stack()),
            'strong_negative_corr': len(correlation_matrix[(correlation_matrix < -0.7)].stack()),
            'columns_analyzed': len(numeric_columns)
        }

        return {
            'correlation_matrix': correlation_table,
            'correlation_heatmap': fig_correlation,
            'correlation_annotated': fig_correlation_annotated,
            'correlation_stats': corr_stats,
            'analyzed_columns': numeric_columns
        }

    except Exception as e:
        st.warning(f"Erreur lors de la création de la matrice de corrélation: {e}")
        return None


def perform_complete_analysis(df, n_clusters=3, use_streamlit=True):
    """Effectue une analyse complète et avancée CPTU avec 3D et géolocalisation"""
    try:
        if use_streamlit:
            st.info("🔄 Analyse complète avancée en cours...")

        # Étape 1: Classification détaillée des sols
        if use_streamlit:
            with st.spinner("🌱 Étape 1/5: Classification détaillée des sols..."):
                df_analyzed = estimate_soil_type(df)
                if df_analyzed is None:
                    raise ValueError("Échec de la classification des sols")
                progress_bar = st.progress(20)
                st.success("✅ Classification détaillée des sols terminée!")
        else:
            df_analyzed = estimate_soil_type(df)
            if df_analyzed is None:
                raise ValueError("Échec de la classification des sols")


        # Étape 2: Calcul avancé du CRR et liquéfaction
        if use_streamlit:
            with st.spinner("🌊 Étape 2/5: Analyse de liquéfaction avancée..."):
                df_crr = calculate_crr(df_analyzed)
                if df_crr is None:
                    raise ValueError("Échec du calcul du CRR")
                progress_bar.progress(40)
                st.success("✅ Analyse de liquéfaction terminée!")
        else:
            df_crr = calculate_crr(df_analyzed)
            if df_crr is None:
                raise ValueError("Échec du calcul du CRR")

        # Étape 3: Identification des couches 3D
        if use_streamlit:
            with st.spinner("🏔️ Étape 3/5: Identification des couches géologiques 3D..."):
                layers_df = identify_soil_layers_3d(df_crr)
                progress_bar.progress(60)
                st.success(f"✅ {len(layers_df)} couches géologiques identifiées!")
        else:
            layers_df = identify_soil_layers_3d(df_crr)

        # Étape 4: Analyse géospatiale
        if use_streamlit:
            with st.spinner("🌍 Étape 4/5: Géolocalisation des points..."):
                gdf = create_geospatial_analysis(df_crr)
                progress_bar.progress(80)
                if gdf is not None:
                    st.success("✅ Géolocalisation terminée!")
                else:
                    st.warning("⚠️ Géolocalisation limitée (GeoPandas non disponible)")
        else:
            gdf = create_geospatial_analysis(df_crr)

        # Étape 5: Clustering avancé
        if use_streamlit:
            with st.spinner("🎯 Étape 5/5: Clustering automatique avancé..."):
                from models.clustering import perform_clustering
                df_clustered, kmeans, scaler, pca = perform_clustering(df_crr, n_clusters)
                if df_clustered is None:
                    raise ValueError("Échec du clustering")
                progress_bar.progress(100)
                st.success("✅ Clustering avancé terminé!")
        else:
            from models.clustering import perform_clustering
            df_clustered, kmeans, scaler, pca = perform_clustering(df_crr, n_clusters)
            if df_clustered is None:
                raise ValueError("Échec du clustering")

        # Création des visualisations avancées
        if use_streamlit:
            with st.spinner("📊 Génération des graphiques avancés..."):
                visualizations = create_advanced_visualizations(df_clustered, layers_df, gdf)
        else:
            visualizations = create_advanced_visualizations(df_clustered, layers_df, gdf)

        # Création du tableau de corrélation complet
        if use_streamlit:
            with st.spinner("📈 Calcul du tableau de corrélation complet..."):
                correlation_results = create_correlation_matrix(df_clustered)
                if correlation_results:
                    st.success("✅ Tableau de corrélation généré!")
                else:
                    st.warning("⚠️ Impossible de générer le tableau de corrélation")
        else:
            correlation_results = create_correlation_matrix(df_clustered)

        # Sauvegarder les modèles et résultats
        models = {
            'kmeans': kmeans,
            'scaler': scaler,
            'pca': pca
        }

        results = {
            'data': df_clustered,
            'layers': layers_df,
            'geospatial': gdf,
            'models': models,
            'visualizations': visualizations,
            'correlation_analysis': correlation_results
        }

        if use_streamlit:
            progress_bar.empty()
            st.success("🎉 Analyse complète avancée terminée avec succès!")
            st.info(f"📊 {len(visualizations)} graphiques avancés générés")

        return df_clustered, models, results

    except Exception as e:
        if use_streamlit:
            st.error(f"❌ Erreur lors de l'analyse complète: {str(e)}")
            st.error("Retour aux données brutes...")
        return df, None, None