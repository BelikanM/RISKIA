"""
Module de simulations et probabilités avancées pour l'analyse de risques
Implémente Monte Carlo, propagation de trajectoires, zones d'impact
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import math

class AdvancedRiskSimulator:
    """Simulateur de risques avancé avec probabilités et trajectoires"""
    
    def __init__(self, image_width: int, image_height: int):
        self.width = image_width
        self.height = image_height
        
    def monte_carlo_fire_propagation(
        self,
        ignition_point: Tuple[float, float],
        wind_direction: float = 0,  # En degrés (0=Nord, 90=Est)
        wind_speed: float = 10,  # km/h
        fuel_density: float = 0.5,  # 0-1
        num_simulations: int = 1000
    ) -> Dict:
        """
        Simulation Monte Carlo de propagation d'incendie
        
        Returns:
            Dict avec zones de probabilité, trajectoires, temps d'arrivée
        """
        print(f"🔥 Simulation Monte Carlo incendie ({num_simulations} itérations)...")
        
        results = {
            'probability_map': np.zeros((self.height, self.width)),
            'trajectories': [],
            'mean_arrival_time': {},
            'confidence_zones': []
        }
        
        # Convertir vent en radians
        wind_rad = math.radians(wind_direction)
        wind_vector = (wind_speed * math.cos(wind_rad), wind_speed * math.sin(wind_rad))
        
        for sim in range(num_simulations):
            # Paramètres aléatoires pour cette simulation
            sim_wind_var = np.random.normal(0, wind_speed * 0.2, 2)
            sim_fuel = fuel_density * np.random.uniform(0.8, 1.2)
            sim_spread_rate = 1.0 + sim_fuel * 0.5  # m/s base
            
            # Propagation par pas de temps
            current_pos = list(ignition_point)
            trajectory = [current_pos.copy()]
            time_step = 0.1  # secondes
            max_time = 300  # 5 minutes
            
            for t in np.arange(0, max_time, time_step):
                # Direction influencée par le vent + variation aléatoire
                direction = wind_rad + np.random.normal(0, 0.3)
                
                # Vitesse de propagation avec variation
                speed = sim_spread_rate * (1 + wind_speed / 20) * np.random.uniform(0.9, 1.1)
                
                # Nouvelle position
                dx = speed * math.cos(direction) * time_step
                dy = speed * math.sin(direction) * time_step
                
                current_pos[0] += dx
                current_pos[1] += dy
                
                # Vérifier limites
                if 0 <= current_pos[0] < self.width and 0 <= current_pos[1] < self.height:
                    trajectory.append(current_pos.copy())
                    
                    # Marquer sur la carte de probabilité
                    x, y = int(current_pos[0]), int(current_pos[1])
                    if 0 <= x < self.width and 0 <= y < self.height:
                        results['probability_map'][y, x] += 1
                        
                        # Enregistrer temps d'arrivée moyen
                        pos_key = (x, y)
                        if pos_key not in results['mean_arrival_time']:
                            results['mean_arrival_time'][pos_key] = []
                        results['mean_arrival_time'][pos_key].append(t)
                else:
                    break
            
            if len(trajectory) > 10:  # Trajectoires significatives seulement
                results['trajectories'].append(trajectory)
        
        # Normaliser la carte de probabilité
        if results['probability_map'].max() > 0:
            results['probability_map'] = results['probability_map'] / results['probability_map'].max()
        
        # Calculer temps d'arrivée moyen par position
        for pos, times in results['mean_arrival_time'].items():
            results['mean_arrival_time'][pos] = np.mean(times)
        
        # Créer zones de confiance (50%, 90%, 99%)
        prob_map = results['probability_map']
        results['confidence_zones'] = [
            {'level': 0.99, 'mask': prob_map > 0.01},   # Zone 99%
            {'level': 0.90, 'mask': prob_map > 0.10},   # Zone 90%
            {'level': 0.50, 'mask': prob_map > 0.50}    # Zone 50%
        ]
        
        print(f"✅ {len(results['trajectories'])} trajectoires simulées")
        return results
    
    def chemical_dispersion_simulation(
        self,
        source_point: Tuple[float, float],
        release_rate: float = 1.0,  # kg/s
        wind_direction: float = 0,
        wind_speed: float = 10,
        stability_class: str = 'D',  # Classes Pasquill A-F
        duration: float = 600  # secondes
    ) -> Dict:
        """
        Simulation de dispersion de nuage chimique (modèle Gaussien simplifié)
        
        Returns:
            Dict avec concentrations, zones IDLH, zones d'évacuation
        """
        print(f"☁️ Simulation dispersion chimique (Modèle Gaussien, classe {stability_class})...")
        
        # Coefficients de dispersion selon classe de stabilité
        dispersion_coeffs = {
            'A': (0.22, 0.20),  # Très instable
            'B': (0.16, 0.12),  # Instable
            'C': (0.11, 0.08),  # Légèrement instable
            'D': (0.08, 0.06),  # Neutre
            'E': (0.06, 0.03),  # Stable
            'F': (0.04, 0.016)  # Très stable
        }
        
        sigma_y_coeff, sigma_z_coeff = dispersion_coeffs.get(stability_class, (0.08, 0.06))
        
        # Convertir vent
        wind_rad = math.radians(wind_direction)
        
        # Créer grille de concentration
        concentration_map = np.zeros((self.height, self.width))
        
        # Calculer dispersion
        x_src, y_src = source_point
        
        for x in range(self.width):
            for y in range(self.height):
                # Distance downwind
                dx = (x - x_src) * math.cos(wind_rad) + (y - y_src) * math.sin(wind_rad)
                
                if dx > 0:  # Seulement downwind
                    # Distance crosswind
                    dy = -(x - x_src) * math.sin(wind_rad) + (y - y_src) * math.cos(wind_rad)
                    
                    # Écarts-types de dispersion
                    sigma_y = sigma_y_coeff * dx ** 0.894
                    sigma_z = sigma_z_coeff * dx ** 0.894
                    
                    # Concentration (modèle Gaussien)
                    Q = release_rate
                    u = wind_speed / 3.6  # Convertir en m/s
                    
                    if u > 0 and sigma_y > 0 and sigma_z > 0:
                        C = (Q / (2 * math.pi * u * sigma_y * sigma_z)) * \
                            math.exp(-0.5 * (dy / sigma_y) ** 2) * \
                            math.exp(-0.5 * (0 / sigma_z) ** 2)  # z=0 (niveau sol)
                        
                        concentration_map[y, x] = C
        
        # Normaliser pour visualisation
        if concentration_map.max() > 0:
            concentration_map_normalized = concentration_map / concentration_map.max()
        else:
            concentration_map_normalized = concentration_map
        
        # Définir zones de danger
        results = {
            'concentration_map': concentration_map,
            'concentration_normalized': concentration_map_normalized,
            'danger_zones': [
                {'level': 'IDLH', 'threshold': 0.5, 'mask': concentration_map_normalized > 0.5},
                {'level': 'Évacuation', 'threshold': 0.2, 'mask': concentration_map_normalized > 0.2},
                {'level': 'Alerte', 'threshold': 0.05, 'mask': concentration_map_normalized > 0.05}
            ],
            'max_concentration': concentration_map.max(),
            'affected_area_sqm': np.count_nonzero(concentration_map_normalized > 0.05)
        }
        
        print(f"✅ Zone affectée: {results['affected_area_sqm']} pixels")
        return results
    
    def explosion_blast_zones(
        self,
        epicenter: Tuple[float, float],
        tnt_equivalent: float = 100,  # kg TNT
        obstacles: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Calcul des zones d'effet d'explosion
        
        Returns:
            Dict avec zones de surpression, projectiles, dommages
        """
        print(f"💥 Simulation explosion ({tnt_equivalent} kg TNT équivalent)...")
        
        # Rayons d'effet selon surpression (en mètres)
        # Formules de Kinney-Graham simplifiées
        scale_factor = tnt_equivalent ** (1/3)
        
        blast_zones = [
            {
                'name': 'Destruction totale',
                'overpressure_psi': 20,  # > 140 kPa
                'radius_m': 4.8 * scale_factor,
                'effects': 'Bâtiments détruits, blessures graves/mortelles',
                'color': (255, 0, 0, 180)
            },
            {
                'name': 'Dommages structurels graves',
                'overpressure_psi': 10,  # 70 kPa
                'radius_m': 7.5 * scale_factor,
                'effects': 'Effondrement murs, toitures arrachées',
                'color': (255, 100, 0, 150)
            },
            {
                'name': 'Dommages modérés',
                'overpressure_psi': 5,  # 35 kPa
                'radius_m': 11 * scale_factor,
                'effects': 'Vitres brisées, portes arrachées',
                'color': (255, 200, 0, 120)
            },
            {
                'name': 'Dommages légers',
                'overpressure_psi': 2,  # 14 kPa
                'radius_m': 17 * scale_factor,
                'effects': 'Vitres fissurées, dommages mineurs',
                'color': (255, 255, 0, 100)
            }
        ]
        
        # Convertir en pixels (supposer 1 pixel = 1 mètre)
        x_center, y_center = epicenter
        zones_pixel = []
        
        for zone in blast_zones:
            radius_px = int(zone['radius_m'])
            zones_pixel.append({
                **zone,
                'center': epicenter,
                'radius_px': radius_px,
                'bbox': [
                    int(x_center - radius_px),
                    int(y_center - radius_px),
                    int(x_center + radius_px),
                    int(y_center + radius_px)
                ]
            })
        
        # Simulation de trajectoires de projectiles
        projectiles = []
        num_projectiles = 50
        
        for i in range(num_projectiles):
            angle = (i / num_projectiles) * 2 * math.pi
            velocity = np.random.uniform(20, 100)  # m/s
            
            # Trajectoire balistique simplifiée
            vx = velocity * math.cos(angle)
            vy = velocity * math.sin(angle)
            
            max_range = (velocity ** 2) / 9.81  # Portée maximale
            
            projectiles.append({
                'start': epicenter,
                'angle': angle,
                'velocity': velocity,
                'max_range': max_range,
                'end': (
                    x_center + max_range * math.cos(angle),
                    y_center + max_range * math.sin(angle)
                )
            })
        
        results = {
            'blast_zones': zones_pixel,
            'projectiles': projectiles[:20],  # Top 20 pour visualisation
            'tnt_equivalent': tnt_equivalent,
            'max_radius_m': blast_zones[-1]['radius_m']
        }
        
        print(f"✅ {len(blast_zones)} zones d'effet calculées, rayon max: {results['max_radius_m']:.1f}m")
        return results
    
    def flood_simulation(
        self,
        water_sources: List[Tuple[float, float]],
        terrain_elevation: Optional[np.ndarray] = None,
        water_level_rise: float = 2.0  # mètres
    ) -> Dict:
        """
        Simulation d'inondation par montée des eaux
        
        Returns:
            Dict avec zones inondables, profondeurs, vitesses
        """
        print(f"🌊 Simulation inondation (montée: {water_level_rise}m)...")
        
        if terrain_elevation is None:
            # Générer terrain synthétique si non fourni
            terrain_elevation = np.random.rand(self.height, self.width) * 5
        
        # Simuler montée des eaux
        flooded_mask = terrain_elevation < water_level_rise
        water_depth = np.maximum(0, water_level_rise - terrain_elevation)
        
        # Vitesse d'écoulement (formule Manning simplifiée)
        velocity_map = np.zeros_like(water_depth)
        gradient_y, gradient_x = np.gradient(terrain_elevation)
        slope = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Vitesse proportionnelle à profondeur et pente
        velocity_map = water_depth ** 0.5 * slope * 5  # m/s
        
        # Zones de danger selon profondeur
        danger_zones = [
            {'level': 'Très dangereux', 'depth_min': 1.5, 'mask': water_depth > 1.5},
            {'level': 'Dangereux', 'depth_min': 0.5, 'mask': (water_depth > 0.5) & (water_depth <= 1.5)},
            {'level': 'Risque modéré', 'depth_min': 0.1, 'mask': (water_depth > 0.1) & (water_depth <= 0.5)}
        ]
        
        results = {
            'flooded_mask': flooded_mask,
            'water_depth': water_depth,
            'velocity_map': velocity_map,
            'danger_zones': danger_zones,
            'flooded_area_sqm': np.count_nonzero(flooded_mask),
            'max_depth': water_depth.max(),
            'max_velocity': velocity_map.max()
        }
        
        print(f"✅ Surface inondée: {results['flooded_area_sqm']} pixels, profondeur max: {results['max_depth']:.2f}m")
        return results
    
    def calculate_bayesian_risk(
        self,
        prior_probability: float,
        likelihood_given_evidence: float,
        evidence_probability: float
    ) -> float:
        """
        Calcul de probabilité bayésienne mise à jour
        P(Risque|Evidence) = P(Evidence|Risque) * P(Risque) / P(Evidence)
        """
        posterior = (likelihood_given_evidence * prior_probability) / evidence_probability
        return min(1.0, max(0.0, posterior))
    
    def failure_probability_weibull(
        self,
        time: float,
        shape: float = 2.0,
        scale: float = 100
    ) -> float:
        """
        Probabilité de défaillance selon distribution de Weibull
        Utilisé pour équipements vieillissants
        """
        return 1 - math.exp(-(time / scale) ** shape)
