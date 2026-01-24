# tools/geo.py
import requests
import logging

def get_geo_context():
    """
    Récupère la géolocalisation par IP (Fallback gratuit et illimité).
    """
    try:
        # Utilisation de ip-api.com (Gratuit, fiable, pas de clé requise)
        response = requests.get('http://ip-api.com/json/', timeout=5)
        data = response.json()
        
        if data.get('status') == 'success':
            return {
                "ip": data.get('query'),
                "country": data.get('country', 'Gabon'),
                "city": data.get('city', 'Libreville'),
                "latitude": float(data.get('lat', 0.3908)),
                "longitude": float(data.get('lon', 9.4544)),
                "timezone": data.get('timezone', 'Africa/Libreville'),
                "method": "IP (Approximatif)",
                "lang": "Français"
            }
        else:
            raise Exception("Erreur API ip-api")
            
    except Exception as e:
        logging.warning(f"Échec IP Geo: {e}")
        return {
            "ip": "127.0.0.1", "country": "Gabon", "city": "Libreville",
            "latitude": 0.3908, "longitude": 9.4544, "timezone": "Africa/Libreville",
            "method": "Valeurs par défaut", "lang": "Français"
        }