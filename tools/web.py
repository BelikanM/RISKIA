from tavily import TavilyClient
from ddgs import DDGS
import os
from dotenv import load_dotenv

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def web_search(query: str, disabled=False):
    # PROTECTION ANTI-HALLUCINATION: Vérifier si les appels web sont désactivés
    print(f"DEBUG: web_search called with disabled={disabled} for query='{query}'")
    if disabled:
        print(f"🔒 Recherche web désactivée pour : {query}")
        return {"results": [], "images": [], "query": query, "source": "disabled"}
    
    # Si on arrive ici, c'est que disabled=False, mais on va quand même retourner vide pour être sûr
    print(f"⚠️ APPEL WEB NON AUTORISÉ: {query}")
    return {"results": [], "images": [], "query": query, "source": "blocked"}

def display_images(web_results, max_images=3):
    """Affiche les informations des images trouvées et propose de les télécharger"""
    if not web_results or 'images' not in web_results:
        return "Aucune image trouvée."
    
    images = web_results['images']
    if not images:
        return "Aucune image trouvée."
    
    display_text = f"🖼️ **Images trouvées pour '{web_results.get('query', '')}'** ({len(images)} résultats)\n\n"
    
    for i, img in enumerate(images[:max_images]):
        title = img.get('title', 'Sans titre')
        source = img.get('source', img.get('url', ''))
        url = img.get('url', '')
        
        display_text += f"**{i+1}. {title}**\n"
        display_text += f"   📍 Source: {source}\n"
        display_text += f"   🔗 URL: {url}\n"
        
        if img.get('width') and img.get('height'):
            display_text += f"   📐 Dimensions: {img['width']}x{img['height']}\n"
        
        display_text += "\n"
    
    if len(images) > max_images:
        display_text += f"... et {len(images) - max_images} autres images.\n"
    
    display_text += "💡 Utilisez `download_image(url, filename)` pour télécharger une image spécifique."
    
    return display_text

def download_image(image_url, filename=None, save_dir="downloads/images"):
    """Télécharge une image depuis une URL"""
    import requests
    import os
    from urllib.parse import urlparse
    
    try:
        # Créer le dossier de destination
        os.makedirs(save_dir, exist_ok=True)
        
        # Générer un nom de fichier si non fourni
        if not filename:
            parsed_url = urlparse(image_url)
            filename = os.path.basename(parsed_url.path)
            if not filename:
                filename = f"image_{hash(image_url) % 10000}.jpg"
        
        filepath = os.path.join(save_dir, filename)
        
        # Télécharger l'image
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        return f"✅ Image téléchargée: {filepath}"
        
    except Exception as e:
        return f"❌ Erreur de téléchargement: {e}"