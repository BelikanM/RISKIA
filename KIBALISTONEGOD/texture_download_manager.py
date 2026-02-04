"""
Module de Téléchargement et Gestion Automatique de Textures PBR
Télécharge, stocke et injecte automatiquement les textures PBR
Développé par NYUNDU FRANCIS ARNAUD pour SETRAF GABON
"""

import sqlite3
import requests
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple
import hashlib
from PIL import Image
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Sources de textures PBR gratuites avec API
TEXTURE_SOURCES = {
    "polyhaven": {
        "name": "Poly Haven",
        "api_base": "https://api.polyhaven.com",
        "download_base": "https://dl.polyhaven.com",
        "license": "CC0"
    },
    "ambientcg": {
        "name": "ambientCG",
        "search_base": "https://ambientcg.com/api/v2/full_json",
        "license": "CC0"
    }
}


class TextureDownloadManager:
    """
    Gestionnaire de téléchargement et stockage automatique de textures PBR
    """
    
    def __init__(self, storage_path: str = "./texture_library"):
        """
        Initialise le gestionnaire de textures
        
        Args:
            storage_path: Chemin de stockage des textures
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.storage_path / "textures.db"
        self._init_database()
        
        print(f"📦 TextureDownloadManager initialisé : {self.storage_path}")
    
    def _init_database(self):
        """Initialise la base de données SQLite"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Table des textures
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS textures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                material_type TEXT NOT NULL,
                source TEXT NOT NULL,
                resolution TEXT,
                license TEXT,
                download_url TEXT,
                local_path TEXT,
                thumbnail_path TEXT,
                file_size INTEGER,
                downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                hash TEXT UNIQUE
            )
        """)
        
        # Table des maps PBR par texture
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS texture_maps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                texture_id INTEGER,
                map_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER,
                FOREIGN KEY (texture_id) REFERENCES textures (id)
            )
        """)
        
        # Index pour recherche rapide
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_material_type ON textures(material_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON textures(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_map_type ON texture_maps(map_type)")
        
        conn.commit()
        conn.close()
        
        print("✅ Base de données SQLite initialisée")
    
    def search_polyhaven_textures(self, material_keywords: List[str], 
                                  resolution: str = "2k") -> List[Dict]:
        """
        Recherche des textures sur Poly Haven
        
        Args:
            material_keywords: Mots-clés (concrete, metal, wood, etc.)
            resolution: Résolution (1k, 2k, 4k, 8k)
            
        Returns:
            Liste de textures disponibles
        """
        results = []
        
        try:
            # API Poly Haven pour lister les textures
            response = requests.get(f"{TEXTURE_SOURCES['polyhaven']['api_base']}/assets?t=textures", 
                                  timeout=10)
            
            if response.status_code == 200:
                all_textures = response.json()
                
                # Filtrer par mots-clés
                for tex_id, tex_info in all_textures.items():
                    tex_name = tex_info.get('name', '').lower()
                    
                    # Vérifier si un mot-clé correspond
                    if any(keyword.lower() in tex_name for keyword in material_keywords):
                        results.append({
                            "id": tex_id,
                            "name": tex_info.get('name', tex_id),
                            "source": "polyhaven",
                            "categories": tex_info.get('categories', []),
                            "resolution": resolution,
                            "thumbnail": f"https://cdn.polyhaven.com/asset_img/thumbs/{tex_id}.png"
                        })
        
        except Exception as e:
            print(f"⚠️ Erreur recherche Poly Haven: {e}")
        
        return results
    
    def get_polyhaven_download_links(self, texture_id: str, 
                                    resolution: str = "2k") -> Dict[str, str]:
        """
        Obtient les liens de téléchargement Poly Haven pour toutes les maps
        
        Args:
            texture_id: ID de la texture
            resolution: Résolution (1k, 2k, 4k)
            
        Returns:
            Dict avec liens pour chaque map (albedo, normal, roughness, etc.)
        """
        download_links = {}
        
        try:
            # API pour obtenir les fichiers disponibles
            response = requests.get(
                f"{TEXTURE_SOURCES['polyhaven']['api_base']}/files/{texture_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                files = response.json()
                
                # Trouver les maps PBR
                for file_type, resolutions in files.items():
                    if resolution in resolutions:
                        res_data = resolutions[resolution]
                        
                        # Mapper les types de fichiers
                        map_type = None
                        if "diff" in file_type or "col" in file_type:
                            map_type = "albedo"
                        elif "nor" in file_type or "nrm" in file_type:
                            map_type = "normal"
                        elif "rough" in file_type:
                            map_type = "roughness"
                        elif "disp" in file_type:
                            map_type = "displacement"
                        elif "ao" in file_type or "arm" in file_type:
                            map_type = "ao"
                        elif "metal" in file_type:
                            map_type = "metallic"
                        
                        if map_type and "url" in res_data:
                            download_links[map_type] = res_data["url"]
        
        except Exception as e:
            print(f"⚠️ Erreur obtention liens: {e}")
        
        return download_links
    
    def download_texture(self, texture_info: Dict, resolution: str = "2k") -> Optional[int]:
        """
        Télécharge une texture complète (toutes les maps)
        
        Args:
            texture_info: Infos de la texture
            resolution: Résolution à télécharger
            
        Returns:
            ID de la texture dans la base, ou None si échec
        """
        texture_id = texture_info["id"]
        texture_name = texture_info["name"]
        source = texture_info["source"]
        
        print(f"📥 Téléchargement : {texture_name} ({resolution})")
        
        # Créer dossier pour cette texture
        texture_folder = self.storage_path / source / texture_id
        texture_folder.mkdir(parents=True, exist_ok=True)
        
        # Obtenir les liens de téléchargement
        if source == "polyhaven":
            download_links = self.get_polyhaven_download_links(texture_id, resolution)
        else:
            print(f"⚠️ Source {source} non supportée")
            return None
        
        if not download_links:
            print(f"❌ Aucune map trouvée pour {texture_name}")
            return None
        
        # Télécharger toutes les maps
        downloaded_maps = {}
        total_size = 0
        
        for map_type, url in download_links.items():
            try:
                print(f"  ⬇️ {map_type}...")
                response = requests.get(url, timeout=30, stream=True)
                
                if response.status_code == 200:
                    # Déterminer extension
                    ext = url.split('.')[-1].split('?')[0]
                    if ext not in ['jpg', 'jpeg', 'png', 'exr']:
                        ext = 'jpg'
                    
                    file_path = texture_folder / f"{map_type}.{ext}"
                    
                    # Sauvegarder
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    downloaded_maps[map_type] = str(file_path)
                    
                    print(f"    ✅ {map_type} : {file_size / 1024:.1f} KB")
                
            except Exception as e:
                print(f"    ⚠️ Erreur {map_type}: {e}")
        
        if not downloaded_maps:
            print(f"❌ Échec téléchargement {texture_name}")
            return None
        
        # Télécharger thumbnail
        thumbnail_path = None
        if "thumbnail" in texture_info:
            try:
                response = requests.get(texture_info["thumbnail"], timeout=10)
                if response.status_code == 200:
                    thumbnail_path = texture_folder / "thumbnail.png"
                    with open(thumbnail_path, 'wb') as f:
                        f.write(response.content)
            except:
                pass
        
        # Enregistrer dans la base
        texture_db_id = self._save_to_database(
            name=texture_name,
            material_type=texture_info.get("categories", ["unknown"])[0] if texture_info.get("categories") else "unknown",
            source=source,
            resolution=resolution,
            license=TEXTURE_SOURCES[source]["license"],
            download_url=texture_info.get("thumbnail", ""),
            local_path=str(texture_folder),
            thumbnail_path=str(thumbnail_path) if thumbnail_path else None,
            file_size=total_size,
            metadata=json.dumps(texture_info),
            hash=hashlib.md5(f"{source}_{texture_id}_{resolution}".encode()).hexdigest()
        )
        
        # Enregistrer les maps
        for map_type, file_path in downloaded_maps.items():
            self._save_map_to_database(texture_db_id, map_type, file_path)
        
        print(f"✅ Texture {texture_name} téléchargée (ID: {texture_db_id})")
        return texture_db_id
    
    def _save_to_database(self, **kwargs) -> int:
        """Sauvegarde une texture dans la base"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO textures (name, material_type, source, resolution, license,
                                download_url, local_path, thumbnail_path, file_size, 
                                metadata, hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            kwargs['name'], kwargs['material_type'], kwargs['source'], 
            kwargs['resolution'], kwargs['license'], kwargs['download_url'],
            kwargs['local_path'], kwargs['thumbnail_path'], kwargs['file_size'],
            kwargs['metadata'], kwargs['hash']
        ))
        
        texture_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # cursor.lastrowid peut retourner None si l'insertion échoue
        # mais dans notre cas, si on arrive ici, l'insertion a réussi
        return int(texture_id) if texture_id is not None else 0
    
    def _save_map_to_database(self, texture_id: int, map_type: str, file_path: str):
        """Enregistre une map PBR dans la base"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        file_size = Path(file_path).stat().st_size
        
        cursor.execute("""
            INSERT INTO texture_maps (texture_id, map_type, file_path, file_size)
            VALUES (?, ?, ?, ?)
        """, (texture_id, map_type, file_path, file_size))
        
        conn.commit()
        conn.close()
    
    def search_local_textures(self, material_type: Optional[str] = None,
                             resolution: Optional[str] = None) -> List[Dict]:
        """
        Recherche dans les textures téléchargées localement
        
        Args:
            material_type: Filtrer par type (concrete, metal, etc.)
            resolution: Filtrer par résolution
            
        Returns:
            Liste des textures correspondantes
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        query = "SELECT * FROM textures WHERE 1=1"
        params = []
        
        if material_type:
            query += " AND material_type = ?"
            params.append(material_type)
        
        if resolution:
            query += " AND resolution = ?"
            params.append(resolution)
        
        query += " ORDER BY downloaded_at DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            texture = {
                "id": row[0],
                "name": row[1],
                "material_type": row[2],
                "source": row[3],
                "resolution": row[4],
                "license": row[5],
                "local_path": row[7],
                "thumbnail_path": row[8],
                "file_size": row[9],
                "downloaded_at": row[10]
            }
            
            # Récupérer les maps associées
            cursor.execute("SELECT map_type, file_path FROM texture_maps WHERE texture_id = ?", (row[0],))
            maps = cursor.fetchall()
            texture["maps"] = {map_type: file_path for map_type, file_path in maps}
            
            results.append(texture)
        
        conn.close()
        return results
    
    def batch_download(self, material_keywords: List[str], 
                      max_textures: int = 5,
                      resolution: str = "2k") -> List[int]:
        """
        Télécharge plusieurs textures en parallèle
        
        Args:
            material_keywords: Mots-clés de recherche
            max_textures: Nombre max de textures à télécharger
            resolution: Résolution
            
        Returns:
            Liste des IDs téléchargés
        """
        print(f"🔍 Recherche de textures pour : {', '.join(material_keywords)}")
        
        # Rechercher sur Poly Haven
        available = self.search_polyhaven_textures(material_keywords, resolution)
        
        if not available:
            print("❌ Aucune texture trouvée")
            return []
        
        # Limiter le nombre
        to_download = available[:max_textures]
        print(f"📦 {len(to_download)} textures trouvées, téléchargement...")
        
        # Téléchargement parallèle
        downloaded_ids = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(self.download_texture, tex, resolution): tex 
                      for tex in to_download}
            
            for future in as_completed(futures):
                texture_id = future.result()
                if texture_id:
                    downloaded_ids.append(texture_id)
        
        print(f"✅ {len(downloaded_ids)} textures téléchargées avec succès")
        return downloaded_ids
    
    def get_texture_for_injection(self, material_type: str) -> Optional[Dict]:
        """
        Récupère la meilleure texture disponible pour injection automatique
        
        Args:
            material_type: Type de matériau
            
        Returns:
            Dict avec chemins des maps PBR
        """
        textures = self.search_local_textures(material_type=material_type)
        
        if not textures:
            return None
        
        # Prendre la première (plus récente)
        best = textures[0]
        
        return {
            "name": best["name"],
            "maps": best["maps"],
            "local_path": best["local_path"]
        }
    
    def get_library_stats(self) -> Dict:
        """Statistiques de la bibliothèque locale"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM textures")
        total_textures = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(file_size) FROM textures")
        total_size = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT material_type, COUNT(*) FROM textures GROUP BY material_type")
        by_material = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            "total_textures": total_textures,
            "total_size_mb": total_size / (1024 * 1024),
            "by_material": by_material
        }


if __name__ == "__main__":
    # Test
    manager = TextureDownloadManager()
    
    # Recherche
    results = manager.search_polyhaven_textures(["concrete"], "1k")
    print(f"Trouvé {len(results)} textures")
    
    # Stats
    stats = manager.get_library_stats()
    print(f"Bibliothèque : {stats['total_textures']} textures ({stats['total_size_mb']:.1f} MB)")
