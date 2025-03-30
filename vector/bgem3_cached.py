import os
import json
import hashlib
import torch
from typing import Union, List, Dict, Optional
import pickle
from pathlib import Path

from bgem3_encoder import BGEM3Encoder

class CachedBGEM3Encoder(BGEM3Encoder):
    """Extension de BGEM3Encoder avec cache des embeddings sur disque"""
    
    def __init__(self, 
                model_name: str = "BAAI/bge-m3", 
                device: str = "auto",
                cache_dir: str = ".embedding_cache"):
        """Initialise l'encodeur avec support de cache"""
        super().__init__(model_name, device)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.cache_index: Dict[str, str] = self._load_cache_index()
        
    def _load_cache_index(self) -> Dict[str, str]:
        """Charge l'index de cache existant ou crée un nouveau"""
        index_path = self.cache_dir / "index.json"
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    def _save_cache_index(self) -> None:
        """Sauvegarde l'index de cache sur disque"""
        index_path = self.cache_dir / "index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(self.cache_index, f, ensure_ascii=False, indent=2)
    
    def _get_text_hash(self, text: str) -> str:
        """Génère un hash unique pour un texte donné"""
        return hashlib.md5(text.encode("utf-8")).hexdigest()
    
    def _get_cache_path(self, text_hash: str) -> Path:
        """Retourne le chemin du fichier de cache pour un hash donné"""
        return self.cache_dir / f"{text_hash}.pt"
    
    def _is_in_cache(self, text: str) -> bool:
        """Vérifie si un texte est déjà dans le cache"""
        text_hash = self._get_text_hash(text)
        return text_hash in self.cache_index
    
    def _get_from_cache(self, text: str) -> Optional[torch.Tensor]:
        """Récupère l'embedding depuis le cache si disponible"""
        if not self._is_in_cache(text):
            return None
            
        text_hash = self._get_text_hash(text)
        cache_path = self._get_cache_path(text_hash)
        
        if not cache_path.exists():
            # Nettoyage de l'index si le fichier n'existe plus
            del self.cache_index[text_hash]
            self._save_cache_index()
            return None
            
        try:
            return torch.load(cache_path)
        except Exception as e:
            print(f"Erreur lors du chargement du cache: {e}")
            return None
    
    def _save_to_cache(self, text: str, embedding: torch.Tensor) -> None:
        """Sauvegarde un embedding dans le cache"""
        text_hash = self._get_text_hash(text)
        cache_path = self._get_cache_path(text_hash)
        
        torch.save(embedding, cache_path)
        self.cache_index[text_hash] = text[:100] + "..." if len(text) > 100 else text
        self._save_cache_index()
    
    def encode(self, 
              texts: Union[str, List[str]], 
              batch_size: int = 32,
              max_length: int = 8192,
              pooling_method: str = "cls",
              use_cache: bool = True) -> torch.Tensor:
        """Encode les textes avec support de cache"""
        
        # Cas d'un texte unique
        if isinstance(texts, str):
            if use_cache:
                cached_embedding = self._get_from_cache(texts)
                if cached_embedding is not None:
                    return cached_embedding
                    
            embedding = super().encode(texts, batch_size, max_length, pooling_method)
            
            if use_cache:
                self._save_to_cache(texts, embedding)
                
            return embedding
        
        # Cas d'une liste de textes
        if use_cache:
            # Vérifier quels textes sont dans le cache
            cached_embeddings = []
            texts_to_encode = []
            indices_to_encode = []
            
            for i, text in enumerate(texts):
                cached_embedding = self._get_from_cache(text)
                if cached_embedding is not None:
                    cached_embeddings.append((i, cached_embedding))
                else:
                    texts_to_encode.append(text)
                    indices_to_encode.append(i)
            
            # Si tous les textes sont dans le cache
            if not texts_to_encode:
                # Trier par indice original et extraire les embeddings
                cached_embeddings.sort(key=lambda x: x[0])
                return torch.stack([emb for _, emb in cached_embeddings])
            
            # Encoder les textes manquants
            new_embeddings = super().encode(texts_to_encode, batch_size, max_length, pooling_method)
            
            # Sauvegarder les nouveaux embeddings dans le cache
            for i, text in enumerate(texts_to_encode):
                self._save_to_cache(text, new_embeddings[i])
            
            # Combiner les embeddings cachés et nouveaux
            all_embeddings = torch.zeros(
                (len(texts), new_embeddings.shape[1]), 
                dtype=new_embeddings.dtype
            )
            
            # Placer les embeddings cachés
            for idx, emb in cached_embeddings:
                all_embeddings[idx] = emb
                
            # Placer les nouveaux embeddings
            for i, orig_idx in enumerate(indices_to_encode):
                all_embeddings[orig_idx] = new_embeddings[i]
                
            return all_embeddings
        
        # Sans cache, utiliser l'implémentation de base
        return super().encode(texts, batch_size, max_length, pooling_method)
    
    def clear_cache(self, older_than_days: Optional[int] = None) -> int:
        """Nettoie le cache, optionnellement en fonction de l'âge des fichiers
        
        Args:
            older_than_days: Si spécifié, supprime uniquement les fichiers plus anciens que ce nombre de jours
            
        Returns:
            Nombre de fichiers supprimés
        """
        import time
        from datetime import datetime, timedelta
        
        count = 0
        current_time = time.time()
        
        # Parcourir tous les fichiers de cache
        for cache_file in self.cache_dir.glob("*.pt"):
            if older_than_days is not None:
                file_mtime = cache_file.stat().st_mtime
                file_age_days = (current_time - file_mtime) / (60 * 60 * 24)
                
                if file_age_days < older_than_days:
                    continue
                    
            # Supprimer le fichier
            cache_file.unlink()
            count += 1
            
            # Mettre à jour l'index
            file_hash = cache_file.stem
            if file_hash in self.cache_index:
                del self.cache_index[file_hash]
        
        # Sauvegarder l'index mis à jour
        self._save_cache_index()
        return count
    
    def get_cache_stats(self) -> Dict:
        """Retourne des statistiques sur le cache"""
        cache_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pt"))
        return {
            "entries_count": len(self.cache_index),
            "cache_size_mb": cache_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir)
        }


# Exemple d'utilisation
if __name__ == "__main__":
    import time
    
    # Initialiser l'encodeur avec cache
    encoder = CachedBGEM3Encoder(cache_dir=".embedding_cache")
    
    texts = [
        "La révolution française de 1789",
        "量子コンピューティングの基本原理",
        "Theoretical foundations of deep learning",
        "¿Cómo funciona el mecanismo de atención en los transformers?"
    ]
    
    # Premier appel (sans cache)
    print("Premier appel (sans cache):")
    start_time = time.time()
    embeddings = encoder.encode(texts)
    print(f"Temps d'inférence: {time.time() - start_time:.2f}s")
    print(f"Embedding shape: {embeddings.shape}")
    
    # Deuxième appel (avec cache)
    print("\nDeuxième appel (avec cache):")
    start_time = time.time()
    embeddings = encoder.encode(texts)
    print(f"Temps d'inférence: {time.time() - start_time:.2f}s")
    
    # Statistiques du cache
    print("\nStatistiques du cache:")
    print(encoder.get_cache_stats())
