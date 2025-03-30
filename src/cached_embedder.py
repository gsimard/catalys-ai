import hashlib
import os
import pickle
from pathlib import Path
import torch
from typing import Union, List, Dict
import logging
from datetime import datetime

from src.embedding_model import BGEM3Embedder

class CachedBGEM3Embedder(BGEM3Embedder):
    def __init__(self, cache_dir: str = "./embeddings_cache", **kwargs):
        super().__init__(**kwargs)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.hits = 0
        self.misses = 0
        
        # Configuration du logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("CachedEmbedder")
        
        self.logger.info(f"Cache initialisé dans {self.cache_dir}")
        self.logger.info(f"Modèle chargé sur {self.device}")

    def _get_cache_path(self, text: str, pooling: str, max_length: int) -> Path:
        """
        Génère un chemin de cache unique basé sur le texte et les paramètres d'embedding.
        
        Args:
            text: Texte à encoder
            pooling: Méthode de pooling utilisée
            max_length: Longueur maximale de séquence
            
        Returns:
            Chemin du fichier de cache
        """
        # Inclure les paramètres dans la clé de hachage pour différencier les embeddings
        # avec différentes configurations
        hash_input = f"{text}_{pooling}_{max_length}"
        hash_key = hashlib.sha256(hash_input.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.pkl"

    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        max_length: int = 8192,
        pooling: str = "cls",
        use_cache: bool = True
    ) -> torch.Tensor:
        """
        Génère des embeddings pour une liste de textes avec mise en cache.
        
        Args:
            texts: Un texte ou une liste de textes à encoder
            batch_size: Taille des lots pour le traitement
            max_length: Longueur maximale des séquences
            pooling: Méthode de pooling ('cls', 'mean', 'max')
            use_cache: Activer/désactiver l'utilisation du cache
            
        Returns:
            Tensor contenant les embeddings normalisés
        """
        # Conversion en liste si un seul texte est fourni
        if isinstance(texts, str):
            texts = [texts]
            
        # Si le cache est désactivé, utiliser l'implémentation de base
        if not use_cache:
            return super().embed(texts, batch_size, max_length, pooling)
        
        # Vérifier quels textes sont dans le cache
        cached_embeddings = {}
        texts_to_embed = []
        indices_to_embed = []
        
        for i, text in enumerate(texts):
            cache_path = self._get_cache_path(text, pooling, max_length)
            
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        cached_embeddings[i] = pickle.load(f)
                    self.hits += 1
                except (pickle.PickleError, EOFError):
                    # En cas d'erreur de lecture du cache, recalculer l'embedding
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
                    self.misses += 1
                    self.logger.warning(f"Erreur de lecture du cache pour {cache_path}, recalcul de l'embedding")
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
                self.misses += 1
        
        # Calculer les embeddings manquants
        if texts_to_embed:
            new_embeddings = super().embed(texts_to_embed, batch_size, max_length, pooling)
            
            # Mettre en cache les nouveaux embeddings
            for idx, text_idx in enumerate(indices_to_embed):
                embedding = new_embeddings[idx].unsqueeze(0)
                cached_embeddings[text_idx] = embedding
                
                cache_path = self._get_cache_path(texts[text_idx], pooling, max_length)
                with open(cache_path, 'wb') as f:
                    pickle.dump(embedding, f)
        
        # Assembler tous les embeddings dans l'ordre original
        all_embeddings = []
        for i in range(len(texts)):
            all_embeddings.append(cached_embeddings[i])
            
        # Log des statistiques du cache
        self.logger.info(f"Statistiques du cache - Hits: {self.hits}, Misses: {self.misses}, " 
                         f"Ratio: {self.hits/(self.hits+self.misses):.2f}")
            
        return torch.cat(all_embeddings, dim=0)
    
    def clear_cache(self):
        """Vide le cache d'embeddings."""
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
            count += 1
        
        self.logger.info(f"Cache vidé: {count} fichiers supprimés")
        return count
    
    def get_cache_stats(self) -> Dict:
        """
        Retourne des statistiques sur le cache.
        
        Returns:
            Dictionnaire contenant les statistiques du cache
        """
        cache_files = list(self.cache_dir.glob("*.pkl"))
        cache_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
            "cache_files": len(cache_files),
            "cache_size_bytes": cache_size,
            "cache_size_mb": cache_size / (1024 * 1024)
        }
