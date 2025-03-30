from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from typing import List, Tuple, Union, Dict

class EmbeddingComparator:
    @staticmethod
    def cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
        """
        Calcule la matrice de similarité cosinus entre tous les embeddings.
        
        Args:
            embeddings: Matrice d'embeddings de forme (n_samples, n_features)
            
        Returns:
            Matrice de similarité de forme (n_samples, n_samples)
        """
        return cosine_similarity(embeddings)

    @staticmethod
    def find_most_similar(query_embedding: np.ndarray, corpus_embeddings: np.ndarray, top_k: int = 1) -> Union[int, List[int]]:
        """
        Trouve les indices des embeddings les plus similaires à l'embedding de requête.
        
        Args:
            query_embedding: Embedding de requête de forme (1, n_features) ou (n_features,)
            corpus_embeddings: Matrice d'embeddings de forme (n_samples, n_features)
            top_k: Nombre de résultats similaires à retourner
            
        Returns:
            Indice du document le plus similaire ou liste des top_k indices
        """
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        similarities = cosine_similarity(query_embedding, corpus_embeddings).flatten()
        
        if top_k == 1:
            return np.argmax(similarities)
        else:
            return list(np.argsort(similarities)[-top_k:][::-1])
    
    @staticmethod
    def get_similarity_scores(query_embedding: np.ndarray, corpus_embeddings: np.ndarray) -> np.ndarray:
        """
        Calcule les scores de similarité entre l'embedding de requête et tous les embeddings du corpus.
        
        Args:
            query_embedding: Embedding de requête de forme (1, n_features) ou (n_features,)
            corpus_embeddings: Matrice d'embeddings de forme (n_samples, n_features)
            
        Returns:
            Tableau des scores de similarité
        """
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        return cosine_similarity(query_embedding, corpus_embeddings).flatten()
    
    @staticmethod
    def search_by_similarity(
        query_embedding: np.ndarray, 
        corpus_embeddings: np.ndarray, 
        texts: List[str], 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Recherche les textes les plus similaires à partir d'un embedding de requête.
        
        Args:
            query_embedding: Embedding de requête
            corpus_embeddings: Matrice d'embeddings du corpus
            texts: Liste des textes correspondant aux embeddings
            top_k: Nombre de résultats à retourner
            
        Returns:
            Liste de tuples (texte, score de similarité) triés par similarité décroissante
        """
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        similarities = cosine_similarity(query_embedding, corpus_embeddings).flatten()
        indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in indices:
            results.append((texts[idx], float(similarities[idx])))
            
        return results

def convert_to_numpy(embeddings: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Convertit des embeddings en tableau numpy.
    
    Args:
        embeddings: Embeddings sous forme de tensor PyTorch ou tableau numpy
        
    Returns:
        Embeddings sous forme de tableau numpy
    """
    if isinstance(embeddings, torch.Tensor):
        return embeddings.cpu().numpy()
    return embeddings
