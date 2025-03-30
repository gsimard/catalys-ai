import sys
import os
import torch
import numpy as np
import time
from pathlib import Path

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(str(Path(__file__).parent.parent))

from src.embedding_model import BGEM3Embedder
from src.cached_embedder import CachedBGEM3Embedder
from src.embedding_utils import EmbeddingComparator, convert_to_numpy

def basic_usage_example():
    """Exemple d'utilisation basique du modèle d'embedding."""
    print("\n=== Exemple d'utilisation basique ===")
    
    embedder = BGEM3Embedder()
    texts = ["Bonjour le monde", "Hello world", "Hola mundo"]
    
    print(f"Modèle chargé sur: {embedder.device}")
    
    start_time = time.time()
    embeddings = embedder.embed(texts)
    elapsed = time.time() - start_time
    
    print(f"Dimensions des embeddings: {embeddings.shape}")
    print(f"Temps d'inférence: {elapsed:.2f} secondes")
    
    # Calcul de similarité
    embeddings_np = convert_to_numpy(embeddings)
    similarity_matrix = EmbeddingComparator.cosine_similarity(embeddings_np)
    
    print("\nMatrice de similarité:")
    for i, text in enumerate(texts):
        print(f"{text}: {similarity_matrix[i]}")

def cached_embedder_example():
    """Exemple d'utilisation du modèle avec cache."""
    print("\n=== Exemple d'utilisation avec cache ===")
    
    cache_dir = "./examples/cache"
    embedder = CachedBGEM3Embedder(cache_dir=cache_dir)
    
    texts = [
        "L'intelligence artificielle révolutionne de nombreux domaines.",
        "Les modèles de langage permettent de générer du texte cohérent.",
        "Le deep learning est une branche de l'apprentissage automatique."
    ]
    
    print("Premier appel (sans cache):")
    start_time = time.time()
    embeddings1 = embedder.embed(texts)
    elapsed1 = time.time() - start_time
    print(f"Temps d'inférence: {elapsed1:.2f} secondes")
    
    print("\nDeuxième appel (avec cache):")
    start_time = time.time()
    embeddings2 = embedder.embed(texts)
    elapsed2 = time.time() - start_time
    print(f"Temps d'inférence: {elapsed2:.2f} secondes")
    
    # Vérification que les embeddings sont identiques
    are_equal = torch.allclose(embeddings1, embeddings2)
    print(f"Les embeddings sont identiques: {are_equal}")
    
    # Statistiques du cache
    stats = embedder.get_cache_stats()
    print("\nStatistiques du cache:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

def similarity_search_example():
    """Exemple de recherche par similarité."""
    print("\n=== Exemple de recherche par similarité ===")
    
    embedder = BGEM3Embedder()
    
    corpus = [
        "Paris est la capitale de la France.",
        "Londres est la capitale du Royaume-Uni.",
        "Berlin est la capitale de l'Allemagne.",
        "Rome est la capitale de l'Italie.",
        "Madrid est la capitale de l'Espagne."
    ]
    
    query = "Quelle est la capitale de la France ?"
    
    print(f"Requête: {query}")
    print(f"Corpus: {len(corpus)} documents")
    
    # Génération des embeddings
    corpus_embeddings = embedder.embed(corpus)
    query_embedding = embedder.embed(query)
    
    # Conversion en numpy
    corpus_embeddings_np = convert_to_numpy(corpus_embeddings)
    query_embedding_np = convert_to_numpy(query_embedding)
    
    # Recherche des documents les plus similaires
    results = EmbeddingComparator.search_by_similarity(
        query_embedding_np, 
        corpus_embeddings_np, 
        corpus, 
        top_k=3
    )
    
    print("\nRésultats de la recherche:")
    for i, (text, score) in enumerate(results):
        print(f"{i+1}. {text} (score: {score:.4f})")

if __name__ == "__main__":
    basic_usage_example()
    cached_embedder_example()
    similarity_search_example()
