import argparse
import json
import numpy as np
from bge_embeddings import EmbeddingGenerator

def load_kb(kb_file):
    """
    Charge une base de connaissances préparée.
    
    Args:
        kb_file: Chemin vers le fichier JSON de la base de connaissances
        
    Returns:
        Textes, sources et embeddings
    """
    with open(kb_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    
    texts = [doc["content"] for doc in documents]
    sources = [doc["source"] for doc in documents]
    
    # Si les embeddings sont déjà dans le fichier JSON
    if "embedding" in documents[0]:
        embeddings = np.array([doc["embedding"] for doc in documents])
    else:
        embeddings = None
    
    return texts, sources, embeddings

def search(query, texts, sources, embeddings, generator, top_k=5):
    """
    Effectue une recherche sémantique.
    
    Args:
        query: Requête
        texts: Textes de la base de connaissances
        sources: Sources des textes
        embeddings: Embeddings des textes
        generator: Générateur d'embeddings
        top_k: Nombre de résultats à retourner
        
    Returns:
        Résultats de la recherche
    """
    # Génération de l'embedding pour la requête
    query_embedding = generator.generate_embeddings([query])[0]
    
    # Calcul des similarités
    similarities = np.dot(embeddings, query_embedding)
    
    # Récupération des indices des documents les plus similaires
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Récupération des résultats
    results = []
    for i in top_indices:
        results.append({
            "score": float(similarities[i]),
            "content": texts[i],
            "source": sources[i]
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Recherche sémantique avec bge-m3")
    parser.add_argument("--kb", type=str, required=True, 
                        help="Fichier JSON de la base de connaissances")
    parser.add_argument("--query", type=str, 
                        help="Requête à traiter")
    parser.add_argument("--top-k", type=int, default=5, 
                        help="Nombre de résultats à retourner")
    parser.add_argument("--device", type=str, default=None, 
                        help="Appareil à utiliser")
    parser.add_argument("--interactive", action="store_true", 
                        help="Mode interactif")
    
    args = parser.parse_args()
    
    # Chargement de la base de connaissances
    print(f"Chargement de la base de connaissances depuis {args.kb}...")
    texts, sources, embeddings = load_kb(args.kb)
    print(f"Base de connaissances chargée: {len(texts)} documents")
    
    # Si les embeddings ne sont pas dans le fichier, les générer
    if embeddings is None:
        print("Génération des embeddings...")
        generator = EmbeddingGenerator(device=args.device)
        embeddings = generator.generate_embeddings(texts)
    else:
        generator = EmbeddingGenerator(device=args.device)
    
    # Mode interactif
    if args.interactive:
        print("\nMode interactif. Tapez 'exit' pour quitter.")
        while True:
            query = input("\nRecherche: ")
            if query.lower() == 'exit':
                break
                
            results = search(query, texts, sources, embeddings, generator, args.top_k)
            
            print("\nRésultats:")
            for i, result in enumerate(results):
                print(f"\n{i+1}. Score: {result['score']:.4f}")
                print(f"Source: {result['source']}")
                print(f"Contenu: {result['content'][:200]}...")
    
    # Mode non interactif
    elif args.query:
        results = search(args.query, texts, sources, embeddings, generator, args.top_k)
        
        print("\nRésultats:")
        for i, result in enumerate(results):
            print(f"\n{i+1}. Score: {result['score']:.4f}")
            print(f"Source: {result['source']}")
            print(f"Contenu: {result['content'][:200]}...")
    
    else:
        print("Veuillez spécifier une requête ou utiliser le mode interactif.")


if __name__ == "__main__":
    main()
