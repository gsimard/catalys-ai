import argparse
import json
import pickle # Ajout de pickle
import numpy as np
from bge_embeddings import EmbeddingGenerator

def load_kb(kb_file):
    """
    Charge une base de connaissances préparée depuis un fichier Pickle.

    Args:
        kb_file: Chemin vers le fichier Pickle (.pkl) de la base de connaissances

    Returns:
        Tuple: (Textes, sources, embeddings) ou lève une exception en cas d'erreur.
    """
    print(f"Chargement de la base de connaissances depuis {kb_file}...")
    try:
        with open(kb_file, 'rb') as f: # Utiliser 'rb' pour le binaire
            documents = pickle.load(f)
    except FileNotFoundError:
        print(f"Erreur: Le fichier {kb_file} n'a pas été trouvé.")
        raise
    except Exception as e:
        print(f"Erreur lors du chargement du fichier pickle {kb_file}: {e}")
        raise

    if not documents:
        print("Attention: Le fichier de base de connaissances est vide.")
        return [], [], np.array([])

    texts = [doc.get("content", "") for doc in documents]
    sources = [doc.get("source", "Inconnue") for doc in documents]

    # Les embeddings devraient être présents et être des tableaux numpy
    if documents and "embedding" in documents[0] and documents[0]["embedding"] is not None:
        # Empiler les embeddings (probablement F16)
        embeddings = np.array([doc["embedding"] for doc in documents])
        print(f"Embeddings chargés (dtype: {embeddings.dtype})")
    else:
        # Ceci ne devrait pas arriver si le fichier a été créé par prepare_kb.py
        print("Attention: Embeddings non trouvés dans le fichier. Retour d'embeddings vides.")
        embeddings = None # Ou np.array([]) ? Retourner None pour forcer la regénération si nécessaire

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
    
    print("Recherche des documents pertinents...")
    from tqdm import tqdm
    
    # Calcul des similarités avec barre de progression
    # Diviser le calcul en batches pour montrer la progression
    batch_size = 1000  # Taille de batch adaptée
    similarities = np.zeros(len(embeddings))
    
    for i in tqdm(range(0, len(embeddings), batch_size), 
                 desc="Recherche de documents", unit="batch"):
        end_idx = min(i + batch_size, len(embeddings))
        batch_embeddings = embeddings[i:end_idx]
        similarities[i:end_idx] = np.dot(batch_embeddings, query_embedding)
    
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
                        help="Fichier PKL de la base de connaissances préparée") # Changement JSON -> PKL
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
    try:
        texts, sources, embeddings = load_kb(args.kb)
        if embeddings is None:
             # Si load_kb retourne None pour embeddings, c'est qu'ils manquaient.
             print("Embeddings non trouvés dans le fichier PKL, regénération nécessaire...")
             if not texts:
                 print("Aucun texte à encoder. Arrêt.")
                 return
             generator = EmbeddingGenerator(device=args.device)
             embeddings = generator.generate_embeddings(texts)
             # Optionnel: Sauvegarder le fichier PKL mis à jour ? Pour l'instant, non.
        elif len(texts) == 0:
             print("La base de connaissances est vide. Arrêt.")
             return
        else:
             # Embeddings chargés, on a juste besoin du générateur pour la requête
             generator = EmbeddingGenerator(device=args.device)

        print(f"Base de connaissances chargée: {len(texts)} documents")

    except Exception as e:
        print(f"Erreur lors du chargement de la base de connaissances: {e}")
        return # Arrêt si le chargement échoue
    
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
