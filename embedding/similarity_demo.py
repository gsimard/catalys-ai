import numpy as np
from bge_m3_embeddings import EmbeddingGenerator
import argparse

def cosine_similarity(a, b):
    """Calcule la similarité cosinus entre deux vecteurs."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    parser = argparse.ArgumentParser(description="Démonstration de similarité avec bge-m3")
    parser.add_argument("--device", type=str, default=None, 
                        help="Appareil à utiliser (cpu, cuda, cuda:0, etc.)")
    
    args = parser.parse_args()
    
    # Initialisation du générateur d'embeddings
    generator = EmbeddingGenerator(device=args.device)
    
    # Exemples de phrases pour démontrer la similarité
    queries = [
        "Quelle est la capitale de la France?",
        "Paris est la capitale de quel pays?",
        "Comment fonctionne l'intelligence artificielle?",
        "Expliquez-moi le fonctionnement des réseaux de neurones.",
        "Quelle est la recette de la tarte aux pommes?",
    ]
    
    # Génération des embeddings
    print("Génération des embeddings...")
    embeddings = generator.generate_embeddings(queries)
    
    # Calcul et affichage de la matrice de similarité
    print("\nMatrice de similarité cosinus:")
    similarity_matrix = np.zeros((len(queries), len(queries)))
    
    for i in range(len(queries)):
        for j in range(len(queries)):
            similarity = cosine_similarity(embeddings[i], embeddings[j])
            similarity_matrix[i, j] = similarity
    
    # Affichage formaté
    print("\n" + " " * 10 + " ".join([f"Phrase {i+1:<8}" for i in range(len(queries))]))
    for i in range(len(queries)):
        print(f"Phrase {i+1:<3}", end=" ")
        for j in range(len(queries)):
            print(f"{similarity_matrix[i, j]:.4f}    ", end="")
        print()
    
    print("\nDétail des phrases:")
    for i, query in enumerate(queries):
        print(f"Phrase {i+1}: {query}")
    
    # Trouver les phrases les plus similaires
    print("\nPaires de phrases les plus similaires:")
    for i in range(len(queries)):
        for j in range(i+1, len(queries)):
            if similarity_matrix[i, j] > 0.5:  # Seuil arbitraire
                print(f"Similarité {similarity_matrix[i, j]:.4f} entre:")
                print(f"  - {queries[i]}")
                print(f"  - {queries[j]}")
                print()

if __name__ == "__main__":
    main()
