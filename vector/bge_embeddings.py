import torch
from transformers import AutoTokenizer, AutoModel
import argparse
import json
import pickle # Ajout de pickle
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name="BAAI/bge-m3", device=None):
        """
        Initialise le générateur d'embeddings avec le modèle bge-m3.
        
        Args:
            model_name: Nom du modèle à charger
            device: Appareil sur lequel exécuter le modèle (None pour auto-détection)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Chargement du modèle {model_name} sur {self.device}...")
        
        # Chargement du tokenizer et du modèle
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        print(f"Modèle chargé avec succès!")
        
        # Informations sur l'utilisation de la VRAM si disponible
        if self.device == "cuda":
            print(f"VRAM utilisée: {torch.cuda.memory_allocated() / 1024**2:.2f} Mo")
            print(f"VRAM totale: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} Mo")
    
    def generate_embeddings(self, texts, batch_size=8):
        """
        Génère des embeddings pour une liste de textes.
        
        Args:
            texts: Liste de textes à encoder
            batch_size: Taille du batch pour le traitement
            
        Returns:
            Tableau numpy des embeddings normalisés
        """
        embeddings = []
        
        # Traitement par batch pour économiser la mémoire
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            # Tokenisation
            inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                   max_length=512, return_tensors="pt").to(self.device)
            
            # Génération des embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Utilisation des embeddings de la couche [CLS]
            batch_embeddings = outputs.last_hidden_state[:, 0].cpu().numpy()
            
            # Normalisation L2
            batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            
            embeddings.append(batch_embeddings)
        
        # Concaténation de tous les embeddings
        return np.vstack(embeddings)
    
    def save_embeddings(self, texts, embeddings, output_file):
        """
        Sauvegarde les textes et leurs embeddings dans un fichier JSON.
        
        Args:
            texts: Liste des textes
            embeddings: Tableau numpy des embeddings
            output_file: Chemin du fichier de sortie
        """
        results = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            results.append({
                "id": i,
                "text": text,
                "embedding": embedding.tolist()
            })
            
        with open(output_file, 'w', encoding='utf-8') as f:
            # Structure de sauvegarde : un dictionnaire contenant les textes et l'array numpy des embeddings
            data_to_save = {
                "texts": texts,
                "embeddings": embeddings # embeddings est déjà un array numpy
            }
            pickle.dump(data_to_save, f)

        print(f"Textes et embeddings sauvegardés dans {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Génération d'embeddings avec bge-m3")
    parser.add_argument("--input", type=str, help="Fichier texte avec une entrée par ligne")
    parser.add_argument("--output", type=str, default="embeddings.pkl", # Changement .json -> .pkl
                        help="Fichier de sortie pour les textes et embeddings (PKL)") # Changement JSON -> PKL
    parser.add_argument("--device", type=str, default=None,
                        help="Appareil à utiliser (cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Taille du batch pour le traitement")
    parser.add_argument("--texts", nargs="+", default=[], 
                        help="Textes à encoder directement via la ligne de commande")
    
    args = parser.parse_args()
    
    # Vérification des entrées
    if not args.input and not args.texts:
        print("Erreur: Veuillez fournir soit un fichier d'entrée, soit des textes directement.")
        return
    
    # Chargement des textes
    texts = []
    if args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    
    if args.texts:
        texts.extend(args.texts)
    
    print(f"Traitement de {len(texts)} textes...")
    
    # Génération des embeddings
    generator = EmbeddingGenerator(device=args.device)
    embeddings = generator.generate_embeddings(texts, batch_size=args.batch_size)

    # Conversion en float16 avant sauvegarde
    embeddings_f16 = embeddings.astype(np.float16)
    print(f"Embeddings générés et convertis en {embeddings_f16.dtype}")

    # Sauvegarde des résultats (textes et embeddings F16)
    generator.save_embeddings(texts, embeddings_f16, args.output)
    
    # Affichage d'un exemple
    if len(texts) > 0:
        print("\nExemple d'embedding:")
        print(f"Texte: {texts[0]}")
        print(f"Dimension de l'embedding: {embeddings[0].shape}")
        print(f"Premiers éléments: {embeddings[0][:5]}")


if __name__ == "__main__":
    main()
