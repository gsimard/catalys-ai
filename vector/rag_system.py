import os
import numpy as np
import json
import argparse
from bge_embeddings import EmbeddingGenerator
import torch

# Vérification de la disponibilité de transformers
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Le package transformers n'est pas installé. La génération de réponses ne sera pas disponible.")


class RAGSystem:
    def __init__(self, embedding_model="BAAI/bge-m3", llm_model="meta-llama/Llama-2-7b-chat-hf", device=None):
        """
        Initialise le système RAG avec un modèle d'embedding et un LLM.
        
        Args:
            embedding_model: Modèle d'embedding à utiliser
            llm_model: Modèle de langage à utiliser
            device: Appareil sur lequel exécuter les modèles
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Initialisation du système RAG sur {self.device}...")
        
        # Chargement du générateur d'embeddings
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model, device=self.device)
        
        # Chargement du LLM si disponible
        self.llm = None
        self.tokenizer_llm = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                print(f"Chargement du LLM {llm_model}...")
                self.tokenizer_llm = AutoTokenizer.from_pretrained(llm_model)
                self.llm = AutoModelForCausalLM.from_pretrained(llm_model).to(self.device)
                print("LLM chargé avec succès!")
            except Exception as e:
                print(f"Erreur lors du chargement du LLM: {e}")
                print("Le système fonctionnera en mode recherche seulement.")
        
        # Base de connaissances
        self.kb_texts = []
        self.kb_sources = []
        self.kb_embeddings = None
        
    def load_knowledge_base(self, kb_file):
        """
        Charge une base de connaissances à partir d'un fichier texte.
        
        Args:
            kb_file: Chemin vers le fichier de la base de connaissances
        """
        print(f"Chargement de la base de connaissances depuis {kb_file}...")
        
        with open(kb_file, 'r', encoding='utf-8') as f:
            self.kb_texts = [line.strip() for line in f if line.strip()]
            self.kb_sources = [kb_file] * len(self.kb_texts)
            
        print(f"Génération des embeddings pour {len(self.kb_texts)} documents...")
        self.kb_embeddings = self.embedding_generator.generate_embeddings(self.kb_texts)
        
        print("Base de connaissances chargée avec succès!")
        
    def load_knowledge_base_from_directory(self, directory, extensions=['.txt', '.md']):
        """
        Charge une base de connaissances à partir d'un répertoire.
        
        Args:
            directory: Chemin vers le répertoire contenant les documents
            extensions: Extensions de fichiers à considérer
        """
        print(f"Chargement de la base de connaissances depuis {directory}...")
        
        self.kb_texts = []
        self.kb_sources = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            # Diviser le contenu en paragraphes ou chunks
                            chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
                            self.kb_texts.extend(chunks)
                            self.kb_sources.extend([file_path] * len(chunks))
                    except Exception as e:
                        print(f"Erreur lors de la lecture de {file_path}: {e}")
        
        print(f"Génération des embeddings pour {len(self.kb_texts)} documents...")
        self.kb_embeddings = self.embedding_generator.generate_embeddings(self.kb_texts)
        
        print("Base de connaissances chargée avec succès!")
        
    def load_knowledge_base_from_json(self, json_file):
        """
        Charge une base de connaissances préparée au format JSON.
        
        Args:
            json_file: Chemin vers le fichier JSON
        """
        print(f"Chargement de la base de connaissances depuis {json_file}...")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        self.kb_texts = [doc["content"] for doc in documents]
        self.kb_sources = [doc["source"] for doc in documents]
        
        # Si les embeddings sont déjà dans le fichier JSON
        if "embedding" in documents[0]:
            self.kb_embeddings = np.array([doc["embedding"] for doc in documents])
            print(f"Embeddings chargés depuis le fichier JSON pour {len(self.kb_texts)} documents.")
        else:
            print(f"Génération des embeddings pour {len(self.kb_texts)} documents...")
            self.kb_embeddings = self.embedding_generator.generate_embeddings(self.kb_texts)
        
        print("Base de connaissances chargée avec succès!")
        
    def retrieve(self, query, top_k=3):
        """
        Récupère les documents les plus pertinents pour une requête.
        
        Args:
            query: Requête de l'utilisateur
            top_k: Nombre de documents à récupérer
            
        Returns:
            Liste des documents les plus pertinents
        """
        if self.kb_embeddings is None or len(self.kb_texts) == 0:
            raise ValueError("La base de connaissances n'est pas chargée.")
            
        # Génération de l'embedding pour la requête
        query_embedding = self.embedding_generator.generate_embeddings([query])[0]
        
        # Calcul des similarités avec tous les documents
        similarities = np.dot(self.kb_embeddings, query_embedding)
        
        # Récupération des indices des documents les plus similaires
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Récupération des documents et de leurs scores
        top_docs = [(self.kb_texts[i], self.kb_sources[i], similarities[i]) for i in top_indices]
        
        return top_docs
        
    def generate(self, query, top_k=3, max_length=512):
        """
        Génère une réponse à une requête en utilisant le RAG.
        
        Args:
            query: Requête de l'utilisateur
            top_k: Nombre de documents à récupérer
            max_length: Longueur maximale de la réponse
            
        Returns:
            Réponse générée
        """
        # Vérification de la disponibilité du LLM
        if not TRANSFORMERS_AVAILABLE or self.llm is None:
            raise ValueError("Le LLM n'est pas disponible. Installez transformers et chargez un modèle.")
        
        # Récupération des documents pertinents
        relevant_docs = self.retrieve(query, top_k=top_k)
        
        # Construction du contexte
        context = "\n\n".join([doc for doc, _, _ in relevant_docs])
        
        # Construction du prompt
        prompt = f"""Contexte:
{context}

Question: {query}

Réponse:"""
        
        # Tokenisation
        inputs = self.tokenizer_llm(prompt, return_tensors="pt").to(self.device)
        
        # Génération
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
            )
        
        # Décodage
        response = self.tokenizer_llm.decode(outputs[0], skip_special_tokens=True)
        
        # Extraction de la réponse (après "Réponse:")
        response = response.split("Réponse:")[-1].strip()
        
        return response, relevant_docs


def main():
    parser = argparse.ArgumentParser(description="Système RAG avec bge-m3")
    parser.add_argument("--kb", type=str, help="Fichier ou répertoire de la base de connaissances")
    parser.add_argument("--device", type=str, default=None, 
                        help="Appareil à utiliser (cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--llm", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help="Modèle de langage à utiliser")
    parser.add_argument("--interactive", action="store_true", 
                        help="Mode interactif")
    parser.add_argument("--query", type=str, 
                        help="Requête à traiter (mode non interactif)")
    parser.add_argument("--search-only", action="store_true",
                        help="Mode recherche seulement (sans génération)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Nombre de documents à récupérer")
    
    args = parser.parse_args()
    
    # Initialisation du système RAG
    rag_system = RAGSystem(llm_model=args.llm, device=args.device)
    
    # Chargement de la base de connaissances
    if args.kb:
        if args.kb.endswith('.json'):
            rag_system.load_knowledge_base_from_json(args.kb)
        elif os.path.isdir(args.kb):
            rag_system.load_knowledge_base_from_directory(args.kb)
        else:
            rag_system.load_knowledge_base(args.kb)
    else:
        print("Aucune base de connaissances spécifiée.")
        return
    
    # Mode interactif
    if args.interactive:
        print("\nMode interactif. Tapez 'exit' pour quitter.")
        while True:
            query = input("\nQuestion: ")
            if query.lower() == 'exit':
                break
                
            try:
                # Mode recherche seulement
                if args.search_only or not TRANSFORMERS_AVAILABLE or rag_system.llm is None:
                    docs = rag_system.retrieve(query, top_k=args.top_k)
                    
                    print("\nDocuments pertinents:")
                    for i, (doc, source, score) in enumerate(docs):
                        print(f"{i+1}. [{score:.4f}] Source: {source}")
                        print(f"   {doc[:200]}...")
                # Mode RAG complet
                else:
                    response, docs = rag_system.generate(query, top_k=args.top_k)
                    
                    print("\nDocuments pertinents:")
                    for i, (doc, source, score) in enumerate(docs):
                        print(f"{i+1}. [{score:.4f}] Source: {source}")
                        print(f"   {doc[:200]}...")
                        
                    print(f"\nRéponse: {response}")
            except Exception as e:
                print(f"Erreur: {e}")
    
    # Mode non interactif
    elif args.query:
        try:
            # Mode recherche seulement
            if args.search_only or not TRANSFORMERS_AVAILABLE or rag_system.llm is None:
                docs = rag_system.retrieve(args.query, top_k=args.top_k)
                
                print("\nDocuments pertinents:")
                for i, (doc, source, score) in enumerate(docs):
                    print(f"{i+1}. [{score:.4f}] Source: {source}")
                    print(f"   {doc[:200]}...")
            # Mode RAG complet
            else:
                response, docs = rag_system.generate(args.query, top_k=args.top_k)
                
                print("\nDocuments pertinents:")
                for i, (doc, source, score) in enumerate(docs):
                    print(f"{i+1}. [{score:.4f}] Source: {source}")
                    print(f"   {doc[:200]}...")
                    
                print(f"\nRéponse: {response}")
        except Exception as e:
            print(f"Erreur: {e}")
    
    else:
        print("Veuillez spécifier une requête ou utiliser le mode interactif.")


if __name__ == "__main__":
    main()
