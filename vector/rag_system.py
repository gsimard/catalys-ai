import os
import numpy as np
import os
import numpy as np
import json
import argparse
from bge_embeddings import EmbeddingGenerator
import torch
from dotenv import load_dotenv
import openai

# Vérification de la disponibilité de transformers
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Le package transformers n'est pas installé. La génération de réponses avec un LLM local ne sera pas disponible.")

# Vérification de la disponibilité de prompt_toolkit
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False
    print("Le package prompt_toolkit n'est pas installé. L'historique et la navigation avancée en mode interactif ne seront pas disponibles.")

# Charger les variables d'environnement depuis un fichier .env s'il existe
load_dotenv()

class RAGSystem:
    def __init__(self, embedding_model="BAAI/bge-m3", llm_model="meta-llama/Llama-2-7b-chat-hf", device=None, use_openai_endpoint=False):
        """
        Initialise le système RAG avec un modèle d'embedding et un LLM (local ou via API OpenAI).

        Args:
            embedding_model: Modèle d'embedding à utiliser.
            llm_model: Modèle de langage local à utiliser (si use_openai_endpoint=False).
            device: Appareil sur lequel exécuter les modèles locaux.
            use_openai_endpoint: Si True, utilise l'endpoint OpenAI configuré via les variables d'environnement.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Initialisation du système RAG...")
        print(f"Utilisation du device: {self.device} pour les embeddings.")

        # Chargement du générateur d'embeddings
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model, device=self.device)

        # Configuration du LLM
        self.llm = None
        self.tokenizer_llm = None
        self.openai_client = None
        self.openai_model = None
        self.use_openai = False

        # Vérifier si on doit utiliser l'endpoint OpenAI
        nebius_api_base = os.getenv("NEBIUS_API_BASE")
        nebius_api_key = os.getenv("NEBIUS_API_KEY")
        nebius_model = os.getenv("NEBIUS_AI_MODEL")

        if use_openai_endpoint and nebius_api_base and nebius_api_key and nebius_model:
            print(f"Configuration de l'endpoint OpenAI/Nebius: {nebius_api_base}")
            try:
                self.openai_client = openai.OpenAI(
                    base_url=nebius_api_base,
                    api_key=nebius_api_key,
                )
                self.openai_model = nebius_model
                self.use_openai = True
                print(f"Client OpenAI initialisé pour le modèle: {self.openai_model}")
            except Exception as e:
                print(f"Erreur lors de l'initialisation du client OpenAI: {e}")
                print("Retour à l'utilisation du LLM local si possible.")
                self.use_openai = False
        
        # Si on n'utilise pas OpenAI, essayer de charger le LLM local
        if not self.use_openai:
            if TRANSFORMERS_AVAILABLE:
                print(f"Tentative de chargement du LLM local {llm_model} sur {self.device}...")
                try:
                    self.tokenizer_llm = AutoTokenizer.from_pretrained(llm_model)
                    # Charger le modèle sur le CPU si CUDA n'est pas disponible ou si device est 'cpu'
                    model_device = self.device if self.device.startswith("cuda") else "cpu" 
                    self.llm = AutoModelForCausalLM.from_pretrained(llm_model).to(model_device)
                    print(f"LLM local chargé avec succès sur {model_device}!")
                except Exception as e:
                    print(f"Erreur lors du chargement du LLM local: {e}")
                    print("La génération de réponses ne sera pas disponible.")
            else:
                 print("Transformers non disponible et endpoint OpenAI non configuré. Génération de réponses impossible.")

        # Base de connaissances
        self.kb_texts = []
        self.kb_sources = []
        self.kb_chunk_ids = [] # Ajout pour stocker les IDs des chunks
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
        # Extraire les données, en gérant l'absence potentielle de chunk_id
        self.kb_texts = [doc.get("content", "") for doc in documents]
        self.kb_sources = [doc.get("source", "Inconnue") for doc in documents]
        self.kb_chunk_ids = [doc.get("chunk_id", -1) for doc in documents] # Charger chunk_id, avec -1 par défaut

        # Si les embeddings sont déjà dans le fichier JSON
        if documents and "embedding" in documents[0]:
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
        # Récupération des documents, sources, scores et chunk_ids
        top_docs = [
            (self.kb_texts[i], self.kb_sources[i], similarities[i], self.kb_chunk_ids[i]) 
            for i in top_indices
        ]
        
        return top_docs
        
    def generate(self, query, top_k=3, max_length=512):
        """
        Génère une réponse à une requête en utilisant le RAG.
        
        Args:
            query: Requête de l'utilisateur
            top_k: Nombre de documents à récupérer
            max_length: Longueur maximale de la réponse
            
        Returns:
            Tuple: (Réponse générée, Liste des documents pertinents)
        """
        # Récupération des documents pertinents
        relevant_docs = self.retrieve(query, top_k=top_k)
        
        # Construction du contexte
        # On ne prend que le premier élément (le texte du chunk) de chaque tuple retourné par retrieve
        context = "\n\n".join([doc for doc, _, _, _ in relevant_docs])
        
        # Construction du prompt
        prompt = f"""Contexte:
{context}

Question: {query}

Réponse:"""

        print(f"\n--- Prompt envoyé au LLM ---\n{prompt}\n---------------------------\n") # Ajout de l'impression du prompt

        response = "Impossible de générer une réponse."

        if self.use_openai and self.openai_client:
            # Génération via API OpenAI
            try:
                print("Génération via l'API OpenAI...")
                chat_completion = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": "Tu es un assistant IA qui répond aux questions en se basant sur le contexte fourni."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_length,
                    temperature=0.7,
                )
                response = chat_completion.choices[0].message.content.strip()
            except Exception as e:
                print(f"Erreur lors de l'appel à l'API OpenAI: {e}")
                response = f"Erreur lors de la génération via API: {e}"

        elif self.llm and self.tokenizer_llm:
             # Génération via LLM local
            print("Génération via le LLM local...")
             # S'assurer que les inputs sont sur le bon device (CPU ou CUDA)
            model_device = self.llm.device
            inputs = self.tokenizer_llm(prompt, return_tensors="pt").to(model_device)

            # Génération
            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + max_length, # Ajuster max_length
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer_llm.eos_token_id # Éviter les warnings
                )

            # Décodage
            # S'assurer de décoder seulement les nouveaux tokens générés
            output_text = self.tokenizer_llm.decode(outputs[0], skip_special_tokens=True)
            
            # Extraction de la réponse (après "Réponse:")
            # Trouver la dernière occurrence pour éviter les problèmes si "Réponse:" est dans le contexte
            response_marker = "Réponse:"
            marker_index = output_text.rfind(response_marker)
            if marker_index != -1:
                response = output_text[marker_index + len(response_marker):].strip()
            else:
                # Fallback si le marqueur n'est pas trouvé (peut arriver avec certains modèles)
                # On prend ce qui suit le prompt original
                prompt_lines = prompt.count('\n') + 1
                response_lines = output_text.split('\n')
                if len(response_lines) > prompt_lines:
                     response = "\n".join(response_lines[prompt_lines:]).strip()
                else: # Si le modèle n'a rien ajouté ou a mal formaté
                     response = output_text # Retourner tout pour inspection

        else:
            raise ValueError("Aucun LLM (local ou distant) n'est configuré ou disponible pour la génération.")

        return response, relevant_docs


def main():
    parser = argparse.ArgumentParser(description="Système RAG avec bge-m3")
    parser.add_argument("--kb", type=str, help="Fichier ou répertoire de la base de connaissances")
    parser.add_argument("--device", type=str, default=None, 
                        help="Appareil à utiliser (cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--llm", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help="Modèle de langage local à utiliser (ignoré si --use-openai)")
    parser.add_argument("--use-openai", action="store_true",
                        help="Utiliser l'endpoint OpenAI/Nebius configuré via les variables d'environnement")
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
    rag_system = RAGSystem(llm_model=args.llm, device=args.device, use_openai_endpoint=args.use_openai)

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
        print("\nMode interactif. Tapez 'exit' ou Ctrl+D pour quitter.")
        
        if PROMPT_TOOLKIT_AVAILABLE:
            # Utiliser prompt_toolkit si disponible
            session = PromptSession(history=FileHistory('.rag_history'))
            while True:
                try:
                    query = session.prompt("\nQuestion: ")
                    if query.lower() == 'exit':
                        break
                    if not query.strip(): # Ignorer les entrées vides
                        continue
                except (EOFError, KeyboardInterrupt):
                    break # Quitter proprement avec Ctrl+D ou Ctrl+C
                
                try:
                    # Déterminer si la génération est possible
                    generation_possible = rag_system.use_openai or (TRANSFORMERS_AVAILABLE and rag_system.llm is not None)

                    # Mode recherche seulement ou si la génération n'est pas possible
                    if args.search_only or not generation_possible:
                        if not generation_possible and not args.search_only:
                             print("\nAvertissement: Génération impossible (LLM non chargé ou non configuré). Affichage des résultats de recherche seulement.")
                        
                        docs = rag_system.retrieve(query, top_k=args.top_k)
                        
                        print("\nDocuments pertinents:")
                        for i, (doc, source, score, chunk_id) in enumerate(docs):
                            print(f"{i+1}. [{score:.4f}] Source: {source} (Chunk ID: {chunk_id})")
                            print(f"   Contenu: {doc}") # Affichage complet
                            print("-" * 20) # Séparateur
                    # Mode RAG complet
                    else:
                        response, docs = rag_system.generate(query, top_k=args.top_k)
                        
                        print("\nDocuments pertinents utilisés pour la réponse:")
                        for i, (doc, source, score, chunk_id) in enumerate(docs):
                            print(f"{i+1}. [{score:.4f}] Source: {source} (Chunk ID: {chunk_id})")
                            print(f"   Contenu: {doc}") # Affichage complet
                            print("-" * 20) # Séparateur
                            
                        print(f"\nRéponse:\n{response}")
                except Exception as e:
                    print(f"Erreur: {e}")
        else:
            # Fallback vers input() si prompt_toolkit n'est pas disponible
            while True:
                try:
                    query = input("\nQuestion: ")
                    if query.lower() == 'exit':
                        break
                    if not query.strip(): # Ignorer les entrées vides
                        continue
                except EOFError:
                    break # Quitter avec Ctrl+D
                
                try:
                    # Déterminer si la génération est possible
                    generation_possible = rag_system.use_openai or (TRANSFORMERS_AVAILABLE and rag_system.llm is not None)

                    # Mode recherche seulement ou si la génération n'est pas possible
                    if args.search_only or not generation_possible:
                        if not generation_possible and not args.search_only:
                             print("\nAvertissement: Génération impossible (LLM non chargé ou non configuré). Affichage des résultats de recherche seulement.")
                        
                        docs = rag_system.retrieve(query, top_k=args.top_k)
                        
                        print("\nDocuments pertinents:")
                        for i, (doc, source, score, chunk_id) in enumerate(docs):
                            print(f"{i+1}. [{score:.4f}] Source: {source} (Chunk ID: {chunk_id})")
                            print(f"   Contenu: {doc}") # Affichage complet
                            print("-" * 20) # Séparateur
                    # Mode RAG complet
                    else:
                        response, docs = rag_system.generate(query, top_k=args.top_k)
                        
                        print("\nDocuments pertinents utilisés pour la réponse:")
                        for i, (doc, source, score, chunk_id) in enumerate(docs):
                            print(f"{i+1}. [{score:.4f}] Source: {source} (Chunk ID: {chunk_id})")
                            print(f"   Contenu: {doc}") # Affichage complet
                            print("-" * 20) # Séparateur
                            
                        print(f"\nRéponse:\n{response}")
                except Exception as e:
                    print(f"Erreur: {e}")

    # Mode non interactif
    elif args.query:
        try:
            # Déterminer si la génération est possible
                generation_possible = rag_system.use_openai or (TRANSFORMERS_AVAILABLE and rag_system.llm is not None)

                # Mode recherche seulement ou si la génération n'est pas possible
                if args.search_only or not generation_possible:
                    if not generation_possible and not args.search_only:
                         print("\nAvertissement: Génération impossible (LLM non chargé ou non configuré). Affichage des résultats de recherche seulement.")
                    
                    docs = rag_system.retrieve(query, top_k=args.top_k)
                    
                    print("\nDocuments pertinents:")
                    for i, (doc, source, score, chunk_id) in enumerate(docs):
                        print(f"{i+1}. [{score:.4f}] Source: {source} (Chunk ID: {chunk_id})")
                        print(f"   Contenu: {doc}") # Affichage complet
                        print("-" * 20) # Séparateur
                # Mode RAG complet
                else:
                    response, docs = rag_system.generate(query, top_k=args.top_k)
                    
                    print("\nDocuments pertinents utilisés pour la réponse:")
                    for i, (doc, source, score, chunk_id) in enumerate(docs):
                        print(f"{i+1}. [{score:.4f}] Source: {source} (Chunk ID: {chunk_id})")
                        print(f"   Contenu: {doc}") # Affichage complet
                        print("-" * 20) # Séparateur
                        
                    print(f"\nRéponse:\n{response}")
            except Exception as e:
                print(f"Erreur: {e}")
    
    # Mode non interactif
    elif args.query:
        try:
            # Déterminer si la génération est possible
            generation_possible = rag_system.use_openai or (TRANSFORMERS_AVAILABLE and rag_system.llm is not None)

            # Mode recherche seulement ou si la génération n'est pas possible
            if args.search_only or not generation_possible:
                if not generation_possible and not args.search_only:
                    print("\nAvertissement: Génération impossible (LLM non chargé ou non configuré). Affichage des résultats de recherche seulement.")
                
                docs = rag_system.retrieve(args.query, top_k=args.top_k)
                
                print("\nDocuments pertinents:")
                for i, (doc, source, score, chunk_id) in enumerate(docs):
                    print(f"{i+1}. [{score:.4f}] Source: {source} (Chunk ID: {chunk_id})")
                    print(f"   Contenu: {doc}") # Affichage complet
                    print("-" * 20) # Séparateur
            # Mode RAG complet
            else:
                response, docs = rag_system.generate(args.query, top_k=args.top_k)
                
                print("\nDocuments pertinents utilisés pour la réponse:")
                for i, (doc, source, score, chunk_id) in enumerate(docs):
                    print(f"{i+1}. [{score:.4f}] Source: {source} (Chunk ID: {chunk_id})")
                    print(f"   Contenu: {doc}") # Affichage complet
                    print("-" * 20) # Séparateur
                    
                print(f"\nRéponse:\n{response}")
        except Exception as e:
            print(f"Erreur: {e}")
    
    else:
        print("Veuillez spécifier une requête ou utiliser le mode interactif.")


if __name__ == "__main__":
    main()
