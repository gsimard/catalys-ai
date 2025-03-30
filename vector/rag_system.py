import os
import numpy as np
import json
import pickle # Ajout de pickle
import argparse
import threading
import time
import sys
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
    def __init__(self, embedding_model="BAAI/bge-m3", llm_model="meta-llama/Llama-2-7b-chat-hf", device=None, force_local_llm=False, debug=False):
        """
        Initialise le système RAG avec un modèle d'embedding et un LLM.
        Priorise l'utilisation d'un endpoint externe (type OpenAI) si configuré via .env,
        sauf si force_local_llm est True.

        Args:
            embedding_model: Modèle d'embedding à utiliser.
            llm_model: Modèle de langage local à utiliser (si l'endpoint externe n'est pas utilisé).
            device: Appareil sur lequel exécuter les modèles locaux.
            force_local_llm: Si True, force l'utilisation du LLM local même si un endpoint externe est configuré.
            debug: Si True, active les messages de débogage.
        """
        self.debug = debug # Stocker l'état de débogage
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
        self.use_openai = False # Sera mis à True si l'endpoint externe est utilisé

        # Déterminer quel LLM utiliser
        nebius_api_base = os.getenv("NEBIUS_API_BASE")
        nebius_api_key = os.getenv("NEBIUS_API_KEY")
        nebius_model = os.getenv("NEBIUS_AI_MODEL")

        # Condition pour utiliser l'endpoint externe : variables définies ET on ne force PAS le local
        should_use_external = nebius_api_base and nebius_api_key and nebius_model and not force_local_llm

        if should_use_external:
            print(f"Configuration de l'endpoint externe (type OpenAI) détectée: {nebius_api_base}")
            try:
                self.openai_client = openai.OpenAI(
                    base_url=nebius_api_base,
                    api_key=nebius_api_key,
                )
                self.openai_model = nebius_model
                self.use_openai = True # Marquer comme utilisant l'API externe
                print(f"Client externe (type OpenAI) initialisé pour le modèle: {self.openai_model}")
            except Exception as e:
                print(f"Erreur lors de l'initialisation du client externe: {e}")
                print("Retour à la tentative d'utilisation du LLM local si possible.")
                self.use_openai = False # Échec de l'initialisation externe

        # Si on n'utilise PAS l'endpoint externe (soit par échec, soit par choix --force-local-llm, soit non configuré)
        if not self.use_openai:
            if force_local_llm:
                print("Utilisation forcée du LLM local.")
            elif not (nebius_api_base and nebius_api_key and nebius_model):
                print("Endpoint externe non configuré via les variables d'environnement.")
            
            if TRANSFORMERS_AVAILABLE:
                print(f"Tentative de chargement du LLM local '{llm_model}' sur {self.device}...")
                try:
                    self.tokenizer_llm = AutoTokenizer.from_pretrained(llm_model)
                    # Charger le modèle sur le CPU si CUDA n'est pas disponible ou si device est 'cpu'
                    model_device = self.device if self.device.startswith("cuda") else "cpu" 
                    self.llm = AutoModelForCausalLM.from_pretrained(llm_model).to(model_device)
                    print(f"LLM local chargé avec succès sur {model_device}!")
                except Exception as e:
                    print(f"Erreur lors du chargement du LLM local: {e}")
                    print("La génération de réponses avec le LLM local ne sera pas disponible.")
            else:
                 print("Le package 'transformers' n'est pas installé. Impossible de charger un LLM local.")
                 if not self.use_openai: # Si l'externe n'a pas été chargé non plus
                     print("AVERTISSEMENT: Aucun LLM (externe ou local) n'est disponible. La génération de réponses est impossible.")

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

    def load_knowledge_base_from_pickle(self, pkl_file):
        """
        Charge une base de connaissances préparée au format Pickle.

        Args:
            pkl_file: Chemin vers le fichier Pickle (.pkl)
        """
        print(f"Chargement de la base de connaissances depuis {pkl_file}...")

        try:
            with open(pkl_file, 'rb') as f: # Utiliser 'rb' pour le binaire
                documents = pickle.load(f)
        except FileNotFoundError:
            print(f"Erreur: Le fichier {pkl_file} n'a pas été trouvé.")
            raise
        except Exception as e:
            print(f"Erreur lors du chargement du fichier pickle {pkl_file}: {e}")
            raise

        if not documents:
            print("Attention: Le fichier de base de connaissances est vide.")
            self.kb_texts = []
            self.kb_sources = []
            self.kb_chunk_ids = []
            self.kb_embeddings = np.array([])
            return

        # Extraire les données
        self.kb_texts = [doc.get("content", "") for doc in documents]
        self.kb_sources = [doc.get("source", "Inconnue") for doc in documents]
        self.kb_chunk_ids = [doc.get("chunk_id", -1) for doc in documents]

        # Les embeddings sont déjà des tableaux numpy (probablement F16)
        # Il suffit de les empiler dans un seul grand tableau numpy
        # S'assurer qu'ils sont bien présents
        if "embedding" in documents[0] and documents[0]["embedding"] is not None:
             # Convertir en F32 pour les calculs de similarité si nécessaire,
             # bien que numpy gère les opérations mixtes F16/F32.
             # Gardons-les en F16 pour l'instant pour économiser la RAM.
            self.kb_embeddings = np.array([doc["embedding"] for doc in documents])
            print(f"Embeddings chargés depuis le fichier PKL pour {len(self.kb_texts)} documents (dtype: {self.kb_embeddings.dtype}).")
        else:
            # Fallback si les embeddings manquent (ne devrait pas arriver avec prepare_kb.py)
            print(f"Attention: Embeddings non trouvés dans {pkl_file}. Regénération...")
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
        
        print("Recherche des documents pertinents...")
        from tqdm import tqdm
        
        # Calcul des similarités avec tous les documents avec barre de progression
        # Diviser le calcul en batches pour montrer la progression
        batch_size = 1000  # Taille de batch adaptée
        similarities = np.zeros(len(self.kb_embeddings))
        
        for i in tqdm(range(0, len(self.kb_embeddings), batch_size), 
                     desc="Recherche de documents", unit="batch"):
            end_idx = min(i + batch_size, len(self.kb_embeddings))
            batch_embeddings = self.kb_embeddings[i:end_idx]
            similarities[i:end_idx] = np.dot(batch_embeddings, query_embedding)
        
        # Récupération des indices des documents les plus similaires
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        # Récupération des documents, sources, scores et chunk_ids
        top_docs = [
            (self.kb_texts[i], self.kb_sources[i], similarities[i], self.kb_chunk_ids[i]) 
            for i in top_indices
        ]
        
        return top_docs
        
    def generate(self, query, top_k=3, max_length=1024):
        """
        Génère une réponse à une requête en utilisant le RAG.
        
        Args:
            query: Requête de l'utilisateur
            top_k: Nombre de documents à récupérer
            max_length: Longueur maximale de la réponse (en tokens)
            
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

        if self.debug:
            print(f"\n--- Prompt envoyé au LLM ---\n{prompt}\n---------------------------\n")

        response = "Impossible de générer une réponse."

        # Fonction pour animer un curseur tournant
        def spinning_cursor():
            spinner = ['|', '/', '-', '\\']
            i = 0
            while True:
                sys.stdout.write('\r' + spinner[i % len(spinner)])
                sys.stdout.flush()
                i += 1
                time.sleep(0.1)

        if self.use_openai and self.openai_client:
            # Génération via API OpenAI
            try:
                print("Génération via l'API OpenAI...")
                
                # Démarrer l'animation du curseur dans un thread séparé
                stop_spinner = threading.Event()
                spinner_thread = threading.Thread(target=lambda: (
                    spinning_cursor() if not stop_spinner.is_set() else None
                ))
                spinner_thread.daemon = True  # Le thread s'arrêtera quand le programme principal se termine
                spinner_thread.start()
                
                try:
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
                finally:
                    # Arrêter l'animation du curseur
                    stop_spinner.set()
                    sys.stdout.write('\r')  # Effacer le curseur
                    sys.stdout.flush()
                    print("Réponse reçue!                ")  # Espaces pour effacer tout reste du spinner
            except Exception as e:
                print(f"\rErreur lors de l'appel à l'API OpenAI: {e}")
                response = f"Erreur lors de la génération via API: {e}"

        elif self.llm and self.tokenizer_llm:
            # Génération via LLM local
            print("Génération via le LLM local...")
            
            # Démarrer l'animation du curseur dans un thread séparé
            stop_spinner = threading.Event()
            spinner_thread = threading.Thread(target=lambda: (
                spinning_cursor() if not stop_spinner.is_set() else None
            ))
            spinner_thread.daemon = True
            spinner_thread.start()
            
            try:
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
            
            finally:
                # Arrêter l'animation du curseur
                stop_spinner.set()
                sys.stdout.write('\r')  # Effacer le curseur
                sys.stdout.flush()
                print("Réponse reçue!                ")  # Espaces pour effacer tout reste du spinner
            
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
                        help="Appareil à utiliser pour les modèles locaux (embeddings, LLM local) (cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--llm", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help="Modèle de langage local à utiliser (si l'endpoint externe n'est pas utilisé ou si --force-local-llm est activé)")
    parser.add_argument("--force-local-llm", action="store_true",
                        help="Forcer l'utilisation du LLM local même si un endpoint externe est configuré via les variables d'environnement")
    parser.add_argument("--interactive", action="store_true",
                        help="Mode interactif")
    parser.add_argument("--query", type=str,
                        help="Requête à traiter (mode non interactif)")
    parser.add_argument("--search-only", action="store_true",
                        help="Mode recherche seulement (sans génération)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Nombre de documents à récupérer")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Longueur maximale de la réponse générée (en tokens)")
    parser.add_argument("--debug", action="store_true",
                        help="Activer l'affichage de débogage (chunks complets, prompt LLM)")
    
    args = parser.parse_args()

    # Initialisation du système RAG
    rag_system = RAGSystem(llm_model=args.llm, device=args.device, force_local_llm=args.force_local_llm, debug=args.debug)

    # Chargement de la base de connaissances
    if args.kb:
        # Priorité au chargement depuis un fichier préparé (PKL)
        if args.kb.lower().endswith('.pkl'):
            try:
                rag_system.load_knowledge_base_from_pickle(args.kb)
            except Exception as e:
                print(f"Impossible de charger la base de connaissances depuis {args.kb}. Erreur: {e}")
                return # Arrêter si le chargement échoue
        # Sinon, essayer de charger depuis un répertoire ou un fichier texte simple
        elif os.path.isdir(args.kb):
            # Note: load_knowledge_base_from_directory génère les embeddings à la volée
            rag_system.load_knowledge_base_from_directory(args.kb)
        elif os.path.isfile(args.kb):
            # Note: load_knowledge_base génère les embeddings à la volée
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
                            if args.debug:
                                print(f"   Contenu: {doc}") # Affichage complet si debug
                                print("-" * 20) # Séparateur si debug
                    # Mode RAG complet
                    else:
                        response, docs = rag_system.generate(query, top_k=args.top_k)
                        
                        print("\nDocuments pertinents utilisés pour la réponse:")
                        for i, (doc, source, score, chunk_id) in enumerate(docs):
                            print(f"{i+1}. [{score:.4f}] Source: {source} (Chunk ID: {chunk_id})")
                            if args.debug:
                                print(f"   Contenu: {doc}") # Affichage complet si debug
                                print("-" * 20) # Séparateur si debug
                            
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
                            if args.debug:
                                print(f"   Contenu: {doc}") # Affichage complet si debug
                                print("-" * 20) # Séparateur si debug
                    # Mode RAG complet
                    else:
                        response, docs = rag_system.generate(query, top_k=args.top_k)
                        
                        print("\nDocuments pertinents utilisés pour la réponse:")
                        for i, (doc, source, score, chunk_id) in enumerate(docs):
                            print(f"{i+1}. [{score:.4f}] Source: {source} (Chunk ID: {chunk_id})")
                            if args.debug:
                                print(f"   Contenu: {doc}") # Affichage complet si debug
                                print("-" * 20) # Séparateur si debug
                            
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
                        if args.debug:
                            print(f"   Contenu: {doc}") # Affichage complet si debug
                            print("-" * 20) # Séparateur si debug
                # Mode RAG complet
                else:
                    response, docs = rag_system.generate(query, top_k=args.top_k, max_length=args.max_length)
                    
                    print("\nDocuments pertinents utilisés pour la réponse:")
                    for i, (doc, source, score, chunk_id) in enumerate(docs):
                        print(f"{i+1}. [{score:.4f}] Source: {source} (Chunk ID: {chunk_id})")
                        if args.debug:
                            print(f"   Contenu: {doc}") # Affichage complet si debug
                            print("-" * 20) # Séparateur si debug
                        
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
                    if args.debug:
                        print(f"   Contenu: {doc}") # Affichage complet si debug
                        print("-" * 20) # Séparateur si debug
            # Mode RAG complet
            else:
                response, docs = rag_system.generate(args.query, top_k=args.top_k, max_length=args.max_length)
                
                print("\nDocuments pertinents utilisés pour la réponse:")
                for i, (doc, source, score, chunk_id) in enumerate(docs):
                    print(f"{i+1}. [{score:.4f}] Source: {source} (Chunk ID: {chunk_id})")
                    if args.debug:
                        print(f"   Contenu: {doc}") # Affichage complet si debug
                        print("-" * 20) # Séparateur si debug
                    
                print(f"\nRéponse:\n{response}")
        except Exception as e:
            print(f"Erreur: {e}")
    
    else:
        print("Veuillez spécifier une requête ou utiliser le mode interactif.")


if __name__ == "__main__":
    main()
