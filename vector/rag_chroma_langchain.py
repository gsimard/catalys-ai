import sys
import os
import argparse
import traceback
from dotenv import load_dotenv
import torch
import chromadb

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory # For potential history management
from langchain_openai import ChatOpenAI # Wrapper for OpenAI compatible APIs
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline # Wrapper for local HF models

# Transformers for local LLM loading (if needed)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Le package transformers n'est pas installé. La génération avec un LLM local ne sera pas disponible.", file=sys.stderr)

# Prompt toolkit for better interactive experience
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False
    print("Le package prompt_toolkit n'est pas installé. L'historique et la navigation avancée en mode interactif ne seront pas disponibles.", file=sys.stderr)

# Load environment variables from .env file
load_dotenv()

# --- Helper Functions ---

def detect_device(requested_device):
    """Detects the appropriate device (CPU or CUDA)."""
    if requested_device:
        return requested_device
    return "cuda" if torch.cuda.is_available() else "cpu"

def format_docs(docs):
    """Formats retrieved documents into a single string."""
    return "\n\n".join(f"Source: {doc.metadata.get('source', 'N/A')}\n{doc.page_content}" for doc in docs)

def load_embedding_model(model_name, device):
    """Loads the HuggingFace BGE embedding model."""
    print(f"Chargement du modèle d'embedding {model_name} sur {device}...")
    try:
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True} # Crucial for BGE
        )
        print("Modèle d'embedding chargé.")
        return embeddings
    except Exception as e:
        print(f"Erreur lors du chargement du modèle d'embedding: {e}", file=sys.stderr)
        raise

def connect_chromadb(host, port, collection_name, embedding_function):
    """Connects to ChromaDB and returns the VectorStore."""
    print(f"Connexion à ChromaDB ({host}:{port}), collection '{collection_name}'...")
    try:
        chroma_client = chromadb.HttpClient(host=host, port=port)
        chroma_client.heartbeat() # Test connection

        vector_store = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embedding_function,
        )
        print(f"Connecté à la collection ChromaDB '{collection_name}'.")
        # Quick check
        count = vector_store._collection.count()
        if count == 0:
            print(f"Attention: La collection '{collection_name}' est vide.", file=sys.stderr)
        else:
            print(f"La collection contient {count} documents.")
        return vector_store
    except Exception as e:
        print(f"Erreur de connexion à ChromaDB: {e}", file=sys.stderr)
        print("Vérifiez que le serveur ChromaDB est lancé et que la collection existe.", file=sys.stderr)
        raise

def load_llm(use_openai, openai_config, local_llm_config, device):
    """Loads the appropriate LLM (OpenAI compatible API or local HuggingFace)."""
    if use_openai:
        print(f"Configuration de l'API externe (type OpenAI) : {openai_config['base_url']} (Modèle: {openai_config['model']})")
        try:
            llm = ChatOpenAI(
                openai_api_base=openai_config['base_url'],
                openai_api_key=openai_config['api_key'],
                model=openai_config['model'],
                temperature=0.7, # Example temperature
                # max_tokens=openai_config.get('max_tokens', 1024) # Can be set here or in chain
            )
            print("Client API externe initialisé.")
            return llm
        except Exception as e:
            print(f"Erreur lors de l'initialisation du client API externe: {e}", file=sys.stderr)
            print("Retour à la tentative d'utilisation du LLM local si possible.", file=sys.stderr)
            # Fall through to local LLM loading if API fails

    # Try loading local LLM if not using OpenAI or if OpenAI failed
    if not TRANSFORMERS_AVAILABLE:
        print("Transformers non disponible, impossible de charger un LLM local.", file=sys.stderr)
        return None

    print(f"Tentative de chargement du LLM local '{local_llm_config['model_id']}' sur {device}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(local_llm_config['model_id'])
        # Load model on the correct device
        model = AutoModelForCausalLM.from_pretrained(local_llm_config['model_id']).to(device)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=local_llm_config.get('max_new_tokens', 512), # Max tokens to generate
            device=0 if device.startswith("cuda") else -1 # pipeline expects device index or -1 for CPU
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        print(f"LLM local chargé avec succès sur {device}!")
        return llm
    except Exception as e:
        print(f"Erreur lors du chargement du LLM local '{local_llm_config['model_id']}': {e}", file=sys.stderr)
        traceback.print_exc()
        return None

# --- Main Application Logic ---

def main():
    parser = argparse.ArgumentParser(description="Système RAG avec LangChain et ChromaDB")
    # ChromaDB args
    parser.add_argument("--chroma-host", type=str, default=os.getenv("CHROMA_HOST", "localhost"),
                        help="Adresse de l'hôte ChromaDB (ou variable d'env CHROMA_HOST)")
    parser.add_argument("--chroma-port", type=int, default=int(os.getenv("CHROMA_PORT", 8000)),
                        help="Port du serveur ChromaDB (ou variable d'env CHROMA_PORT)")
    parser.add_argument("--collection", type=str, default=os.getenv("CHROMA_COLLECTION", "rag_collection"),
                        help="Nom de la collection ChromaDB (ou variable d'env CHROMA_COLLECTION)")
    # Embedding model args
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-m3",
                        help="Modèle d'embedding HuggingFace à utiliser")
    # LLM args
    parser.add_argument("--llm", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help="ID du modèle LLM local HuggingFace (si API externe non utilisée ou forcée)")
    parser.add_argument("--force-local-llm", action="store_true",
                        help="Forcer l'utilisation du LLM local même si une API externe est configurée via .env")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Nombre maximum de nouveaux tokens à générer par le LLM")
    # General args
    parser.add_argument("--device", type=str, default=None,
                        help="Appareil à utiliser pour les modèles locaux (cpu, cuda, cuda:0, etc.) - auto-détecté si non fourni")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Nombre de documents à récupérer de ChromaDB")
    parser.add_argument("--interactive", action="store_true",
                        help="Lancer en mode interactif")
    parser.add_argument("--query", type=str,
                        help="Requête à traiter (mode non interactif)")
    parser.add_argument("--debug", action="store_true",
                        help="Activer l'affichage de débogage (contexte complet, etc.)")

    args = parser.parse_args()

    # --- Configuration ---
    effective_device = detect_device(args.device)
    print(f"Utilisation du device: {effective_device}")

    # OpenAI API Config from .env
    nebius_api_base = os.getenv("NEBIUS_API_BASE")
    nebius_api_key = os.getenv("NEBIUS_API_KEY")
    nebius_model = os.getenv("NEBIUS_AI_MODEL")
    use_openai_api = nebius_api_base and nebius_api_key and nebius_model and not args.force_local_llm

    openai_config = {
        "base_url": nebius_api_base,
        "api_key": nebius_api_key,
        "model": nebius_model,
        "max_tokens": args.max_new_tokens
    }
    local_llm_config = {
        "model_id": args.llm,
        "max_new_tokens": args.max_new_tokens
    }

    # --- Initialisation des composants ---
    try:
        # 1. Embedding Model
        embedding_function = load_embedding_model(args.embedding_model, effective_device)

        # 2. ChromaDB VectorStore
        vector_store = connect_chromadb(args.chroma_host, args.chroma_port, args.collection, embedding_function)

        # 3. Retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": args.top_k})
        print(f"Retriever ChromaDB configuré pour récupérer {args.top_k} documents.")

        # 4. LLM
        llm = load_llm(use_openai_api, openai_config, local_llm_config, effective_device)
        if llm is None:
            print("Erreur: Aucun LLM n'a pu être chargé (ni API externe, ni local).", file=sys.stderr)
            print("La génération de réponse est impossible. Vous pouvez utiliser ce script pour la recherche seulement.", file=sys.stderr)
            # Allow continuing for retrieval-only tasks if needed, but generation will fail.
            # Consider adding a --search-only flag if this is a desired mode.
            # For now, we exit if interactive/query mode needs generation.
            if args.interactive or args.query:
                 print("Arrêt du programme.", file=sys.stderr)
                 return 1 # Exit with error code

    except Exception as e:
        print(f"\nErreur lors de l'initialisation: {e}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 1 # Exit with error code

    # --- Définition de la chaîne RAG avec LangChain Expression Language (LCEL) ---

    # Template de prompt
    # Adaptez ce template selon le modèle LLM utilisé (certains préfèrent des formats spécifiques)
    template = """Tu es un assistant IA serviable. Réponds à la question en te basant UNIQUEMENT sur le contexte suivant:
{context}

Question: {question}

Réponse:"""
    prompt = ChatPromptTemplate.from_template(template)

    # Définition de la chaîne principale
    # Cette chaîne prend une question en entrée ("question")
    # 1. Récupère les documents pertinents (`retriever`)
    # 2. Formate les documents récupérés en une seule chaîne (`format_docs`)
    # 3. Prépare les entrées pour le prompt (contexte formaté, question originale)
    # 4. Applique le template de prompt
    # 5. Appelle le LLM
    # 6. Parse la sortie du LLM en chaîne de caractères

    # Chaîne pour formater et appeler le LLM une fois le contexte et la question prêts
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    # Chaîne complète qui inclut la récupération et passe les sources
    # RunnableParallel permet d'exécuter le retriever et de passer la question en parallèle
    # .assign(answer=...) ajoute la sortie de rag_chain_from_docs au dictionnaire final
    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    # --- Exécution ---

    def process_query(query):
        if not query or not query.strip():
            return

        print("\n--- Recherche et Génération ---")
        start_time = time.time() # Simple timer

        try:
            # Invoquer la chaîne RAG
            result = rag_chain_with_source.invoke(query)
            end_time = time.time()

            print(f"\nRéponse (générée en {end_time - start_time:.2f}s):")
            print(result["answer"])

            print("\nDocuments sources utilisés:")
            if result["context"]:
                for i, doc in enumerate(result["context"]):
                    source = doc.metadata.get('source', 'Inconnue')
                    chunk_id = doc.metadata.get('chunk_id', 'N/A')
                    # score = doc.metadata.get('score', None) # Chroma might not add score by default here
                    print(f"{i+1}. Source: {source} (Chunk ID: {chunk_id})")
                    if args.debug:
                        print(f"   Contenu: {doc.page_content[:250]}...") # Show beginning of chunk if debug
                        print("-" * 20)
            else:
                print("Aucun document pertinent trouvé dans la base.")

        except Exception as e:
            print(f"\nErreur lors du traitement de la requête: {e}", file=sys.stderr)
            if args.debug:
                traceback.print_exc()

    # Mode interactif
    if args.interactive:
        print("\nMode interactif. Tapez 'exit' ou Ctrl+D pour quitter.")
        if PROMPT_TOOLKIT_AVAILABLE:
            session = PromptSession(history=FileHistory('.rag_chroma_history'))
            while True:
                try:
                    user_input = session.prompt("\nQuestion: ")
                    if user_input.lower() == 'exit':
                        break
                    process_query(user_input)
                except (EOFError, KeyboardInterrupt):
                    break
        else: # Fallback basic input
            while True:
                try:
                    user_input = input("\nQuestion: ")
                    if user_input.lower() == 'exit':
                        break
                    process_query(user_input)
                except EOFError:
                    break
        print("\nFin du mode interactif.")

    # Mode non interactif (requête unique)
    elif args.query:
        process_query(args.query)

    else:
        print("Veuillez fournir une requête avec --query ou utiliser le mode --interactive.", file=sys.stderr)
        parser.print_help()
        return 1

    return 0 # Exit successfully

if __name__ == "__main__":
    # Add basic time import for process_query timer
    import time
    sys.exit(main())
