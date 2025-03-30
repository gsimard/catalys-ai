import os
import argparse
import re
import traceback
from tqdm import tqdm

# LangChain and ChromaDB imports
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import chromadb

# --- Text Extraction Logic (Copied and adapted from original prepare_kb.py) ---

# Vérification de la disponibilité de PyPDF2
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("PyPDF2 n'est pas installé. L'extraction de texte des PDF sera limitée.")

# Vérification de la disponibilité de pdfminer.six
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False
    print("pdfminer.six n'est pas installé. L'extraction de texte des PDF sera limitée.")

# Vérification de la disponibilité de pytesseract et pdf2image pour l'OCR
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("pytesseract ou pdf2image n'est pas installé. L'OCR pour les PDF scannés ne sera pas disponible.")


def extract_text_from_pdf(pdf_path):
    """
    Extrait le texte d'un fichier PDF en utilisant plusieurs méthodes.
    (Copied from original prepare_kb.py)
    """
    text = ""
    errors = []

    # Méthode 1: PyPDF2 (si disponible)
    if PYPDF2_AVAILABLE:
        try:
            text_pypdf = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                if pdf_reader.is_encrypted:
                    print(f"Le PDF {pdf_path} est crypté, tentative de déchiffrement...")
                    try:
                        success = pdf_reader.decrypt('')
                        if not success:
                            print(f"Impossible de déchiffrer le PDF {pdf_path}")
                    except:
                        print(f"Erreur lors de la tentative de déchiffrement du PDF {pdf_path}")

                for page_num in range(len(pdf_reader.pages)):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            text_pypdf += page_text + "\n\n"
                    except Exception as e:
                        print(f"Erreur lors de l'extraction de la page {page_num} de {pdf_path}: {str(e)}")

            if text_pypdf.strip():
                text = text_pypdf
                print(f"Texte extrait avec succès de {pdf_path} en utilisant PyPDF2")
                return clean_text(text)
        except Exception as e:
            errors.append(f"PyPDF2: {str(e)}")

    # Méthode 2: pdfminer.six (si disponible)
    if PDFMINER_AVAILABLE and not text.strip():
        try:
            text_pdfminer = pdfminer_extract_text(pdf_path)
            if text_pdfminer.strip():
                text = text_pdfminer
                print(f"Texte extrait avec succès de {pdf_path} en utilisant pdfminer.six")
                return clean_text(text)
        except Exception as e:
            errors.append(f"pdfminer.six: {str(e)}")

    # Méthode 3: OCR (si disponible et nécessaire)
    if OCR_AVAILABLE and not text.strip():
        try:
            text_ocr = extract_text_from_scanned_pdf(pdf_path)
            if text_ocr.strip():
                text = text_ocr
                print(f"Texte extrait avec succès de {pdf_path} en utilisant OCR")
                return clean_text(text)
        except Exception as e:
            errors.append(f"OCR: {str(e)}")

    # Si aucune méthode n'a fonctionné
    if not text.strip():
        error_msg = "Aucune méthode d'extraction n'a réussi à extraire du texte."
        if errors:
            error_msg += f" Erreurs: {'; '.join(errors)}"
        raise Exception(error_msg)

    return clean_text(text)


def extract_text_from_scanned_pdf(pdf_path):
    """
    Extrait le texte d'un PDF scanné en utilisant OCR.
    (Copied from original prepare_kb.py)
    """
    if not OCR_AVAILABLE:
        return ""

    import tempfile

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            images = convert_from_path(pdf_path)
            text = ""
            for i, image in enumerate(images):
                image_path = os.path.join(temp_dir, f'page_{i}.png')
                image.save(image_path, 'PNG')
                page_text = pytesseract.image_to_string(image_path, lang='fra+eng')
                text += page_text + "\n\n"
            return text
    except Exception as e:
        print(f"Erreur lors de l'OCR sur {pdf_path}: {e}")
        return ""


def clean_text(text):
    """
    Nettoie le texte extrait.
    (Copied from original prepare_kb.py)
    """
    text = re.sub(r'\s+', ' ', text)
    text = ''.join(c for c in text if c.isprintable() or c in ['\n', '\t'])
    return text.strip()


def chunk_text(text, chunk_size=512, overlap=50):
    """
    Découpe un texte en chunks de taille fixe avec chevauchement (version générateur).
    (Copied from original prepare_kb.py)
    """
    if not text:
        return

    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            split_pos = -1
            last_newline = text.rfind('\n', max(0, end - overlap), end)
            last_space = text.rfind(' ', max(0, end - overlap), end)
            split_pos = max(last_newline, last_space)

            if split_pos != -1 and split_pos > start:
                end = split_pos

        chunk = text[start:end].strip()
        if chunk:
            yield chunk

        next_start = start + chunk_size - overlap
        if next_start <= start:
             if end == start + chunk_size:
                 next_start = end
             else:
                 next_start = start + 1

        start = next_start
        if start >= text_len:
             break

# --- End of Text Extraction Logic ---


def process_file(file_path, chunk_size=512, overlap=50):
    """
    Traite un fichier, extrait le texte et le découpe en LangChain Documents.

    Args:
        file_path: Chemin vers le fichier
        chunk_size: Taille des chunks
        overlap: Chevauchement entre les chunks

    Returns:
        Liste d'objets LangChain Document
    """
    try:
        content = ""
        if file_path.lower().endswith('.pdf'):
            if not PYPDF2_AVAILABLE and not PDFMINER_AVAILABLE and not OCR_AVAILABLE:
                print(f"Impossible de traiter le PDF {file_path}. Manque PyPDF2/pdfminer/OCR.")
                return []
            try:
                content = extract_text_from_pdf(file_path)
                if not content.strip():
                    print(f"Attention: Aucun texte n'a pu être extrait de {file_path}")
                    return []
            except Exception as e:
                print(f"Erreur lors de l'extraction du texte de {file_path}: {str(e)}")
                return []
        else:
            # Pour les fichiers texte
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                except Exception as e:
                    print(f"Erreur lors de la lecture de {file_path} avec encodage latin-1: {str(e)}")
                    return []
            except Exception as e:
                print(f"Erreur lors de la lecture de {file_path}: {str(e)}")
                return []

        if not content.strip():
            print(f"Attention: {file_path} ne contient pas de texte exploitable.")
            return []

        # Découper en chunks et créer des Documents LangChain
        docs = []
        for i, chunk_content in enumerate(chunk_text(content, chunk_size, overlap)):
            doc = Document(
                page_content=chunk_content,
                metadata={
                    "source": file_path,
                    "chunk_id": i
                }
            )
            docs.append(doc)

        if not docs:
            print(f"Attention: Aucun chunk n'a pu être créé à partir de {file_path}")
        return docs

    except Exception as e:
        print(f"Erreur grave lors du traitement de {file_path}: {str(e)}")
        traceback.print_exc()
        return []


def main():
    parser = argparse.ArgumentParser(description="Préparation de la base de connaissances pour ChromaDB avec LangChain")
    parser.add_argument("--input", type=str, required=True,
                        help="Fichier ou répertoire d'entrée")
    parser.add_argument("--chroma-host", type=str, default="localhost",
                        help="Adresse de l'hôte ChromaDB")
    parser.add_argument("--chroma-port", type=int, default=8000,
                        help="Port du serveur ChromaDB")
    parser.add_argument("--collection", type=str, default="rag_collection",
                        help="Nom de la collection ChromaDB")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-m3",
                        help="Nom du modèle d'embedding HuggingFace (ex: BAAI/bge-m3)")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Taille des chunks (en caractères)")
    parser.add_argument("--overlap", type=int, default=50,
                        help="Chevauchement entre les chunks")
    parser.add_argument("--device", type=str, default=None,
                        help="Appareil à utiliser pour les embeddings (cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--extensions", type=str, default=".txt,.md,.pdf",
                        help="Extensions de fichiers à traiter (séparées par des virgules)")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Nombre de documents à ajouter à ChromaDB par batch")
    parser.add_argument("--debug", action="store_true",
                        help="Mode débogage avec plus d'informations")

    args = parser.parse_args()

    # Mode débogage
    if args.debug:
        print("Mode débogage activé")
        # (La fonction process_file a déjà la gestion d'erreur avec traceback)

    # Initialisation de la fonction d'embedding LangChain
    print(f"Initialisation du modèle d'embedding {args.embedding_model} sur device '{args.device}'...")
    try:
        embedding_function = HuggingFaceBgeEmbeddings(
            model_name=args.embedding_model,
            model_kwargs={'device': args.device},
            encode_kwargs={'normalize_embeddings': True} # Normalisation L2 importante pour BGE
        )
        print("Modèle d'embedding initialisé.")
    except Exception as e:
        print(f"Erreur lors de l'initialisation du modèle d'embedding: {e}")
        return

    # Initialisation du client ChromaDB
    print(f"Connexion au client ChromaDB à {args.chroma_host}:{args.chroma_port}...")
    try:
        chroma_client = chromadb.HttpClient(host=args.chroma_host, port=args.chroma_port)
        # Test de connexion simple
        chroma_client.heartbeat()
        print("Client ChromaDB connecté.")
    except Exception as e:
        print(f"Erreur de connexion au client ChromaDB: {e}")
        print("Assurez-vous que le serveur ChromaDB est lancé et accessible.")
        print("Vous pouvez le lancer avec: chroma run --path /chemin/vers/db --host localhost --port 8000")
        return

    # Conversion des extensions
    extensions = args.extensions.split(',')

    # Collecte et traitement des fichiers
    all_docs = []
    files_to_process = []

    if os.path.isdir(args.input):
        print(f"Scan du répertoire {args.input}...")
        for root, _, files in os.walk(args.input):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    files_to_process.append(os.path.join(root, file))
    elif os.path.isfile(args.input):
         if any(args.input.endswith(ext) for ext in extensions):
             files_to_process.append(args.input)
         else:
             print(f"Le fichier {args.input} n'a pas une extension supportée ({args.extensions}).")
             return
    else:
        print(f"Erreur: Le chemin d'entrée {args.input} n'est ni un fichier ni un répertoire valide.")
        return

    if not files_to_process:
        print("Aucun fichier à traiter trouvé.")
        return

    print(f"Traitement de {len(files_to_process)} fichiers...")
    for file_path in tqdm(files_to_process, desc="Traitement des fichiers"):
        docs = process_file(file_path, args.chunk_size, args.overlap)
        all_docs.extend(docs)

    print(f"Nombre total de documents (chunks) générés: {len(all_docs)}")

    if not all_docs:
        print("Aucun document n'a pu être généré. Vérifiez les erreurs ci-dessus.")
        return

    # Ajout des documents à ChromaDB par batches
    print(f"Ajout des documents à la collection ChromaDB '{args.collection}'...")
    # Utilisation de LangChain Chroma wrapper pour gérer l'ajout et l'embedding
    try:
        vector_store = Chroma(
            client=chroma_client,
            collection_name=args.collection,
            embedding_function=embedding_function,
        )

        # Ajout par batches pour gérer la mémoire et afficher la progression
        num_batches = (len(all_docs) + args.batch_size - 1) // args.batch_size
        for i in tqdm(range(0, len(all_docs), args.batch_size), total=num_batches, desc="Ajout à ChromaDB"):
            batch = all_docs[i:i + args.batch_size]
            # Extraire les textes et métadonnées pour add_texts (plus robuste parfois)
            batch_texts = [doc.page_content for doc in batch]
            batch_metadatas = [doc.metadata for doc in batch]
            vector_store.add_texts(texts=batch_texts, metadatas=batch_metadatas)

        print(f"Documents ajoutés avec succès à la collection '{args.collection}'.")
        # Vérification rapide
        count = vector_store._collection.count()
        print(f"Nombre total d'éléments dans la collection '{args.collection}': {count}")

    except Exception as e:
        print(f"Erreur lors de l'ajout des documents à ChromaDB: {e}")
        if args.debug:
            traceback.print_exc()
        return

    print("Terminé!")


if __name__ == "__main__":
    main()
