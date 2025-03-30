import os
import argparse
import json
import re
from bge_embeddings import EmbeddingGenerator

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
    
    Args:
        pdf_path: Chemin vers le fichier PDF
        
    Returns:
        Texte extrait du PDF
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
                        # Essayer avec un mot de passe vide
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
    
    Args:
        pdf_path: Chemin vers le fichier PDF
        
    Returns:
        Texte extrait du PDF
    """
    if not OCR_AVAILABLE:
        return ""
    
    import tempfile
    
    try:
        # Création d'un dossier temporaire pour les images
        with tempfile.TemporaryDirectory() as temp_dir:
            # Conversion du PDF en images
            images = convert_from_path(pdf_path)
            
            text = ""
            for i, image in enumerate(images):
                # Sauvegarde de l'image
                image_path = os.path.join(temp_dir, f'page_{i}.png')
                image.save(image_path, 'PNG')
                
                # OCR sur l'image
                page_text = pytesseract.image_to_string(image_path, lang='fra+eng')
                text += page_text + "\n\n"
            
            return text
    except Exception as e:
        print(f"Erreur lors de l'OCR sur {pdf_path}: {e}")
        return ""


def clean_text(text):
    """
    Nettoie le texte extrait.
    
    Args:
        text: Texte à nettoyer
        
    Returns:
        Texte nettoyé
    """
    # Remplace les espaces multiples par un seul espace
    text = re.sub(r'\s+', ' ', text)
    # Supprime les caractères non imprimables
    text = ''.join(c for c in text if c.isprintable() or c in ['\n', '\t'])
    return text.strip()


def chunk_text(text, chunk_size=512, overlap=50):
    """
    Découpe un texte en chunks de taille fixe avec chevauchement.
    
    Args:
        text: Texte à découper
        chunk_size: Taille des chunks (en caractères)
        overlap: Chevauchement entre les chunks
        
    Returns:
        Liste des chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # Si ce n'est pas le dernier chunk, essayer de couper à un espace
        if end < len(text):
            # Chercher le dernier espace dans la fenêtre [end-50, end]
            last_space = text.rfind(' ', end - 50, end)
            if last_space != -1:
                end = last_space
        
        chunks.append(text[start:end].strip())
        start = end - overlap
    
    # Ensure the last part is included if it's smaller than overlap
    if start < len(text):
        chunks.append(text[start:].strip())
        
    return chunks


def chunk_text(text, chunk_size=512, overlap=50):
    """
    Découpe un texte en chunks de taille fixe avec chevauchement (version générateur).
    
    Args:
        text: Texte à découper
        chunk_size: Taille des chunks (en caractères)
        overlap: Chevauchement entre les chunks
        
    Yields:
        Chunks de texte
    """
    if not text: # Handle empty text case
        return

    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # Si ce n'est pas le dernier chunk, essayer de couper à un espace ou une nouvelle ligne
        # pour éviter de couper en plein milieu d'un mot.
        if end < text_len:
            # Chercher le dernier espace ou retour à la ligne dans la fenêtre [end-overlap, end]
            split_pos = -1
            # Prioritize newline, then space
            last_newline = text.rfind('\n', max(0, end - overlap), end)
            last_space = text.rfind(' ', max(0, end - overlap), end)
            
            split_pos = max(last_newline, last_space)

            # If a split point is found and it's after the start, use it
            if split_pos != -1 and split_pos > start:
                end = split_pos + 1 # Include the space/newline in the previous chunk for context? Or end = split_pos? Let's try end = split_pos
            # If no suitable split point found, just cut at chunk_size
            # (This part is implicitly handled by end = min(start + chunk_size, text_len))

        chunk = text[start:end].strip()
        if chunk: # Yield only non-empty chunks
            yield chunk
        
        # Calculer le prochain début
        next_start = start + chunk_size - overlap
        
        # Assurer la progression pour éviter une boucle infinie si overlap >= chunk_size
        # ou si end ne bouge pas.
        if next_start <= start:
             # Si end n'a pas pu être ajusté (e.g. très longue ligne sans espace/retour),
             # forcer la progression au-delà du chunk actuel.
             if end == start + chunk_size:
                 next_start = end
             else: # Sinon, avancer d'au moins un caractère
                 next_start = start + 1 

        start = next_start
        
        # Safety break if start doesn't advance significantly (should not happen with above logic)
        # This indicates a potential logic error we need to fix if it occurs.
        if start >= text_len:
             break


def process_file(file_path, chunk_size=512, overlap=50):
    """
    Traite un fichier et le découpe en chunks.
    
    Args:
        file_path: Chemin vers le fichier
        chunk_size: Taille des chunks
        overlap: Chevauchement entre les chunks
        
    Returns:
        Liste des chunks
    """
    try:
        # Traitement différent selon l'extension du fichier
        if file_path.lower().endswith('.pdf'):
            if not PYPDF2_AVAILABLE and not PDFMINER_AVAILABLE:
                print(f"Impossible de traiter le PDF {file_path}. Installez PyPDF2 ou pdfminer.six.")
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
                # Essayer avec une autre encodage
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                except Exception as e:
                    print(f"Erreur lors de la lecture de {file_path} avec encodage latin-1: {str(e)}")
                    return []
            except Exception as e:
                print(f"Erreur lors de la lecture de {file_path}: {str(e)}")
                return []
        
        # Si le contenu est vide, retourner une liste vide
        if not content.strip():
            print(f"Attention: {file_path} ne contient pas de texte exploitable.")
            return []
        
        # Utiliser list() pour collecter les chunks du générateur
        chunks = list(chunk_text(content, chunk_size, overlap))
        if not chunks:
            print(f"Attention: Aucun chunk n'a pu être créé à partir de {file_path}")
        return chunks
    except Exception as e:
        print(f"Erreur lors du traitement de {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()  # Affiche la trace complète pour le débogage
        return []


def main():
    parser = argparse.ArgumentParser(description="Préparation de la base de connaissances")
    parser.add_argument("--input", type=str, required=True, 
                        help="Fichier ou répertoire d'entrée")
    parser.add_argument("--output", type=str, required=True, 
                        help="Fichier de sortie (JSON)")
    parser.add_argument("--chunk-size", type=int, default=512, 
                        help="Taille des chunks (en caractères)")
    parser.add_argument("--overlap", type=int, default=50, 
                        help="Chevauchement entre les chunks")
    parser.add_argument("--device", type=str, default=None, 
                        help="Appareil à utiliser pour les embeddings")
    parser.add_argument("--extensions", type=str, default=".txt,.md,.pdf", 
                        help="Extensions de fichiers à traiter (séparées par des virgules)")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Ne pas générer les embeddings (utile pour le débogage)")
    parser.add_argument("--debug", action="store_true",
                        help="Mode débogage avec plus d'informations")
    
    args = parser.parse_args()
    
    # Mode débogage
    if args.debug:
        import traceback
        print("Mode débogage activé")
        
        # Activer le mode débogage
        def process_file_with_debug(file_path, chunk_size=512, overlap=50):
            try:
                return process_file(file_path, chunk_size, overlap)
            except Exception as e:
                print(f"ERREUR CRITIQUE lors du traitement de {file_path}:")
                traceback.print_exc()
                return []
        
        # Remplacer la fonction process_file par la version avec débogage
        global process_file
        original_process_file = process_file
        process_file = process_file_with_debug
    
    # Conversion des extensions
    extensions = args.extensions.split(',')
    
    # Collecte des documents
    documents = []
    
    if os.path.isdir(args.input):
        print(f"Traitement du répertoire {args.input}...")
        
        for root, _, files in os.walk(args.input):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    print(f"Traitement de {file_path}...")
                    chunks = process_file(file_path, args.chunk_size, args.overlap)
                    
                    for i, chunk in enumerate(chunks):
                        documents.append({
                            "source": file_path,
                            "chunk_id": i,
                            "content": chunk
                        })
    else:
        print(f"Traitement du fichier {args.input}...")
        chunks = process_file(args.input, args.chunk_size, args.overlap)
        
        for i, chunk in enumerate(chunks):
            documents.append({
                "source": args.input,
                "chunk_id": i,
                "content": chunk
            })
    
    print(f"Nombre total de chunks: {len(documents)}")
    
    # Vérification si des documents ont été traités
    if not documents:
        print("Aucun document n'a pu être traité. Vérifiez les messages d'erreur ci-dessus.")
        # Sauvegarde d'un fichier JSON vide
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        print(f"Fichier JSON vide créé: {args.output}")
        return
    
    # Génération des embeddings
    if not args.skip_embeddings:
        print("Génération des embeddings...")
        generator = EmbeddingGenerator(device=args.device)
        print(f"EmbeddingGenerator utilise le périphérique: {generator.device}") # Ajout pour vérification
        
        texts = [doc["content"] for doc in documents]
        embeddings = generator.generate_embeddings(texts)
        
        # Ajout des embeddings aux documents
        for i, embedding in enumerate(embeddings):
            documents[i]["embedding"] = embedding.tolist()
    
    # Sauvegarde des résultats
    print(f"Sauvegarde des résultats dans {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    print("Terminé!")


if __name__ == "__main__":
    main()
