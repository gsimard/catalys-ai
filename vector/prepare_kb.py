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
    
    # Méthode 1: pdfminer.six (si disponible)
    if PDFMINER_AVAILABLE:
        try:
            text = pdfminer_extract_text(pdf_path)
            if len(text.strip()) > 100:  # Si on a extrait suffisamment de texte
                return clean_text(text)
        except Exception as e:
            print(f"Erreur lors de l'extraction avec pdfminer: {e}")
    
    # Méthode 2: PyPDF2 (si disponible)
    if PYPDF2_AVAILABLE:
        try:
            text_pypdf = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text_pypdf += page.extract_text() + "\n\n"
            
            if len(text_pypdf.strip()) > len(text.strip()):
                text = text_pypdf
                if len(text.strip()) > 100:  # Si on a extrait suffisamment de texte
                    return clean_text(text)
        except Exception as e:
            print(f"Erreur lors de l'extraction avec PyPDF2: {e}")
    
    # Méthode 3: OCR (si disponible et nécessaire)
    if OCR_AVAILABLE and len(text.strip()) < 100:
        try:
            text_ocr = extract_text_from_scanned_pdf(pdf_path)
            if len(text_ocr.strip()) > len(text.strip()):
                text = text_ocr
        except Exception as e:
            print(f"Erreur lors de l'OCR: {e}")
    
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
    
    return chunks


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
            content = extract_text_from_pdf(file_path)
        else:
            # Pour les fichiers texte
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        
        # Si le contenu est vide, retourner une liste vide
        if not content.strip():
            print(f"Attention: {file_path} ne contient pas de texte exploitable.")
            return []
        
        return chunk_text(content, chunk_size, overlap)
    except Exception as e:
        print(f"Erreur lors du traitement de {file_path}: {e}")
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
    
    args = parser.parse_args()
    
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
    
    # Génération des embeddings
    if not args.skip_embeddings:
        print("Génération des embeddings...")
        generator = EmbeddingGenerator(device=args.device)
        
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
