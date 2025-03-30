# Outils Vectoriels pour RAG et Recherche Sémantique

Ce répertoire contient des scripts Python pour construire et interroger une base de connaissances vectorielle en utilisant des modèles d'embedding (comme BGE-M3) et un système RAG (Retrieval-Augmented Generation).

## Fonctionnalités

*   **Préparation de la base de connaissances (`prepare_kb.py`)**: Traite des fichiers texte, Markdown ou PDF, les découpe en chunks, génère les embeddings et sauvegarde le tout dans un fichier `.pkl` optimisé (embeddings en F16).
*   **Système RAG (`rag_system.py`)**: Permet de poser des questions en langage naturel. Le système récupère les chunks les plus pertinents de la base de connaissances et les utilise comme contexte pour générer une réponse avec un LLM.
    *   **Sélection du LLM**:
        *   **Par défaut**: Utilise un endpoint externe (type OpenAI, configuré via les variables d'environnement `NEBIUS_API_BASE`, `NEBIUS_API_KEY`, `NEBIUS_AI_MODEL`) s'il est configuré.
        *   **Fallback**: Si l'endpoint externe n'est pas configuré ou si l'option `--force-local-llm` est utilisée, tente de charger un LLM local via la bibliothèque `transformers`.
*   **Recherche Sémantique (`semantic_search.py`)**: Recherche les chunks les plus similaires à une requête donnée dans la base de connaissances, sans génération de réponse par un LLM.
*   **Démonstration de Similarité (`similarity_demo.py`)**: Calcule et affiche la similarité cosinus entre plusieurs phrases d'exemple.
*   **Génération d'Embeddings (`bge_embeddings.py`)**: Script de base pour générer et sauvegarder des embeddings pour une liste de textes.

## Installation

1.  **Clonez le dépôt** (si ce n'est pas déjà fait).
2.  **Installez les dépendances Python**:
    ```bash
    pip install -r requirements.txt 
    # Ou installez les paquets nécessaires manuellement:
    # pip install torch transformers numpy scikit-learn python-dotenv openai tqdm pickle5 PyPDF2 pdfminer.six pytesseract pdf2image prompt_toolkit
    ```
    *   `torch`: Nécessaire pour les modèles `transformers`. Installez la version appropriée (CPU ou CUDA) selon votre matériel. Voir [pytorch.org](https://pytorch.org/).
    *   `transformers`: Pour charger les modèles d'embedding (BGE-M3) et le LLM local (si utilisé).
    *   `numpy`, `scikit-learn`: Pour les opérations sur les vecteurs.
    *   `python-dotenv`: Pour charger les variables d'environnement depuis un fichier `.env`.
    *   `openai`: Pour interagir avec l'API externe (Nebius/OpenAI).
    *   `tqdm`: Pour les barres de progression.
    *   `pickle5`: Assure la compatibilité du format pickle.
    *   `PyPDF2`, `pdfminer.six`: Pour l'extraction de texte des PDF.
    *   `pytesseract`, `pdf2image`: Pour l'OCR sur les PDF scannés (nécessite l'installation de Tesseract OCR sur votre système).
    *   `prompt_toolkit`: Pour une meilleure expérience en mode interactif.

3.  **(Optionnel) Configuration de l'endpoint LLM externe**:
    Créez un fichier `.env` à la racine du projet ou dans le répertoire `vector/` avec les variables suivantes si vous souhaitez utiliser un LLM externe (comme celui de Nebius AI) :
    ```dotenv
    NEBIUS_API_BASE="URL_DE_VOTRE_ENDPOINT"
    NEBIUS_API_KEY="VOTRE_CLE_API"
    NEBIUS_AI_MODEL="NOM_DU_MODELE_DEPLOYE" 
    ```

## Utilisation

### 1. Préparer la Base de Connaissances (`prepare_kb.py`)

Ce script lit les fichiers d'un répertoire (ou un fichier unique), les découpe en chunks, génère les embeddings pour chaque chunk et sauvegarde le résultat dans un fichier `.pkl`.

```bash
python vector/prepare_kb.py --input chemin/vers/vos/documents --output kb.pkl --device cuda --chunk-size 512 --overlap 50 --extensions .txt,.md,.pdf
```

*   `--input`: Chemin vers un fichier ou un répertoire contenant les documents sources.
*   `--output`: Chemin vers le fichier `.pkl` de sortie qui contiendra la base de connaissances.
*   `--device`: `cuda` pour utiliser le GPU, `cpu` pour le CPU. Détecté automatiquement si omis.
*   `--chunk-size`: Taille maximale des chunks en caractères.
*   `--overlap`: Nombre de caractères de chevauchement entre les chunks consécutifs.
*   `--extensions`: Extensions des fichiers à traiter (séparées par des virgules).
*   `--skip-embeddings`: (Optionnel) Pour ne pas générer les embeddings (utile pour tester le découpage).
*   `--debug`: (Optionnel) Active des messages de débogage plus détaillés.

### 2. Interroger avec RAG (`rag_system.py`)

Ce script charge la base de connaissances (`.pkl`) et permet de poser des questions. Il récupère les informations pertinentes et génère une réponse.

**Mode interactif :**

```bash
python vector/rag_system.py --kb kb.pkl --interactive [--force-local-llm] [--device cuda] [--llm nom/modele_local] [--top-k 5] [--debug]
```

*   `--kb`: Chemin vers le fichier `.pkl` de la base de connaissances.
*   `--interactive`: Lance le mode interactif pour poser plusieurs questions.
*   `--force-local-llm`: Force l'utilisation d'un LLM local même si un endpoint externe est configuré dans `.env`.
*   `--device`: Spécifie le device (`cuda` ou `cpu`) pour les modèles locaux (embeddings et LLM local).
*   `--llm`: (Utilisé si `--force-local-llm` ou si l'endpoint externe n'est pas configuré) Nom ou chemin du modèle LLM local à charger via `transformers`.
*   `--top-k`: Nombre de chunks pertinents à récupérer pour le contexte.
*   `--search-only`: Ne fait que la recherche de chunks pertinents, sans passer par le LLM pour la génération.
*   `--debug`: Affiche les chunks récupérés et le prompt envoyé au LLM.

**Mode non interactif (une seule question) :**

```bash
python vector/rag_system.py --kb kb.pkl --query "Votre question ici" [--force-local-llm] [--device cuda] [...]
```

*   `--query`: La question à poser.

### 3. Recherche Sémantique (`semantic_search.py`)

Ce script effectue une recherche de similarité entre une requête et les chunks de la base de connaissances, sans générer de réponse.

**Mode interactif :**

```bash
python vector/semantic_search.py --kb kb.pkl --interactive [--top-k 5] [--device cuda]
```

**Mode non interactif :**

```bash
python vector/semantic_search.py --kb kb.pkl --query "Votre recherche ici" [--top-k 5] [--device cuda]
```

*   `--kb`: Chemin vers le fichier `.pkl` de la base de connaissances.
*   `--query`/`--interactive`: Votre requête ou mode interactif.
*   `--top-k`: Nombre de résultats à afficher.
*   `--device`: Device pour le modèle d'embedding.

### 4. Démonstration de Similarité (`similarity_demo.py`)

Ce script simple montre comment calculer la similarité cosinus entre les embeddings de plusieurs phrases.

```bash
python vector/similarity_demo.py [--device cuda]
```

## Notes

*   **Modèles**: Les scripts utilisent par défaut `BAAI/bge-m3` pour les embeddings et `meta-llama/Llama-2-7b-chat-hf` comme LLM local. Vous pouvez spécifier d'autres modèles compatibles avec `transformers`.
*   **Performance**: La génération d'embeddings et l'inférence LLM locale peuvent être gourmandes en ressources (VRAM, RAM, CPU). L'utilisation d'un GPU (`--device cuda`) est fortement recommandée. L'utilisation de l'endpoint externe est généralement plus rapide si vous y avez accès.
*   **Format Pickle**: Le format `.pkl` avec des embeddings `float16` est utilisé pour réduire l'espace disque par rapport à un stockage JSON des vecteurs.
