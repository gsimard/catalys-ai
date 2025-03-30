# Outils Vectoriels pour RAG avec LangChain et ChromaDB

Ce répertoire contient des scripts Python pour construire et interroger une base de connaissances vectorielle en utilisant des modèles d'embedding (comme BGE-M3), la base de données vectorielle ChromaDB, et le framework LangChain pour l'orchestration RAG (Retrieval-Augmented Generation).

## Fonctionnalités

*   **Préparation de la base de connaissances (`prepare_kb_chroma.py`)**: Traite des fichiers texte, Markdown ou PDF, les découpe en chunks, génère les embeddings (via LangChain) et les ajoute directement à une collection dans une instance ChromaDB.
*   **Système RAG (`rag_chroma_langchain.py`)**: Permet de poser des questions en langage naturel. Le système utilise LangChain pour :
    *   Se connecter à la collection ChromaDB.
    *   Récupérer les chunks les plus pertinents de la base de connaissances via un Retriever LangChain.
    *   Utiliser ces chunks comme contexte pour générer une réponse avec un LLM (local ou externe) via une Chaîne RAG LangChain.
    *   **Sélection du LLM**:
        *   **Par défaut**: Utilise un endpoint externe (type OpenAI, configuré via les variables d'environnement `NEBIUS_API_BASE`, `NEBIUS_API_KEY`, `NEBIUS_AI_MODEL`) s'il est configuré.
        *   **Fallback**: Si l'endpoint externe n'est pas configuré ou si l'option `--force-local-llm` est utilisée, tente de charger un LLM local via la bibliothèque `transformers` (encapsulé dans un wrapper LangChain).
*   **Démonstration de Similarité (`similarity_demo.py`)**: Calcule et affiche la similarité cosinus entre plusieurs phrases d'exemple (indépendant de ChromaDB/LangChain).
*   **Génération d'Embeddings (`bge_embeddings.py`)**: Script de base pour générer et sauvegarder des embeddings pour une liste de textes (sauvegarde en `.pkl`, non utilisé par le flux principal ChromaDB).
*   **(Déprécié)** `semantic_search.py` et `prepare_kb.py`: Anciens scripts basés sur le stockage Pickle. Non compatibles avec le flux ChromaDB.

## Installation

1.  **Clonez le dépôt** (si ce n'est pas déjà fait).
2.  **Installez les dépendances Python**:
    Idéalement, utilisez un environnement virtuel (`python -m venv .venv`, `source .venv/bin/activate`).
    ```bash
    pip install torch transformers numpy python-dotenv openai tqdm PyPDF2 pdfminer.six pytesseract pdf2image prompt_toolkit langchain langchain-community langchain-core langchain-openai chromadb-client sentence-transformers
    # Ou utilisez un fichier requirements.txt s'il est fourni et à jour.
    ```
    *   `torch`: Nécessaire pour `transformers`. Installez la version appropriée (CPU ou CUDA) selon votre matériel. Voir [pytorch.org](https://pytorch.org/).
    *   `transformers`: Pour charger les modèles d'embedding (BGE-M3) et le LLM local (si utilisé).
    *   `numpy`: Pour les opérations numériques.
    *   `python-dotenv`: Pour charger les variables d'environnement depuis un fichier `.env`.
    *   `openai`: Pour interagir avec l'API externe (Nebius/OpenAI).
    *   `tqdm`: Pour les barres de progression.
    *   `PyPDF2`, `pdfminer.six`: Pour l'extraction de texte des PDF.
    *   `pytesseract`, `pdf2image`: Pour l'OCR sur les PDF scannés (nécessite l'installation de Tesseract OCR sur votre système).
    *   `prompt_toolkit`: Pour une meilleure expérience en mode interactif.
    *   `langchain`, `langchain-community`, `langchain-core`, `langchain-openai`: Composants clés du framework LangChain.
    *   `chromadb-client`: Client Python pour interagir avec le serveur ChromaDB.
    *   `sentence-transformers`: Souvent requis par les wrappers d'embedding HuggingFace de LangChain.

3.  **Lancez un serveur ChromaDB**:
    ChromaDB stocke les vecteurs et métadonnées. Vous devez avoir un serveur ChromaDB en cours d'exécution. Pour un test local simple :
    ```bash
    # Installez le serveur si nécessaire: pip install chromadb
    # Lancez le serveur (il créera un répertoire 'chroma_db' pour stocker les données)
    chroma run --path ./chroma_db --host localhost --port 8000
    ```
    Gardez ce serveur actif pendant l'utilisation des scripts `prepare_kb_chroma.py` et `rag_chroma_langchain.py`.

4.  **(Optionnel) Configuration de l'endpoint LLM externe**:
    Créez un fichier `.env` à la racine du projet ou dans le répertoire `vector/` avec les variables suivantes si vous souhaitez utiliser un LLM externe (comme celui de Nebius AI) pour la génération de réponses :
    ```dotenv
    NEBIUS_API_BASE="URL_DE_VOTRE_ENDPOINT"
    NEBIUS_API_KEY="VOTRE_CLE_API"
    NEBIUS_AI_MODEL="NOM_DU_MODELE_DEPLOYE" 
    ```

## Utilisation

Assurez-vous que votre serveur ChromaDB est lancé (voir Installation étape 3).

### 1. Préparer la Base de Connaissances (`prepare_kb_chroma.py`)

Ce script lit les fichiers d'un répertoire (ou un fichier unique), les découpe en chunks, génère les embeddings et les ajoute à une collection dans votre serveur ChromaDB.

```bash
python vector/prepare_kb_chroma.py \
    --input chemin/vers/vos/documents \
    --chroma-host localhost \
    --chroma-port 8000 \
    --collection ma_collection_rag \
    --embedding-model BAAI/bge-m3 \
    --chunk-size 512 \
    --overlap 50 \
    --extensions .txt,.md,.pdf \
    --device auto # ou cuda, cpu
```

*   `--input`: Chemin vers un fichier ou un répertoire contenant les documents sources.
*   `--chroma-host`, `--chroma-port`: Adresse et port de votre serveur ChromaDB.
*   `--collection`: Nom que vous souhaitez donner à la collection dans ChromaDB. **Important :** Utilisez le même nom lors de l'interrogation.
*   `--embedding-model`: Modèle d'embedding à utiliser (doit être le même pour la préparation et l'interrogation).
*   `--device`: `cuda` pour utiliser le GPU, `cpu` pour le CPU, `auto` pour détection automatique.
*   `--chunk-size`: Taille maximale des chunks en caractères.
*   `--overlap`: Nombre de caractères de chevauchement entre les chunks consécutifs.
*   `--extensions`: Extensions des fichiers à traiter (séparées par des virgules).
*   `--batch-size`: (Optionnel) Nombre de documents à envoyer à ChromaDB en un seul batch.
*   `--debug`: (Optionnel) Active des messages de débogage plus détaillés.

Ce script est **additif** : si vous le relancez avec de nouveaux documents, ils seront ajoutés à la collection existante (attention aux doublons si les mêmes fichiers sont retraités).

### 2. Interroger avec RAG (`rag_chroma_langchain.py`)

Ce script se connecte à la collection ChromaDB spécifiée, récupère les informations pertinentes pour votre question en utilisant LangChain, et génère une réponse avec un LLM.

**Mode interactif :**

```bash
python vector/rag_chroma_langchain.py \
    --collection ma_collection_rag \
    --interactive \
    [--chroma-host localhost] \
    [--chroma-port 8000] \
    [--top-k 5] \
    [--force-local-llm] \
    [--device auto] \
    [--llm nom/modele_local] \
    [--debug]
```

*   `--collection`: **Obligatoire.** Nom de la collection ChromaDB à interroger (doit correspondre à celui utilisé lors de la préparation).
*   `--interactive`: Lance le mode interactif pour poser plusieurs questions.
*   `--chroma-host`, `--chroma-port`: Spécifiez si votre serveur ChromaDB n'est pas sur `localhost:8000`.
*   `--top-k`: Nombre de chunks pertinents à récupérer pour le contexte.
*   `--force-local-llm`: Force l'utilisation d'un LLM local même si un endpoint externe est configuré dans `.env`.
*   `--device`: Spécifie le device (`cuda`, `cpu`, `auto`) pour les modèles locaux (embeddings et LLM local).
*   `--llm`: (Utilisé si `--force-local-llm` ou si l'endpoint externe n'est pas configuré) Nom ou chemin du modèle LLM local à charger via `transformers`.
*   `--debug`: Affiche les documents récupérés (contenu tronqué) et potentiellement d'autres informations de débogage.

**Mode non interactif (une seule question) :**

```bash
python vector/rag_chroma_langchain.py \
    --collection ma_collection_rag \
    --query "Votre question ici" \
    [--top-k 3] \
    [...] # Autres options comme ci-dessus
```

*   `--query`: La question à poser.

### 3. Démonstration de Similarité (`similarity_demo.py`)

Ce script simple montre comment calculer la similarité cosinus entre les embeddings de plusieurs phrases.

```bash
python vector/similarity_demo.py [--device cuda]
```

## Notes

*   **Modèles**: Les scripts utilisent par défaut `BAAI/bge-m3` pour les embeddings et `meta-llama/Llama-2-7b-chat-hf` comme LLM local par défaut. Vous pouvez spécifier d'autres modèles compatibles avec `transformers` via les arguments `--embedding-model` et `--llm`.
*   **Serveur ChromaDB**: N'oubliez pas de lancer votre serveur ChromaDB avant d'exécuter les scripts `prepare_kb_chroma.py` et `rag_chroma_langchain.py`.
*   **Performance**: La génération d'embeddings et l'inférence LLM locale peuvent être gourmandes en ressources (VRAM, RAM, CPU). L'utilisation d'un GPU (`--device cuda`) est fortement recommandée pour les modèles locaux. L'utilisation de l'endpoint externe est généralement plus rapide si vous y avez accès.
*   **LangChain**: Ce projet utilise LangChain pour l'orchestration. Les concepts clés sont les `Embeddings`, `VectorStores` (Chroma), `Retrievers`, `Prompts`, `LLMs` (wrappers locaux ou API), et les `Chains` (LCEL).
