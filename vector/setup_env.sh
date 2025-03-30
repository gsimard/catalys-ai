#!/bin/bash
# Script pour configurer un environnement compatible avec ChromaDB et les transformers

# Créer un environnement virtuel
python -m venv bge-env

# Activer l'environnement
source bge-env/bin/activate

# Installer les dépendances avec les versions compatibles
pip install -r requirements.txt

# Vérifier l'installation de tokenizers
python -c "import tokenizers; print('Tokenizers version:', tokenizers.__version__)"

echo "Environnement configuré avec succès!"
