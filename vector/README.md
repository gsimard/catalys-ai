# Utilisation du modèle BGE-M3 en local

Ce projet permet d'utiliser le modèle multilingue BGE-M3 en local avec accélération GPU pour générer des embeddings de texte.

## Installation

```bash
# Installation des dépendances
pip install -r requirements.txt

# Pour utiliser CUDA (remplacer cu118 par votre version CUDA)
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118
```

## Utilisation

```bash
python main.py
```

## Fonctionnalités

- Génération d'embeddings pour des textes multilingues
- Calcul de similarité cosinus entre textes
- Recherche des textes les plus similaires
- Support de l'accélération GPU avec FP16

## Structure du projet

- `main.py` : Point d'entrée du programme avec exemple d'utilisation
- `embedding_model.py` : Classe pour charger le modèle et générer des embeddings
- `similarity_utils.py` : Fonctions utilitaires pour calculer les similarités

## Personnalisation

Vous pouvez modifier les paramètres du modèle dans `embedding_model.py` :
- Changer le modèle utilisé
- Ajuster la longueur maximale des tokens
- Activer/désactiver l'optimisation FP16
