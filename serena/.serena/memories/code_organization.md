# Organisation du code

Le projet est organisé en plusieurs sous-projets, chacun dans son propre répertoire:

```
catalys-ai/
├── agno/         # Agent/wrapper pour modèles IA
├── chromadb/     # Probablement une base de données vectorielle
├── gpt/          # Utilitaires pour GPT
├── magnet/       # Module inconnu
├── mcp/          # Module inconnu
├── serena/       # Outil de développement
├── vector/       # Probablement lié aux embeddings
└── voice/        # Probablement lié à la voix/audio
```

Pour le sous-projet `agno`, voici sa structure actuelle:
```
agno/
├── .env          # Configuration des API et modèles
├── agent.py      # Définition de l'agent
└── playground.py # Interface pour tester l'agent
```