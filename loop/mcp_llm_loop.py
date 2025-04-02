#!/usr/bin/env python3
import asyncio
import argparse
import sys
import logging
from llm_loop import main as llm_main

# Configuration du logging
logging.basicConfig(
    filename='mcp_llm.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    parser = argparse.ArgumentParser(description="Assistant LLM avec intégration MCP")
    parser.add_argument("--model", choices=["nebius", "claude", "gemini"], default="nebius",
                        help="Modèle LLM à utiliser (nebius, claude ou gemini)")
    parser.add_argument("--mcp-server", type=str, required=True,
                        help="Chemin vers le script du serveur MCP ou adresse TCP (host:port)")
    parser.add_argument("--tcp", action="store_true", 
                        help="Utiliser une connexion TCP au lieu de stdio")
    
    args = parser.parse_args()
    
    # Si --tcp est spécifié mais que l'adresse ne contient pas de port, ajouter le port par défaut
    if args.tcp and ':' not in args.mcp_server:
        args.mcp_server = f"{args.mcp_server}:8765"
        logger.info(f"Utilisation du port par défaut: {args.mcp_server}")
    
    # Passer les arguments à llm_loop.py
    sys.argv = [sys.argv[0], "--model", args.model, "--mcp-server", args.mcp_server]
    
    # Lancer la boucle LLM
    await llm_main()

if __name__ == "__main__":
    asyncio.run(main())
