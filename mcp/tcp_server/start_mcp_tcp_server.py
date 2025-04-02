#!/usr/bin/env python3
import asyncio
import argparse
import sys
import os
import logging
import importlib.util
import subprocess

# Ajouter le répertoire parent au path pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("start_mcp_tcp_server")

async def main():
    parser = argparse.ArgumentParser(description="Démarrer un serveur MCP TCP")
    parser.add_argument("module_path", help="Chemin vers le module MCP (.py)")
    parser.add_argument("--host", default="0.0.0.0", help="Adresse d'écoute (défaut: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Port d'écoute (défaut: 8765)")
    
    args = parser.parse_args()
    
    # Vérifier si le module existe
    if not os.path.exists(args.module_path):
        logger.error(f"Le fichier module {args.module_path} n'existe pas")
        sys.exit(1)
    
    # Vérifier si le module tcp_server.py existe
    tcp_server_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "tcp_server.py")
    
    if not os.path.exists(tcp_server_path):
        logger.error(f"Le fichier tcp_server.py n'existe pas à l'emplacement: {tcp_server_path}")
        sys.exit(1)
    
    # Démarrer le serveur TCP
    logger.info(f"Démarrage du serveur MCP TCP pour le module {args.module_path} sur {args.host}:{args.port}")
    
    cmd = [
        sys.executable,
        tcp_server_path,
        args.module_path,
        "--host", args.host,
        "--port", str(args.port)
    ]
    
    try:
        # Exécuter le serveur TCP
        process = subprocess.Popen(cmd)
        logger.info(f"Serveur MCP TCP démarré avec PID {process.pid}")
        print(f"Serveur MCP TCP démarré avec PID {process.pid}")
        print(f"Écoute sur {args.host}:{args.port}")
        print("Appuyez sur Ctrl+C pour arrêter le serveur")
        
        # Attendre que le processus se termine
        process.wait()
        
    except KeyboardInterrupt:
        logger.info("Arrêt du serveur demandé par l'utilisateur")
        print("\nArrêt du serveur...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Le serveur ne s'est pas arrêté proprement, utilisation de kill")
            process.kill()
    
    except Exception as e:
        logger.error(f"Erreur lors du démarrage du serveur: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
