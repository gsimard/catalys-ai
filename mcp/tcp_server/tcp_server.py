#!/usr/bin/env python3
import asyncio
import logging
import argparse
# importlib.util n'est plus nécessaire
import sys
import os
from pathlib import Path
# Ajout pour create_subprocess_exec
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Renommer le logger pour refléter le rôle de wrapper générique
logger = logging.getLogger("mcp_tcp_wrapper")

class MCPTCPServer:
    # Accepter la commande et le cwd
    def __init__(self, host='0.0.0.0', port=8765, mcp_command=None, mcp_cwd=None):
        self.host = host
        self.port = port
        self.server = None
        self.clients = set() # Garder une trace des tâches de gestion client
        self.mcp_command = mcp_command
        self.mcp_cwd = mcp_cwd
        if not self.mcp_command:
            raise ValueError("mcp_command is required")
        if not self.mcp_cwd:
            raise ValueError("mcp_cwd is required")

    async def start_server(self):
        """Démarre le serveur TCP qui écoute les connexions."""
        self.server = await asyncio.start_server(
            self.handle_client, self.host, self.port
        )

        addr = self.server.sockets[0].getsockname()
        logger.info(f'Wrapper MCP TCP démarré sur {addr[0]}:{addr[1]}')
        logger.info(f"Prêt à lancer la commande: {' '.join(self.mcp_command)} dans {self.mcp_cwd}")

        async with self.server:
            await self.server.serve_forever()

    async def pipe_stream(self, reader, writer, source_name, dest_name):
        """Lit depuis reader et écrit vers writer jusqu'à EOF."""
        try:
            while not reader.at_eof():
                data = await reader.read(4096) # Lire par blocs
                if not data:
                    break # EOF
                # logger.debug(f"Piping {len(data)} bytes from {source_name} to {dest_name}")
                writer.write(data)
                await writer.drain()
        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError) as e:
            logger.info(f"Pipe from {source_name} to {dest_name} closed: {type(e).__name__}")
        except Exception as e:
            logger.error(f"Error piping from {source_name} to {dest_name}: {e}", exc_info=True)
        finally:
            # logger.debug(f"Closing writer for {dest_name}")
            if not writer.is_closing():
                 writer.close()
                 try:
                     await writer.wait_closed()
                 except Exception:
                     pass # Peut échouer si déjà fermé
            logger.info(f"Pipe from {source_name} to {dest_name} finished.")


    async def handle_client(self, client_reader, client_writer):
        """Gère une connexion client en lançant le serveur MCP et en reliant les flux."""
        peername = client_writer.get_extra_info('peername', ('?', '?'))
        client_id = f"{peername[0]}:{peername[1]}"
        logger.info(f"Nouvelle connexion client: {client_id}")

        process = None
        # Utiliser un set pour les tâches de ce client spécifique
        client_tasks = set()
        try:
            logger.info(f"Lancement du processus MCP pour {client_id}: {' '.join(self.mcp_command)}")
            process = await asyncio.create_subprocess_exec(
                *self.mcp_command, # Dépaqueter la commande et ses arguments
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, # Capturer stderr aussi
                cwd=self.mcp_cwd # Définir le répertoire de travail
            )
            logger.info(f"Processus MCP démarré pour {client_id} avec PID {process.pid}")

            # Créer les tâches pour relier les flux
            # Tâche 1: Client TCP -> stdin du processus MCP
            to_mcp_task = asyncio.create_task(
                self.pipe_stream(client_reader, process.stdin, f"Client {client_id}", f"MCP stdin (PID {process.pid})"),
                name=f"pipe_client_to_mcp_{client_id}"
            )

            # Tâche 2: stdout du processus MCP -> Client TCP
            from_mcp_task = asyncio.create_task(
                self.pipe_stream(process.stdout, client_writer, f"MCP stdout (PID {process.pid})", f"Client {client_id}"),
                name=f"pipe_mcp_to_client_{client_id}"
            )

            # Tâche 3 (Optionnel mais recommandé): stderr du processus MCP -> Logs du wrapper
            async def log_stderr():
                try:
                    while not process.stderr.at_eof():
                        line = await process.stderr.readline()
                        if line:
                            logger.warning(f"MCP stderr (PID {process.pid}): {line.decode(errors='ignore').strip()}")
                        else:
                            break
                except asyncio.CancelledError:
                    logger.info(f"Stderr logging cancelled for MCP (PID {process.pid})")
                except Exception as e:
                    logger.error(f"Error reading MCP stderr (PID {process.pid}): {e}", exc_info=True)
                finally:
                    logger.info(f"Stderr logging finished for MCP (PID {process.pid})")


            stderr_task = asyncio.create_task(log_stderr(), name=f"log_stderr_{client_id}")

            client_tasks.add(to_mcp_task)
            client_tasks.add(from_mcp_task)
            client_tasks.add(stderr_task)

            # Attendre que les tâches de communication se terminent ou que le processus finisse
            # Ajouter process.wait() à l'ensemble des tâches à attendre
            process_wait_task = asyncio.create_task(process.wait(), name=f"process_wait_{client_id}")
            client_tasks.add(process_wait_task)

            done, pending = await asyncio.wait(
                 client_tasks, return_when=asyncio.FIRST_COMPLETED
            )

            # Log des tâches terminées (pour debug)
            for task in done:
                 exception = task.exception()
                 if exception:
                     logger.error(f"Task completed with error for {client_id}: {task.get_name()} - Exception: {exception}", exc_info=exception)
                 else:
                     logger.debug(f"Task completed normally for {client_id}: {task.get_name()} - Result: {task.result()}")


            logger.info(f"Communication terminée ou processus MCP arrêté pour {client_id}. Nettoyage...")

        except Exception as e:
            logger.error(f"Erreur lors de la gestion du client {client_id} ou du lancement du processus: {e}", exc_info=True)
        finally:
            logger.info(f"Début du nettoyage pour {client_id}")
            # Annuler les tâches en cours s'il y en a pour ce client
            for task in client_tasks:
                if not task.done():
                    logger.debug(f"Annulation de la tâche en cours: {task.get_name()}")
                    task.cancel()

            # Attendre que les tâches annulées se terminent (avec timeout)
            if client_tasks:
                # Créer une copie car le set peut être modifié pendant l'itération
                tasks_to_wait = list(client_tasks)
                try:
                    # Ne pas attendre process_wait_task s'il est déjà terminé
                    if process_wait_task in tasks_to_wait and process_wait_task.done():
                        tasks_to_wait.remove(process_wait_task)

                    if tasks_to_wait: # Seulement attendre s'il reste des tâches
                         _, pending_after_cancel = await asyncio.wait(tasks_to_wait, timeout=2.0)
                         if pending_after_cancel:
                              logger.warning(f"{len(pending_after_cancel)} tâches ne se sont pas terminées après annulation pour {client_id}")
                              for p_task in pending_after_cancel:
                                   logger.warning(f" - Tâche en attente: {p_task.get_name()}")

                except asyncio.TimeoutError:
                    logger.warning(f"Timeout en attendant la fin des tâches pour {client_id}")
                except asyncio.CancelledError:
                     pass # Normal si le serveur s'arrête
                except Exception as e:
                     logger.error(f"Erreur pendant l'attente des tâches annulées pour {client_id}: {e}", exc_info=True)


            # Fermer la connexion client proprement
            if client_writer and not client_writer.is_closing():
                logger.debug(f"Fermeture de la connexion client {client_id}")
                client_writer.close()
                try:
                    await writer.wait_closed()
                except Exception as e:
                    logger.warning(f"Erreur lors de wait_closed pour le client {client_id}: {e}")


            # S'assurer que le processus MCP est terminé
            if process and process.returncode is None:
                logger.warning(f"Le processus MCP (PID {process.pid}) ne s'est pas terminé, tentative de terminaison...")
                try:
                    process.terminate()
                    # Attendre un peu que le processus se termine
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                    logger.info(f"Processus MCP (PID {process.pid}) terminé avec code {process.returncode}")
                except asyncio.TimeoutError:
                    logger.error(f"Timeout lors de la terminaison du processus MCP (PID {process.pid}), tentative de kill...")
                    try:
                        process.kill()
                        # Attendre après kill
                        await process.wait()
                        logger.info(f"Processus MCP (PID {process.pid}) tué, code retour {process.returncode}.")
                    except ProcessLookupError:
                         logger.warning(f"Processus MCP (PID {process.pid}) déjà terminé avant kill.")
                    except Exception as kill_e:
                         logger.error(f"Erreur lors du kill du processus MCP (PID {process.pid}): {kill_e}")

                except ProcessLookupError: # Processus déjà terminé
                     logger.warning(f"Processus MCP (PID {process.pid}) déjà terminé avant la tentative de terminaison.")
                except Exception as e:
                    logger.error(f"Erreur lors de la terminaison/kill du processus MCP (PID {process.pid}): {e}")
            elif process:
                 logger.info(f"Processus MCP (PID {process.pid}) s'est terminé avec le code {process.returncode}")


            logger.info(f"Nettoyage terminé pour {client_id}")


async def run_server():
    """Parse arguments and run the MCP TCP Wrapper server."""
    parser = argparse.ArgumentParser(
        description="Démarre un wrapper TCP générique pour un serveur MCP exécutable via stdio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Exemple:
  python tcp_server.py --cwd /path/to/mysql/server -- python src/mysql_mcp_server/server.py
  python tcp_server.py --cwd /path/to/go/server -- ./my-go-mcp-server --config config.json
"""
    )
    parser.add_argument("--host", default="0.0.0.0", help="Adresse d'écoute du wrapper TCP (défaut: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Port d'écoute du wrapper TCP (défaut: 8765)")
    parser.add_argument("--cwd", required=True, help="Répertoire de travail pour lancer la commande MCP.")
    # Utiliser nargs=argparse.REMAINDER pour capturer la commande et tous ses arguments
    parser.add_argument('mcp_command', nargs=argparse.REMAINDER, help="La commande complète pour lancer le serveur MCP.")

    args = parser.parse_args()

    if not args.mcp_command:
        logger.error("Aucune commande MCP fournie.")
        parser.print_help()
        sys.exit(1)

    # Vérifier si le premier élément de la commande est un chemin relatif/absolu et s'il existe
    # C'est une heuristique, pourrait être améliorée
    cmd_path = args.mcp_command[0]
    # Utiliser shutil.which pour vérifier si la commande est dans le PATH
    if not shutil.which(cmd_path):
         # Si pas dans le PATH, vérifier si c'est un chemin relatif au CWD et exécutable
         potential_path = os.path.join(args.cwd, cmd_path)
         if not os.path.isfile(potential_path) or not os.access(potential_path, os.X_OK):
              # Log un warning mais continuer, l'utilisateur sait peut-être ce qu'il fait (ex: commande shell interne, ou python script)
              logger.warning(f"La commande '{cmd_path}' n'a pas été trouvée dans le PATH et n'est pas un fichier exécutable dans {args.cwd}. Tentative de lancement quand même.")
         else:
              # Si c'est un chemin relatif/absolu valide, l'utiliser
              logger.info(f"Utilisation du chemin '{potential_path}' pour la commande.")
              # On pourrait modifier args.mcp_command[0] = potential_path mais create_subprocess_exec le gère bien avec cwd

    if not os.path.isdir(args.cwd):
        logger.error(f"Le répertoire de travail spécifié n'existe pas: {args.cwd}")
        sys.exit(1)

    # Démarrer le serveur TCP wrapper directement
    logger.info(f"Configuration du wrapper TCP pour lancer la commande : {' '.join(args.mcp_command)}")
    logger.info(f"Répertoire de travail pour la commande : {args.cwd}")
    logger.info(f"Wrapper TCP écoutera sur {args.host}:{args.port}")

    # Passer la commande (liste) et le cwd au serveur TCP
    server = MCPTCPServer(args.host, args.port, args.mcp_command, args.cwd)

    try:
        print(f"Wrapper MCP TCP démarrant...")
        print(f"Écoute sur {args.host}:{args.port}")
        print(f"Prêt à lancer : {' '.join(args.mcp_command)} dans {args.cwd}")
        print("Appuyez sur Ctrl+C pour arrêter le wrapper")

        # La méthode start_server gère maintenant la boucle serveur asyncio
        await server.start_server()

    except KeyboardInterrupt:
        logger.info("Arrêt du wrapper demandé par l'utilisateur")
        print("\nArrêt du wrapper...")
        # La gestion de l'arrêt doit être dans MCPTCPServer ou gérée par asyncio

    except Exception as e:
        logger.error(f"Erreur lors du démarrage ou de l'exécution du wrapper: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("Programme interrompu.")
