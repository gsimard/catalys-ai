import subprocess
import os
import datetime
from typing import Optional

# Fonctions MCP à enregistrer
def notification(msg: str) -> None:
    """Envoie une notification et arrête la boucle."""
    try:
        script_path = "/app/scripts/notification.sh"
        subprocess.run([script_path, msg], check=True)
        print(f"Notification envoyée: {msg}")
        return None  # Retourne None pour indiquer d'arrêter la boucle
    except Exception as e:
        error_msg = f"Erreur lors de l'envoi de la notification: {str(e)}"
        print(error_msg)
        return None  # Retourne None pour indiquer d'arrêter la boucle

def command_line(cmd: str) -> str:
    """Exécute une ligne de commande Linux bash."""
    try:
        # Utiliser Popen au lieu de run pour pouvoir gérer l'interruption
        process = subprocess.Popen(
            cmd, 
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )
        
        stdout_data = []
        stderr_data = []
        
        # Lire la sortie de manière non bloquante
        import select
        import time
        
        # Configurer les descripteurs de fichiers pour select
        readable_list = [process.stdout, process.stderr]
        
        try:
            # Tant que le processus est en cours d'exécution
            while process.poll() is None:
                # Vérifier si des données sont disponibles (timeout de 0.1s)
                ready, _, _ = select.select(readable_list, [], [], 0.1)
                
                for stream in ready:
                    line = stream.readline()
                    if line:
                        if stream == process.stdout:
                            stdout_data.append(line)
                            print(line, end='')  # Afficher en temps réel
                        else:
                            stderr_data.append(line)
                            print(f"STDERR: {line}", end='')  # Afficher les erreurs
                
                time.sleep(0.1)  # Petite pause pour éviter de surcharger le CPU
                
        except KeyboardInterrupt:
            # Terminer le processus
            process.terminate()
            try:
                # Attendre que le processus se termine avec un timeout
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # Si le processus ne se termine pas proprement, le tuer
                process.kill()
            
            return "Commande interrompue."
        
        # Récupérer les données restantes
        stdout_remainder, stderr_remainder = process.communicate()
        if stdout_remainder:
            stdout_data.append(stdout_remainder)
        if stderr_remainder:
            stderr_data.append(stderr_remainder)
        
        # Vérifier le code de retour
        if process.returncode != 0:
            return f"Erreur lors de l'exécution de la commande (code {process.returncode}):\n{''.join(stderr_data)}"
        
        return f"Commande exécutée avec succès:\n{''.join(stdout_data)}"
        
    except Exception as e:
        return f"Erreur lors de l'exécution de la commande: {str(e)}"

def write_file(file_name: str, contents: str) -> str:
    """Écrit contents dans le fichier file-name."""
    try:
        # Assurez-vous que le répertoire existe
        os.makedirs(os.path.dirname(os.path.abspath(file_name)), exist_ok=True)
        
        # Traiter les caractères d'échappement dans la chaîne de contenu
        # Convertir les séquences d'échappement littérales comme \n en véritables caractères
        processed_contents = contents
        if r'\n' in contents or r'\t' in contents or r'\"' in contents:
            processed_contents = contents.encode().decode('unicode_escape')
        
        # Supprimer les guillemets au début et à la fin si présents
        if processed_contents.startswith('"') and processed_contents.endswith('"'):
            processed_contents = processed_contents[1:-1]
        
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(processed_contents)
        
        return f"Fichier '{file_name}' créé avec succès."
    except Exception as e:
        return f"Erreur lors de la création du fichier '{file_name}': {str(e)}"

def read_file(file_name: str) -> str:
    """Lit et retourne le contenu du fichier spécifié."""
    try:
        with open(file_name, "r", encoding="utf-8") as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return f"Erreur: Le fichier '{file_name}' n'existe pas."
    except Exception as e:
        return f"Erreur lors de la lecture du fichier '{file_name}': {str(e)}"

def read_file_about_line(file_name: str, line_number: int, context: int = 5) -> str:
    """Lit et retourne le contenu du fichier spécifié autour d'une ligne donnée."""
    try:
        with open(file_name, "r", encoding="utf-8") as file:
            lines = file.readlines()
        
        total_lines = len(lines)

        print(f"read_file_about_line: {str}")
        
        if line_number < 1 or line_number > total_lines:
            return f"Erreur: Le numéro de ligne {line_number} est hors limites (1-{total_lines})."
        
        start_line = max(1, line_number - context)
        end_line = min(total_lines, line_number + context)
        
        # Ajuster pour l'indexation basée sur 0
        selected_lines = lines[start_line-1:end_line]
        
        return ''.join(selected_lines)
    except FileNotFoundError:
        return f"Erreur: Le fichier '{file_name}' n'existe pas."
    except Exception as e:
        return f"Erreur lors de la lecture du fichier '{file_name}': {str(e)}"

def update_file(file_name: str, old_str: str, new_str: str) -> str:
    """Remplace une unique occurrence de old_str par new_str dans file_name."""
    log_file_path = "work/update_debug.log"
    log_entry_prefix = f"--- {datetime.datetime.now()} ---\nFichier: {file_name}\n"
    
    try:
        # Traiter les caractères d'échappement dans les chaînes
        processed_old_str = old_str
        if r'\n' in old_str or r'\t' in old_str or r'\"' in old_str:
            processed_old_str = old_str.encode().decode('unicode_escape')
        if processed_old_str.startswith('"') and processed_old_str.endswith('"'):
             processed_old_str = processed_old_str[1:-1]

        processed_new_str = new_str
        if r'\n' in new_str or r'\t' in new_str or r'\"' in new_str:
            processed_new_str = new_str.encode().decode('unicode_escape')
        if processed_new_str.startswith('"') and processed_new_str.endswith('"'):
             processed_new_str = processed_new_str[1:-1]

        # Lire le contenu du fichier
        try:
            with open(file_name, "r", encoding="utf-8") as file:
                content = file.read()
        except FileNotFoundError:
            error_msg = f"Erreur: Le fichier '{file_name}' n'existe pas."
            # Log de l'erreur
            try:
                with open(log_file_path, "a", encoding="utf-8") as log_file:
                    log_file.write(f"{log_entry_prefix}Résultat: ÉCHEC - Fichier non trouvé\n====================\n")
            except Exception as log_e:
                print(f"Avertissement: Erreur lors de l'écriture dans {log_file_path}: {log_e}")
            return error_msg
        except Exception as e:
            error_msg = f"Erreur lors de la lecture du fichier '{file_name}': {str(e)}"
            # Log de l'erreur
            try:
                with open(log_file_path, "a", encoding="utf-8") as log_file:
                    log_file.write(f"{log_entry_prefix}Résultat: ÉCHEC - Erreur lecture fichier\nErreur: {str(e)}\n====================\n")
            except Exception as log_e:
                print(f"Avertissement: Erreur lors de l'écriture dans {log_file_path}: {log_e}")
            return error_msg

        # Compter les occurrences de old_str
        occurrences = content.count(processed_old_str)

        # Préparer le log détaillé
        log_detail = (
            f"{log_entry_prefix}"
            f"--- Old String ---\n{processed_old_str}\n"
            f"--- New String ---\n{processed_new_str}\n"
            f"--- Occurrences trouvées: {occurrences} ---\n"
        )

        if occurrences == 0:
            result_msg = f"Erreur: La chaîne à remplacer n'a pas été trouvée dans '{file_name}'."
            log_detail += f"Résultat: ÉCHEC - Chaîne non trouvée\n====================\n"
        elif occurrences > 1:
            result_msg = f"Erreur: La chaîne à remplacer n'est pas unique ({occurrences} occurrences trouvées) dans '{file_name}'. Remplacement annulé."
            log_detail += f"Résultat: ÉCHEC - Chaîne non unique\n====================\n"
        else:
            # Remplacer l'unique occurrence
            new_content = content.replace(processed_old_str, processed_new_str, 1)
            
            # Écrire le nouveau contenu dans le fichier
            try:
                with open(file_name, "w", encoding="utf-8") as file:
                    file.write(new_content)
                result_msg = f"Fichier '{file_name}' mis à jour avec succès."
                log_detail += f"Résultat: SUCCÈS\n====================\n"
            except Exception as e:
                result_msg = f"Erreur lors de l'écriture dans le fichier '{file_name}': {str(e)}"
                log_detail += f"Résultat: ÉCHEC - Erreur écriture fichier\nErreur: {str(e)}\n====================\n"

        # Écrire dans le fichier log
        try:
            os.makedirs(os.path.dirname(os.path.abspath(log_file_path)), exist_ok=True)
            with open(log_file_path, "a", encoding="utf-8") as log_file:
                log_file.write(log_detail)
        except Exception as log_e:
            print(f"Avertissement: Erreur lors de l'écriture dans {log_file_path}: {log_e}")
            
        return result_msg

    except Exception as e:
        error_msg = f"Erreur inattendue dans update_file: {str(e)}"
        # Log de l'erreur inattendue
        try:
            with open(log_file_path, "a", encoding="utf-8") as log_file:
                 log_file.write(f"{log_entry_prefix}Résultat: ÉCHEC - Erreur inattendue\nErreur: {str(e)}\n====================\n")
        except Exception as log_e:
            print(f"Avertissement: Erreur lors de l'écriture dans {log_file_path}: {log_e}")
        return error_msg

# def patch_file(file_name: str, contents: str, strip: int = 0) -> str:
#     """Applique un patch (diff) au fichier spécifié sans passer par le shell."""
#     try:
#         # Vérifier si le fichier existe
#         if not os.path.exists(file_name):
#             return f"Erreur: Le fichier '{file_name}' n'existe pas."
# 
#         # Traiter les caractères d'échappement dans la chaîne de contenu
#         # Convertir les séquences d'échappement littérales comme \n en véritables caractères
#         processed_contents = contents
#         if r'\n' in contents or r'\t' in contents or r'\"' in contents:
#             processed_contents = contents.encode().decode('unicode_escape')
#         
#         # Supprimer les guillemets au début et à la fin si présents
#         if processed_contents.startswith('"') and processed_contents.endswith('"'):
#             processed_contents = processed_contents[1:-1]
# 
#         # --- Début Log pour débogage patch ---
#         try:
#             with open(file_name, "r", encoding="utf-8") as f_orig:
#                 original_content = f_orig.read()
#             
#             log_entry = (
#                 f"--- {datetime.datetime.now()} ---\n"
#                 f"Fichier cible: {file_name}\n"
#                 f"--- Contenu Original ---\n{original_content}\n"
#                 f"--- Patch Proposé ---\n{processed_contents}\n"
#                 f"====================\n"
#             )
#             
#             with open("work/patch_debug.log", "a", encoding="utf-8") as log_file:
#                 log_file.write(log_entry)
#                 
#         except Exception as log_e:
#             print(f"Avertissement: Erreur lors de l'écriture dans patch_debug.log: {log_e}")
#         # --- Fin Log pour débogage patch ---
#         
#         # Créer un fichier temporaire pour le patch
#         import tempfile
#         with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.patch') as temp_file:
#             temp_file_name = temp_file.name
#             temp_file.write(processed_contents)
#         
#         # Exécuter la commande patch avec le fichier temporaire
#         cmd = ["patch", f"-p{strip}", "-i", temp_file_name, file_name]
#         
#         process = subprocess.Popen(
#             cmd,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=True
#         )
#         
#         stdout, stderr = process.communicate()
#         
#         # Supprimer le fichier temporaire
#         try:
#             1
#             #os.unlink(temp_file_name)
#         except:
#             pass
#         
#         if process.returncode != 0:
#             return f"Erreur d'application du patch: {stderr.strip()}"
#         
#         return f"Patch appliqué avec succès à '{file_name}':\n{stdout.strip()}"
#     except Exception as e:
#         return f"Erreur: {str(e)}"

#def memory_checkpoint(summary: str) -> str:
#    """Crée un point de contrôle mémoire pour réduire la taille de l'historique."""
#    try:
#        checkpoint_file = "work/memory_checkpoint.txt"
#        with open(checkpoint_file, "w", encoding="utf-8") as file:
#            file.write(summary)
#        return f"Point de contrôle mémoire créé avec succès."
#    except Exception as e:
#        return f"Erreur lors de la création du point de contrôle mémoire: {str(e)}"
