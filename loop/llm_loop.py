import os
import re
import json
import inspect
import signal
import sys
import logging
import time
import datetime
import argparse
import asyncio
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import google.generativeai as genai

# Configuration du logging
logging.basicConfig(
    filename='llm.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Variable globale pour gérer CTRL+C
interrupt_requested = False

# Gestionnaire de signal pour CTRL+C
def handle_sigint(sig, frame):
    """Gère le signal SIGINT (CTRL+C)."""
    global interrupt_requested
    if interrupt_requested:
        print("\nSortie.")
        logger.info("Sortie de l'application (deuxième CTRL+C)")
        sys.exit(0)
    else:
        interrupt_requested = True
        print("\nAnnulation de la requête en cours...")
        # Ne pas logger ici pour éviter le spam si l'utilisateur appuie plusieurs fois rapidement

# Charger les variables d'environnement
load_dotenv()

# Récupérer les variables d'environnement
nebius_api_base = os.getenv("NEBIUS_API_BASE")
nebius_api_key = os.getenv("NEBIUS_API_KEY")
nebius_model = os.getenv("NEBIUS_AI_MODEL")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
max_tokens = int(os.getenv("NEBIUS_MAX_TOKENS", "8192"))
memory_tokens_threshold = int(os.getenv("MEMORY_TOKENS_THRESHOLD", "4000"))

# Paramètres de génération par défaut
default_temperature = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
default_top_p = float(os.getenv("DEFAULT_TOP_P", "0.95"))

# Constantes pour les modèles
NEBIUS_MODEL = "nebius"
CLAUDE_MODEL = "claude"
GEMINI_MODEL = "gemini"
ANTHROPIC_AI_MODEL = os.getenv("ANTHROPIC_AI_MODEL", "claude-3-7-sonnet-latest")
GEMINI_AI_MODEL = "gemini-2.5-pro-exp-03-25"

# Classe pour gérer les appels aux différents LLMs
class LLMClient:
    def __init__(self, model_type=NEBIUS_MODEL):
        self.model_type = model_type
        self.last_request_time = {} # Stocke le timestamp de la dernière requête par type de modèle
        # Délais par défaut (en secondes). 30s pour Gemini = 2 RPM. 0 = pas de limite.
        self.request_delay_seconds = {
            NEBIUS_MODEL: 0,
            CLAUDE_MODEL: 0,
            GEMINI_MODEL: 30.0
        }
        
        # Paramètres de génération par modèle
        self.generation_params = {
            NEBIUS_MODEL: {"temperature": default_temperature, "top_p": default_top_p, "min_p": 0.0},
            CLAUDE_MODEL: {"temperature": default_temperature, "top_p": default_top_p},
            GEMINI_MODEL: {"temperature": default_temperature, "top_p": default_top_p}
        }

        # Initialiser les clients selon le modèle
        self.openai_client = OpenAI(
            base_url=nebius_api_base,
            api_key=nebius_api_key
        )
        
        self.anthropic_client = anthropic.Anthropic(
            api_key=anthropic_api_key
        )

        # Configurer Gemini
        self.gemini_model = None
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel(GEMINI_AI_MODEL)
            logger.info(f"Client Gemini configuré pour le modèle: {GEMINI_AI_MODEL}")
        else:
            logger.warning("Clé API Gemini non trouvée. Le modèle Gemini ne sera pas disponible.")
            
        # Mapper les types de modèles à leurs méthodes de complétion
        self.completion_methods = {
            NEBIUS_MODEL: self._get_nebius_completion,
            CLAUDE_MODEL: self._get_anthropic_completion,
            GEMINI_MODEL: self._get_gemini_completion
        }

        logger.info(f"LLM initialisé avec le modèle par défaut: {model_type}")
        logger.info(f"Délais initiaux (secondes): {self.request_delay_seconds}")

    def set_rpm(self, model_type, rpm):
        if rpm <= 0:
            delay = 0.0 # Pas de limite
        else:
            delay = 60.0 / rpm
        
        if model_type in self.request_delay_seconds:
            self.request_delay_seconds[model_type] = delay
            logger.info(f"Délai pour {model_type} réglé à {delay:.2f} secondes ({rpm} RPM)")
            return f"Délai pour {model_type} réglé à {delay:.2f} secondes ({rpm} RPM)."
        else:
            logger.warning(f"Tentative de régler le RPM pour un modèle inconnu: {model_type}")
            return f"Modèle inconnu: {model_type}"

    def get_rpm_settings(self):
        settings = {}
        for model, delay in self.request_delay_seconds.items():
            rpm = "inf" if delay <= 0 else f"{60.0 / delay:.2f}"
            settings[model] = {"delay_seconds": f"{delay:.2f}", "rpm": rpm}
        return settings
        
    def set_generation_param(self, param_name, value, model_type=None):
        """Configure un paramètre de génération pour un modèle spécifique ou le modèle actuel"""
        if param_name not in ["temperature", "top_p"]:
            return f"Paramètre inconnu: {param_name}. Paramètres disponibles: temperature, top_p"
            
        try:
            value = float(value)
            # Validation des valeurs
            if param_name == "temperature" and (value < 0.0 or value > 1.0):
                return f"La température doit être entre 0.0 et 1.0, valeur reçue: {value}"
            if param_name == "top_p" and (value < 0.0 or value > 1.0):
                return f"Le top_p doit être entre 0.0 et 1.0, valeur reçue: {value}"
                
            # Appliquer le paramètre au modèle spécifié ou au modèle actuel
            target_model = model_type if model_type else self.model_type
            
            if target_model in self.generation_params:
                self.generation_params[target_model][param_name] = value
                logger.info(f"Paramètre {param_name} réglé à {value} pour le modèle {target_model}")
                return f"Paramètre {param_name} réglé à {value} pour le modèle {target_model}."
            else:
                return f"Modèle inconnu: {target_model}"
        except ValueError:
            return f"Erreur: La valeur doit être un nombre à virgule flottante."
            
    def get_generation_params(self, model_type=None):
        """Récupère les paramètres de génération pour un modèle spécifique ou tous les modèles"""
        if model_type:
            if model_type in self.generation_params:
                return {model_type: self.generation_params[model_type]}
            else:
                return {"error": f"Modèle inconnu: {model_type}"}
        else:
            return self.generation_params

    def set_model(self, model_type):
        if model_type == GEMINI_MODEL and not self.gemini_model:
            error_msg = "Impossible de passer au modèle Gemini: Clé API non configurée."
            logger.error(error_msg)
            return error_msg
            
        self.model_type = model_type
        logger.info(f"Modèle changé pour: {model_type}")
        return f"Modèle changé pour: {model_type}"
    
    def get_model_name(self):
        if self.model_type == NEBIUS_MODEL:
            return nebius_model
        elif self.model_type == CLAUDE_MODEL:
            return ANTHROPIC_AI_MODEL
        elif self.model_type == GEMINI_MODEL:
            return GEMINI_AI_MODEL
        else:
            return "Modèle inconnu"

    def _handle_rate_limit(self):
        """Gère la limitation de débit entre les requêtes"""
        current_time = time.time()
        last_time = self.last_request_time.get(self.model_type, 0)
        delay = self.request_delay_seconds.get(self.model_type, 0)
        
        if delay > 0:
            time_since_last = current_time - last_time
            wait_time = delay - time_since_last
            if wait_time > 0:
                logger.info(f"Limitation RPM pour {self.model_type}: Attente de {wait_time:.2f} secondes...")
                print(f"Système: Limitation RPM pour {self.model_type}, attente de {wait_time:.2f} secondes...")
                time.sleep(wait_time)

    def get_chat_completion(self, messages, max_tokens_value):
        # Gestion du délai entre requêtes
        self._handle_rate_limit()
        
        # Mettre à jour l'heure de la dernière requête *avant* l'appel réel
        self.last_request_time[self.model_type] = time.time()

        # Appel à la méthode de complétion appropriée
        try:
            if self.model_type not in self.completion_methods:
                raise ValueError(f"Type de modèle non supporté: {self.model_type}")
                
            if self.model_type == GEMINI_MODEL and not self.gemini_model:
                raise ValueError("Client Gemini non initialisé. Vérifiez la clé API.")
                
            return self.completion_methods[self.model_type](messages, max_tokens_value)
            
        except Exception as e:
            logger.error(f"Erreur pendant l'appel API pour {self.model_type} après gestion du délai: {e}")
            raise e # Propage l'exception pour qu'elle soit gérée plus haut


    def _get_nebius_completion(self, messages, max_tokens_value):
        logger.info(f"Appel au modèle Nebius: {nebius_model}")
        # Récupérer les paramètres de génération pour ce modèle
        params = self.generation_params[NEBIUS_MODEL]
        logger.info(f"Paramètres de génération: temperature={params['temperature']}, top_p={params['top_p']}")
        
        response = self.openai_client.chat.completions.create(
            model=nebius_model,
            messages=messages,
            max_tokens=max_tokens_value,
            temperature=params["temperature"],
            top_p=params["top_p"]
        )
        return response.choices[0].message.content
    
    def _get_anthropic_completion(self, messages, max_tokens_value):
        logger.info(f"Appel au modèle Claude: {ANTHROPIC_AI_MODEL}")
        
        # Récupérer les paramètres de génération pour ce modèle
        params = self.generation_params[CLAUDE_MODEL]
        logger.info(f"Paramètres de génération: temperature={params['temperature']}, top_p={params['top_p']}")
        
        # Conversion du format OpenAI vers Anthropic
        system_message = ""
        anthropic_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            elif msg["role"] == "user":
                anthropic_messages.append({
                    "role": "user",
                    "content": msg["content"]
                })
            elif msg["role"] == "assistant":
                anthropic_messages.append({
                    "role": "assistant",
                    "content": msg["content"]
                })
        
        # Appel à l'API Anthropic
        response = self.anthropic_client.messages.create(
            model=ANTHROPIC_AI_MODEL,
            system=system_message,
            messages=anthropic_messages,
            max_tokens=max_tokens_value,
            temperature=params["temperature"],
            top_p=params["top_p"]
        )
        
        # Vérifier si la réponse est bloquée (Anthropic utilise response.content qui est une liste)
        if not response.content:
            # Tenter d'obtenir la raison de l'arrêt si disponible (peut varier selon l'API version/modèle)
            stop_reason = response.stop_reason
            error_details = f"Réponse vide ou bloquée par Anthropic. Raison d'arrêt: {stop_reason}"
            logger.warning(error_details)
            # Anthropic peut retourner des détails dans response.error ou via des exceptions spécifiques
            return f"Erreur: {error_details}"

        # Extraire le contenu texte du premier bloc (généralement le seul)
        # La structure est response.content = [TextBlock(text='...', type='text')]
        try:
            # Accéder au texte du premier bloc de contenu s'il existe et est de type 'text'
            if response.content and hasattr(response.content[0], 'text'):
                 return response.content[0].text
            else:
                 # Gérer le cas où le contenu est vide ou n'a pas le format attendu
                 error_details = f"Format de réponse Anthropic inattendu ou contenu vide. Contenu brut: {response.content}"
                 logger.warning(error_details)
                 return f"Erreur: {error_details}"
        except Exception as e:
            # Autres erreurs potentielles lors de l'accès au contenu
            error_details = f"Erreur inattendue lors de l'extraction de la réponse Anthropic: {str(e)}"
            logger.error(error_details)
            return f"Erreur: {error_details}"


    def _get_gemini_completion(self, messages, max_tokens_value):
        logger.info(f"Appel au modèle Gemini: {GEMINI_AI_MODEL}")
        
        # Conversion du format OpenAI vers Gemini
        gemini_messages = []
        system_instruction = None
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                # Gemini gère les instructions système séparément
                system_instruction = content
            elif role == "user":
                gemini_messages.append({'role': 'user', 'parts': [content]})
            elif role == "assistant":
                # Le rôle 'assistant' de OpenAI correspond à 'model' dans Gemini
                gemini_messages.append({'role': 'model', 'parts': [content]})
            # Ignorer les autres rôles potentiels pour l'instant

        # Récupérer les paramètres de génération pour ce modèle
        params = self.generation_params[GEMINI_MODEL]
        logger.info(f"Paramètres de génération: temperature={params['temperature']}, top_p={params['top_p']}")
        
        # Configuration de la génération
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens_value,
            temperature=params["temperature"],
            top_p=params["top_p"]
        )

        # Appel à l'API Gemini
        try:
            if system_instruction:
                model_instance = genai.GenerativeModel(
                    GEMINI_AI_MODEL,
                    system_instruction=system_instruction,
                    generation_config=generation_config
                )
            else:
                 model_instance = genai.GenerativeModel(
                    GEMINI_AI_MODEL,
                    generation_config=generation_config
                )

            # S'assurer que l'historique commence par un message 'user' si nécessaire
            # Gemini API requiert une alternance user/model, commençant par user.
            # Si le premier message après le system prompt est 'model', on pourrait avoir un souci.
            # Cependant, notre boucle ajoute toujours 'user' avant 'assistant', donc ça devrait aller.
            # On garde une vérification simple pour le cas où l'historique serait manipulé autrement.
            if gemini_messages and gemini_messages[0]['role'] == 'model':
                 logger.warning("Le premier message pour Gemini est 'model', ce qui pourrait causer une erreur. Ajout d'un message utilisateur vide.")
                 # Optionnel: Insérer un message utilisateur vide si nécessaire, bien que cela puisse altérer la conversation.
                 # gemini_messages.insert(0, {'role': 'user', 'parts': ["Continue."]}) # Alternative: loguer et laisser l'API échouer.

            response = model_instance.generate_content(gemini_messages)

            # Vérifier si la réponse est bloquée
            if not response.candidates:
                 # Essayer d'extraire la raison du blocage si disponible
                try:
                    block_reason = response.prompt_feedback.block_reason
                    # Gemini peut ne pas fournir de block_reason_message, vérifier sa présence
                    block_message = getattr(response.prompt_feedback, 'block_reason_message', 'Non spécifié')
                    error_details = f"Réponse bloquée par Gemini. Raison: {block_reason}. Message: {block_message}"
                except AttributeError:
                     # Si prompt_feedback n'existe pas ou n'a pas block_reason
                     error_details = "Réponse bloquée par Gemini pour une raison inconnue (pas de candidats retournés, feedback indisponible)."
                except Exception as e:
                     error_details = f"Réponse bloquée par Gemini, erreur lors de l'accès aux détails du blocage: {e}"
                logger.warning(error_details)
                return f"Erreur: {error_details}"

            # Vérifier si le contenu est présent
            try:
                # Accéder au texte via response.text
                return response.text
            except ValueError:
                # Si response.text lève une ValueError (par exemple, si bloqué pour sécurité)
                # Essayer d'obtenir plus d'infos depuis les candidats s'ils existent (même si la réponse globale est bloquée)
                finish_reason = "Inconnu"
                safety_ratings = "Indisponible"
                if response.candidates: # Vérifier si la liste candidates existe et n'est pas vide
                    try:
                        candidate = response.candidates[0]
                        finish_reason = getattr(candidate, 'finish_reason', 'Inconnu')
                        safety_ratings = getattr(candidate, 'safety_ratings', 'Indisponible')
                    except Exception as e:
                         logger.error(f"Erreur lors de l'accès aux détails du candidat Gemini: {e}")

                error_details = f"Impossible d'extraire le texte de la réponse Gemini (ValueError). Raison d'arrêt: {finish_reason}. Évaluations de sécurité: {safety_ratings}"
                logger.warning(error_details)
                return f"Erreur: {error_details}"
            except Exception as e:
                 # Autres erreurs potentielles
                 error_details = f"Erreur inattendue lors de l'extraction de la réponse Gemini: {str(e)}"
                 logger.error(error_details)
                 return f"Erreur: {error_details}"

        except Exception as e:
            error_msg = f"Erreur lors de l'appel à l'API Gemini: {str(e)}"
            logger.error(error_msg)
            # Ne pas retourner directement le message d'erreur comme réponse valide, lever ou retourner un indicateur d'erreur
            raise  # Propager l'exception pour une gestion centralisée


# Registre de fonctions disponibles
function_registry = {}

# Fonction pour enregistrer automatiquement toutes les fonctions d'un module
def register_all_functions_from_module(module):
    for name, obj in inspect.getmembers(module):
        # Vérifier si c'est une fonction définie dans ce module (pas importée)
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            function_registry[name] = obj

# Enregistrer toutes les fonctions du module mcp_functions
import mcp_functions
register_all_functions_from_module(mcp_functions)

# Générer la documentation des fonctions pour le system prompt par réflexion
def generate_function_docs():
    docs = []
    # Trier les fonctions par nom pour une documentation plus lisible
    for func_name in sorted(function_registry.keys()):
        func = function_registry[func_name]
        # Convertir les underscores en tirets pour l'affichage
        display_name = func_name.replace("_", "-")
        
        # Obtenir la signature de la fonction
        sig = inspect.signature(func)
        
        # Construire la représentation des paramètres pour la documentation
        params = []
        for param_name, param in sig.parameters.items():
            # Détecter le type à partir des annotations ou par défaut string
            param_type = "string"  # Type par défaut
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == str:
                    param_type = "string"
                elif param.annotation == int:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                else:
                    param_type = str(param.annotation).replace("<class '", "").replace("'>", "")
            
            params.append(f"{param_name}: {param_type}")
        
        # Détecter le type de retour
        return_type = "void"  # Type par défaut
        if func.__annotations__.get('return', None):
            return_annotation = func.__annotations__['return']
            if return_annotation == str:
                return_type = "string"
            elif return_annotation == int:
                return_type = "number"
            elif return_annotation == bool:
                return_type = "boolean"
            elif return_annotation == None or return_annotation == type(None):
                return_type = "void"
            else:
                return_type = str(return_annotation).replace("<class '", "").replace("'>", "")
        
        # Récupérer la docstring comme commentaire
        comment = ""
        if func.__doc__:
            comment = " // " + func.__doc__.strip().split('\n')[0]
        
        # Construire un exemple Gemma
        gemma_params = []
        
        # Ajouter des valeurs d'exemple pour chaque paramètre
        for param_name, param in sig.parameters.items():
            if param.default == inspect.Parameter.empty:
                # Paramètre obligatoire
                if param_type == "string":
                    gemma_params.append(f'{param_name}="valeur_{param_name}"')
                elif param_type == "number":
                    gemma_params.append(f'{param_name}=0')
                elif param_type == "boolean":
                    gemma_params.append(f'{param_name}=True')
                else:
                    gemma_params.append(f'{param_name}=None')
            else:
                # Paramètre optionnel, n'ajouter que si la valeur par défaut n'est pas None
                if param.default is not None:
                    if isinstance(param.default, str):
                        gemma_params.append(f'{param_name}="{param.default}"')
                    else:
                        gemma_params.append(f'{param_name}={param.default}')
        
        # Créer l'exemple Gemma
        gemma_str = f"[{func_name}({', '.join(gemma_params)})]"
        
        # Construire la documentation complète
        function_signature = f"{display_name}({', '.join(params)}): {return_type}{comment}"
        docs.append(function_signature)
    
    return "\n".join(docs)

# Classe pour gérer la connexion au serveur MCP et l'intégration des outils
class MCPToolsProvider:
    def __init__(self, server_script_path=None):
        self.client = None
        self.server_script_path = server_script_path
        self.tools_documentation = ""
        
    async def initialize(self):
        """Initialise la connexion au serveur MCP et récupère les outils disponibles"""
        if not self.server_script_path:
            logger.info("Aucun serveur MCP spécifié, fonctionnement sans outils externes")
            return False
            
        try:
            from mcp_client import MCPClient
            
            self.client = MCPClient()
            await self.client.connect_to_server(self.server_script_path)
            
            # Récupérer la documentation des outils
            self.tools_documentation = await self.client.get_tools_documentation()
            logger.info(f"Documentation des outils MCP récupérée: {self.tools_documentation}")
            
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du client MCP: {str(e)}")
            return False
    
    def get_tools_documentation(self):
        """Retourne la documentation des outils MCP"""
        return self.tools_documentation
        
    async def process_tool_call(self, tool_name, tool_args):
        """Traite un appel d'outil via le client MCP"""
        if not self.client or not self.client.session:
            return "Erreur: Client MCP non initialisé"
            
        try:
            result = await self.client.session.call_tool(tool_name, tool_args)
            return result.content
        except Exception as e:
            error_msg = f"Erreur lors de l'appel à l'outil {tool_name}: {str(e)}"
            logger.error(error_msg)
            return error_msg

# Lire le system prompt et ajouter la documentation des fonctions
def load_system_prompt(mcp_tools_provider=None):
    try:
        with open("system_prompt.fr.txt", "r", encoding="utf-8") as file:
            system_prompt = file.read()
        
        # Ajouter la documentation des fonctions avec la syntaxe JSON
        system_prompt += generate_function_docs()
        
        # Ajouter la documentation des outils MCP si disponible
        if mcp_tools_provider and mcp_tools_provider.get_tools_documentation():
            system_prompt += "\n\n# Outils MCP externes disponibles\n"
            system_prompt += mcp_tools_provider.get_tools_documentation()
        
        return system_prompt
    
    except FileNotFoundError:
        try:
            with open("system_prompt.txt", "r", encoding="utf-8") as file:
                system_prompt = file.read()
            
            # Ajouter la documentation des fonctions avec la syntaxe JSON
            system_prompt += generate_function_docs()
            
            # Ajouter la documentation des outils MCP si disponible
            if mcp_tools_provider and mcp_tools_provider.get_tools_documentation():
                system_prompt += "\n\n# Outils MCP externes disponibles\n"
                system_prompt += mcp_tools_provider.get_tools_documentation()

            return system_prompt
        
        except FileNotFoundError:
            print("Erreur: Les fichiers system_prompt.fr.txt et system_prompt.txt n'ont pas été trouvés.")
            logger.warning("Fichiers system prompt non trouvés, utilisation du prompt par défaut")
            default_prompt = "Vous êtes un assistant IA utile."
            
            # Ajouter la documentation des fonctions avec la syntaxe JSON
            default_prompt += generate_function_docs()
            
            # Ajouter la documentation des outils MCP si disponible
            if mcp_tools_provider and mcp_tools_provider.get_tools_documentation():
                default_prompt += "\n\n# Outils MCP externes disponibles\n"
                default_prompt += mcp_tools_provider.get_tools_documentation()
            
            return default_prompt


# Estimation approximative du nombre de tokens dans un message
def estimate_tokens(text):
    # Estimation grossière : ~4 caractères par token en moyenne pour les langues européennes
    return len(text) // 4

# Estimation du nombre total de tokens dans les messages
def estimate_total_tokens(messages):
    total = 0
    for msg in messages:
        total += estimate_tokens(msg["content"])
    return total

# Vérifier si le seuil de tokens est dépassé
def check_token_threshold(messages):
    total_tokens = estimate_total_tokens(messages)
    logger.info(f"Estimation de tokens: {total_tokens}")
    return total_tokens, total_tokens > memory_tokens_threshold

# Charger un checkpoint mémoire s'il existe
def load_memory_checkpoint():
    try:
        with open("memory_checkpoint.txt", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return None

# Parser pour les appels de fonction au format Gemma
async def parse_function_call(text, mcp_tools_provider=None):
    # Format Gemma/Gemini (fonction avec paramètres nommés)
    #gemma_pattern = r"\[(\w+(?:-\w+)?)\((.*?)\)\]"
    gemma_pattern = r"\[([\w_-]+)\((.*?)\)\]"
    gemma_match = re.search(gemma_pattern, text, re.DOTALL)
    if gemma_match:
        try:
            # Normaliser le nom de la fonction (remplacer les tirets par des underscores)
            function_name = gemma_match.group(1).replace("-", "_")
            params_str = gemma_match.group(2).strip()
            
            # Vérifier si c'est un outil MCP
            is_mcp_tool = False
            if mcp_tools_provider and mcp_tools_provider.client and mcp_tools_provider.client.session:
                response = await mcp_tools_provider.client.session.list_tools()
                mcp_tools = [tool.name for tool in response.tools]
                if function_name in mcp_tools:
                    is_mcp_tool = True
            
            # Pas de paramètres
            if not params_str:
                return function_name, [], {}, is_mcp_tool
            
            # Parser les paramètres
            kwargs = {}
            args = []
            
            # Gestion avancée des paramètres nommés avec support pour les chaînes avec virgules
            # Utiliser une approche plus robuste pour gérer les chaînes entre guillemets
            i = 0
            while i < len(params_str):
                # Trouver le nom du paramètre
                match = re.match(r'(\w+)=', params_str[i:])
                if not match:
                    i += 1
                    continue
                
                param_name = match.group(1)
                i += len(param_name) + 1  # +1 pour le signe =
                
                # Vérifier si la valeur est une chaîne entre guillemets
                if i < len(params_str) and params_str[i] == '"':
                    # Trouver la fin de la chaîne (en tenant compte des guillemets échappés)
                    start = i + 1
                    i += 1
                    while i < len(params_str):
                        if params_str[i] == '"' and params_str[i-1] != '\\':
                            break
                        i += 1
                    
                    if i < len(params_str):
                        param_value = params_str[start:i]
                        i += 1  # Passer le guillemet fermant
                    else:
                        param_value = params_str[start:]
                    
                    # Traiter les caractères d'échappement
                    param_value = param_value.replace('\\"', '"')
                else:
                    # Pour les valeurs non-chaînes (nombres, booléens, etc.)
                    match = re.match(r'([^,]+)', params_str[i:])
                    if match:
                        param_value = match.group(1).strip()
                        i += len(match.group(1))
                        
                        # Convertir les valeurs si nécessaire
                        if param_value.lower() == 'true':
                            param_value = True
                        elif param_value.lower() == 'false':
                            param_value = False
                        elif param_value.isdigit():
                            param_value = int(param_value)
                        elif re.match(r'^-?\d+(\.\d+)?$', param_value):
                            param_value = float(param_value)
                    else:
                        i += 1
                        continue
                
                kwargs[param_name] = param_value
                
                # Passer la virgule si présente
                if i < len(params_str) and params_str[i] == ',':
                    i += 1
                
                kwargs[param_name] = param_value
            
            return function_name, args, kwargs, is_mcp_tool
        except Exception as e:
            error_msg = f"Erreur lors du parsing du format Gemma: {str(e)}"
            logger.error(error_msg)
            return None, None, None, False
    
    return None, None, None, False

# Exécuter une fonction à partir de son nom et ses arguments
def execute_function(function_name, args, kwargs):
    # Normaliser le nom de la fonction (remplacer les tirets par des underscores)
    normalized_name = function_name.replace("-", "_")
    
    if normalized_name in function_registry:
        try:
            logger.info(f"Exécution de la fonction: {normalized_name}")
            return function_registry[normalized_name](*args, **kwargs)
        except Exception as e:
            error_msg = f"Erreur lors de l'exécution de {normalized_name}: {str(e)}"
            logger.error(error_msg)
            return error_msg
    return f"Fonction '{function_name}' non trouvée dans le registre des fonctions."

# Formatage des paramètres pour l'affichage et le log
def format_function_params(args, kwargs):
    params = []
    for arg in args:
        if isinstance(arg, str):
            # Limiter l'affichage des chaînes longues
            if len(arg) > 100:
                display_str = arg[:97] + "..."
                params.append(f'"{display_str}"')
            else:
                params.append(f'"{arg}"')
        else:
            params.append(str(arg))
    
    for key, value in kwargs.items():
        if isinstance(value, str):
            # Limiter l'affichage des chaînes longues
            if len(value) > 100:
                display_str = value[:97] + "..."
                params.append(f'{key}="{display_str}"')
            else:
                params.append(f'{key}="{value}"')
        else:
            params.append(f'{key}={value}')
    
    return ", ".join(params)

# Analyser les arguments de ligne de commande
def parse_args():
    parser = argparse.ArgumentParser(description="Assistant LLM avec fonctions MCP")
    parser.add_argument("--model", choices=[NEBIUS_MODEL, CLAUDE_MODEL, GEMINI_MODEL], default=NEBIUS_MODEL,
                        help=f"Modèle LLM à utiliser ({NEBIUS_MODEL}, {CLAUDE_MODEL} ou {GEMINI_MODEL})")
    parser.add_argument("--mcp-server", type=str, help="Chemin vers le script du serveur MCP")
    return parser.parse_args()

# Fonctions utilitaires pour la boucle principale
# Cette fonction n'est plus utilisée car remplacée par la logique dans main()
# def setup_environment():
#     """Configure l'environnement initial pour la boucle LLM"""
#     # Analyser les arguments
#     args = parse_args()
#     
#     logger.info("Démarrage de l'application")
#     
#     # Initialiser le client LLM avec le modèle choisi
#     llm_client = LLMClient(args.model)
#     
#     system_prompt = load_system_prompt()
#     messages = [{"role": "system", "content": system_prompt}]
#     
#     # Charger un checkpoint mémoire s'il existe
#     checkpoint = load_memory_checkpoint()
#     if checkpoint:
#         messages.append({"role": "user", "content": f"Voici un résumé des échanges précédents pour économiser des jetons:\n\n{checkpoint}"})
#         logger.info("Point de contrôle mémoire chargé")
#     
#     logger.info(f"System prompt chargé: {system_prompt}")
#     logger.info(f"Modèle initial: {llm_client.get_model_name()}")
#     
#     return llm_client, messages

def print_welcome_message(llm_client):
    """Affiche le message de bienvenue et les commandes disponibles"""
    print(f"Démarrage de la boucle d'interaction LLM avec le modèle: {llm_client.get_model_name()}")
    print("Commandes disponibles:")
    print("  /clear   - Effacer l'historique de conversation")
    print("  /compact - Demander au LLM de résumer l'historique (mémoire uniquement)")
    print("  /forget  - Effacer l'historique ET le checkpoint mémoire sur disque")
    print("  /exit, /quit - Quitter la boucle")
    print(f"  /model [{NEBIUS_MODEL}|{CLAUDE_MODEL}|{GEMINI_MODEL}] - Changer de modèle (actuel: {llm_client.model_type})")
    print(f"  /rpm [{NEBIUS_MODEL}|{CLAUDE_MODEL}|{GEMINI_MODEL}] [valeur] - Voir/régler le RPM (Requêtes Par Minute) pour un modèle (0 = illimité)")
    print(f"  /param [temperature|top_p] [valeur] [{NEBIUS_MODEL}|{CLAUDE_MODEL}|{GEMINI_MODEL}] - Configurer les paramètres de génération")
    print(f"  /param [modèle] - Voir les paramètres d'un modèle spécifique")
    print(f"  /param - Voir tous les paramètres de génération")
    print("-" * 50)

async def get_user_input(use_prompt_toolkit=False, session=None):
    """Obtient l'entrée utilisateur avec gestion des interruptions"""
    try:
        if use_prompt_toolkit and session:
            # Utiliser la version asynchrone de prompt
            user_input = await session.prompt_async("Vous: ")
        else:
            # Pour input standard, on utilise un executor pour éviter de bloquer la boucle
            loop = asyncio.get_event_loop()
            user_input = await loop.run_in_executor(None, lambda: input("Vous: "))
        logger.info(f"Entrée utilisateur: {user_input}")
        return user_input
    except EOFError: # Gérer uniquement EOFError (Ctrl+D) ici
        print("\nAu revoir! (EOF reçu)")
        logger.info("Sortie de l'application (EOFError)")
        return None
    except KeyboardInterrupt:
        # Laisser le gestionnaire de signal handle_sigint gérer la logique
        # Si l'utilisateur appuie sur CTRL+C pendant l'input,
        # handle_sigint s'exécute. S'il ne quitte pas,
        # on retourne None pour potentiellement réafficher le prompt
        # ou laisser la boucle principale décider.
        # Retourner une chaîne vide pourrait être une autre option
        # pour simplement réafficher le prompt sans message d'erreur.
        return "" # Retourne une chaîne vide pour réafficher le prompt

def handle_special_commands(user_input, messages, llm_client):
    """Gère les commandes spéciales et retourne True si une commande a été traitée"""
    # Commande de sortie
    if user_input.lower() in ["/exit", "/quit"]:
        print("Au revoir!")
        logger.info("Sortie de l'application (/exit ou /quit)")
        exit(0)  # Quitte immédiatement le programme
    
    # Commande pour effacer l'historique
    if user_input.lower() == "/clear":
        messages[:] = [messages[0]]  # Garder uniquement le system prompt
        print("Historique de conversation effacé.")
        logger.info("Historique de conversation effacé (/clear)")
        return True

    # Commande pour effacer l'historique et le checkpoint
    if user_input.lower() == "/forget":
        messages[:] = [messages[0]]  # Garder uniquement le system prompt
        print("Historique de conversation effacé.")
        logger.info("Historique de conversation effacé (/forget)")
        
        # Supprimer le checkpoint mémoire
        try:
            checkpoint_file = "memory_checkpoint.txt"
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                print(f"Fichier '{checkpoint_file}' supprimé.")
                logger.info(f"Fichier checkpoint '{checkpoint_file}' supprimé (/forget)")
            else:
                print(f"Fichier '{checkpoint_file}' non trouvé, rien à supprimer.")
        except Exception as e:
            error_msg = f"Erreur lors de la suppression de '{checkpoint_file}': {str(e)}"
            print(error_msg)
            logger.error(error_msg)
        return True

    # Commande pour changer de modèle
    if user_input.lower().startswith("/model"):
        parts = user_input.split()
        available_models = [NEBIUS_MODEL, CLAUDE_MODEL, GEMINI_MODEL]
        if len(parts) > 1 and parts[1].lower() in available_models:
            new_model = parts[1].lower()
            result = llm_client.set_model(new_model)
            print(result)
            # Afficher le nom complet du modèle après le changement
            if "Impossible" not in result: # Vérifier si le changement a réussi
                print(f"Modèle actuel: {llm_client.get_model_name()}")
            logger.info(f"Tentative de changement de modèle pour: {new_model}. Résultat: {result}")
        else:
            print(f"Usage: /model [{NEBIUS_MODEL}|{CLAUDE_MODEL}|{GEMINI_MODEL}]")
            print(f"Modèle actuel: {llm_client.model_type} ({llm_client.get_model_name()})")
        return True

    # Commande pour gérer le RPM
    if user_input.lower().startswith("/rpm"):
        parts = user_input.split()
        available_models = [NEBIUS_MODEL, CLAUDE_MODEL, GEMINI_MODEL]
        
        if len(parts) == 1:
            # Afficher les paramètres RPM actuels
            settings = llm_client.get_rpm_settings()
            print("Paramètres RPM actuels:")
            for model, config in settings.items():
                print(f"  - {model}: {config['rpm']} RPM (délai: {config['delay_seconds']}s)")
        elif len(parts) == 3:
            # Définir le RPM pour un modèle
            model_to_set = parts[1].lower()
            try:
                rpm_value = float(parts[2])
                if model_to_set in available_models:
                    result = llm_client.set_rpm(model_to_set, rpm_value)
                    print(result)
                else:
                    print(f"Modèle non reconnu: {model_to_set}. Modèles disponibles: {', '.join(available_models)}")
            except ValueError:
                print("Erreur: La valeur RPM doit être un nombre.")
        else:
            # Mauvais usage
            print(f"Usage: /rpm [{NEBIUS_MODEL}|{CLAUDE_MODEL}|{GEMINI_MODEL}] [valeur]")
            print("   ou: /rpm  (pour voir les paramètres actuels)")
        
        return True
        
    # Commande pour gérer les paramètres de génération (temperature, top_p)
    if user_input.lower().startswith("/param"):
        parts = user_input.split()
        available_models = [NEBIUS_MODEL, CLAUDE_MODEL, GEMINI_MODEL]
        available_params = ["temperature", "top_p"]
        
        if len(parts) == 1:
            # Afficher les paramètres actuels pour tous les modèles
            params = llm_client.get_generation_params()
            print("Paramètres de génération actuels:")
            for model, model_params in params.items():
                print(f"  - {model}:")
                for param_name, param_value in model_params.items():
                    print(f"      {param_name}: {param_value}")
        elif len(parts) == 2 and parts[1].lower() in available_models:
            # Afficher les paramètres pour un modèle spécifique
            model_to_show = parts[1].lower()
            params = llm_client.get_generation_params(model_to_show)
            print(f"Paramètres de génération pour {model_to_show}:")
            for param_name, param_value in params[model_to_show].items():
                print(f"  {param_name}: {param_value}")
        elif len(parts) >= 3 and parts[1].lower() in available_params:
            # Définir un paramètre pour le modèle actuel ou un modèle spécifique
            param_name = parts[1].lower()
            param_value = parts[2]
            
            # Vérifier si un modèle spécifique est fourni
            model_to_set = None
            if len(parts) == 4 and parts[3].lower() in available_models:
                model_to_set = parts[3].lower()
                
            result = llm_client.set_generation_param(param_name, param_value, model_to_set)
            print(result)
        else:
            # Mauvais usage
            print(f"Usage: /param [temperature|top_p] [valeur] [{NEBIUS_MODEL}|{CLAUDE_MODEL}|{GEMINI_MODEL}]")
            print("   ou: /param [modèle]  (pour voir les paramètres d'un modèle)")
            print("   ou: /param  (pour voir tous les paramètres)")
        
        return True

    # Commande pour compacter la mémoire (sans fichier)
    if user_input.lower() == "/compact":
        handle_compact_command(messages, llm_client)
        return True
        
    # Si aucune commande spéciale n'a été traitée
    return False

def handle_compact_command(messages, llm_client):
    """Gère la commande /compact pour résumer la conversation"""
    print("Demande de compaction de la mémoire au LLM...")
    logger.info("Début de la commande /compact")
    
    # Préparer une requête spécifique pour le résumé
    compaction_request_msg = "Résume la conversation actuelle de manière concise pour réduire l'utilisation de la mémoire. Ne pas utiliser la fonction memory-checkpoint. Réponds uniquement avec le résumé."
    temp_messages = messages + [{"role": "user", "content": compaction_request_msg}]
    
    try:
        # Appel au LLM pour obtenir le résumé
        logger.info(f"Envoi de la requête de compaction au modèle {llm_client.get_model_name()}")
        summary = llm_client.get_chat_completion(temp_messages, max_tokens // 2) # Utiliser moins de tokens pour le résumé
        logger.info(f"Résumé reçu pour /compact: {summary}")

        # Réinitialiser l'historique avec le system prompt et le résumé
        system_content = messages[0]["content"]
        messages[:] = [{"role": "system", "content": system_content}]
        messages.append({"role": "user", "content": f"Voici un résumé des échanges précédents (compaction manuelle):\n\n{summary}"})
        
        print("Mémoire compactée avec succès (en mémoire uniquement).")
        logger.info("Mémoire compactée avec succès via /compact")
        
    except Exception as e:
        error_msg = f"Erreur lors de la compaction mémoire (/compact): {str(e)}"
        print(error_msg)
        logger.error(error_msg)

async def process_user_input(user_input, messages, llm_client, mcp_tools_provider=None):
    """Traite l'entrée utilisateur normale et gère l'interaction avec le LLM"""
    # Ajouter l'entrée utilisateur aux messages
    messages.append({"role": "user", "content": user_input})
    
    # Vérifier si le seuil de tokens est dépassé
    total_tokens, threshold_exceeded = check_token_threshold(messages)
    
    # Si le seuil est dépassé, demander explicitement un checkpoint
    if threshold_exceeded:
        checkpoint_request = f"IMPORTANT: L'historique actuel est estimé à environ {total_tokens} tokens, ce qui dépasse le seuil de {memory_tokens_threshold}. Créez immédiatement un résumé avec memory-checkpoint."
        print(f"Système: {checkpoint_request}")
        logger.info(f"Demande explicite de checkpoint: {total_tokens} > {memory_tokens_threshold}")
        messages.append({"role": "user", "content": checkpoint_request})
    
    # Boucle de traitement des réponses et appels de fonction
    await process_llm_responses(messages, llm_client, mcp_tools_provider)

async def process_llm_responses(messages, llm_client, mcp_tools_provider=None):
    """Gère les réponses du LLM et les appels de fonction potentiels"""
    global interrupt_requested # Nécessaire pour vérifier et réinitialiser
    while True:
        # Réinitialiser le flag d'interruption avant chaque tentative d'appel LLM
        interrupt_requested = False
        try:
            # Appel au LLM
            logger.info(f"Envoi de la requête au modèle {llm_client.get_model_name()}")
            assistant_response = llm_client.get_chat_completion(messages, max_tokens)
            logger.info(f"Réponse du modèle reçue: {assistant_response}")

            # Si l'appel réussit sans interruption, s'assurer que le flag est bien False
            # (Normalement déjà fait au début de la boucle, mais par sécurité)
            interrupt_requested = False
            
            # Ajouter la réponse aux messages
            messages.append({"role": "assistant", "content": assistant_response})
            
            # Vérifier à nouveau le seuil de tokens après la réponse
            check_token_threshold_after_response(messages)
            
            # Vérifier si la réponse contient un appel de fonction
            function_name, args, kwargs, is_mcp_tool = await parse_function_call(assistant_response, mcp_tools_provider)
            
            if function_name:
                # Traiter l'appel de fonction et vérifier s'il faut continuer la boucle
                should_continue = await handle_function_call(function_name, args, kwargs, messages, is_mcp_tool, mcp_tools_provider)
                if not should_continue:
                    # Si la fonction a retourné None, sortir de la boucle
                    print("Système: Fonction exécutée, attente de la prochaine entrée utilisateur.")
                    break
            else:
                # Pas d'appel de fonction, afficher la réponse et sortir de la boucle
                print(f"Assistant: {assistant_response}")
                break

        except KeyboardInterrupt:
            if interrupt_requested:
                # Interruption gérée par notre handler pendant l'appel LLM
                print("Requête LLM annulée.")
                logger.warning("Requête LLM annulée par l'utilisateur (CTRL+C)")
                # Pas besoin de réinitialiser interrupt_requested ici,
                # car on sort de la boucle et il sera réinitialisé
                # avant le prochain appel dans process_llm_responses.
                break # Sortir de la boucle while True de process_llm_responses
            else:
                # Interruption inattendue (ne devrait pas arriver si le handler fonctionne)
                print("\nInterruption clavier inattendue.")
                logger.warning("KeyboardInterrupt inattendue interceptée dans process_llm_responses")
                # Peut-être quitter ou simplement sortir de la boucle de traitement
                break # Sortir de la boucle while True

        except Exception as e:
            error_msg = f"Erreur lors de l'appel au LLM: {str(e)}"
            print(error_msg)
            logger.error(error_msg)
            # Réinitialiser le flag en cas d'erreur autre qu'une interruption
            interrupt_requested = False
            break

def check_token_threshold_after_response(messages):
    """Vérifie si le seuil de tokens est dépassé après une réponse du LLM"""
    total_tokens, threshold_exceeded = check_token_threshold(messages)
    
    if threshold_exceeded:
        checkpoint_request = f"IMPORTANT: L'historique actuel est estimé à environ {total_tokens} tokens, ce qui dépasse le seuil de {memory_tokens_threshold}. Créez immédiatement un résumé avec memory-checkpoint."
        logger.info(f"Seuil de tokens dépassé après réponse: {total_tokens} > {memory_tokens_threshold}")
        
        # Vérifier si un message de demande de checkpoint a déjà été envoyé récemment
        recent_checkpoint_request = False
        if len(messages) >= 4:
            for i in range(1, 5):  # Vérifier les 4 derniers messages
                if i <= len(messages) and "role" in messages[-i] and messages[-i]["role"] == "user" and "memory-checkpoint" in messages[-i]["content"]:
                    recent_checkpoint_request = True
                    break
        
        if not recent_checkpoint_request:
            print(f"Système: {checkpoint_request}")
            messages.append({"role": "user", "content": checkpoint_request})

async def handle_function_call(function_name, args, kwargs, messages, is_mcp_tool=False, mcp_tools_provider=None):
    """Gère un appel de fonction détecté dans la réponse du LLM"""
    # Formater les paramètres pour l'affichage
    params_display = format_function_params(args, kwargs)
    
    function_call_message = f"Assistant a appelé {'l\'outil MCP' if is_mcp_tool else 'la fonction'}: {function_name}({params_display})"
    print(function_call_message)
    logger.info(f"Appel {'d\'outil MCP' if is_mcp_tool else 'de fonction'} détecté: {function_name}({params_display})")
    
    # Pour write_file et patch_file, enregistrer le contenu détaillé dans les logs pour debugging
    if not is_mcp_tool and function_name in ["write_file", "patch_file"] and len(args) >= 2:
        logger.debug(f"Contenu pour '{args[0]}': {repr(args[1])}")
    
    # Exécuter la fonction ou l'outil MCP
    if is_mcp_tool and mcp_tools_provider:
        # Convertir les arguments en format attendu par MCP
        tool_args = {}
        if kwargs:
            tool_args = kwargs
        
        function_result = await mcp_tools_provider.process_tool_call(function_name, tool_args)
    else:
        # Pour les fonctions internes
        function_result = execute_function(function_name, args, kwargs)
    
    logger.info(f"Résultat de {'l\'outil MCP' if is_mcp_tool else 'la fonction'}: {function_result}")
    
    # Si c'est un checkpoint mémoire, réinitialiser l'historique
    if function_name == "memory_checkpoint":
        handle_memory_checkpoint(messages, args)
    
    # Si la fonction retourne None, indiquer qu'il faut arrêter la boucle
    if function_result is None:
        logger.info(f"La fonction {function_name} a retourné None, arrêt de la boucle")
        return False
    
    # Ajouter le résultat comme message utilisateur
    messages.append({"role": "user", "content": f"Résultat de {function_name}: {function_result}"})
    
    # Par défaut, continuer la boucle
    return True

def handle_memory_checkpoint(messages, args):
    """Gère la réinitialisation de l'historique après un checkpoint mémoire"""
    # Garder seulement le system prompt et ajouter le résumé
    system_content = messages[0]["content"]
    checkpoint_content = args[0] if args else ""
    
    # Réinitialiser les messages avec seulement le system prompt
    messages[:] = [{"role": "system", "content": system_content}]
    
    # Ajouter le checkpoint comme premier message utilisateur
    messages.append({"role": "user", "content": f"Voici un résumé des échanges précédents pour économiser des jetons:\n\n{checkpoint_content}"})
    
    memory_reset_message = "MEMORY-CHECKPOINT invoqué : Mémoire réinitialisée avec point de contrôle."
    print(memory_reset_message)
    logger.info("Mémoire réinitialisée avec memory-checkpoint")

# Configuration de l'interface de saisie
def setup_prompt_interface():
    """Configure l'interface de saisie utilisateur avec prompt_toolkit si disponible"""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        
        # Utiliser prompt_toolkit si disponible
        use_prompt_toolkit = True
        session = PromptSession(
            history=FileHistory('.llm_history'),
            auto_suggest=AutoSuggestFromHistory()
        )
    except ImportError:
        # Fallback vers input standard si prompt_toolkit n'est pas installé
        use_prompt_toolkit = False
        session = None
        print("Conseil: installez 'prompt_toolkit' pour une meilleure expérience (historique, flèches): pip install prompt_toolkit")
    
    return use_prompt_toolkit, session

# Boucle principale d'interaction avec le LLM
async def main():
    # Analyser les arguments
    parser = argparse.ArgumentParser(description="Assistant LLM avec fonctions MCP")
    parser.add_argument("--model", choices=[NEBIUS_MODEL, CLAUDE_MODEL, GEMINI_MODEL], default=NEBIUS_MODEL,
                        help=f"Modèle LLM à utiliser ({NEBIUS_MODEL}, {CLAUDE_MODEL} ou {GEMINI_MODEL})")
    parser.add_argument("--mcp-server", type=str, help="Chemin vers le script du serveur MCP")
    args = parser.parse_args()
    
    logger.info("Démarrage de l'application")
    
    # Initialiser le client MCP si un serveur est spécifié
    mcp_tools_provider = None
    if args.mcp_server:
        mcp_tools_provider = MCPToolsProvider(args.mcp_server)
        mcp_initialized = await mcp_tools_provider.initialize()
        if mcp_initialized:
            logger.info(f"Client MCP initialisé avec succès pour le serveur: {args.mcp_server}")
            print(f"Client MCP connecté au serveur: {args.mcp_server}")
        else:
            logger.warning(f"Échec de l'initialisation du client MCP pour le serveur: {args.mcp_server}")
            print(f"Échec de la connexion au serveur MCP: {args.mcp_server}")
            mcp_tools_provider = None
    
    # Configuration de l'interface de saisie
    use_prompt_toolkit, session = setup_prompt_interface()
    # Enregistrer le gestionnaire de signal pour SIGINT
    signal.signal(signal.SIGINT, handle_sigint)

    # Configuration initiale
    llm_client = LLMClient(args.model)
    
    system_prompt = load_system_prompt(mcp_tools_provider)
    messages = [{"role": "system", "content": system_prompt}]
    
    # Charger un checkpoint mémoire s'il existe
    checkpoint = load_memory_checkpoint()
    if checkpoint:
        messages.append({"role": "user", "content": f"Voici un résumé des échanges précédents pour économiser des jetons:\n\n{checkpoint}"})
        logger.info("Point de contrôle mémoire chargé")

    # Afficher le message de bienvenue
    print_welcome_message(llm_client)
    
    # Boucle principale
    while True:
        # Obtenir l'entrée utilisateur
        user_input = await get_user_input(use_prompt_toolkit, session)
        
        # Sortir si l'utilisateur a fait Ctrl+D (EOFError retourne None)
        # ou si get_user_input retourne None pour une autre raison.
        if user_input is None:
            break

        # Si l'utilisateur a fait CTRL+C pendant l'input, user_input est ""
        # On ignore et on réaffiche le prompt.
        if user_input == "":
            continue

        # Traiter les commandes spéciales
        if handle_special_commands(user_input, messages, llm_client):
            continue
            
        # Traitement normal de l'entrée utilisateur
        await process_user_input(user_input, messages, llm_client, mcp_tools_provider)

if __name__ == "__main__":
    asyncio.run(main())
