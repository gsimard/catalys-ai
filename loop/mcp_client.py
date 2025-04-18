import asyncio
import socket
import logging # Ajouter l'import logging
from typing import Optional, Tuple, Union
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from tcp import tcp_client # Correction: Importer depuis le fichier local tcp.py

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

# Récupérer le logger (peut être configuré dans llm_loop.py ou ici)
logger = logging.getLogger(__name__)

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.connection_type = None  # 'stdio' or 'tcp'

    async def connect_to_server(self, server_path_or_address: str):
        """Connect to an MCP server via stdio or TCP
        
        Args:
            server_path_or_address: Either a path to server script (.py/.js) or a TCP address (host:port)
        """
        # Check if it's a TCP address (host:port)
        if ':' in server_path_or_address and not server_path_or_address.endswith(('.py', '.js')):
            await self._connect_via_tcp(server_path_or_address)
        else:
            await self._connect_via_stdio(server_path_or_address)
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print(f"\nConnected to server ({self.connection_type}) with tools:", [tool.name for tool in tools])
    
    async def _connect_via_stdio(self, server_script_path: str):
        """Connect to an MCP server via stdio
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        self.connection_type = 'stdio'
    
    async def _connect_via_tcp(self, address: str):
        """Connect to an MCP server via TCP
        
        Args:
            address: TCP address in format "host:port"
        """
        try:
            host, port_str = address.split(':')
            port = int(port_str)
            logger.info(f"Tentative de connexion TCP à {host}:{port}...")
            
            # Utiliser un timeout pour open_connection
            try:
                tcp_transport = await asyncio.wait_for(
                    self.exit_stack.enter_async_context(tcp_client(host, port)),
                    timeout=10.0 # Timeout de 10 secondes pour la connexion
                )
                logger.info(f"Connexion TCP établie avec {host}:{port}")
            except asyncio.TimeoutError:
                logger.error(f"Timeout lors de la connexion TCP à {host}:{port}")
                raise ConnectionError(f"Timeout lors de la connexion à {address}")
            except ConnectionRefusedError as e:
                 logger.error(f"Connexion refusée par {host}:{port}. Vérifiez que le serveur écoute.")
                 raise ConnectionRefusedError(f"Connexion refusée à {address}. Serveur en cours d'exécution?") from e
            except OSError as e:
                 # Autres erreurs réseau potentielles (ex: No route to host)
                 logger.error(f"Erreur réseau lors de la connexion à {host}:{port}: {e}")
                 raise ConnectionError(f"Erreur réseau lors de la connexion à {address}: {e}") from e

            self.stdio, self.write = tcp_transport
            logger.info("Transport TCP obtenu, création de la session MCP...")
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            logger.info("Session MCP créée, initialisation...")
            
            try:
                # Utiliser un timeout pour l'initialisation MCP
                await asyncio.wait_for(self.session.initialize(), timeout=15.0) # Timeout de 15s pour l'initialisation
                logger.info("Initialisation de la session MCP réussie.")
                self.connection_type = 'tcp'
            except asyncio.TimeoutError:
                 logger.error(f"Timeout lors de l'initialisation de la session MCP avec {address}")
                 raise TimeoutError(f"Timeout lors de l'initialisation MCP avec {address}")
            except Exception as e:
                 logger.error(f"Erreur lors de l'initialisation de la session MCP avec {address}: {e}", exc_info=True)
                 raise RuntimeError(f"Erreur d'initialisation MCP avec {address}: {e}") from e

        except ValueError:
            logger.error(f"Format d'adresse TCP invalide: {address}")
            raise ValueError(f"Format d'adresse TCP invalide: {address}. Format attendu: host:port")
        # ConnectionRefusedError est maintenant gérée dans le bloc try interne
        # except ConnectionRefusedError:
        #     logger.error(f"Connexion refusée à {address}. Assurez-vous que le serveur est lancé.")
        #     raise ConnectionRefusedError(f"Connexion refusée à {address}. Assurez-vous que le serveur est lancé.")

    async def get_tools_documentation(self) -> str:
        """Récupère la documentation des outils disponibles sur le serveur MCP
        
        Returns:
            Une chaîne formatée contenant la documentation de tous les outils
        """
        if not self.session:
            return "Erreur: Non connecté au serveur MCP"
            
        response = await self.session.list_tools()
        tools = response.tools
        
        docs = []
        for tool in tools:
            # Formater le nom de l'outil
            tool_name = tool.name
            
            # Récupérer la description
            description = tool.description or "Pas de description disponible"
            
            # Analyser le schéma d'entrée pour extraire les paramètres
            params = []
            if tool.inputSchema and "properties" in tool.inputSchema:
                properties = tool.inputSchema.get("properties", {})
                required = tool.inputSchema.get("required", [])
                
                for param_name, param_info in properties.items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    is_required = param_name in required
                    
                    param_str = f"{param_name}: {param_type}"
                    if is_required:
                        param_str += " (obligatoire)"
                    if param_desc:
                        param_str += f" - {param_desc}"
                        
                    params.append(param_str)
            
            # Construire la documentation de l'outil
            tool_doc = f"{tool_name}({', '.join(params)})"
            if description:
                tool_doc += f" // {description}"
                
            docs.append(tool_doc)
        
        return "\n".join(docs)
        
    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{ 
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

        # Initial Claude API call
        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )

        # Process response and handle tool calls
        final_text = []

        for content in response.content:
            if content.type == 'text':
                final_text.append(content.text)
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input
                
                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                # Continue conversation with tool results
                if hasattr(content, 'text') and content.text:
                    messages.append({
                      "role": "assistant",
                      "content": content.text
                    })
                messages.append({
                    "role": "user", 
                    "content": result.content
                })

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                )

                final_text.append(response.content[0].text)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())
