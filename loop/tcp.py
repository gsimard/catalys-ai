import asyncio
from typing import Tuple, AsyncIterator, Any
from contextlib import asynccontextmanager
import asyncio

# Importer les composants nécessaires de MCP pour l'encodage
# Correction: Essayer d'importer depuis mcp.shared (basé sur la trace d'erreur originale)
from mcp.shared import JSONRPCMessage, Encoder 

# Wrapper pour asyncio.StreamWriter fournissant la méthode send attendue par MCP
class MCPStreamWriterWrapper:
    def __init__(self, writer: asyncio.StreamWriter):
        self._writer = writer
        self._encoder = Encoder() # Utiliser l'encodeur de MCP

    async def send(self, message: JSONRPCMessage) -> None:
        """Encode et envoie un message JSONRPC."""
        try:
            encoded_message = self._encoder.encode(message)
            self._writer.write(encoded_message + b'\n') # MCP ajoute une nouvelle ligne
            await self._writer.drain()
        except ConnectionError as e:
            # Gérer les erreurs de connexion potentielles lors de l'écriture
            print(f"Erreur de connexion lors de l'envoi: {e}")
            # Vous pourriez vouloir lever une exception personnalisée ou logger ici
            raise
        except Exception as e:
            # Gérer d'autres erreurs potentielles
            print(f"Erreur inattendue lors de l'envoi: {e}")
            raise

    # Pas besoin de méthode close ici, car le context manager tcp_client gère la fermeture du writer original.

@asynccontextmanager
async def tcp_client(host: str, port: int) -> AsyncIterator[Tuple[AsyncIterator[bytes], MCPStreamWriterWrapper]]:
    """
    Connect to an MCP server over TCP.
    
    Args:
        host: The hostname or IP address of the MCP server
        port: The port number of the MCP server
        
    Returns:
        A tuple of (reader_stream, writer_function)
    """
    reader, writer = await asyncio.open_connection(host, port)
    
    async def read_stream() -> AsyncIterator[bytes]:
        while True:
            data = await reader.readline()
            if not data:
                break
            yield data
    
    # Créer le wrapper pour le writer
    mcp_writer = MCPStreamWriterWrapper(writer)
    
    try:
        # Yield the reader stream function and the MCP writer wrapper object
        yield read_stream(), mcp_writer
    finally:
        # Ensure the original writer is closed properly
        if not writer.is_closing():
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass # Ignore errors during cleanup, might already be closed
