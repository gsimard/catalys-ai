import asyncio
from typing import Tuple, AsyncIterator, Any, Optional
from contextlib import asynccontextmanager

@asynccontextmanager
async def tcp_client(host: str, port: int) -> AsyncIterator[Tuple[AsyncIterator[Any], Any]]:
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
    
    try:
        # Yield the reader stream function and the raw writer object
        yield read_stream(), writer 
    finally:
        # Ensure the writer is closed properly
        if not writer.is_closing():
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass # Ignore errors during cleanup, might already be closed
        # Original cleanup logic (redundant now but kept for safety)
        # writer.close() 
        # try:
            await writer.wait_closed()
        except:
            pass  # Ignore errors during cleanup
