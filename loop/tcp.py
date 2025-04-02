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
    
    async def write_function(data: bytes) -> None:
        writer.write(data)
        await writer.drain()
    
    try:
        yield read_stream(), write_function
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except:
            pass  # Ignore errors during cleanup
