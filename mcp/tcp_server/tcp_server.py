#!/usr/bin/env python3
import asyncio
import logging
import argparse
import importlib.util
import sys
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_tcp_server")

class MCPTCPServer:
    def __init__(self, host='0.0.0.0', port=8765):
        self.host = host
        self.port = port
        self.server = None
        self.clients = set()
        
    async def start(self, module_path):
        """Start the TCP server and load the MCP module"""
        # Import the MCP module dynamically
        try:
            # Get the absolute path
            abs_path = os.path.abspath(module_path)
            logger.info(f"Loading MCP module from: {abs_path}")
            
            # Extract module name from file path
            module_name = Path(module_path).stem
            
            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, abs_path)
            if spec is None:
                logger.error(f"Could not load module spec from {abs_path}")
                return False
                
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Get the app instance
            if not hasattr(module, 'app'):
                logger.error(f"Module {module_name} does not have an 'app' attribute")
                return False
                
            self.app = module.app
            logger.info(f"Successfully loaded MCP app: {self.app.name}")
            
            # Start the server
            self.server = await asyncio.start_server(
                self.handle_client, self.host, self.port
            )
            
            addr = self.server.sockets[0].getsockname()
            logger.info(f'MCP TCP Server running on {addr[0]}:{addr[1]}')
            
            async with self.server:
                await self.server.serve_forever()
                
        except Exception as e:
            logger.error(f"Error starting MCP TCP server: {str(e)}", exc_info=True)
            return False
    
    async def handle_client(self, reader, writer):
        """Handle a client connection"""
        addr = writer.get_extra_info('peername')
        client_id = f"{addr[0]}:{addr[1]}"
        logger.info(f"New client connected: {client_id}")
        
        self.clients.add((reader, writer))
        
        try:
            # Run the MCP app with this client's streams
            await self.app.run(
                reader, 
                writer,
                self.app.create_initialization_options()
            )
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {str(e)}", exc_info=True)
        finally:
            # Clean up
            if (reader, writer) in self.clients:
                self.clients.remove((reader, writer))
            
            try:
                writer.close()
                await writer.wait_closed()
                logger.info(f"Client disconnected: {client_id}")
            except Exception as e:
                logger.error(f"Error closing client connection {client_id}: {str(e)}")

async def main():
    parser = argparse.ArgumentParser(description="MCP TCP Server")
    parser.add_argument("module_path", help="Path to the MCP module file")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    
    args = parser.parse_args()
    
    server = MCPTCPServer(args.host, args.port)
    await server.start(args.module_path)

if __name__ == "__main__":
    asyncio.run(main())
