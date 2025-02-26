import asyncio
import time
import websockets
import logging
import json
import signal
from typing import List, Union, Generator, Iterator

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(message)s",
    level=logging.INFO,
)

class Pipeline:
    def __init__(self):
        self.name = "WebSocket Pipeline"
        self.websocket = None
        self.server_host = "crewai-container"  # Adjust to your container name
        self.server_port = "8765"
        self.connected = False
        self.terminate = False
        # To store pending responses
        self.responses = {}
        self.response_events = {}

    async def on_startup(self):
        """Called when the server is started."""
        logging.info(f"Starting pipeline: {self.name}")
        # Start the websocket connection
        asyncio.create_task(self.connect_websocket())

    async def on_shutdown(self):
        """Called when the server is shutdown."""
        logging.info(f"Shutting down pipeline: {self.name}")
        self.terminate = True
        if self.websocket and self.connected:
            try:
                await self.websocket.send(json.dumps({"action": "terminate"}))
                await self.websocket.close()
            except Exception as e:
                logging.error(f"Error during shutdown: {e}")

    async def connect_websocket(self):
        """Establish and maintain a websocket connection to the server."""
        while not self.terminate:
            try:
                logging.info(f"Connecting to ws://{self.server_host}:{self.server_port}")
                async with websockets.connect(
                    f"ws://{self.server_host}:{self.server_port}",
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10,
                    max_size=10 * 1024 * 1024  # 10MB max message size
                ) as websocket:
                    self.websocket = websocket
                    self.connected = True
                    logging.info("Connected to server")
                    
                    # Listen for messages from the server
                    await self.listen_for_messages()
            except (websockets.exceptions.ConnectionError, 
                    websockets.exceptions.InvalidStatusCode,
                    ConnectionRefusedError) as e:
                self.connected = False
                if not self.terminate:
                    logging.error(f"Connection error: {e}. Retrying in 5 seconds...")
                    await asyncio.sleep(5)
                else:
                    break
            except Exception as e:
                self.connected = False
                logging.error(f"Unexpected connection error: {e}", exc_info=True)
                if not self.terminate:
                    await asyncio.sleep(5)
                else:
                    break

    async def listen_for_messages(self):
        """Listen for incoming messages from the server."""
        try:
            async for message in self.websocket:
                logging.info(f"Received raw message: {message[:200]}...")  # Log first 200 chars
                try:
                    # Parse the response
                    response_data = json.loads(message)
                    message_id = response_data.get("original_id")
                    
                    logging.info(f"Parsed message_id: {message_id}, active event keys: {list(self.response_events.keys())}")
                    
                    if message_id is not None:
                        # Store the response
                        self.responses[message_id] = response_data
                        if message_id in self.response_events:
                            logging.info(f"Setting event for message_id: {message_id}")
                            self.response_events[message_id].set()
                        else:
                            logging.warning(f"Received response for unknown message_id: {message_id}")
                except json.JSONDecodeError as e:
                    logging.error(f"Invalid JSON response: {message}. Error: {e}")
                except Exception as e:
                    logging.error(f"Error processing response: {e}", exc_info=True)
        except websockets.exceptions.ConnectionClosed as e:
            logging.error(f"Connection to server was closed: code={e.code}, reason={e.reason}")
            self.connected = False
        except Exception as e:
            logging.error(f"Error in message listener: {e}", exc_info=True)
            self.connected = False

    async def send_message(self, user_message, message_id):
        """Send a message to the server and wait for the response."""
        if not self.connected:
            # Try to reconnect if not connected
            logging.warning("Not connected to server. Attempting to reconnect...")
            # Create a task to reconnect but don't wait for it to complete
            reconnect_task = asyncio.create_task(self.connect_websocket())
            
            # Wait a short time for the connection to establish
            for _ in range(10):  # Try for 5 seconds (10 * 0.5s)
                if self.connected:
                    break
                await asyncio.sleep(0.5)
            
            if not self.connected:
                logging.error("Failed to reconnect to server")
                return {"error": "Could not connect to server"}
        
        try:
            # Initialize response tracking before sending
            # This prevents race conditions if response arrives very quickly
            self.responses[message_id] = None
            self.response_events[message_id] = asyncio.Event()
            
            # Prepare the message
            data = {
                "id": message_id,
                "action": "process",
                "data": user_message
            }
            
            # Send the message
            message = json.dumps(data)
            logging.info(f"Sending message with ID {message_id}: {message}")
            await self.websocket.send(message)
            
            # Wait for the response with timeout
            try:
                logging.info(f"Waiting for response to message {message_id}")
                # Increased timeout to 120 seconds (2 minutes)
                await asyncio.wait_for(self.response_events[message_id].wait(), timeout=120)
                
                # Get the response
                response = self.responses.pop(message_id, {"error": "No response received"})
                logging.info(f"Got response for message {message_id}: {str(response)[:200]}...")
                return response
            except asyncio.TimeoutError:
                logging.error(f"Timeout waiting for response to message {message_id}")
                return {"error": "Timeout waiting for response"}
        except websockets.exceptions.ConnectionClosed as e:
            logging.error(f"Connection closed while sending message: {e}")
            self.connected = False
            return {"error": f"Connection closed: {e}"}
        except Exception as e:
            logging.error(f"Error sending message: {e}", exc_info=True)
            return {"error": str(e)}
        finally:
            # Clean up to prevent memory leaks
            self.response_events.pop(message_id, None)
    
    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        """Process a user message and return the response."""
        logging.info(f"Processing user message with pipeline: {user_message[:100]}...")
        
        # Create a unique message ID
        message_id = hash(user_message + str(time.time()))
        
        # Use an event loop to send the message and wait for the response
        loop = asyncio.new_event_loop()
        try:
            response_data = loop.run_until_complete(self.send_message(user_message, message_id))
            
            # Extract the result from the response
            if "error" in response_data:
                error_message = response_data["error"]
                logging.error(f"Error in pipeline response: {error_message}")
                return f"Error: {error_message}"
            
            result = response_data.get("result", "No result returned from CrewAI")
            logging.info(f"Pipeline returning result: {result[:100]}...")
            return result
        finally:
            loop.close()