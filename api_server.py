"""
RealAI API Server

A simple HTTP server that provides an OpenAI-compatible REST API.
This allows you to use RealAI with any OpenAI-compatible client libraries.

Run with: python api_server.py
"""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from realai import RealAI


class RealAIAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for RealAI API."""
    
    def __init__(self, *args, **kwargs):
        self.model = RealAI()
        super().__init__(*args, **kwargs)
    
    def _send_response(self, status_code: int, data: dict):
        """Send JSON response."""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _read_body(self) -> dict:
        """Read and parse JSON body."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        return json.loads(body.decode()) if body else {}
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/v1/models':
            # List available models
            response = {
                "object": "list",
                "data": [
                    {
                        "id": "realai-1.0",
                        "object": "model",
                        "created": 1708308000,
                        "owned_by": "realai",
                        "permission": [],
                        "root": "realai-1.0",
                        "parent": None,
                    }
                ]
            }
            self._send_response(200, response)
        
        elif parsed_path.path == '/v1/models/realai-1.0':
            # Get specific model info
            response = self.model.get_model_info()
            response["object"] = "model"
            response["id"] = self.model.model_name
            self._send_response(200, response)
        
        elif parsed_path.path == '/health':
            # Health check endpoint
            self._send_response(200, {"status": "healthy", "model": "realai-1.0"})
        
        else:
            self._send_response(404, {"error": "Not found"})
    
    def do_POST(self):
        """Handle POST requests."""
        parsed_path = urlparse(self.path)
        
        try:
            body = self._read_body()
            
            if parsed_path.path == '/v1/chat/completions':
                # Chat completion endpoint
                response = self.model.chat_completion(
                    messages=body.get('messages', []),
                    temperature=body.get('temperature', 0.7),
                    max_tokens=body.get('max_tokens'),
                    stream=body.get('stream', False)
                )
                self._send_response(200, response)
            
            elif parsed_path.path == '/v1/completions':
                # Text completion endpoint
                response = self.model.text_completion(
                    prompt=body.get('prompt', ''),
                    temperature=body.get('temperature', 0.7),
                    max_tokens=body.get('max_tokens')
                )
                self._send_response(200, response)
            
            elif parsed_path.path == '/v1/images/generations':
                # Image generation endpoint
                response = self.model.generate_image(
                    prompt=body.get('prompt', ''),
                    size=body.get('size', '1024x1024'),
                    quality=body.get('quality', 'standard'),
                    n=body.get('n', 1)
                )
                self._send_response(200, response)
            
            elif parsed_path.path == '/v1/embeddings':
                # Embeddings endpoint
                response = self.model.create_embeddings(
                    input_text=body.get('input', ''),
                    model=body.get('model', 'realai-embeddings')
                )
                self._send_response(200, response)
            
            elif parsed_path.path == '/v1/audio/transcriptions':
                # Audio transcription endpoint
                response = self.model.transcribe_audio(
                    audio_file=body.get('file', ''),
                    language=body.get('language'),
                    prompt=body.get('prompt')
                )
                self._send_response(200, response)
            
            elif parsed_path.path == '/v1/audio/speech':
                # Audio generation endpoint
                response = self.model.generate_audio(
                    text=body.get('input', ''),
                    voice=body.get('voice', 'alloy'),
                    model=body.get('model', 'realai-tts')
                )
                self._send_response(200, response)
            
            else:
                self._send_response(404, {"error": "Endpoint not found"})
        
        except json.JSONDecodeError:
            self._send_response(400, {"error": "Invalid JSON"})
        except Exception as e:
            self._send_response(500, {"error": str(e)})
    
    def log_message(self, format, *args):
        """Log API requests."""
        print(f"[{self.log_date_time_string()}] {format % args}")


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start the RealAI API server.
    
    Args:
        host (str): Host to bind to
        port (int): Port to listen on
    """
    server_address = (host, port)
    httpd = HTTPServer(server_address, RealAIAPIHandler)
    
    print("="*60)
    print("RealAI API Server")
    print("="*60)
    print(f"Server running at http://{host}:{port}")
    print("\nAvailable endpoints:")
    print("  GET  /health")
    print("  GET  /v1/models")
    print("  GET  /v1/models/realai-1.0")
    print("  POST /v1/chat/completions")
    print("  POST /v1/completions")
    print("  POST /v1/images/generations")
    print("  POST /v1/embeddings")
    print("  POST /v1/audio/transcriptions")
    print("  POST /v1/audio/speech")
    print("\nPress Ctrl+C to stop the server")
    print("="*60)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        httpd.shutdown()


if __name__ == "__main__":
    run_server()
