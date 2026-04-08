"""
RealAI API Server

A simple HTTP server that provides an OpenAI-compatible REST API.
This allows you to use RealAI with any OpenAI-compatible client libraries.

Run with: python api_server.py

API Key handling
----------------
Pass your provider API key in the standard ``Authorization: Bearer <key>``
header.  RealAI auto-detects the provider from the key prefix and forwards
requests to the real AI service.  You can also supply ``X-Provider`` to pick
the provider explicitly, and ``X-Base-URL`` to override the endpoint.
"""

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from realai import RealAI, PROVIDER_CONFIGS, PROVIDER_ENV_VARS


class RealAIAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for RealAI API."""

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

    def _get_model(self, model_name: str = "realai-2.0") -> RealAI:
        """Build a :class:`~realai.RealAI` instance from request headers.

        Reads the API key from the ``Authorization: Bearer <key>`` header, the
        optional provider override from ``X-Provider``, and the optional base
        URL from ``X-Base-URL``.

        When no ``Authorization`` header is present the method falls back to
        ``REALAI_<PROVIDER>_API_KEY`` environment variables so the GUI launcher
        can pass keys via the process environment without requiring callers to
        set the header explicitly.
        """
        auth = self.headers.get("Authorization", "")
        api_key = auth[len("Bearer "):].strip() if auth.startswith("Bearer ") else None
        provider = self.headers.get("X-Provider") or None
        base_url = self.headers.get("X-Base-URL") or None

        # Fall back to environment variables set by the GUI launcher.
        # Priority follows the insertion order of PROVIDER_ENV_VARS
        # (openai → anthropic → grok → gemini); the first key found wins.
        if not api_key:
            for _provider, _env_var in PROVIDER_ENV_VARS.items():
                _key = os.environ.get(_env_var, "")
                if _key:
                    api_key = _key
                    if not provider:
                        provider = _provider
                    break

        return RealAI(model_name=model_name, api_key=api_key,
                      provider=provider, base_url=base_url)

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers',
                         'Content-Type, Authorization, X-Provider, X-Base-URL')
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/v1/models':
            # List available models: RealAI's own plus any configured providers.
            models = [
                {
                    "id": "realai-2.0",
                    "object": "model",
                    "created": 1708308000,
                    "owned_by": "realai",
                    "permission": [],
                    "root": "realai-2.0",
                    "parent": None,
                }
            ]
            for provider_name, cfg in PROVIDER_CONFIGS.items():
                models.append({
                    "id": cfg["default_model"],
                    "object": "model",
                    "created": 1708308000,
                    "owned_by": provider_name,
                    "permission": [],
                    "root": cfg["default_model"],
                    "parent": None,
                })
            self._send_response(200, {"object": "list", "data": models})

        elif parsed_path.path.startswith('/v1/models/'):
            model_id = parsed_path.path[len('/v1/models/'):]
            model = self._get_model(model_name=model_id)
            response = model.get_model_info()
            response["object"] = "model"
            response["id"] = model_id
            self._send_response(200, response)

        elif parsed_path.path == '/health':
            self._send_response(200, {"status": "healthy", "model": "realai-2.0"})

        else:
            self._send_response(404, {"error": "Not found"})

    def do_POST(self):
        """Handle POST requests."""
        parsed_path = urlparse(self.path)

        try:
            body = self._read_body()
            # The 'model' field in the body is passed through to RealAI as the
            # preferred provider model name.  Provider routing still depends on
            # the API key and X-Provider header; if no key is supplied the
            # response falls back to RealAI's placeholder regardless of model.
            model_name = body.get('model', 'realai-2.0')
            model = self._get_model(model_name=model_name)

            if parsed_path.path == '/v1/chat/completions':
                response = model.chat_completion(
                    messages=body.get('messages', []),
                    temperature=body.get('temperature', 0.7),
                    max_tokens=body.get('max_tokens'),
                    stream=body.get('stream', False)
                )
                self._send_response(200, response)

            elif parsed_path.path == '/v1/completions':
                response = model.text_completion(
                    prompt=body.get('prompt', ''),
                    temperature=body.get('temperature', 0.7),
                    max_tokens=body.get('max_tokens')
                )
                self._send_response(200, response)

            elif parsed_path.path == '/v1/images/generations':
                response = model.generate_image(
                    prompt=body.get('prompt', ''),
                    size=body.get('size', '1024x1024'),
                    quality=body.get('quality', 'standard'),
                    n=body.get('n', 1)
                )
                self._send_response(200, response)

            elif parsed_path.path == '/v1/videos/generations':
                response = model.generate_video(
                    prompt=body.get('prompt', ''),
                    image_url=body.get('image_url'),
                    size=body.get('size', '1280x720'),
                    duration=body.get('duration', 5),
                    fps=body.get('fps', 24),
                    n=body.get('n', 1),
                    response_format=body.get('response_format', 'url'),
                    model=body.get('model')
                )
                self._send_response(200, response)

            elif parsed_path.path == '/v1/embeddings':
                response = model.create_embeddings(
                    input_text=body.get('input', ''),
                    model=body.get('model', 'realai-embeddings')
                )
                self._send_response(200, response)

            elif parsed_path.path == '/v1/audio/transcriptions':
                response = model.transcribe_audio(
                    audio_file=body.get('file', ''),
                    language=body.get('language'),
                    prompt=body.get('prompt')
                )
                self._send_response(200, response)

            elif parsed_path.path == '/v1/audio/speech':
                response = model.generate_audio(
                    text=body.get('input', ''),
                    voice=body.get('voice', 'alloy'),
                    model=body.get('model', 'realai-tts')
                )
                self._send_response(200, response)

            elif parsed_path.path == '/v1/reasoning/chain':
                response = model.chain_of_thought(
                    problem=body.get('problem', ''),
                    domain=body.get('domain')
                )
                self._send_response(200, response)

            elif parsed_path.path == '/v1/synthesis/knowledge':
                response = model.synthesize_knowledge(
                    topics=body.get('topics', []),
                    output_format=body.get('output_format', 'narrative')
                )
                self._send_response(200, response)

            elif parsed_path.path == '/v1/reflection/analyze':
                response = model.self_reflect(
                    interaction_history=body.get('interaction_history'),
                    focus=body.get('focus', 'general')
                )
                self._send_response(200, response)

            elif parsed_path.path == '/v1/agents/orchestrate':
                response = model.orchestrate_agents(
                    task=body.get('task', ''),
                    agent_roles=body.get('agent_roles')
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
    print("  GET  /v1/models/<model-id>")
    print("  POST /v1/chat/completions")
    print("  POST /v1/completions")
    print("  POST /v1/images/generations")
    print("  POST /v1/embeddings")
    print("  POST /v1/audio/transcriptions")
    print("  POST /v1/audio/speech")
    print("  POST /v1/reasoning/chain")
    print("  POST /v1/synthesis/knowledge")
    print("  POST /v1/reflection/analyze")
    print("  POST /v1/agents/orchestrate")
    print("\nPass your API key via:  Authorization: Bearer <key>")
    print("Override provider via:  X-Provider: openai|anthropic|grok|gemini|openrouter|mistral|together|deepseek|perplexity")
    print("Override base URL via:  X-Base-URL: https://...")
    print("\nPress Ctrl+C to stop the server")
    print("="*60)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        httpd.shutdown()


if __name__ == "__main__":
    run_server()
