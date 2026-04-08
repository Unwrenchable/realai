"""
Vercel-compatible Python entrypoint.

This file exists so platforms like Vercel can detect a standard Python
entrypoint (`main.py`) and import a callable named `app`.
"""

from api_server import run_server


def app(environ, start_response):
    """Minimal WSGI app for serverless platform health checks."""
    path = environ.get("PATH_INFO", "/")
    method = environ.get("REQUEST_METHOD", "GET")

    if method == "GET" and path == "/health":
        body = b'{"status":"healthy","model":"realai-2.0"}'
        headers = [("Content-Type", "application/json"), ("Content-Length", str(len(body)))]
        start_response("200 OK", headers)
        return [body]

    body = b'{"message":"RealAI deployment is live. Use /health for status."}'
    headers = [("Content-Type", "application/json"), ("Content-Length", str(len(body)))]
    start_response("200 OK", headers)
    return [body]


if __name__ == "__main__":
    run_server()
