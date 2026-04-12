"""
Vercel entry point for RealAI.

Vercel's Python web-framework runtime detects an ``app`` callable in
``api/index.py`` and serves it as an ASGI/WSGI serverless function.

We expose the FastAPI application from :mod:`realai_api.app` as ``app`` so
that all /v1/* endpoints work out of the box on Vercel.  A lightweight
fallback WSGI stub is used if FastAPI or its dependencies are not installed.
"""

import json
import os
import sys

# Ensure the project root is on sys.path so realai_api is importable.
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

# ---------------------------------------------------------------------------
# Fallback: lightweight WSGI stub — defined first so ``app`` is always a
# top-level name that Vercel's static scanner can find unconditionally.
# ---------------------------------------------------------------------------


def app(environ, start_response):  # type: ignore[misc]
    """Minimal WSGI fallback when realai_api dependencies are missing."""
    path = environ.get("PATH_INFO", "/")

    if path == "/health":
        body = json.dumps({"status": "ok", "version": "0.1.0"}).encode()
    else:
        body = json.dumps({
            "message": "RealAI API is live.",
            "docs": "/docs",
            "health": "/health",
            "models": "/v1/models",
        }).encode()

    headers = [
        ("Content-Type", "application/json"),
        ("Content-Length", str(len(body))),
    ]
    start_response("200 OK", headers)
    return [body]


# ---------------------------------------------------------------------------
# Primary: replace the stub with the real FastAPI ASGI app when available.
# ---------------------------------------------------------------------------

try:
    from realai_api.app import app  # noqa: F401  (re-exported for Vercel)
except Exception:
    pass  # fallback ``app`` defined above remains in effect
