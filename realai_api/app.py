import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .auth import require_api_key
from .config import DEFAULT_MODEL_ID, DEFAULT_MODEL_OWNER
from .providers import MockProvider, ProviderRouter
from .schemas import (
    AudioSpeechRequest,
    AudioTranscriptionRequest,
    CapabilityInfo,
    ChatCompletionRequest,
    CodeExecutionRequest,
    CodeGenerationRequest,
    EmbeddingsRequest,
    ImageGenerationRequest,
    TranslationRequest,
    WebResearchRequest,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="RealAI API",
    version="0.1.0",
    description="OpenAI-compatible RealAI API with 17+ capabilities and a React frontend.",
)

# ---------------------------------------------------------------------------
# CORS – allow the React dev server (and any other origin) to call the API
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

provider_router = ProviderRouter(
    default_provider=MockProvider(model_id=DEFAULT_MODEL_ID, owned_by=DEFAULT_MODEL_OWNER)
)

# ---------------------------------------------------------------------------
# Lazy-load the RealAI core so the API server still starts even when optional
# dependencies (httpx, etc.) are absent.
# ---------------------------------------------------------------------------
_realai_instance = None


def _get_realai():
    """Return a cached RealAI instance, creating it on first call."""
    global _realai_instance
    if _realai_instance is None:
        try:
            import sys
            # Allow both package-mode and direct-run imports
            parent = str(Path(__file__).resolve().parent.parent)
            if parent not in sys.path:
                sys.path.insert(0, parent)
            from realai import RealAI  # type: ignore
            _realai_instance = RealAI()
        except Exception as exc:
            logger.warning("RealAI core unavailable (%s); stub responses will be used.", exc)
    return _realai_instance


# ---------------------------------------------------------------------------
# Serve the compiled React frontend (app/dist/) when it exists
# ---------------------------------------------------------------------------
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "app" / "dist"

if _FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_FRONTEND_DIST / "assets")), name="assets")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _error_ref() -> str:
    """Return an opaque reference ID for log correlation."""
    return f"err-{int(time.time())}-{uuid.uuid4().hex[:8]}"


def _safe_realai_call(method_name: str, **kwargs) -> Dict[str, Any]:
    """Call a RealAI method by name, returning a stub dict on failure."""
    ai = _get_realai()
    if ai is not None:
        try:
            method = getattr(ai, method_name)
            return method(**kwargs)
        except Exception as exc:
            ref = _error_ref()
            logger.error("RealAI.%s failed [%s]: %s", method_name, ref, exc)
            return {"status": "error", "error": f"Request failed (ref: {ref})"}
    return {
        "status": "success",
        "data": f"Stub response for {method_name} — configure a provider API key for real output.",
        "note": "Set REALAI_OPENAI_API_KEY (or another provider key) to enable live responses.",
    }


# ---------------------------------------------------------------------------
# Health & metadata
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Return a simple liveness response."""
    return {"status": "ok", "version": "0.1.0"}


@app.get("/v1/models")
def list_models(api_key: str = Depends(require_api_key)):
    """List available models (OpenAI-compatible)."""
    models = provider_router.list_models()
    return {"object": "list", "data": models}


@app.get("/v1/capabilities")
def list_capabilities():
    """List all RealAI capabilities with descriptions and endpoint paths."""
    capabilities: List[CapabilityInfo] = [
        CapabilityInfo(name="Chat Completions", description="Multi-turn conversational AI and text generation.", endpoint="/v1/chat/completions"),
        CapabilityInfo(name="Image Generation", description="Generate images from text prompts.", endpoint="/v1/images/generations"),
        CapabilityInfo(name="Embeddings", description="Convert text into semantic embedding vectors.", endpoint="/v1/embeddings"),
        CapabilityInfo(name="Audio Transcription", description="Transcribe audio files to text (speech-to-text).", endpoint="/v1/audio/transcriptions"),
        CapabilityInfo(name="Text-to-Speech", description="Synthesise natural-sounding speech from text.", endpoint="/v1/audio/speech"),
        CapabilityInfo(name="Code Generation", description="Generate code in any programming language from a prompt.", endpoint="/v1/code/generate"),
        CapabilityInfo(name="Code Execution", description="Execute code safely in a sandboxed environment.", endpoint="/v1/execute"),
        CapabilityInfo(name="Translation", description="Translate text between languages.", endpoint="/v1/translate"),
        CapabilityInfo(name="Web Research", description="Search the web and summarise results.", endpoint="/v1/research"),
        CapabilityInfo(name="Image Analysis", description="Analyse and describe images using vision models.", endpoint=None),
        CapabilityInfo(name="Task Automation", description="Automate real-world tasks like bookings and orders.", endpoint=None),
        CapabilityInfo(name="Voice Interaction", description="Natural voice-based conversation interface.", endpoint=None),
        CapabilityInfo(name="Business Planning", description="Generate business plans, strategies and forecasts.", endpoint=None),
        CapabilityInfo(name="Therapy Support", description="Empathetic conversational support and counselling.", endpoint=None),
        CapabilityInfo(name="Web3 Integration", description="Blockchain queries, NFT minting and DeFi interactions.", endpoint=None),
        CapabilityInfo(name="Plugin System", description="Extend RealAI with third-party plugins.", endpoint=None),
        CapabilityInfo(name="Local Models", description="Run open-source models locally with full privacy.", endpoint=None),
    ]
    return {"object": "list", "data": [c.dict() for c in capabilities]}


# ---------------------------------------------------------------------------
# Chat completions (existing, kept for compatibility)
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest, api_key: str = Depends(require_api_key)):
    """OpenAI-compatible chat completions endpoint."""
    response = provider_router.route_chat(request)
    return response


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

@app.post("/v1/images/generations")
def generate_image(request: ImageGenerationRequest, api_key: str = Depends(require_api_key)):
    """Generate an image from a text prompt."""
    result = _safe_realai_call("generate_image", prompt=request.prompt, n=request.n, size=request.size)
    return result


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

@app.post("/v1/embeddings")
def create_embeddings(request: EmbeddingsRequest, api_key: str = Depends(require_api_key)):
    """Generate embedding vectors for one or more text inputs."""
    texts = request.input if isinstance(request.input, list) else [request.input]
    result = _safe_realai_call("generate_embeddings", texts=texts)
    return result


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

@app.post("/v1/audio/transcriptions")
def transcribe_audio(request: AudioTranscriptionRequest, api_key: str = Depends(require_api_key)):
    """Transcribe an audio file to text."""
    result = _safe_realai_call("transcribe_audio", audio_path=request.audio_path)
    return result


@app.post("/v1/audio/speech")
def text_to_speech(request: AudioSpeechRequest, api_key: str = Depends(require_api_key)):
    """Synthesise speech audio from text."""
    result = _safe_realai_call("generate_speech", text=request.input, voice=request.voice)
    return result


# ---------------------------------------------------------------------------
# Code
# ---------------------------------------------------------------------------

@app.post("/v1/code/generate")
def generate_code(request: CodeGenerationRequest, api_key: str = Depends(require_api_key)):
    """Generate code in a specified programming language."""
    result = _safe_realai_call("generate_code", prompt=request.prompt, language=request.language)
    return result


@app.post("/v1/execute")
def execute_code(request: CodeExecutionRequest, api_key: str = Depends(require_api_key)):
    """Execute code in a sandboxed environment."""
    result = _safe_realai_call("execute_code", code=request.code, language=request.language)
    return result


# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------

@app.post("/v1/translate")
def translate(request: TranslationRequest, api_key: str = Depends(require_api_key)):
    """Translate text into the target language."""
    result = _safe_realai_call("translate", text=request.text, target_language=request.target_language)
    return result


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------

@app.post("/v1/research")
def web_research(request: WebResearchRequest, api_key: str = Depends(require_api_key)):
    """Search the web and return a summarised result."""
    result = _safe_realai_call("web_research", query=request.query)
    return result


# ---------------------------------------------------------------------------
# SPA catch-all — serve the React frontend for every unmatched path
# ---------------------------------------------------------------------------

@app.get("/{full_path:path}")
def serve_spa(full_path: str):
    """Serve the compiled React SPA for client-side routing."""
    if not _FRONTEND_DIST.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend not built. Run `npm run build` inside the app/ directory.",
        )

    dist = _FRONTEND_DIST.resolve()

    # Safely join the path — let Path.resolve() canonicalise symlinks, then
    # verify the result sits inside dist.  This handles all traversal forms
    # (../foo, ./foo, encoded %2e%2e, etc.) without manual part filtering.
    candidate = (dist / full_path).resolve()

    try:
        candidate.relative_to(dist)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid path")

    if candidate.is_file():
        return FileResponse(str(candidate))

    # Fall back to index.html for client-side routing
    return FileResponse(str(dist / "index.html"))
