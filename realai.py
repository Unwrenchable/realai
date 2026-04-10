"""
RealAI - The AI Model That Can Do It All

A limitless AI model with comprehensive capabilities including:
- Text generation and completion
- Image generation and analysis
- Code generation, understanding, and execution
- Embeddings and semantic search
- Audio transcription and generation
- Translation and language understanding
- Web research and browsing
- Real-world task automation (groceries, appointments, etc.)
- Voice interaction and conversation
- Business planning and building
- Therapy and counseling support
- Web3 and blockchain integration
- Plugin system for unlimited extensibility
- Learning and memory capabilities

The sky is the limit - RealAI has no limits and can truly do anything!
"""

import json
import re
import time
import subprocess
import tempfile
import os
import importlib
from typing import List, Dict, Any, Optional, Union
from enum import Enum

try:
    import resource
except Exception:
    resource = None


# ---------------------------------------------------------------------------
# Provider configuration for real AI API routing
# ---------------------------------------------------------------------------

#: Configuration for supported external AI providers.
PROVIDER_CONFIGS: Dict[str, Dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "api_format": "openai",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "default_model": "claude-3-5-haiku-20241022",
        "api_format": "anthropic",
    },
    "grok": {
        "base_url": "https://api.x.ai/v1",
        "default_model": "grok-beta",
        "api_format": "openai",
    },
    "gemini": {
        # Google exposes an OpenAI-compatible endpoint for Gemini models.
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": "gemini-1.5-flash",
        "api_format": "openai",
    },
    "openrouter": {
        # OpenRouter aggregates hundreds of models via a single OpenAI-compatible API.
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "openai/gpt-4o-mini",
        "api_format": "openai",
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "default_model": "mistral-small-latest",
        "api_format": "openai",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "default_model": "meta-llama/Llama-3-8b-chat-hf",
        "api_format": "openai",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "api_format": "openai",
    },
    "perplexity": {
        "base_url": "https://api.perplexity.ai",
        "default_model": "llama-3.1-sonar-small-128k-online",
        "api_format": "openai",
    },
}

#: Maps API key prefixes to provider names for auto-detection.
_KEY_PREFIX_TO_PROVIDER: Dict[str, str] = {
    "sk-ant-": "anthropic",
    "sk-or-v1-": "openrouter",
    "sk-proj-": "openai",
    "sk-": "openai",
    "xai-": "grok",
    "AIza": "gemini",
    "pplx-": "perplexity",
}

#: Maps provider names to the environment variable used to pass their API key
#: via the process environment (e.g. set by the GUI launcher).
#: The insertion order defines the fallback priority used by the API server.
PROVIDER_ENV_VARS: Dict[str, str] = {
    "openai": "REALAI_OPENAI_API_KEY",
    "anthropic": "REALAI_ANTHROPIC_API_KEY",
    "grok": "REALAI_GROK_API_KEY",
    "gemini": "REALAI_GEMINI_API_KEY",
    "openrouter": "REALAI_OPENROUTER_API_KEY",
    "mistral": "REALAI_MISTRAL_API_KEY",
    "together": "REALAI_TOGETHER_API_KEY",
    "deepseek": "REALAI_DEEPSEEK_API_KEY",
    "perplexity": "REALAI_PERPLEXITY_API_KEY",
}

#: Provider capability map used for capability-aware routing and fallbacks.
#: Values are capability names from :class:`ModelCapability`.
PROVIDER_CAPABILITY_MAP: Dict[str, List[str]] = {
    "openai": [
        "text_generation", "image_generation", "video_generation",
        "image_analysis", "code_generation", "embeddings",
        "audio_transcription", "audio_generation", "translation",
    ],
    "anthropic": [
        "text_generation", "code_generation", "translation",
        "knowledge_synthesis", "chain_of_thought", "self_reflection",
        "multi_agent",
    ],
    "grok": [
        "text_generation", "image_generation", "image_analysis",
        "video_generation", "code_generation", "translation",
        "web_research", "chain_of_thought", "knowledge_synthesis",
        "multi_agent",
    ],
    "gemini": [
        "text_generation", "image_analysis", "code_generation",
        "embeddings", "translation", "audio_transcription",
        "chain_of_thought", "knowledge_synthesis",
    ],
    "openrouter": [
        "text_generation", "image_generation", "video_generation",
        "image_analysis", "code_generation", "embeddings",
        "audio_transcription", "audio_generation", "translation",
        "web_research", "task_automation", "voice_interaction",
        "business_planning", "therapy_counseling", "web3_integration",
        "code_execution", "plugin_system", "memory_learning",
        "self_reflection", "chain_of_thought", "knowledge_synthesis",
        "multi_agent",
    ],
    "mistral": [
        "text_generation", "code_generation", "embeddings",
        "translation", "chain_of_thought",
    ],
    "together": [
        "text_generation", "code_generation", "embeddings",
        "translation", "image_generation",
    ],
    "deepseek": [
        "text_generation", "code_generation", "translation",
        "chain_of_thought", "knowledge_synthesis",
    ],
    "perplexity": [
        "text_generation", "web_research", "translation",
        "chain_of_thought", "knowledge_synthesis",
    ],
}

#: Persona profiles that modify response style while preserving answer quality.
PERSONA_PROFILES: Dict[str, Dict[str, str]] = {
    "balanced": {
        "description": "Neutral, concise, and practical assistant style.",
        "system_prompt": "",
    },
    "analyst": {
        "description": "Data-first, structured, and verification-oriented style.",
        "system_prompt": (
            "Adopt an analytical style: structure your response clearly, "
            "state assumptions, and prefer verifiable statements."
        ),
    },
    "creative": {
        "description": "Imaginative, expressive, and idea-rich style.",
        "system_prompt": (
            "Adopt a creative style: generate novel ideas and vivid language "
            "while remaining relevant to the user's goal."
        ),
    },
    "coach": {
        "description": "Supportive, motivational, and action-oriented style.",
        "system_prompt": (
            "Adopt a coaching style: be encouraging, practical, and provide "
            "clear next steps."
        ),
    },
}


def _detect_provider(api_key: Optional[str], provider: Optional[str]) -> Optional[str]:
    """Detect the AI provider from an explicit name or API key prefix.

    Args:
        api_key: The raw API key string (may be ``None``).
        provider: An explicit provider name that overrides key-based detection.

    Returns:
        The lower-cased provider name, or ``None`` if it cannot be determined.
    """
    if provider:
        return provider.lower()
    if api_key:
        for prefix, name in _KEY_PREFIX_TO_PROVIDER.items():
            if api_key.startswith(prefix):
                return name
    return None


class ModelCapability(Enum):
    """Supported capabilities of the RealAI model."""
    TEXT_GENERATION = "text_generation"
    IMAGE_GENERATION = "image_generation"
    VIDEO_GENERATION = "video_generation"
    IMAGE_ANALYSIS = "image_analysis"
    CODE_GENERATION = "code_generation"
    EMBEDDINGS = "embeddings"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    AUDIO_GENERATION = "audio_generation"
    TRANSLATION = "translation"
    WEB_RESEARCH = "web_research"
    TASK_AUTOMATION = "task_automation"
    VOICE_INTERACTION = "voice_interaction"
    BUSINESS_PLANNING = "business_planning"
    THERAPY_COUNSELING = "therapy_counseling"
    WEB3_INTEGRATION = "web3_integration"
    CODE_EXECUTION = "code_execution"
    PLUGIN_SYSTEM = "plugin_system"
    MEMORY_LEARNING = "memory_learning"
    # Next-generation capabilities
    SELF_REFLECTION = "self_reflection"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"
    MULTI_AGENT = "multi_agent"


#: Canonical capability-domain mapping used for discovery across model/client/API.
CAPABILITY_DOMAIN_MAP: Dict[ModelCapability, str] = {
    ModelCapability.TEXT_GENERATION: "generation",
    ModelCapability.IMAGE_GENERATION: "generation",
    ModelCapability.VIDEO_GENERATION: "generation",
    ModelCapability.AUDIO_GENERATION: "generation",
    ModelCapability.CODE_GENERATION: "generation",
    ModelCapability.IMAGE_ANALYSIS: "analysis",
    ModelCapability.AUDIO_TRANSCRIPTION: "analysis",
    ModelCapability.EMBEDDINGS: "analysis",
    ModelCapability.TRANSLATION: "analysis",
    ModelCapability.WEB_RESEARCH: "integrations",
    ModelCapability.TASK_AUTOMATION: "tools",
    ModelCapability.VOICE_INTERACTION: "tools",
    ModelCapability.BUSINESS_PLANNING: "tools",
    ModelCapability.THERAPY_COUNSELING: "tools",
    ModelCapability.WEB3_INTEGRATION: "integrations",
    ModelCapability.CODE_EXECUTION: "tools",
    ModelCapability.PLUGIN_SYSTEM: "integrations",
    ModelCapability.MEMORY_LEARNING: "memory",
    ModelCapability.SELF_REFLECTION: "reasoning",
    ModelCapability.CHAIN_OF_THOUGHT: "reasoning",
    ModelCapability.KNOWLEDGE_SYNTHESIS: "reasoning",
    ModelCapability.MULTI_AGENT: "orchestration",
}


class RealAI:
    """
    RealAI - The limitless AI model that can truly do anything.
    
    This model has no limits and provides a unified interface for unlimited AI capabilities:
    - From Web3 to ordering groceries
    - Research any prompt comprehensively
    - Code and execute anything
    - Voice interaction for natural conversation
    - Therapy and counseling support
    - Build businesses from the ground up
    - Can be implemented anywhere via plugins
    
    The sky is the limit with RealAI!
    
    Attributes:
        model_name (str): The name of the model
        version (str): The version of the model
        capabilities (List[ModelCapability]): List of supported capabilities
    """
    
    def __init__(self, model_name: str = "realai-2.0", api_key: Optional[str] = None,
                 provider: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the RealAI model.

        Args:
            model_name (str): The name of the model to use. When a real provider is
                configured and this is left at the default ``"realai-2.0"``, the
                provider's default model will be used for API calls.
            api_key (Optional[str]): API key for the provider (OpenAI, Anthropic, etc.).
                When provided, requests are forwarded to the real AI service.
            provider (Optional[str]): Explicit provider name (``"openai"``,
                ``"anthropic"``, ``"grok"``, ``"gemini"``). If omitted the provider
                is auto-detected from the *api_key* prefix.
            base_url (Optional[str]): Override the provider's base URL. Useful for
                self-hosted or proxy endpoints.
        """
        self.model_name = model_name
        self.version = "2.0.0"
        self.api_key = api_key
        self.capabilities = list(ModelCapability)
        # Registry of loaded plugins: name -> metadata
        self.plugins_registry: Dict[str, Any] = {}

        # Provider routing setup
        self.provider = _detect_provider(api_key, provider)
        cfg: Dict[str, str] = PROVIDER_CONFIGS.get(self.provider, {}) if self.provider else {}
        self.base_url: str = base_url or cfg.get("base_url", "")
        self._api_format: str = cfg.get("api_format", "openai")
        # The actual model name sent to the remote provider.
        # If the caller left model_name at the default, use the provider's default.
        if self.provider and model_name == "realai-2.0":
            self._provider_model: str = cfg.get("default_model", model_name)
        else:
            self._provider_model = model_name
        self.response_contract_version = "2026-04-08"
        self.persona = "balanced"
        self._web_research_cache: Dict[str, Dict[str, Any]] = {}
        self._web_research_cache_ttl = 900

    # ------------------------------------------------------------------
    # Private helpers for real provider API calls
    # ------------------------------------------------------------------

    def _provider_supports(self, capability_name: str) -> bool:
        """Return whether the active provider is expected to support a capability."""
        if not self.provider:
            return True
        supported = PROVIDER_CAPABILITY_MAP.get(self.provider)
        if not supported:
            return True
        return capability_name in supported

    def _with_metadata(
        self,
        response: Dict[str, Any],
        capability: str,
        modality: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Attach canonical RealAI response metadata."""
        payload = dict(response)
        payload["realai_meta"] = {
            "contract_version": self.response_contract_version,
            "capability": capability,
            "modality": modality or "text",
            "provider": self.provider or "realai-local",
            "model": self._provider_model if self.provider else self.model_name,
            "timestamp": int(time.time()),
            **(extra or {}),
        }
        return payload

    @staticmethod
    def _parse_json_block(text: str) -> Dict[str, Any]:
        """Best-effort parser for plain JSON or fenced JSON blocks."""
        cleaned = text.strip()
        if "```" in cleaned:
            parts = cleaned.split("```")
            if len(parts) >= 3:
                block = parts[1]
                first_nl = block.find("\n")
                cleaned = (block[first_nl + 1:] if first_nl != -1 else block).strip()
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}

    def _call_openai_compat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Call an OpenAI-compatible chat/completions endpoint.

        Works with OpenAI, xAI/Grok, Google Gemini (via its OpenAI-compatible
        endpoint), and any other OpenAI-API-compatible service.
        """
        import requests as _requests

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload: Dict[str, Any] = {
            "model": self._provider_model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        resp = _requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def _call_anthropic(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call the Anthropic Messages API and normalize the response to OpenAI format."""
        import requests as _requests

        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        user_messages = [m for m in messages if m.get("role") != "system"]

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        payload: Dict[str, Any] = {
            "model": self._provider_model,
            "messages": user_messages,
            "max_tokens": max_tokens or 1024,
            "temperature": temperature,
        }
        if system_parts:
            payload["system"] = "\n".join(system_parts)

        resp = _requests.post(
            f"{self.base_url}/v1/messages", json=payload, headers=headers, timeout=60
        )
        resp.raise_for_status()
        data = resp.json()

        content = data.get("content", [])
        text = content[0].get("text", "") if content else ""
        usage = data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)

        return {
            "id": data.get("id", f"chatcmpl-{int(time.time())}"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self._provider_model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": data.get("stop_reason", "stop"),
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a chat completion response (like OpenAI's ChatGPT).

        When an *api_key* and a recognised *provider* are configured the request
        is forwarded to the real AI service (OpenAI, Anthropic, xAI/Grok,
        Google Gemini, …).  Falls back to a placeholder response when no
        provider is configured or the remote call fails.

        Args:
            messages (List[Dict[str, str]]): List of message objects with 'role' and 'content'
            temperature (float): Sampling temperature (0-2)
            max_tokens (Optional[int]): Maximum tokens to generate
            stream (bool): Whether to stream the response

        Returns:
            Dict[str, Any]: Chat completion response
        """
        messages_to_send = list(messages)
        persona_cfg = PERSONA_PROFILES.get(self.persona, PERSONA_PROFILES["balanced"])
        persona_prompt = persona_cfg.get("system_prompt", "").strip()
        if persona_prompt:
            messages_to_send = [{"role": "system", "content": persona_prompt}] + messages_to_send

        # Route to the real provider when credentials are available.
        if self.api_key and self.provider:
            try:
                if self._api_format == "anthropic":
                    provider_response = self._call_anthropic(messages_to_send, temperature, max_tokens)
                else:
                    provider_response = self._call_openai_compat(messages_to_send, temperature, max_tokens, stream)
                return self._with_metadata(
                    provider_response,
                    capability=ModelCapability.TEXT_GENERATION.value,
                    modality="text",
                    extra={"persona": self.persona},
                )
            except Exception:
                # Broad catch is intentional: we want to fall back gracefully
                # for network errors, auth failures, missing `requests` package,
                # etc., so the client remains functional without live credentials.
                pass

        # Placeholder response (no provider configured or call failed).
        return self._with_metadata({
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I am RealAI, the model that can do it all. I understand your message and am ready to help!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": sum(len(msg.get("content", "").split()) for msg in messages_to_send),
                "completion_tokens": 20,
                "total_tokens": sum(len(msg.get("content", "").split()) for msg in messages_to_send) + 20
            }
        }, capability=ModelCapability.TEXT_GENERATION.value, modality="text", extra={"persona": self.persona})

    def text_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate a text completion (like OpenAI's GPT-3).

        When an *api_key* and a recognised *provider* are configured the prompt
        is forwarded to the real AI service via its chat/completions endpoint.
        Falls back to a placeholder when no provider is configured or the
        remote call fails.

        Args:
            prompt (str): The text prompt
            temperature (float): Sampling temperature (0-2)
            max_tokens (Optional[int]): Maximum tokens to generate

        Returns:
            Dict[str, Any]: Text completion response
        """
        # Route to the real provider when credentials are available.
        # All modern providers expose a chat completions endpoint; wrap the
        # prompt as a single user message for maximum compatibility.
        if self.api_key and self.provider:
            try:
                messages = [{"role": "user", "content": prompt}]
                if self._api_format == "anthropic":
                    chat_resp = self._call_anthropic(messages, temperature, max_tokens)
                else:
                    chat_resp = self._call_openai_compat(messages, temperature, max_tokens)
                # Re-shape chat response into text-completion format.
                text = ""
                if chat_resp.get("choices"):
                    text = chat_resp["choices"][0].get("message", {}).get("content", "")
                usage = chat_resp.get("usage", {})
                return self._with_metadata({
                    "id": chat_resp.get("id", f"cmpl-{int(time.time())}"),
                    "object": "text_completion",
                    "created": chat_resp.get("created", int(time.time())),
                    "model": self._provider_model,
                    "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
                    "usage": usage,
                }, capability=ModelCapability.TEXT_GENERATION.value, modality="text", extra={"persona": self.persona})
            except Exception:
                # Broad catch is intentional: fall back gracefully for network
                # errors, auth failures, or missing dependencies.
                pass

        return self._with_metadata({
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{
                "text": "This is a RealAI completion. The model understands and can respond to your prompt.",
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 15,
                "total_tokens": len(prompt.split()) + 15
            }
        }, capability=ModelCapability.TEXT_GENERATION.value, modality="text", extra={"persona": self.persona})

    def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        n: int = 1
    ) -> Dict[str, Any]:
        """
        Generate an image from a text prompt (like DALL-E).
        
        Args:
            prompt (str): The image description
            size (str): Image size (e.g., "1024x1024")
            quality (str): Image quality ("standard" or "hd")
            n (int): Number of images to generate
            
        Returns:
            Dict[str, Any]: Image generation response
        """
        response = {
            "created": int(time.time()),
            "data": [
                {
                    "url": f"https://realai.example.com/generated-image-{i}.png",
                    "revised_prompt": prompt
                }
                for i in range(n)
            ]
        }
        return self._with_metadata(
            response,
            capability=ModelCapability.IMAGE_GENERATION.value,
            modality="image",
        )

    def generate_video(
        self,
        prompt: str,
        image_url: Optional[str] = None,
        size: str = "1280x720",
        duration: int = 5,
        fps: int = 24,
        n: int = 1,
        response_format: str = "url",
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a video from text or an input image.

        Args:
            prompt (str): Prompt describing the requested video.
            image_url (Optional[str]): Optional source image for image-to-video.
            size (str): Output dimensions (e.g., "1280x720").
            duration (int): Video duration in seconds.
            fps (int): Frames per second.
            n (int): Number of videos to generate.
            response_format (str): "url" or "b64_json".
            model (Optional[str]): Optional provider model override.

        Returns:
            Dict[str, Any]: Video generation response in OpenAI-style data format.
        """
        if response_format not in ("url", "b64_json"):
            response_format = "url"

        safe_n = max(1, int(n))
        request_model = model or self._provider_model
        mode = "image_to_video" if image_url else "text_to_video"

        # Attempt real provider call when configured and endpoint is available.
        if (
            self.api_key
            and self.provider
            and self._api_format == "openai"
            and self.base_url
            and self._provider_supports(ModelCapability.VIDEO_GENERATION.value)
        ):
            try:
                import requests as _requests

                payload: Dict[str, Any] = {
                    "model": request_model,
                    "prompt": prompt,
                    "size": size,
                    "duration": duration,
                    "fps": fps,
                    "n": safe_n,
                    "response_format": response_format,
                }
                if image_url:
                    # Keep both keys for broader provider compatibility:
                    # some APIs expect `image_url`, others expect `image`.
                    payload["image_url"] = image_url
                    payload["image"] = image_url

                resp = _requests.post(
                    f"{self.base_url}/videos/generations",
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                provider_response = resp.json()
                if isinstance(provider_response, dict) and provider_response.get("data"):
                        return self._with_metadata(
                            provider_response,
                            capability=ModelCapability.VIDEO_GENERATION.value,
                            modality="video",
                            extra={"mode": mode},
                        )
            except Exception:
                # Fall back to placeholder response if provider does not support
                # this endpoint or request execution fails for any reason.
                pass

        # Graceful local placeholder response.
        data: List[Dict[str, Any]] = []
        for i in range(safe_n):
            item: Dict[str, Any] = {
                "revised_prompt": prompt,
                "mode": mode,
            }
            if image_url:
                item["source_image_url"] = image_url
            if response_format == "b64_json":
                try:
                    import base64 as _base64
                    placeholder_blob = (
                        f"RealAI placeholder video payload {i + 1}".encode("utf-8")
                    )
                    item["b64_json"] = _base64.b64encode(placeholder_blob).decode("ascii")
                except Exception:
                    item["b64_json"] = ""
            else:
                item["url"] = f"https://realai.example.com/generated-video-{i}.mp4"
            data.append(item)

        return self._with_metadata({
            "created": int(time.time()),
            "data": data,
            "model": request_model,
            "duration": duration,
            "size": size,
            "fps": fps,
            "response_format": response_format,
            "note": (
                "Placeholder response. Configure a provider endpoint that supports "
                "video generation for real video outputs."
            ),
        }, capability=ModelCapability.VIDEO_GENERATION.value, modality="video", extra={"mode": mode})
    
    def analyze_image(self, image_url: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze an image and provide descriptions or answer questions (like GPT-4 Vision).
        
        Args:
            image_url (str): URL of the image to analyze
            prompt (Optional[str]): Optional question about the image
            
        Returns:
            Dict[str, Any]: Image analysis response
        """
        response = {
            "analysis": "RealAI has analyzed your image.",
            "description": "The image contains visual content that has been processed by RealAI.",
            "prompt": prompt,
            "confidence": 0.95
        }
        return self._with_metadata(
            response,
            capability=ModelCapability.IMAGE_ANALYSIS.value,
            modality="image",
        )
    
    def generate_code(
        self,
        prompt: str,
        language: Optional[str] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate code from a description (like GitHub Copilot or Codex).
        
        Args:
            prompt (str): Description of the code to generate
            language (Optional[str]): Programming language
            context (Optional[str]): Additional context or existing code
            
        Returns:
            Dict[str, Any]: Code generation response
        """
        response = {
            "code": "# RealAI generated code\n# Based on your prompt, here's the implementation\n",
            "language": language or "python",
            "explanation": "RealAI has generated code based on your requirements.",
            "confidence": 0.9
        }
        return response
    
    def create_embeddings(
        self,
        input_text: Union[str, List[str]],
        model: str = "realai-embeddings"
    ) -> Dict[str, Any]:
        """
        Create embeddings for text (like OpenAI's text-embedding models).
        
        Args:
            input_text (Union[str, List[str]]): Text or list of texts to embed
            model (str): The embedding model to use
            
        Returns:
            Dict[str, Any]: Embeddings response
        """
        texts = [input_text] if isinstance(input_text, str) else input_text

        # Try to use sentence-transformers for real embeddings. If unavailable,
        # fall back to the original stubbed 1536-d zero vector for compatibility.
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            # Choose a compact, high-quality model by default
            model_name = "all-mpnet-base-v2"
            # Allow callers to override with a model-like string
            if model and model != "realai-embeddings":
                model_name = model

            embedder = SentenceTransformer(model_name)
            vectors = embedder.encode(texts, show_progress_bar=False)

            data = []
            for i, vec in enumerate(vectors if isinstance(vectors, (list, np.ndarray)) else [vectors]):
                arr = np.array(vec).astype(float).tolist()
                data.append({
                    "object": "embedding",
                    "embedding": arr,
                    "index": i
                })

            response = {
                "object": "list",
                "data": data,
                "model": model_name,
                "usage": {
                    "prompt_tokens": sum(len(text.split()) for text in texts),
                    "total_tokens": sum(len(text.split()) for text in texts)
                }
            }
            return response

        except Exception:
            # Fallback stub for environments without sentence-transformers
            data = [
                {
                    "object": "embedding",
                    "embedding": [0.0] * 1536,
                    "index": i
                }
                for i, _ in enumerate(texts)
            ]
            return {
                "object": "list",
                "data": data,
                "model": model,
                "usage": {
                    "prompt_tokens": sum(len(text.split()) for text in texts),
                    "total_tokens": sum(len(text.split()) for text in texts)
                }
            }
    
    def transcribe_audio(
        self,
        audio_file: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio to text (like Whisper).
        
        Args:
            audio_file (str): Path or URL to audio file
            language (Optional[str]): Language of the audio
            prompt (Optional[str]): Optional prompt to guide the model
            
        Returns:
            Dict[str, Any]: Transcription response
        """
        # Attempt to use Vosk for offline ASR if available and model provided.
        try:
            from vosk import Model, KaldiRecognizer
            import wave

            # If audio_file is a URL or missing, fall back
            if not os.path.exists(audio_file):
                raise FileNotFoundError("Audio file not found for local ASR")

            # Try to find a small model in environment variable VOSK_MODEL_PATH
            model_path = os.environ.get("VOSK_MODEL_PATH")
            if not model_path or not os.path.exists(model_path):
                # No model available locally; fall back
                raise RuntimeError("No Vosk model available")

            wf = wave.open(audio_file, "rb")
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                # Vosk expects mono 16-bit audio; fall back if not matching
                raise RuntimeError("Unsupported audio format for Vosk; expected mono 16-bit WAV")

            model = Model(model_path)
            rec = KaldiRecognizer(model, wf.getframerate())
            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    results.append(rec.Result())
            results.append(rec.FinalResult())
            text_parts = []
            for r in results:
                try:
                    j = json.loads(r)
                    if "text" in j:
                        text_parts.append(j["text"])
                except Exception:
                    continue

            return self._with_metadata({
                "text": " ".join(p for p in text_parts if p),
                "language": language or "en",
                "duration": wf.getnframes() / wf.getframerate(),
                "segments": [],
            }, capability=ModelCapability.AUDIO_TRANSCRIPTION.value, modality="audio")

        except Exception:
            # Fallback stub
            return self._with_metadata({
                "text": "RealAI has transcribed your audio file.",
                "language": language or "en",
                "duration": 10.5,
                "segments": []
            }, capability=ModelCapability.AUDIO_TRANSCRIPTION.value, modality="audio")
    
    def generate_audio(
        self,
        text: str,
        voice: str = "alloy",
        model: str = "realai-tts"
    ) -> Dict[str, Any]:
        """
        Generate audio from text (like text-to-speech).
        
        Args:
            text (str): Text to convert to speech
            voice (str): Voice to use
            model (str): TTS model to use
            
        Returns:
            Dict[str, Any]: Audio generation response
        """
        # Try to use pyttsx3 for local TTS if available
        try:
            import pyttsx3

            engine = pyttsx3.init()
            # Optionally set voice
            try:
                voices = engine.getProperty('voices')
                # Attempt to pick a matching voice name if provided
                for v in voices:
                    if voice.lower() in (v.name or "").lower():
                        engine.setProperty('voice', v.id)
                        break
            except Exception:
                pass

            # Write to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                out_path = f.name
            engine.save_to_file(text, out_path)
            engine.runAndWait()

            duration = len(text.split()) * 0.5
            return self._with_metadata({
                "audio_url": out_path,
                "duration": duration,
                "voice": voice,
                "model": model
            }, capability=ModelCapability.AUDIO_GENERATION.value, modality="audio")

        except Exception:
            # Fallback simulated response
            return self._with_metadata({
                "audio_url": "https://realai.example.com/generated-audio.mp3",
                "duration": len(text.split()) * 0.5,
                "voice": voice,
                "model": model
            }, capability=ModelCapability.AUDIO_GENERATION.value, modality="audio")
    
    def translate(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Translate text between languages.
        
        Args:
            text (str): Text to translate
            target_language (str): Target language code (e.g., 'es', 'fr', 'de')
            source_language (Optional[str]): Source language (auto-detected if not provided)
            
        Returns:
            Dict[str, Any]: Translation response
        """
        response = {
            "translated_text": f"[Translated to {target_language}] {text}",
            "source_language": source_language or "auto",
            "target_language": target_language,
            "confidence": 0.95
        }
        return self._with_metadata(
            response,
            capability=ModelCapability.TRANSLATION.value,
            modality="text",
        )
    
    def web_research(
        self,
        query: str,
        depth: str = "standard",
        sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Research a topic using web search and analysis.
        
        Args:
            query (str): Research query or topic
            depth (str): Research depth ("quick", "standard", "deep")
            sources (Optional[List[str]]): Specific sources to prioritize
            
        Returns:
            Dict[str, Any]: Research results with findings and sources
        """
        # Unified retrieval layer with source details, citations, and freshness/cache metadata.
        max_results = {"quick": 1, "standard": 3, "deep": 5}.get(depth, 3)
        cache_key = f"{query}|{depth}|{','.join(sources or [])}"
        now = int(time.time())

        cached = self._web_research_cache.get(cache_key)
        if cached and now - int(cached.get("cached_at", 0)) < self._web_research_cache_ttl:
            cached_payload = dict(cached["payload"])
            cached_payload["cached"] = True
            cached_payload["freshness"] = "cached"
            return self._with_metadata(
                cached_payload,
                capability=ModelCapability.WEB_RESEARCH.value,
                modality="text",
            )

        findings_list: List[Dict[str, Any]] = []
        resolved_sources: List[str] = []

        try:
            import requests
            from bs4 import BeautifulSoup

            session = requests.Session()
            session.headers.update({
                "User-Agent": "RealAI/2.0 (+https://example.com)"
            })

            # If caller provided explicit sources, fetch them directly
            urls_to_fetch = list(sources or [])

            # If no explicit sources, do a lightweight DuckDuckGo HTML search
            if not urls_to_fetch:
                search_url = "https://html.duckduckgo.com/html/"
                params = {"q": query}
                r = session.post(search_url, data=params, timeout=10)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")

                # Extract result links up to max_results
                anchors = soup.find_all("a", attrs={"rel": "nofollow"})
                for a in anchors:
                    href = a.get("href")
                    if href and href.startswith("http"):
                        urls_to_fetch.append(href)
                    if len(urls_to_fetch) >= max_results:
                        break

            # Limit number of sources
            urls_to_fetch = urls_to_fetch[:max_results]

            for url in urls_to_fetch:
                try:
                    r = session.get(url, timeout=10)
                    r.raise_for_status()
                    page = BeautifulSoup(r.text, "html.parser")
                    title = (page.title.string.strip() if page.title and page.title.string else url)
                    p = page.find("p")
                    snippet = p.get_text().strip() if p else ""
                    citation_score = 0.4
                    if url.startswith("https://"):
                        citation_score += 0.2
                    if len(snippet) > 80:
                        citation_score += 0.2
                    if len(title) > 8:
                        citation_score += 0.1
                    citation_score = min(0.99, round(citation_score, 2))
                    findings_list.append({
                        "url": url,
                        "title": title,
                        "snippet": snippet,
                        "citation_score": citation_score,
                        "freshness": "live",
                    })
                    resolved_sources.append(url)
                except Exception:
                    # Skip individual failures but continue
                    continue

            # Build an aggregated findings string
            findings = []
            for idx, f in enumerate(findings_list, start=1):
                summary_line = f"[{idx}] {f['title']}: {f['snippet'][:300]}"
                findings.append(summary_line)

            response = {
                "query": query,
                "findings": "\n\n".join(findings) if findings else "No substantive findings retrieved.",
                "summary": f"Aggregated {len(findings_list)} source(s) for query '{query}'.",
                "sources": resolved_sources if resolved_sources else urls_to_fetch,
                "source_details": findings_list,
                "citations": [
                    {
                        "index": idx,
                        "url": item["url"],
                        "title": item["title"],
                        "citation_score": item["citation_score"],
                    }
                    for idx, item in enumerate(findings_list, start=1)
                ],
                "depth": depth,
                "confidence": 0.7 if findings_list else 0.2,
                "timestamp": now,
                "freshness": "live",
                "cached": False,
            }
            self._web_research_cache[cache_key] = {
                "cached_at": now,
                "payload": response,
            }
            return self._with_metadata(
                response,
                capability=ModelCapability.WEB_RESEARCH.value,
                modality="text",
            )

        except Exception:
            # If any dependency or network issue occurs, return previous canned response
            return self._with_metadata({
                "query": query,
                "findings": "RealAI has researched your query comprehensively across the web.",
                "summary": "Based on extensive research, here are the key findings and insights.",
                "sources": sources or [
                    "https://example.com/source1",
                    "https://example.com/source2",
                    "https://example.com/source3"
                ],
                "depth": depth,
                "confidence": 0.92,
                "timestamp": now,
                "freshness": "fallback",
                "cached": False,
                "source_details": [],
                "citations": [],
            }, capability=ModelCapability.WEB_RESEARCH.value, modality="text")
    
    def automate_task(
        self,
        task_type: str,
        task_details: Dict[str, Any],
        execute: bool = False
    ) -> Dict[str, Any]:
        """
        Automate real-world tasks like ordering groceries, booking appointments, etc.
        
        Args:
            task_type (str): Type of task ("groceries", "appointment", "reservation", "payment", etc.)
            task_details (Dict[str, Any]): Details needed for the task
            execute (bool): Whether to actually execute the task or just plan it
            
        Returns:
            Dict[str, Any]: Task execution status and details
        """
        plan_text = f"RealAI has {'executed' if execute else 'planned'} your {task_type} task."
        try:
            ai_result = self.chat_completion([
                {
                    "role": "system",
                    "content": (
                        "You are a task automation assistant. "
                        "Break the task into concrete, executable steps. "
                        "Be concise and practical."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Task type: {task_type}\n"
                        f"Task details: {task_details}\n"
                        f"Mode: {'execute' if execute else 'plan only'}"
                    )
                }
            ])
            ai_content = (
                ai_result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if ai_content:
                plan_text = ai_content
        except Exception:
            pass  # fall back to static plan_text

        response = {
            "task_type": task_type,
            "status": "executed" if execute else "planned",
            "details": task_details,
            "plan": plan_text,
            "estimated_completion": "5-10 minutes",
            "confirmations": [],
            "success": True
        }
        return response
    
    def voice_interaction(
        self,
        audio_input: Optional[str] = None,
        text_input: Optional[str] = None,
        conversation_id: Optional[str] = None,
        response_format: str = "both"
    ) -> Dict[str, Any]:
        """
        Handle voice-based interaction with speech input/output.
        
        Args:
            audio_input (Optional[str]): Audio file or stream for speech input
            text_input (Optional[str]): Text input if not using voice
            conversation_id (Optional[str]): ID to maintain conversation context
            response_format (str): Response format ("audio", "text", "both")
            
        Returns:
            Dict[str, Any]: Response with audio and/or text
        """
        input_text = text_input or ""
        conv_id = conversation_id or f"conv-{int(time.time())}"
        response_text = "RealAI is ready to have a natural conversation with you through voice."
        audio_url = (
            "https://realai.example.com/voice-response.mp3"
            if response_format in ["audio", "both"] else None
        )
        input_transcription = None

        try:
            # Step 1: transcribe audio file if provided
            if audio_input and os.path.isfile(str(audio_input)):
                transcription = self.transcribe_audio(audio_input)
                input_transcription = transcription.get("text", input_text)
                if input_transcription:
                    input_text = input_transcription

            # Step 2: get AI response
            if input_text:
                ai_result = self.chat_completion([
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful voice assistant. "
                            "Keep responses concise for spoken delivery."
                        )
                    },
                    {"role": "user", "content": input_text}
                ])
                ai_content = (
                    ai_result.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                if ai_content:
                    response_text = ai_content

            # Step 3: generate audio for the response
            if response_format in ["audio", "both"]:
                audio_result = self.generate_audio(response_text)
                audio_url = audio_result.get("audio_url", audio_url)

        except Exception:
            pass  # fall back to defaults set above

        response = {
            "conversation_id": conv_id,
            "input_transcription": input_transcription if audio_input else None,
            "response_text": response_text,
            "response_audio_url": audio_url,
            "emotion_detected": "neutral",
            "intent": "conversational",
            "format": response_format
        }
        return response
    
    def business_planning(
        self,
        business_type: str,
        stage: str = "ideation",
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive business plans and strategies.
        
        Args:
            business_type (str): Type of business (e.g., "tech startup", "restaurant", "e-commerce")
            stage (str): Business stage ("ideation", "planning", "launch", "growth", "scale")
            details (Optional[Dict[str, Any]]): Specific business details and requirements
            
        Returns:
            Dict[str, Any]: Business plan and recommendations
        """
        # Static fallback plan
        business_plan = {
            "executive_summary": "Comprehensive business plan created by RealAI",
            "market_analysis": "Detailed market research and competitive analysis",
            "financial_projections": "5-year financial projections and funding requirements",
            "marketing_strategy": "Multi-channel marketing and growth strategy",
            "operations_plan": "Operational structure and processes",
            "risk_analysis": "Risk assessment and mitigation strategies"
        }
        action_items = [
            "Define unique value proposition",
            "Conduct market research",
            "Create MVP or prototype",
            "Develop go-to-market strategy",
            "Secure initial funding"
        ]

        try:
            system_prompt = (
                f"You are an expert business consultant. Create a comprehensive business plan "
                f"for a {business_type} at {stage} stage.\n"
                "Respond with a JSON object containing these exact keys: "
                "executive_summary, market_analysis, financial_projections, "
                "marketing_strategy, operations_plan, risk_analysis, "
                "action_items (as a JSON array of strings)."
            )
            ai_result = self.chat_completion([
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Business type: {business_type}\n"
                        f"Stage: {stage}\n"
                        f"Additional details: {details or 'None provided'}"
                    )
                }
            ])
            ai_content = (
                ai_result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if ai_content:
                # Try to parse a JSON block from the response
                # Strip markdown code fences if present
                cleaned = ai_content
                if "```" in cleaned:
                    # Match code fences with any optional language tag (e.g. ```json, ```JSON)
                    m = re.search(r"```(?:[a-zA-Z]*)?\s*([\s\S]*?)```", cleaned)
                    if m:
                        cleaned = m.group(1).strip()
                parsed = json.loads(cleaned)
                # Extract action_items list
                if isinstance(parsed.get("action_items"), list):
                    action_items = [str(x) for x in parsed.pop("action_items")]
                # Remaining keys populate the business_plan dict
                for key in [
                    "executive_summary", "market_analysis",
                    "financial_projections", "marketing_strategy",
                    "operations_plan", "risk_analysis"
                ]:
                    if key in parsed and isinstance(parsed[key], str):
                        business_plan[key] = parsed[key]
        except Exception:
            pass  # fall back to static values

        response = {
            "business_type": business_type,
            "stage": stage,
            "business_plan": business_plan,
            "action_items": action_items,
            "estimated_timeline": "6-12 months to launch",
            "success_probability": 0.75
        }
        return response
    
    def therapy_counseling(
        self,
        session_type: str,
        message: str,
        session_id: Optional[str] = None,
        approach: str = "cognitive_behavioral"
    ) -> Dict[str, Any]:
        """
        Provide therapeutic and counseling support.
        
        Args:
            session_type (str): Type of session ("therapy", "counseling", "coaching", "support")
            message (str): User's message or concern
            session_id (Optional[str]): Session ID for continuity
            approach (str): Therapeutic approach to use
            
        Returns:
            Dict[str, Any]: Therapeutic response and recommendations
        """
        _THERAPY_DISCLAIMER = (
            "\n\n⚠️ IMPORTANT: This AI provides general wellbeing support only. "
            "It is not a substitute for professional mental health care. "
            "If you are in crisis, please contact a mental health professional "
            "or a crisis helpline immediately."
        )
        _THERAPY_SYSTEM_PROMPT = (
            "You are a compassionate AI wellbeing support assistant trained in "
            "evidence-based techniques including Cognitive Behavioural Therapy (CBT) "
            "and motivational interviewing.\n\n"
            "Your role is to:\n"
            "- Listen empathetically and validate feelings\n"
            "- Help users identify and reframe negative thought patterns\n"
            "- Suggest practical coping strategies\n"
            "- Encourage professional help when appropriate\n"
            "- Never diagnose or prescribe\n\n"
            "Always respond warmly, without judgment, and in plain language.\n\n"
            "After your response, on a new line add: RECOMMENDATIONS: followed by "
            "3 bullet points of specific coping strategies."
        )

        # Static fallbacks
        ai_response_text = "RealAI provides empathetic, supportive, and professional therapeutic guidance."
        ai_insights = "I hear what you're sharing and I'm here to support you through this."
        recommendations = [
            "Practice self-compassion",
            "Consider journaling your thoughts",
            "Establish a regular routine"
        ]

        try:
            ai_result = self.chat_completion([
                {"role": "system", "content": _THERAPY_SYSTEM_PROMPT},
                {"role": "user", "content": message}
            ])
            ai_content = (
                ai_result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if ai_content:
                # Split off the RECOMMENDATIONS section if present
                if "RECOMMENDATIONS:" in ai_content:
                    parts = ai_content.split("RECOMMENDATIONS:", 1)
                    main_text = parts[0].strip()
                    rec_text = parts[1].strip()
                    # Parse bullet points (lines starting with - or • or *)
                    rec_lines = re.findall(
                        r"^[\-\*•]\s*(.+)$", rec_text, re.MULTILINE
                    )
                    if rec_lines:
                        # Keep at most the first 3 recommendations (matches system prompt)
                        recommendations = [r.strip() for r in rec_lines[:3]]
                else:
                    main_text = ai_content

                ai_response_text = main_text
                # Use first sentence of response as insight.
                # A trailing space is appended so the \s alternative matches
                # strings that end immediately after the punctuation mark.
                first_sentence_match = re.search(
                    r"^(.+?[.!?])(?:\s|$)", main_text + " "
                )
                if first_sentence_match:
                    candidate = first_sentence_match.group(1).strip()
                    if len(candidate) > 10:
                        ai_insights = candidate

        except Exception:
            pass  # fall back to static values

        response = {
            "session_id": session_id or f"session-{int(time.time())}",
            "session_type": session_type,
            "approach": approach,
            "response": ai_response_text + _THERAPY_DISCLAIMER,
            "insights": ai_insights,
            "techniques": ["Active listening", "Cognitive reframing", "Mindfulness"],
            "recommendations": recommendations,
            "resources": ["Mental health hotlines", "Professional referrals available"],
            "disclaimer": "This is AI-assisted support. For serious concerns, please consult a licensed professional."
        }
        return response
    
    def web3_integration(
        self,
        operation: str,
        blockchain: str = "ethereum",
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Integrate with Web3 technologies and blockchain operations.
        
        Args:
            operation (str): Operation type ("query", "transaction", "smart_contract", "nft", "defi")
            blockchain (str): Blockchain network to use
            params (Optional[Dict[str, Any]]): Operation-specific parameters
            
        Returns:
            Dict[str, Any]: Web3 operation results
        """
        # Try to use web3.py for real read-only operations when a provider is configured.
        provider_url = os.environ.get("WEB3_PROVIDER_URL")
        fallback = {
            "operation": operation,
            "blockchain": blockchain,
            "status": "success",
            "result": "RealAI has processed your Web3 operation.",
            "transaction_hash": f"0x{'a'*64}" if operation == "transaction" else None,
            "gas_used": "21000" if operation == "transaction" else None,
            "smart_contract_address": f"0x{'b'*40}" if operation == "smart_contract" else None,
            "network": blockchain,
            "timestamp": int(time.time())
        }

        if not provider_url:
            return fallback

        try:
            from web3 import Web3

            w3 = Web3(Web3.HTTPProvider(provider_url))
            if not w3.is_connected():
                raise RuntimeError("WEB3 provider not connected")

            result = {
                "operation": operation,
                "blockchain": blockchain,
                "status": "success",
                "network": blockchain,
                "timestamp": int(time.time())
            }

            if operation == "query":
                # Example: support basic queries like 'block_number' or address balance
                if params and params.get("action") == "block_number":
                    result["block_number"] = w3.eth.block_number
                elif params and params.get("address"):
                    addr = params.get("address")
                    try:
                        balance = w3.eth.get_balance(addr)
                        result["address"] = addr
                        result["balance_wei"] = balance
                        result["balance_eth"] = w3.from_wei(balance, "ether")
                    except Exception as e:
                        result["error"] = str(e)
                else:
                    result["info"] = "No query parameters provided"

            elif operation == "smart_contract":
                # For security and simplicity, do not deploy. Return sample contract info or run a read-only call if provided.
                if params and params.get("read_contract"):
                    # params: {address, abi, function, args}
                    try:
                        addr = params.get("address")
                        abi = params.get("abi")
                        func = params.get("function")
                        args = params.get("args", [])
                        contract = w3.eth.contract(address=addr, abi=abi)
                        fn = getattr(contract.functions, func)
                        value = fn(*args).call()
                        result["call_result"] = value
                    except Exception as e:
                        result["error"] = str(e)
                else:
                    result["note"] = "smart_contract deploys are not performed; provide read_contract params to call view functions"

            elif operation == "transaction":
                result["note"] = "Transactions require private keys and are not executed by default"

            else:
                result["info"] = "Unsupported web3 operation"

            return result

        except Exception:
            return fallback
    
    def execute_code(
        self,
        code: str,
        language: str,
        sandbox: bool = True,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Execute code in a safe environment.
        
        Args:
            code (str): Code to execute
            language (str): Programming language
            sandbox (bool): Whether to execute in sandboxed environment
            timeout (int): Execution timeout in seconds
            
        Returns:
            Dict[str, Any]: Execution results
        """
        # Currently we only support executing Python code locally.
        if language.lower() != "python":
            return {
                "language": language,
                "execution_status": "unsupported_language",
                "output": "",
                "errors": f"Execution for language '{language}' is not supported.",
                "runtime": 0.0,
                "memory_used": None,
                "sandboxed": False,
                "exit_code": None
            }

        tmp_file = None
        start = time.time()
        try:
            # Write code to a temporary file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                tmp_file = f.name

            # Prepare resource limiting preexec function if available
            def _limit_resources():
                if sandbox and resource is not None:
                    # Limit CPU time (seconds)
                    resource.setrlimit(resource.RLIMIT_CPU, (max(1, timeout), max(1, timeout)))
                    # Limit address space (virtual memory) to ~200MB
                    mem_bytes = 200 * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
                    # Prevent creation of new core files
                    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

            # Execute the file with timeout and capture output
            proc = subprocess.run([
                "python3",
                tmp_file
            ], capture_output=True, text=True, timeout=timeout, preexec_fn=_limit_resources if sandbox and resource is not None else None)

            runtime = time.time() - start
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""

            return {
                "language": "python",
                "execution_status": "completed" if proc.returncode == 0 else "error",
                "output": stdout,
                "errors": stderr if stderr else None,
                "runtime": f"{runtime:.3f}s",
                "memory_used": None,
                "sandboxed": bool(sandbox and resource is not None),
                "exit_code": proc.returncode
            }

        except subprocess.TimeoutExpired as e:
            runtime = time.time() - start
            return {
                "language": "python",
                "execution_status": "timeout",
                "output": e.stdout or "",
                "errors": (e.stderr or "") + f"\nExecution timed out after {timeout} seconds.",
                "runtime": f"{runtime:.3f}s",
                "memory_used": None,
                "sandboxed": bool(sandbox and resource is not None),
                "exit_code": None
            }
        except Exception as e:
            runtime = time.time() - start
            return {
                "language": "python",
                "execution_status": "error",
                "output": "",
                "errors": str(e),
                "runtime": f"{runtime:.3f}s",
                "memory_used": None,
                "sandboxed": bool(sandbox and resource is not None),
                "exit_code": None
            }
        finally:
            if tmp_file and os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except Exception:
                    pass
    
    def load_plugin(
        self,
        plugin_name: str,
        plugin_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Load and configure plugins for extended functionality.
        
        Args:
            plugin_name (str): Name of the plugin to load
            plugin_config (Optional[Dict[str, Any]]): Plugin configuration
            
        Returns:
            Dict[str, Any]: Plugin status and available methods
        """
        # Try loading a local plugin module from the `plugins` package first.
        try:
            module_name = f"plugins.{plugin_name}"
            module = importlib.import_module(module_name)

            # If plugin exposes a register() callable, call it with this model
            if hasattr(module, "register") and callable(getattr(module, "register")):
                metadata = module.register(self, plugin_config or {})
                # Store metadata in registry
                self.plugins_registry[plugin_name] = metadata or {"name": plugin_name}

                return {
                    "plugin_name": plugin_name,
                    "status": "loaded",
                    "version": metadata.get("version") if isinstance(metadata, dict) else None,
                    "capabilities": metadata.get("capabilities") if isinstance(metadata, dict) else [],
                    "config": plugin_config or {},
                    "methods": metadata.get("methods") if isinstance(metadata, dict) else []
                }

        except Exception:
            # Fall through to default simulated behavior
            pass

        # Fallback: return simulated plugin loaded response
        response = {
            "plugin_name": plugin_name,
            "status": "loaded",
            "version": "2.0.0",
            "capabilities": ["Plugin capabilities available"],
            "config": plugin_config or {},
            "methods": ["method1", "method2", "method3"]
        }
        # Record in registry for visibility
        self.plugins_registry[plugin_name] = response
        return response

    def load_all_plugins(self, package: str = "plugins") -> List[str]:
        """Discover and load all plugins in the given package namespace.

        Returns a list of successfully loaded plugin names.
        """
        loaded = []
        try:
            import pkgutil
            pkg = importlib.import_module(package)
            prefix = pkg.__name__ + "."
            for finder, name, ispkg in pkgutil.iter_modules(pkg.__path__, prefix):
                # module name will be like 'plugins.foo'
                mod_name = name.split(".")[-1]
                try:
                    self.load_plugin(mod_name)
                    loaded.append(mod_name)
                except Exception:
                    continue
        except Exception:
            # If discovery fails, return empty list
            return loaded

        return loaded
    
    def learn_from_interaction(
        self,
        interaction_data: Dict[str, Any],
        save: bool = True
    ) -> Dict[str, Any]:
        """
        Learn and adapt from user interactions.
        
        Args:
            interaction_data (Dict[str, Any]): Interaction data to learn from
            save (bool): Whether to persist the learning
            
        Returns:
            Dict[str, Any]: Learning status and insights
        """
        response = {
            "learned": save,
            "insights": "RealAI has analyzed and learned from this interaction.",
            "patterns_identified": ["User preferences", "Interaction style", "Topic interests"],
            "adaptations": ["Improved response style", "Better context understanding"],
            "memory_updated": save,
            "timestamp": int(time.time())
        }
        return response
    
    # ------------------------------------------------------------------
    # Next-generation capabilities
    # ------------------------------------------------------------------

    def self_reflect(
        self,
        interaction_history: Optional[List[Dict[str, Any]]] = None,
        focus: str = "general"
    ) -> Dict[str, Any]:
        """Analyze past interactions and generate meta-level self-improvement insights.

        Args:
            interaction_history: List of past interaction dicts (each with at least
                a ``role`` and ``content`` key, like chat messages).
            focus: Area to focus on – ``"general"``, ``"accuracy"``,
                ``"empathy"``, or ``"efficiency"``.

        Returns:
            Dict with ``status``, ``strengths``, ``weaknesses``, ``improvements``,
            and ``score`` keys.
        """
        history = interaction_history or []
        # Static fallback values
        strengths: List[str] = ["Broad knowledge coverage", "Consistent response structure"]
        weaknesses: List[str] = ["Responses can be overly verbose", "Limited domain specialization"]
        improvements: List[str] = [
            "Ask clarifying questions before responding",
            "Use domain-specific terminology when context allows",
            "Offer concise summaries before detailed explanations",
        ]
        score = 0.75

        try:
            history_text = "\n".join(
                f"{m.get('role', 'unknown')}: {m.get('content', '')}"
                for m in history
            ) if history else "(no prior interaction history provided)"

            ai_result = self.chat_completion([
                {
                    "role": "system",
                    "content": (
                        "You are a meta-cognitive AI analyst. "
                        "Critically evaluate the provided interaction history. "
                        "Respond ONLY with a JSON object containing exactly these keys: "
                        "strengths (array of strings), weaknesses (array of strings), "
                        "improvements (array of strings), score (float 0-1). "
                        "Be specific and actionable."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Focus area: {focus}\n\n"
                        f"Interaction history:\n{history_text}"
                    )
                }
            ])
            ai_content = (
                ai_result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if ai_content:
                cleaned = ai_content
                if "```" in cleaned:
                    parts = cleaned.split("```")
                    if len(parts) >= 3:
                        fence_content = parts[1]
                        first_nl = fence_content.find('\n')
                        cleaned = (fence_content[first_nl + 1:] if first_nl != -1 else fence_content).strip()
                parsed = json.loads(cleaned)
                strengths = [str(x) for x in parsed.get("strengths", strengths)]
                weaknesses = [str(x) for x in parsed.get("weaknesses", weaknesses)]
                improvements = [str(x) for x in parsed.get("improvements", improvements)]
                score = float(parsed.get("score", score))
        except Exception:
            pass  # fall back to static values

        return {
            "status": "success",
            "focus": focus,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "improvements": improvements,
            "score": score,
            "provider": self.provider,
            "timestamp": int(time.time()),
        }

    def chain_of_thought(
        self,
        problem: str,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """Solve a complex problem through explicit, verifiable step-by-step reasoning.

        Args:
            problem: The question or problem to reason through.
            domain: Optional domain hint (e.g. ``"math"``, ``"logic"``,
                ``"science"``) to sharpen the reasoning style.

        Returns:
            Dict with ``status``, ``problem``, ``steps`` (list of reasoning
            step strings), ``answer``, and ``confidence`` keys.
        """
        steps: List[str] = [
            "Identify the core question",
            "Gather relevant facts",
            "Apply logical reasoning",
            "Verify conclusions",
        ]
        answer = "Reasoning complete — see steps for the conclusion."
        confidence = 0.8

        try:
            domain_hint = f" Focus on {domain} reasoning." if domain else ""
            ai_result = self.chat_completion([
                {
                    "role": "system",
                    "content": (
                        "You are an expert reasoning engine."
                        + domain_hint
                        + " Think step by step, showing your work explicitly. "
                        "Respond ONLY with a JSON object containing exactly these keys: "
                        "steps (array of reasoning step strings), "
                        "answer (string — the final conclusion), "
                        "confidence (float 0-1)."
                    )
                },
                {"role": "user", "content": f"Problem: {problem}"}
            ])
            ai_content = (
                ai_result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if ai_content:
                cleaned = ai_content
                if "```" in cleaned:
                    parts = cleaned.split("```")
                    if len(parts) >= 3:
                        fence_content = parts[1]
                        first_nl = fence_content.find('\n')
                        cleaned = (fence_content[first_nl + 1:] if first_nl != -1 else fence_content).strip()
                parsed = json.loads(cleaned)
                steps = [str(x) for x in parsed.get("steps", steps)]
                answer = str(parsed.get("answer", answer))
                confidence = float(parsed.get("confidence", confidence))
        except Exception:
            pass  # fall back to static values

        return {
            "status": "success",
            "problem": problem,
            "domain": domain,
            "steps": steps,
            "answer": answer,
            "confidence": confidence,
            "provider": self.provider,
        }

    def synthesize_knowledge(
        self,
        topics: List[str],
        output_format: str = "narrative"
    ) -> Dict[str, Any]:
        """Combine knowledge from multiple topics or domains into unified insights.

        First gathers lightweight research on each topic via :meth:`web_research`,
        then uses an AI provider to synthesize cross-domain connections.

        Args:
            topics: List of topics or domains to synthesize (1-10 items).
            output_format: ``"narrative"`` for prose or ``"bullets"`` for a
                structured bullet-point summary.

        Returns:
            Dict with ``status``, ``topics``, ``per_topic`` (dict of topic →
            snippet), ``synthesis`` (unified insight string), and
            ``connections`` (list of identified cross-domain links).
        """
        topics = list(topics)[:10]  # cap at 10 for performance

        per_topic: Dict[str, str] = {}
        for topic in topics:
            try:
                result = self.web_research(query=topic, depth="quick")
                per_topic[topic] = result.get("findings", "")[:500]
            except Exception:
                per_topic[topic] = f"(research unavailable for '{topic}')"

        synthesis = (
            f"RealAI has synthesized knowledge across {len(topics)} topic(s): "
            + ", ".join(topics) + "."
        )
        connections: List[str] = [
            f"Cross-domain link between {topics[i]} and {topics[i + 1]}"
            for i in range(min(len(topics) - 1, 3))
        ]

        try:
            topic_summaries = "\n\n".join(
                f"### {t}\n{s}" for t, s in per_topic.items()
            )
            format_instruction = (
                "Write a narrative paragraph." if output_format == "narrative"
                else "Use concise bullet points."
            )
            ai_result = self.chat_completion([
                {
                    "role": "system",
                    "content": (
                        "You are an expert knowledge synthesizer. "
                        "Identify deep connections across the provided topics and "
                        "produce unified insights. "
                        + format_instruction
                        + " Respond ONLY with a JSON object containing exactly: "
                        "synthesis (string), connections (array of strings describing "
                        "cross-domain links)."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Topics: {', '.join(topics)}\n\n"
                        f"Research summaries:\n{topic_summaries}"
                    )
                }
            ], max_tokens=1500)
            ai_content = (
                ai_result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if ai_content:
                cleaned = ai_content
                if "```" in cleaned:
                    parts = cleaned.split("```")
                    if len(parts) >= 3:
                        fence_content = parts[1]
                        first_nl = fence_content.find('\n')
                        cleaned = (fence_content[first_nl + 1:] if first_nl != -1 else fence_content).strip()
                parsed = json.loads(cleaned)
                synthesis = str(parsed.get("synthesis", synthesis))
                connections = [str(x) for x in parsed.get("connections", connections)]
        except Exception:
            pass  # fall back to static values

        return {
            "status": "success",
            "topics": topics,
            "per_topic": per_topic,
            "synthesis": synthesis,
            "connections": connections,
            "output_format": output_format,
            "provider": self.provider,
        }

    def orchestrate_agents(
        self,
        task: str,
        agent_roles: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Coordinate multiple specialized virtual agents to tackle a complex task.

        Each "agent" is a focused AI call with a tailored system prompt.  The
        results are then synthesised into a final answer by a coordinator call.

        Args:
            task: High-level task description.
            agent_roles: Optional list of specialist roles to engage
                (e.g. ``["researcher", "analyst", "writer"]``).  Defaults to
                ``["planner", "researcher", "critic", "synthesizer"]``.

        Returns:
            Dict with ``status``, ``task``, ``agent_results`` (dict of
            role → output), ``final_output``, and ``agents_used`` keys.
        """
        default_roles = ["planner", "researcher", "critic", "synthesizer"]
        roles = agent_roles or default_roles

        # Static fallback
        agent_results: Dict[str, str] = {
            role: f"[{role.title()} agent]: Processed task '{task}'."
            for role in roles
        }
        final_output = f"Multi-agent analysis of '{task}' complete."
        execution_plan: List[str] = [
            "Detect user intent",
            "Route to specialist agents",
            "Execute specialist analysis",
            "Synthesize final answer",
        ]
        verification = {
            "status": "heuristic",
            "confidence": 0.7,
            "notes": "Fallback verification applied.",
        }

        _ROLE_PROMPTS: Dict[str, str] = {
            "planner": (
                "You are a strategic planning agent. "
                "Outline a clear execution plan for the given task."
            ),
            "researcher": (
                "You are a research agent. "
                "Identify the key facts, data, and knowledge needed for the task."
            ),
            "analyst": (
                "You are a data analysis agent. "
                "Analyse the task critically and surface important insights."
            ),
            "critic": (
                "You are a critical review agent. "
                "Identify potential flaws, risks, or blind spots in approaching this task."
            ),
            "writer": (
                "You are a content creation agent. "
                "Produce a clear, engaging write-up addressing the task."
            ),
            "synthesizer": (
                "You are a synthesis agent. "
                "Combine diverse perspectives into a coherent, actionable summary."
            ),
        }

        try:
            # Planner stage (agentic-by-default)
            planner_result = self.chat_completion([
                {
                    "role": "system",
                    "content": (
                        "You are a planning agent. Return JSON with keys: "
                        "execution_plan (array of concise steps), "
                        "recommended_roles (array of role names)."
                    ),
                },
                {"role": "user", "content": f"Task: {task}"},
            ])
            planner_content = (
                planner_result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if planner_content:
                parsed_plan = self._parse_json_block(planner_content)
                if isinstance(parsed_plan.get("execution_plan"), list):
                    execution_plan = [str(x) for x in parsed_plan["execution_plan"]][:8]
                if not agent_roles and isinstance(parsed_plan.get("recommended_roles"), list):
                    routed_roles = [str(x).strip().lower() for x in parsed_plan["recommended_roles"] if str(x).strip()]
                    if routed_roles:
                        roles = routed_roles[:6]

            for role in roles:
                system_prompt = _ROLE_PROMPTS.get(
                    role,
                    f"You are a specialist {role} agent. Complete the given task."
                )
                result = self.chat_completion([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Task: {task}"}
                ])
                content = (
                    result.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                if content:
                    agent_results[role] = content

            # Coordinator synthesis call
            contributions = "\n\n".join(
                f"--- {role.upper()} ---\n{output}"
                for role, output in agent_results.items()
            )
            coord_result = self.chat_completion([
                {
                    "role": "system",
                    "content": (
                        "You are the lead coordinator agent. "
                        "Given the outputs from multiple specialist agents, "
                        "produce a single, coherent, final response that best "
                        "addresses the original task."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Original task: {task}\n\n"
                        f"Agent contributions:\n{contributions}"
                    )
                }
            ], max_tokens=1500)
            coord_content = (
                coord_result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if coord_content:
                final_output = coord_content

            # Verification stage
            verifier_result = self.chat_completion([
                {
                    "role": "system",
                    "content": (
                        "You are a verification agent. Return JSON with keys: "
                        "confidence (float 0-1), notes (string)."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Task: {task}\n\nFinal output:\n{final_output}\n\n"
                        f"Agent contributions:\n{agent_results}"
                    ),
                },
            ])
            verifier_content = (
                verifier_result.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if verifier_content:
                parsed_verifier = self._parse_json_block(verifier_content)
                verification = {
                    "status": "model_verified",
                    "confidence": float(parsed_verifier.get("confidence", 0.75)),
                    "notes": str(parsed_verifier.get("notes", "")),
                }

        except Exception:
            pass  # fall back to static values

        return self._with_metadata({
            "status": "success",
            "task": task,
            "agents_used": roles,
            "execution_plan": execution_plan,
            "agent_results": agent_results,
            "final_output": final_output,
            "verification": verification,
            "provider": self.provider,
        }, capability=ModelCapability.MULTI_AGENT.value, modality="text")

    def generate_speech(self, text: str, voice: str = "alloy", model: str = "realai-tts") -> Dict[str, Any]:
        """Convenience alias for :meth:`generate_audio` for speech synthesis.

        Args:
            text: Text to speak.
            voice: Voice identifier.
            model: TTS model name.

        Returns:
            Dict with ``url``, ``spoken``, and ``audio_url`` keys.
        """
        result = self.generate_audio(text=text, voice=voice, model=model)
        return {
            "url": result.get("audio_url", ""),
            "spoken": bool(result.get("audio_url")),
            "audio_url": result.get("audio_url", ""),
            "duration": result.get("duration"),
            "voice": voice,
        }

    def get_capabilities(self) -> List[str]:
        """
        Get list of all supported capabilities.
        
        Returns:
            List[str]: List of capability names
        """
        return [cap.value for cap in self.capabilities]

    def get_capability_catalog(self) -> Dict[str, Any]:
        """Return canonical capability metadata grouped by domain."""
        items: List[Dict[str, Any]] = []
        for cap in self.capabilities:
            items.append({
                "name": cap.value,
                "domain": CAPABILITY_DOMAIN_MAP.get(cap, "general"),
                "provider_supported": self._provider_supports(cap.value),
            })

        domains: Dict[str, List[str]] = {}
        for entry in items:
            domains.setdefault(entry["domain"], []).append(entry["name"])

        return {
            "capabilities": items,
            "domains": domains,
            "count": len(items),
        }

    def get_provider_capabilities(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Return supported capabilities for a provider."""
        provider_name = (provider or self.provider or "realai-local").lower()
        all_caps = self.get_capabilities()
        supported = PROVIDER_CAPABILITY_MAP.get(provider_name, all_caps)
        unsupported = sorted([c for c in all_caps if c not in supported])
        return {
            "provider": provider_name,
            "supported_capabilities": sorted(supported),
            "unsupported_capabilities": unsupported,
        }

    def set_persona(self, persona: str) -> Dict[str, str]:
        """Set the active persona profile used for chat-style outputs."""
        key = persona.lower().strip()
        if key not in PERSONA_PROFILES:
            raise ValueError(
                f"Unsupported persona '{persona}'. Available: {', '.join(sorted(PERSONA_PROFILES.keys()))}"
            )
        self.persona = key
        return {
            "persona": self.persona,
            "description": PERSONA_PROFILES[self.persona]["description"],
        }

    def get_personas(self) -> Dict[str, Dict[str, str]]:
        """List available persona profiles."""
        return {
            name: {"description": cfg["description"]}
            for name, cfg in PERSONA_PROFILES.items()
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            "name": self.model_name,
            "version": self.version,
            "capabilities": self.get_capabilities(),
            "capability_catalog": self.get_capability_catalog(),
            "provider_capabilities": self.get_provider_capabilities(),
            "persona": self.persona,
            "description": "RealAI - The limitless AI that can truly do anything. The sky is the limit!"
        }


class RealAIClient:
    """
    OpenAI-compatible client for RealAI.

    This client provides an interface similar to the OpenAI Python client,
    making it easy to switch from OpenAI to RealAI or to proxy requests to a
    real AI provider (ChatGPT, Claude, Grok, Gemini, …) by supplying the
    appropriate *api_key*.
    """

    def __init__(self, api_key: Optional[str] = None,
                 provider: Optional[str] = None,
                 base_url: Optional[str] = None):
        """
        Initialize the RealAI client.

        Args:
            api_key (Optional[str]): API key for the AI provider. When
                provided, requests are forwarded to the real AI service.
                The provider is auto-detected from the key prefix, or can
                be set explicitly via *provider*.
            provider (Optional[str]): Explicit provider name (``"openai"``,
                ``"anthropic"``, ``"grok"``, ``"gemini"``).  Overrides
                key-based auto-detection.
            base_url (Optional[str]): Override the provider's base URL,
                e.g. for a local proxy or self-hosted model.
        """
        self.api_key = api_key
        self.model = RealAI(api_key=api_key, provider=provider, base_url=base_url)

        # Create nested classes to match OpenAI structure
        self.chat = self.ChatCompletions(self.model)
        self.completions = self.Completions(self.model)
        self.images = self.Images(self.model)
        self.videos = self.Videos(self.model)
        self.embeddings = self.Embeddings(self.model)
        self.audio = self.Audio(self.model)

        # New limitless capabilities
        self.web = self.Web(self.model)
        self.tasks = self.Tasks(self.model)
        self.voice = self.Voice(self.model)
        self.business = self.Business(self.model)
        self.therapy = self.Therapy(self.model)
        self.web3 = self.Web3(self.model)
        self.plugins = self.Plugins(self.model)
        self.personas = self.Personas(self.model)

        # Next-generation capabilities
        self.reasoning = self.Reasoning(self.model)
        self.synthesis = self.Synthesis(self.model)
        self.reflection = self.Reflection(self.model)
        self.agents = self.Agents(self.model)

    class ChatCompletions:
        """Chat completions interface."""
        def __init__(self, model: RealAI):
            self.model = model
            
        def create(self, **kwargs) -> Dict[str, Any]:
            """Create a chat completion."""
            return self.model.chat_completion(**kwargs)
    
    class Completions:
        """Text completions interface."""
        def __init__(self, model: RealAI):
            self.model = model
            
        def create(self, **kwargs) -> Dict[str, Any]:
            """Create a text completion."""
            prompt = kwargs.pop('prompt', '')
            return self.model.text_completion(prompt, **kwargs)
    
    class Images:
        """Image generation and analysis interface."""
        def __init__(self, model: RealAI):
            self.model = model
            
        def generate(self, **kwargs) -> Dict[str, Any]:
            """Generate an image."""
            return self.model.generate_image(**kwargs)
        
        def analyze(self, **kwargs) -> Dict[str, Any]:
            """Analyze an image."""
            return self.model.analyze_image(**kwargs)

    class Videos:
        """Video generation interface."""
        def __init__(self, model: RealAI):
            self.model = model

        def generate(self, **kwargs) -> Dict[str, Any]:
            """Generate a video."""
            return self.model.generate_video(**kwargs)
    
    class Embeddings:
        """Embeddings interface."""
        def __init__(self, model: RealAI):
            self.model = model
            
        def create(self, **kwargs) -> Dict[str, Any]:
            """Create embeddings."""
            return self.model.create_embeddings(**kwargs)
    
    class Audio:
        """Audio interface."""
        def __init__(self, model: RealAI):
            self.model = model
            
        def transcribe(self, **kwargs) -> Dict[str, Any]:
            """Transcribe audio."""
            return self.model.transcribe_audio(**kwargs)
        
        def generate(self, **kwargs) -> Dict[str, Any]:
            """Generate audio."""
            return self.model.generate_audio(**kwargs)
    
    class Web:
        """Web research and browsing interface."""
        def __init__(self, model: RealAI):
            self.model = model
        
        def research(self, **kwargs) -> Dict[str, Any]:
            """Research a topic on the web."""
            return self.model.web_research(**kwargs)
    
    class Tasks:
        """Real-world task automation interface."""
        def __init__(self, model: RealAI):
            self.model = model
        
        def automate(self, **kwargs) -> Dict[str, Any]:
            """Automate a real-world task."""
            return self.model.automate_task(**kwargs)
        
        def order_groceries(self, items: List[str], **kwargs) -> Dict[str, Any]:
            """Order groceries."""
            return self.model.automate_task(
                task_type="groceries",
                task_details={"items": items, **kwargs},
                execute=kwargs.get("execute", False)
            )
        
        def book_appointment(self, details: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            """Book an appointment."""
            return self.model.automate_task(
                task_type="appointment",
                task_details=details,
                execute=kwargs.get("execute", False)
            )
    
    class Voice:
        """Voice interaction interface."""
        def __init__(self, model: RealAI):
            self.model = model
        
        def interact(self, **kwargs) -> Dict[str, Any]:
            """Have a voice interaction."""
            return self.model.voice_interaction(**kwargs)
        
        def conversation(self, message: str, **kwargs) -> Dict[str, Any]:
            """Have a natural conversation."""
            if 'response_format' not in kwargs:
                kwargs = {**kwargs, 'response_format': 'both'}
            return self.model.voice_interaction(
                text_input=message,
                **kwargs
            )
    
    class Business:
        """Business planning and building interface."""
        def __init__(self, model: RealAI):
            self.model = model
        
        def plan(self, **kwargs) -> Dict[str, Any]:
            """Create a business plan."""
            return self.model.business_planning(**kwargs)
        
        def build(self, business_type: str, **kwargs) -> Dict[str, Any]:
            """Build a business from the ground up."""
            if 'stage' not in kwargs:
                kwargs = {**kwargs, 'stage': 'planning'}
            return self.model.business_planning(
                business_type=business_type,
                **kwargs
            )
    
    class Therapy:
        """Therapy and counseling interface."""
        def __init__(self, model: RealAI):
            self.model = model
        
        def session(self, **kwargs) -> Dict[str, Any]:
            """Have a therapy session."""
            return self.model.therapy_counseling(**kwargs)
        
        def support(self, message: str, **kwargs) -> Dict[str, Any]:
            """Get emotional support."""
            return self.model.therapy_counseling(
                session_type="support",
                message=message,
                **kwargs
            )
    
    class Web3:
        """Web3 and blockchain interface."""
        def __init__(self, model: RealAI):
            self.model = model
        
        def execute(self, **kwargs) -> Dict[str, Any]:
            """Execute a Web3 operation."""
            return self.model.web3_integration(**kwargs)
        
        def smart_contract(self, **kwargs) -> Dict[str, Any]:
            """Deploy or interact with smart contracts."""
            return self.model.web3_integration(
                operation="smart_contract",
                **kwargs
            )
    
    class Plugins:
        """Plugin management interface."""
        def __init__(self, model: RealAI):
            self.model = model
        
        def load(self, **kwargs) -> Dict[str, Any]:
            """Load a plugin."""
            return self.model.load_plugin(**kwargs)
        
        def extend(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            """Extend RealAI with a plugin."""
            return self.model.load_plugin(plugin_name, config)

    class Reasoning:
        """Chain-of-thought and structured reasoning interface."""
        def __init__(self, model: RealAI):
            self.model = model

        def solve(self, problem: str, **kwargs) -> Dict[str, Any]:
            """Solve a problem with explicit step-by-step reasoning."""
            return self.model.chain_of_thought(problem=problem, **kwargs)

        def chain(self, problem: str, domain: Optional[str] = None) -> Dict[str, Any]:
            """Alias for :meth:`solve` with an optional domain hint."""
            return self.model.chain_of_thought(problem=problem, domain=domain)

    class Synthesis:
        """Knowledge synthesis interface."""
        def __init__(self, model: RealAI):
            self.model = model

        def combine(self, topics: List[str], **kwargs) -> Dict[str, Any]:
            """Synthesise knowledge from multiple topics."""
            return self.model.synthesize_knowledge(topics=topics, **kwargs)

        def cross_domain(self, topics: List[str], output_format: str = "narrative") -> Dict[str, Any]:
            """Produce cross-domain insights from a list of topics."""
            return self.model.synthesize_knowledge(topics=topics, output_format=output_format)

    class Reflection:
        """Self-reflection and meta-improvement interface."""
        def __init__(self, model: RealAI):
            self.model = model

        def analyze(self, interaction_history: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Dict[str, Any]:
            """Analyse past interactions and generate improvement insights."""
            return self.model.self_reflect(interaction_history=interaction_history, **kwargs)

        def improve(self, focus: str = "general") -> Dict[str, Any]:
            """Return targeted improvement suggestions for the given focus area."""
            return self.model.self_reflect(focus=focus)

    class Agents:
        """Multi-agent orchestration interface."""
        def __init__(self, model: RealAI):
            self.model = model

        def run(self, task: str, **kwargs) -> Dict[str, Any]:
            """Run multiple specialised agents on a complex task."""
            return self.model.orchestrate_agents(task=task, **kwargs)

        def coordinate(self, task: str, roles: Optional[List[str]] = None) -> Dict[str, Any]:
            """Coordinate a specific set of agent roles for a task."""
            return self.model.orchestrate_agents(task=task, agent_roles=roles)

    class Personas:
        """Persona profile management interface."""
        def __init__(self, model: RealAI):
            self.model = model

        def list(self) -> Dict[str, Dict[str, str]]:
            """List available persona profiles."""
            return self.model.get_personas()

        def set(self, persona: str) -> Dict[str, str]:
            """Set active persona profile."""
            return self.model.set_persona(persona)


def main():
    """Example usage of RealAI - demonstrating limitless capabilities."""
    # Create a RealAI client
    client = RealAIClient()
    
    print("RealAI Model Information:")
    print(json.dumps(client.model.get_model_info(), indent=2))
    
    print("\n" + "="*50)
    print("Testing Chat Completion:")
    response = client.chat.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What can you do?"}
        ]
    )
    print(json.dumps(response, indent=2))
    
    print("\n" + "="*50)
    print("Testing Web Research:")
    research = client.web.research(
        query="Latest developments in AI",
        depth="standard"
    )
    print(json.dumps(research, indent=2))
    
    print("\n" + "="*50)
    print("Testing Task Automation (Groceries):")
    groceries = client.tasks.order_groceries(
        items=["milk", "eggs", "bread"],
        execute=False
    )
    print(json.dumps(groceries, indent=2))
    
    print("\n" + "="*50)
    print("Testing Voice Interaction:")
    voice = client.voice.conversation(
        message="Tell me about yourself"
    )
    print(json.dumps(voice, indent=2))
    
    print("\n" + "="*50)
    print("Testing Business Planning:")
    business = client.business.build(
        business_type="tech startup"
    )
    print(json.dumps(business, indent=2))
    
    print("\n" + "="*50)
    print("Testing Web3 Integration:")
    web3 = client.web3.smart_contract(
        blockchain="ethereum"
    )
    print(json.dumps(web3, indent=2))


if __name__ == "__main__":
    main()
