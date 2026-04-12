from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=None, alias="max_tokens")
    temperature: Optional[float] = None
    stream: Optional[bool] = False

    class Config:
        allow_population_by_field_name = True
        extra = "ignore"


class ImageGenerationRequest(BaseModel):
    """Request body for POST /v1/images/generations."""

    prompt: str
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"

    class Config:
        extra = "ignore"


class EmbeddingsRequest(BaseModel):
    """Request body for POST /v1/embeddings."""

    input: Union[str, List[str]]
    model: Optional[str] = None

    class Config:
        extra = "ignore"


class AudioTranscriptionRequest(BaseModel):
    """Request body for POST /v1/audio/transcriptions (JSON variant)."""

    audio_path: str
    language: Optional[str] = None

    class Config:
        extra = "ignore"


class AudioSpeechRequest(BaseModel):
    """Request body for POST /v1/audio/speech."""

    input: str
    voice: Optional[str] = "alloy"
    model: Optional[str] = None

    class Config:
        extra = "ignore"


class CodeGenerationRequest(BaseModel):
    """Request body for POST /v1/code/generate."""

    prompt: str
    language: Optional[str] = "python"

    class Config:
        extra = "ignore"


class TranslationRequest(BaseModel):
    """Request body for POST /v1/translate."""

    text: str
    target_language: str

    class Config:
        extra = "ignore"


class WebResearchRequest(BaseModel):
    """Request body for POST /v1/research."""

    query: str

    class Config:
        extra = "ignore"


class CodeExecutionRequest(BaseModel):
    """Request body for POST /v1/execute."""

    code: str
    language: Optional[str] = "python"

    class Config:
        extra = "ignore"


class CapabilityInfo(BaseModel):
    """Describes a single RealAI capability."""

    name: str
    description: str
    endpoint: Optional[str] = None
