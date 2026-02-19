# RealAI API Documentation

## Overview

RealAI provides a comprehensive, OpenAI-compatible API for various AI capabilities. This document describes all available endpoints and methods.

## Table of Contents

- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [API Methods](#api-methods)
  - [Chat Completions](#chat-completions)
  - [Text Completions](#text-completions)
  - [Image Generation](#image-generation)
  - [Image Analysis](#image-analysis)
  - [Code Generation](#code-generation)
  - [Embeddings](#embeddings)
  - [Audio Transcription](#audio-transcription)
  - [Audio Generation](#audio-generation)
  - [Translation](#translation)
- [REST API Server](#rest-api-server)

## Installation

```bash
pip install -e git+https://github.com/Unwrenchable/realai.git#egg=realai
```

Or from source:

```bash
git clone https://github.com/Unwrenchable/realai.git
cd realai
pip install -e .
```

## Basic Usage

```python
from realai import RealAIClient

# Initialize the client
client = RealAIClient(api_key="optional-api-key")

# Use any of the available methods
response = client.chat.create(
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## API Methods

### Chat Completions

Create a conversational response (like ChatGPT).

**Method:** `client.chat.create(**kwargs)`

**Parameters:**
- `messages` (List[Dict[str, str]], required): List of messages with 'role' and 'content'
- `temperature` (float, optional): Sampling temperature (0-2), default: 0.7
- `max_tokens` (int, optional): Maximum tokens to generate
- `stream` (bool, optional): Whether to stream the response, default: False

**Example:**
```python
response = client.chat.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What can you do?"}
    ],
    temperature=0.7,
    max_tokens=150
)
```

**Response:**
```json
{
  "id": "chatcmpl-1234567890",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "realai-1.0",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "I am RealAI..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

### Text Completions

Generate text completions (like GPT-3).

**Method:** `client.completions.create(**kwargs)`

**Parameters:**
- `prompt` (str, required): The text prompt
- `temperature` (float, optional): Sampling temperature (0-2), default: 0.7
- `max_tokens` (int, optional): Maximum tokens to generate

**Example:**
```python
response = client.completions.create(
    prompt="The future of AI is",
    temperature=0.8,
    max_tokens=100
)
```

### Image Generation

Generate images from text descriptions (like DALL-E).

**Method:** `client.images.generate(**kwargs)`

**Parameters:**
- `prompt` (str, required): The image description
- `size` (str, optional): Image size, default: "1024x1024"
- `quality` (str, optional): Image quality ("standard" or "hd"), default: "standard"
- `n` (int, optional): Number of images to generate, default: 1

**Example:**
```python
response = client.images.generate(
    prompt="A beautiful sunset over mountains",
    size="1024x1024",
    quality="hd",
    n=2
)
```

### Image Analysis

Analyze and describe images (like GPT-4 Vision).

**Method:** `client.images.analyze(**kwargs)`

**Parameters:**
- `image_url` (str, required): URL of the image to analyze
- `prompt` (str, optional): Question about the image

**Example:**
```python
response = client.images.analyze(
    image_url="https://example.com/image.jpg",
    prompt="What's in this image?"
)
```

### Code Generation

Generate code from descriptions (like Codex).

**Method:** `client.model.generate_code(**kwargs)`

**Parameters:**
- `prompt` (str, required): Description of the code to generate
- `language` (str, optional): Programming language
- `context` (str, optional): Additional context or existing code

**Example:**
```python
response = client.model.generate_code(
    prompt="Create a function to calculate factorial",
    language="python"
)
```

### Embeddings

Create semantic text embeddings.

**Method:** `client.embeddings.create(**kwargs)`

**Parameters:**
- `input_text` (Union[str, List[str]], required): Text or list of texts to embed
- `model` (str, optional): Embedding model to use, default: "realai-embeddings"

**Example:**
```python
response = client.embeddings.create(
    input_text=["Hello world", "AI is amazing"]
)
```

### Audio Transcription

Transcribe audio to text (like Whisper).

**Method:** `client.audio.transcribe(**kwargs)`

**Parameters:**
- `audio_file` (str, required): Path or URL to audio file
- `language` (str, optional): Language of the audio
- `prompt` (str, optional): Optional prompt to guide the model

**Example:**
```python
response = client.audio.transcribe(
    audio_file="recording.mp3",
    language="en"
)
```

### Audio Generation

Generate audio from text (text-to-speech).

**Method:** `client.audio.generate(**kwargs)`

**Parameters:**
- `text` (str, required): Text to convert to speech
- `voice` (str, optional): Voice to use, default: "alloy"
- `model` (str, optional): TTS model to use, default: "realai-tts"

**Example:**
```python
response = client.audio.generate(
    text="Hello from RealAI!",
    voice="alloy"
)
```

### Translation

Translate text between languages.

**Method:** `client.model.translate(**kwargs)`

**Parameters:**
- `text` (str, required): Text to translate
- `target_language` (str, required): Target language code (e.g., 'es', 'fr', 'de')
- `source_language` (str, optional): Source language (auto-detected if not provided)

**Example:**
```python
response = client.model.translate(
    text="Hello, how are you?",
    target_language="es"
)
```

## REST API Server

RealAI includes a built-in HTTP server that provides an OpenAI-compatible REST API.

### Starting the Server

```bash
python api_server.py
```

The server will start on `http://0.0.0.0:8000` by default.

### Available Endpoints

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model": "realai-1.0"
}
```

#### GET /v1/models
List available models.

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "realai-1.0",
      "object": "model",
      "created": 1708308000,
      "owned_by": "realai"
    }
  ]
}
```

#### POST /v1/chat/completions
Create a chat completion.

**Request Body:**
```json
{
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "temperature": 0.7,
  "max_tokens": 150
}
```

#### POST /v1/completions
Create a text completion.

**Request Body:**
```json
{
  "prompt": "Once upon a time",
  "temperature": 0.7,
  "max_tokens": 100
}
```

#### POST /v1/images/generations
Generate an image.

**Request Body:**
```json
{
  "prompt": "A beautiful sunset",
  "size": "1024x1024",
  "quality": "hd",
  "n": 1
}
```

#### POST /v1/embeddings
Create embeddings.

**Request Body:**
```json
{
  "input": "Hello world",
  "model": "realai-embeddings"
}
```

#### POST /v1/audio/transcriptions
Transcribe audio.

**Request Body:**
```json
{
  "file": "audio.mp3",
  "language": "en"
}
```

#### POST /v1/audio/speech
Generate speech.

**Request Body:**
```json
{
  "input": "Hello from RealAI",
  "voice": "alloy"
}
```

## Migration from OpenAI

RealAI is designed to be a drop-in replacement for OpenAI's client. Simply replace:

```python
# Old OpenAI code
from openai import OpenAI
client = OpenAI(api_key="...")

# New RealAI code
from realai import RealAIClient
client = RealAIClient(api_key="...")
```

The API structure and method calls remain the same!

## Examples

See [examples.py](examples.py) for comprehensive usage examples.

## Testing

Run the test suite:

```bash
python test_realai.py
```

## License

MIT License - See [LICENSE](LICENSE) file for details.
