# RealAI - The AI Model That Can Do It All

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

> *All the AI models want to be me* 🚀

RealAI is a comprehensive AI model designed to be used like OpenAI was supposed to be - complete, powerful, and easy to use. It provides a unified interface for various AI capabilities in one model.

## Features

RealAI combines multiple AI capabilities into a single, cohesive model:

- 💬 **Chat Completion** - Conversational AI like ChatGPT
- 📝 **Text Generation** - Complete text from prompts like GPT-3
- 🎨 **Image Generation** - Create images from text like DALL-E
- 👁️ **Image Analysis** - Understand and describe images like GPT-4 Vision
- 💻 **Code Generation** - Generate and understand code like Codex
- 🔤 **Embeddings** - Create semantic embeddings for text
- 🎤 **Audio Transcription** - Convert speech to text like Whisper
- 🔊 **Audio Generation** - Text-to-speech capabilities
- 🌍 **Translation** - Translate between languages

## Installation

### From Source

```bash
git clone https://github.com/Unwrenchable/realai.git
cd realai
pip install -e .
```

### Quick Install

```bash
pip install -e git+https://github.com/Unwrenchable/realai.git#egg=realai
```

## Quick Start

```python
from realai import RealAIClient

# Initialize the client (OpenAI-compatible interface)
client = RealAIClient()

# Chat completion
response = client.chat.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What can you do?"}
    ]
)
print(response)

# Image generation
image = client.images.generate(
    prompt="A beautiful sunset over mountains",
    size="1024x1024"
)
print(image)

# Code generation
code = client.model.generate_code(
    prompt="Create a function to sort a list",
    language="python"
)
print(code)
```

## Usage Examples

### Chat Completion (ChatGPT-style)

```python
from realai import RealAIClient

client = RealAIClient()

response = client.chat.create(
    messages=[
        {"role": "system", "content": "You are a coding expert."},
        {"role": "user", "content": "Explain recursion in simple terms."}
    ],
    temperature=0.7,
    max_tokens=150
)

print(response['choices'][0]['message']['content'])
```

### Text Completion (GPT-3-style)

```python
response = client.completions.create(
    prompt="The future of AI is",
    temperature=0.8,
    max_tokens=100
)

print(response['choices'][0]['text'])
```

### Image Generation (DALL-E-style)

```python
image_response = client.images.generate(
    prompt="A futuristic cityscape with neon lights",
    size="1024x1024",
    quality="hd",
    n=1
)

print(f"Generated image URL: {image_response['data'][0]['url']}")
```

### Image Analysis (GPT-4 Vision-style)

```python
analysis = client.images.analyze(
    image_url="https://example.com/photo.jpg",
    prompt="Describe what you see in this image"
)

print(analysis['description'])
```

### Code Generation (Codex-style)

```python
code_response = client.model.generate_code(
    prompt="Create a REST API endpoint in Flask",
    language="python"
)

print(code_response['code'])
```

### Embeddings

```python
embeddings = client.embeddings.create(
    input_text=["Hello world", "RealAI is powerful"]
)

print(f"Embedding dimension: {len(embeddings['data'][0]['embedding'])}")
```

### Audio Transcription (Whisper-style)

```python
transcription = client.audio.transcribe(
    audio_file="recording.mp3",
    language="en"
)

print(transcription['text'])
```

### Audio Generation (TTS)

```python
audio = client.audio.generate(
    text="Hello from RealAI!",
    voice="alloy"
)

print(f"Audio URL: {audio['audio_url']}")
```

### Translation

```python
translation = client.model.translate(
    text="Hello, how are you?",
    target_language="es"
)

print(translation['translated_text'])
```

## API Reference

### RealAIClient

The main client class that provides an OpenAI-compatible interface.

```python
client = RealAIClient(api_key="your-api-key")  # api_key is optional
```

#### Available Methods

- `client.chat.create(**kwargs)` - Create chat completions
- `client.completions.create(**kwargs)` - Create text completions
- `client.images.generate(**kwargs)` - Generate images
- `client.images.analyze(**kwargs)` - Analyze images
- `client.embeddings.create(**kwargs)` - Create embeddings
- `client.audio.transcribe(**kwargs)` - Transcribe audio
- `client.audio.generate(**kwargs)` - Generate audio

### RealAI Model

Direct access to the underlying model:

```python
from realai import RealAI

model = RealAI(model_name="realai-1.0")

# Get model information
info = model.get_model_info()

# Get supported capabilities
capabilities = model.get_capabilities()
```

## Command Line Usage

After installation, you can run the example directly:

```bash
python -m realai
```

Or use the console script:

```bash
realai
```

## Architecture

RealAI is designed with the following principles:

1. **Unified Interface** - One model for all AI tasks
2. **OpenAI Compatibility** - Drop-in replacement for OpenAI's API structure
3. **Comprehensive Capabilities** - Text, images, code, audio, and more
4. **Easy to Use** - Simple, intuitive API
5. **Self-Contained** - No heavy dependencies required

## Capabilities

RealAI supports the following capabilities out of the box:

| Capability | Description | Similar To |
|------------|-------------|------------|
| Text Generation | Generate and complete text | GPT-3, GPT-4 |
| Chat Completion | Conversational AI | ChatGPT |
| Image Generation | Create images from text | DALL-E |
| Image Analysis | Understand and describe images | GPT-4 Vision |
| Code Generation | Generate and understand code | Codex, Copilot |
| Embeddings | Semantic text embeddings | text-embedding-ada-002 |
| Audio Transcription | Speech to text | Whisper |
| Audio Generation | Text to speech | TTS |
| Translation | Multilingual translation | Google Translate |

## Why RealAI?

- ✅ **All-in-One**: Everything you need in one model
- ✅ **OpenAI-Compatible**: Easy migration from OpenAI
- ✅ **Simple API**: Clean, intuitive interface
- ✅ **No Lock-in**: Open source and free to use
- ✅ **Lightweight**: Minimal dependencies
- ✅ **Extensible**: Easy to add new capabilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues and questions, please open an issue on [GitHub](https://github.com/Unwrenchable/realai/issues).

---

**RealAI** - The AI model that can do it all. All the AI models want to be me. 🌟
