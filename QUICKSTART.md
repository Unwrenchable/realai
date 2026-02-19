# Quick Start Guide

Get started with RealAI in under 5 minutes!

## Step 1: Installation

Clone and install:

```bash
git clone https://github.com/Unwrenchable/realai.git
cd realai
pip install -e .
```

## Step 2: Run Examples

Try the built-in examples:

```bash
python examples.py
```

This will demonstrate all 8 capabilities of RealAI.

## Step 3: Use in Your Code

Create a new Python file:

```python
from realai import RealAIClient

# Initialize client
client = RealAIClient()

# Use chat completion
response = client.chat.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ]
)

print(response['choices'][0]['message']['content'])
```

## Step 4: Start API Server (Optional)

For REST API access:

```bash
python api_server.py
```

Then test with curl:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Step 5: Run Tests

Verify everything works:

```bash
python test_realai.py
```

## What Can You Do?

RealAI provides 8 major capabilities:

1. **Chat** - Conversational AI
2. **Text** - Text generation and completion
3. **Images** - Generate and analyze images
4. **Code** - Generate and understand code
5. **Embeddings** - Semantic text vectors
6. **Audio** - Transcribe and generate audio
7. **Translation** - Multilingual translation
8. **Analysis** - Image and content understanding

## Learn More

- See [README.md](README.md) for overview
- See [API.md](API.md) for detailed API documentation
- See [examples.py](examples.py) for code examples

## Need Help?

Open an issue on [GitHub](https://github.com/Unwrenchable/realai/issues)

---

**RealAI** - The AI model that can do it all! 🚀
