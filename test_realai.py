"""
Tests for RealAI model.

Run with: python test_realai.py
"""

import sys
from realai import RealAI, RealAIClient, ModelCapability


def test_model_initialization():
    """Test model initialization."""
    print("Testing model initialization...")
    model = RealAI()
    assert model.model_name == "realai-1.0"
    assert model.version == "1.0.0"
    assert len(model.capabilities) > 0
    print("✓ Model initialization test passed")


def test_client_initialization():
    """Test client initialization."""
    print("Testing client initialization...")
    client = RealAIClient()
    assert client.model is not None
    assert hasattr(client, 'chat')
    assert hasattr(client, 'completions')
    assert hasattr(client, 'images')
    assert hasattr(client, 'embeddings')
    assert hasattr(client, 'audio')
    print("✓ Client initialization test passed")


def test_chat_completion():
    """Test chat completion."""
    print("Testing chat completion...")
    client = RealAIClient()
    response = client.chat.create(
        messages=[
            {"role": "user", "content": "Hello"}
        ]
    )
    assert 'id' in response
    assert 'choices' in response
    assert len(response['choices']) > 0
    assert 'message' in response['choices'][0]
    print("✓ Chat completion test passed")


def test_text_completion():
    """Test text completion."""
    print("Testing text completion...")
    client = RealAIClient()
    response = client.completions.create(
        prompt="Hello world"
    )
    assert 'id' in response
    assert 'choices' in response
    assert len(response['choices']) > 0
    print("✓ Text completion test passed")


def test_image_generation():
    """Test image generation."""
    print("Testing image generation...")
    client = RealAIClient()
    response = client.images.generate(
        prompt="A sunset"
    )
    assert 'data' in response
    assert len(response['data']) > 0
    assert 'url' in response['data'][0]
    print("✓ Image generation test passed")


def test_image_analysis():
    """Test image analysis."""
    print("Testing image analysis...")
    client = RealAIClient()
    response = client.images.analyze(
        image_url="https://example.com/test.jpg"
    )
    assert 'analysis' in response
    assert 'description' in response
    print("✓ Image analysis test passed")


def test_code_generation():
    """Test code generation."""
    print("Testing code generation...")
    model = RealAI()
    response = model.generate_code(
        prompt="Sort a list",
        language="python"
    )
    assert 'code' in response
    assert 'language' in response
    print("✓ Code generation test passed")


def test_embeddings():
    """Test embeddings creation."""
    print("Testing embeddings...")
    client = RealAIClient()
    response = client.embeddings.create(
        input_text="Test text"
    )
    assert 'data' in response
    assert len(response['data']) > 0
    assert 'embedding' in response['data'][0]
    assert len(response['data'][0]['embedding']) == 1536
    print("✓ Embeddings test passed")


def test_audio_transcription():
    """Test audio transcription."""
    print("Testing audio transcription...")
    client = RealAIClient()
    response = client.audio.transcribe(
        audio_file="test.mp3"
    )
    assert 'text' in response
    assert 'language' in response
    print("✓ Audio transcription test passed")


def test_audio_generation():
    """Test audio generation."""
    print("Testing audio generation...")
    client = RealAIClient()
    response = client.audio.generate(
        text="Hello"
    )
    assert 'audio_url' in response
    print("✓ Audio generation test passed")


def test_translation():
    """Test translation."""
    print("Testing translation...")
    model = RealAI()
    response = model.translate(
        text="Hello",
        target_language="es"
    )
    assert 'translated_text' in response
    assert 'target_language' in response
    print("✓ Translation test passed")


def test_model_capabilities():
    """Test getting model capabilities."""
    print("Testing model capabilities...")
    model = RealAI()
    capabilities = model.get_capabilities()
    assert len(capabilities) >= 8
    assert 'text_generation' in capabilities
    assert 'image_generation' in capabilities
    assert 'code_generation' in capabilities
    print("✓ Model capabilities test passed")


def test_model_info():
    """Test getting model info."""
    print("Testing model info...")
    model = RealAI()
    info = model.get_model_info()
    assert 'name' in info
    assert 'version' in info
    assert 'capabilities' in info
    print("✓ Model info test passed")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Running RealAI Tests")
    print("="*60 + "\n")
    
    tests = [
        test_model_initialization,
        test_client_initialization,
        test_chat_completion,
        test_text_completion,
        test_image_generation,
        test_image_analysis,
        test_code_generation,
        test_embeddings,
        test_audio_transcription,
        test_audio_generation,
        test_translation,
        test_model_capabilities,
        test_model_info
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n✓ All tests passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
