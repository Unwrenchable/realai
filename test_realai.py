"""
Tests for RealAI model.

Run with: python test_realai.py
"""

import sys
from realai import RealAI, RealAIClient, ModelCapability, PROVIDER_CONFIGS, PROVIDER_ENV_VARS, _detect_provider


def test_model_initialization():
    """Test model initialization."""
    print("Testing model initialization...")
    model = RealAI()
    assert model.model_name == "realai-2.0"
    assert model.version == "2.0.0"
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
    assert hasattr(client, 'videos')
    assert hasattr(client, 'embeddings')
    assert hasattr(client, 'audio')
    # New capabilities
    assert hasattr(client, 'web')
    assert hasattr(client, 'tasks')
    assert hasattr(client, 'voice')
    assert hasattr(client, 'business')
    assert hasattr(client, 'therapy')
    assert hasattr(client, 'web3')
    assert hasattr(client, 'plugins')
    assert hasattr(client, 'personas')
    # Next-generation capability sub-clients
    assert hasattr(client, 'reasoning')
    assert hasattr(client, 'synthesis')
    assert hasattr(client, 'reflection')
    assert hasattr(client, 'agents')
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


def test_video_generation():
    """Test video generation."""
    print("Testing video generation...")
    client = RealAIClient()
    response = client.videos.generate(
        prompt="A drone flyover of a futuristic city",
        duration=4,
        n=1
    )
    assert 'created' in response
    assert 'data' in response
    assert len(response['data']) > 0
    assert 'url' in response['data'][0]
    print("✓ Video generation test passed")


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
    assert len(response['data'][0]['embedding']) > 0
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
    assert len(capabilities) >= 17
    assert 'text_generation' in capabilities
    assert 'image_generation' in capabilities
    assert 'video_generation' in capabilities
    assert 'code_generation' in capabilities
    assert 'web_research' in capabilities
    assert 'task_automation' in capabilities
    assert 'voice_interaction' in capabilities
    assert 'business_planning' in capabilities
    assert 'therapy_counseling' in capabilities
    assert 'web3_integration' in capabilities
    # Next-generation capabilities
    assert 'self_reflection' in capabilities
    assert 'chain_of_thought' in capabilities
    assert 'knowledge_synthesis' in capabilities
    assert 'multi_agent' in capabilities
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


def test_web_research():
    """Test web research capability."""
    print("Testing web research...")
    model = RealAI()
    response = model.web_research(
        query="AI developments",
        depth="standard"
    )
    assert 'query' in response
    assert 'findings' in response
    assert 'sources' in response
    assert 'citations' in response
    assert 'freshness' in response
    print("✓ Web research test passed")


def test_capability_catalog_and_provider_capabilities():
    """Test unified capability catalog and provider capability map access."""
    print("Testing capability catalog/provider capabilities...")
    model = RealAI(api_key="sk-testkey123")
    catalog = model.get_capability_catalog()
    assert 'capabilities' in catalog
    assert 'domains' in catalog
    assert catalog['count'] >= 17
    provider_caps = model.get_provider_capabilities()
    assert provider_caps['provider'] == 'openai'
    assert 'supported_capabilities' in provider_caps
    assert 'unsupported_capabilities' in provider_caps
    print("✓ Capability catalog/provider capabilities test passed")


def test_persona_profiles_and_chat_metadata():
    """Test persona switching and canonical response metadata."""
    print("Testing personas and response metadata...")
    model = RealAI()
    personas = model.get_personas()
    assert 'balanced' in personas
    model.set_persona("analyst")
    response = model.chat_completion(messages=[{"role": "user", "content": "Hello"}])
    assert 'realai_meta' in response
    assert response['realai_meta']['capability'] == 'text_generation'
    assert response['realai_meta']['persona'] == 'analyst'
    print("✓ Personas and response metadata test passed")


def test_task_automation():
    """Test task automation capability."""
    print("Testing task automation...")
    client = RealAIClient()
    response = client.tasks.order_groceries(
        items=["milk", "eggs"],
        execute=False
    )
    assert 'task_type' in response
    assert 'status' in response
    assert response['task_type'] == 'groceries'
    print("✓ Task automation test passed")


def test_voice_interaction():
    """Test voice interaction capability."""
    print("Testing voice interaction...")
    client = RealAIClient()
    response = client.voice.conversation(
        message="Hello"
    )
    assert 'conversation_id' in response
    assert 'response_text' in response
    print("✓ Voice interaction test passed")


def test_business_planning():
    """Test business planning capability."""
    print("Testing business planning...")
    client = RealAIClient()
    response = client.business.build(
        business_type="tech startup"
    )
    assert 'business_type' in response
    assert 'business_plan' in response
    assert 'action_items' in response
    print("✓ Business planning test passed")


def test_therapy_counseling():
    """Test therapy/counseling capability."""
    print("Testing therapy counseling...")
    model = RealAI()
    response = model.therapy_counseling(
        session_type="support",
        message="I need support"
    )
    assert 'session_id' in response
    assert 'response' in response
    assert 'recommendations' in response
    print("✓ Therapy counseling test passed")


def test_web3_integration():
    """Test Web3 integration capability."""
    print("Testing Web3 integration...")
    client = RealAIClient()
    response = client.web3.smart_contract(
        blockchain="ethereum"
    )
    assert 'operation' in response
    assert 'blockchain' in response
    assert 'status' in response
    print("✓ Web3 integration test passed")


def test_code_execution():
    """Test code execution capability."""
    print("Testing code execution...")
    model = RealAI()
    response = model.execute_code(
        code="print('Hello')",
        language="python"
    )
    assert 'execution_status' in response
    assert 'output' in response
    assert response['execution_status'] == 'completed', f"Expected 'completed', got '{response['execution_status']}'"
    assert response['output'].strip() == 'Hello', f"Expected 'Hello', got '{response['output'].strip()}'"
    print("✓ Code execution test passed")


def test_plugin_system():
    """Test plugin system capability."""
    print("Testing plugin system...")
    client = RealAIClient()
    response = client.plugins.load(
        plugin_name="test_plugin"
    )
    assert 'plugin_name' in response
    assert 'status' in response
    print("✓ Plugin system test passed")


def test_local_plugin_loading():
    """Test loading a local plugin from the `plugins` package."""
    print("Testing local plugin loading...")
    client = RealAIClient()
    response = client.plugins.load(plugin_name="sample_plugin")
    assert response.get('plugin_name') == 'sample_plugin'
    # Plugin should be recorded in model registry and attach a callable
    assert 'sample_plugin' in client.model.plugins_registry
    assert hasattr(client.model, 'sample_action')
    result = client.model.sample_action({'x': 1})
    assert isinstance(result, dict)
    assert result.get('ok') is True
    print("✓ Local plugin loading test passed")


def test_load_all_plugins():
    """Test discovery and loading of all plugins in `plugins` package."""
    print("Testing load_all_plugins...")
    client = RealAIClient()
    loaded = client.model.load_all_plugins()
    # At least our sample_plugin should be discovered and loaded
    assert 'sample_plugin' in loaded
    print("✓ load_all_plugins test passed")


def test_memory_learning():
    """Test memory/learning capability."""
    print("Testing memory learning...")
    model = RealAI()
    response = model.learn_from_interaction(
        interaction_data={"test": "data"},
        save=True
    )
    assert 'learned' in response
    assert 'insights' in response
    print("✓ Memory learning test passed")


def test_provider_detection():
    """Test automatic provider detection from API key prefixes."""
    print("Testing provider detection...")
    assert _detect_provider("sk-abc123", None) == "openai"
    assert _detect_provider("sk-proj-abc123", None) == "openai"
    assert _detect_provider("sk-ant-abc123", None) == "anthropic"
    assert _detect_provider("xai-abc123", None) == "grok"
    assert _detect_provider("AIzaXYZ", None) == "gemini"
    assert _detect_provider("sk-or-v1-abc123", None) == "openrouter"
    assert _detect_provider("pplx-abc123", None) == "perplexity"
    assert _detect_provider(None, None) is None
    assert _detect_provider(None, "openai") == "openai"
    assert _detect_provider("sk-abc123", "anthropic") == "anthropic"  # explicit wins
    print("✓ Provider detection test passed")


def test_provider_configs():
    """Test that PROVIDER_CONFIGS contains all expected providers."""
    print("Testing provider configs...")
    for name in ("openai", "anthropic", "grok", "gemini",
                 "openrouter", "mistral", "together", "deepseek", "perplexity"):
        assert name in PROVIDER_CONFIGS, f"Missing provider: {name}"
        cfg = PROVIDER_CONFIGS[name]
        assert "base_url" in cfg
        assert "default_model" in cfg
        assert "api_format" in cfg
    print("✓ Provider configs test passed")


def test_provider_env_vars():
    """Test that PROVIDER_ENV_VARS covers all providers in PROVIDER_CONFIGS."""
    print("Testing provider env vars...")
    for name in ("openai", "anthropic", "grok", "gemini",
                 "openrouter", "mistral", "together", "deepseek", "perplexity"):
        assert name in PROVIDER_ENV_VARS, f"Missing env var entry for: {name}"
        env_var = PROVIDER_ENV_VARS[name]
        assert env_var.startswith("REALAI_"), f"Unexpected env var name: {env_var}"
        assert env_var.endswith("_API_KEY"), f"Unexpected env var name: {env_var}"
    print("✓ Provider env vars test passed")


def test_realai_provider_init():
    """Test RealAI initialises provider routing fields correctly."""
    print("Testing RealAI provider init...")

    # No key → no provider
    m = RealAI()
    assert m.provider is None
    assert m._provider_model == "realai-2.0"

    # OpenAI key auto-detection
    m = RealAI(api_key="sk-testkey123")
    assert m.provider == "openai"
    assert m.base_url == PROVIDER_CONFIGS["openai"]["base_url"]
    assert m._provider_model == PROVIDER_CONFIGS["openai"]["default_model"]

    # Explicit model name preserved when not default
    m = RealAI(api_key="sk-testkey123", model_name="gpt-4o")
    assert m._provider_model == "gpt-4o"

    # Anthropic key auto-detection
    m = RealAI(api_key="sk-ant-testkey123")
    assert m.provider == "anthropic"
    assert m._api_format == "anthropic"

    # OpenRouter key auto-detection
    m = RealAI(api_key="sk-or-v1-testkey123")
    assert m.provider == "openrouter"
    assert m.base_url == PROVIDER_CONFIGS["openrouter"]["base_url"]

    # Perplexity key auto-detection
    m = RealAI(api_key="pplx-testkey123")
    assert m.provider == "perplexity"
    assert m.base_url == PROVIDER_CONFIGS["perplexity"]["base_url"]

    # Explicit provider override
    m = RealAI(api_key="sk-testkey123", provider="grok")
    assert m.provider == "grok"
    assert m.base_url == PROVIDER_CONFIGS["grok"]["base_url"]

    # Mistral via explicit provider (no unique key prefix)
    m = RealAI(api_key="somekey", provider="mistral")
    assert m.provider == "mistral"
    assert m.base_url == PROVIDER_CONFIGS["mistral"]["base_url"]

    # Together AI via explicit provider
    m = RealAI(api_key="somekey", provider="together")
    assert m.provider == "together"
    assert m.base_url == PROVIDER_CONFIGS["together"]["base_url"]

    # DeepSeek via explicit provider
    m = RealAI(api_key="somekey", provider="deepseek")
    assert m.provider == "deepseek"
    assert m.base_url == PROVIDER_CONFIGS["deepseek"]["base_url"]

    # Custom base_url override
    m = RealAI(api_key="sk-testkey123", base_url="http://localhost:11434/v1")
    assert m.base_url == "http://localhost:11434/v1"

    print("✓ RealAI provider init test passed")


def test_client_provider_params():
    """Test RealAIClient forwards provider/base_url to its underlying model."""
    print("Testing RealAIClient provider params...")
    client = RealAIClient(api_key="sk-ant-testkey", provider="anthropic",
                          base_url="https://custom.example.com")
    assert client.model.provider == "anthropic"
    assert client.model.base_url == "https://custom.example.com"
    assert client.model._api_format == "anthropic"
    print("✓ RealAIClient provider params test passed")


def test_chat_fallback_without_key():
    """Placeholder response is returned when no API key is configured."""
    print("Testing chat fallback without key...")
    client = RealAIClient()
    response = client.chat.create(
        messages=[{"role": "user", "content": "Hello"}]
    )
    assert 'id' in response
    assert 'choices' in response
    assert response['choices'][0]['message']['role'] == 'assistant'
    print("✓ Chat fallback without key test passed")


def test_chain_of_thought():
    """Test chain-of-thought reasoning capability."""
    print("Testing chain-of-thought reasoning...")
    model = RealAI()
    response = model.chain_of_thought(
        problem="If all cats are mammals and all mammals breathe air, do cats breathe air?",
        domain="logic"
    )
    assert response['status'] == 'success'
    assert 'steps' in response
    assert isinstance(response['steps'], list)
    assert len(response['steps']) > 0
    assert 'answer' in response
    assert 'confidence' in response
    print("✓ Chain-of-thought reasoning test passed")


def test_chain_of_thought_client():
    """Test chain-of-thought via RealAIClient.reasoning."""
    print("Testing chain-of-thought via client...")
    client = RealAIClient()
    response = client.reasoning.chain(
        problem="What is 2 + 2?",
        domain="math"
    )
    assert response['status'] == 'success'
    assert 'steps' in response
    assert 'answer' in response
    print("✓ Chain-of-thought client test passed")


def test_synthesize_knowledge():
    """Test knowledge synthesis capability."""
    print("Testing knowledge synthesis...")
    model = RealAI()
    response = model.synthesize_knowledge(
        topics=["artificial intelligence", "neuroscience"],
        output_format="bullets"
    )
    assert response['status'] == 'success'
    assert 'topics' in response
    assert 'per_topic' in response
    assert 'synthesis' in response
    assert 'connections' in response
    assert isinstance(response['connections'], list)
    print("✓ Knowledge synthesis test passed")


def test_synthesize_knowledge_client():
    """Test knowledge synthesis via RealAIClient.synthesis."""
    print("Testing knowledge synthesis via client...")
    client = RealAIClient()
    response = client.synthesis.combine(
        topics=["climate change", "renewable energy", "economics"]
    )
    assert response['status'] == 'success'
    assert 'synthesis' in response
    assert len(response['topics']) == 3
    print("✓ Knowledge synthesis client test passed")


def test_self_reflect():
    """Test self-reflection capability."""
    print("Testing self-reflection...")
    model = RealAI()
    history = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]
    response = model.self_reflect(interaction_history=history, focus="accuracy")
    assert response['status'] == 'success'
    assert 'strengths' in response
    assert isinstance(response['strengths'], list)
    assert 'weaknesses' in response
    assert isinstance(response['weaknesses'], list)
    assert 'improvements' in response
    assert isinstance(response['improvements'], list)
    assert 'score' in response
    print("✓ Self-reflection test passed")


def test_self_reflect_client():
    """Test self-reflection via RealAIClient.reflection."""
    print("Testing self-reflection via client...")
    client = RealAIClient()
    response = client.reflection.improve(focus="efficiency")
    assert response['status'] == 'success'
    assert 'improvements' in response
    print("✓ Self-reflection client test passed")


def test_orchestrate_agents():
    """Test multi-agent orchestration capability."""
    print("Testing multi-agent orchestration...")
    model = RealAI()
    response = model.orchestrate_agents(
        task="Evaluate the pros and cons of electric vehicles",
        agent_roles=["researcher", "analyst", "critic"]
    )
    assert response['status'] == 'success'
    assert 'task' in response
    assert 'agents_used' in response
    assert 'agent_results' in response
    assert isinstance(response['agent_results'], dict)
    assert 'final_output' in response
    assert 'execution_plan' in response
    assert 'verification' in response
    assert len(response['agent_results']) == 3
    print("✓ Multi-agent orchestration test passed")


def test_orchestrate_agents_client():
    """Test multi-agent orchestration via RealAIClient.agents."""
    print("Testing multi-agent orchestration via client...")
    client = RealAIClient()
    response = client.agents.run(
        task="Summarize the key trends in renewable energy"
    )
    assert response['status'] == 'success'
    assert 'final_output' in response
    assert 'agents_used' in response
    print("✓ Multi-agent orchestration client test passed")


def test_generate_speech():
    """Test generate_speech convenience method."""
    print("Testing generate_speech...")
    model = RealAI()
    response = model.generate_speech("Hello, world!")
    assert 'url' in response
    assert 'spoken' in response
    assert 'audio_url' in response
    print("✓ generate_speech test passed")


def test_new_capabilities_in_model():
    """Test that all next-gen capabilities appear in model.get_capabilities()."""
    print("Testing next-gen capabilities in model...")
    model = RealAI()
    caps = model.get_capabilities()
    assert 'self_reflection' in caps
    assert 'chain_of_thought' in caps
    assert 'knowledge_synthesis' in caps
    assert 'multi_agent' in caps
    assert 'video_generation' in caps
    print("✓ Next-gen capabilities in model test passed")


def test_new_client_attributes():
    """Test that RealAIClient exposes the new sub-client attributes."""
    print("Testing new client attributes...")
    client = RealAIClient()
    assert hasattr(client, 'reasoning')
    assert hasattr(client, 'synthesis')
    assert hasattr(client, 'reflection')
    assert hasattr(client, 'agents')
    assert hasattr(client, 'videos')
    print("✓ New client attributes test passed")


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
        test_video_generation,
        test_code_generation,
        test_embeddings,
        test_audio_transcription,
        test_audio_generation,
        test_translation,
        test_web_research,
        test_capability_catalog_and_provider_capabilities,
        test_persona_profiles_and_chat_metadata,
        test_task_automation,
        test_voice_interaction,
        test_business_planning,
        test_therapy_counseling,
        test_web3_integration,
        test_code_execution,
        test_plugin_system,
        test_memory_learning,
        test_model_capabilities,
        test_model_info,
        # Provider routing tests
        test_provider_detection,
        test_provider_configs,
        test_provider_env_vars,
        test_realai_provider_init,
        test_client_provider_params,
        test_chat_fallback_without_key,
        # Plugin real-world tests
        test_local_plugin_loading,
        test_load_all_plugins,
        # Next-generation capability tests
        test_chain_of_thought,
        test_chain_of_thought_client,
        test_synthesize_knowledge,
        test_synthesize_knowledge_client,
        test_self_reflect,
        test_self_reflect_client,
        test_orchestrate_agents,
        test_orchestrate_agents_client,
        test_generate_speech,
        test_new_capabilities_in_model,
        test_new_client_attributes,
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
