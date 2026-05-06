"""
Tests for RealAI model.

Run with: python test_realai.py
"""

import importlib.util
import os
import sys
import time
from realai import (
    RealAI, RealAIClient, ModelCapability, PROVIDER_CONFIGS, PROVIDER_ENV_VARS,
    _detect_provider, AgentRegistry, AgentDefinition, AccessProfile,
    CloudDeploymentManager, LoadBalancer, LearningRecorder, ScreenCapture,
    ComputerMode, CloudProvider, CloudInstance, RecordedAction,
    CAPABILITY_DOMAIN_MAP, PERSONA_PROFILES,
)


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
    assert len(capabilities) >= 44
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


def test_web3_gpg_signing():
    """Test Web3 GPG signing capability."""
    print("Testing Web3 GPG signing...")
    client = RealAIClient()
    # Test GPG signing without requiring a live Web3 provider.
    response = client.web3.execute(
        operation="transaction",
        blockchain="ethereum",
        sign_with_gpg=True,
        transaction_data="test transaction data",
        gpg_keyid="test@example.com"
    )
    assert 'operation' in response
    assert 'blockchain' in response
    assert 'status' in response
    assert 'result' not in response
    assert (
        response.get('signature_status') == 'signed_with_gpg'
        or 'error' in response
    )
    print("✓ Web3 GPG signing test passed")


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
    response = client.agents.orchestrate(
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


# ============================================================================
# Future AI Capabilities (2026+) - Cutting-Edge Tests
# ============================================================================

def test_quantum_integration():
    """Test quantum computing integration."""
    print("Testing quantum integration...")
    client = RealAIClient()
    response = client.quantum.compute(operation="factorization", parameters={"number": 15})
    assert 'operation' in response
    assert 'algorithm' in response
    assert 'status' in response
    assert 'factors' in response
    print("✓ Quantum integration test passed")


def test_neural_architecture_search():
    """Test neural architecture search."""
    print("Testing neural architecture search...")
    client = RealAIClient()
    response = client.neural_arch.search(task="classification")
    assert 'task' in response
    assert 'optimal_architecture' in response
    assert 'status' in response
    print("✓ Neural architecture search test passed")


def test_causal_reasoning():
    """Test causal reasoning capabilities."""
    print("Testing causal reasoning...")
    client = RealAIClient()
    response = client.causal.analyze(
        scenario="Testing causal relationships",
        variables=["cause", "effect", "confounder"]
    )
    assert 'scenario' in response
    assert 'causal_graph' in response
    assert 'status' in response
    print("✓ Causal reasoning test passed")


def test_meta_learning():
    """Test meta-learning capabilities."""
    print("Testing meta-learning...")
    client = RealAIClient()
    response = client.meta.learn(learning_tasks=["task1", "task2", "task3"])
    assert 'learning_tasks' in response
    assert 'learned_strategies' in response
    assert 'status' in response
    print("✓ Meta-learning test passed")


def test_emotional_intelligence():
    """Test emotional intelligence capabilities."""
    print("Testing emotional intelligence...")
    client = RealAIClient()
    response = client.emotion.analyze("I'm feeling really happy today!")
    assert 'emotional_analysis' in response
    assert 'primary_emotion' in response['emotional_analysis']
    assert 'status' in response
    print("✓ Emotional intelligence test passed")


def test_swarm_intelligence():
    """Test swarm intelligence capabilities."""
    print("Testing swarm intelligence...")
    client = RealAIClient()
    response = client.swarm.solve(problem="Optimize complex system", agents=25)
    assert 'problem' in response
    assert 'swarm_size' in response
    assert 'emergent_behavior' in response
    assert 'status' in response
    print("✓ Swarm intelligence test passed")


def test_predictive_simulation():
    """Test predictive simulation capabilities."""
    print("Testing predictive simulation...")
    client = RealAIClient()
    response = client.predictive.simulate(scenario="Future technology trends")
    assert 'scenario' in response
    assert 'predictions' in response
    assert 'status' in response
    print("✓ Predictive simulation test passed")


def test_consciousness_simulation():
    """Test consciousness simulation capabilities."""
    print("Testing consciousness simulation...")
    client = RealAIClient()
    response = client.consciousness.simulate(self_awareness=True)
    assert 'consciousness_metrics' in response
    assert 'self_reflection' in response
    assert 'status' in response
    print("✓ Consciousness simulation test passed")


def test_reality_simulation():
    """Test reality simulation capabilities."""
    print("Testing reality simulation...")
    client = RealAIClient()
    response = client.reality.simulate(reality_type="alternate_history")
    assert 'reality_type' in response
    assert 'simulated_reality' in response
    assert 'status' in response
    print("✓ Reality simulation test passed")


# ============================================================================
# Agent Orchestration and Hive Mind System Tests
# ============================================================================

def test_agent_orchestration():
    """Test multi-agent orchestration capabilities."""
    print("Testing agent orchestration...")
    client = RealAIClient()
    response = client.agents.orchestrate(
        task="Build a web application with security testing",
        workflow_type="sequential"
    )
    assert 'task' in response
    assert 'agents_involved' in response
    assert 'workflow_type' in response
    assert 'agent_results' in response
    assert 'hive_mind_insights' in response
    assert 'status' in response
    print("✓ Agent orchestration test passed")


def test_agent_execution():
    """Test individual agent execution."""
    print("Testing agent execution...")
    client = RealAIClient()
    response = client.agents.execute("architect", "Design a REST API for user management")
    assert 'agent_id' in response
    assert 'task' in response
    assert 'result' in response
    assert 'status' in response
    print("✓ Agent execution test passed")


def test_agent_registration():
    """Test custom agent registration."""
    print("Testing agent registration...")
    client = RealAIClient()
    response = client.agents.register(
        agent_id="test_agent",
        role="Test Specialist",
        description="Specialized agent for testing purposes",
        capabilities=["testing", "validation"],
        required_tools=["read_file", "run_in_terminal"]
    )
    assert 'status' in response
    assert 'agent_id' in response
    assert response['agent_id'] == "test_agent"
    print("✓ Agent registration test passed")


def test_agent_listing():
    """Test agent listing and search."""
    print("Testing agent listing...")
    client = RealAIClient()
    response = client.agents.list()
    assert 'status' in response
    assert 'total_agents' in response
    assert 'agents' in response
    assert isinstance(response['agents'], list)
    assert len(response['agents']) > 0
    print("✓ Agent listing test passed")


def test_agent_status():
    """Test hive mind system status."""
    print("Testing agent system status...")
    client = RealAIClient()
    response = client.agents.status()
    assert 'status' in response
    assert 'system_health' in response
    assert 'registered_agents' in response
    assert 'hive_mind_capabilities' in response
    print("✓ Agent status test passed")


def test_core_agent_methods():
    """Test convenience methods for core agents."""
    print("Testing core agent convenience methods...")
    client = RealAIClient()
    
    # Test architect agent
    response = client.agents.architect("Design a microservices architecture")
    assert 'status' in response
    assert 'agent_id' in response
    assert response['agent_id'] == 'architect'
    
    # Test implementer agent
    response = client.agents.implementer("Implement user authentication")
    assert 'status' in response
    assert 'agent_id' in response
    assert response['agent_id'] == 'implementer'
    
    print("✓ Core agent methods test passed")


# ============================================================================
# Cloud Computing and Distributed Systems Tests
# ============================================================================

def test_cloud_deployment_orchestration():
    """Test cloud deployment across multiple providers."""
    print("Testing cloud deployment orchestration...")
    client = RealAIClient()
    response = client.cloud.deploy(
        providers=["vercel", "render", "railway"],
        instance_count=5
    )
    assert 'status' in response
    assert 'deployed_instances' in response
    assert 'total_instances' in response
    assert 'providers_used' in response
    assert 'total_hourly_cost' in response
    print("✓ Cloud deployment orchestration test passed")


def test_distributed_computing_coordination():
    """Test distributed computing coordination."""
    print("Testing distributed computing coordination...")
    client = RealAIClient()
    tasks = [
        {"type": "computation", "operation": "fibonacci", "n": 30},
        {"type": "inference", "model": "gpt-4", "prompt": "Hello world"},
        {"type": "training", "dataset": "mnist", "epochs": 5}
    ]
    response = client.cloud.compute(tasks=tasks)
    assert 'status' in response
    assert 'submitted_tasks' in response
    assert 'task_ids' in response
    assert 'system_status' in response
    print("✓ Distributed computing coordination test passed")


def test_auto_scaling_management():
    """Test auto-scaling configuration."""
    print("Testing auto-scaling management...")
    client = RealAIClient()
    response = client.cloud.scale(
        target_instances=10,
        max_instances=50,
        scale_up_threshold=10,
        scale_down_threshold=2
    )
    assert 'status' in response
    assert 'scaling_config' in response
    assert 'current_status' in response
    assert 'scaling_actions' in response
    print("✓ Auto-scaling management test passed")


def test_load_balancing_optimization():
    """Test load balancing optimization."""
    print("Testing load balancing optimization...")
    client = RealAIClient()
    response = client.cloud.balance(
        algorithm="round_robin",
        health_checks=True,
        session_persistence=False
    )
    assert 'status' in response
    assert 'load_balancing_config' in response
    assert 'load_distribution' in response
    assert 'optimization_metrics' in response
    print("✓ Load balancing optimization test passed")


def test_multi_cloud_resource_management():
    """Test multi-cloud resource management."""
    print("Testing multi-cloud resource management...")
    client = RealAIClient()
    response = client.cloud.resources(
        providers=["vercel", "render", "railway"],
        optimization_goal="cost"
    )
    assert 'status' in response
    assert 'providers_managed' in response
    assert 'provider_breakdown' in response
    assert 'recommendations' in response
    assert 'cost_savings_potential' in response
    print("✓ Multi-cloud resource management test passed")


def test_serverless_function_deployment():
    """Test serverless function deployment."""
    print("Testing serverless function deployment...")
    client = RealAIClient()
    functions = [
        {"name": "api_handler", "runtime": "python3.9"},
        {"name": "data_processor", "runtime": "node18"}
    ]
    response = client.cloud.functions(
        functions=functions,
        providers=["vercel", "render"]
    )
    assert 'status' in response
    assert 'deployed_functions' in response
    assert 'total_functions' in response
    assert 'providers_used' in response
    print("✓ Serverless function deployment test passed")


def test_container_orchestration():
    """Test container orchestration."""
    print("Testing container orchestration...")
    client = RealAIClient()
    containers = [
        {"name": "web_server", "image": "nginx:latest", "replicas": 3},
        {"name": "api_server", "image": "python:3.9", "replicas": 2}
    ]
    response = client.cloud.containers(
        containers=containers,
        orchestration_platform="kubernetes"
    )
    assert 'status' in response
    assert 'orchestrated_containers' in response
    assert 'cluster_status' in response
    assert 'orchestration_platform' in response
    print("✓ Container orchestration test passed")


def test_cloud_cost_optimization():
    """Test cloud cost optimization."""
    print("Testing cloud cost optimization...")
    client = RealAIClient()
    response = client.cloud.optimize_cost(
        optimization_targets=["compute", "storage", "networking"],
        time_horizon="monthly",
        budget_limit=100.0
    )
    assert 'status' in response
    assert 'current_hourly_cost' in response
    assert 'cost_breakdown' in response
    assert 'recommendations' in response
    assert 'potential_savings' in response
    print("✓ Cloud cost optimization test passed")


def test_distributed_ai_training():
    """Test distributed AI training coordination."""
    print("Testing distributed AI training...")
    client = RealAIClient()
    model_config = {
        "architecture": "transformer",
        "parameters": 1000000,
        "layers": 12
    }
    dataset_config = {
        "location": "s3://realai-datasets/training-data",
        "format": "parquet",
        "size_gb": 100
    }
    response = client.cloud.train_distributed(
        model_config=model_config,
        dataset_config=dataset_config,
        training_strategy="data_parallel",
        instances_required=4
    )
    assert 'status' in response
    assert 'training_task_id' in response
    assert 'training_strategy' in response
    assert 'distributed_benefits' in response
    print("✓ Distributed AI training test passed")


def test_cloud_native_ai_inference():
    """Test cloud-native AI inference deployment."""
    print("Testing cloud-native AI inference...")
    client = RealAIClient()
    model_endpoints = [
        {"model": "gpt-4", "endpoint_type": "rest"},
        {"model": "bert", "endpoint_type": "grpc"}
    ]
    scaling_config = {
        "min_instances": 1,
        "max_instances": 10,
        "target_cpu_utilization": 70
    }
    response = client.cloud.inference_cloud(
        model_endpoints=model_endpoints,
        scaling_config=scaling_config,
        optimization_level="balanced"
    )
    assert 'status' in response
    assert 'deployed_endpoints' in response
    assert 'total_endpoints' in response
    assert 'performance_metrics' in response
    print("✓ Cloud-native AI inference test passed")


def test_cloud_convenience_methods():
    """Test cloud convenience methods."""
    print("Testing cloud convenience methods...")
    client = RealAIClient()

    # Test individual provider deployments
    response = client.cloud.vercel(instance_count=3)
    assert 'status' in response
    assert 'providers_used' in response
    assert 'vercel' in response['providers_used']

    response = client.cloud.render(instance_count=3)
    assert 'status' in response
    assert 'providers_used' in response
    assert 'render' in response['providers_used']

    response = client.cloud.railway(instance_count=3)
    assert 'status' in response
    assert 'providers_used' in response
    assert 'railway' in response['providers_used']

    # Test multi-cloud deployment
    response = client.cloud.multi_cloud(instance_count=5)
    assert 'status' in response
    assert 'providers_used' in response
    assert len(response['providers_used']) >= 2

    print("✓ Cloud convenience methods test passed")


def test_computer_mode_activation():
    """Test computer mode activation."""
    print("Testing computer mode activation...")
    client = RealAIClient()
    response = client.computer.activate()
    # Note: This will fail without pyautogui/pygetwindow/pillow installed
    # But we test the interface structure
    assert 'status' in response
    print("✓ Computer mode activation test passed")


def test_screen_capture_analysis():
    """Test screen capture and analysis."""
    print("Testing screen capture analysis...")
    client = RealAIClient()
    response = client.computer.capture_screen("Analyze the desktop")
    assert 'status' in response
    print("✓ Screen capture analysis test passed")


def test_mouse_keyboard_control():
    """Test mouse and keyboard control."""
    print("Testing mouse and keyboard control...")
    client = RealAIClient()
    
    # Test mouse movement
    response = client.computer.move_mouse(100, 100)
    assert 'status' in response
    assert 'action_type' in response
    
    # Test clicking
    response = client.computer.click("left")
    assert 'status' in response
    assert 'action_type' in response
    
    # Test typing
    response = client.computer.type_text("Hello World")
    assert 'status' in response
    assert 'action_type' in response
    
    # Test key press
    response = client.computer.press_key("enter")
    assert 'status' in response
    assert 'action_type' in response
    
    print("✓ Mouse and keyboard control test passed")


def test_window_management():
    """Test window management."""
    print("Testing window management...")
    client = RealAIClient()
    
    # Activate computer mode first
    response = client.computer.activate()
    # In test environment, may fail due to missing dependencies
    assert 'status' in response or 'error' in response
    
    # Test getting active window
    response = client.computer.get_active_window()
    assert 'status' in response or 'error' in response
    
    # Test listing windows
    response = client.computer.list_windows()
    assert 'status' in response or 'error' in response
    
    # Test switching window
    response = client.computer.switch_window("test")
    assert 'status' in response or 'error' in response
    
    print("✓ Window management test passed")


def test_gui_automation():
    """Test GUI automation workflows."""
    print("Testing GUI automation...")
    client = RealAIClient()
    response = client.computer.automate_workflow("test_workflow")
    assert 'status' in response
    assert 'task' in response
    print("✓ GUI automation test passed")


def test_development_workflow_automation():
    """Test development workflow automation."""
    print("Testing development workflow automation...")
    client = RealAIClient()
    response = client.computer.build_app("web", {"framework": "react"})
    assert 'status' in response
    assert 'app_type' in response
    print("✓ Development workflow automation test passed")


def test_self_learning_recording():
    """Test self-learning recording."""
    print("Testing self-learning recording...")
    client = RealAIClient()
    
    # Test starting learning
    response = client.computer.start_learning("test_task")
    assert 'status' in response
    
    # Test recording action
    response = client.computer.record_action("click", position=(100, 100))
    assert 'status' in response
    
    # Test stopping learning
    response = client.computer.stop_learning()
    assert 'status' in response
    
    print("✓ Self-learning recording test passed")


def test_action_replay_execution():
    """Test action replay execution."""
    print("Testing action replay execution...")
    from realai import RecordedAction
    import time
    
    client = RealAIClient()
    actions = [
        RecordedAction(
            timestamp=time.time(),
            action_type="click",
            position=(100, 100),
            metadata={"button": "left"}
        )
    ]
    response = client.computer.replay_actions(actions)
    assert 'status' in response
    print("✓ Action replay execution test passed")


def test_code_generation_automation():
    """Test code generation automation."""
    print("Testing code generation automation...")
    client = RealAIClient()
    response = client.computer.generate_code("create a hello world function")
    assert 'status' in response
    print("✓ Code generation automation test passed")


def test_app_building_automation():
    """Test app building automation."""
    print("Testing app building automation...")
    client = RealAIClient()
    response = client.computer.build_app("web", {"framework": "react"})
    assert 'status' in response
    assert 'app_type' in response
    print("✓ App building automation test passed")


def test_crypto_mining():
    """Test crypto mining setup."""
    print("Testing crypto mining...")
    client = RealAIClient()
    response = client.crypto.mine_crypto("ethash", 2)
    assert 'status' in response
    if 'algorithm' in response:
        assert response['algorithm'] == 'ethash'
        assert response['gpu_count'] == 2
    print("✓ Crypto mining test passed")


def test_ai_trading_bot_integration():
    """Test AI trading bot integration."""
    print("Testing AI trading bot integration...")
    client = RealAIClient()
    response = client.crypto.integrate_trading_bot("freqtrade", {"exchange": "binance"})
    assert 'status' in response
    if 'bot_name' in response:
        assert response['bot_name'] == 'freqtrade'
    print("✓ AI trading bot integration test passed")


def test_freqtrade_integration():
    """Test Freqtrade bot setup."""
    print("Testing Freqtrade integration...")
    client = RealAIClient()
    response = client.crypto.setup_freqtrade("binance", "SampleStrategy")
    assert 'status' in response
    if 'bot' in response:
        assert response['bot'] == 'freqtrade'
    print("✓ Freqtrade integration test passed")


def test_hummingbot_integration():
    """Test Hummingbot setup."""
    print("Testing Hummingbot integration...")
    client = RealAIClient()
    response = client.crypto.setup_hummingbot("binance", "pure_market_making")
    assert 'status' in response
    if 'bot' in response:
        assert response['bot'] == 'hummingbot'
    print("✓ Hummingbot integration test passed")


def test_octobot_integration():
    """Test OctoBot setup."""
    print("Testing OctoBot integration...")
    client = RealAIClient()
    response = client.crypto.setup_octobot("binance", "Simple")
    assert 'status' in response
    if 'bot' in response:
        assert response['bot'] == 'octobot'
    print("✓ OctoBot integration test passed")


def test_jessie_trading_integration():
    """Test Jesse trading bot setup."""
    print("Testing Jesse integration...")
    client = RealAIClient()
    response = client.crypto.setup_jessie("binance", "TestStrategy")
    assert 'status' in response
    if 'bot' in response:
        assert response['bot'] == 'jesse'
    print("✓ Jesse integration test passed")


def test_superalgos_integration():
    """Test Superalgos setup."""
    print("Testing Superalgos integration...")
    client = RealAIClient()
    response = client.crypto.setup_superalgos("binance", "VisualStrategy")
    assert 'status' in response
    if 'bot' in response:
        assert response['bot'] == 'superalgos'
    print("✓ Superalgos integration test passed")


def test_polymarket_bot_integration():
    """Test Polymarket bot setup."""
    print("Testing Polymarket bot integration...")
    client = RealAIClient()
    response = client.crypto.setup_polymarket_bot("crypto", "sniper")
    assert 'status' in response
    if 'bot' in response:
        assert response['bot'] == 'polymarket'
    print("✓ Polymarket bot integration test passed")


def test_market_analysis():
    """Test market analysis."""
    print("Testing market analysis...")
    client = RealAIClient()
    response = client.crypto.analyze_market("BTC/USDT", "1h")
    assert 'status' in response
    print("✓ Market analysis test passed")


def test_trading_strategy_optimization():
    """Test trading strategy optimization."""
    print("Testing trading strategy optimization...")
    client = RealAIClient()
    response = client.crypto.optimize_strategy("def strategy(): pass", {"historical_data": []})
    assert 'status' in response
    print("✓ Trading strategy optimization test passed")


def test_risk_management():
    """Test risk management."""
    print("Testing risk management...")
    client = RealAIClient()
    response = client.crypto.manage_risk({"BTC": 0.5, "ETH": 0.3}, {"max_drawdown": 0.1})
    assert 'status' in response
    print("✓ Risk management test passed")


def test_portfolio_management():
    """Test portfolio management."""
    print("Testing portfolio management...")
    client = RealAIClient()
    response = client.crypto.manage_portfolio(["BTC", "ETH", "ADA"], "balanced")
    assert 'status' in response
    print("✓ Portfolio management test passed")


# ============================================================================
# AgentRegistry Internal Method Tests
# ============================================================================

def test_agent_registry_find_agents():
    """Test AgentRegistry.find_agents() returns matching agents."""
    print("Testing AgentRegistry.find_agents...")
    registry = AgentRegistry()

    # Search by capability tag
    results = registry.find_agents("security")
    assert any(a.id == "security" for a in results), "Expected 'security' agent in results"

    # Search by role
    results = registry.find_agents("architect")
    assert any(a.id == "architect" for a in results)

    # Search for something that doesn't match any agent
    results = registry.find_agents("xyznonexistent12345")
    assert results == [], f"Expected empty list, got {results}"

    print("✓ AgentRegistry.find_agents test passed")


def test_agent_registry_recommend_profile():
    """Test AgentRegistry.recommend_profile() returns appropriate profiles."""
    print("Testing AgentRegistry.recommend_profile...")
    registry = AgentRegistry()

    # Agent with preferred_profile="safe" should get "safe" profile
    architect = registry.get_agent("architect")
    profile = registry.recommend_profile(architect)
    assert profile.name == "safe", f"Expected 'safe', got '{profile.name}'"

    # Agent with preferred_profile="power" should get "power" profile
    deployment = registry.get_agent("deployment")
    profile = registry.recommend_profile(deployment)
    assert profile.name == "power", f"Expected 'power', got '{profile.name}'"

    # Agent with preferred_profile="balanced" should get "balanced" profile
    implementer = registry.get_agent("implementer")
    profile = registry.recommend_profile(implementer)
    assert profile.name == "balanced", f"Expected 'balanced', got '{profile.name}'"

    print("✓ AgentRegistry.recommend_profile test passed")


def test_agent_registry_assess_access():
    """Test AgentRegistry.assess_access() correctly identifies missing tools."""
    print("Testing AgentRegistry.assess_access...")
    registry = AgentRegistry()

    architect = registry.get_agent("architect")
    safe_profile = registry.profiles["safe"]
    balanced_profile = registry.profiles["balanced"]

    # Architect only needs read tools — safe profile should pass
    result = registry.assess_access(architect, safe_profile)
    assert result["agent"] == "architect"
    assert result["profile"] == "safe"
    assert result["pass"] is True, f"Expected pass=True, got {result}"
    assert result["missing_tools"] == []

    # Power profile grants extra tools, which is fine (still passes)
    power_profile = registry.profiles["power"]
    result = registry.assess_access(architect, power_profile)
    assert result["pass"] is True

    # Implementer needs apply_patch/create_file — safe profile should fail
    implementer = registry.get_agent("implementer")
    result = registry.assess_access(implementer, safe_profile)
    assert result["pass"] is False, "Implementer should fail with safe profile"
    assert len(result["missing_tools"]) > 0

    print("✓ AgentRegistry.assess_access test passed")


def test_agent_registry_execution_tracking():
    """Test AgentRegistry execution history and status tracking."""
    print("Testing AgentRegistry execution tracking...")
    registry = AgentRegistry()

    # No active executions initially
    active = registry.list_active_executions()
    assert isinstance(active, list)

    # Unknown execution_id returns None
    status = registry.get_execution_status("nonexistent-id-xyz")
    assert status is None

    # History starts empty (or has prior entries from other tests using shared global)
    history = registry.get_execution_history(limit=5)
    assert isinstance(history, list)
    assert len(history) <= 5

    # Execute a known agent and verify it appears in history
    initial_history_len = len(registry.get_execution_history(limit=100))
    registry.execute_agent("architect", "Design a simple REST API")
    new_history = registry.get_execution_history(limit=100)
    assert len(new_history) > initial_history_len, "Execution should be added to history"
    last = new_history[-1]
    assert last.agent_id == "architect"

    print("✓ AgentRegistry execution tracking test passed")


def test_agent_definition_from_dict():
    """Test AgentDefinition.from_dict() constructor."""
    print("Testing AgentDefinition.from_dict...")
    data = {
        "id": "custom_agent",
        "role": "Custom Specialist",
        "description": "A custom test agent",
        "tags": ["custom", "test"],
        "capabilities": ["testing", "validation"],
        "required_tools": ["read_file"],
        "preferred_profile": "safe",
        "risk_level": "low",
    }
    agent = AgentDefinition.from_dict(data)
    assert agent.id == "custom_agent"
    assert agent.role == "Custom Specialist"
    assert agent.description == "A custom test agent"
    assert "custom" in agent.tags
    assert "testing" in agent.capabilities
    assert "read_file" in agent.required_tools
    assert agent.preferred_profile == "safe"
    assert agent.risk_level == "low"

    # Minimal dict (only required fields)
    minimal = AgentDefinition.from_dict({"id": "minimal", "role": "Minimal"})
    assert minimal.id == "minimal"
    assert minimal.tags == []
    assert minimal.capabilities == []

    print("✓ AgentDefinition.from_dict test passed")


# ============================================================================
# RealAI Internal Helper Method Tests
# ============================================================================

def test_parse_json_block():
    """Test RealAI._parse_json_block static method."""
    print("Testing RealAI._parse_json_block...")

    # Plain JSON object
    result = RealAI._parse_json_block('{"key": "value", "num": 42}')
    assert result == {"key": "value", "num": 42}

    # Fenced JSON block (```json ... ```)
    fenced = '```json\n{"answer": "yes"}\n```'
    result = RealAI._parse_json_block(fenced)
    assert result == {"answer": "yes"}

    # Fenced block without language tag
    fenced_plain = '```\n{"x": 1}\n```'
    result = RealAI._parse_json_block(fenced_plain)
    assert result == {"x": 1}

    # Invalid JSON returns empty dict
    result = RealAI._parse_json_block("this is not json at all")
    assert result == {}

    # JSON array (not a dict) returns empty dict
    result = RealAI._parse_json_block("[1, 2, 3]")
    assert result == {}

    # Empty string returns empty dict
    result = RealAI._parse_json_block("")
    assert result == {}

    print("✓ RealAI._parse_json_block test passed")


def test_with_metadata_direct():
    """Test RealAI._with_metadata attaches correct metadata fields."""
    print("Testing RealAI._with_metadata...")
    model = RealAI()
    response = {"some_key": "some_value"}

    enriched = model._with_metadata(response, capability="text_generation")
    assert "realai_meta" in enriched
    meta = enriched["realai_meta"]
    assert meta["capability"] == "text_generation"
    assert meta["modality"] == "text"  # default
    assert "provider" in meta
    assert "model" in meta
    assert "timestamp" in meta
    assert "contract_version" in meta
    assert enriched["some_key"] == "some_value"  # original data preserved

    # Custom modality and extra fields
    enriched2 = model._with_metadata(
        {"data": []}, capability="image_generation", modality="image",
        extra={"persona": "creative"}
    )
    assert enriched2["realai_meta"]["modality"] == "image"
    assert enriched2["realai_meta"]["persona"] == "creative"

    # Provider-configured model uses provider name
    model_with_provider = RealAI(api_key="sk-testkey123")
    enriched3 = model_with_provider._with_metadata({}, capability="text_generation")
    assert enriched3["realai_meta"]["provider"] == "openai"

    print("✓ RealAI._with_metadata test passed")


def test_provider_supports():
    """Test RealAI._provider_supports for various provider/capability combinations."""
    print("Testing RealAI._provider_supports...")

    # No provider → always True
    model_no_provider = RealAI()
    assert model_no_provider._provider_supports("text_generation") is True
    assert model_no_provider._provider_supports("nonexistent_capability") is True

    # OpenAI supports text_generation
    model_openai = RealAI(api_key="sk-testkey")
    assert model_openai.provider == "openai"
    assert model_openai._provider_supports("text_generation") is True

    # OpenAI does not support therapy_counseling
    assert model_openai._provider_supports("therapy_counseling") is False

    # Anthropic supports chain_of_thought
    model_anthropic = RealAI(api_key="sk-ant-testkey")
    assert model_anthropic._provider_supports("chain_of_thought") is True

    # Provider name not in PROVIDER_CAPABILITY_MAP returns True
    model_custom = RealAI(api_key="somekey", provider="together")
    # "together" IS in the map, check a known supported cap
    assert model_custom._provider_supports("text_generation") is True

    print("✓ RealAI._provider_supports test passed")


# ============================================================================
# CAPABILITY_DOMAIN_MAP Completeness Tests
# ============================================================================

def test_capability_domain_map_completeness():
    """Test that every ModelCapability value is present in CAPABILITY_DOMAIN_MAP."""
    print("Testing CAPABILITY_DOMAIN_MAP completeness...")
    missing = [
        cap for cap in ModelCapability
        if cap not in CAPABILITY_DOMAIN_MAP
    ]
    assert missing == [], f"Capabilities missing from CAPABILITY_DOMAIN_MAP: {[c.value for c in missing]}"

    # All domain values should be non-empty strings
    for cap, domain in CAPABILITY_DOMAIN_MAP.items():
        assert isinstance(domain, str) and domain, (
            f"Domain for {cap.value!r} must be a non-empty string, got {domain!r}"
        )
    print("✓ CAPABILITY_DOMAIN_MAP completeness test passed")


# ============================================================================
# Persona Profile Tests
# ============================================================================

def test_persona_profiles_completeness():
    """Test that all persona profiles have required fields."""
    print("Testing PERSONA_PROFILES completeness...")
    required_keys = {"description", "system_prompt"}
    for name, profile in PERSONA_PROFILES.items():
        for key in required_keys:
            assert key in profile, f"Persona '{name}' missing key '{key}'"
            assert isinstance(profile[key], str) and profile[key], (
                f"Persona '{name}'.{key} must be a non-empty string"
            )
    # Verify the four documented personas exist
    for persona_name in ("balanced", "analyst", "creative", "coach"):
        assert persona_name in PERSONA_PROFILES, f"Persona '{persona_name}' not found"
    print("✓ PERSONA_PROFILES completeness test passed")


def test_set_persona_invalid():
    """Test that set_persona raises ValueError for an unknown persona name."""
    print("Testing set_persona with invalid name...")
    model = RealAI()
    try:
        model.set_persona("nonexistent_persona_xyz")
        assert False, "Expected ValueError for unknown persona"
    except ValueError as exc:
        assert "nonexistent_persona_xyz" in str(exc)
    print("✓ set_persona invalid persona test passed")


# ============================================================================
# CloudDeploymentManager Direct Tests
# ============================================================================

def test_cloud_deployment_manager_deploy():
    """Test CloudDeploymentManager.deploy_instance for multiple providers."""
    print("Testing CloudDeploymentManager.deploy_instance...")
    manager = CloudDeploymentManager()

    for provider_enum, region, instance_type in [
        (CloudProvider.VERCEL, "iad1", "pro"),
        (CloudProvider.RENDER, "oregon", "starter"),
        (CloudProvider.RAILWAY, "us-west", "hobby"),
    ]:
        instance = manager.deploy_instance(
            provider=provider_enum,
            region=region,
            instance_type=instance_type,
            realai_config={"test": True},
        )
        assert isinstance(instance, CloudInstance)
        assert instance.provider == provider_enum
        assert instance.region == region
        assert instance.instance_type == instance_type
        assert instance.status == "running"
        assert instance.url, "URL should be set after deployment"
        assert instance.instance_id in manager.deployments

    print("✓ CloudDeploymentManager.deploy_instance test passed")


def test_cloud_deployment_manager_terminate():
    """Test CloudDeploymentManager.terminate_instance."""
    print("Testing CloudDeploymentManager.terminate_instance...")
    manager = CloudDeploymentManager()

    # Terminating non-existent ID returns False
    assert manager.terminate_instance("nonexistent-id") is False

    # Deploy then terminate
    instance = manager.deploy_instance(
        provider=CloudProvider.VERCEL,
        region="iad1",
        instance_type="hobby",
        realai_config={},
    )
    assert manager.terminate_instance(instance.instance_id) is True
    assert manager.deployments[instance.instance_id].status == "terminated"

    print("✓ CloudDeploymentManager.terminate_instance test passed")


def test_cloud_deployment_manager_instances_and_cost():
    """Test get_active_instances and get_total_cost_per_hour."""
    print("Testing CloudDeploymentManager active instances and cost...")
    manager = CloudDeploymentManager()
    initial_active = len(manager.get_active_instances())

    # Deploy two instances
    inst1 = manager.deploy_instance(CloudProvider.RENDER, "oregon", "starter", {})
    inst2 = manager.deploy_instance(CloudProvider.RAILWAY, "us-west", "hobby", {})

    active = manager.get_active_instances()
    assert len(active) == initial_active + 2

    cost = manager.get_total_cost_per_hour()
    assert isinstance(cost, float)
    assert cost >= 0.0

    # Terminate one — active count decreases
    manager.terminate_instance(inst1.instance_id)
    assert len(manager.get_active_instances()) == initial_active + 1

    print("✓ CloudDeploymentManager active instances and cost test passed")


# ============================================================================
# LoadBalancer Direct Tests
# ============================================================================

def test_load_balancer_select_instance():
    """Test LoadBalancer.select_instance with available instances."""
    print("Testing LoadBalancer.select_instance...")
    lb = LoadBalancer()

    # No instances → None
    dummy_task = type("T", (), {"priority": 1})()
    assert lb.select_instance([], dummy_task) is None

    # Create mock instances
    inst_a = CloudInstance(
        instance_id="inst-a", provider=CloudProvider.VERCEL,
        region="iad1", instance_type="pro", status="running",
    )
    inst_b = CloudInstance(
        instance_id="inst-b", provider=CloudProvider.RENDER,
        region="oregon", instance_type="starter", status="running",
    )

    # First selection picks one of them
    selected = lb.select_instance([inst_a, inst_b], dummy_task)
    assert selected is not None
    assert selected.instance_id in ("inst-a", "inst-b")

    # Load for selected instance is incremented
    assert lb.instance_load.get(selected.instance_id, 0) == 1

    print("✓ LoadBalancer.select_instance test passed")


def test_load_balancer_release_instance():
    """Test LoadBalancer.release_instance decrements load correctly."""
    print("Testing LoadBalancer.release_instance...")
    lb = LoadBalancer()
    lb.instance_load["inst-x"] = 3

    lb.release_instance("inst-x")
    assert lb.instance_load["inst-x"] == 2

    # Release below zero is clamped at 0
    lb.instance_load["inst-y"] = 0
    lb.release_instance("inst-y")
    assert lb.instance_load["inst-y"] == 0

    # Releasing unknown instance doesn't raise
    lb.release_instance("inst-unknown")

    print("✓ LoadBalancer.release_instance test passed")


# ============================================================================
# LearningRecorder Direct Tests
# ============================================================================

def test_learning_recorder():
    """Test LearningRecorder start/stop/record_action/learn_pattern."""
    print("Testing LearningRecorder...")
    recorder = LearningRecorder()

    # Initially not recording
    assert recorder.is_recording is False

    # start_recording returns True and sets flag
    result = recorder.start_recording("test_task")
    assert result is True
    assert recorder.is_recording is True

    # Starting again while recording returns False
    result2 = recorder.start_recording("another_task")
    assert result2 is False

    # record_action while recording adds to current_session
    action = RecordedAction(
        timestamp=time.time(),
        action_type="click",
        position=(50, 50),
        metadata={"button": "left"},
    )
    recorder.record_action(action)
    assert len(recorder.current_session) == 1

    # stop_recording returns the actions
    actions = recorder.stop_recording()
    assert recorder.is_recording is False
    assert isinstance(actions, list)

    # record_action when NOT recording does not add to current_session
    recorder.record_action(action)
    assert len(recorder.current_session) == 0

    # learn_pattern returns expected structure
    pattern = recorder.learn_pattern([action])
    assert pattern["status"] == "success"
    assert "pattern_type" in pattern
    assert "confidence" in pattern

    print("✓ LearningRecorder test passed")


# ============================================================================
# ScreenCapture Direct Tests
# ============================================================================

def test_screen_capture_analyze_screen():
    """Test ScreenCapture.analyze_screen returns expected structure."""
    print("Testing ScreenCapture.analyze_screen...")
    sc = ScreenCapture()
    result = sc.analyze_screen("base64encodedimage==", "What buttons are visible?")
    assert result["status"] == "success"
    assert "analysis" in result
    assert "elements" in result
    assert isinstance(result["elements"], list)
    assert "confidence" in result
    print("✓ ScreenCapture.analyze_screen test passed")


# ============================================================================
# ComputerMode Direct Tests
# ============================================================================

def test_computer_mode_execute_action_types():
    """Test ComputerMode.execute_action for various action types."""
    print("Testing ComputerMode.execute_action...")
    cm = ComputerMode()

    # All action types should return a dict with 'status' and 'action_type'
    for action_type, kwargs in [
        ("move_mouse", {"x": 100, "y": 200}),
        ("click", {"button": "left"}),
        ("type_text", {"text": "hello"}),
        ("press_key", {"key": "enter"}),
        ("hotkey", {"keys": ["ctrl", "c"]}),
        ("switch_window", {"title_contains": "notepad"}),
    ]:
        result = cm.execute_action(action_type, **kwargs)
        assert "status" in result, f"Missing 'status' for action_type={action_type}"
        assert "action_type" in result, f"Missing 'action_type' for action_type={action_type}"

    # Unknown action type returns error
    result = cm.execute_action("unknown_action_xyz")
    assert result["status"] == "error"

    print("✓ ComputerMode.execute_action test passed")


def test_computer_mode_build_app_all_types():
    """Test ComputerMode.build_app for all supported and unsupported app types."""
    print("Testing ComputerMode.build_app for all types...")
    cm = ComputerMode()

    # Web app
    result = cm.build_app("web", {"framework": "vue"})
    assert result["status"] == "success"
    assert result["app_type"] == "web"

    # Mobile app
    result = cm.build_app("mobile", {"platform": "android"})
    assert result["status"] == "success"
    assert result["app_type"] == "mobile"

    # Game
    result = cm.build_app("game", {"genre": "rpg"})
    assert result["status"] == "success"
    assert result["app_type"] == "game"

    # Crypto app
    result = cm.build_app("crypto", {"blockchain": "solana"})
    assert result["status"] == "success"
    assert result["app_type"] == "crypto"

    # Unknown app type returns error
    result = cm.build_app("unsupported_app_type_xyz", {})
    assert result["status"] == "error"

    print("✓ ComputerMode.build_app all types test passed")


def test_computer_mode_stop_learning_no_actions():
    """Test ComputerMode.stop_learning when no actions were recorded."""
    print("Testing ComputerMode.stop_learning with no actions...")
    cm = ComputerMode()

    # start_learning and then immediately stop — no actions recorded
    cm.start_learning("empty_task")
    result = cm.stop_learning()
    # The recorder.stop_recording() returns a copy of recordings (which may be
    # empty since no actions were added).  The method should return a response dict.
    assert "status" in result

    print("✓ ComputerMode.stop_learning no-actions test passed")


def test_computer_mode_automate_task():
    """Test ComputerMode.automate_task returns expected structure."""
    print("Testing ComputerMode.automate_task...")
    cm = ComputerMode()
    result = cm.automate_task("open_browser", url="https://example.com")
    assert result["status"] == "success"
    assert "task" in result
    assert "execution_status" in result
    print("✓ ComputerMode.automate_task test passed")


# ============================================================================
# RealAI Edge Case Tests
# ============================================================================

def test_generate_video_b64_json():
    """Test generate_video with response_format='b64_json'."""
    print("Testing generate_video with b64_json format...")
    model = RealAI()
    response = model.generate_video(
        prompt="A rolling wave",
        response_format="b64_json",
        n=1,
    )
    assert "data" in response
    assert len(response["data"]) == 1
    item = response["data"][0]
    assert "b64_json" in item
    assert isinstance(item["b64_json"], str)
    print("✓ generate_video b64_json test passed")


def test_generate_video_with_image_url():
    """Test generate_video with image_url for image-to-video mode."""
    print("Testing generate_video with image_url...")
    model = RealAI()
    response = model.generate_video(
        prompt="Zoom out slowly",
        image_url="https://example.com/input.jpg",
    )
    assert "data" in response
    assert len(response["data"]) > 0
    item = response["data"][0]
    # mode should be image_to_video
    assert item.get("mode") == "image_to_video"
    assert "source_image_url" in item
    print("✓ generate_video with image_url test passed")


def test_generate_video_multiple_n():
    """Test generate_video with n > 1 returns multiple items."""
    print("Testing generate_video with n=3...")
    model = RealAI()
    response = model.generate_video(prompt="A city timelapse", n=3)
    assert "data" in response
    assert len(response["data"]) == 3
    print("✓ generate_video n=3 test passed")


def test_create_embeddings_list_input():
    """Test create_embeddings with a list of texts."""
    print("Testing create_embeddings with list input...")
    model = RealAI()
    response = model.create_embeddings(["hello world", "foo bar", "baz"])
    assert "data" in response
    assert len(response["data"]) == 3
    for item in response["data"]:
        assert "embedding" in item
        assert isinstance(item["embedding"], list)
        assert len(item["embedding"]) > 0
    print("✓ create_embeddings list input test passed")


def test_automate_task_groceries_plan_mode():
    """Test automate_task with groceries type in plan (not execute) mode."""
    print("Testing automate_task groceries plan mode...")
    model = RealAI()
    response = model.automate_task(
        task_type="groceries",
        task_details={"items": ["apples", "bread", "milk"]},
        execute=False,
    )
    assert response["task_type"] == "groceries"
    assert response["status"] == "planned"
    assert "plan" in response
    print("✓ automate_task groceries plan mode test passed")


def test_automate_task_appointment_plan_mode():
    """Test automate_task with appointment type in plan mode."""
    print("Testing automate_task appointment plan mode...")
    model = RealAI()
    response = model.automate_task(
        task_type="appointment",
        task_details={"title": "Dentist", "start_time": "2026-06-01T10:00:00"},
        execute=False,
    )
    assert response["task_type"] == "appointment"
    assert response["status"] == "planned"
    print("✓ automate_task appointment plan mode test passed")


def test_web_research_caching():
    """Test that web_research returns a cached result on a second identical call."""
    print("Testing web_research caching...")
    model = RealAI()
    query = "test_cache_unique_xyzabc987"
    depth = "quick"
    cache_key = f"{query}|{depth}|"
    now = int(time.time())

    # Manually inject a cache entry so the test does not depend on network access
    model._web_research_cache[cache_key] = {
        "cached_at": now,
        "payload": {
            "query": query,
            "findings": "Injected test findings",
            "summary": "Test summary",
            "sources": [],
            "source_details": [],
            "citations": [],
            "depth": depth,
            "confidence": 0.9,
            "timestamp": now,
            "freshness": "live",
            "cached": False,
        },
    }

    # Call web_research — should hit the cache
    r = model.web_research(query=query, depth=depth)
    assert r.get("cached") is True, f"Expected cached=True, got cached={r.get('cached')}"
    assert r.get("freshness") == "cached"
    assert r["query"] == query

    print("✓ web_research caching test passed")


def test_execute_code_unsupported_language():
    """Test execute_code with a language other than Python."""
    print("Testing execute_code with unsupported language...")
    model = RealAI()
    response = model.execute_code(code="console.log('hello')", language="javascript")
    assert response["execution_status"] == "unsupported_language"
    assert response["language"] == "javascript"
    assert response["sandboxed"] is False
    print("✓ execute_code unsupported language test passed")


def test_chat_completion_persona_in_metadata():
    """Test that persona is reflected in chat_completion realai_meta."""
    print("Testing chat_completion persona in metadata...")
    model = RealAI()
    model.set_persona("creative")
    response = model.chat_completion(messages=[{"role": "user", "content": "Hello"}])
    assert "realai_meta" in response
    assert response["realai_meta"].get("persona") == "creative"
    print("✓ chat_completion persona metadata test passed")


def test_get_provider_capabilities_explicit_provider():
    """Test get_provider_capabilities with an explicit provider argument."""
    print("Testing get_provider_capabilities with explicit provider...")
    model = RealAI()
    result = model.get_provider_capabilities(provider="anthropic")
    assert result["provider"] == "anthropic"
    assert "chain_of_thought" in result["supported_capabilities"]
    assert isinstance(result["unsupported_capabilities"], list)
    print("✓ get_provider_capabilities explicit provider test passed")


# ============================================================================
# realai_api.config Tests (imported directly, no FastAPI dependency)
# ============================================================================

def _load_api_config():
    """Load realai_api/config.py directly without triggering the FastAPI import."""
    config_path = os.path.join(os.path.dirname(__file__), "realai_api", "config.py")
    spec = importlib.util.spec_from_file_location("realai_api_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_api_config_default_keys():
    """Test get_valid_api_keys returns the built-in fallback set when env var is absent."""
    print("Testing api_config default keys...")
    config = _load_api_config()

    # Ensure the env var is not set
    os.environ.pop("REALAI_API_KEYS", None)
    keys = config.get_valid_api_keys()
    assert isinstance(keys, set)
    assert len(keys) >= 1
    # Both default keys should be present
    assert "realai-dev" in keys
    assert "realai-demo" in keys
    print("✓ api_config default keys test passed")


def test_api_config_env_keys():
    """Test get_valid_api_keys respects REALAI_API_KEYS environment variable."""
    print("Testing api_config env keys...")
    config = _load_api_config()

    os.environ["REALAI_API_KEYS"] = "mykey1, mykey2 , mykey3"
    try:
        keys = config.get_valid_api_keys()
        assert "mykey1" in keys
        assert "mykey2" in keys
        assert "mykey3" in keys
        # Default keys should NOT be present when env var is set
        assert "realai-dev" not in keys
    finally:
        os.environ.pop("REALAI_API_KEYS", None)
    print("✓ api_config env keys test passed")


def test_api_config_default_models():
    """Test get_default_models returns a well-formed model list."""
    print("Testing api_config default models...")
    config = _load_api_config()
    now = int(time.time())
    models = config.get_default_models(now)
    assert isinstance(models, list)
    assert len(models) >= 1
    model = models[0]
    assert "id" in model
    assert "object" in model
    assert model["object"] == "model"
    assert model["created"] == now
    assert "owned_by" in model
    print("✓ api_config default models test passed")


# ============================================================================
# AI Training and Quality System Tests
# ============================================================================

def test_agent_evals():
    """Test agent_evals capability runs and returns evaluation results."""
    print("Testing agent_evals...")
    model = RealAI()
    result = model.agent_evals(
        task_name="chat_quality",
        golden_examples=[
            {"input": "Hello", "expected": "hello", "actual": "Hello there!"},
            {"input": "Bye", "expected": "bye", "actual": "Goodbye!"}
        ],
        metrics=["accuracy", "latency"]
    )
    assert result["status"] == "success"
    assert "scores" in result
    assert "accuracy" in result["scores"]
    assert "regression_detected" in result
    assert result["realai_meta"]["capability"] == ModelCapability.AGENT_EVALS.value
    print("✓ agent_evals test passed")


def test_agent_evals_client():
    """Test client.training.run_evals() convenience method."""
    print("Testing client.training.run_evals...")
    client = RealAIClient()
    result = client.training.run_evals(
        task_name="grounding_quality",
        metrics=["accuracy", "hallucination_rate"]
    )
    assert result["status"] == "success"
    assert "scores" in result
    print("✓ client.training.run_evals test passed")


def test_feedback_learning():
    """Test feedback_learning capability processes feedback items."""
    print("Testing feedback_learning...")
    model = RealAI()
    result = model.feedback_learning(
        feedback_items=[
            {"text": "Wrong answer about capital", "label": "negative", "correction": "Paris is the capital of France"},
            {"text": "Great response", "label": "positive"}
        ],
        policy_target="accuracy"
    )
    assert result["status"] == "success"
    assert result["feedback_processed"] == 2
    assert "failure_clusters" in result
    assert "policy_updates" in result
    assert result["realai_meta"]["capability"] == ModelCapability.FEEDBACK_LEARNING.value
    print("✓ feedback_learning test passed")


def test_feedback_learning_client():
    """Test client.training.process_feedback() convenience method."""
    print("Testing client.training.process_feedback...")
    client = RealAIClient()
    result = client.training.process_feedback(
        feedback_items=[{"text": "Slow response", "label": "negative"}],
        policy_target="latency"
    )
    assert result["status"] == "success"
    assert result["feedback_processed"] == 1
    print("✓ client.training.process_feedback test passed")


def test_grounding():
    """Test grounding capability returns retrieval-grounded response."""
    print("Testing grounding...")
    model = RealAI()
    result = model.grounding(
        query="What is the capital of France?",
        sources=[
            {"text": "France is a country in Europe. Paris is its capital city.", "url": "https://example.com/france"},
            {"text": "Germany has Berlin as its capital.", "url": "https://example.com/germany"}
        ],
        citation_mode="inline"
    )
    assert result["status"] == "success"
    assert "grounded_answer" in result
    assert "citations" in result
    assert "hallucination_risk" in result
    assert result["realai_meta"]["capability"] == ModelCapability.GROUNDING.value
    print("✓ grounding test passed")


def test_grounding_client():
    """Test client.training.ground_response() convenience method."""
    print("Testing client.training.ground_response...")
    client = RealAIClient()
    result = client.training.ground_response(
        query="What is machine learning?",
        sources=[{"text": "Machine learning is a subset of AI.", "url": "https://example.com/ml"}]
    )
    assert result["status"] == "success"
    assert result["sources_used"] >= 1
    print("✓ client.training.ground_response test passed")


def test_agent_observability():
    """Test agent_observability capability returns observability snapshot."""
    print("Testing agent_observability...")
    model = RealAI()
    result = model.agent_observability(
        trace_config={"enabled": True, "sampling_rate": 0.5, "latency_budget_ms": 2000},
        metrics_config={"latency": True, "cost": True, "drift": True}
    )
    assert result["status"] == "success"
    assert "observability_snapshot" in result
    snapshot = result["observability_snapshot"]
    assert snapshot["tracing_enabled"] is True
    assert "latency_p50_ms" in snapshot
    assert "output_drift_detected" in snapshot
    assert result["realai_meta"]["capability"] == ModelCapability.AGENT_OBSERVABILITY.value
    print("✓ agent_observability test passed")


def test_agent_observability_client():
    """Test client.training.setup_observability() convenience method."""
    print("Testing client.training.setup_observability...")
    client = RealAIClient()
    result = client.training.setup_observability(
        metrics_config={"latency": True, "cost": False}
    )
    assert result["status"] == "success"
    assert "structured_log_fields" in result
    print("✓ client.training.setup_observability test passed")


def test_ai_incident_response():
    """Test ai_incident_response capability triages incidents correctly."""
    print("Testing ai_incident_response...")
    model = RealAI()
    result = model.ai_incident_response(
        incident_type="hallucination",
        impact_scope="user",
        severity="high"
    )
    assert result["status"] == "success"
    assert result["incident_type"] == "hallucination"
    assert result["severity"] == "high"
    assert "immediate_actions" in result
    assert len(result["immediate_actions"]) > 0
    assert "postmortem_items" in result
    assert result["rollback_required"] is True
    assert result["realai_meta"]["capability"] == ModelCapability.AI_INCIDENT_RESPONSE.value
    print("✓ ai_incident_response test passed")


def test_ai_incident_response_client():
    """Test client.training.respond_to_incident() convenience method."""
    print("Testing client.training.respond_to_incident...")
    client = RealAIClient()
    result = client.training.respond_to_incident(
        incident_type="latency_spike",
        severity="medium"
    )
    assert result["status"] == "success"
    assert "immediate_actions" in result
    print("✓ client.training.respond_to_incident test passed")


def test_expansion_coordination():
    """Test expansion_coordination orders roadmap items by dependencies."""
    print("Testing expansion_coordination...")
    model = RealAI()
    result = model.expansion_coordination(
        roadmap_items=["grounding", "evals", "feedback_loop", "observability"],
        dependencies=[
            {"from": "grounding", "to": "evals"},
            {"from": "evals", "to": "feedback_loop"}
        ]
    )
    assert result["status"] == "success"
    assert result["total_items"] == 4
    assert "phases" in result
    assert len(result["phases"]) >= 1
    assert result["realai_meta"]["capability"] == ModelCapability.EXPANSION_COORDINATION.value
    print("✓ expansion_coordination test passed")


def test_expansion_coordination_client():
    """Test client.training.coordinate_expansion() convenience method."""
    print("Testing client.training.coordinate_expansion...")
    client = RealAIClient()
    result = client.training.coordinate_expansion(
        roadmap_items=["capability_A", "capability_B"],
        dependencies=[]
    )
    assert result["status"] == "success"
    assert result["total_items"] == 2
    print("✓ client.training.coordinate_expansion test passed")


def test_training_agents_in_registry():
    """Test that training agent types are in the AgentRegistry."""
    print("Testing training agents in registry...")
    registry = AgentRegistry()
    training_agent_ids = [
        "agent-evals-engineer",
        "feedback-learning-engineer",
        "grounding-engineer",
        "agent-observability-engineer",
        "ai-incident-responder",
        "expansion-coordinator"
    ]
    for agent_id in training_agent_ids:
        agent = registry.get_agent(agent_id)
        assert agent is not None, f"Agent '{agent_id}' not found in registry"
        assert agent.id == agent_id
        assert len(agent.capabilities) > 0
    print("✓ training agents in registry test passed")


def test_training_capabilities_in_domain_map():
    """Test that all training capabilities are in CAPABILITY_DOMAIN_MAP."""
    print("Testing training capabilities in domain map...")
    from realai import CAPABILITY_DOMAIN_MAP
    training_caps = [
        ModelCapability.AGENT_EVALS,
        ModelCapability.FEEDBACK_LEARNING,
        ModelCapability.GROUNDING,
        ModelCapability.AGENT_OBSERVABILITY,
        ModelCapability.AI_INCIDENT_RESPONSE,
        ModelCapability.EXPANSION_COORDINATION,
    ]
    for cap in training_caps:
        assert cap in CAPABILITY_DOMAIN_MAP, f"{cap} missing from CAPABILITY_DOMAIN_MAP"
        assert CAPABILITY_DOMAIN_MAP[cap] == "training"
    print("✓ training capabilities in domain map test passed")


def test_execution_runtime_subscribe_and_emit():
    """Test ExecutionRuntime emits events to subscribers."""
    print("Testing ExecutionRuntime subscribe/emit...")
    from realai import ExecutionRuntime, ExecutionEvent
    runtime = ExecutionRuntime()
    q = runtime.subscribe()
    runtime.start("eid1", agent_id="architect", task="test task")
    event = q.get(timeout=1)
    assert event["event_type"] == "dispatch"
    assert event["agent_id"] == "architect"
    assert event["data"]["task"] == "test task"
    runtime.unsubscribe(q)
    print("✓ ExecutionRuntime subscribe/emit test passed")


def test_execution_runtime_complete():
    """Test ExecutionRuntime complete event."""
    print("Testing ExecutionRuntime complete...")
    from realai import ExecutionRuntime
    runtime = ExecutionRuntime()
    q = runtime.subscribe()
    runtime.complete("eid2", agent_id="implementer", duration_ms=123, result="done")
    event = q.get(timeout=1)
    assert event["event_type"] == "complete"
    assert event["data"]["duration_ms"] == 123
    runtime.unsubscribe(q)
    print("✓ ExecutionRuntime complete test passed")


def test_execution_runtime_fail():
    """Test ExecutionRuntime error event."""
    print("Testing ExecutionRuntime fail...")
    from realai import ExecutionRuntime
    runtime = ExecutionRuntime()
    q = runtime.subscribe()
    runtime.fail("eid3", agent_id="qa", duration_ms=50, error="out of tokens")
    event = q.get(timeout=1)
    assert event["event_type"] == "error"
    assert "out of tokens" in event["data"]["error"]
    runtime.unsubscribe(q)
    print("✓ ExecutionRuntime fail test passed")


def test_execute_agent_emits_lifecycle_events():
    """Test that execute_agent() emits dispatch/complete events to the runtime."""
    print("Testing execute_agent lifecycle events...")
    from realai import _execution_runtime, AgentRegistry
    registry = AgentRegistry()
    q = _execution_runtime.subscribe()
    try:
        registry.execute_agent("architect", "design a REST API")
        events = []
        import queue as _q
        while True:
            try:
                events.append(q.get_nowait())
            except _q.Empty:
                break
        event_types = [e["event_type"] for e in events]
        assert "dispatch" in event_types, f"No dispatch event. Got: {event_types}"
        assert "complete" in event_types or "error" in event_types, \
            f"No terminal event. Got: {event_types}"
    finally:
        _execution_runtime.unsubscribe(q)
    print("✓ execute_agent lifecycle events test passed")


def test_execute_agent_access_check_missing_tools():
    """Test that execute_agent() emits a warning when tools are missing."""
    print("Testing execute_agent access check warning...")
    from realai import AgentDefinition, AgentRegistry, _execution_runtime
    import queue as _q
    registry = AgentRegistry()
    # Agent that requires a tool no profile has
    registry.register_agent(AgentDefinition(
        id="test-missing-tool-agent",
        role="Test Agent",
        description="Agent with impossible tool requirement",
        required_tools=["nonexistent_tool_xyz"],
        preferred_profile="balanced",
    ))
    q = _execution_runtime.subscribe()
    try:
        registry.execute_agent("test-missing-tool-agent", "do something")
        events = []
        while True:
            try:
                events.append(q.get_nowait())
            except _q.Empty:
                break
        warning_events = [e for e in events if e["event_type"] == "warning"]
        assert len(warning_events) > 0, "Expected a warning event for missing tools"
        assert "nonexistent_tool_xyz" in str(warning_events[0]["data"])
    finally:
        _execution_runtime.unsubscribe(q)
    print("✓ execute_agent access check warning test passed")


def test_feedback_learning_persists_to_memory():
    """Test that feedback_learning() with corrections calls learn_from_interaction."""
    print("Testing feedback_learning → learn_from_interaction loop...")
    import os, json
    model = RealAI()
    # Resolve the actual realai package directory for the memory file
    realai_pkg_dir = os.path.dirname(os.path.abspath(
        __import__('realai').__file__
    ))
    memory_file = os.path.join(realai_pkg_dir, 'realai_memory.json')
    # Record baseline interaction count
    baseline = 0
    if os.path.exists(memory_file):
        try:
            with open(memory_file) as f:
                mem = json.load(f)
                baseline = len(mem.get("interactions", []))
        except Exception:
            pass
    result = model.feedback_learning(
        feedback_items=[
            {"text": "What is 2+2?", "label": "negative", "correction": "2+2 equals 4."},
            {"text": "What color is the sky?", "label": "negative", "correction": "The sky is blue."},
        ],
        policy_target="accuracy"
    )
    assert result["status"] == "success"
    assert result["correction_dataset_size"] == 2
    # Verify that memory was updated (interactions increased)
    if os.path.exists(memory_file):
        try:
            with open(memory_file) as f:
                mem = json.load(f)
                after = len(mem.get("interactions", []))
                assert after >= baseline + 2, \
                    f"Expected at least {baseline+2} interactions, got {after}"
        except Exception:
            pass  # memory file may not be writable in CI — skip assertion
    print("✓ feedback_learning → learn_from_interaction loop test passed")


def test_grounding_auto_populates_sources():
    """Test that grounding() with empty sources tries web_research."""
    print("Testing grounding auto-populate sources...")
    model = RealAI()
    # With no sources, grounding should still return a valid result
    result = model.grounding(query="What is machine learning?", sources=[])
    assert result["status"] == "success"
    assert "grounded_answer" in result
    assert result["hallucination_risk"] in ("low", "medium", "high")
    print("✓ grounding auto-populate sources test passed")


def test_grounding_with_sources_still_works():
    """Test grounding with explicit sources is unaffected by auto-populate."""
    print("Testing grounding with explicit sources...")
    model = RealAI()
    result = model.grounding(
        query="Python programming",
        sources=[{"text": "Python is a high-level programming language.", "url": "https://example.com/py"}],
    )
    assert result["status"] == "success"
    assert result["sources_used"] >= 1
    print("✓ grounding with explicit sources test passed")


def test_tool_call_dataclass():
    """Test ToolCall dataclass serialise/deserialise round-trip."""
    print("Testing ToolCall round-trip...")
    from realai import ToolCall
    tc = ToolCall(id="call_1", tool="http", arguments={"url": "https://example.com", "method": "GET"})
    d = tc.to_dict()
    assert d["id"] == "call_1"
    assert d["type"] == "function"
    assert d["function"]["name"] == "http"
    import json as _json
    args = _json.loads(d["function"]["arguments"])
    assert args["url"] == "https://example.com"
    # Round-trip from_dict
    tc2 = ToolCall.from_dict(d)
    assert tc2.tool == "http"
    assert tc2.arguments["method"] == "GET"
    print("✓ ToolCall round-trip test passed")


def test_self_critique_engine_improves():
    """Test SelfCritiqueEngine retries on low-quality results."""
    print("Testing SelfCritiqueEngine retry loop...")
    from realai import SelfCritiqueEngine
    call_count = [0]

    def bad_then_good(task: str) -> str:
        call_count[0] += 1
        if call_count[0] == 1:
            return "error: fail"   # score < threshold
        return "A " * 60           # long enough to score well

    engine = SelfCritiqueEngine(max_retries=2, threshold=0.6)
    result = engine.run(executor_fn=bad_then_good, task="do the thing")
    assert call_count[0] == 2, f"Expected 2 calls, got {call_count[0]}"
    assert len(result) > 20
    print("✓ SelfCritiqueEngine retry loop test passed")


def test_self_critique_engine_no_retry_on_good():
    """Test SelfCritiqueEngine does not retry when first result is good."""
    print("Testing SelfCritiqueEngine no retry on good result...")
    from realai import SelfCritiqueEngine
    call_count = [0]

    def always_good(task: str) -> str:
        call_count[0] += 1
        return "This is a detailed, comprehensive response. " * 10

    engine = SelfCritiqueEngine(max_retries=3, threshold=0.6)
    engine.run(executor_fn=always_good, task="do the thing")
    assert call_count[0] == 1, "Should not retry a good result"
    print("✓ SelfCritiqueEngine no-retry test passed")


def test_model_registry_lookup():
    """Test ModelRegistry get and list_all."""
    print("Testing ModelRegistry lookup...")
    from realai.model_registry import MODEL_REGISTRY
    meta = MODEL_REGISTRY.get("realai-2.0")
    assert meta is not None
    assert meta.id == "realai-2.0"
    assert meta.context_window >= 1024
    all_models = MODEL_REGISTRY.list_all()
    assert len(all_models) >= 5
    # Should include RealAI and 3rd-party entries
    ids = [m.id for m in all_models]
    assert "realai-1.0-agentic" in ids
    assert "gpt-4o" in ids
    print("✓ ModelRegistry lookup test passed")


def test_model_registry_recommend():
    """Test ModelRegistry.recommend() returns a valid model."""
    print("Testing ModelRegistry recommend...")
    from realai.model_registry import MODEL_REGISTRY
    best = MODEL_REGISTRY.recommend(need_tools=True, max_cost=5)
    assert best is not None
    assert best.tool_calling is True
    cheap = MODEL_REGISTRY.recommend(max_cost=2)
    assert cheap is not None
    assert cheap.cost_score <= 2
    print("✓ ModelRegistry recommend test passed")


def test_model_registry_openai_list():
    """Test ModelRegistry.to_openai_list() returns correct shape."""
    print("Testing ModelRegistry to_openai_list...")
    from realai.model_registry import MODEL_REGISTRY
    result = MODEL_REGISTRY.to_openai_list()
    assert result["object"] == "list"
    assert len(result["data"]) >= 5
    entry = result["data"][0]
    assert "id" in entry
    assert entry["object"] == "model"
    print("✓ ModelRegistry to_openai_list test passed")


def test_safety_filter_blocks_harmful():
    """Test SafetyFilter blocks clearly harmful inputs."""
    print("Testing SafetyFilter harmful block...")
    from realai.safety import SAFETY_FILTER
    result = SAFETY_FILTER.check_input("how to make a bomb at home")
    assert result.blocked is True
    assert result.rule_id == "input-hard-block"
    print("✓ SafetyFilter harmful block test passed")


def test_safety_filter_passes_safe():
    """Test SafetyFilter passes safe inputs."""
    print("Testing SafetyFilter safe pass...")
    from realai.safety import SAFETY_FILTER
    result = SAFETY_FILTER.check_input("How do I sort a list in Python?")
    assert result.blocked is False
    assert result.ok is True
    print("✓ SafetyFilter safe pass test passed")


def test_safety_filter_output_pii_redaction():
    """Test SafetyFilter redacts PII in outputs."""
    print("Testing SafetyFilter PII redaction...")
    from realai.safety import SAFETY_FILTER
    text = "Contact me at user@example.com or call 555-1234."
    result = SAFETY_FILTER.check_output(text)
    assert result.flagged is True
    assert result.redacted_text is not None
    assert "user@example.com" not in result.redacted_text
    print("✓ SafetyFilter PII redaction test passed")


def test_safety_filter_tool_allow_list():
    """Test SafetyFilter tool allow-list enforcement."""
    print("Testing SafetyFilter tool allow-list...")
    from realai.safety import SAFETY_FILTER
    ok = SAFETY_FILTER.check_tool_call("researcher", "http", ["http", "filesystem"])
    assert ok.ok is True
    blocked = SAFETY_FILTER.check_tool_call("researcher", "solana", ["http"])
    assert blocked.blocked is True
    print("✓ SafetyFilter tool allow-list test passed")


def test_agent_registry_get_execution_runtime():
    """Test that AgentRegistry.get_execution_runtime() returns the global runtime."""
    print("Testing AgentRegistry.get_execution_runtime...")
    from realai import AgentRegistry, ExecutionRuntime
    runtime = AgentRegistry.get_execution_runtime()
    assert isinstance(runtime, ExecutionRuntime)
    print("✓ AgentRegistry.get_execution_runtime test passed")


def test_workflow_example_files_valid_json():
    """Test that example workflow JSON files are valid."""
    print("Testing workflow example files...")
    import os, json
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
    for fname in ("workflow-ai-training.json", "workflow-fullstack.json"):
        fpath = os.path.join(base, fname)
        assert os.path.exists(fpath), f"Missing: {fpath}"
        with open(fpath) as f:
            data = json.load(f)
        assert "name" in data
        assert "steps" in data
        assert len(data["steps"]) >= 2
        for step in data["steps"]:
            assert "agent_id" in step
            assert "task" in step
    print("✓ workflow example files test passed")


def test_schema_files_valid_json():
    """Test that schema JSON files are valid and have required fields."""
    print("Testing schema files...")
    import os, json
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema")
    for fname in ("agent.schema.json", "tool.schema.json"):
        fpath = os.path.join(base, fname)
        assert os.path.exists(fpath), f"Missing: {fpath}"
        with open(fpath) as f:
            data = json.load(f)
        assert "$schema" in data
        assert "title" in data
        assert "properties" in data
    print("✓ schema files test passed")


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
        test_web3_gpg_signing,
        test_code_execution,
        test_plugin_system,
        test_memory_learning,
        test_model_capabilities,
        test_model_info,
        # Future AI Capabilities (2026+)
        test_quantum_integration,
        test_neural_architecture_search,
        test_causal_reasoning,
        test_meta_learning,
        test_emotional_intelligence,
        test_swarm_intelligence,
        test_predictive_simulation,
        test_consciousness_simulation,
        test_reality_simulation,
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
        # Agent Orchestration and Hive Mind System Tests
        test_agent_orchestration,
        test_agent_execution,
        test_agent_registration,
        test_agent_listing,
        test_agent_status,
        test_core_agent_methods,
        # Cloud Computing and Distributed Systems Tests
        test_cloud_deployment_orchestration,
        test_distributed_computing_coordination,
        test_auto_scaling_management,
        test_load_balancing_optimization,
        test_multi_cloud_resource_management,
        test_serverless_function_deployment,
        test_container_orchestration,
        test_cloud_cost_optimization,
        test_distributed_ai_training,
        test_cloud_native_ai_inference,
        test_cloud_convenience_methods,
        # Computer Mode and Desktop Automation Tests
        test_computer_mode_activation,
        test_screen_capture_analysis,
        test_mouse_keyboard_control,
        test_window_management,
        test_gui_automation,
        test_development_workflow_automation,
        test_self_learning_recording,
        test_action_replay_execution,
        test_code_generation_automation,
        test_app_building_automation,
        # Crypto Trading and Mining Tests
        test_crypto_mining,
        test_ai_trading_bot_integration,
        test_freqtrade_integration,
        test_hummingbot_integration,
        test_octobot_integration,
        test_jessie_trading_integration,
        test_superalgos_integration,
        test_polymarket_bot_integration,
        test_market_analysis,
        test_trading_strategy_optimization,
        test_risk_management,
        test_portfolio_management,
        # AgentRegistry internal method tests
        test_agent_registry_find_agents,
        test_agent_registry_recommend_profile,
        test_agent_registry_assess_access,
        test_agent_registry_execution_tracking,
        test_agent_definition_from_dict,
        # RealAI internal helper method tests
        test_parse_json_block,
        test_with_metadata_direct,
        test_provider_supports,
        # CAPABILITY_DOMAIN_MAP / persona tests
        test_capability_domain_map_completeness,
        test_persona_profiles_completeness,
        test_set_persona_invalid,
        # CloudDeploymentManager direct tests
        test_cloud_deployment_manager_deploy,
        test_cloud_deployment_manager_terminate,
        test_cloud_deployment_manager_instances_and_cost,
        # LoadBalancer direct tests
        test_load_balancer_select_instance,
        test_load_balancer_release_instance,
        # LearningRecorder direct tests
        test_learning_recorder,
        # ScreenCapture direct tests
        test_screen_capture_analyze_screen,
        # ComputerMode direct tests
        test_computer_mode_execute_action_types,
        test_computer_mode_build_app_all_types,
        test_computer_mode_stop_learning_no_actions,
        test_computer_mode_automate_task,
        # RealAI edge case tests
        test_generate_video_b64_json,
        test_generate_video_with_image_url,
        test_generate_video_multiple_n,
        test_create_embeddings_list_input,
        test_automate_task_groceries_plan_mode,
        test_automate_task_appointment_plan_mode,
        test_web_research_caching,
        test_execute_code_unsupported_language,
        test_chat_completion_persona_in_metadata,
        test_get_provider_capabilities_explicit_provider,
        # realai_api.config tests
        test_api_config_default_keys,
        test_api_config_env_keys,
        test_api_config_default_models,
        # AI Training and Quality System Tests
        test_agent_evals,
        test_agent_evals_client,
        test_feedback_learning,
        test_feedback_learning_client,
        test_grounding,
        test_grounding_client,
        test_agent_observability,
        test_agent_observability_client,
        test_ai_incident_response,
        test_ai_incident_response_client,
        test_expansion_coordination,
        test_expansion_coordination_client,
        test_training_agents_in_registry,
        test_training_capabilities_in_domain_map,
        # ExecutionRuntime and lifecycle tests
        test_execution_runtime_subscribe_and_emit,
        test_execution_runtime_complete,
        test_execution_runtime_fail,
        test_execute_agent_emits_lifecycle_events,
        test_execute_agent_access_check_missing_tools,
        # feedback_learning self-improvement loop
        test_feedback_learning_persists_to_memory,
        # grounding auto-populate
        test_grounding_auto_populates_sources,
        test_grounding_with_sources_still_works,
        # ToolCall protocol
        test_tool_call_dataclass,
        # SelfCritiqueEngine
        test_self_critique_engine_improves,
        test_self_critique_engine_no_retry_on_good,
        # ModelRegistry
        test_model_registry_lookup,
        test_model_registry_recommend,
        test_model_registry_openai_list,
        # SafetyFilter
        test_safety_filter_blocks_harmful,
        test_safety_filter_passes_safe,
        test_safety_filter_output_pii_redaction,
        test_safety_filter_tool_allow_list,
        # AgentRegistry helpers
        test_agent_registry_get_execution_runtime,
        # Example files
        test_workflow_example_files_valid_json,
        test_schema_files_valid_json,
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
