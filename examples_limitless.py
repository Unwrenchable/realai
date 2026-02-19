"""
Comprehensive examples of RealAI's limitless capabilities.

This script demonstrates all 17 capabilities of RealAI - the AI that has no limits!
"""

from realai import RealAIClient
import json


def example_web_research():
    """Example: Web research capability."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Web Research - Research Anything!")
    print("="*70)
    
    client = RealAIClient()
    
    research = client.web.research(
        query="Latest breakthroughs in quantum computing",
        depth="deep"
    )
    
    print(json.dumps(research, indent=2))


def example_task_automation():
    """Example: Real-world task automation."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Task Automation - Order Groceries!")
    print("="*70)
    
    client = RealAIClient()
    
    # Order groceries
    groceries = client.tasks.order_groceries(
        items=["organic milk", "free-range eggs", "whole wheat bread"],
        execute=False  # Set to True to actually execute
    )
    print("\nGrocery Order:")
    print(json.dumps(groceries, indent=2))
    
    # Book an appointment
    appointment = client.tasks.book_appointment(
        details={
            "type": "dental checkup",
            "date": "2026-03-15",
            "time": "2:00 PM"
        },
        execute=False
    )
    print("\nAppointment Booking:")
    print(json.dumps(appointment, indent=2))


def example_voice_interaction():
    """Example: Natural voice conversations."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Voice Interaction - Talk Naturally!")
    print("="*70)
    
    client = RealAIClient()
    
    conversation = client.voice.conversation(
        message="Tell me about the future of AI",
        response_format="both"
    )
    
    print(json.dumps(conversation, indent=2))


def example_business_planning():
    """Example: Build businesses from the ground up."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Business Planning - Build Your Business!")
    print("="*70)
    
    client = RealAIClient()
    
    business_plan = client.business.build(
        business_type="AI-powered SaaS startup",
        stage="ideation"
    )
    
    print(json.dumps(business_plan, indent=2))


def example_therapy_counseling():
    """Example: Therapy and emotional support."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Therapy & Counseling - Get Support!")
    print("="*70)
    
    client = RealAIClient()
    
    # Emotional support
    support = client.therapy.support(
        message="I'm feeling stressed about work deadlines"
    )
    print("\nEmotional Support:")
    print(json.dumps(support, indent=2))
    
    # Full therapy session
    session = client.therapy.session(
        session_type="therapy",
        message="I want to work on managing my anxiety",
        approach="cognitive_behavioral"
    )
    print("\nTherapy Session:")
    print(json.dumps(session, indent=2))


def example_web3_integration():
    """Example: Web3 and blockchain capabilities."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Web3 Integration - Blockchain & Smart Contracts!")
    print("="*70)
    
    client = RealAIClient()
    
    # Deploy smart contract
    smart_contract = client.web3.smart_contract(
        blockchain="ethereum",
        params={"contract_type": "ERC20", "name": "MyToken"}
    )
    print("\nSmart Contract Deployment:")
    print(json.dumps(smart_contract, indent=2))
    
    # Query blockchain
    query = client.web3.execute(
        operation="query",
        blockchain="ethereum"
    )
    print("\nBlockchain Query:")
    print(json.dumps(query, indent=2))


def example_code_execution():
    """Example: Code generation AND execution."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Code Execution - Generate & Run Code!")
    print("="*70)
    
    client = RealAIClient()
    
    # Generate code
    code_gen = client.model.generate_code(
        prompt="Create a function to calculate prime numbers",
        language="python"
    )
    print("\nGenerated Code:")
    print(json.dumps(code_gen, indent=2))
    
    # Execute code
    execution = client.model.execute_code(
        code="""
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

for i in range(10):
    print(f"Fib({i}) = {fibonacci(i)}")
""",
        language="python",
        sandbox=True
    )
    print("\nCode Execution:")
    print(json.dumps(execution, indent=2))


def example_plugin_system():
    """Example: Plugin system for unlimited extensibility."""
    print("\n" + "="*70)
    print("EXAMPLE 8: Plugin System - Extend Anywhere!")
    print("="*70)
    
    client = RealAIClient()
    
    # Load a custom plugin
    plugin = client.plugins.load(
        plugin_name="custom_analytics",
        plugin_config={
            "api_key": "demo_key",
            "features": ["data_analysis", "visualization"]
        }
    )
    print("\nPlugin Loaded:")
    print(json.dumps(plugin, indent=2))
    
    # Extend with IoT integration
    iot_extension = client.plugins.extend(
        plugin_name="iot_integration",
        config={
            "devices": ["smart_home", "wearables"],
            "protocols": ["MQTT", "HTTP"]
        }
    )
    print("\nIoT Extension:")
    print(json.dumps(iot_extension, indent=2))


def example_learning_memory():
    """Example: Learning and adapting from interactions."""
    print("\n" + "="*70)
    print("EXAMPLE 9: Learning & Memory - Adaptive AI!")
    print("="*70)
    
    client = RealAIClient()
    
    learning = client.model.learn_from_interaction(
        interaction_data={
            "user_preferences": {
                "response_style": "concise",
                "technical_level": "advanced",
                "topics_of_interest": ["AI", "quantum computing", "startups"]
            },
            "context": "User is a technical founder",
            "feedback": "positive"
        },
        save=True
    )
    
    print(json.dumps(learning, indent=2))


def example_all_core_capabilities():
    """Example: All core AI capabilities in one place."""
    print("\n" + "="*70)
    print("EXAMPLE 10: Core AI Capabilities - Everything in One!")
    print("="*70)
    
    client = RealAIClient()
    
    # Chat
    chat = client.chat.create(
        messages=[{"role": "user", "content": "What makes you limitless?"}]
    )
    print("\n💬 Chat:")
    print(f"Response: {chat['choices'][0]['message']['content']}")
    
    # Images
    image = client.images.generate(prompt="A limitless AI in the clouds")
    print(f"\n🎨 Image: {image['data'][0]['url']}")
    
    # Code
    code = client.model.generate_code(prompt="Sort algorithm", language="python")
    print(f"\n💻 Code: {code['code'][:50]}...")
    
    # Embeddings
    emb = client.embeddings.create(input_text="RealAI is limitless")
    print(f"\n🔤 Embeddings: Dimension {len(emb['data'][0]['embedding'])}")
    
    # Audio
    audio = client.audio.generate(text="I have no limits!")
    print(f"\n🔊 Audio: {audio['audio_url']}")
    
    # Translation
    trans = client.model.translate(text="No limits", target_language="es")
    print(f"\n🌍 Translation: {trans['translated_text']}")


def main():
    """Run all examples showcasing RealAI's limitless capabilities."""
    print("="*70)
    print(" "*10 + "REALAI - THE LIMITLESS AI")
    print(" "*15 + "No Limits. Sky is the Limit!")
    print("="*70)
    
    examples = [
        example_web_research,
        example_task_automation,
        example_voice_interaction,
        example_business_planning,
        example_therapy_counseling,
        example_web3_integration,
        example_code_execution,
        example_plugin_system,
        example_learning_memory,
        example_all_core_capabilities
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\n❌ Error in {example.__name__}: {e}")
    
    print("\n" + "="*70)
    print("✅ All examples completed!")
    print("RealAI truly has no limits - from Web3 to groceries,")
    print("from therapy to building businesses from the ground up!")
    print("The sky is the limit! 🚀✨")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
