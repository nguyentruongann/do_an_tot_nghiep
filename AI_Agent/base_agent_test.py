# test_agents.py
import asyncio
import json
from agents.validate_agent import ValidationAgent
from agents.scoring_agent import ScoringAgent
from agents.explanation_agent import ExplanationAgent
from agents.communication_agent import CommunicationAgent
from agents.coordinator import MultiAgentCoordinator

# Test data
test_payload = {
    "ID": 1,
    "age": 58,
    "job": "management",
    "marital": 2,
    "education": "tertiary",
    "default": 0,
    "balance": 2143,
    "housing": 1,
    "loan": 1,
    "day": 5,
    "month": "may",
    "duration": 231,
    "campaign": 1,
    "pdays": -1,
    "previous": 0
}

async def test_validation_agent():
    """Test Validation Agent"""
    print("🧪 Testing Validation Agent...")
    
    agent = ValidationAgent()
    input_data = {"payload": test_payload}
    
    result = await agent.process(input_data)
    
    print(f"Status: {result.status.value}")
    print(f"Data: {json.dumps(result.data, indent=2, ensure_ascii=False)}")
    print(f"Confidence: {result.confidence}")
    print(f"Processing Time: {result.processing_time:.3f}s")
    print("-" * 50)
    
    return result

async def test_scoring_agent():
    """Test Scoring Agent"""
    print("🧪 Testing Scoring Agent...")
    
    agent = ScoringAgent()
    input_data = {"payload": test_payload}
    context = {
        "data_type": "bank_marketing",
        "validation_result": {
            "quality_score": 85.0,
            "missing_fields": []
        }
    }
    
    result = await agent.process(input_data, context)
    
    print(f"Status: {result.status.value}")
    print(f"Data: {json.dumps(result.data, indent=2, ensure_ascii=False)}")
    print(f"Confidence: {result.confidence}")
    print(f"Processing Time: {result.processing_time:.3f}s")
    print("-" * 50)
    
    return result

async def test_explanation_agent():
    """Test Explanation Agent"""
    print("🧪 Testing Explanation Agent...")
    
    agent = ExplanationAgent()
    input_data = {"payload": test_payload}
    context = {
        "scoring_result": {
            "score": 75.5,
            "grade": "B",
            "rationale": "Good customer profile",
            "contrib": [{"name": "duration", "impact": 0.4}],
            "confidence": 0.8
        },
        "validation_result": {
            "quality_score": 85.0,
            "missing_fields": []
        }
    }
    
    result = await agent.process(input_data, context)
    
    print(f"Status: {result.status.value}")
    print(f"Data: {json.dumps(result.data, indent=2, ensure_ascii=False)}")
    print(f"Confidence: {result.confidence}")
    print(f"Processing Time: {result.processing_time:.3f}s")
    print("-" * 50)
    
    return result

async def test_communication_agent():
    """Test Communication Agent"""
    print("🧪 Testing Communication Agent...")
    
    agent = CommunicationAgent()
    input_data = {"payload": test_payload}
    context = {
        "scoring_result": {
            "score": 75.5,
            "grade": "B",
            "confidence": 0.8
        },
        "explanation_result": {
            "user_explanation": "✅ Khách hàng có tiềm năng tốt",
            "recommendations": ["📞 Liên hệ trong 48h"]
        }
    }
    
    result = await agent.process(input_data, context)
    
    print(f"Status: {result.status.value}")
    print(f"Data: {json.dumps(result.data, indent=2, ensure_ascii=False)}")
    print(f"Confidence: {result.confidence}")
    print(f"Processing Time: {result.processing_time:.3f}s")
    print("-" * 50)
    
    return result

async def test_full_pipeline():
    """Test Full Multi-Agent Pipeline"""
    print("🧪 Testing Full Multi-Agent Pipeline...")
    
    coordinator = MultiAgentCoordinator()
    input_data = {
        "data_type": "bank_marketing",
        "payload": test_payload
    }
    
    result = await coordinator.process(input_data)
    
    print(f"Status: {result.status.value}")
    print(f"Overall Confidence: {result.confidence}")
    print(f"Processing Time: {result.processing_time:.3f}s")
    print("\n📊 Final Result:")
    print(json.dumps(result.data, indent=2, ensure_ascii=False))
    print("-" * 50)
    
    return result

async def test_health_status():
    """Test Health Status"""
    print("🧪 Testing Health Status...")
    
    coordinator = MultiAgentCoordinator()
    health = coordinator.get_agent_health_status()
    
    print("Health Status:")
    print(json.dumps(health, indent=2, ensure_ascii=False))
    print("-" * 50)

async def main():
    """Run all tests"""
    print("🚀 Starting Multi-Agent System Tests...\n")
    
    try:
        # Test individual agents
        await test_validation_agent()
        await test_scoring_agent()
        await test_explanation_agent()
        await test_communication_agent()
        
        # Test full pipeline
        await test_full_pipeline()
        
        # Test health status
        await test_health_status()
        
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())