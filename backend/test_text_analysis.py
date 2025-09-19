"""
Test suite for Text Analysis API using DistilBERT
Tests all endpoints and functionality
"""

import pytest
import asyncio
import httpx
import json
from typing import Dict, Any

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_TEXTS = [
    "This is a great product! I love using it every day. The quality is excellent and the service is outstanding.",
    "I hate this terrible product. It's awful and completely useless. Very disappointed with the purchase.",
    "The weather today is nice. It's sunny outside. I might go for a walk later.",
    "machine learning artificial intelligence data science algorithms neural networks deep learning",
    "This text has some grammar errors and it dont make much sense sometimes but overall readable",
    "",  # Empty text for testing validation
    "A" * 5000,  # Long text for testing limits
]

class TestTextAnalysisAPI:
    """Test suite for the Text Analysis API"""
    
    @pytest.fixture(scope="session")
    def event_loop(self):
        """Create an instance of the default event loop for the test session."""
        loop = asyncio.get_event_loop_policy().new_event_loop()
        yield loop
        loop.close()
    
    @pytest.fixture(scope="session")
    async def client(self):
        """Create an HTTP client for testing"""
        async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
            yield client
    
    async def test_root_endpoint(self, client: httpx.AsyncClient):
        """Test the root endpoint"""
        response = await client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "1.0.0"
    
    async def test_health_endpoint(self, client: httpx.AsyncClient):
        """Test the health check endpoint"""
        response = await client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_status" in data
        assert data["model_status"] in ["loaded", "mock"]
        assert "uptime" in data
        assert "timestamp" in data
    
    async def test_analyze_single_text_valid(self, client: httpx.AsyncClient):
        """Test analyzing a single valid text"""
        test_text = TEST_TEXTS[0]  # Positive text
        payload = {
            "text": test_text,
            "include_details": True
        }
        
        response = await client.post("/analyze", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        
        # Check response structure
        assert "sentiment" in data
        assert "grammar" in data
        assert "coherence" in data
        assert "overall_score" in data
        assert "analysis_id" in data
        assert "timestamp" in data
        
        # Check sentiment scores
        sentiment = data["sentiment"]
        assert 0 <= sentiment["positive"] <= 1
        assert 0 <= sentiment["negative"] <= 1
        assert 0 <= sentiment["overall"] <= 1
        
        # Check grammar scores
        grammar = data["grammar"]
        assert 0 <= grammar["grammar_score"] <= 1
        assert grammar["errors_detected"] >= 0
        
        # Check coherence scores
        coherence = data["coherence"]
        assert 0 <= coherence["coherence_score"] <= 1
        assert coherence["sentence_count"] >= 0
        
        # Check overall score
        assert 0 <= data["overall_score"] <= 1
    
    async def test_analyze_single_text_negative(self, client: httpx.AsyncClient):
        """Test analyzing negative text"""
        test_text = TEST_TEXTS[1]  # Negative text
        payload = {
            "text": test_text,
            "include_details": True
        }
        
        response = await client.post("/analyze", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        sentiment = data["sentiment"]
        
        # Negative text should have higher negative sentiment
        assert sentiment["negative"] > sentiment["positive"]
    
    async def test_analyze_empty_text(self, client: httpx.AsyncClient):
        """Test analyzing empty text"""
        payload = {
            "text": "",
            "include_details": True
        }
        
        response = await client.post("/analyze", json=payload)
        assert response.status_code == 422  # Validation error
    
    async def test_analyze_very_long_text(self, client: httpx.AsyncClient):
        """Test analyzing very long text"""
        long_text = "A" * 15000  # Exceeds limit
        payload = {
            "text": long_text,
            "include_details": True
        }
        
        response = await client.post("/analyze", json=payload)
        assert response.status_code == 422  # Validation error
    
    async def test_batch_analysis_valid(self, client: httpx.AsyncClient):
        """Test batch analysis with valid texts"""
        valid_texts = [text for text in TEST_TEXTS[:4] if text.strip()]
        payload = {
            "texts": valid_texts,
            "include_details": True
        }
        
        response = await client.post("/analyze/batch", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        
        # Check response structure
        assert "results" in data
        assert "total_texts" in data
        assert "batch_id" in data
        assert "timestamp" in data
        
        # Check results
        results = data["results"]
        assert len(results) == len(valid_texts)
        
        for i, result in enumerate(results):
            assert result["index"] == i
            assert "text_preview" in result
            assert "analysis" in result
            
            # Check analysis structure
            analysis = result["analysis"]
            assert "sentiment" in analysis
            assert "grammar" in analysis
            assert "coherence" in analysis
            assert "overall_score" in analysis
    
    async def test_batch_analysis_empty_list(self, client: httpx.AsyncClient):
        """Test batch analysis with empty list"""
        payload = {
            "texts": [],
            "include_details": True
        }
        
        response = await client.post("/analyze/batch", json=payload)
        assert response.status_code == 422  # Validation error
    
    async def test_batch_analysis_too_many_texts(self, client: httpx.AsyncClient):
        """Test batch analysis with too many texts"""
        too_many_texts = ["Test text"] * 60  # Exceeds limit of 50
        payload = {
            "texts": too_many_texts,
            "include_details": True
        }
        
        response = await client.post("/analyze/batch", json=payload)
        assert response.status_code == 422  # Validation error
    
    async def test_sentiment_only_endpoint(self, client: httpx.AsyncClient):
        """Test sentiment-only analysis endpoint"""
        test_text = TEST_TEXTS[0]
        
        response = await client.get(f"/analyze/sentiment?text={test_text}")
        assert response.status_code == 200
        
        data = response.json()
        assert "sentiment" in data
        assert "text_preview" in data
        assert "analysis_id" in data
        
        sentiment = data["sentiment"]
        assert 0 <= sentiment["positive"] <= 1
        assert 0 <= sentiment["negative"] <= 1
    
    async def test_grammar_only_endpoint(self, client: httpx.AsyncClient):
        """Test grammar-only analysis endpoint"""
        test_text = TEST_TEXTS[4]  # Text with grammar errors
        
        response = await client.get(f"/analyze/grammar?text={test_text}")
        assert response.status_code == 200
        
        data = response.json()
        assert "grammar" in data
        assert "text_preview" in data
        assert "analysis_id" in data
        
        grammar = data["grammar"]
        assert 0 <= grammar["grammar_score"] <= 1
        assert grammar["errors_detected"] >= 0
    
    async def test_coherence_only_endpoint(self, client: httpx.AsyncClient):
        """Test coherence-only analysis endpoint"""
        test_text = TEST_TEXTS[2]
        
        response = await client.get(f"/analyze/coherence?text={test_text}")
        assert response.status_code == 200
        
        data = response.json()
        assert "coherence" in data
        assert "text_preview" in data
        assert "analysis_id" in data
        
        coherence = data["coherence"]
        assert 0 <= coherence["coherence_score"] <= 1
        assert coherence["sentence_count"] >= 0
    
    async def test_stats_endpoint(self, client: httpx.AsyncClient):
        """Test API statistics endpoint"""
        response = await client.get("/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_analyses" in data
        assert "uptime" in data
        assert "model_status" in data
        assert "api_version" in data
        assert data["total_analyses"] >= 0
    
    async def test_missing_text_parameter(self, client: httpx.AsyncClient):
        """Test endpoints with missing text parameter"""
        response = await client.get("/analyze/sentiment")
        assert response.status_code == 422  # Missing required parameter
        
        response = await client.get("/analyze/grammar")
        assert response.status_code == 422
        
        response = await client.get("/analyze/coherence")
        assert response.status_code == 422
    
    def test_score_consistency(self):
        """Test that scores are consistent and within expected ranges"""
        # This would be run after other tests to ensure consistency
        pass

if __name__ == "__main__":
    # Run tests manually
    import asyncio
    
    async def run_manual_tests():
        """Run tests manually without pytest"""
        print("🧪 Running manual tests for Text Analysis API...")
        
        async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
            # Test health endpoint
            try:
                response = await client.get("/health")
                print(f"✅ Health check: {response.status_code}")
                print(f"   Response: {response.json()}")
            except Exception as e:
                print(f"❌ Health check failed: {e}")
            
            # Test single analysis
            try:
                test_text = "This is a great product! I love it."
                payload = {"text": test_text, "include_details": True}
                response = await client.post("/analyze", json=payload)
                print(f"✅ Single analysis: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"   Overall score: {data['overall_score']}")
                    print(f"   Sentiment: {data['sentiment']['overall']}")
                    print(f"   Grammar: {data['grammar']['grammar_score']}")
                    print(f"   Coherence: {data['coherence']['coherence_score']}")
                else:
                    print(f"   Error: {response.text}")
            except Exception as e:
                print(f"❌ Single analysis failed: {e}")
            
            # Test batch analysis
            try:
                texts = [
                    "Great product!",
                    "Terrible service.",
                    "The weather is nice today."
                ]
                payload = {"texts": texts, "include_details": True}
                response = await client.post("/analyze/batch", json=payload)
                print(f"✅ Batch analysis: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"   Processed {data['total_texts']} texts")
                    print(f"   Average processing time: {data['avg_processing_time']:.3f}s")
                else:
                    print(f"   Error: {response.text}")
            except Exception as e:
                print(f"❌ Batch analysis failed: {e}")
    
    print("🚀 Starting manual test runner...")
    print(f"📡 Testing API at: {API_BASE_URL}")
    print("⚠️  Make sure the API server is running!")
    print()
    
    asyncio.run(run_manual_tests())