#!/usr/bin/env python3
"""
Simple test script for Text Analysis API
Tests basic functionality without requiring pytest
"""

import requests
import json
import time
from typing import Dict, Any, List

# Configuration
API_BASE_URL = "http://localhost:8000"
TIMEOUT = 30

def test_health_check() -> bool:
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data['status']}")
            print(f"   Model status: {data['model_status']}")
            print(f"   Uptime: {data['uptime']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_single_analysis() -> bool:
    """Test single text analysis"""
    try:
        test_text = "This is a fantastic product! I absolutely love using it every day. The quality is exceptional and the customer service is outstanding."
        payload = {
            "text": test_text,
            "include_details": True
        }
        
        print(f"🔍 Testing single analysis...")
        print(f"   Text: {test_text[:50]}...")
        
        response = requests.post(f"{API_BASE_URL}/analyze", json=payload, timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Single analysis passed")
            print(f"   Overall score: {data['overall_score']:.3f}")
            print(f"   Sentiment (positive): {data['sentiment']['positive']:.3f}")
            print(f"   Grammar score: {data['grammar']['grammar_score']:.3f}")
            print(f"   Coherence score: {data['coherence']['coherence_score']:.3f}")
            print(f"   Processing time: {data['processing_time']:.3f}s")
            print(f"   Model status: {data['model_status']}")
            return True
        else:
            print(f"❌ Single analysis failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Single analysis error: {e}")
        return False

def test_batch_analysis() -> bool:
    """Test batch text analysis"""
    try:
        test_texts = [
            "I love this amazing product! It's absolutely wonderful.",
            "This is terrible. I hate it completely. Very disappointing.",
            "The weather today is nice. It's sunny outside and perfect for a walk.",
            "Machine learning and artificial intelligence are fascinating fields of study."
        ]
        
        payload = {
            "texts": test_texts,
            "include_details": True
        }
        
        print(f"📊 Testing batch analysis with {len(test_texts)} texts...")
        
        response = requests.post(f"{API_BASE_URL}/analyze/batch", json=payload, timeout=TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch analysis passed")
            print(f"   Total texts processed: {data['total_texts']}")
            print(f"   Average processing time: {data['avg_processing_time']:.3f}s")
            print(f"   Total processing time: {data['total_processing_time']:.3f}s")
            
            # Show results for each text
            for i, result in enumerate(data['results'][:2]):  # Show first 2 results
                analysis = result['analysis']
                print(f"   Text {i+1}: Overall={analysis['overall_score']:.3f}, "
                      f"Sentiment={analysis['sentiment']['overall']:.3f}")
            
            return True
        else:
            print(f"❌ Batch analysis failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Batch analysis error: {e}")
        return False

def test_individual_endpoints() -> bool:
    """Test individual analysis endpoints"""
    try:
        test_text = "This is a great example of well-written text with good grammar and clear meaning."
        
        # Test sentiment endpoint
        response = requests.get(f"{API_BASE_URL}/analyze/sentiment", 
                              params={"text": test_text}, timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Sentiment analysis: {data['sentiment']['overall']:.3f}")
        else:
            print(f"❌ Sentiment analysis failed: {response.status_code}")
            return False
        
        # Test grammar endpoint
        response = requests.get(f"{API_BASE_URL}/analyze/grammar", 
                              params={"text": test_text}, timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Grammar analysis: {data['grammar']['grammar_score']:.3f}")
        else:
            print(f"❌ Grammar analysis failed: {response.status_code}")
            return False
        
        # Test coherence endpoint
        response = requests.get(f"{API_BASE_URL}/analyze/coherence", 
                              params={"text": test_text}, timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Coherence analysis: {data['coherence']['coherence_score']:.3f}")
        else:
            print(f"❌ Coherence analysis failed: {response.status_code}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Individual endpoints error: {e}")
        return False

def test_validation_errors() -> bool:
    """Test validation and error handling"""
    try:
        print(f"🔒 Testing validation...")
        
        # Test empty text
        payload = {"text": "", "include_details": True}
        response = requests.post(f"{API_BASE_URL}/analyze", json=payload, timeout=TIMEOUT)
        if response.status_code == 422:
            print(f"✅ Empty text validation works")
        else:
            print(f"❌ Empty text validation failed: {response.status_code}")
            return False
        
        # Test too long text
        long_text = "A" * 15000  # Exceeds 10,000 character limit
        payload = {"text": long_text, "include_details": True}
        response = requests.post(f"{API_BASE_URL}/analyze", json=payload, timeout=TIMEOUT)
        if response.status_code == 422:
            print(f"✅ Long text validation works")
        else:
            print(f"❌ Long text validation failed: {response.status_code}")
            return False
        
        # Test empty batch
        payload = {"texts": [], "include_details": True}
        response = requests.post(f"{API_BASE_URL}/analyze/batch", json=payload, timeout=TIMEOUT)
        if response.status_code == 422:
            print(f"✅ Empty batch validation works")
        else:
            print(f"❌ Empty batch validation failed: {response.status_code}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Validation test error: {e}")
        return False

def test_api_stats() -> bool:
    """Test API statistics endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API stats retrieved")
            print(f"   Total analyses: {data['total_analyses']}")
            print(f"   Uptime: {data['uptime']}")
            print(f"   Model status: {data['model_status']}")
            return True
        else:
            print(f"❌ API stats failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API stats error: {e}")
        return False

def run_performance_test() -> bool:
    """Test API performance with multiple requests"""
    try:
        print(f"⚡ Running performance test...")
        
        test_texts = [
            "Excellent product quality!",
            "Terrible customer service.",
            "Average experience overall.",
            "Outstanding delivery speed.",
            "Poor value for money."
        ]
        
        start_time = time.time()
        
        for i, text in enumerate(test_texts):
            payload = {"text": text, "include_details": True}
            response = requests.post(f"{API_BASE_URL}/analyze", json=payload, timeout=TIMEOUT)
            if response.status_code != 200:
                print(f"❌ Performance test failed at request {i+1}")
                return False
        
        total_time = time.time() - start_time
        avg_time = total_time / len(test_texts)
        
        print(f"✅ Performance test passed")
        print(f"   {len(test_texts)} requests in {total_time:.2f}s")
        print(f"   Average: {avg_time:.3f}s per request")
        print(f"   Throughput: {len(test_texts)/total_time:.1f} requests/sec")
        
        return True
    except Exception as e:
        print(f"❌ Performance test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Text Analysis API Test Suite")
    print("=" * 50)
    print(f"📡 Testing API at: {API_BASE_URL}")
    print(f"⏱️  Timeout: {TIMEOUT}s")
    print()
    
    # Check if API is accessible
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print(f"❌ API not accessible at {API_BASE_URL}")
            print("   Make sure the server is running with: python -m uvicorn api.text_analysis_api:app --reload")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("   Make sure the server is running with: python -m uvicorn api.text_analysis_api:app --reload")
        return
    
    print("🚀 API is accessible, starting tests...")
    print()
    
    tests = [
        ("Health Check", test_health_check),
        ("Single Analysis", test_single_analysis),
        ("Batch Analysis", test_batch_analysis),
        ("Individual Endpoints", test_individual_endpoints),
        ("Validation & Errors", test_validation_errors),
        ("API Statistics", test_api_stats),
        ("Performance Test", run_performance_test),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"🔍 {test_name}...")
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The Text Analysis API is working correctly.")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Check the output above for details.")
    
    print()
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")

if __name__ == "__main__":
    main()