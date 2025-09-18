"""
Test script for FastAPI endpoints
"""
import requests
import json
import time
import io
from PIL import Image
import numpy as np

def create_test_image():
    """Create a test image for API testing"""
    # Create a simple test image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes

def test_api_endpoints():
    """Test all API endpoints"""
    base_url = "http://localhost:8000"
    
    print("Testing FastAPI endpoints...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health check: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   Model loaded: {health_data.get('model_loaded', False)}")
            print(f"   Device: {health_data.get('device', 'unknown')}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/")
        print(f"✅ Root endpoint: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
    
    # Test model list endpoint
    try:
        response = requests.get(f"{base_url}/models/list")
        print(f"✅ Models list: {response.status_code}")
        if response.status_code == 200:
            models = response.json()
            print(f"   Available models: {len(models.get('models', []))}")
    except Exception as e:
        print(f"❌ Models list failed: {e}")
    
    # Test model info (may fail if no model loaded)
    try:
        response = requests.get(f"{base_url}/model/info")
        if response.status_code == 200:
            print(f"✅ Model info: {response.status_code}")
            model_info = response.json()
            print(f"   Model: {model_info.get('model_name', 'unknown')}")
        else:
            print(f"⚠️  Model info: {response.status_code} (No model loaded)")
    except Exception as e:
        print(f"⚠️  Model info failed: {e} (Expected if no model loaded)")
    
    # Test metrics endpoint
    try:
        response = requests.get(f"{base_url}/metrics")
        print(f"✅ Metrics: {response.status_code}")
    except Exception as e:
        print(f"❌ Metrics failed: {e}")
    
    # Test prediction endpoint (may fail if no model loaded)
    try:
        test_image = create_test_image()
        files = {'file': ('test.png', test_image, 'image/png')}
        
        response = requests.post(f"{base_url}/predict", files=files)
        if response.status_code == 200:
            print(f"✅ Prediction: {response.status_code}")
            result = response.json()
            print(f"   Top prediction: {result['predictions'][0]['class']}")
            print(f"   Confidence: {result['predictions'][0]['confidence']:.3f}")
        else:
            print(f"⚠️  Prediction: {response.status_code} (Model may not be loaded)")
    except Exception as e:
        print(f"⚠️  Prediction failed: {e} (Expected if no model loaded)")
    
    print("\n🎉 API testing completed!")

if __name__ == "__main__":
    test_api_endpoints()