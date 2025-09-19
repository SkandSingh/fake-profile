#!/bin/bash

# Text Analysis API Startup Script
# Starts the FastAPI server for text analysis using DistilBERT

echo "🚀 Starting Text Analysis API..."

# Check if we're in the correct directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found. Please run from the backend directory."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if models need to be downloaded
echo "🤖 Checking AI models..."
python3 -c "
import sys
try:
    from transformers import pipeline
    # Try to load a small model to check if transformers is working
    print('✅ Transformers library is working')
    print('📚 Models will be downloaded automatically on first use')
except Exception as e:
    print(f'⚠️  Warning: {e}')
    print('🔄 API will run in mock mode without transformers')
"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start the API server
echo "🌐 Starting FastAPI server on http://localhost:8000"
echo "📖 API Documentation available at http://localhost:8000/docs"
echo "🔍 Health check at http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"

# Run the server
cd api
python3 -m uvicorn text_analysis_api:app --host 0.0.0.0 --port 8000 --reload