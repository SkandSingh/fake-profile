#!/bin/bash

# 🚀 Profile Purity Detector - Complete System Startup
# One command to start all services: backend APIs + frontend

set -e
PROJECT_DIR="/Users/skand/Dev/Projects/Hackathon 101"
cd "$PROJECT_DIR"

echo "🚀 Starting Complete Profile Purity Detector System"
echo "=================================================="

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🧹 Shutting down all services..."
    pkill -f "uvicorn" 2>/dev/null || true
    pkill -f "next.*dev" 2>/dev/null || true
    pkill -f "text_analysis_simple" 2>/dev/null || true
    pkill -f "vision_detection_simple" 2>/dev/null || true
    pkill -f "profile_extraction_simple" 2>/dev/null || true
    echo "✅ All services stopped"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Clean up any existing processes
echo "🧹 Cleaning up existing processes..."
pkill -f "uvicorn" 2>/dev/null || true
pkill -f "next.*dev" 2>/dev/null || true
sleep 2

# Setup Python environment
echo "🐍 Setting up Python environment..."
if [ ! -d "venv" ]; then
    echo "   📦 Creating virtual environment..."
    python3 -m venv venv
fi

echo "   🔧 Activating virtual environment..."
source venv/bin/activate

echo "   📚 Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet fastapi uvicorn python-multipart aiohttp pydantic
pip install --quiet requests beautifulsoup4 lxml Pillow || {
    echo "   ⚠️  Some packages failed to install, using fallback implementations"
}

echo "   ✅ Python environment ready"

# Start all backend services
echo ""
echo "🛠️  Starting backend services..."
cd backend/api

echo "   📊 Starting Tabular API (port 8003)..."
nohup python3 -m uvicorn tabular_classification_api:app --host 127.0.0.1 --port 8003 > ../tabular_api.log 2>&1 &
sleep 1

echo "   🧠 Starting Ensemble API (port 8004)..."
nohup python3 -m uvicorn ensemble_api:app --host 127.0.0.1 --port 8004 > ../ensemble_api.log 2>&1 &
sleep 1

echo "   🔤 Starting Text Analysis API (port 8000)..."
nohup python3 text_analysis_simple.py > ../text_api.log 2>&1 &
sleep 1

echo "   👁️  Starting Vision Detection API (port 8002)..."
nohup python3 vision_detection_simple.py > ../vision_api.log 2>&1 &
sleep 1

echo "   🔍 Starting Profile Extraction API (port 8005)..."
nohup python3 profile_extraction_simple.py > ../extraction_api.log 2>&1 &
sleep 1

cd ../..

# Start frontend
echo ""
echo "⚛️  Starting frontend..."
cd frontend

echo "   📦 Installing npm dependencies (if needed)..."
if [ ! -d "node_modules" ]; then
    npm install --silent
fi

echo "   🌐 Starting Next.js frontend (port 3000)..."
nohup npm run dev > ../frontend.log 2>&1 &
sleep 3

cd ..

# Wait for services to initialize
echo ""
echo "⏳ Waiting for all services to start (10 seconds)..."
sleep 10

# Check service health
echo ""
echo "🔍 Checking service health..."

services_running=0
total_services=6

check_service() {
    local port=$1
    local name=$2
    local endpoint=${3:-""}
    
    if curl -s -f "http://localhost:$port$endpoint" > /dev/null 2>&1; then
        echo "   ✅ $name (port $port)"
        return 0
    else
        echo "   ❌ $name (port $port)"
        return 1
    fi
}

# Check each service
check_service 8000 "Text Analysis API" "/health" && services_running=$((services_running + 1))
check_service 8002 "Vision Detection API" "/health" && services_running=$((services_running + 1))
check_service 8003 "Tabular Classification API" "/health" && services_running=$((services_running + 1))
check_service 8004 "Ensemble Learning API" "/health" && services_running=$((services_running + 1))
check_service 8005 "Profile Extraction API" "/health" && services_running=$((services_running + 1))
check_service 3000 "Frontend Application" "" && services_running=$((services_running + 1))

echo ""
echo "📊 Service Status: $services_running/$total_services services running"

if [ $services_running -ge 4 ]; then
    echo ""
    echo "🎉 SUCCESS! Profile Purity Detector is running!"
    echo ""
    echo "🌐 Access your application:"
    echo "   📱 Frontend:     http://localhost:3000"
    echo ""
    echo "📋 API Documentation:"
    echo "   🔤 Text API:     http://localhost:8000/docs"
    echo "   👁️  Vision API:   http://localhost:8002/docs"
    echo "   📊 Tabular API:  http://localhost:8003/docs"
    echo "   🧠 Ensemble API: http://localhost:8004/docs"
    echo "   🔍 Extract API:  http://localhost:8005/docs"
    echo ""
    echo "✨ Features available:"
    echo "   • Automatic profile extraction from URLs"
    echo "   • AI-powered text analysis"
    echo "   • Computer vision fake detection"
    echo "   • Ensemble learning predictions"
    echo "   • Real-time confidence scoring"
    echo ""
    echo "🎯 How to use:"
    echo "   1. Open http://localhost:3000 in your browser"
    echo "   2. Paste any social media profile URL"
    echo "   3. Click 'Auto-Extract & Analyze Profile'"
    echo "   4. Get instant fake profile detection results!"
    echo ""
    echo "⏹️  Press Ctrl+C to stop all services"
    echo ""
    
    # Keep script running and monitor services
    while true; do
        sleep 30
        # Quick health check every 30 seconds
        if ! curl -s -f "http://localhost:3000" > /dev/null 2>&1; then
            echo "⚠️  Frontend may have stopped, but backend APIs are still running"
        fi
    done
else
    echo ""
    echo "❌ Not enough services started successfully"
    echo ""
    echo "🔍 Check logs for details:"
    echo "   📁 Backend logs: backend/*.log"
    echo "   📁 Frontend log: frontend.log"
    echo ""
    echo "💡 Try running individual services manually:"
    echo "   cd backend/api && python3 text_analysis_simple.py"
    echo "   cd frontend && npm run dev"
    
    cleanup
fi