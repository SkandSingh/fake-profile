#!/bin/bash

# Quick setup script for the ML project

echo "🚀 Setting up ML Full-Stack Application..."

# Backend setup
echo "📦 Setting up Python backend..."
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3.11 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Backend setup complete!"

# Frontend setup
echo "📦 Setting up Next.js frontend..."
cd ../frontend

# Install dependencies
echo "Installing Node.js dependencies..."
npm install

echo "✅ Frontend setup complete!"

# Back to root
cd ..

echo "🎉 Setup complete!"
echo ""
echo "To start the application:"
echo "1. Start backend: cd backend && source venv/bin/activate && cd api && python main.py"
echo "2. Start frontend: cd frontend && npm run dev"
echo ""
echo "Then visit http://localhost:3000"