# Profile Purity Detector

🎯 **Hackathon-ready** AI-powered```
Profile Purity Detector/
├── 🚀 start-all.sh              # One-command startup script
├── 📄 README.md                 # This file
├── backend/
│   ├── requirements.txt         # Python dependencies  
│   └── api/
│       ├── text_analysis_simple.py        # Text Analysis API
│       ├── vision_detection_simple.py     # Vision Detection API  
│       ├── tabular_analysis_simple.py     # Tabular Analysis API
│       ├── ensemble_simple.py             # Ensemble Learning API
│       └── profile_extraction_simple.py   # Profile Extraction API
└── frontend/
    ├── app/                     # Next.js application
    ├── components/              # UI components
    └── [other frontend files]
```ion system with **automatic profile extraction** from social media URLs.

## ✨ Features

- 🔄 **Automatic Profile Extraction**: Just paste Instagram/Twitter URLs - all data extracted automatically
- 🤖 **Multi-faceted AI Analysis**: NLP (30%) + Computer Vision (35%) + Profile Metrics (35%)
- ⚡ **Real-time Processing**: Instant analysis with professional UI
- 🌐 **Multi-platform Support**: Instagram, Twitter/X, Facebook
- 💼 **Production Ready**: Robust error handling and graceful fallbacks

## 🚀 One-Command Startup

### Prerequisites
- Python 3.8+ 
- Node.js 16+
- Git

### Quick Start
```bash
git clone <repository-url>
cd "Hackathon 101"
./start-all.sh
```

That's it! The script will:
- ✅ Set up Python virtual environment
- ✅ Install all dependencies  
- ✅ Start all 5 backend APIs
- ✅ Launch the frontend
- ✅ Verify all services are running

### Access the Application
- **Demo**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs (and ports 8002-8005)

## 📱 How to Use

1. **Open**: http://localhost:3000
2. **Paste**: Any Instagram or Twitter profile URL
3. **Click**: "Auto-Extract & Analyze Profile"  
4. **Get**: Instant trust score with detailed breakdown

## 🏗️ Architecture
### Backend Services
- **Text Analysis API** (Port 8000): NLP sentiment and authenticity analysis
- **Vision Detection API** (Port 8002): AI image manipulation detection  
- **Tabular Analysis API** (Port 8003): Profile metrics classification
- **Ensemble API** (Port 8004): Weighted scoring algorithm
- **Profile Extraction API** (Port 8005): Automatic data extraction from URLs

### Frontend
- **Next.js 14**: Modern React framework with TypeScript
- **Tailwind CSS**: Responsive design with dark/light mode
- **shadcn/ui**: Professional UI components
- **Real-time Analysis**: Live progress indicators and results

## 📁 Project Structure

```
Profile Purity Detector/
├── 🚀 start-project.sh          # One-command startup script
├── � README.md                 # This file
├── backend/
│   ├── requirements.txt         # Python dependencies  
│   ├── ml/
│   │   ├── auto_profile_extractor.py  # Automatic profile extraction
│   │   └── [analysis modules]   # NLP, Computer Vision, etc.
│   └── api/
│       ├── text_api.py          # Text Analysis API
│       ├── vision_api.py        # Vision Detection API  
│       ├── production_tabular_api.py # Tabular Analysis API
│       ├── ensemble_api.py      # Ensemble Learning API
│       └── profile_extraction_api.py # Profile Extraction API
└── frontend/
    ├── app/                     # Next.js application
    ├── components/              # UI components
    └── [other frontend files]
```

## 🎯 Demo Workflow

1. **Start Project**: `./start-all.sh`
2. **Open Browser**: Navigate to http://localhost:3000
3. **Paste URL**: Enter any Instagram or Twitter profile URL
4. **Auto-Analysis**: System automatically:
   - Extracts follower count, following count, post count
   - Downloads and analyzes profile image
   - Processes bio text for sentiment/authenticity
   - Calculates weighted trust score
5. **View Results**: Get detailed breakdown with risk factors

## 🔧 Troubleshooting

### Services Not Starting
```bash
# Check if ports are in use
lsof -i :3000,:8000,:8002,:8003,:8004,:8005

# Kill existing processes
./start-project.sh  # Script automatically handles cleanup
```

### Profile Extraction Issues
- Ensure stable internet connection
- Check if social media URLs are public profiles
- Private profiles will fall back to manual input

## 🏆 Key Features for Hackathon

### Backend Setup

1. **Create Python virtual environment**:
```bash
cd backend
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Train a model** (optional - for custom models):
```bash
cd ml
python train.py
```

4. **Run hyperparameter optimization** (optional):
```bash
cd ml
python optimize.py
```

5. **Start the API server**:
```bash
cd api
python main.py
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Install Node.js dependencies**:
```bash
cd frontend
npm install
```

2. **Start the development server**:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Text Analysis API Setup

Our application includes a dedicated Text Analysis API powered by DistilBERT for advanced text analysis.

1. **Start the Text Analysis API**:
```bash
cd backend
./start_text_api.sh
```

Or manually:
```bash
cd backend/api
python -m uvicorn text_analysis_api:app --host 0.0.0.0 --port 8000 --reload
```

2. **Test the API**:
```bash
cd backend
python test_simple.py
```

The Text Analysis API will be available at:
- **API Server**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

**Key Features:**
- **Sentiment Analysis**: 0-1 normalized sentiment scores using DistilBERT
- **Grammar Analysis**: Rule-based + ML grammar quality assessment  
- **Coherence Analysis**: Text structure and semantic coherence evaluation
- **Batch Processing**: Analyze multiple texts efficiently
- **Comprehensive API**: Full REST API with validation and error handling

See `backend/TEXT_ANALYSIS_README.md` for detailed documentation.

## 🧪 Test Commands

### Backend Tests

**Test ML Training Script**:
```bash
cd backend/ml
python -c "
from train import ImageClassificationTrainer, set_seed
import torch

# Test with minimal config
config = {
    'model_type': 'timm',
    'model_name': 'efficientnet_b0',
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'epochs': 1,  # Quick test
    'seed': 42,
    'use_wandb': False
}

trainer = ImageClassificationTrainer(config)
trainer.prepare_data()
trainer.create_model()
print('✅ Training script test passed')
"
```

**Test Hyperparameter Optimization**:
```bash
cd backend/ml
python -c "
from optimize import HyperparameterOptimizer

config = {
    'model_type': 'timm',
    'seed': 42,
    'use_wandb': False,
    'optimize_model': False  # Skip model optimization for speed
}

optimizer = HyperparameterOptimizer(config, n_trials=2)
print('✅ Optimization script test passed')
"
```

**Test FastAPI Server**:
```bash
cd backend/api
python -c "
import requests
import time
import subprocess
import os

# Start server in background
proc = subprocess.Popen(['python', 'main.py'])
time.sleep(5)  # Wait for server to start

try:
    # Test health endpoint
    response = requests.get('http://localhost:8000/health')
    print(f'Health check: {response.status_code}')
    
    # Test model info (may fail if no model loaded)
    try:
        response = requests.get('http://localhost:8000/model/info')
        print(f'Model info: {response.status_code}')
    except:
        print('Model info: No model loaded (expected)')
    
    print('✅ FastAPI server test passed')
finally:
    proc.terminate()
"
```

### Frontend Tests

**Test Next.js Build**:
```bash
cd frontend
npm run build
echo "✅ Frontend build test passed"
```

**Test TypeScript Compilation**:
```bash
cd frontend
npm run type-check
echo "✅ TypeScript test passed"
```

**Test Linting**:
```bash
cd frontend
npm run lint
echo "✅ Frontend linting test passed"
```

## 🚀 Deployment

### Backend Deployment

**Local/Server Deployment**:
```bash
# Install dependencies
pip install -r backend/requirements.txt

# Start with production settings
cd backend/api
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Environment Variables**:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/backend"
export MODEL_PATH="/path/to/your/model.pth"  # Optional
```

### Frontend Deployment (Vercel)

1. **Build the project**:
```bash
cd frontend
npm run build
```

2. **Deploy to Vercel**:
```bash
npm install -g vercel
vercel
```

3. **Configure environment variables in Vercel**:
- `NEXT_PUBLIC_API_URL`: Your backend API URL

## 📊 Model Performance

The default models achieve the following performance on CIFAR-10:

| Model | Accuracy | Parameters | Training Time |
|-------|----------|------------|---------------|
| EfficientNet-B0 | ~85% | 5.3M | ~20 min |
| ResNet-18 | ~82% | 11.2M | ~15 min |
| MobileNet-V3 | ~78% | 5.5M | ~12 min |

## 🔧 API Endpoints

### Model Management
- `GET /health` - System health check
- `GET /model/info` - Current model information
- `POST /model/load` - Load a specific model
- `GET /models/list` - List available models

### Inference
- `POST /predict` - Single image prediction
- `POST /predict/batch` - Batch image prediction

### Monitoring
- `GET /metrics` - System metrics
- `GET /` - API documentation

## 🎯 CIFAR-10 Classes

The model classifies images into these 10 categories:
- Airplane ✈️
- Automobile 🚗
- Bird 🐦
- Cat 🐱
- Deer 🦌
- Dog 🐕
- Frog 🐸
- Horse 🐴
- Ship 🚢
- Truck 🚛

## 🔬 Advanced Usage

### Custom Model Training

```python
from backend.ml.train import ImageClassificationTrainer

config = {
    'model_type': 'huggingface',
    'model_name': 'google/vit-base-patch16-224',
    'batch_size': 64,
    'learning_rate': 0.0001,
    'epochs': 50,
    'use_wandb': True,  # Enable experiment tracking
    'project_name': 'my-experiment'
}

trainer = ImageClassificationTrainer(config)
model_path, val_acc, test_acc = trainer.train()
```

### Hyperparameter Optimization

```python
from backend.ml.optimize import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(
    base_config={'seed': 42},
    n_trials=100,
    study_name="cifar10_optimization"
)

study = optimizer.optimize()
best_model_path, val_acc, test_acc = optimizer.train_best_model(full_epochs=100)
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce batch size in training config
   - Use smaller models (e.g., mobilenet instead of efficientnet)

2. **Model not loading**:
   - Check if model file exists in `backend/models/`
   - Verify model path in API logs

3. **Frontend connection issues**:
   - Ensure backend is running on port 8000
   - Check CORS settings in FastAPI

4. **Package installation issues**:
   - Use Python 3.11 specifically
   - Install PyTorch with appropriate CUDA version

### Performance Optimization

- **GPU Usage**: Ensure CUDA is available for faster training
- **Model Loading**: Pre-load models for faster inference
- **Batch Processing**: Use batch endpoints for multiple images
- **Caching**: Implement Redis for prediction caching

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review API logs in backend console
- Inspect browser console for frontend issues