"""
Production-ready FastAPI server for ML model inference
Includes proper error handling, validation, and monitoring
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import timm
from transformers import AutoModelForImageClassification, AutoImageProcessor
import numpy as np
import io
import logging
import time
from typing import List, Dict, Optional, Union
from pathlib import Path
import json
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model
model = None
model_config = None
image_processor = None
transforms_pipeline = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Thread pool for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=4)

class PredictionRequest(BaseModel):
    """Request model for batch predictions"""
    image_urls: List[str] = Field(..., description="List of image URLs to predict")
    top_k: int = Field(default=5, ge=1, le=10, description="Number of top predictions to return")

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    model_config = {"protected_namespaces": ()}
    
    predictions: List[Dict[str, Union[str, float]]]
    confidence_score: float
    processing_time_ms: float
    model_info: Dict[str, str]

class HealthResponse(BaseModel):
    """Response model for health check"""
    model_config = {"protected_namespaces": ()}
    
    status: str
    model_loaded: bool
    device: str
    memory_usage: Optional[Dict[str, float]] = None

class ModelInfo(BaseModel):
    """Model information response"""
    model_config = {"protected_namespaces": ()}
    
    model_name: str
    model_type: str
    num_classes: int
    device: str
    classes: List[str]

def load_model(model_path: str = None):
    """Load the trained model"""
    global model, model_config, image_processor, transforms_pipeline
    
    if model_path is None:
        # Try to find the latest model
        models_dir = Path("./models")
        if not models_dir.exists():
            raise FileNotFoundError("Models directory not found")
        
        model_files = list(models_dir.glob("*.pth"))
        if not model_files:
            raise FileNotFoundError("No model files found")
        
        # Get the most recent model
        model_path = max(model_files, key=lambda x: x.stat().st_mtime)
    
    logger.info(f"Loading model from {model_path}")
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['config']
    
    # Create model based on config
    if model_config['model_type'] == 'timm':
        model = timm.create_model(
            model_config['model_name'],
            pretrained=False,
            num_classes=10
        )
    elif model_config['model_type'] == 'huggingface':
        model = AutoModelForImageClassification.from_pretrained(
            model_config['model_name'],
            num_labels=10,
            ignore_mismatched_sizes=True
        )
        image_processor = AutoImageProcessor.from_pretrained(model_config['model_name'])
    else:
        raise ValueError(f"Unsupported model type: {model_config['model_type']}")
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create transforms pipeline
    transforms_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    logger.info(f"Model loaded successfully on {device}")
    logger.info(f"Model: {model_config['model_name']} ({model_config['model_type']})")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Starting up ML API server...")
    try:
        load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Don't fail startup, allow manual model loading
    
    yield
    
    # Shutdown
    logger.info("Shutting down ML API server...")
    executor.shutdown(wait=True)

# Create FastAPI app
app = FastAPI(
    title="ML Model Inference API",
    description="Production-ready API for image classification with PyTorch models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to check if model is loaded
async def get_model():
    """Dependency to ensure model is loaded"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please load a model first.")
    return model

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model inference"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    if model_config['model_type'] == 'huggingface' and image_processor:
        # Use HuggingFace processor
        inputs = image_processor(image, return_tensors="pt")
        return inputs['pixel_values'].to(device)
    else:
        # Use torchvision transforms
        tensor = transforms_pipeline(image).unsqueeze(0)
        return tensor.to(device)

def predict_image(image_tensor: torch.Tensor, top_k: int = 5) -> List[Dict[str, Union[str, float]]]:
    """Make prediction on preprocessed image tensor"""
    with torch.no_grad():
        if model_config['model_type'] == 'huggingface':
            outputs = model(pixel_values=image_tensor)
            logits = outputs.logits
        else:
            logits = model(image_tensor)
        
        # Apply softmax to get probabilities
        probabilities = F.softmax(logits, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
        
        predictions = []
        for i in range(top_k):
            predictions.append({
                'class': CIFAR10_CLASSES[top_indices[0][i].item()],
                'confidence': float(top_probs[0][i].item())
            })
        
        return predictions

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "ML Model Inference API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    memory_usage = None
    
    if torch.cuda.is_available():
        memory_usage = {
            "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
            "cached_mb": torch.cuda.memory_reserved() / 1024 / 1024
        }
    
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=str(device),
        memory_usage=memory_usage
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info(current_model=Depends(get_model)):
    """Get information about the loaded model"""
    return ModelInfo(
        model_name=model_config['model_name'],
        model_type=model_config['model_type'],
        num_classes=10,
        device=str(device),
        classes=CIFAR10_CLASSES
    )

@app.post("/model/load")
async def load_model_endpoint(model_path: Optional[str] = None):
    """Load a model from file"""
    try:
        load_model(model_path)
        return {"message": "Model loaded successfully", "model_name": model_config['model_name']}
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_single_image(
    file: UploadFile = File(...),
    top_k: int = 5,
    current_model=Depends(get_model)
):
    """Predict on a single uploaded image"""
    start_time = time.time()
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess image
        image_tensor = preprocess_image(image)
        
        # Make prediction
        predictions = predict_image(image_tensor, top_k)
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            predictions=predictions,
            confidence_score=predictions[0]['confidence'],
            processing_time_ms=processing_time,
            model_info={
                "model_name": model_config['model_name'],
                "model_type": model_config['model_type']
            }
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch_images(
    files: List[UploadFile] = File(...),
    top_k: int = 5,
    current_model=Depends(get_model)
):
    """Predict on multiple uploaded images"""
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 10 images")
    
    results = []
    
    for file in files:
        start_time = time.time()
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            continue  # Skip non-image files
        
        try:
            # Read and process image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            
            # Preprocess image
            image_tensor = preprocess_image(image)
            
            # Make prediction
            predictions = predict_image(image_tensor, top_k)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            results.append(PredictionResponse(
                predictions=predictions,
                confidence_score=predictions[0]['confidence'],
                processing_time_ms=processing_time,
                model_info={
                    "model_name": model_config['model_name'],
                    "model_type": model_config['model_type']
                }
            ))
            
        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {e}")
            continue  # Skip failed images
    
    return results

@app.get("/models/list")
async def list_available_models():
    """List all available trained models"""
    models_dir = Path("./models")
    if not models_dir.exists():
        return {"models": []}
    
    models = []
    for model_file in models_dir.glob("*.pth"):
        try:
            # Get model info without loading
            checkpoint = torch.load(model_file, map_location='cpu')
            model_info = {
                "filename": model_file.name,
                "model_name": checkpoint['config']['model_name'],
                "model_type": checkpoint['config']['model_type'],
                "created": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat(),
                "size_mb": round(model_file.stat().st_size / 1024 / 1024, 2)
            }
            models.append(model_info)
        except Exception as e:
            logger.warning(f"Could not read model {model_file}: {e}")
    
    return {"models": models}

@app.get("/metrics")
async def get_metrics():
    """Get basic metrics for monitoring"""
    metrics = {
        "model_loaded": model is not None,
        "device": str(device),
        "timestamp": datetime.now().isoformat()
    }
    
    if torch.cuda.is_available():
        metrics.update({
            "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
            "gpu_memory_cached_mb": torch.cuda.memory_reserved() / 1024 / 1024,
            "gpu_utilization": torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else None
        })
    
    return metrics

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )