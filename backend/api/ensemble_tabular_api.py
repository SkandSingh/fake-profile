"""
Ensemble Tabular Profile Classification API
===========================================

Advanced API with ensemble methods, model persistence, and comprehensive monitoring.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
import asyncio
import logging
from datetime import datetime
import sys
import os
import json
from pathlib import Path
import time
import psutil

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import advanced classifier, fall back to simple
try:
    from ml.advanced_tabular_classifier import AdvancedTabularClassifier
    ADVANCED_MODELS_AVAILABLE = True
    logger.info("✅ Advanced ML models available")
except ImportError as e:
    logger.warning(f"⚠️ Advanced ML models not available: {e}")
    ADVANCED_MODELS_AVAILABLE = False

# Always import simple classifier as fallback
from ml.simple_tabular_classifier import SimpleTabularClassifier

# Initialize FastAPI app
app = FastAPI(
    title="Ensemble Tabular Profile Classification API",
    description="Advanced API with ensemble methods, model persistence, and monitoring",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
class APIState:
    def __init__(self):
        self.models = {}
        self.simple_model = SimpleTabularClassifier()
        self.request_count = 0
        self.start_time = datetime.now()
        self.prediction_history = []
        self.model_performance = {}

state = APIState()

# Pydantic models with enhanced validation
class ProfileFeatures(BaseModel):
    """Enhanced profile features for classification."""
    account_age_days: float = Field(..., ge=0, le=10000, description="Days since account creation")
    followers_following_ratio: float = Field(..., ge=0, le=1000, description="Followers/following ratio")
    post_frequency: float = Field(..., ge=0, le=100, description="Posts per day")
    engagement_per_post: float = Field(..., ge=0, le=10000, description="Average engagement per post")
    
    # Optional enhanced features
    follower_count: Optional[float] = Field(None, ge=0, description="Total follower count")
    following_count: Optional[float] = Field(None, ge=0, description="Total following count")
    post_count: Optional[float] = Field(None, ge=0, description="Total post count")
    verification_status: Optional[bool] = Field(None, description="Account verification status")
    
    class Config:
        schema_extra = {
            "example": {
                "account_age_days": 365,
                "followers_following_ratio": 1.5,
                "post_frequency": 0.2,
                "engagement_per_post": 25.0,
                "follower_count": 1500,
                "following_count": 1000,
                "post_count": 73,
                "verification_status": False
            }
        }

class PredictionRequest(BaseModel):
    """Request for profile prediction with ensemble options."""
    features: ProfileFeatures
    model_type: str = Field(default="ensemble", description="Model: 'randomforest', 'xgboost', 'ensemble', 'simple', 'auto'")
    include_explanation: bool = Field(default=False, description="Include prediction explanation")
    
    @validator('model_type')
    def validate_model_type(cls, v):
        valid_types = ['randomforest', 'xgboost', 'ensemble', 'simple', 'auto']
        if v.lower() not in valid_types:
            raise ValueError(f'model_type must be one of {valid_types}')
        return v.lower()

class BatchPredictionRequest(BaseModel):
    """Request for batch profile prediction."""
    profiles: List[ProfileFeatures] = Field(..., max_items=100, description="List of profiles")
    model_type: str = Field(default="ensemble", description="Model type")
    include_explanation: bool = Field(default=False, description="Include explanations")
    
    @validator('profiles')
    def validate_profiles_length(cls, v):
        if len(v) > 100:
            raise ValueError('Maximum 100 profiles allowed per request')
        return v

class PredictionExplanation(BaseModel):
    """Explanation for prediction decision."""
    key_factors: List[str]
    risk_indicators: List[str]
    confidence_factors: List[str]
    feature_contributions: Dict[str, float]

class PredictionResponse(BaseModel):
    """Enhanced response for profile prediction."""
    probability_real: float = Field(..., description="Probability that profile is real (0-1)")
    classification: str = Field(..., description="Binary classification: 'real' or 'fake'")
    confidence: str = Field(..., description="Confidence level: 'low', 'medium', 'high'")
    model_type: str = Field(..., description="Model used for prediction")
    prediction_id: str = Field(..., description="Unique prediction identifier")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    explanation: Optional[PredictionExplanation] = Field(None, description="Prediction explanation")

class BatchPredictionResponse(BaseModel):
    """Response for batch prediction."""
    predictions: List[PredictionResponse]
    total_processed: int
    success_count: int
    error_count: int
    total_processing_time_ms: float
    average_processing_time_ms: float
    model_used: str

class ModelInfo(BaseModel):
    """Comprehensive model information."""
    available_models: Dict[str, bool]
    loaded_models: Dict[str, Any]
    model_performance: Dict[str, Dict[str, float]]
    feature_names: List[str]
    api_version: str
    system_info: Dict[str, Any]

class HealthResponse(BaseModel):
    """Enhanced health check response."""
    status: str
    timestamp: str
    uptime_seconds: float
    request_count: int
    models_loaded: Dict[str, bool]
    system_metrics: Dict[str, Any]
    recent_performance: Dict[str, float]

@app.on_event("startup")
async def startup_event():
    """Initialize models and monitoring on startup."""
    logger.info("Starting Ensemble Tabular Profile Classification API...")
    
    if ADVANCED_MODELS_AVAILABLE:
        try:
            # Initialize ensemble model
            logger.info("Loading ensemble model...")
            ensemble_model = AdvancedTabularClassifier(model_type="ensemble")
            
            # Check for existing models
            models_dir = Path("models")
            if models_dir.exists():
                model_files = list(models_dir.glob("*.joblib"))
                if model_files:
                    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                    logger.info(f"Loading existing model: {latest_model}")
                    ensemble_model.load_model(latest_model)
                else:
                    logger.info("Training new ensemble model...")
                    ensemble_model.train()
                    ensemble_model.save_model()
            else:
                logger.info("Training new ensemble model...")
                ensemble_model.train()
                ensemble_model.save_model()
            
            state.models['ensemble'] = ensemble_model
            
            # Load individual models too
            for model_type in ['randomforest', 'xgboost']:
                try:
                    model = AdvancedTabularClassifier(model_type=model_type)
                    model.train()
                    state.models[model_type] = model
                    logger.info(f"✅ {model_type} model loaded")
                except Exception as e:
                    logger.error(f"❌ Failed to load {model_type}: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error loading advanced models: {e}")
    
    # Simple model is always available
    state.models['simple'] = state.simple_model
    logger.info("✅ Simple rule-based model loaded")
    
    logger.info("🚀 API initialization completed")

def get_best_available_model(requested_type: str):
    """Get the best available model based on request."""
    if requested_type in state.models:
        return state.models[requested_type], requested_type
    
    if requested_type == "auto":
        # Prefer ensemble, then individual models, then simple
        for model_type in ['ensemble', 'randomforest', 'xgboost', 'simple']:
            if model_type in state.models:
                return state.models[model_type], model_type
    
    # Fallback to simple
    return state.models['simple'], 'simple'

def generate_prediction_explanation(features: Dict[str, float], probability: float) -> PredictionExplanation:
    """Generate human-readable explanation for prediction."""
    key_factors = []
    risk_indicators = []
    confidence_factors = []
    
    # Analyze key features
    age = features.get('account_age_days', 0)
    ratio = features.get('followers_following_ratio', 0)
    freq = features.get('post_frequency', 0)
    engagement = features.get('engagement_per_post', 0)
    
    # Account age analysis
    if age > 365:
        key_factors.append("Mature account (>1 year old)")
        confidence_factors.append("Established account history")
    elif age < 30:
        risk_indicators.append("Very new account (<30 days)")
        confidence_factors.append("Limited account history")
    
    # Ratio analysis
    if 0.5 <= ratio <= 10:
        key_factors.append("Healthy follower/following ratio")
    elif ratio > 50:
        risk_indicators.append("Unusually high follower ratio (possible bought followers)")
    elif ratio < 0.01:
        risk_indicators.append("Very low follower ratio (follows many, few followers)")
    
    # Posting frequency analysis
    if freq > 10:
        risk_indicators.append("Extremely high posting frequency (>10 posts/day)")
    elif 0.1 <= freq <= 2:
        key_factors.append("Normal posting frequency")
    
    # Engagement analysis
    if engagement > 20:
        key_factors.append("High engagement rate")
        confidence_factors.append("Strong user interaction")
    elif engagement < 1:
        risk_indicators.append("Very low engagement rate")
    
    # Feature contributions (simplified)
    feature_contributions = {
        "account_age": min(age / 365, 1.0) * 0.3,
        "social_ratio": min(max(ratio, 0), 10) / 10 * 0.25,
        "posting_pattern": min(freq, 2) / 2 * 0.25,
        "engagement": min(engagement, 50) / 50 * 0.2
    }
    
    return PredictionExplanation(
        key_factors=key_factors,
        risk_indicators=risk_indicators,
        confidence_factors=confidence_factors,
        feature_contributions=feature_contributions
    )

def get_confidence_level(probability: float) -> str:
    """Determine confidence level based on probability."""
    if probability < 0.2 or probability > 0.8:
        return "high"
    elif probability < 0.35 or probability > 0.65:
        return "medium"
    else:
        return "low"

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check with system metrics."""
    uptime = (datetime.now() - state.start_time).total_seconds()
    
    # Get system metrics
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent()
    
    # Calculate recent performance
    recent_predictions = state.prediction_history[-100:] if state.prediction_history else []
    avg_response_time = sum(p.get('processing_time', 0) for p in recent_predictions) / max(len(recent_predictions), 1)
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        uptime_seconds=uptime,
        request_count=state.request_count,
        models_loaded={name: model is not None for name, model in state.models.items()},
        system_metrics={
            "memory_usage_percent": memory.percent,
            "cpu_usage_percent": cpu_percent,
            "available_memory_gb": memory.available / (1024**3)
        },
        recent_performance={
            "average_response_time_ms": avg_response_time,
            "total_predictions": len(state.prediction_history)
        }
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_profile(request: PredictionRequest):
    """Advanced profile prediction with ensemble methods."""
    start_time = time.time()
    prediction_id = f"pred_{int(time.time() * 1000)}"
    
    try:
        state.request_count += 1
        
        # Get the appropriate model
        model, model_type = get_best_available_model(request.model_type)
        
        # Convert features to dictionary
        features_dict = request.features.dict()
        
        # Get prediction
        probability = model.predict_probability(features_dict)
        
        # Determine classification and confidence
        classification = "real" if probability > 0.5 else "fake"
        confidence = get_confidence_level(probability)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Generate explanation if requested
        explanation = None
        if request.include_explanation:
            explanation = generate_prediction_explanation(features_dict, probability)
        
        # Record prediction for monitoring
        prediction_record = {
            "id": prediction_id,
            "probability": probability,
            "model_type": model_type,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        state.prediction_history.append(prediction_record)
        
        # Keep only last 1000 predictions
        if len(state.prediction_history) > 1000:
            state.prediction_history = state.prediction_history[-1000:]
        
        return PredictionResponse(
            probability_real=probability,
            classification=classification,
            confidence=confidence,
            model_type=model_type,
            prediction_id=prediction_id,
            processing_time_ms=processing_time,
            explanation=explanation
        )
        
    except Exception as e:
        logger.error(f"Error in prediction {prediction_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Advanced batch prediction with monitoring."""
    start_time = time.time()
    
    try:
        # Get the appropriate model
        model, model_type = get_best_available_model(request.model_type)
        
        # Process predictions
        predictions = []
        success_count = 0
        error_count = 0
        
        for i, profile in enumerate(request.profiles):
            try:
                prediction_start = time.time()
                prediction_id = f"batch_{int(time.time() * 1000)}_{i}"
                
                features_dict = profile.dict()
                probability = model.predict_probability(features_dict)
                
                classification = "real" if probability > 0.5 else "fake"
                confidence = get_confidence_level(probability)
                processing_time = (time.time() - prediction_start) * 1000
                
                explanation = None
                if request.include_explanation:
                    explanation = generate_prediction_explanation(features_dict, probability)
                
                predictions.append(PredictionResponse(
                    probability_real=probability,
                    classification=classification,
                    confidence=confidence,
                    model_type=model_type,
                    prediction_id=prediction_id,
                    processing_time_ms=processing_time,
                    explanation=explanation
                ))
                
                success_count += 1
                
            except Exception as e:
                logger.error(f"Error in batch prediction {i}: {str(e)}")
                error_count += 1
        
        total_processing_time = (time.time() - start_time) * 1000
        avg_processing_time = total_processing_time / len(request.profiles) if request.profiles else 0
        
        state.request_count += 1
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(request.profiles),
            success_count=success_count,
            error_count=error_count,
            total_processing_time_ms=total_processing_time,
            average_processing_time_ms=avg_processing_time,
            model_used=model_type
        )
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get comprehensive model information."""
    # Get system info
    memory = psutil.virtual_memory()
    
    return ModelInfo(
        available_models={
            "randomforest": ADVANCED_MODELS_AVAILABLE,
            "xgboost": ADVANCED_MODELS_AVAILABLE,
            "ensemble": ADVANCED_MODELS_AVAILABLE,
            "simple": True
        },
        loaded_models={name: bool(model) for name, model in state.models.items()},
        model_performance=state.model_performance,
        feature_names=[
            "account_age_days", "followers_following_ratio", 
            "post_frequency", "engagement_per_post"
        ],
        api_version="2.0.0",
        system_info={
            "memory_total_gb": memory.total / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "cpu_count": psutil.cpu_count()
        }
    )

@app.get("/stats/performance")
async def get_performance_stats():
    """Get detailed performance statistics."""
    if not state.prediction_history:
        return {"message": "No predictions recorded yet"}
    
    recent_predictions = state.prediction_history[-100:]
    
    stats = {
        "total_predictions": len(state.prediction_history),
        "recent_predictions": len(recent_predictions),
        "average_response_time_ms": sum(p.get('processing_time', 0) for p in recent_predictions) / len(recent_predictions),
        "model_usage": {},
        "classification_distribution": {"real": 0, "fake": 0}
    }
    
    # Count model usage
    for pred in recent_predictions:
        model_type = pred.get('model_type', 'unknown')
        stats["model_usage"][model_type] = stats["model_usage"].get(model_type, 0) + 1
        
        # Count classifications
        if pred.get('probability', 0) > 0.5:
            stats["classification_distribution"]["real"] += 1
        else:
            stats["classification_distribution"]["fake"] += 1
    
    return stats

@app.get("/")
async def root():
    """Root endpoint with comprehensive API information."""
    return {
        "message": "Ensemble Tabular Profile Classification API",
        "version": "2.0.0",
        "description": "Advanced fake profile detection with ensemble methods and monitoring",
        "features": [
            "Multi-model ensemble (RandomForest + XGBoost)",
            "Advanced feature engineering",
            "Model persistence and loading",
            "Comprehensive monitoring and metrics",
            "Prediction explanations",
            "Batch processing with error handling",
            "Real-time performance tracking"
        ],
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "model_info": "/model/info",
            "performance_stats": "/stats/performance",
            "documentation": "/docs"
        },
        "models": list(state.models.keys()),
        "advanced_features_available": ADVANCED_MODELS_AVAILABLE,
        "system_status": "operational"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)