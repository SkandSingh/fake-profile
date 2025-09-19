"""
Hybrid Tabular Profile Classification API
=========================================

FastAPI application that attempts to use ML models (RandomForest/XGBoost) 
but falls back to a simple rule-based classifier if dependencies fail.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import logging
from datetime import datetime
import sys
import os

# Add the parent directory to the path to import ml modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import ML models, fall back to simple classifier
try:
    from ml.tabular_profile_classifier import TabularProfileClassifier
    ML_MODELS_AVAILABLE = True
    logger.info("✅ Advanced ML models available")
except ImportError as e:
    logger.warning(f"⚠️ ML models not available: {e}")
    ML_MODELS_AVAILABLE = False

# Always import the simple classifier as fallback
from ml.simple_tabular_classifier import SimpleTabularClassifier

# Initialize FastAPI app
app = FastAPI(
    title="Hybrid Tabular Profile Classification API",
    description="API for detecting fake profiles using tabular behavioral features with ML fallback",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instances
rf_model = None
xgb_model = None
simple_model = SimpleTabularClassifier()

# Pydantic models for request/response validation
class ProfileFeatures(BaseModel):
    """Profile features for classification."""
    account_age_days: float = Field(..., ge=0, description="Days since account creation")
    followers_following_ratio: float = Field(..., ge=0, description="Followers divided by following count")
    post_frequency: float = Field(..., ge=0, description="Posts per day")
    engagement_per_post: float = Field(..., ge=0, description="Average engagement per post")
    
    @validator('followers_following_ratio')
    def validate_ratio(cls, v):
        if v > 1000:  # Reasonable upper limit
            logger.warning(f"High followers/following ratio: {v}")
        return v
    
    @validator('post_frequency')
    def validate_frequency(cls, v):
        if v > 100:  # More than 100 posts per day seems excessive
            logger.warning(f"High post frequency: {v}")
        return v

class ProfilePredictionRequest(BaseModel):
    """Request for profile prediction."""
    features: ProfileFeatures
    model_type: str = Field(default="auto", description="Model type: 'randomforest', 'xgboost', 'simple', or 'auto'")
    
    @validator('model_type')
    def validate_model_type(cls, v):
        valid_types = ['randomforest', 'xgboost', 'simple', 'auto']
        if v.lower() not in valid_types:
            raise ValueError(f'model_type must be one of {valid_types}')
        return v.lower()

class BatchPredictionRequest(BaseModel):
    """Request for batch profile prediction."""
    profiles: List[ProfileFeatures] = Field(..., description="List of profiles (max 100)")
    model_type: str = Field(default="auto", description="Model type: 'randomforest', 'xgboost', 'simple', or 'auto'")
    
    @validator('profiles')
    def validate_profiles_length(cls, v):
        if len(v) > 100:
            raise ValueError('Maximum 100 profiles allowed per request')
        return v
    
    @validator('model_type')
    def validate_model_type(cls, v):
        valid_types = ['randomforest', 'xgboost', 'simple', 'auto']
        if v.lower() not in valid_types:
            raise ValueError(f'model_type must be one of {valid_types}')
        return v.lower()

class PredictionResponse(BaseModel):
    """Response for profile prediction."""
    probability_real: float = Field(..., description="Probability that profile is real (0-1)")
    classification: str = Field(..., description="Binary classification: 'real' or 'fake'")
    confidence: str = Field(..., description="Confidence level: 'low', 'medium', 'high'")
    model_type: str = Field(..., description="Model used for prediction")

class BatchPredictionResponse(BaseModel):
    """Response for batch prediction."""
    predictions: List[PredictionResponse]
    total_processed: int
    processing_time_ms: float
    model_used: str

class ModelInfo(BaseModel):
    """Model information response."""
    ml_models_available: bool
    randomforest_loaded: bool
    xgboost_loaded: bool
    simple_model_available: bool
    feature_names: List[str]
    last_trained: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    models_available: Dict[str, bool]
    uptime_seconds: float

# Store app start time for uptime calculation
app_start_time = datetime.now()

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global rf_model, xgb_model
    
    logger.info("Starting Hybrid Tabular Profile Classification API...")
    
    if ML_MODELS_AVAILABLE:
        try:
            # Initialize RandomForest model
            logger.info("Initializing RandomForest model...")
            rf_model = TabularProfileClassifier(model_type="randomforest")
            rf_model.train()  # Train with synthetic data
            logger.info("✅ RandomForest model loaded successfully")
            
            # Initialize XGBoost model
            logger.info("Initializing XGBoost model...")
            xgb_model = TabularProfileClassifier(model_type="xgboost")
            xgb_model.train()  # Train with synthetic data
            logger.info("✅ XGBoost model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading ML models: {str(e)}")
            rf_model = None
            xgb_model = None
    
    logger.info("✅ Simple rule-based model always available")
    logger.info("🚀 API initialization completed")

def get_confidence_level(probability: float) -> str:
    """Determine confidence level based on probability."""
    if probability < 0.3 or probability > 0.7:
        return "high"
    elif probability < 0.4 or probability > 0.6:
        return "medium"
    else:
        return "low"

def get_best_available_model(requested_type: str):
    """Get the best available model based on request and availability."""
    if requested_type == "simple":
        return simple_model, "simple"
    
    if requested_type == "randomforest" and rf_model and rf_model.is_trained:
        return rf_model, "randomforest"
    
    if requested_type == "xgboost" and xgb_model and xgb_model.is_trained:
        return xgb_model, "xgboost"
    
    if requested_type == "auto":
        # Try RandomForest first, then XGBoost, then simple
        if rf_model and rf_model.is_trained:
            return rf_model, "randomforest"
        elif xgb_model and xgb_model.is_trained:
            return xgb_model, "xgboost"
    
    # Fall back to simple model
    logger.warning(f"Requested model '{requested_type}' not available, using simple model")
    return simple_model, "simple"

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = (datetime.now() - app_start_time).total_seconds()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_available={
            "randomforest": rf_model is not None and (rf_model.is_trained if hasattr(rf_model, 'is_trained') else True),
            "xgboost": xgb_model is not None and (xgb_model.is_trained if hasattr(xgb_model, 'is_trained') else True),
            "simple": True
        },
        uptime_seconds=uptime
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_profile(request: ProfilePredictionRequest):
    """
    Predict if a profile is real or fake based on tabular features.
    
    Returns probability score (0-1) where:
    - 0.0 = very likely fake
    - 1.0 = very likely real
    """
    try:
        # Get the best available model
        model, model_type = get_best_available_model(request.model_type)
        
        # Convert features to dictionary
        features_dict = request.features.dict()
        
        # Get prediction
        probability = model.predict_probability(features_dict)
        
        # Determine classification and confidence
        classification = "real" if probability > 0.5 else "fake"
        confidence = get_confidence_level(probability)
        
        return PredictionResponse(
            probability_real=probability,
            classification=classification,
            confidence=confidence,
            model_type=model_type
        )
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict if multiple profiles are real or fake.
    Maximum 100 profiles per request.
    """
    try:
        start_time = datetime.now()
        
        # Get the best available model
        model, model_type = get_best_available_model(request.model_type)
        
        # Convert features to list of dictionaries
        features_list = [profile.dict() for profile in request.profiles]
        
        # Get batch predictions
        probabilities = model.predict_batch(features_list)
        
        # Create response objects
        predictions = []
        for prob in probabilities:
            classification = "real" if prob > 0.5 else "fake"
            confidence = get_confidence_level(prob)
            
            predictions.append(PredictionResponse(
                probability_real=prob,
                classification=classification,
                confidence=confidence,
                model_type=model_type
            ))
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            processing_time_ms=processing_time,
            model_used=model_type
        )
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about loaded models."""
    feature_names = ["account_age_days", "followers_following_ratio", "post_frequency", "engagement_per_post"]
    
    return ModelInfo(
        ml_models_available=ML_MODELS_AVAILABLE,
        randomforest_loaded=rf_model is not None and (rf_model.is_trained if hasattr(rf_model, 'is_trained') else True),
        xgboost_loaded=xgb_model is not None and (xgb_model.is_trained if hasattr(xgb_model, 'is_trained') else True),
        simple_model_available=True,
        feature_names=feature_names,
        last_trained=datetime.now().isoformat() if rf_model or xgb_model else None
    )

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Hybrid Tabular Profile Classification API",
        "version": "1.0.0",
        "description": "Detect fake profiles using behavioral features with ML fallback",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch", 
            "model_info": "/model/info"
        },
        "features": [
            "account_age_days",
            "followers_following_ratio",
            "post_frequency", 
            "engagement_per_post"
        ],
        "models": ["randomforest", "xgboost", "simple", "auto"],
        "ml_models_available": ML_MODELS_AVAILABLE
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)