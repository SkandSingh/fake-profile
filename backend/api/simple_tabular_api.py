"""
Simple FastAPI for Tabular Profile Classification (Minimal Dependencies)
========================================================================

A lightweight API that doesn't require scikit-learn or heavy ML dependencies.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict
import logging
from datetime import datetime
import random
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Simple Tabular Profile Classification API",
    description="Lightweight API for detecting fake profiles using rule-based scoring",
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

class SimpleClassifier:
    """Simple rule-based classifier."""
    
    def __init__(self):
        random.seed(42)  # For reproducible results
    
    def predict_probability(self, features: Dict[str, float]) -> float:
        """Predict probability using rule-based scoring."""
        score = 0.5  # Start neutral
        
        # Account age scoring
        age = features.get('account_age_days', 0)
        if age > 365:
            score += 0.2
        elif age > 90:
            score += 0.1
        elif age < 7:
            score -= 0.2
        
        # Followers/following ratio scoring
        ratio = features.get('followers_following_ratio', 0)
        if 0.5 <= ratio <= 10:
            score += 0.15
        elif ratio > 50:
            score -= 0.1
        elif ratio < 0.01:
            score -= 0.15
        
        # Post frequency scoring
        freq = features.get('post_frequency', 0)
        if 0.05 <= freq <= 2:
            score += 0.1
        elif freq > 10:
            score -= 0.2
        
        # Engagement scoring
        engagement = features.get('engagement_per_post', 0)
        if engagement > 20:
            score += 0.15
        elif engagement > 5:
            score += 0.1
        elif engagement < 1:
            score -= 0.1
        
        # Add slight randomness for demonstration
        score += random.uniform(-0.02, 0.02)
        
        return max(0.0, min(1.0, score))
    
    def predict_batch(self, features_list: List[Dict[str, float]]) -> List[float]:
        """Predict batch of profiles."""
        return [self.predict_probability(features) for features in features_list]

# Global classifier instance
classifier = SimpleClassifier()
app_start_time = datetime.now()

# Pydantic models
class ProfileFeatures(BaseModel):
    account_age_days: float = Field(..., ge=0, description="Days since account creation")
    followers_following_ratio: float = Field(..., ge=0, description="Followers/following ratio")
    post_frequency: float = Field(..., ge=0, description="Posts per day")
    engagement_per_post: float = Field(..., ge=0, description="Average engagement per post")

class ProfilePredictionRequest(BaseModel):
    features: ProfileFeatures
    model_type: str = Field(default="simple", description="Model type (always 'simple')")

class BatchPredictionRequest(BaseModel):
    profiles: List[ProfileFeatures] = Field(..., description="List of profiles (max 100)")
    model_type: str = Field(default="simple", description="Model type (always 'simple')")
    
    @validator('profiles')
    def validate_profiles_length(cls, v):
        if len(v) > 100:
            raise ValueError('Maximum 100 profiles allowed per request')
        return v

class PredictionResponse(BaseModel):
    probability_real: float = Field(..., description="Probability that profile is real (0-1)")
    classification: str = Field(..., description="Binary classification: 'real' or 'fake'")
    confidence: str = Field(..., description="Confidence level: 'low', 'medium', 'high'")
    model_type: str = Field(..., description="Model used for prediction")

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int
    processing_time_ms: float
    model_used: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    uptime_seconds: float

def get_confidence_level(probability: float) -> str:
    """Determine confidence level based on probability."""
    if probability < 0.3 or probability > 0.7:
        return "high"
    elif probability < 0.4 or probability > 0.6:
        return "medium"
    else:
        return "low"

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = (datetime.now() - app_start_time).total_seconds()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=True,
        uptime_seconds=uptime
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_profile(request: ProfilePredictionRequest):
    """Predict if a profile is real or fake."""
    try:
        features_dict = request.features.dict()
        probability = classifier.predict_probability(features_dict)
        
        classification = "real" if probability > 0.5 else "fake"
        confidence = get_confidence_level(probability)
        
        return PredictionResponse(
            probability_real=probability,
            classification=classification,
            confidence=confidence,
            model_type="simple"
        )
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict multiple profiles."""
    try:
        start_time = datetime.now()
        
        features_list = [profile.dict() for profile in request.profiles]
        probabilities = classifier.predict_batch(features_list)
        
        predictions = []
        for prob in probabilities:
            classification = "real" if prob > 0.5 else "fake"
            confidence = get_confidence_level(prob)
            
            predictions.append(PredictionResponse(
                probability_real=prob,
                classification=classification,
                confidence=confidence,
                model_type="simple"
            ))
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            processing_time_ms=processing_time,
            model_used="simple"
        )
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Simple Tabular Profile Classification API",
        "version": "1.0.0",
        "description": "Detect fake profiles using behavioral features with rule-based scoring",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch"
        },
        "features": [
            "account_age_days",
            "followers_following_ratio",
            "post_frequency", 
            "engagement_per_post"
        ],
        "model": "rule-based classifier",
        "status": "lightweight implementation - no ML dependencies required"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)