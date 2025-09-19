from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime
import json
import random
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ML_AVAILABLE = False
try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb
    import joblib
    ML_AVAILABLE = True
    logger.info("ML libraries loaded successfully")
except ImportError as e:
    logger.warning(f"ML libraries not available: {e}")
    logger.info("Falling back to rule-based classifier")

app = FastAPI(
    title="Tabular Profile Classification API",
    description="RandomForest/XGBoost classifiers for fake profile detection",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MLTabularClassifier:
    
    def __init__(self, model_type: str = "randomforest"):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.feature_names = [
            'account_age_days',
            'followers_following_ratio',
            'post_frequency',
            'engagement_per_post'
        ]
        
        if not ML_AVAILABLE:
            raise ImportError("ML libraries not available")
        
        # Initialize models
        if model_type == "randomforest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )
        elif model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1
            )
        else:
            raise ValueError("model_type must be 'randomforest' or 'xgboost'")
        
        self.scaler = RobustScaler()
    
    def generate_training_data(self, n_samples: int = 10000):
        """Generate synthetic training data."""
        np.random.seed(42)
        
        # Generate real profiles (60%)
        n_real = int(n_samples * 0.6)
        real_data = {
            'account_age_days': np.random.gamma(2, 200, n_real),
            'followers_following_ratio': np.random.gamma(2, 0.5, n_real),
            'post_frequency': np.random.gamma(1.8, 0.1, n_real),
            'engagement_per_post': np.random.gamma(2, 10, n_real)
        }
        
        # Generate fake profiles (40%)
        n_fake = n_samples - n_real
        fake_data = {
            'account_age_days': np.random.gamma(1, 30, n_fake),
            'followers_following_ratio': np.concatenate([
                np.random.gamma(3, 10, n_fake//2),  # High ratio (bought followers)
                np.random.gamma(1, 0.01, n_fake//2)  # Low ratio (following many)
            ]),
            'post_frequency': np.random.gamma(1, 5, n_fake),  # Irregular posting
            'engagement_per_post': np.random.gamma(1, 3, n_fake)  # Low engagement
        }
        
        # Combine data
        all_data = {
            'account_age_days': np.concatenate([real_data['account_age_days'], fake_data['account_age_days']]),
            'followers_following_ratio': np.concatenate([real_data['followers_following_ratio'], fake_data['followers_following_ratio']]),
            'post_frequency': np.concatenate([real_data['post_frequency'], fake_data['post_frequency']]),
            'engagement_per_post': np.concatenate([real_data['engagement_per_post'], fake_data['engagement_per_post']])
        }
        
        # Create labels (1 = real, 0 = fake)
        labels = np.concatenate([np.ones(n_real), np.zeros(n_fake)])
        
        # Create DataFrame and shuffle
        df = pd.DataFrame(all_data)
        indices = np.random.permutation(len(df))
        df = df.iloc[indices].reset_index(drop=True)
        labels = labels[indices]
        
        return df, labels
    
    def train(self):
        """Train the classifier."""
        if not ML_AVAILABLE:
            return False
        
        logger.info(f"Training {self.model_type} classifier...")
        
        # Generate training data
        features_df, labels = self.generate_training_data()
        
        # Prepare features
        X = features_df[self.feature_names].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X = X.clip(lower=0, upper=X.quantile(0.99))
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_prob = self.model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_prob)
        
        logger.info(f"Model trained successfully. AUC Score: {auc_score:.4f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            for i, feature in enumerate(self.feature_names):
                logger.info(f"  {feature}: {importances[i]:.4f}")
        
        self.is_trained = True
        return True
    
    def predict_probability(self, features: Dict[str, float]) -> float:
        """Predict probability that profile is real."""
        if not self.is_trained:
            return 0.5
        
        # Prepare feature vector
        feature_vector = np.array([features.get(name, 0.0) for name in self.feature_names])
        feature_vector = feature_vector.reshape(1, -1)
        
        # Handle outliers and scale
        feature_vector = np.clip(feature_vector, 0, np.percentile(feature_vector, 99))
        feature_vector = self.scaler.transform(feature_vector)
        
        # Predict
        prob = self.model.predict_proba(feature_vector)[0, 1]
        return float(prob)

class RuleBasedClassifier:
    
    def __init__(self):
        random.seed(42)
    
    def predict_probability(self, features: Dict[str, float]) -> float:
        score = 0.5
        
        age = features.get('account_age_days', 0)
        if age > 365: score += 0.2
        elif age > 90: score += 0.1
        elif age < 7: score -= 0.2
        
        ratio = features.get('followers_following_ratio', 0)
        if 0.5 <= ratio <= 10: score += 0.15
        elif ratio > 50: score -= 0.1
        elif ratio < 0.01: score -= 0.15
        
        freq = features.get('post_frequency', 0)
        if 0.05 <= freq <= 2: score += 0.1
        elif freq > 10: score -= 0.2
        
        engagement = features.get('engagement_per_post', 0)
        if engagement > 20: score += 0.15
        elif engagement > 5: score += 0.1
        elif engagement < 1: score -= 0.1
        
        score += random.uniform(-0.02, 0.02)
        
        return max(0.0, min(1.0, score))

# Global classifier instances
rf_classifier = None
xgb_classifier = None
rule_classifier = RuleBasedClassifier()
app_start_time = datetime.now()

# Pydantic models
class ProfileFeatures(BaseModel):
    account_age_days: float = Field(..., ge=0, description="Days since account creation")
    followers_following_ratio: float = Field(..., ge=0, description="Followers/following ratio")
    post_frequency: float = Field(..., ge=0, description="Posts per day")
    engagement_per_post: float = Field(..., ge=0, description="Average engagement per post")

class ProfilePredictionRequest(BaseModel):
    features: ProfileFeatures
    model_type: str = Field(default="auto", description="Model: 'randomforest', 'xgboost', 'rule', or 'auto'")
    
    @validator('model_type')
    def validate_model_type(cls, v):
        valid_types = ['randomforest', 'xgboost', 'rule', 'auto']
        if v.lower() not in valid_types:
            raise ValueError(f'model_type must be one of {valid_types}')
        return v.lower()

class BatchPredictionRequest(BaseModel):
    profiles: List[ProfileFeatures] = Field(..., description="List of profiles (max 100)")
    model_type: str = Field(default="auto", description="Model type")
    
    @validator('profiles')
    def validate_profiles_length(cls, v):
        if len(v) > 100:
            raise ValueError('Maximum 100 profiles allowed per request')
        return v
    
    @validator('model_type')
    def validate_model_type(cls, v):
        valid_types = ['randomforest', 'xgboost', 'rule', 'auto']
        if v.lower() not in valid_types:
            raise ValueError(f'model_type must be one of {valid_types}')
        return v.lower()

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
    ml_available: bool
    models_loaded: Dict[str, bool]
    uptime_seconds: float

@app.on_event("startup")
async def startup_event():
    global rf_classifier, xgb_classifier
    
    logger.info("Starting Tabular Classification API...")
    
    if ML_AVAILABLE:
        try:
            logger.info("Training RandomForest classifier...")
            rf_classifier = MLTabularClassifier(model_type="randomforest")
            rf_classifier.train()
            
            logger.info("Training XGBoost classifier...")
            xgb_classifier = MLTabularClassifier(model_type="xgboost")
            xgb_classifier.train()
            
            logger.info("ML models trained successfully")
        except Exception as e:
            logger.error(f"Error training ML models: {e}")
            rf_classifier = None
            xgb_classifier = None
    
    logger.info("Rule-based classifier available as fallback")
    logger.info("API initialization completed")

def get_confidence_level(probability: float) -> str:
    if probability < 0.3 or probability > 0.7:
        return "high"
    elif probability < 0.4 or probability > 0.6:
        return "medium"
    else:
        return "low"

def get_best_model(requested_type: str):
    if requested_type == "rule":
        return rule_classifier, "rule"
    elif requested_type == "randomforest" and rf_classifier and rf_classifier.is_trained:
        return rf_classifier, "randomforest"
    elif requested_type == "xgboost" and xgb_classifier and xgb_classifier.is_trained:
        return xgb_classifier, "xgboost"
    elif requested_type == "auto":
        if rf_classifier and rf_classifier.is_trained:
            return rf_classifier, "randomforest"
        elif xgb_classifier and xgb_classifier.is_trained:
            return xgb_classifier, "xgboost"
    
    return rule_classifier, "rule"

@app.get("/health", response_model=HealthResponse)
async def health_check():
    uptime = (datetime.now() - app_start_time).total_seconds()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        ml_available=ML_AVAILABLE,
        models_loaded={
            "randomforest": rf_classifier is not None and rf_classifier.is_trained if rf_classifier else False,
            "xgboost": xgb_classifier is not None and xgb_classifier.is_trained if xgb_classifier else False,
            "rule": True
        },
        uptime_seconds=uptime
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_profile(request: ProfilePredictionRequest):
    try:
        model, model_type = get_best_model(request.model_type)
        features_dict = request.features.dict()
        
        probability = model.predict_probability(features_dict)
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
    try:
        start_time = datetime.now()
        model, model_type = get_best_model(request.model_type)
        
        predictions = []
        for profile in request.profiles:
            features_dict = profile.dict()
            probability = model.predict_probability(features_dict)
            classification = "real" if probability > 0.5 else "fake"
            confidence = get_confidence_level(probability)
            
            predictions.append(PredictionResponse(
                probability_real=probability,
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

@app.get("/")
async def root():
    return {
        "message": "Tabular Profile Classification API",
        "version": "1.0.0",
        "description": "RandomForest/XGBoost classifiers with rule-based fallback",
        "features": [
            "account_age_days",
            "followers_following_ratio",
            "post_frequency",
            "engagement_per_post"
        ],
        "models": ["randomforest", "xgboost", "rule", "auto"],
        "ml_available": ML_AVAILABLE,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)