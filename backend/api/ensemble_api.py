from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import numpy as np
import math
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ensemble Learning API",
    description="Meta-learner combining text, image, and metrics scores",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EnsembleInput(BaseModel):
    textScore: float = Field(..., ge=0, le=1, description="Text analysis score (0-1)")
    imageScore: float = Field(..., ge=0, le=1, description="Image analysis score (0-1)")
    metricsScore: float = Field(..., ge=0, le=1, description="Metrics analysis score (0-1)")

class EnsembleResponse(BaseModel):
    trustScore: float = Field(..., description="Final trust score (0-100)")
    confidence: str = Field(..., description="Confidence level")
    individual_scores: Dict[str, float]
    model_performance: Dict[str, float]

class SimpleLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.is_trained = False
        
    def sigmoid(self, z):
        # Clip z to prevent overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Compute cost
            cost = self.compute_cost(y, predictions)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Log progress
            if i % 100 == 0:
                logger.debug(f"Iteration {i}, Cost: {cost:.4f}")
        
        self.is_trained = True
        return self
    
    def compute_cost(self, y_true, y_pred):
        # Binary cross-entropy loss
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)  # Prevent log(0)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

class EnsembleLearner:
    def __init__(self):
        self.model = SimpleLogisticRegression(learning_rate=0.1, max_iterations=1000)
        self.feature_means = None
        self.feature_stds = None
        self.is_trained = False
        self.performance_metrics = {}
        
    def normalize_features(self, X, fit=False):
        """Normalize features to have mean 0 and std 1"""
        if fit:
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0) + 1e-8  # Add small value to prevent division by zero
        
        return (X - self.feature_means) / self.feature_stds
        
    def generate_training_data(self, n_samples: int = 5000):
        """Generate synthetic training data for the ensemble learner"""
        np.random.seed(42)
        
        # Generate positive samples (real profiles)
        n_real = int(n_samples * 0.6)
        # Real profiles have higher scores with some noise
        real_text = np.random.beta(3, 1.5, n_real)
        real_image = np.random.beta(2.5, 1.8, n_real)
        real_metrics = np.random.beta(2.8, 1.5, n_real)
        
        # Generate negative samples (fake profiles)
        n_fake = n_samples - n_real
        # Fake profiles have lower scores
        fake_text = np.random.beta(1.2, 3, n_fake)
        fake_image = np.random.beta(1.5, 2.5, n_fake)
        fake_metrics = np.random.beta(1.3, 2.8, n_fake)
        
        # Combine data
        X = np.column_stack([
            np.concatenate([real_text, fake_text]),
            np.concatenate([real_image, fake_image]),
            np.concatenate([real_metrics, fake_metrics])
        ])
        
        y = np.concatenate([np.ones(n_real), np.zeros(n_fake)])
        
        # Shuffle data
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        return X, y
    
    def train(self):
        """Train the ensemble model"""
        logger.info("Training ensemble learner...")
        
        X, y = self.generate_training_data()
        
        # Split data (80/20 train/test)
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Normalize features
        X_train_norm = self.normalize_features(X_train, fit=True)
        X_test_norm = self.normalize_features(X_test, fit=False)
        
        # Train model
        self.model.fit(X_train_norm, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_norm)
        y_pred_proba = self.model.predict_proba(X_test_norm)
        
        accuracy = np.mean(y_test == y_pred)
        
        # Calculate AUC manually
        auc_score = self.calculate_auc(y_test, y_pred_proba)
        
        self.performance_metrics = {
            "accuracy": float(accuracy),
            "auc_score": float(auc_score),
            "training_samples": len(X_train)
        }
        
        logger.info(f"Model trained - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
        
        # Log feature importance (weights)
        feature_names = ["textScore", "imageScore", "metricsScore"]
        for name, weight in zip(feature_names, self.model.weights):
            logger.info(f"Feature {name}: weight = {weight:.4f}")
        
        self.is_trained = True
        return True
    
    def calculate_auc(self, y_true, y_scores):
        """Calculate AUC-ROC manually"""
        # Create all possible thresholds
        thresholds = np.unique(y_scores)
        thresholds = np.sort(thresholds)[::-1]
        
        tpr_list = []
        fpr_list = []
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        # Calculate AUC using trapezoidal rule
        tpr_list = np.array(tpr_list)
        fpr_list = np.array(fpr_list)
        
        auc = np.trapz(tpr_list, fpr_list)
        return abs(auc)  # Ensure positive AUC
    
    def predict(self, text_score: float, image_score: float, metrics_score: float) -> Dict[str, Any]:
        """Make ensemble prediction"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Prepare input
        X = np.array([[text_score, image_score, metrics_score]])
        X_norm = self.normalize_features(X, fit=False)
        
        # Predict probability
        prob = self.model.predict_proba(X_norm)[0]
        
        # Scale to 0-100
        trust_score = float(prob * 100)
        
        # Determine confidence
        if prob < 0.3 or prob > 0.7:
            confidence = "high"
        elif prob < 0.4 or prob > 0.6:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            "trust_score": trust_score,
            "confidence": confidence,
            "raw_probability": float(prob)
        }

# Global ensemble learner instance
ensemble_learner = EnsembleLearner()

@app.on_event("startup")
async def startup_event():
    """Initialize the ensemble model on startup"""
    logger.info("Starting Ensemble Learning API...")
    
    try:
        ensemble_learner.train()
        logger.info("Ensemble model initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing ensemble model: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_trained": ensemble_learner.is_trained,
        "performance": ensemble_learner.performance_metrics
    }

@app.post("/ensemble", response_model=EnsembleResponse)
async def ensemble_prediction(request: EnsembleInput):
    """Ensemble prediction endpoint"""
    try:
        if not ensemble_learner.is_trained:
            raise HTTPException(status_code=503, detail="Ensemble model not trained")
        
        result = ensemble_learner.predict(
            request.textScore,
            request.imageScore,
            request.metricsScore
        )
        
        return EnsembleResponse(
            trustScore=result["trust_score"],
            confidence=result["confidence"],
            individual_scores={
                "textScore": request.textScore,
                "imageScore": request.imageScore,
                "metricsScore": request.metricsScore
            },
            model_performance=ensemble_learner.performance_metrics
        )
        
    except Exception as e:
        logger.error(f"Error in ensemble prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information and feature importance"""
    if not ensemble_learner.is_trained:
        raise HTTPException(status_code=503, detail="Model not trained")
    
    feature_names = ["textScore", "imageScore", "metricsScore"]
    weights = ensemble_learner.model.weights.tolist() if ensemble_learner.model.weights is not None else []
    
    return {
        "model_type": "SimpleLogisticRegression",
        "feature_weights": dict(zip(feature_names, weights)),
        "performance_metrics": ensemble_learner.performance_metrics,
        "bias": float(ensemble_learner.model.bias) if ensemble_learner.model.bias is not None else 0.0
    }

@app.post("/retrain")
async def retrain_model():
    """Retrain the ensemble model"""
    try:
        ensemble_learner.train()
        return {
            "status": "success",
            "message": "Model retrained successfully",
            "performance": ensemble_learner.performance_metrics
        }
    except Exception as e:
        logger.error(f"Error retraining model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Ensemble Learning API",
        "version": "1.0.0",
        "description": "Meta-learner combining text, image, and metrics scores",
        "inputs": ["textScore", "imageScore", "metricsScore"],
        "output": "trustScore (0-100)",
        "model": "LogisticRegression",
        "endpoints": {
            "ensemble": "/ensemble",
            "health": "/health",
            "model_info": "/model/info",
            "retrain": "/retrain",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)