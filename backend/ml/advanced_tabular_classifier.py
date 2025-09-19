"""
Advanced Tabular Profile Classifier with Model Persistence
==========================================================

Enhanced version with model saving/loading, advanced features, and ensemble methods.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import joblib
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import datetime
import json
import hashlib
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Metadata for trained models."""
    model_type: str
    training_date: str
    training_samples: int
    auc_score: float
    feature_names: List[str]
    model_hash: str
    version: str = "2.0.0"

class AdvancedTabularClassifier:
    """
    Advanced tabular profile classifier with persistence and enhanced features.
    """
    
    def __init__(self, model_type: str = "randomforest", models_dir: str = "models"):
        """
        Initialize the advanced classifier.
        
        Args:
            model_type: "randomforest", "xgboost", or "ensemble"
            models_dir: Directory to save/load models
        """
        self.model_type = model_type.lower()
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Core components
        self.model = None
        self.scaler = None
        self.metadata = None
        self.is_trained = False
        
        # Feature engineering
        self.feature_names = [
            'account_age_days',
            'followers_following_ratio',
            'post_frequency',
            'engagement_per_post',
            # Enhanced features
            'account_age_weeks',
            'follower_growth_rate',
            'engagement_consistency',
            'posting_pattern_score',
            'social_graph_score'
        ]
        
        # Initialize models
        self._initialize_model()
        logger.info(f"Initialized {self.model_type} classifier with enhanced features")
    
    def _initialize_model(self):
        """Initialize the appropriate model."""
        if self.model_type == "randomforest":
            self.model = RandomForestClassifier(
                n_estimators=200,  # Increased for better performance
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=42,
                class_weight='balanced',
                n_jobs=-1  # Use all CPU cores
            )
        elif self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,  # Slower learning for better generalization
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1
            )
        elif self.model_type == "ensemble":
            # Ensemble of multiple models
            self.rf_model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'
            )
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            )
            self.model = {"rf": self.rf_model, "xgb": self.xgb_model}
        else:
            raise ValueError("model_type must be 'randomforest', 'xgboost', or 'ensemble'")
        
        self.scaler = RobustScaler()
    
    def _engineer_features(self, raw_features: Dict[str, float]) -> Dict[str, float]:
        """
        Engineer additional features from raw input.
        
        Args:
            raw_features: Basic features dictionary
            
        Returns:
            Enhanced features dictionary
        """
        enhanced = raw_features.copy()
        
        # Convert account age to weeks for additional perspective
        enhanced['account_age_weeks'] = raw_features.get('account_age_days', 0) / 7.0
        
        # Follower growth rate estimate (synthetic for demo)
        age_days = max(raw_features.get('account_age_days', 1), 1)
        followers_est = raw_features.get('followers_following_ratio', 0) * 100  # Estimate followers
        enhanced['follower_growth_rate'] = followers_est / age_days
        
        # Engagement consistency score
        engagement = raw_features.get('engagement_per_post', 0)
        post_freq = raw_features.get('post_frequency', 0)
        if post_freq > 0:
            enhanced['engagement_consistency'] = min(engagement / (post_freq + 0.1), 100)
        else:
            enhanced['engagement_consistency'] = 0
        
        # Posting pattern score (regular posting is better)
        if 0.05 <= post_freq <= 2.0:  # Healthy range
            enhanced['posting_pattern_score'] = 1.0
        elif post_freq > 10:  # Spammy
            enhanced['posting_pattern_score'] = 0.0
        else:  # Too infrequent or inactive
            enhanced['posting_pattern_score'] = 0.5
        
        # Social graph score (based on ratio and age)
        ratio = raw_features.get('followers_following_ratio', 0)
        if age_days > 90 and 0.5 <= ratio <= 5.0:
            enhanced['social_graph_score'] = 1.0
        elif ratio > 50 or ratio < 0.01:
            enhanced['social_graph_score'] = 0.0
        else:
            enhanced['social_graph_score'] = 0.5
        
        return enhanced
    
    def generate_enhanced_synthetic_data(self, n_samples: int = 15000) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate enhanced synthetic training data with more realistic patterns.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (features_df, labels)
        """
        np.random.seed(42)
        
        # Generate real profiles (65% of data - more realistic distribution)
        n_real = int(n_samples * 0.65)
        
        # Real profile patterns with more variance
        real_data = {
            'account_age_days': np.random.gamma(3, 150, n_real),  # More mature accounts
            'followers_following_ratio': np.random.lognormal(0.5, 0.8, n_real),  # Log-normal distribution
            'post_frequency': np.random.gamma(1.5, 0.1, n_real),  # Moderate posting
            'engagement_per_post': np.random.gamma(2.5, 8, n_real)  # Varied engagement
        }
        
        # Generate fake profiles (35% of data)
        n_fake = n_samples - n_real
        
        # Create different types of fake profiles
        bot_profiles = n_fake // 3
        spam_profiles = n_fake // 3
        inactive_profiles = n_fake - bot_profiles - spam_profiles
        
        # Bot profiles: New accounts, many followers, low engagement
        bot_data = {
            'account_age_days': np.random.gamma(1, 15, bot_profiles),
            'followers_following_ratio': np.random.gamma(4, 50, bot_profiles),
            'post_frequency': np.random.gamma(1, 0.5, bot_profiles),
            'engagement_per_post': np.random.gamma(1, 1, bot_profiles)
        }
        
        # Spam profiles: High posting frequency, low engagement
        spam_data = {
            'account_age_days': np.random.gamma(1.5, 40, spam_profiles),
            'followers_following_ratio': np.random.gamma(1, 0.1, spam_profiles),
            'post_frequency': np.random.gamma(3, 5, spam_profiles),
            'engagement_per_post': np.random.gamma(1, 0.5, spam_profiles)
        }
        
        # Inactive profiles: Old but suspicious patterns
        inactive_data = {
            'account_age_days': np.random.gamma(2, 100, inactive_profiles),
            'followers_following_ratio': np.random.gamma(1, 0.01, inactive_profiles),
            'post_frequency': np.random.gamma(1, 0.01, inactive_profiles),
            'engagement_per_post': np.random.gamma(1, 0.1, inactive_profiles)
        }
        
        # Combine all fake profiles
        fake_data = {
            'account_age_days': np.concatenate([bot_data['account_age_days'], 
                                              spam_data['account_age_days'], 
                                              inactive_data['account_age_days']]),
            'followers_following_ratio': np.concatenate([bot_data['followers_following_ratio'],
                                                       spam_data['followers_following_ratio'],
                                                       inactive_data['followers_following_ratio']]),
            'post_frequency': np.concatenate([bot_data['post_frequency'],
                                            spam_data['post_frequency'],
                                            inactive_data['post_frequency']]),
            'engagement_per_post': np.concatenate([bot_data['engagement_per_post'],
                                                 spam_data['engagement_per_post'],
                                                 inactive_data['engagement_per_post']])
        }
        
        # Combine real and fake data
        all_data = {
            'account_age_days': np.concatenate([real_data['account_age_days'], fake_data['account_age_days']]),
            'followers_following_ratio': np.concatenate([real_data['followers_following_ratio'], fake_data['followers_following_ratio']]),
            'post_frequency': np.concatenate([real_data['post_frequency'], fake_data['post_frequency']]),
            'engagement_per_post': np.concatenate([real_data['engagement_per_post'], fake_data['engagement_per_post']])
        }
        
        # Create labels
        labels = np.concatenate([np.ones(n_real), np.zeros(n_fake)])
        
        # Create DataFrame and engineer features
        basic_df = pd.DataFrame(all_data)
        enhanced_rows = []
        
        for _, row in basic_df.iterrows():
            enhanced_features = self._engineer_features(row.to_dict())
            enhanced_rows.append(enhanced_features)
        
        enhanced_df = pd.DataFrame(enhanced_rows)
        
        # Shuffle the data
        indices = np.random.permutation(len(enhanced_df))
        enhanced_df = enhanced_df.iloc[indices].reset_index(drop=True)
        labels = labels[indices]
        
        logger.info(f"Generated {len(enhanced_df)} enhanced samples ({n_real} real, {n_fake} fake)")
        return enhanced_df, labels
    
    def train(self, features_df: Optional[pd.DataFrame] = None, labels: Optional[np.ndarray] = None):
        """
        Train the classifier with enhanced features and cross-validation.
        
        Args:
            features_df: Training features DataFrame
            labels: Training labels (1 = real, 0 = fake)
        """
        if features_df is None or labels is None:
            logger.info("Generating enhanced synthetic training data...")
            features_df, labels = self.generate_enhanced_synthetic_data()
        
        # Prepare features
        X = features_df[self.feature_names].copy()
        
        # Handle missing values and outliers
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0.0)
        
        # Apply reasonable clipping
        X = X.clip(lower=0, upper=X.quantile(0.99))
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train model(s)
        if self.model_type == "ensemble":
            # Train both models
            logger.info("Training ensemble models...")
            self.model["rf"].fit(X_train, y_train)
            self.model["xgb"].fit(X_train, y_train)
            
            # Evaluate ensemble
            rf_pred = self.model["rf"].predict_proba(X_test)[:, 1]
            xgb_pred = self.model["xgb"].predict_proba(X_test)[:, 1]
            ensemble_pred = (rf_pred + xgb_pred) / 2
            auc_score = roc_auc_score(y_test, ensemble_pred)
            
        else:
            logger.info(f"Training {self.model_type} classifier...")
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            y_prob = self.model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_prob)
        
        # Cross-validation
        if self.model_type != "ensemble":
            cv_scores = cross_val_score(self.model, X_scaled, labels, cv=5, scoring='roc_auc')
            logger.info(f"Cross-validation AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        logger.info(f"Model trained successfully. AUC Score: {auc_score:.4f}")
        
        # Create metadata
        self.metadata = ModelMetadata(
            model_type=self.model_type,
            training_date=datetime.datetime.now().isoformat(),
            training_samples=len(features_df),
            auc_score=auc_score,
            feature_names=self.feature_names,
            model_hash=self._calculate_model_hash()
        )
        
        self.is_trained = True
        
        # Feature importance
        if self.model_type == "randomforest":
            self._log_feature_importance(self.model.feature_importances_)
        elif self.model_type == "xgboost":
            self._log_feature_importance(self.model.feature_importances_)
        elif self.model_type == "ensemble":
            rf_importance = self.model["rf"].feature_importances_
            xgb_importance = self.model["xgb"].feature_importances_
            avg_importance = (rf_importance + xgb_importance) / 2
            self._log_feature_importance(avg_importance)
    
    def _log_feature_importance(self, importances: np.ndarray):
        """Log feature importance in a readable format."""
        feature_importance = list(zip(self.feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        logger.info("Feature Importances:")
        for feature, importance in feature_importance:
            logger.info(f"  {feature}: {importance:.4f}")
    
    def _calculate_model_hash(self) -> str:
        """Calculate a hash of the model for versioning."""
        model_str = f"{self.model_type}_{datetime.datetime.now().isoformat()}"
        return hashlib.md5(model_str.encode()).hexdigest()[:8]
    
    def predict_probability(self, features: Dict[str, float]) -> float:
        """
        Predict the probability that a profile is real.
        
        Args:
            features: Dictionary with profile features
            
        Returns:
            Probability score (0-1) where 1.0 = very likely real
        """
        if not self.is_trained:
            logger.warning("Model not trained, returning heuristic prediction")
            return self._heuristic_prediction(features)
        
        # Engineer features
        enhanced_features = self._engineer_features(features)
        
        # Prepare feature vector
        feature_vector = np.array([enhanced_features.get(name, 0.0) for name in self.feature_names])
        feature_vector = feature_vector.reshape(1, -1)
        
        # Handle outliers
        feature_vector = np.clip(feature_vector, 0, np.percentile(feature_vector, 99))
        
        # Scale features
        feature_vector = self.scaler.transform(feature_vector)
        
        # Predict
        if self.model_type == "ensemble":
            rf_prob = self.model["rf"].predict_proba(feature_vector)[0, 1]
            xgb_prob = self.model["xgb"].predict_proba(feature_vector)[0, 1]
            prob = (rf_prob + xgb_prob) / 2
        else:
            prob = self.model.predict_proba(feature_vector)[0, 1]
        
        return float(prob)
    
    def _heuristic_prediction(self, features: Dict[str, float]) -> float:
        """Fallback heuristic prediction when model is not trained."""
        enhanced = self._engineer_features(features)
        
        score = 0.5
        score += enhanced.get('posting_pattern_score', 0) * 0.2
        score += enhanced.get('social_graph_score', 0) * 0.2
        score += min(enhanced.get('engagement_consistency', 0) / 10, 0.1)
        
        if enhanced.get('account_age_days', 0) > 365:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def save_model(self, filename: Optional[str] = None):
        """Save the trained model and all components."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_type}_model_{timestamp}.joblib"
        
        filepath = self.models_dir / filename
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'metadata': asdict(self.metadata),
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        
        # Save metadata separately
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(asdict(self.metadata), f, indent=2)
        
        logger.info(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath: Union[str, Path]):
        """Load a trained model from file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        # Load metadata
        if 'metadata' in model_data:
            self.metadata = ModelMetadata(**model_data['metadata'])
        
        logger.info(f"Model loaded from {filepath}")
        if self.metadata:
            logger.info(f"Model info: {self.metadata.model_type}, AUC: {self.metadata.auc_score:.4f}")

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test advanced classifier
    classifier = AdvancedTabularClassifier(model_type="ensemble")
    classifier.train()
    
    # Test prediction with enhanced features
    test_features = {
        'account_age_days': 500,
        'followers_following_ratio': 1.8,
        'post_frequency': 0.15,
        'engagement_per_post': 22.0
    }
    
    prob = classifier.predict_probability(test_features)
    print(f"Enhanced prediction: {prob:.4f}")
    
    # Save model
    model_path = classifier.save_model()
    print(f"Model saved to: {model_path}")