import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import datetime

logger = logging.getLogger(__name__)

class TabularProfileClassifier:
    - post_frequency: posts_count / max(account_age_days, 1)
    - engagement_per_post: (total_likes + total_comments) / max(posts_count, 1)
    """
    
    def __init__(self, model_type: str = "randomforest"):
        """
        Initialize the tabular profile classifier.
        
        Args:
            model_type: Either "randomforest" or "xgboost"
        """
        self.model_type = model_type.lower()
        self.model = None
        self.scaler = None
        self.feature_names = [
            'account_age_days',
            'followers_following_ratio', 
            'post_frequency',
            'engagement_per_post'
        ]
        self.is_trained = False
        
        # Initialize model
        if self.model_type == "randomforest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
        else:
            raise ValueError("model_type must be 'randomforest' or 'xgboost'")
        
        # Use RobustScaler to handle outliers better
        self.scaler = RobustScaler()
        
        logger.info(f"Initialized {self.model_type} tabular classifier")
    
    def generate_synthetic_data(self, n_samples: int = 10000) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate synthetic training data with realistic patterns.
        
        Real profiles typically have:
        - Longer account age
        - Balanced followers/following ratio
        - Consistent posting frequency
        - Good engagement rates
        
        Fake profiles typically have:
        - Recent account creation
        - High followers but low following OR very high following
        - Irregular posting patterns
        - Low engagement rates
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (features_df, labels)
        """
        np.random.seed(42)
        
        # Generate real profiles (60% of data)
        n_real = int(n_samples * 0.6)
        
        # Real profile patterns
        real_account_age = np.random.gamma(2, 200, n_real)  # Longer account age
        real_followers = np.random.gamma(2, 500, n_real)
        real_following = np.random.gamma(1.5, 300, n_real)
        real_posts = np.random.gamma(1.8, 50, n_real)
        real_total_engagement = real_posts * np.random.gamma(2, 10, n_real)
        
        # Calculate ratios and frequencies for real profiles
        real_followers_following_ratio = real_followers / np.maximum(real_following, 1)
        real_post_frequency = real_posts / np.maximum(real_account_age, 1)
        real_engagement_per_post = real_total_engagement / np.maximum(real_posts, 1)
        
        # Generate fake profiles (40% of data)
        n_fake = n_samples - n_real
        
        # Fake profile patterns - more extreme and suspicious
        fake_account_age = np.random.gamma(1, 30, n_fake)  # Newer accounts
        
        # Mix of bot patterns: some with many followers, few following
        # Others with few followers, many following
        fake_followers = np.concatenate([
            np.random.gamma(3, 1000, n_fake//2),  # Some bots buy followers
            np.random.gamma(1, 50, n_fake//2)     # Others have few followers
        ])
        fake_following = np.concatenate([
            np.random.gamma(1, 100, n_fake//2),   # Bots with many followers follow few
            np.random.gamma(2, 500, n_fake//2)    # Others follow many
        ])
        
        # Irregular posting patterns
        fake_posts = np.random.gamma(1, 20, n_fake)
        
        # Lower engagement rates
        fake_total_engagement = fake_posts * np.random.gamma(1, 3, n_fake)
        
        # Calculate ratios for fake profiles
        fake_followers_following_ratio = fake_followers / np.maximum(fake_following, 1)
        fake_post_frequency = fake_posts / np.maximum(fake_account_age, 1)
        fake_engagement_per_post = fake_total_engagement / np.maximum(fake_posts, 1)
        
        # Combine data
        account_age_days = np.concatenate([real_account_age, fake_account_age])
        followers_following_ratio = np.concatenate([real_followers_following_ratio, fake_followers_following_ratio])
        post_frequency = np.concatenate([real_post_frequency, fake_post_frequency])
        engagement_per_post = np.concatenate([real_engagement_per_post, fake_engagement_per_post])
        
        # Create labels (1 = real, 0 = fake)
        labels = np.concatenate([np.ones(n_real), np.zeros(n_fake)])
        
        # Create DataFrame
        features_df = pd.DataFrame({
            'account_age_days': account_age_days,
            'followers_following_ratio': followers_following_ratio,
            'post_frequency': post_frequency,
            'engagement_per_post': engagement_per_post
        })
        
        # Shuffle the data
        indices = np.random.permutation(len(features_df))
        features_df = features_df.iloc[indices].reset_index(drop=True)
        labels = labels[indices]
        
        logger.info(f"Generated {len(features_df)} synthetic samples ({n_real} real, {n_fake} fake)")
        return features_df, labels
    
    def preprocess_features(self, features: Dict[str, float]) -> np.ndarray:
        """
        Preprocess raw features into model-ready format.
        
        Args:
            features: Dictionary with keys:
                - account_age_days
                - followers_following_ratio
                - post_frequency
                - engagement_per_post
                
        Returns:
            Preprocessed feature array
        """
        # Convert to DataFrame for consistency
        df = pd.DataFrame([features])
        
        # Handle missing values
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0.0
        
        # Ensure correct order
        df = df[self.feature_names]
        
        # Handle infinite values and outliers
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0.0)
        
        # Clip extreme outliers
        df['followers_following_ratio'] = np.clip(df['followers_following_ratio'], 0, 100)
        df['post_frequency'] = np.clip(df['post_frequency'], 0, 10)
        df['engagement_per_post'] = np.clip(df['engagement_per_post'], 0, 1000)
        
        # Scale features if scaler is fitted
        if self.scaler is not None and hasattr(self.scaler, 'scale_'):
            features_scaled = self.scaler.transform(df)
        else:
            features_scaled = df.values
        
        return features_scaled
    
    def train(self, features_df: Optional[pd.DataFrame] = None, labels: Optional[np.ndarray] = None):
        """
        Train the classifier on the provided data or generate synthetic data.
        
        Args:
            features_df: Training features DataFrame
            labels: Training labels (1 = real, 0 = fake)
        """
        if features_df is None or labels is None:
            logger.info("No training data provided, generating synthetic data...")
            features_df, labels = self.generate_synthetic_data()
        
        # Prepare features
        X = features_df[self.feature_names].copy()
        
        # Handle missing values and outliers
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0.0)
        
        # Clip extreme outliers
        X['followers_following_ratio'] = np.clip(X['followers_following_ratio'], 0, 100)
        X['post_frequency'] = np.clip(X['post_frequency'], 0, 10)
        X['engagement_per_post'] = np.clip(X['engagement_per_post'], 0, 1000)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train model
        logger.info(f"Training {self.model_type} classifier...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        auc_score = roc_auc_score(y_test, y_prob)
        logger.info(f"Model trained successfully. AUC Score: {auc_score:.4f}")
        
        # Print classification report
        report = classification_report(y_test, y_pred, target_names=['Fake', 'Real'])
        logger.info(f"Classification Report:\n{report}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = list(zip(self.feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            logger.info("Feature Importances:")
            for feature, importance in feature_importance:
                logger.info(f"  {feature}: {importance:.4f}")
        
        self.is_trained = True
    
    def predict_probability(self, features: Dict[str, float]) -> float:
        """
        Predict the probability that a profile is real.
        
        Args:
            features: Dictionary with profile features
            
        Returns:
            Probability score (0-1) where 1.0 = very likely real
        """
        if not self.is_trained:
            logger.warning("Model not trained, returning mock prediction")
            # Mock prediction based on simple heuristics
            account_age = features.get('account_age_days', 0)
            ratio = features.get('followers_following_ratio', 0)
            engagement = features.get('engagement_per_post', 0)
            
            # Simple heuristic scoring
            score = 0.5
            if account_age > 100: score += 0.2
            if 0.1 <= ratio <= 10: score += 0.2
            if engagement > 5: score += 0.1
            
            return min(max(score, 0.0), 1.0)
        
        # Preprocess features
        X = self.preprocess_features(features)
        
        # Predict probability
        prob = self.model.predict_proba(X)[0, 1]  # Probability of class 1 (real)
        
        return float(prob)
    
    def predict_batch(self, features_list: List[Dict[str, float]]) -> List[float]:
        """
        Predict probabilities for a batch of profiles.
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            List of probability scores
        """
        if not features_list:
            return []
        
        if not self.is_trained:
            logger.warning("Model not trained, returning mock predictions")
            return [self.predict_probability(features) for features in features_list]
        
        # Process all features
        processed_features = []
        for features in features_list:
            X = self.preprocess_features(features)
            processed_features.append(X[0])
        
        X_batch = np.array(processed_features)
        
        # Predict probabilities
        probs = self.model.predict_proba(X_batch)[:, 1]
        
        return probs.tolist()
    
    def save_model(self, filepath: str):
        """Save the trained model and scaler."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model and scaler."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test RandomForest classifier
    print("Testing RandomForest Classifier:")
    rf_classifier = TabularProfileClassifier(model_type="randomforest")
    rf_classifier.train()
    
    # Test prediction
    test_features = {
        'account_age_days': 365,
        'followers_following_ratio': 2.5,
        'post_frequency': 0.1,
        'engagement_per_post': 15.0
    }
    
    prob = rf_classifier.predict_probability(test_features)
    print(f"Test prediction: {prob:.4f} (probability of being real)")
    
    # Test XGBoost classifier
    print("\nTesting XGBoost Classifier:")
    xgb_classifier = TabularProfileClassifier(model_type="xgboost")
    xgb_classifier.train()
    
    prob = xgb_classifier.predict_probability(test_features)
    print(f"Test prediction: {prob:.4f} (probability of being real)")