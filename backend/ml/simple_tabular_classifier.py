"""
Simple Tabular Profile Classifier (Minimal Dependencies)
========================================================

A lightweight implementation using only core libraries for testing.
"""

import json
import math
import random
from typing import Dict, List

class SimpleTabularClassifier:
    """
    A simple rule-based classifier for tabular profile features.
    This is a fallback implementation that doesn't require scikit-learn.
    """
    
    def __init__(self):
        self.feature_weights = {
            'account_age_days': 0.3,
            'followers_following_ratio': 0.25,
            'post_frequency': 0.25,
            'engagement_per_post': 0.2
        }
        self.thresholds = {
            'account_age_days': 30,  # Accounts older than 30 days are more likely real
            'followers_following_ratio': 0.1,  # Healthy ratio > 0.1
            'post_frequency': 5.0,  # Posting more than 5 times/day is suspicious
            'engagement_per_post': 5.0  # Good engagement > 5
        }
    
    def predict_probability(self, features: Dict[str, float]) -> float:
        """
        Predict probability that a profile is real using rule-based scoring.
        
        Args:
            features: Dictionary with profile features
            
        Returns:
            Probability score (0-1) where 1.0 = very likely real
        """
        score = 0.5  # Start with neutral score
        
        # Account age scoring
        age = features.get('account_age_days', 0)
        if age > 365:  # > 1 year
            score += 0.2
        elif age > 90:  # > 3 months
            score += 0.1
        elif age < 7:  # < 1 week
            score -= 0.2
        
        # Followers/following ratio scoring
        ratio = features.get('followers_following_ratio', 0)
        if 0.5 <= ratio <= 10:  # Healthy ratio
            score += 0.15
        elif ratio > 50:  # Too many followers vs following (bought followers)
            score -= 0.1
        elif ratio < 0.01:  # Too many following vs followers (bot behavior)
            score -= 0.15
        
        # Post frequency scoring
        freq = features.get('post_frequency', 0)
        if 0.05 <= freq <= 2:  # 1 post every 20 days to 2 posts per day
            score += 0.1
        elif freq > 10:  # More than 10 posts per day
            score -= 0.2
        
        # Engagement scoring
        engagement = features.get('engagement_per_post', 0)
        if engagement > 20:  # High engagement
            score += 0.15
        elif engagement > 5:  # Moderate engagement
            score += 0.1
        elif engagement < 1:  # Very low engagement
            score -= 0.1
        
        # Add some randomness for demonstration
        score += random.uniform(-0.05, 0.05)
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))
    
    def predict_batch(self, features_list: List[Dict[str, float]]) -> List[float]:
        """Predict probabilities for a batch of profiles."""
        return [self.predict_probability(features) for features in features_list]

# Test the simple classifier
if __name__ == "__main__":
    classifier = SimpleTabularClassifier()
    
    # Test real profile
    real_features = {
        'account_age_days': 500,
        'followers_following_ratio': 1.5,
        'post_frequency': 0.2,
        'engagement_per_post': 25.0
    }
    
    # Test fake profile
    fake_features = {
        'account_age_days': 3,
        'followers_following_ratio': 0.005,
        'post_frequency': 15.0,
        'engagement_per_post': 0.5
    }
    
    real_prob = classifier.predict_probability(real_features)
    fake_prob = classifier.predict_probability(fake_features)
    
    print(f"Real profile probability: {real_prob:.4f}")
    print(f"Fake profile probability: {fake_prob:.4f}")
    print("✅ Simple classifier working!")