"""
Text Analysis Service using Hugging Face DistilBERT
Provides coherence, grammar, and sentiment analysis with normalized scores (0-1)
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
from functools import lru_cache
import time

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Install with: pip install transformers torch")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextAnalysisService:
    """
    Service for analyzing text using DistilBERT and other transformer models
    Provides sentiment, grammar, and coherence analysis
    """
    
    def __init__(self):
        self.models_loaded = False
        self.sentiment_analyzer = None
        self.grammar_analyzer = None
        self.coherence_analyzer = None
        
        if TRANSFORMERS_AVAILABLE:
            self._load_models()
        else:
            logger.warning("Transformers not available - using mock analysis")
    
    def _load_models(self):
        """Load all required transformer models"""
        try:
            logger.info("Loading transformer models...")
            
            # Sentiment Analysis - DistilBERT fine-tuned on sentiment
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=True
            )
            
            # Grammar Analysis - Using a grammar correction model
            # Note: This is a placeholder - you might want to use a dedicated grammar model
            self.grammar_analyzer = pipeline(
                "text-classification",
                model="martin-ha/toxic-comment-model",  # We'll use this as a proxy for text quality
                return_all_scores=True
            )
            
            # Coherence Analysis - Using a general text classification model
            # For coherence, we'll analyze text structure and flow
            self.coherence_analyzer = pipeline(
                "sentiment-analysis",  # We'll repurpose this for coherence analysis
                model="distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=True
            )
            
            self.models_loaded = True
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.models_loaded = False
    
    @lru_cache(maxsize=128)
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using DistilBERT
        Returns normalized sentiment scores (0-1)
        """
        if not self.models_loaded or not text.strip():
            return self._mock_sentiment_analysis(text)
        
        try:
            results = self.sentiment_analyzer(text)
            
            # Convert to normalized scores
            sentiment_scores = {}
            for result in results[0]:  # First prediction
                label = result['label'].lower()
                score = result['score']
                
                if label == 'positive':
                    sentiment_scores['positive'] = score
                    sentiment_scores['negative'] = 1 - score
                elif label == 'negative':
                    sentiment_scores['negative'] = score
                    sentiment_scores['positive'] = 1 - score
            
            # Calculate overall sentiment score (0 = very negative, 1 = very positive)
            sentiment_scores['overall'] = sentiment_scores.get('positive', 0.5)
            
            return sentiment_scores
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self._mock_sentiment_analysis(text)
    
    def analyze_grammar(self, text: str) -> Dict[str, float]:
        """
        Analyze grammar quality of text
        Returns normalized grammar scores (0-1)
        """
        if not self.models_loaded or not text.strip():
            return self._mock_grammar_analysis(text)
        
        try:
            # Basic grammar analysis using rule-based approach combined with model
            grammar_score = self._calculate_grammar_score(text)
            
            # Use the toxicity model as a proxy for text quality
            results = self.grammar_analyzer(text)
            
            # Convert toxicity score to grammar quality (inverse relationship)
            model_score = 1.0
            for result in results[0]:
                if result['label'] == 'TOXIC':
                    model_score = 1 - result['score']  # Inverse toxicity = quality
                    break
            
            # Combine rule-based and model-based scores
            combined_score = (grammar_score + model_score) / 2
            
            return {
                'grammar_score': combined_score,
                'rule_based': grammar_score,
                'model_based': model_score,
                'errors_detected': self._count_grammar_errors(text)
            }
            
        except Exception as e:
            logger.error(f"Error in grammar analysis: {e}")
            return self._mock_grammar_analysis(text)
    
    def analyze_coherence(self, text: str) -> Dict[str, float]:
        """
        Analyze coherence and flow of text
        Returns normalized coherence scores (0-1)
        """
        if not self.models_loaded or not text.strip():
            return self._mock_coherence_analysis(text)
        
        try:
            # Analyze text structure and coherence
            coherence_score = self._calculate_coherence_score(text)
            
            # Use sentiment model to analyze text flow consistency
            sentences = self._split_into_sentences(text)
            if len(sentences) > 1:
                sentence_scores = []
                for sentence in sentences:
                    if sentence.strip():
                        result = self.coherence_analyzer(sentence)
                        # Use confidence as a proxy for coherence
                        max_confidence = max([r['score'] for r in result[0]])
                        sentence_scores.append(max_confidence)
                
                model_coherence = np.mean(sentence_scores) if sentence_scores else 0.5
                # Lower variance in confidence indicates better coherence
                coherence_variance = np.var(sentence_scores) if len(sentence_scores) > 1 else 0
                model_coherence = model_coherence * (1 - min(coherence_variance, 0.5))
            else:
                model_coherence = 0.5
            
            # Combine structural and model-based coherence
            combined_score = (coherence_score + model_coherence) / 2
            
            return {
                'coherence_score': combined_score,
                'structural': coherence_score,
                'model_based': model_coherence,
                'sentence_count': len(sentences),
                'avg_sentence_length': np.mean([len(s.split()) for s in sentences if s.strip()])
            }
            
        except Exception as e:
            logger.error(f"Error in coherence analysis: {e}")
            return self._mock_coherence_analysis(text)
    
    def analyze_text_complete(self, text: str) -> Dict[str, any]:
        """
        Complete text analysis including sentiment, grammar, and coherence
        Returns comprehensive analysis with normalized scores
        """
        start_time = time.time()
        
        if not text or not text.strip():
            return {
                'error': 'Empty or invalid text provided',
                'sentiment': {},
                'grammar': {},
                'coherence': {},
                'overall_score': 0.0,
                'processing_time': 0.0
            }
        
        # Perform all analyses
        sentiment = self.analyze_sentiment(text)
        grammar = self.analyze_grammar(text)
        coherence = self.analyze_coherence(text)
        
        # Calculate overall combined score
        overall_score = (
            sentiment.get('overall', 0.5) * 0.3 +  # 30% weight for sentiment
            grammar.get('grammar_score', 0.5) * 0.4 +  # 40% weight for grammar
            coherence.get('coherence_score', 0.5) * 0.3  # 30% weight for coherence
        )
        
        processing_time = time.time() - start_time
        
        return {
            'sentiment': sentiment,
            'grammar': grammar,
            'coherence': coherence,
            'overall_score': round(overall_score, 4),
            'text_length': len(text),
            'word_count': len(text.split()),
            'processing_time': round(processing_time, 4),
            'model_status': 'loaded' if self.models_loaded else 'mock'
        }
    
    def _calculate_grammar_score(self, text: str) -> float:
        """Calculate grammar score using rule-based approach"""
        if not text.strip():
            return 0.0
        
        score = 1.0
        
        # Check for basic grammar rules
        sentences = self._split_into_sentences(text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Penalize sentences that don't start with capital letter
            if sentence[0].islower():
                score -= 0.1
            
            # Penalize sentences without proper ending punctuation
            if not sentence.endswith(('.', '!', '?')):
                score -= 0.1
            
            # Check for repeated words
            words = sentence.lower().split()
            if len(words) != len(set(words)):
                score -= 0.05
            
            # Penalize very short or very long sentences
            if len(words) < 3:
                score -= 0.1
            elif len(words) > 50:
                score -= 0.05
        
        return max(score, 0.0)
    
    def _calculate_coherence_score(self, text: str) -> float:
        """Calculate coherence score using structural analysis"""
        if not text.strip():
            return 0.0
        
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return 0.7  # Single sentence gets neutral score
        
        score = 1.0
        
        # Analyze sentence length variance (more consistent = more coherent)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if len(sentence_lengths) > 1:
            length_variance = np.var(sentence_lengths) / np.mean(sentence_lengths)
            if length_variance > 1.0:  # High variance
                score -= 0.2
        
        # Check for transition words/phrases
        transition_words = {
            'however', 'therefore', 'furthermore', 'moreover', 'consequently',
            'additionally', 'meanwhile', 'nevertheless', 'thus', 'hence',
            'in contrast', 'on the other hand', 'as a result', 'for example'
        }
        
        text_lower = text.lower()
        transition_count = sum(1 for word in transition_words if word in text_lower)
        transition_ratio = transition_count / len(sentences)
        
        # Optimal transition ratio is around 0.2-0.4
        if transition_ratio < 0.1:
            score -= 0.1
        elif transition_ratio > 0.6:
            score -= 0.1
        else:
            score += 0.1
        
        return max(min(score, 1.0), 0.0)
    
    def _count_grammar_errors(self, text: str) -> int:
        """Count basic grammar errors"""
        errors = 0
        
        # Check for common errors
        if re.search(r'\bi\s', text):  # Lowercase 'i'
            errors += text.lower().count(' i ')
        
        # Multiple spaces
        if '  ' in text:
            errors += len(re.findall(r'\s{2,}', text))
        
        # Missing spaces after punctuation
        errors += len(re.findall(r'[.!?][a-zA-Z]', text))
        
        return errors
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    # Mock analysis methods for when transformers are not available
    def _mock_sentiment_analysis(self, text: str) -> Dict[str, float]:
        """Mock sentiment analysis"""
        # Simple rule-based sentiment
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'like', 'happy']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'disappointed']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count + neg_count == 0:
            return {'positive': 0.5, 'negative': 0.5, 'overall': 0.5}
        
        positive_score = pos_count / (pos_count + neg_count)
        return {
            'positive': positive_score,
            'negative': 1 - positive_score,
            'overall': positive_score
        }
    
    def _mock_grammar_analysis(self, text: str) -> Dict[str, float]:
        """Mock grammar analysis"""
        grammar_score = self._calculate_grammar_score(text)
        return {
            'grammar_score': grammar_score,
            'rule_based': grammar_score,
            'model_based': grammar_score,
            'errors_detected': self._count_grammar_errors(text)
        }
    
    def _mock_coherence_analysis(self, text: str) -> Dict[str, float]:
        """Mock coherence analysis"""
        coherence_score = self._calculate_coherence_score(text)
        sentences = self._split_into_sentences(text)
        return {
            'coherence_score': coherence_score,
            'structural': coherence_score,
            'model_based': coherence_score,
            'sentence_count': len(sentences),
            'avg_sentence_length': np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0
        }

# Global service instance
text_analysis_service = TextAnalysisService()

def analyze_text(text: str) -> Dict[str, any]:
    """Convenience function for complete text analysis"""
    return text_analysis_service.analyze_text_complete(text)

def analyze_sentiment(text: str) -> Dict[str, float]:
    """Convenience function for sentiment analysis only"""
    return text_analysis_service.analyze_sentiment(text)

def analyze_grammar(text: str) -> Dict[str, float]:
    """Convenience function for grammar analysis only"""
    return text_analysis_service.analyze_grammar(text)

def analyze_coherence(text: str) -> Dict[str, float]:
    """Convenience function for coherence analysis only"""
    return text_analysis_service.analyze_coherence(text)