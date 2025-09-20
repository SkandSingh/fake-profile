from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import logging
import re
import random
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Text Analysis API",
    description="AI-powered text analysis for fake profile detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
    include_details: bool = Field(default=True, description="Include detailed analysis breakdown")

class TextAnalysisResponse(BaseModel):
    analysis_id: str
    text_length: int
    sentiment: Dict[str, float]
    grammar: Dict[str, float] 
    coherence: Dict[str, float]
    overall_score: float
    risk_level: str
    explanation: List[str]
    processing_time: float

def analyze_text_simple(text: str) -> Dict:
    positive_words = ['good', 'great', 'awesome', 'amazing', 'love', 'best', 'excellent', 'wonderful', 'fantastic', 'happy']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'sad', 'angry', 'disappointed', 'frustrated']
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    total_sentiment_words = positive_count + negative_count
    if total_sentiment_words > 0:
        positive_ratio = positive_count / total_sentiment_words
        negative_ratio = negative_count / total_sentiment_words
        neutral_ratio = 1 - positive_ratio - negative_ratio
    else:
        positive_ratio = 0.4
        negative_ratio = 0.2
        neutral_ratio = 0.4
    
    sentences = text.split('.')
    avg_sentence_length = len(text.split()) / max(len(sentences), 1)
    
    cap_count = sum(1 for c in text if c.isupper())
    cap_ratio = cap_count / max(len(text), 1)
    
    grammar_score = min(1.0, max(0.0, 1.0 - abs(avg_sentence_length - 15) / 20))
    if cap_ratio > 0.3:
        grammar_score *= 0.7
    
    words = text.lower().split()
    unique_words = len(set(words))
    word_diversity = unique_words / max(len(words), 1)
    
    # Coherence score based on word diversity and sentence flow
    coherence_score = min(1.0, word_diversity * 1.2)
    
    # Overall authenticity score
    authenticity_factors = [
        grammar_score,
        coherence_score,
        1.0 - abs(positive_ratio - 0.3),
        min(1.0, len(text) / 50)
    ]
    overall_score = sum(authenticity_factors) / len(authenticity_factors)
    
    if overall_score > 0.7:
        risk_level = "low"
        explanation = ["Text appears authentic", "Good grammar and coherence", "Natural sentiment patterns"]
    elif overall_score > 0.4:
        risk_level = "medium"
        explanation = ["Some authenticity concerns", "Grammar or coherence issues detected", "Monitor for additional signals"]
    else:
        risk_level = "high"
        explanation = ["Multiple authenticity issues", "Poor grammar or coherence", "Potential fake content detected"]
    
    return {
        "sentiment": {
            "positive": positive_ratio,
            "negative": negative_ratio,
            "neutral": neutral_ratio,
            "overall": positive_ratio - negative_ratio + 0.5
        },
        "grammar": {
            "grammar_score": grammar_score,
            "avg_sentence_length": avg_sentence_length,
            "capitalization_ratio": cap_ratio
        },
        "coherence": {
            "coherence_score": coherence_score,
            "word_diversity": word_diversity,
            "unique_word_ratio": word_diversity
        },
        "overall_score": overall_score,
        "risk_level": risk_level,
        "explanation": explanation
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "text_analysis", "timestamp": datetime.now().isoformat()}

@app.post("/analyze", response_model=TextAnalysisResponse)
async def analyze_text(request: TextAnalysisRequest):
    try:
        start_time = datetime.now()
        
        if not request.text or len(request.text.strip()) < 1:
            raise HTTPException(status_code=400, detail="Text content is required and cannot be empty")
        
        analysis_result = analyze_text_simple(request.text)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = TextAnalysisResponse(
            analysis_id=f"txt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
            text_length=len(request.text),
            sentiment=analysis_result["sentiment"],
            grammar=analysis_result["grammar"],
            coherence=analysis_result["coherence"],
            overall_score=analysis_result["overall_score"],
            risk_level=analysis_result["risk_level"],
            explanation=analysis_result["explanation"],
            processing_time=processing_time
        )
        
        logger.info(f"Text analysis completed. Score: {analysis_result['overall_score']:.3f}, Risk: {analysis_result['risk_level']}")
        return response
        
    except Exception as e:
        logger.error(f"Error in text analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/batch_analyze")
async def batch_analyze_text(texts: List[str]):
    """Analyze multiple texts in batch"""
    results = []
    for i, text in enumerate(texts):
        try:
            request = TextAnalysisRequest(text=text)
            result = await analyze_text(request)
            results.append({"index": i, "result": result})
        except Exception as e:
            results.append({"index": i, "error": str(e)})
    
    return {"batch_results": results, "total_processed": len(texts)}

if __name__ == "__main__":
    import uvicorn
    import os
    
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(app, host=host, port=port)