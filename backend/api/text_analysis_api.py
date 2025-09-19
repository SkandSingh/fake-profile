from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
import logging
import asyncio
from datetime import datetime
import uuid

try:
    from ml.text_analysis import text_analysis_service, analyze_text
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ml.text_analysis import text_analysis_service, analyze_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Text Analysis API",
    description="AI-powered text analysis using DistilBERT",
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
    
    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()

class BatchTextAnalysisRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=50, description="List of texts to analyze")
    include_details: bool = Field(default=True, description="Include detailed analysis breakdown")
    
    @validator('texts')
    def texts_must_be_valid(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')
        
        valid_texts = []
        for text in v:
            if text and text.strip():
                if len(text.strip()) > 10000:
                    raise ValueError('Each text must be less than 10,000 characters')
                valid_texts.append(text.strip())
        
        if not valid_texts:
            raise ValueError('At least one valid text is required')
        
        return valid_texts

class SentimentScore(BaseModel):
    positive: float = Field(..., ge=0, le=1, description="Positive sentiment score (0-1)")
    negative: float = Field(..., ge=0, le=1, description="Negative sentiment score (0-1)")
    overall: float = Field(..., ge=0, le=1, description="Overall sentiment score (0-1)")

class GrammarScore(BaseModel):
    grammar_score: float = Field(..., ge=0, le=1, description="Overall grammar score (0-1)")
    rule_based: float = Field(..., ge=0, le=1, description="Rule-based grammar score")
    model_based: float = Field(..., ge=0, le=1, description="Model-based grammar score")
    errors_detected: int = Field(..., ge=0, description="Number of grammar errors detected")

class CoherenceScore(BaseModel):
    coherence_score: float = Field(..., ge=0, le=1, description="Overall coherence score (0-1)")
    structural: float = Field(..., ge=0, le=1, description="Structural coherence score")
    model_based: float = Field(..., ge=0, le=1, description="Model-based coherence score")
    sentence_count: int = Field(..., ge=0, description="Number of sentences")
    avg_sentence_length: float = Field(..., ge=0, description="Average sentence length in words")

class TextAnalysisResponse(BaseModel):
    sentiment: SentimentScore
    grammar: GrammarScore
    coherence: CoherenceScore
    overall_score: float = Field(..., ge=0, le=1, description="Combined overall score (0-1)")
    text_length: int = Field(..., ge=0, description="Length of analyzed text")
    word_count: int = Field(..., ge=0, description="Number of words in text")
    processing_time: float = Field(..., ge=0, description="Processing time in seconds")
    model_status: str = Field(..., description="Status of AI models (loaded/mock)")
    analysis_id: str = Field(..., description="Unique analysis identifier")
    timestamp: datetime = Field(..., description="Analysis timestamp")

class BatchAnalysisItem(BaseModel):
    index: int = Field(..., description="Index of the text in the batch")
    text_preview: str = Field(..., description="Preview of the analyzed text")
    analysis: TextAnalysisResponse
    
class BatchTextAnalysisResponse(BaseModel):
    results: List[BatchAnalysisItem]
    total_texts: int = Field(..., description="Total number of texts analyzed")
    avg_processing_time: float = Field(..., description="Average processing time per text")
    total_processing_time: float = Field(..., description="Total processing time")
    batch_id: str = Field(..., description="Unique batch identifier")
    timestamp: datetime = Field(..., description="Batch analysis timestamp")

class HealthResponse(BaseModel):
    status: str = Field(..., description="API status")
    model_status: str = Field(..., description="AI models status")
    uptime: str = Field(..., description="API uptime")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Health check timestamp")

# Global variables for tracking
app_start_time = datetime.now()
analysis_count = 0

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("Starting Text Analysis API...")
    logger.info(f"Model status: {'loaded' if text_analysis_service.models_loaded else 'mock'}")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Text Analysis API using DistilBERT",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = datetime.now() - app_start_time
    
    return HealthResponse(
        status="healthy",
        model_status="loaded" if text_analysis_service.models_loaded else "mock",
        uptime=str(uptime),
        version="1.0.0",
        timestamp=datetime.now()
    )

@app.post("/analyze", response_model=TextAnalysisResponse)
async def analyze_single_text(request: TextAnalysisRequest):
    """
    Analyze a single text for sentiment, grammar, and coherence
    Returns normalized scores (0-1) for all metrics
    """
    global analysis_count
    analysis_count += 1
    
    try:
        # Perform text analysis
        result = analyze_text(request.text)
        
        # Create response with proper structure
        analysis_id = str(uuid.uuid4())
        
        response = TextAnalysisResponse(
            sentiment=SentimentScore(**result['sentiment']),
            grammar=GrammarScore(**result['grammar']),
            coherence=CoherenceScore(**result['coherence']),
            overall_score=result['overall_score'],
            text_length=result['text_length'],
            word_count=result['word_count'],
            processing_time=result['processing_time'],
            model_status=result['model_status'],
            analysis_id=analysis_id,
            timestamp=datetime.now()
        )
        
        logger.info(f"Analysis completed for text (length: {len(request.text)}, score: {result['overall_score']})")
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/batch", response_model=BatchTextAnalysisResponse)
async def analyze_batch_texts(request: BatchTextAnalysisRequest):
    """
    Analyze multiple texts in batch
    Returns analysis for each text with batch summary
    """
    global analysis_count
    
    try:
        batch_start_time = datetime.now()
        batch_id = str(uuid.uuid4())
        results = []
        
        # Process each text
        for i, text in enumerate(request.texts):
            analysis_count += 1
            
            try:
                result = analyze_text(text)
                analysis_id = str(uuid.uuid4())
                
                analysis_response = TextAnalysisResponse(
                    sentiment=SentimentScore(**result['sentiment']),
                    grammar=GrammarScore(**result['grammar']),
                    coherence=CoherenceScore(**result['coherence']),
                    overall_score=result['overall_score'],
                    text_length=result['text_length'],
                    word_count=result['word_count'],
                    processing_time=result['processing_time'],
                    model_status=result['model_status'],
                    analysis_id=analysis_id,
                    timestamp=datetime.now()
                )
                
                batch_item = BatchAnalysisItem(
                    index=i,
                    text_preview=text[:100] + "..." if len(text) > 100 else text,
                    analysis=analysis_response
                )
                
                results.append(batch_item)
                
            except Exception as e:
                logger.error(f"Error analyzing text {i}: {e}")
                # Continue with other texts even if one fails
                continue
        
        batch_end_time = datetime.now()
        total_processing_time = (batch_end_time - batch_start_time).total_seconds()
        avg_processing_time = total_processing_time / len(results) if results else 0
        
        response = BatchTextAnalysisResponse(
            results=results,
            total_texts=len(results),
            avg_processing_time=avg_processing_time,
            total_processing_time=total_processing_time,
            batch_id=batch_id,
            timestamp=datetime.now()
        )
        
        logger.info(f"Batch analysis completed: {len(results)} texts processed in {total_processing_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@app.get("/analyze/sentiment", response_model=Dict[str, Any])
async def analyze_sentiment_only(text: str):
    """Analyze sentiment only for a given text"""
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text parameter is required")
    
    try:
        result = text_analysis_service.analyze_sentiment(text.strip())
        return {
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "sentiment": result,
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")

@app.get("/analyze/grammar", response_model=Dict[str, Any])
async def analyze_grammar_only(text: str):
    """Analyze grammar only for a given text"""
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text parameter is required")
    
    try:
        result = text_analysis_service.analyze_grammar(text.strip())
        return {
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "grammar": result,
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error in grammar analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Grammar analysis failed: {str(e)}")

@app.get("/analyze/coherence", response_model=Dict[str, Any])
async def analyze_coherence_only(text: str):
    """Analyze coherence only for a given text"""
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text parameter is required")
    
    try:
        result = text_analysis_service.analyze_coherence(text.strip())
        return {
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "coherence": result,
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error in coherence analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Coherence analysis failed: {str(e)}")

@app.get("/stats", response_model=Dict[str, Any])
async def get_api_stats():
    """Get API usage statistics"""
    uptime = datetime.now() - app_start_time
    
    return {
        "total_analyses": analysis_count,
        "uptime": str(uptime),
        "uptime_seconds": uptime.total_seconds(),
        "model_status": "loaded" if text_analysis_service.models_loaded else "mock",
        "api_version": "1.0.0",
        "timestamp": datetime.now()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")