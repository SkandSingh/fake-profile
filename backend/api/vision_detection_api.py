from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
import logging
import asyncio
from datetime import datetime
import uuid
import base64
import io
from PIL import Image
import json

try:
    from ml.vision_detection import get_detector, detect_fake_profile
    from ml.image_preprocessing import load_and_preprocess_image, validate_image_input
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ml.vision_detection import get_detector, detect_fake_profile
    from ml.image_preprocessing import load_and_preprocess_image, validate_image_input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vision Fake Profile Detection API",
    description="AI-powered vision API for detecting fake/AI-generated profile pictures",
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

# Pydantic models for request/response validation
class VisionDetectionRequest(BaseModel):
    image_url: Optional[str] = Field(None, description="URL of the image to analyze")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image data")
    model_type: str = Field("resnet50", description="Model type: resnet50 or efficientnet")
    enhance_image: bool = Field(True, description="Apply image enhancement preprocessing")
    detect_face: bool = Field(True, description="Detect and crop face region if present")
    
    @validator('model_type')
    def validate_model_type(cls, v):
        if v not in ['resnet50', 'efficientnet']:
            raise ValueError('model_type must be either "resnet50" or "efficientnet"')
        return v
    
    @validator('image_url', 'image_base64')
    def validate_image_input(cls, v, values):
        if not v and not values.get('image_url') and not values.get('image_base64'):
            raise ValueError('Either image_url or image_base64 must be provided')
        return v

class BatchVisionDetectionRequest(BaseModel):
    images: List[Dict[str, Any]] = Field(..., description="List of image objects with url or base64 data")
    model_type: str = Field("resnet50", description="Model type: resnet50 or efficientnet")
    enhance_image: bool = Field(True, description="Apply image enhancement preprocessing")
    detect_face: bool = Field(True, description="Detect and crop face region if present")
    max_concurrent: int = Field(5, ge=1, le=20, description="Maximum concurrent processing")

class VisionDetectionScore(BaseModel):
    
    model_config = {"protected_namespaces": ()}
    
    is_fake: bool = Field(..., description="Whether the image is classified as fake/AI-generated")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the classification")
    fake_probability: float = Field(..., ge=0.0, le=1.0, description="Probability that image is fake/AI-generated")
    real_probability: float = Field(..., ge=0.0, le=1.0, description="Probability that image is real/authentic")
    explanation: str = Field(..., description="Human-readable explanation of the result")

class VisionDetectionResponse(BaseModel):
    
    model_config = {"protected_namespaces": ()}
    
    detection_result: VisionDetectionScore
    model_type: str = Field(..., description="Model used for detection")
    device: str = Field(..., description="Device used for inference")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_status: str = Field(..., description="Status of the model")
    image_processed: bool = Field(..., description="Whether image was successfully processed")
    face_detected: bool = Field(False, description="Whether a face was detected and cropped")
    analysis_id: str = Field(..., description="Unique analysis identifier")
    timestamp: datetime = Field(..., description="Analysis timestamp")

class BatchVisionDetectionResponse(BaseModel):
    
    model_config = {"protected_namespaces": ()}
    
    results: List[VisionDetectionResponse]
    total_images: int = Field(..., description="Total number of images processed")
    successful_detections: int = Field(..., description="Number of successful detections")
    failed_detections: int = Field(..., description="Number of failed detections")
    average_processing_time: float = Field(..., description="Average processing time per image")
    total_processing_time: float = Field(..., description="Total processing time")
    batch_id: str = Field(..., description="Unique batch identifier")
    timestamp: datetime = Field(..., description="Batch processing timestamp")

class HealthResponse(BaseModel):
    
    model_config = {"protected_namespaces": ()}
    
    status: str
    model_status: str
    available_models: List[str]
    device: str
    uptime: str
    version: str
    timestamp: datetime

# Global variables for tracking
app_start_time = datetime.now()
detection_count = 0

@app.on_event("startup")
async def startup_event():
    
    logger.info("Starting Vision Fake Profile Detection API...")
    try:
        detector = get_detector("resnet50")
        if detector.is_loaded:
            logger.info("✅ Vision models initialized successfully")
        else:
            logger.warning("⚠️ Vision models failed to load - API will run in mock mode")
    except Exception as e:
        logger.error(f"❌ Failed to initialize vision models: {e}")

@app.get("/", response_model=Dict[str, str])
async def root():
    
    return {
        "message": "Vision Fake Profile Detection API",
        "description": "AI-powered detection of fake/AI-generated profile pictures",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    
    uptime = datetime.now() - app_start_time
    
    # Check model status
    try:
        detector = get_detector("resnet50")
        model_status = "loaded" if detector.is_loaded else "not_loaded"
        device = detector.device if detector.is_loaded else "unknown"
    except Exception:
        model_status = "error"
        device = "unknown"
    
    return HealthResponse(
        status="healthy",
        model_status=model_status,
        available_models=["resnet50", "efficientnet"],
        device=device,
        uptime=str(uptime),
        version="1.0.0",
        timestamp=datetime.now()
    )

@app.post("/detect/upload", response_model=VisionDetectionResponse)
async def detect_fake_from_upload(
    file: UploadFile = File(...),
    model_type: str = Form("resnet50"),
    enhance_image: bool = Form(True),
    detect_face: bool = Form(True)
):
    
    global detection_count
    detection_count += 1
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Preprocess image
        processed_image = load_and_preprocess_image(
            image_data, 
            enhance=enhance_image, 
            detect_face=detect_face
        )
        
        # Run detection
        result = detect_fake_profile(processed_image, model_type=model_type)
        
        # Check if there was an error
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        # Build response
        detection_score = VisionDetectionScore(
            is_fake=result['is_fake'],
            confidence=result['confidence'],
            fake_probability=result['fake_probability'],
            real_probability=result['real_probability'],
            explanation=result['explanation']
        )
        
        return VisionDetectionResponse(
            detection_result=detection_score,
            model_type=result.get('model_type', model_type),
            device=result.get('device', 'unknown'),
            processing_time=result.get('processing_time', 0.0),
            model_status=result.get('model_status', 'unknown'),
            image_processed=True,
            face_detected=detect_face,  # Would be set based on actual face detection
            analysis_id=str(uuid.uuid4()),
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Detection failed for uploaded file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect", response_model=VisionDetectionResponse)
async def detect_fake_profile_endpoint(request: VisionDetectionRequest):
    
    global detection_count
    detection_count += 1
    
    try:
        # Determine image input
        if request.image_url:
            image_input = request.image_url
        elif request.image_base64:
            image_input = request.image_base64
        else:
            raise HTTPException(status_code=400, detail="Either image_url or image_base64 must be provided")
        
        # Preprocess image
        processed_image = load_and_preprocess_image(
            image_input,
            enhance=request.enhance_image,
            detect_face=request.detect_face
        )
        
        # Run detection
        result = detect_fake_profile(processed_image, model_type=request.model_type)
        
        # Check if there was an error
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        # Build response
        detection_score = VisionDetectionScore(
            is_fake=result['is_fake'],
            confidence=result['confidence'],
            fake_probability=result['fake_probability'],
            real_probability=result['real_probability'],
            explanation=result['explanation']
        )
        
        return VisionDetectionResponse(
            detection_result=detection_score,
            model_type=result.get('model_type', request.model_type),
            device=result.get('device', 'unknown'),
            processing_time=result.get('processing_time', 0.0),
            model_status=result.get('model_status', 'unknown'),
            image_processed=True,
            face_detected=request.detect_face,  # Would be set based on actual face detection
            analysis_id=str(uuid.uuid4()),
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/batch", response_model=BatchVisionDetectionResponse)
async def detect_fake_profiles_batch(request: BatchVisionDetectionRequest):
    
    global detection_count
    
    if len(request.images) == 0:
        raise HTTPException(status_code=400, detail="No images provided")
    
    if len(request.images) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 images per batch")
    
    batch_id = str(uuid.uuid4())
    start_time = datetime.now()
    results = []
    
    try:
        # Process images with concurrency control
        semaphore = asyncio.Semaphore(request.max_concurrent)
        
        async def process_single_image(image_data: Dict[str, Any], index: int):
            async with semaphore:
                try:
                    detection_count += 1
                    
                    # Determine image input
                    if 'url' in image_data:
                        image_input = image_data['url']
                    elif 'base64' in image_data:
                        image_input = image_data['base64']
                    else:
                        raise ValueError("Image must have 'url' or 'base64' field")
                    
                    # Preprocess image
                    processed_image = load_and_preprocess_image(
                        image_input,
                        enhance=request.enhance_image,
                        detect_face=request.detect_face
                    )
                    
                    # Run detection
                    result = detect_fake_profile(processed_image, model_type=request.model_type)
                    
                    # Check if there was an error
                    if 'error' in result:
                        raise Exception(result['error'])
                    
                    # Build response
                    detection_score = VisionDetectionScore(
                        is_fake=result['is_fake'],
                        confidence=result['confidence'],
                        fake_probability=result['fake_probability'],
                        real_probability=result['real_probability'],
                        explanation=result['explanation']
                    )
                    
                    return VisionDetectionResponse(
                        detection_result=detection_score,
                        model_type=result.get('model_type', request.model_type),
                        device=result.get('device', 'unknown'),
                        processing_time=result.get('processing_time', 0.0),
                        model_status=result.get('model_status', 'unknown'),
                        image_processed=True,
                        face_detected=request.detect_face,
                        analysis_id=f"{batch_id}_{index}",
                        timestamp=datetime.now()
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to process image {index}: {e}")
                    # Return error response for this image
                    detection_score = VisionDetectionScore(
                        is_fake=False,
                        confidence=0.0,
                        fake_probability=0.0,
                        real_probability=1.0,
                        explanation=f"Processing failed: {str(e)}"
                    )
                    
                    return VisionDetectionResponse(
                        detection_result=detection_score,
                        model_type=request.model_type,
                        device="unknown",
                        processing_time=0.0,
                        model_status="error",
                        image_processed=False,
                        face_detected=False,
                        analysis_id=f"{batch_id}_{index}",
                        timestamp=datetime.now()
                    )
        
        # Process all images concurrently
        tasks = [process_single_image(img, i) for i, img in enumerate(request.images)]
        results = await asyncio.gather(*tasks)
        
        # Calculate batch statistics
        end_time = datetime.now()
        total_processing_time = (end_time - start_time).total_seconds()
        successful_detections = sum(1 for r in results if r.image_processed)
        failed_detections = len(results) - successful_detections
        avg_processing_time = sum(r.processing_time for r in results) / len(results) if results else 0
        
        return BatchVisionDetectionResponse(
            results=results,
            total_images=len(request.images),
            successful_detections=successful_detections,
            failed_detections=failed_detections,
            average_processing_time=avg_processing_time,
            total_processing_time=total_processing_time,
            batch_id=batch_id,
            timestamp=end_time
        )
        
    except Exception as e:
        logger.error(f"Batch detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=Dict[str, Any])
async def get_api_stats():
    
    uptime = datetime.now() - app_start_time
    
    try:
        detector = get_detector("resnet50")
        model_status = "loaded" if detector.is_loaded else "not_loaded"
    except Exception:
        model_status = "error"
    
    return {
        "total_detections": detection_count,
        "uptime": str(uptime),
        "model_status": model_status,
        "available_models": ["resnet50", "efficientnet"],
        "api_version": "1.0.0",
        "timestamp": datetime.now()
    }

@app.get("/models", response_model=Dict[str, Any])
async def list_available_models():
    
    models_info = {
        "resnet50": {
            "description": "ResNet50 pretrained on ImageNet, fine-tuned for fake detection",
            "input_size": [224, 224],
            "parameters": "~25M",
            "performance": "High accuracy, moderate speed"
        },
        "efficientnet": {
            "description": "EfficientNet-B0 pretrained on ImageNet, fine-tuned for fake detection",
            "input_size": [224, 224],
            "parameters": "~5M",
            "performance": "Good accuracy, high speed"
        }
    }
    
    return {
        "available_models": list(models_info.keys()),
        "model_details": models_info,
        "default_model": "resnet50"
    }

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    
    logger.error(f"Unhandled exception: {exc}")
    return HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Vision Fake Profile Detection API...")
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")