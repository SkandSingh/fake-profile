"""
Profile Extraction API for Profile Purity Detector
Automatically extracts profile data from social media URLs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any
import logging
import sys
import os

# Add the ml module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml'))

from auto_profile_extractor import AutoProfileExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Profile Extraction API",
    description="Automatically extract profile data from social media URLs for Profile Purity Detector",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the profile extractor
extractor = AutoProfileExtractor()

class ProfileURLRequest(BaseModel):
    url: HttpUrl
    extract_image: Optional[bool] = True
    timeout: Optional[int] = 30

class ProfileData(BaseModel):
    platform: str
    username: str
    displayName: str
    bio: str
    followerCount: int
    followingCount: int
    postCount: int
    verified: bool
    profileImageUrl: str
    accountAge: Optional[int]
    extractionMethod: str
    extractionError: Optional[str] = None
    manualInputRequired: Optional[bool] = False

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "profile-extraction-api"}

@app.post("/extract", response_model=ProfileData)
async def extract_profile_data(request: ProfileURLRequest):
    """
    Extract profile data from social media URL
    
    Args:
        request: ProfileURLRequest containing the URL and options
        
    Returns:
        ProfileData: Extracted profile information
        
    Raises:
        HTTPException: If extraction fails or URL is invalid
    """
    try:
        url_str = str(request.url)
        logger.info(f"Extracting profile data from: {url_str}")
        
        # Extract profile data
        profile_data = extractor.extract_profile_data(url_str)
        
        # Download profile image if requested and URL is available
        if request.extract_image and profile_data.get('profileImageUrl'):
            try:
                image_data = extractor._download_profile_image(profile_data['profileImageUrl'])
                if image_data:
                    profile_data['profileImageData'] = image_data
            except Exception as e:
                logger.warning(f"Could not download profile image: {e}")
        
        # Validate required fields
        if profile_data.get('manualInputRequired'):
            logger.warning(f"Manual input required for URL: {url_str}")
            profile_data['extractionError'] = "Automatic extraction failed - manual input recommended"
        
        return ProfileData(**profile_data)
        
    except Exception as e:
        logger.error(f"Error extracting profile data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract profile data: {str(e)}"
        )

@app.get("/supported-platforms")
async def get_supported_platforms():
    """Get list of supported social media platforms"""
    return {
        "supported_platforms": [
            {
                "name": "Instagram",
                "domains": ["instagram.com", "www.instagram.com"],
                "features": ["follower_count", "following_count", "post_count", "bio", "profile_image", "verification"]
            },
            {
                "name": "Twitter/X",
                "domains": ["twitter.com", "x.com", "www.twitter.com", "www.x.com"],
                "features": ["follower_count", "following_count", "post_count", "bio", "profile_image", "verification", "account_age"]
            },
            {
                "name": "Facebook",
                "domains": ["facebook.com", "www.facebook.com"],
                "features": ["bio", "profile_image"],
                "limitations": ["follower_count_unavailable", "privacy_restrictions"]
            }
        ],
        "extraction_methods": [
            "public_api_endpoints",
            "html_scraping", 
            "metadata_extraction",
            "guest_token_api"
        ]
    }

@app.post("/validate-url")
async def validate_profile_url(request: ProfileURLRequest):
    """
    Validate if a URL is a supported social media profile
    
    Args:
        request: ProfileURLRequest containing the URL to validate
        
    Returns:
        Validation result with platform detection
    """
    try:
        url_str = str(request.url)
        platform = extractor._detect_platform(url_str)
        username = extractor._extract_username_from_url(url_str)
        
        is_supported = platform in ['instagram', 'twitter', 'facebook']
        
        return {
            "url": url_str,
            "is_valid": is_supported,
            "platform": platform,
            "username": username,
            "supported": is_supported,
            "extraction_available": is_supported
        }
        
    except Exception as e:
        logger.error(f"Error validating URL: {e}")
        return {
            "url": str(request.url),
            "is_valid": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)