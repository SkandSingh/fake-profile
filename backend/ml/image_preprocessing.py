"""
Image Preprocessing Utilities for Vision Models
Handles various image formats, validation, and preprocessing
"""

import PIL.Image as Image
import numpy as np
import io
import base64
import requests
from typing import Union, Tuple, Optional
import logging
from pathlib import Path
import cv2

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Handles image preprocessing for vision models"""
    
    @staticmethod
    def validate_image_format(image_data: bytes) -> bool:
        """
        Validate if the image data is in a supported format
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            True if format is supported, False otherwise
        """
        try:
            with Image.open(io.BytesIO(image_data)) as img:
                # Check if format is supported
                supported_formats = ['JPEG', 'PNG', 'BMP', 'TIFF', 'WEBP']
                return img.format in supported_formats
        except Exception:
            return False
    
    @staticmethod
    def load_image_from_url(url: str, timeout: int = 10) -> Image.Image:
        """
        Load image from URL
        
        Args:
            url: Image URL
            timeout: Request timeout in seconds
            
        Returns:
            PIL Image object
        """
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Validate content type
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                raise ValueError(f"Invalid content type: {content_type}")
            
            # Load image
            image_data = response.content
            if not ImageProcessor.validate_image_format(image_data):
                raise ValueError("Unsupported image format")
            
            image = Image.open(io.BytesIO(image_data))
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image from URL {url}: {e}")
            raise
    
    @staticmethod
    def load_image_from_base64(base64_str: str) -> Image.Image:
        """
        Load image from base64 string
        
        Args:
            base64_str: Base64 encoded image
            
        Returns:
            PIL Image object
        """
        try:
            # Handle data URL format
            if base64_str.startswith('data:image'):
                base64_str = base64_str.split(',', 1)[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_str)
            
            # Validate format
            if not ImageProcessor.validate_image_format(image_data):
                raise ValueError("Unsupported image format")
            
            image = Image.open(io.BytesIO(image_data))
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image from base64: {e}")
            raise
    
    @staticmethod
    def load_image_from_file(file_path: Union[str, Path]) -> Image.Image:
        """
        Load image from file path
        
        Args:
            file_path: Path to image file
            
        Returns:
            PIL Image object
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Image file not found: {file_path}")
            
            image = Image.open(file_path)
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image from file {file_path}: {e}")
            raise
    
    @staticmethod
    def normalize_image(image: Image.Image, 
                       target_size: Tuple[int, int] = (224, 224),
                       maintain_aspect_ratio: bool = True) -> Image.Image:
        """
        Normalize image for model input
        
        Args:
            image: PIL Image object
            target_size: Target dimensions (width, height)
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Normalized PIL Image
        """
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            if maintain_aspect_ratio:
                # Resize maintaining aspect ratio, then center crop
                image.thumbnail(target_size, Image.Resampling.LANCZOS)
                
                # Create new image with target size and paste resized image in center
                new_image = Image.new('RGB', target_size, (128, 128, 128))
                
                # Calculate position to center the image
                x = (target_size[0] - image.width) // 2
                y = (target_size[1] - image.height) // 2
                
                new_image.paste(image, (x, y))
                image = new_image
            else:
                # Direct resize without maintaining aspect ratio
                image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to normalize image: {e}")
            raise
    
    @staticmethod
    def enhance_image_quality(image: Image.Image) -> Image.Image:
        """
        Apply basic image enhancement for better model performance
        
        Args:
            image: PIL Image object
            
        Returns:
            Enhanced PIL Image
        """
        try:
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            cv_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Apply slight Gaussian blur to reduce noise
            cv_image = cv2.GaussianBlur(cv_image, (3, 3), 0)
            
            # Convert back to PIL
            enhanced_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            return enhanced_image
            
        except Exception as e:
            logger.warning(f"Image enhancement failed, using original: {e}")
            return image
    
    @staticmethod
    def detect_face_region(image: Image.Image) -> Optional[Image.Image]:
        """
        Detect and crop face region if present
        
        Args:
            image: PIL Image object
            
        Returns:
            Cropped face region or None if no face detected
        """
        try:
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Load face cascade (you might need to install opencv-python-headless)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Get the largest face
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest_face
                
                # Add some padding
                padding = int(min(w, h) * 0.2)
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(cv_image.shape[1] - x, w + 2 * padding)
                h = min(cv_image.shape[0] - y, h + 2 * padding)
                
                # Crop face region
                face_region = cv_image[y:y+h, x:x+w]
                
                # Convert back to PIL
                face_image = Image.fromarray(cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB))
                return face_image
            
            return None
            
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return None
    
    @staticmethod
    def preprocess_for_detection(image_input: Union[str, bytes, Image.Image, np.ndarray],
                                enhance: bool = True,
                                detect_face: bool = True,
                                target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
        """
        Complete preprocessing pipeline for fake detection
        
        Args:
            image_input: Various image input types
            enhance: Whether to apply image enhancement
            detect_face: Whether to detect and crop face region
            target_size: Target image size
            
        Returns:
            Preprocessed PIL Image ready for model inference
        """
        try:
            # Load image from various input types
            if isinstance(image_input, str):
                if image_input.startswith(('http://', 'https://')):
                    image = ImageProcessor.load_image_from_url(image_input)
                elif image_input.startswith('data:image'):
                    image = ImageProcessor.load_image_from_base64(image_input)
                else:
                    image = ImageProcessor.load_image_from_file(image_input)
            elif isinstance(image_input, bytes):
                image = Image.open(io.BytesIO(image_input))
            elif isinstance(image_input, np.ndarray):
                image = Image.fromarray(image_input)
            elif isinstance(image_input, Image.Image):
                image = image_input.copy()
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
            # Try to detect and crop face region
            if detect_face:
                face_image = ImageProcessor.detect_face_region(image)
                if face_image is not None:
                    image = face_image
                    logger.info("Face region detected and cropped")
            
            # Apply image enhancement
            if enhance:
                image = ImageProcessor.enhance_image_quality(image)
            
            # Normalize image
            image = ImageProcessor.normalize_image(image, target_size)
            
            return image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise

# Convenience functions
def load_and_preprocess_image(image_input: Union[str, bytes, Image.Image, np.ndarray],
                             **kwargs) -> Image.Image:
    """Convenience function for image preprocessing"""
    return ImageProcessor.preprocess_for_detection(image_input, **kwargs)

def validate_image_input(image_input: Union[str, bytes, Image.Image, np.ndarray]) -> bool:
    """Validate if image input is processable"""
    try:
        ImageProcessor.preprocess_for_detection(image_input)
        return True
    except Exception:
        return False