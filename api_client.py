"""
API client for facial recognition services.
Handles AWS Rekognition and Google Cloud Vision APIs (Azure excluded due to issues).
"""
import boto3
import json
import logging
from typing import Dict, List, Optional, Tuple
from google.cloud import vision
from PIL import Image
import requests
from io import BytesIO
import base64

from config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognitionClient:
    """Client for managing multiple face recognition APIs."""
    
    def __init__(self):
        self.aws_client = None
        self.google_client = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize API clients with error handling."""
        try:
            # Initialize AWS Rekognition
            self.aws_client = boto3.client(
                'rekognition',
                aws_access_key_id=config.aws_access_key,
                aws_secret_access_key=config.aws_secret_key,
                region_name=config.aws_region
            )
            logger.info("AWS Rekognition client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AWS Rekognition: {e}")
        
        try:
            # Initialize Google Cloud Vision
            self.google_client = vision.ImageAnnotatorClient()
            logger.info("Google Cloud Vision client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Google Cloud Vision: {e}")
    
    def test_connections(self, test_image_url: str = None) -> Dict[str, bool]:
        """Test API connections with a sample image."""
        if not test_image_url:
            test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/User_icon_2.svg/220px-User_icon_2.svg.png"
        
        results = {}
        
        # Test AWS Rekognition
        try:
            response = requests.get(test_image_url)
            image_bytes = response.content
            
            aws_response = self.aws_client.detect_faces(
                Image={'Bytes': image_bytes},
                Attributes=['ALL']
            )
            results['aws'] = len(aws_response.get('FaceDetails', [])) >= 0
            logger.info("AWS Rekognition test: SUCCESS")
        except Exception as e:
            results['aws'] = False
            logger.error(f"AWS Rekognition test failed: {e}")
        
        # Test Google Cloud Vision
        try:
            response = requests.get(test_image_url)
            image_bytes = response.content
            
            image = vision.Image(content=image_bytes)
            google_response = self.google_client.face_detection(image=image)
            results['google'] = len(google_response.face_annotations) >= 0
            logger.info("Google Cloud Vision test: SUCCESS")
        except Exception as e:
            results['google'] = False
            logger.error(f"Google Cloud Vision test failed: {e}")
        
        return results
    
    def analyze_face_aws(self, image_bytes: bytes) -> Dict:
        """Analyze face using AWS Rekognition."""
        try:
            response = self.aws_client.detect_faces(
                Image={'Bytes': image_bytes},
                Attributes=['ALL']
            )
            
            faces = []
            for face_detail in response.get('FaceDetails', []):
                face_data = {
                    'confidence': face_detail.get('Confidence', 0),
                    'age_range': face_detail.get('AgeRange', {}),
                    'gender': face_detail.get('Gender', {}),
                    'emotions': face_detail.get('Emotions', []),
                    'quality': face_detail.get('Quality', {}),
                    'bounding_box': face_detail.get('BoundingBox', {}),
                    'landmarks': face_detail.get('Landmarks', [])
                }
                faces.append(face_data)
            
            return {
                'service': 'aws',
                'faces': faces,
                'face_count': len(faces)
            }
        except Exception as e:
            logger.error(f"AWS face analysis failed: {e}")
            return {'service': 'aws', 'error': str(e), 'faces': [], 'face_count': 0}
    
    def analyze_face_google(self, image_bytes: bytes) -> Dict:
        """Analyze face using Google Cloud Vision."""
        try:
            image = vision.Image(content=image_bytes)
            response = self.google_client.face_detection(image=image)
            
            faces = []
            for face in response.face_annotations:
                face_data = {
                    'confidence': face.detection_confidence,
                    'bounding_box': {
                        'vertices': [(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]
                    },
                    'landmarks': [
                        {
                            'type': landmark.type_.name,
                            'position': {'x': landmark.position.x, 'y': landmark.position.y}
                        }
                        for landmark in face.landmarks
                    ],
                    'emotions': {
                        'joy': face.joy_likelihood.name,
                        'sorrow': face.sorrow_likelihood.name,
                        'anger': face.anger_likelihood.name,
                        'surprise': face.surprise_likelihood.name
                    },
                    'attributes': {
                        'headwear': face.headwear_likelihood.name,
                        'blurred': face.blurred_likelihood.name,
                        'under_exposed': face.under_exposed_likelihood.name
                    }
                }
                faces.append(face_data)
            
            return {
                'service': 'google',
                'faces': faces,
                'face_count': len(faces)
            }
        except Exception as e:
            logger.error(f"Google face analysis failed: {e}")
            return {'service': 'google', 'error': str(e), 'faces': [], 'face_count': 0}
    
    def analyze_image(self, image_path: str = None, image_url: str = None, image_bytes: bytes = None) -> Dict:
        """Analyze image using all available APIs."""
        # Get image bytes
        if image_bytes is None:
            if image_url:
                response = requests.get(image_url)
                image_bytes = response.content
            elif image_path:
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
            else:
                raise ValueError("Must provide image_path, image_url, or image_bytes")
        
        results = {}
        
        # Analyze with AWS
        if self.aws_client:
            results['aws'] = self.analyze_face_aws(image_bytes)
        
        # Analyze with Google
        if self.google_client:
            results['google'] = self.analyze_face_google(image_bytes)
        
        return results
