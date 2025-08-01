"""
This module defines the model configurations used by the various agents
in the brand analysis system.
"""

from dotenv import load_dotenv
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.secret_manager import get_secret

load_dotenv()

class Modelconfig:
    """Model configuration class for ADK agents"""
    
    # Default models - adjust these based on your available models
    flash_lite_model = "gemini-2.0-flash-lite-001"  # For fast, lightweight tasks
    pro_model = "gemini-2.5-pro"          # For complex analysis tasks
    flash_model ="gemini-2.5-flash"
    flash_image = "gemini-2.0-flash-preview-image-generation"
    imagen4_fast ="imagen-4.0-fast-generate-preview-06-06"
    imagen4_ultra ="imagen-4.0-ultra-generate-preview-06-06"
    imagen3 ="imagen-3.0-generate-002"
    

    # Model settings
    temperature = 0.1  # Low temperature for consistent analysis
    max_tokens = 8192  # Maximum output tokens

class SecretConfig:
    """Configuration class that handles secrets from Secret Manager with fallbacks"""
    
    @staticmethod
    def get_qloo_api_key() -> str:
        """Get Qloo API key from Secret Manager or environment variable"""
        return get_secret("qloo-api-key", "QLOO_API_KEY")
    
    @staticmethod
    def get_supabase_url() -> str:
        """Get Supabase URL from Secret Manager or environment variable"""
        return get_secret("REACT_APP_SUPABASE_URL", "REACT_APP_SUPABASE_URL")
    
    @staticmethod
    def get_supabase_secret_key() -> str:
        """Get Supabase secret key from Secret Manager or environment variable"""
        return get_secret("SUPABASE_SECRET_KEY", "SUPABASE_SECRET_KEY")
    
    @staticmethod
    def get_google_cloud_project() -> str:
        """Get Google Cloud Project ID from environment variable"""
        # This can stay as env var since it's not sensitive
        return os.getenv("GOOGLE_CLOUD_PROJECT", "energyagentai")
    
    @staticmethod
    def get_google_cloud_location() -> str:
        """Get Google Cloud Location from environment variable"""
        # This can stay as env var since it's not sensitive
        return os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    
class Settings:
    """Configuration settings for the video generation system."""
    
    # Google Cloud Configuration
    GOOGLE_CLOUD_PROJECT: str = os.getenv("GOOGLE_CLOUD_PROJECT", "energyagentai")
    GOOGLE_CLOUD_LOCATION: str = "us-central1"
    GOOGLE_GENAI_USE_VERTEXAI: bool = True
    GCS_BUCKET_NAME = "resonance_political_ai"
    
    

    @classmethod
    def get_gcs_bucket_url(cls, path: str = "") -> str:
        """Generate GCS bucket URL for a given path."""
        base_url = f"gs://{cls.GCS_BUCKET_NAME}"
        return f"{base_url}/{path}" if path else base_url
    
    @classmethod
    def get_public_gcs_url(cls, path: str) -> str:
        """Generate public GCS URL for a given path."""
        return f"https://storage.googleapis.com/{cls.GCS_BUCKET_NAME}/{path}"

