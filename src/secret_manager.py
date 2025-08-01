import os
import logging
from google.cloud import secretmanager
from typing import Optional, Dict
import functools

logger = logging.getLogger(__name__)
project_id = 'energyagentai'
class SecretManager:
    """Utility class for accessing Google Cloud Secret Manager"""
    
    def __init__(self, project_id: Optional[str] = None):
        """
        Initialize SecretManager client
        
        Args:
            project_id: Google Cloud Project ID. If None, uses GOOGLE_CLOUD_PROJECT env var
        """
        self.project_id = "energyagentai"
        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable or project_id parameter required")
        
        try:
            self.client = secretmanager.SecretManagerServiceClient()
        except Exception as e:
            logger.error(f"Failed to initialize Secret Manager client: {e}")
            raise
    
    @functools.lru_cache(maxsize=32)
    def get_secret(self, secret_name: str, version: str = "latest") -> str:
        """
        Retrieve a secret from Google Cloud Secret Manager with caching
        
        Args:
            secret_name: Name of the secret
            version: Version of the secret (default: "latest")
            
        Returns:
            Secret value as string
            
        Raises:
            Exception: If secret cannot be retrieved
        """
        try:
            name = f"projects/{self.project_id}/secrets/{secret_name}/versions/{version}"
            response = self.client.access_secret_version(request={"name": name})
            secret_value = response.payload.data.decode("UTF-8")
            logger.info(f"✅ Successfully retrieved secret: {secret_name}")
            return secret_value
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve secret {secret_name}: {e}")
            raise
    
    def get_secret_with_fallback(self, secret_name: str, env_var_name: str) -> str:
        """
        Try to get secret from Secret Manager, fallback to environment variable
        
        Args:
            secret_name: Name of the secret in Secret Manager
            env_var_name: Environment variable name as fallback
            
        Returns:
            Secret value from Secret Manager or environment variable
        """
        try:
            return self.get_secret(secret_name)
        except Exception as e:
            logger.warning(f"Failed to get {secret_name} from Secret Manager, falling back to env var: {e}")
            value = os.getenv(env_var_name)
            if not value:
                raise ValueError(f"Neither secret '{secret_name}' nor env var '{env_var_name}' found")
            return value


# Global instance - will be initialized when first imported
_secret_manager: Optional[SecretManager] = None

def get_secret_manager() -> SecretManager:
    """Get or create global SecretManager instance"""
    global _secret_manager
    if _secret_manager is None:
        _secret_manager = SecretManager()
    return _secret_manager

def get_secret(secret_name: str, fallback_env_var: str = None) -> str:
    """
    Convenience function to get a secret
    
    Args:
        secret_name: Name of the secret in Secret Manager
        fallback_env_var: Optional environment variable to use as fallback
        
    Returns:
        Secret value
    """
    sm = get_secret_manager()
    if fallback_env_var:
        return sm.get_secret_with_fallback(secret_name, fallback_env_var)
    return sm.get_secret(secret_name)

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