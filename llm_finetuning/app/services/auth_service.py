import os
from fastapi import HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define API key header scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_admin_token(api_key: str = Security(api_key_header)):
    """
    Verify the admin token for protected endpoints
    """
    # In production, use a more secure approach like environment variables or a secure vault
    admin_token = os.getenv("ADMIN_API_KEY", "admin_secret_key")
    
    if not api_key:
        logger.warning("API key not provided")
        raise HTTPException(
            status_code=401,
            detail="Missing API key",
        )
    
    if api_key != admin_token:
        logger.warning("Invalid API key provided")
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )
    
    return api_key
