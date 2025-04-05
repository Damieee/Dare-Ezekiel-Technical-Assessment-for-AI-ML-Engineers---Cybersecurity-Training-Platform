import json
import os
import time
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def load_training_data(file_path: str) -> List[Dict[str, str]]:
    """
    Load training data from a JSON file
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
            
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data.get("training_examples", [])
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return []

def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"

def log_api_request(endpoint: str, request_data: Dict[str, Any], response_time: float) -> None:
    """
    Log API request details for monitoring
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{timestamp}] Endpoint: {endpoint} | Response time: {format_time(response_time)}")
