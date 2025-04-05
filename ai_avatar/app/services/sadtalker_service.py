import os
import time
import logging
import httpx
import json
import base64
from typing import Optional, Dict, Any, BinaryIO
from pathlib import Path
import tempfile
import asyncio

from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

class SadTalkerService:
    """
    Service for facial animation using SadTalker API
    """
    def __init__(self, api_key: str, api_url: Optional[str] = None):
        """Initialize the SadTalker service with API key and URL"""
        self.api_key = api_key
        self.api_url = api_url or settings.SADTALKER_API_URL
        self.timeout = httpx.Timeout(settings.TIMEOUT_SECONDS)
        logger.info("SadTalker service initialized")
    
    async def generate_animation(
        self,
        image_path: str,
        audio_path: str,
        output_path: Optional[str] = None,
        enhance_lipsync: bool = True,
        motion_strength: float = 1.0,
        still_mode: bool = False
    ) -> str:
        """
        Generate facial animation from image and audio
        
        Args:
            image_path: Path to the image file
            audio_path: Path to the audio file
            output_path: Path to save the output video
            enhance_lipsync: Whether to enhance lip synchronization
            motion_strength: Strength of facial motion (0.0 to 2.0)
            still_mode: Whether to generate a still image animation
            
        Returns:
            Path to the generated video
        """
        if not self.api_key:
            raise ValueError("SadTalker API key is not set")
            
        # Verify files exist
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Create output path if not provided
        if not output_path:
            output_dir = settings.OUTPUT_DIR
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"animation_{int(time.time())}.mp4")
        
        try:
            logger.info(f"Generating animation from {image_path} and {audio_path}")
            
            # Prepare request data
            with open(image_path, "rb") as img_file:
                encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
                
            with open(audio_path, "rb") as audio_file:
                encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')
            
            payload = {
                "image": encoded_img,
                "audio": encoded_audio,
                "enhance_lipsync": enhance_lipsync,
                "motion_strength": motion_strength,
                "still_mode": still_mode
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Make API request
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info("Sending request to SadTalker API...")
                response = await client.post(
                    self.api_url,
                    json=payload,
                    headers=headers
                )
                
                if response.status_code != 200:
                    logger.error(f"API error: {response.status_code}, {response.text}")
                    raise Exception(f"SadTalker API error: {response.status_code}, {response.text}")
                
                data = response.json()
                
                if "error" in data:
                    logger.error(f"API returned error: {data['error']}")
                    raise Exception(f"SadTalker API returned error: {data['error']}")
                
                # Extract and decode video data
                video_data = base64.b64decode(data["video"])
                
                # Save video to output path
                with open(output_path, "wb") as out_file:
                    out_file.write(video_data)
                
                logger.info(f"Animation saved to {output_path}")
                return output_path
                
        except httpx.RequestError as e:
            logger.error(f"Request error: {str(e)}")
            raise Exception(f"Request to SadTalker API failed: {str(e)}")
        except Exception as e:
            logger.error(f"Error generating animation: {str(e)}")
            raise Exception(f"Animation generation failed: {str(e)}")
    
    async def generate_from_urls(
        self,
        image_url: str,
        audio_url: str,
        output_path: Optional[str] = None,
        enhance_lipsync: bool = True,
        motion_strength: float = 1.0,
        still_mode: bool = False
    ) -> str:
        """
        Generate facial animation from image and audio URLs
        
        Args:
            image_url: URL to the image
            audio_url: URL to the audio
            output_path: Path to save the output video
            enhance_lipsync: Whether to enhance lip synchronization
            motion_strength: Strength of facial motion
            still_mode: Whether to generate a still image animation
            
        Returns:
            Path to the generated video
        """
        try:
            # Create temporary files for downloaded content
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_tmp, \
                 tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as audio_tmp:
                
                # Download files
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    # Download image
                    img_response = await client.get(image_url)
                    img_response.raise_for_status()
                    img_tmp.write(img_response.content)
                    
                    # Download audio
                    audio_response = await client.get(audio_url)
                    audio_response.raise_for_status()
                    audio_tmp.write(audio_response.content)
                
                img_path = img_tmp.name
                audio_path = audio_tmp.name
            
            # Generate animation
            result = await self.generate_animation(
                image_path=img_path,
                audio_path=audio_path,
                output_path=output_path,
                enhance_lipsync=enhance_lipsync,
                motion_strength=motion_strength,
                still_mode=still_mode
            )
            
            # Clean up temporary files
            os.unlink(img_path)
            os.unlink(audio_path)
            
            return result
            
        except httpx.RequestError as e:
            logger.error(f"Request error: {str(e)}")
            raise Exception(f"Failed to download image or audio: {str(e)}")
        except Exception as e:
            logger.error(f"Error generating animation from URLs: {str(e)}")
            # Clean up temporary files in case of error
            if 'img_path' in locals():
                try:
                    os.unlink(img_path)
                except:
                    pass
            if 'audio_path' in locals():
                try:
                    os.unlink(audio_path)
                except:
                    pass
            raise Exception(f"Animation generation from URLs failed: {str(e)}")
