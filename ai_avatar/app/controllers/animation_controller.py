import os
import time
import logging
import tempfile
import uuid
from typing import Optional, Dict, Any
from pathlib import Path
from fastapi import UploadFile

from app.services.whisper_service import WhisperService
from app.services.sadtalker_service import SadTalkerService
from app.models.requests import VoiceType, AudioFormat
from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

class AnimationController:
    """
    Controller for managing animation generation workflow
    """
    def __init__(
        self,
        whisper_service: WhisperService,
        sadtalker_service: SadTalkerService
    ):
        """Initialize the animation controller with required services"""
        self.whisper_service = whisper_service
        self.sadtalker_service = sadtalker_service
        logger.info("Animation controller initialized")
        
    async def generate_speech(
        self,
        text: str,
        voice: VoiceType = VoiceType.NEUTRAL,
        speed: float = 1.0,
        output_format: AudioFormat = AudioFormat.MP3
    ) -> str:
        """
        Generate speech from text
        
        Args:
            text: The text to convert to speech
            voice: Voice type to use
            speed: Speech speed
            output_format: Output audio format
            
        Returns:
            Path to the generated audio file
        """
        try:
            logger.info(f"Generating speech for text: {text[:50]}...")
            
            # In a real implementation, this would call a TTS service like ElevenLabs or AWS Polly
            # For this demonstration, we'll simulate with a placeholder audio file
            
            # Create output file
            output_dir = settings.OUTPUT_DIR
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate a unique filename
            filename = f"speech_{uuid.uuid4()}.{output_format}"
            output_path = os.path.join(output_dir, filename)
            
            # Simulate TTS processing time
            await asyncio.sleep(1)
            
            # In a real implementation, the code would look like this:
            # response = await self.tts_client.synthesize_speech(
            #     text=text,
            #     voice_id=self._map_voice_to_provider_id(voice),
            #     speed=speed,
            #     output_format=output_format
            # )
            # with open(output_path, "wb") as f:
            #     f.write(response.audio_content)
            
            # For now, we'll create a placeholder file
            with open(output_path, "w") as f:
                f.write(f"TTS PLACEHOLDER: {text}")
            
            logger.info(f"Speech generated and saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            raise Exception(f"Speech generation failed: {str(e)}")
    
    async def generate_animation_from_text(
        self,
        text: str,
        avatar_image: Optional[str] = None,
        voice: VoiceType = VoiceType.NEUTRAL,
        enhance_lipsync: bool = True,
        motion_strength: float = 1.0
    ) -> str:
        """
        Generate animation from text input
        
        Args:
            text: The text for the avatar to speak
            avatar_image: Path or URL to avatar image
            voice: Voice type to use
            enhance_lipsync: Whether to enhance lip synchronization
            motion_strength: Strength of facial motion
            
        Returns:
            Path to the generated video
        """
        try:
            logger.info(f"Generating animation for text: {text[:50]}...")
            
            # Step 1: Get or use default avatar image
            if not avatar_image:
                # Use default avatar
                avatar_path = os.path.join(settings.STATIC_DIR, settings.DEFAULT_AVATAR)
            elif avatar_image.startswith(('http://', 'https://')):
                # Download from URL to temporary file
                async with httpx.AsyncClient() as client:
                    response = await client.get(avatar_image)
                    response.raise_for_status()
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(response.content)
                        avatar_path = tmp.name
            else:
                # Use local file from static directory
                avatar_path = os.path.join(settings.STATIC_DIR, avatar_image)
                if not os.path.exists(avatar_path):
                    raise FileNotFoundError(f"Avatar image not found: {avatar_path}")
            
            # Step 2: Generate speech audio from text
            audio_path = await self.generate_speech(text, voice)
            
            # Step 3: Generate animation from image and audio
            output_path = os.path.join(
                settings.OUTPUT_DIR, 
                f"animation_{uuid.uuid4()}.mp4"
            )
            
            # Step 4: Call SadTalker service
            result = await self.sadtalker_service.generate_animation(
                image_path=avatar_path,
                audio_path=audio_path,
                output_path=output_path,
                enhance_lipsync=enhance_lipsync,
                motion_strength=motion_strength
            )
            
            # Clean up temporary files
            if avatar_image and avatar_image.startswith(('http://', 'https://')):
                os.unlink(avatar_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating animation from text: {str(e)}")
            raise Exception(f"Animation generation failed: {str(e)}")
    
    async def process_audio_to_animation(
        self,
        audio_file: UploadFile,
        avatar_image: Optional[str] = None,
        enhance_lipsync: bool = True,
        motion_strength: float = 1.0
    ) -> str:
        """
        Process audio file to animated video
        
        Args:
            audio_file: The audio file to process
            avatar_image: Path or URL to avatar image
            enhance_lipsync: Whether to enhance lip synchronization
            motion_strength: Strength of facial motion
            
        Returns:
            Path to the generated video
        """
        try:
            # Step 1: Save audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.filename.split('.')[-1]}") as tmp:
                contents = await audio_file.read()
                tmp.write(contents)
                audio_path = tmp.name
            
            # Step 2: Get or use default avatar image
            if not avatar_image:
                # Use default avatar
                avatar_path = os.path.join(settings.STATIC_DIR, settings.DEFAULT_AVATAR)
            elif avatar_image.startswith(('http://', 'https://')):
                # Download from URL to temporary file
                async with httpx.AsyncClient() as client:
                    response = await client.get(avatar_image)
                    response.raise_for_status()
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(response.content)
                        avatar_path = tmp.name
            else:
                # Use local file from static directory
                avatar_path = os.path.join(settings.STATIC_DIR, avatar_image)
                if not os.path.exists(avatar_path):
                    raise FileNotFoundError(f"Avatar image not found: {avatar_path}")
            
            # Step 3: Generate animation from image and audio
            output_path = os.path.join(
                settings.OUTPUT_DIR, 
                f"animation_{uuid.uuid4()}.mp4"
            )
            
            # Step 4: Call SadTalker service
            result = await self.sadtalker_service.generate_animation(
                image_path=avatar_path,
                audio_path=audio_path,
                output_path=output_path,
                enhance_lipsync=enhance_lipsync,
                motion_strength=motion_strength
            )
            
            # Clean up temporary files
            os.unlink(audio_path)
            if avatar_image and avatar_image.startswith(('http://', 'https://')):
                os.unlink(avatar_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing audio to animation: {str(e)}")
            # Clean up temporary files in case of error
            if 'audio_path' in locals():
                try:
                    os.unlink(audio_path)
                except:
                    pass
            if 'avatar_path' in locals() and avatar_image and avatar_image.startswith(('http://', 'https://')):
                try:
                    os.unlink(avatar_path)
                except:
                    pass
            raise Exception(f"Audio processing failed: {str(e)}")
