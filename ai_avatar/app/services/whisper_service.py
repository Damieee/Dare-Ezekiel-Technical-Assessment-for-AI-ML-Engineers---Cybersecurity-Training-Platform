import os
import tempfile
import logging
from typing import Optional, Dict, Any, BinaryIO
import openai
from fastapi import UploadFile

from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

class WhisperService:
    """
    Service for speech-to-text conversion using OpenAI's Whisper API
    """
    def __init__(self, api_key: str):
        """Initialize the Whisper service with API key"""
        self.api_key = api_key
        openai.api_key = api_key
        logger.info("Whisper service initialized")
        
    async def transcribe_audio(
        self, 
        audio_file: UploadFile, 
        language: Optional[str] = None,
        prompt: Optional[str] = None
    ) -> str:
        """
        Transcribe audio to text using OpenAI's Whisper API
        
        Args:
            audio_file: The audio file to transcribe
            language: Optional language code (e.g., 'en', 'fr')
            prompt: Optional prompt to guide the transcription
            
        Returns:
            Transcribed text
        """
        if not self.api_key:
            raise ValueError("OpenAI API key is not set")
            
        try:
            # Create temporary file to store the uploaded audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.filename.split('.')[-1]}") as tmp:
                # Write uploaded file content to temporary file
                contents = await audio_file.read()
                tmp.write(contents)
                tmp_path = tmp.name
            
            logger.info(f"Transcribing audio file: {audio_file.filename}")
            
            # Prepare optional parameters
            params = {}
            if language:
                params["language"] = language
            if prompt:
                params["prompt"] = prompt
            
            # Call Whisper API
            with open(tmp_path, "rb") as audio:
                response = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio,
                    **params
                )
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            # Extract and return transcribed text
            transcription = response.text
            logger.info(f"Audio transcribed successfully ({len(transcription)} chars)")
            
            return transcription
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            # Clean up temporary file in case of error
            if 'tmp_path' in locals():
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            raise Exception(f"Transcription failed: {str(e)}")
    
    async def transcribe_from_path(
        self, 
        audio_path: str, 
        language: Optional[str] = None,
        prompt: Optional[str] = None
    ) -> str:
        """
        Transcribe audio from a file path
        
        Args:
            audio_path: Path to the audio file
            language: Optional language code
            prompt: Optional prompt to guide the transcription
            
        Returns:
            Transcribed text
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        try:
            logger.info(f"Transcribing audio from path: {audio_path}")
            
            # Prepare optional parameters
            params = {}
            if language:
                params["language"] = language
            if prompt:
                params["prompt"] = prompt
            
            # Call Whisper API
            with open(audio_path, "rb") as audio:
                response = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio,
                    **params
                )
            
            # Extract and return transcribed text
            transcription = response.text
            logger.info(f"Audio transcribed successfully ({len(transcription)} chars)")
            
            return transcription
            
        except Exception as e:
            logger.error(f"Error transcribing audio from path: {str(e)}")
            raise Exception(f"Transcription failed: {str(e)}")
