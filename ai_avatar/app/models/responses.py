from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class TranscriptionResponse(BaseModel):
    text: str = Field(..., description="Transcribed text from audio")
    language: Optional[str] = Field(None, description="Detected language")
    confidence: Optional[float] = Field(None, description="Confidence score of transcription")
    
class SpeechResponse(BaseModel):
    audio_url: str = Field(..., description="URL to the generated speech audio")
    duration: float = Field(..., description="Duration of the audio in seconds")
    format: str = Field(..., description="Audio format")
    
class AnimationResponse(BaseModel):
    video_url: str = Field(..., description="URL to the generated animation video")
    duration: float = Field(..., description="Duration of the video in seconds")
    text: str = Field(..., description="Text that was spoken in the animation")
    avatar_used: str = Field(..., description="Avatar image that was used")
    
class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error details")
    error_code: Optional[str] = Field(None, description="Error code")
    location: Optional[str] = Field(None, description="Error location in code")
