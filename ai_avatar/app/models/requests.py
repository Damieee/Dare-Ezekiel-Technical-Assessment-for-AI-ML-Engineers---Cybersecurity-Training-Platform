from pydantic import BaseModel, Field, HttpUrl, validator
from typing import Optional, List
from enum import Enum
import os

class AudioFormat(str, Enum):
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"

class VoiceType(str, Enum):
    MALE_1 = "male_1"
    MALE_2 = "male_2"
    FEMALE_1 = "female_1"
    FEMALE_2 = "female_2"
    NEUTRAL = "neutral"

class TextToSpeechRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, 
                     description="The text to convert to speech")
    voice: VoiceType = Field(default=VoiceType.NEUTRAL, 
                           description="The voice type to use for synthesis")
    output_format: AudioFormat = Field(default=AudioFormat.MP3, 
                                    description="The output audio format")
    speed: Optional[float] = Field(default=1.0, ge=0.5, le=2.0, 
                                 description="Speech speed multiplier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, this is a test of the AI avatar speech synthesis system.",
                "voice": "female_1",
                "output_format": "mp3",
                "speed": 1.0
            }
        }

class GenerateAnimationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, 
                     description="The text the avatar will speak")
    avatar_image: Optional[str] = Field(default=None, 
                                     description="URL or filename of the avatar image")
    voice: VoiceType = Field(default=VoiceType.NEUTRAL, 
                           description="The voice to use for the speech")
    
    @validator('avatar_image')
    def validate_avatar_image(cls, v):
        if v is None:
            return v
        
        # Check if it's a URL
        if v.startswith(('http://', 'https://')):
            return v
            
        # Check if it's a filename that exists in the static directory
        if os.path.isfile(os.path.join('static', v)):
            return v
            
        raise ValueError("Avatar image must be a URL or a valid filename in the static directory")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, this is an example of facial animation with an AI avatar.",
                "avatar_image": "default_avatar.jpg",
                "voice": "female_1"
            }
        }

class AnimationStyle(BaseModel):
    motion_strength: Optional[float] = Field(default=1.0, ge=0.0, le=2.0, 
                                          description="Strength of facial motion")
    still_mode: Optional[bool] = Field(default=False, 
                                    description="Whether to generate a still image animation")
    enhance_lipsync: Optional[bool] = Field(default=True, 
                                         description="Whether to enhance lip synchronization")
    
    class Config:
        json_schema_extra = {
            "example": {
                "motion_strength": 1.0,
                "still_mode": False,
                "enhance_lipsync": True
            }
        }
