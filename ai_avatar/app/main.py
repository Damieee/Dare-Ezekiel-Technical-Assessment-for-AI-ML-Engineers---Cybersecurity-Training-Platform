from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import os
import logging
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.models.requests import TextToSpeechRequest, GenerateAnimationRequest
from app.services.whisper_service import WhisperService
from app.services.sadtalker_service import SadTalkerService
from app.controllers.animation_controller import AnimationController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Avatar Speech & Animation API",
    description="API for speech-to-text conversion and facial animation generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directory
output_dir = Path("./outputs")
output_dir.mkdir(exist_ok=True)

# Mount static directory
static_dir = Path("./static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize services
whisper_service = WhisperService(api_key=settings.OPENAI_API_KEY)
sadtalker_service = SadTalkerService(api_key=settings.SADTALKER_API_KEY)

# Initialize controllers
animation_controller = AnimationController(whisper_service, sadtalker_service)

@app.get("/")
async def root():
    return {
        "message": "AI Avatar Speech & Animation API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

@app.post("/api/speech-to-text")
async def speech_to_text(
    audio_file: UploadFile = File(...),
    language: Optional[str] = Form(None)
):
    """Convert speech audio to text using OpenAI Whisper"""
    try:
        text = await whisper_service.transcribe_audio(
            audio_file=audio_file,
            language=language
        )
        return {"text": text}
    except Exception as e:
        logger.error(f"Speech-to-text error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Speech-to-text failed: {str(e)}")

@app.post("/api/text-to-speech")
async def text_to_speech(request: TextToSpeechRequest):
    """Convert text to speech audio"""
    try:
        audio_path = await animation_controller.generate_speech(
            text=request.text,
            voice=request.voice,
            output_format=request.output_format
        )
        return FileResponse(
            path=audio_path,
            media_type="audio/mpeg",
            filename=os.path.basename(audio_path)
        )
    except Exception as e:
        logger.error(f"Text-to-speech error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {str(e)}")

@app.post("/api/generate-animation")
async def generate_animation(request: GenerateAnimationRequest):
    """Generate a facial animation from text"""
    try:
        video_path = await animation_controller.generate_animation_from_text(
            text=request.text,
            avatar_image=request.avatar_image,
            voice=request.voice
        )
        return FileResponse(
            path=video_path,
            media_type="video/mp4",
            filename=os.path.basename(video_path)
        )
    except Exception as e:
        logger.error(f"Animation generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Animation generation failed: {str(e)}")

@app.post("/api/process-audio")
async def process_audio(
    audio_file: UploadFile = File(...),
    avatar_image: Optional[str] = Form(None)
):
    """Process audio file to animated video"""
    try:
        video_path = await animation_controller.process_audio_to_animation(
            audio_file=audio_file,
            avatar_image=avatar_image
        )
        return FileResponse(
            path=video_path,
            media_type="video/mp4",
            filename=os.path.basename(video_path)
        )
    except Exception as e:
        logger.error(f"Audio processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8002, reload=True)
