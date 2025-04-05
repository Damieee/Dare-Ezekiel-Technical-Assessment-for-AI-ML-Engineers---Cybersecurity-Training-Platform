import click
import asyncio
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.services.whisper_service import WhisperService
from app.services.sadtalker_service import SadTalkerService
from app.controllers.animation_controller import AnimationController
from app.models.requests import VoiceType

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize services
whisper_service = WhisperService(api_key=settings.OPENAI_API_KEY)
sadtalker_service = SadTalkerService(api_key=settings.SADTALKER_API_KEY)
animation_controller = AnimationController(whisper_service, sadtalker_service)

@click.group()
def cli():
    """AI Avatar CLI for generating animations from text or audio"""
    pass

@cli.command("transcribe")
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--language", "-l", help="Language code (e.g., 'en', 'fr')")
@click.option("--output", "-o", help="Output text file")
def transcribe(audio_file, language, output):
    """Transcribe audio to text using Whisper"""
    async def _transcribe():
        try:
            # Call Whisper service
            text = await whisper_service.transcribe_from_path(
                audio_path=audio_file,
                language=language
            )
            
            # Print or save output
            if output:
                with open(output, "w", encoding="utf-8") as f:
                    f.write(text)
                click.echo(f"Transcription saved to {output}")
            else:
                click.echo("Transcription:")
                click.echo("-" * 40)
                click.echo(text)
                click.echo("-" * 40)
                
        except Exception as e:
            click.echo(f"Error: {str(e)}", err=True)
            sys.exit(1)
    
    asyncio.run(_transcribe())

@cli.command("generate-animation")
@click.option("--text", "-t", required=True, help="Text for the avatar to speak")
@click.option("--avatar", "-a", help="Path or URL to avatar image")
@click.option("--voice", "-v", type=click.Choice(["male_1", "male_2", "female_1", "female_2", "neutral"]), 
              default="neutral", help="Voice type")
@click.option("--output", "-o", help="Output video file")
@click.option("--lipsync/--no-lipsync", default=True, help="Enable/disable lip sync enhancement")
@click.option("--motion", type=float, default=1.0, help="Motion strength (0.0-2.0)")
def generate_animation(text, avatar, voice, output, lipsync, motion):
    """Generate a facial animation from text"""
    async def _generate():
        try:
            # Map voice type
            voice_type = VoiceType(voice)
            
            # Generate animation
            output_path = await animation_controller.generate_animation_from_text(
                text=text,
                avatar_image=avatar,
                voice=voice_type,
                enhance_lipsync=lipsync,
                motion_strength=motion
            )
            
            # Copy to requested output location if specified
            if output:
                import shutil
                shutil.copy2(output_path, output)
                click.echo(f"Animation saved to {output}")
            else:
                click.echo(f"Animation saved to {output_path}")
                
        except Exception as e:
            click.echo(f"Error: {str(e)}", err=True)
            sys.exit(1)
    
    asyncio.run(_generate())

@cli.command("process-audio")
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--avatar", "-a", help="Path or URL to avatar image")
@click.option("--output", "-o", help="Output video file")
@click.option("--lipsync/--no-lipsync", default=True, help="Enable/disable lip sync enhancement")
@click.option("--motion", type=float, default=1.0, help="Motion strength (0.0-2.0)")
def process_audio(audio_file, avatar, output, lipsync, motion):
    """Process audio file to animated video"""
    async def _process():
        try:
            # Create a dummy UploadFile-like object
            class DummyUploadFile:
                def __init__(self, filename, file_path):
                    self.filename = filename
                    self.file_path = file_path
                
                async def read(self):
                    with open(self.file_path, "rb") as f:
                        return f.read()
            
            # Create dummy upload file
            upload_file = DummyUploadFile(
                filename=os.path.basename(audio_file),
                file_path=audio_file
            )
            
            # Process audio
            output_path = await animation_controller.process_audio_to_animation(
                audio_file=upload_file,
                avatar_image=avatar,
                enhance_lipsync=lipsync,
                motion_strength=motion
            )
            
            # Copy to requested output location if specified
            if output:
                import shutil
                shutil.copy2(output_path, output)
                click.echo(f"Animation saved to {output}")
            else:
                click.echo(f"Animation saved to {output_path}")
                
        except Exception as e:
            click.echo(f"Error: {str(e)}", err=True)
            sys.exit(1)
    
    asyncio.run(_process())

if __name__ == "__main__":
    cli()
