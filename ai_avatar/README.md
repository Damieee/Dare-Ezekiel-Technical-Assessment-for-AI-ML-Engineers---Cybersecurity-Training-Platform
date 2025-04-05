# AI Avatar Speech & Face Animation

This application integrates speech-to-text using OpenAI Whisper and facial animation with SadTalker to create an AI avatar that can speak and animate from text input.

## Features

- Speech-to-text conversion using OpenAI Whisper API
- Facial animation pipeline using SadTalker API
- Clean OOP architecture with proper error handling
- Flexible configuration for different avatar settings
- API endpoints for generating animations

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export OPENAI_API_KEY=your_openai_api_key_here
export SADTALKER_API_KEY=your_sadtalker_api_key_here
```

3. Run the application:
```bash
uvicorn app.main:app --reload
```

## Usage

This application provides both a REST API and a command-line interface:

### API Endpoints

- `POST /api/speech-to-text` - Convert audio to text using Whisper
- `POST /api/text-to-speech` - Convert text to speech audio
- `POST /api/generate-animation` - Generate a facial animation from text
- `POST /api/process-audio` - Process audio file to animated video

### Command-line Interface

```bash
python -m app.cli generate-animation --text "This is a test animation" --output video.mp4
```

## Project Structure

- `app/` - Main application code
  - `main.py` - FastAPI application entry point
  - `models/` - Data models and schemas
  - `services/` - Service classes for external APIs
  - `controllers/` - API controllers
  - `core/` - Core application functionality
  - `cli.py` - Command-line interface
- `config/` - Configuration files
- `static/` - Static assets (default avatar images)
