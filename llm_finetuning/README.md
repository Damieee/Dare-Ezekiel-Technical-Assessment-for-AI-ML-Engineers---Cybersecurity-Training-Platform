# Cybersecurity LLM Fine-tuning & Deployment

This application fine-tunes Qwen 2.5 on cybersecurity data and provides a REST API for querying the model.

## Why Qwen 2.5?

Qwen 2.5 was chosen for several reasons:
- Efficient performance on smaller datasets compared to Llama 3.3
- Good balance of size and capability (smaller than Llama 3.3, better optimized than DeepSeek V3)
- Superior instruction-following capabilities for security-specific tasks
- Better documentation and community support for fine-tuning
- More parameter-efficient fine-tuning options

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Set your Hugging Face API token:
```
export HUGGINGFACE_TOKEN=your_token_here
```

3. Run the application:
```
uvicorn app.main:app --reload
```

## API Endpoints

- `POST /api/query`: Query the fine-tuned model with your security questions
- `GET /api/model_info`: Get information about the deployed model
- `POST /api/train`: Trigger fine-tuning (admin only)

## Project Structure

- `app/`: Main application code
  - `main.py`: FastAPI application entry point
  - `models/`: Data models and schemas
  - `services/`: Business logic
  - `api/`: API endpoints
- `config/`: Configuration files
- `data/`: Training data samples
- `utils/`: Helper utilities
