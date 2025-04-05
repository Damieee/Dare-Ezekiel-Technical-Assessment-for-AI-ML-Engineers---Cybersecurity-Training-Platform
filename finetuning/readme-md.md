# Qwen Cybersecurity Model API

A production-ready FastAPI application for fine-tuning and inferencing with Qwen 2.5 language models on cybersecurity datasets.

## Features

- **Model Management**: Load and use different Qwen models for inference
- **Fine-tuning**: Train custom models on cybersecurity datasets
- **Efficient Training**: Support for LoRA parameter-efficient fine-tuning
- **API Integration**: RESTful API for model training and inference
- **Production-Ready**: Error handling, logging, and containerization

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/qwen-cybersecurity-api.git
cd qwen-cybersecurity-api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the API:
```bash
uvicorn app.main:app --reload
```

### Using Docker

Build and run with Docker:
```bash
docker build -t qwen-cybersecurity-api .
docker run -p 8000:8000 qwen-cybersecurity-api
```

## API Documentation

Once running, access the API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Inference

- `POST /api/inference/generate`: Generate text from a prompt
- `GET /api/inference/models`: List available models
- `GET /api/inference/model/{model_id}/config`: Get model configuration
- `DELETE /api/inference/model/{model_id}`: Unload a model from memory

### Training

- `POST /api/training/start`: Start a fine-tuning job
- `POST /api/training/upload`: Upload a dataset for training
- `GET /api/training/jobs`: List all training jobs
- `GET /api/training/jobs/{job_id}`: Get status of a specific job

## Example Usage

### Inference Request

```python
import requests
import json

url = "http://localhost:8000/api/inference/generate"
payload = {
    "prompt": "What are the common types of phishing attacks?",
    "model_id": "fine-tuned",
    "max_new_tokens": 200,
    "temperature": 0.7
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)
print(json.dumps(response.json(), indent=2))
```

### Training Request

```python
import requests
import json

url = "http://localhost:8000/api/training/start"
payload = {
    "base_model_id": "Qwen/Qwen1.5-0.5B",
    "data_source_type": "huggingface_dataset",
    "data_source": "Anurag-Saharan/cybersecurity-threat-intel",
    "use_lora": true,
    "training_args": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4
    }
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)
print(json.dumps(response.json(), indent=2))
```

## Configuration

Edit `app/config.py` to customize model parameters, training settings, and API behavior.

## Project Structure

```
qwen-cybersecurity-api/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application entry point
│   ├── config.py                # Configuration settings
│   ├── models/
│   │   ├── __init__.py
│   │   └── ml_models.py         # ML model classes
│   ├── services/
│   │   ├── __init__.py
│   │   ├── model_service.py     # Model loading/inference service
│   │   └── training_service.py  # Model training service
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── request_schemas.py   # Pydantic schemas for API requests/responses
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── inference.py         # Inference endpoints
│   │   └── training.py          # Training endpoints
│   └── utils/
│       ├── __init__.py
│       └── preprocessing.py     # Data preprocessing utilities
├── requirements.txt             # Project dependencies
├── Dockerfile                   # For containerization
└── README.md                    # Project documentation
```

## License

MIT
