# FastAPI Qwen 2.5 Cybersecurity Model API

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
