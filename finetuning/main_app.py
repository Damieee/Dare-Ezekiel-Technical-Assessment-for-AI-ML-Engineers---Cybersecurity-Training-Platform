import os
from typing import Dict, Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from .config import settings
from .routers import inference, training

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Include routers
app.include_router(inference.router)
app.include_router(training.router)


@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint that returns API information"""
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "description": settings.API_DESCRIPTION,
        "docs_url": "/docs",
        "status": "operational"
    }


@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy"}


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all requests"""
    # Log the request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    try:
        # Process the request
        response = await call_next(request)
        return response
    except Exception as e:
        # Log the error
        logger.error(f"Request failed: {str(e)}")
        
        # Return error response
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )


# Run the application using Uvicorn when executed directly
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("finetuning.main_app:app", host="0.0.0.0", port=port, reload=True)
