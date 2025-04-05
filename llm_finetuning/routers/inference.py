from typing import List, Dict, Any, Optional
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from app.schemas.request_schemas import InferenceRequest, InferenceResponse
from app.services.model_service import model_service

router = APIRouter(
    prefix="/api/inference",
    tags=["inference"],
    responses={404: {"description": "Not found"}},
)


@router.post("/generate", response_model=InferenceResponse)
async def generate_text(request: InferenceRequest):
    """Generate text from a prompt using the specified model
    
    Args:
        request: InferenceRequest with prompt and generation parameters
        
    Returns:
        Generated text response
    """
    try:
        # Extract generation config
        generation_config = {}
        for param in ["max_new_tokens", "temperature", "top_p", "repetition_penalty"]:
            value = getattr(request, param, None)
            if value is not None:
                generation_config[param] = value
        
        # Measure inference time
        start_time = time.time()
        
        # Run inference
        response = model_service.perform_inference(
            prompt=request.prompt,
            model_id=request.model_id,
            generation_config=generation_config
        )
        
        # Calculate time in milliseconds
        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000
        
        return InferenceResponse(
            response=response,
            model_id=request.model_id or "default",
            generation_time_ms=round(inference_time_ms, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.get("/models", response_model=List[str])
async def list_models():
    """List available models for inference
    
    Returns:
        List of model identifiers
    """
    try:
        models = model_service.list_available_models()
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.get("/model/{model_id}/config", response_model=Dict[str, Any])
async def get_model_config(model_id: str):
    """Get generation configuration for a model
    
    Args:
        model_id: Model identifier
        
    Returns:
        Model generation configuration
    """
    try:
        model = model_service.get_model(model_id)
        config = model.get_generation_config()
        return config
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model config: {str(e)}")


@router.delete("/model/{model_id}", response_model=Dict[str, bool])
async def unload_model(model_id: str):
    """Unload a model from memory
    
    Args:
        model_id: Model identifier to unload
        
    Returns:
        Success status
    """
    try:
        success = model_service.unload_model(model_id)
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")
