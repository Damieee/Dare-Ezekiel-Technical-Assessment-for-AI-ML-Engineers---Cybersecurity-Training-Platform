from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any

from app.models.request_models import QueryRequest, FineTuneRequest
from app.models.response_models import QueryResponse, ModelInfoResponse, TrainStatusResponse
from app.services.model_service import ModelService, get_model_service
from app.services.auth_service import verify_admin_token
import time

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_model(request: QueryRequest, 
                      model_service: ModelService = Depends(get_model_service)):
    """
    Query the fine-tuned cybersecurity LLM with a question or prompt
    """
    try:
        start_time = time.time()
        response = await model_service.generate_response(
            request.query,
            temperature=request.temperature,
            max_length=request.max_length
        )
        processing_time = time.time() - start_time
        
        return QueryResponse(
            response=response,
            processing_time=processing_time,
            model_id=model_service.model_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@router.get("/model_info", response_model=ModelInfoResponse)
async def get_model_info(model_service: ModelService = Depends(get_model_service)):
    """
    Get information about the currently deployed model
    """
    try:
        return ModelInfoResponse(
            model_id=model_service.model_id,
            base_model="Qwen/Qwen2.5-0.5B",
            fine_tuned=True,
            capabilities=[
                "Vulnerability assessment",
                "Security best practices",
                "Threat intelligence",
                "Security tool recommendations",
                "Code security analysis"
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}")

@router.post("/train", response_model=TrainStatusResponse)
async def fine_tune_model(
    request: FineTuneRequest,
    background_tasks: BackgroundTasks,
    admin_token: str = Depends(verify_admin_token),
    model_service: ModelService = Depends(get_model_service)
):
    """
    Start fine-tuning the model with new cybersecurity data (admin only)
    """
    try:
        job_id = f"ft-{int(time.time())}"
        
        # Schedule the training job in the background
        background_tasks.add_task(
            model_service.fine_tune_model,
            request.training_data,
            epochs=request.epochs,
            learning_rate=request.learning_rate,
            job_id=job_id
        )
        
        return TrainStatusResponse(
            job_id=job_id,
            status="scheduled",
            message="Fine-tuning job has been scheduled",
            estimated_time="30-60 minutes"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scheduling fine-tuning job: {str(e)}")
