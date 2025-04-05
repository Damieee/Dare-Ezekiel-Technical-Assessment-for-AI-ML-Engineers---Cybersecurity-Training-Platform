from typing import List, Dict, Any, Optional
import os

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Body
from pydantic import parse_obj_as
import json

from ..schemas.request_schemas import TrainingRequest, TrainingResponse, DataSourceType
from ..services.training_service import training_service
from ..services.model_service import model_service
from ..config import settings

router = APIRouter(
    prefix="/api/training",
    tags=["training"],
    responses={404: {"description": "Not found"}},
)


@router.post("/start", response_model=TrainingResponse)
async def start_training(
    background_tasks: BackgroundTasks,
    request: TrainingRequest
):
    """Start a model fine-tuning job
    
    Args:
        background_tasks: FastAPI background tasks
        request: Training configuration
        
    Returns:
        Training job status
    """
    try:
        # Get base model ID
        base_model_id = request.base_model_id or settings.DEFAULT_MODEL_ID
        
        # Create a job ID for tracking
        job_id = f"training_job_{base_model_id.replace('/', '_')}"
        
        # Define an async background task for training
        async def train_in_background():
            try:
                # Prepare model and tokenizer
                qwen_model, transformers_model = training_service.prepare_model_for_training(
                    model_id=base_model_id,
                    use_lora=request.use_lora,
                    lora_config=request.lora_config
                )
                
                # Run training
                result = training_service.train_model(
                    dataset=request.data_source,
                    tokenizer=qwen_model.tokenizer,
                    model=transformers_model,
                    training_args=request.training_args,
                    output_dir=request.output_dir or settings.FINE_TUNED_MODEL_PATH
                )
                
                # Push to hub if requested
                if request.push_to_hub and request.hub_model_id:
                    training_service.push_to_hub(
                        local_model_path=result["model_path"],
                        hub_model_id=request.hub_model_id
                    )
            except Exception as e:
                print(f"Background training failed: {str(e)}")
        
        # Start training in background
        background_tasks.add_task(train_in_background)
        
        return TrainingResponse(
            job_id=job_id,
            status="started",
            model_id=base_model_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")


@router.post("/upload", response_model=Dict[str, str])
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_name: str = Form(...)
):
    """Upload a dataset file for training
    
    Args:
        file: Dataset file (CSV, JSON, etc.)
        dataset_name: Name to assign to the dataset
        
    Returns:
        Upload status
    """
    try:
        # Create data directory if it doesn't exist
        data_dir = os.path.join("data", "uploads")
        os.makedirs(data_dir, exist_ok=True)
        
        # Save file
        file_path = os.path.join(data_dir, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return {
            "status": "success",
            "message": f"Dataset uploaded as {dataset_name}",
            "file_path": file_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/jobs", response_model=List[Dict[str, Any]])
async def list_training_jobs():
    """List all training jobs
    
    Returns:
        List of training jobs
    """
    try:
        jobs = training_service.list_training_jobs()
        return jobs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list jobs: {str(e)}")


@router.get("/jobs/{job_id}", response_model=Dict[str, Any])
async def get_job_status(job_id: str):
    """Get status of a specific training job
    
    Args:
        job_id: Training job ID
        
    Returns:
        Job status information
    """
    try:
        status = training_service.get_training_job_status(job_id)
        return status
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")
