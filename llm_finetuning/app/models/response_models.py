from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class QueryResponse(BaseModel):
    response: str = Field(..., description="The model's response to the query")
    processing_time: float = Field(..., description="Time taken to process the query in seconds")
    model_id: str = Field(..., description="ID of the model used for inference")
    
class ModelInfoResponse(BaseModel):
    model_id: str = Field(..., description="ID of the deployed model")
    base_model: str = Field(..., description="Base model used for fine-tuning")
    fine_tuned: bool = Field(..., description="Whether the model is fine-tuned")
    specialization: str = Field("Cybersecurity", description="Model specialization area")
    capabilities: List[str] = Field(..., description="Model capabilities")
    
class TrainStatusResponse(BaseModel):
    job_id: str = Field(..., description="Training job ID")
    status: str = Field(..., description="Status of the training job")
    message: str = Field(..., description="Additional information about the training")
    estimated_time: Optional[str] = Field(None, description="Estimated time to completion")
