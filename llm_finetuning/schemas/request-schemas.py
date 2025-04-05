from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """Schema for inference requests"""
    
    prompt: str = Field(..., description="Input prompt for text generation")
    model_id: Optional[str] = Field(None, description="Model ID or 'fine-tuned'")
    max_new_tokens: Optional[int] = Field(None, description="Maximum number of tokens to generate")
    temperature: Optional[float] = Field(None, description="Temperature for sampling")
    top_p: Optional[float] = Field(None, description="Top p for nucleus sampling")
    repetition_penalty: Optional[float] = Field(None, description="Penalty for repetition")

    class Config:
        schema_extra = {
            "example": {
                "prompt": "What are the common types of phishing attacks?",
                "model_id": "fine-tuned",
                "max_new_tokens": 200,
                "temperature": 0.7
            }
        }


class InferenceResponse(BaseModel):
    """Schema for inference responses"""
    
    response: str = Field(..., description="Generated text response")
    model_id: str = Field(..., description="Model ID used for inference")
    generation_time_ms: float = Field(..., description="Generation time in milliseconds")


class DataSourceType(str, Enum):
    """Enum for data source types"""
    
    HF_DATASET = "huggingface_dataset"
    LOCAL_CSV = "local_csv"
    LOCAL_JSON = "local_json"


class TrainingRequest(BaseModel):
    """Schema for training requests"""
    
    base_model_id: Optional[str] = Field(None, description="Base model ID to fine-tune")
    data_source_type: DataSourceType = Field(..., description="Type of data source")
    data_source: Union[str, Dict[str, str]] = Field(
        ..., 
        description="Dataset name or path to data files"
    )
    use_lora: bool = Field(True, description="Whether to use LoRA for efficient fine-tuning")
    lora_config: Optional[Dict[str, Any]] = Field(None, description="LoRA configuration")
    training_args: Optional[Dict[str, Any]] = Field(None, description="Training arguments")
    output_dir: Optional[str] = Field(None, description="Output directory for the model")
    push_to_hub: bool = Field(False, description="Whether to push the model to Hugging Face Hub")
    hub_model_id: Optional[str] = Field(None, description="Hugging Face Hub model ID")

    class Config:
        schema_extra = {
            "example": {
                "base_model_id": "Qwen/Qwen1.5-0.5B",
                "data_source_type": "huggingface_dataset",
                "data_source": "Anurag-Saharan/cybersecurity-threat-intel",
                "use_lora": True,
                "training_args": {
                    "num_train_epochs": 3,
                    "per_device_train_batch_size": 4
                }
            }
        }


class TrainingResponse(BaseModel):
    """Schema for training job status responses"""
    
    job_id: str = Field(..., description="Training job ID")
    status: str = Field(..., description="Job status")
    start_time: Optional[str] = Field(None, description="Start time")
    end_time: Optional[str] = Field(None, description="End time")
    model_id: Optional[str] = Field(None, description="Model ID being trained")