from abc import ABC, abstractmethod
import os
from typing import Dict, Any, List, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)

from app.config import settings


class BaseModel(ABC):
    """Abstract base class for ML models"""

    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load model from a given path"""
        pass
    
    @abstractmethod
    def predict(self, input_text: str) -> str:
        """Generate predictions from input text"""
        pass
    
    @abstractmethod
    def save(self, save_path: str) -> None:
        """Save model to a given path"""
        pass


class QwenModel(BaseModel):
    """Class for handling Qwen language models"""
    
    def __init__(
        self, 
        model_id: str = settings.DEFAULT_MODEL_ID,
        device: str = None,
        use_gpu: bool = settings.USE_GPU
    ):
        """Initialize QwenModel with model_id
        
        Args:
            model_id: Hugging Face model ID or local path
            device: Device to load model on ('cpu', 'cuda', etc.)
            use_gpu: Whether to use GPU if available
        """
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        
        # Determine device
        if device is None:
            self.device = "cuda" if (torch.cuda.is_available() and use_gpu) else "cpu"
        else:
            self.device = device
            
        # Generation parameters
        self.generation_config = {
            "max_new_tokens": settings.MAX_NEW_TOKENS,
            "temperature": settings.TEMPERATURE,
            "top_p": settings.TOP_P,
            "do_sample": True,
            "repetition_penalty": settings.REPETITION_PENALTY
        }
    
    def load(self, model_path: Optional[str] = None) -> None:
        """Load model and tokenizer from Hugging Face Hub or local path
        
        Args:
            model_path: Override the default model_id if provided
        """
        try:
            path = model_path if model_path else self.model_id
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                path, 
                trust_remote_code=True
            )
            
            load_kwargs = {
                "device_map": "auto" if self.device == "cuda" else None,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "trust_remote_code": True
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
                
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def predict(self, input_text: str) -> str:
        """Generate text from input prompt
        
        Args:
            input_text: The prompt text to generate from
            
        Returns:
            Generated text response
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded before prediction")
        
        try:
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            
            # Set EOS token if available
            if hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
                self.generation_config["eos_token_id"] = self.tokenizer.eos_token_id
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **self.generation_config)
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the response if it includes the prompt
            if response.startswith(input_text):
                response = response[len(input_text):].strip()
                
            return response
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def save(self, save_path: str) -> None:
        """Save model and tokenizer to the specified path
        
        Args:
            save_path: Directory to save the model and tokenizer
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded before saving")
        
        try:
            os.makedirs(save_path, exist_ok=True)
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            print(f"Model saved to {save_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {str(e)}")
    
    def update_generation_config(self, config: Dict[str, Any]) -> None:
        """Update text generation parameters
        
        Args:
            config: Dictionary of generation parameters to update
        """
        self.generation_config.update(config)
    
    def get_generation_config(self) -> Dict[str, Any]:
        """Get current generation config"""
        return self.generation_config.copy()
