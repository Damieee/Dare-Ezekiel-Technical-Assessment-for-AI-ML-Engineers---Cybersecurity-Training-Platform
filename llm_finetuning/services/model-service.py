from typing import Dict, Any, Optional, List
import os

from app.models.ml_models import QwenModel
from app.config import settings


class ModelService:
    """Service for managing ML model instances and operations"""
    
    def __init__(self):
        """Initialize the model service"""
        self._models = {}
        self._default_model_id = settings.DEFAULT_MODEL_ID
        self._fine_tuned_model_path = settings.FINE_TUNED_MODEL_PATH
    
    def get_model(self, model_id: Optional[str] = None) -> QwenModel:
        """Get or create a model instance
        
        Args:
            model_id: Model identifier or path
            
        Returns:
            QwenModel instance
        """
        model_key = model_id or self._default_model_id
        
        # Check if model is already loaded
        if model_key not in self._models:
            # If asking for fine-tuned model and it exists, use it
            if model_id == "fine-tuned" and os.path.exists(self._fine_tuned_model_path):
                self._models[model_key] = QwenModel(model_id=self._fine_tuned_model_path)
                self._models[model_key].load()
            else:
                # Otherwise load from Hugging Face or specified path
                real_model_id = model_key if model_key != "fine-tuned" else self._default_model_id
                self._models[model_key] = QwenModel(model_id=real_model_id)
                self._models[model_key].load()
                
        return self._models[model_key]
    
    def perform_inference(
        self, 
        prompt: str, 
        model_id: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Run inference using the specified model
        
        Args:
            prompt: Text prompt for generation
            model_id: Model identifier
            generation_config: Optional parameters for text generation
            
        Returns:
            Generated text response
        """
        model = self.get_model(model_id)
        
        # Update generation config if provided
        if generation_config:
            model.update_generation_config(generation_config)
            
        response = model.predict(prompt)
        return response
    
    def list_available_models(self) -> List[str]:
        """List available models
        
        Returns:
            List of available model identifiers
        """
        models = list(self._models.keys())
        
        # Always show the default and fine-tuned (if exists) options
        if self._default_model_id not in models:
            models.append(self._default_model_id)
            
        if os.path.exists(self._fine_tuned_model_path) and "fine-tuned" not in models:
            models.append("fine-tuned")
            
        return models
    
    def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory
        
        Args:
            model_id: Model identifier to unload
            
        Returns:
            True if unloaded successfully, False otherwise
        """
        if model_id in self._models:
            del self._models[model_id]
            return True
        return False


# Create a singleton instance
model_service = ModelService()
