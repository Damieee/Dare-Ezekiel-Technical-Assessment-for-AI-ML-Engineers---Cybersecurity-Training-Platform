from typing import Dict, Any, Optional, List, Union, Tuple
import os
import json
from datetime import datetime

import torch
from transformers import (
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset, DatasetDict
import huggingface_hub

from app.models.ml_models import QwenModel
from app.utils.preprocessing import DataPreprocessor
from app.config import settings


class TrainingService:
    """Service for fine-tuning models"""
    
    def __init__(self):
        """Initialize the training service"""
        self.fine_tuned_model_path = settings.FINE_TUNED_MODEL_PATH
        self.current_training_job = None
        self.training_history = []
    
    def prepare_model_for_training(
        self, 
        model_id: str = settings.DEFAULT_MODEL_ID,
        use_lora: bool = True,
        lora_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[QwenModel, Any]:
        """Prepare a model for training, optionally with LoRA
        
        Args:
            model_id: Base model ID
            use_lora: Whether to use LoRA parameter-efficient fine-tuning
            lora_config: LoRA configuration parameters
            
        Returns:
            Tuple of (QwenModel, transformers model)
        """
        # Create and load model
        qwen_model = QwenModel(model_id=model_id)
        qwen_model.load()
        
        # Get transformers model
        model = qwen_model.model
        
        # If using LoRA, prepare model
        if use_lora:
            # Default LoRA config
            default_lora_config = {
                "r": 16,  # rank
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": TaskType.CAUSAL_LM,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
            }
            
            # Update with user-provided config
            if lora_config:
                default_lora_config.update(lora_config)
                
            # Create LoRA config
            peft_config = LoraConfig(**default_lora_config)
            
            # Prepare model for LoRA
            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                model = prepare_model_for_kbit_training(model)
                
            # Apply LoRA adapter
            model = get_peft_model(model, peft_config)
        
        return qwen_model, model
        
    def train_model(
        self,
        dataset: Union[str, Dict[str, Any], Dataset, DatasetDict],
        tokenizer,
        model,
        training_args: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train/fine-tune a model
        
        Args:
            dataset: Input dataset
            tokenizer: Tokenizer for the model
            model: The model to train
            training_args: Training arguments
            output_dir: Output directory for the trained model
            
        Returns:
            Training metrics
        """
        # Set output directory
        if not output_dir:
            output_dir = self.fine_tuned_model_path
        
        # Prepare dataset
        data_preprocessor = DataPreprocessor(tokenizer)
        
        if isinstance(dataset, (str, dict)):
            # Load dataset if string or dict
            if isinstance(dataset, str):
                loaded_dataset = data_preprocessor.load_dataset(dataset_name=dataset)
            else:
                loaded_dataset = data_preprocessor.load_dataset(data_files=dataset)
            
            processed_dataset = data_preprocessor.prepare_dataset(loaded_dataset)
        else:
            # Already a Dataset or DatasetDict
            processed_dataset = data_preprocessor.prepare_dataset(dataset)
        
        # Default training arguments
        default_training_args = {
            "output_dir": output_dir,
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "per_device_train_batch_size": settings.TRAINING_BATCH_SIZE,
            "per_device_eval_batch_size": settings.EVAL_BATCH_SIZE,
            "learning_rate": settings.LEARNING_RATE,
            "num_train_epochs": settings.NUM_TRAIN_EPOCHS,
            "weight_decay": 0.01,
            "logging_dir": os.path.join(output_dir, "logs"),
            "logging_steps": 100,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "report_to": "none",  # Disable wandb, etc.
            "remove_unused_columns": False
        }
        
        # Update with user-provided args
        if training_args:
            default_training_args.update(training_args)
        
        # Create TrainingArguments
        args = TrainingArguments(**default_training_args)
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # For causal language modeling
        )
        
        # Create Trainer
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=processed_dataset["train"],
            eval_dataset=processed_dataset["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Start training
        training_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.current_training_job = {
            "id": f"training_job_{training_time}",
            "status": "running",
            "start_time": training_time,
            "model_id": getattr(model, "name_or_path", str(model.__class__)),
            "output_dir": output_dir
        }
        
        try:
            train_result = trainer.train()
            metrics = train_result.metrics
            
            # Save model
            trainer.save_model(output_dir)
            
            # Save tokenizer
            tokenizer.save_pretrained(output_dir)
            
            # Save training args
            with open(os.path.join(output_dir, "training_args.json"), "w") as f:
                json.dump(default_training_args, f)
            
            # Update job status
            self.current_training_job["status"] = "completed"
            self.current_training_job["end_time"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.current_training_job["metrics"] = metrics
            self.training_history.append(self.current_training_job)
            
            return {
                "job_id": self.current_training_job["id"],
                "status": "completed",
                "metrics": metrics,
                "model_path": output_dir
            }
            
        except Exception as e:
            # Update job status on failure
            self.current_training_job["status"] = "failed"
            self.current_training_job["error"] = str(e)
            self.current_training_job["end_time"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.training_history.append(self.current_training_job)
            
            raise RuntimeError(f"Training failed: {str(e)}")
    
    def push_to_hub(self, local_model_path: str, hub_model_id: str) -> str:
        """Push a fine-tuned model to Hugging Face Hub
        
        Args:
            local_model_path: Path to the local model
            hub_model_id: Hugging Face Hub model ID (username/model-name)
            
        Returns:
            URL of the model on the Hub
        """
        if not settings.HF_TOKEN:
            raise ValueError("Hugging Face token not set in environment variables or config")
        
        # Login to Hugging Face
        huggingface_hub.login(token=settings.HF_TOKEN)
        
        # Push model to Hub
        model_url = huggingface_hub.push_to_hub_model(
            repo_id=hub_model_id,
            model_path=local_model_path
        )
        
        return model_url
    
    def get_training_job_status(self, job_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of a training job
        
        Args:
            job_id: ID of the training job
            
        Returns:
            Status information for the job
        """
        if job_id:
            # Find specific job
            for job in self.training_history:
                if job["id"] == job_id:
                    return job
            
            raise ValueError(f"Training job with ID {job_id} not found")
        else:
            # Return current job if any
            if self.current_training_job:
                return self.current_training_job
            else:
                return {"status": "no_job_running"}
    
    def list_training_jobs(self) -> List[Dict[str, Any]]:
        """List all training jobs
        
        Returns:
            List of training job information
        """
        return self.training_history


# Create a singleton instance
training_service = TrainingService()
