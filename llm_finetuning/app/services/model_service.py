from typing import List, Dict, Any, Optional
import os
import logging
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    pipeline
)
from trl import SFTTrainer
from datasets import Dataset
from huggingface_hub import HfApi, login

from app.models.request_models import TrainingExample
from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_id = settings.FINETUNED_MODEL_ID
        self.initialized = False
        self.training_in_progress = False
        
        # Login to Hugging Face Hub if token is provided
        if settings.HF_TOKEN:
            try:
                login(token=settings.HF_TOKEN)
                logger.info("Successfully logged in to Hugging Face Hub")
            except Exception as e:
                logger.error(f"Failed to login to Hugging Face Hub: {e}")
    
    async def initialize_model(self):
        """Initialize the model for inference"""
        try:
            if not self.initialized:
                logger.info(f"Initializing model: {self.model_id}")
                
                # Use 8-bit quantization to reduce memory footprint
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_4bit_compute_dtype="float16"
                )
                
                # Load model and tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="auto",
                    quantization_config=quantization_config
                )
                
                self.initialized = True
                logger.info("Model initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            return False
    
    async def generate_response(self, query: str, temperature: float = 0.7, max_length: int = 500) -> str:
        """Generate a response from the model"""
        try:
            # Initialize model if not already done
            if not self.initialized:
                success = await self.initialize_model()
                if not success:
                    raise Exception("Failed to initialize model")
            
            # Create generation config
            gen_kwargs = {
                "max_length": max_length,
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
                "num_return_sequences": 1,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            
            # Format prompt for Qwen-specific format
            prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate response
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs
            )
            
            # Extract response text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Extract only the assistant's response
            response_start = generated_text.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
            response_end = generated_text.find("<|im_end|>", response_start)
            
            if response_end == -1:  # If no end token, take the rest
                response = generated_text[response_start:]
            else:
                response = generated_text[response_start:response_end]
                
            return response.strip()
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise Exception(f"Failed to generate response: {str(e)}")
    
    async def fine_tune_model(self, 
                             training_data: List[TrainingExample], 
                             epochs: int = 3,
                             learning_rate: float = 5e-5,
                             job_id: str = None) -> Dict[str, Any]:
        """Fine-tune the model on cybersecurity data"""
        try:
            if self.training_in_progress:
                return {"status": "error", "message": "Training already in progress"}
            
            self.training_in_progress = True
            logger.info(f"Starting fine-tuning job {job_id}")
            
            # Convert training examples to the format expected by the model
            formatted_data = []
            for example in training_data:
                formatted_data.append({
                    "text": f"<|im_start|>user\n{example.input}<|im_end|>\n<|im_start|>assistant\n{example.output}<|im_end|>"
                })
            
            # Create a dataset
            dataset = Dataset.from_list(formatted_data)
            
            # Load base model and tokenizer
            base_model_id = settings.BASE_MODEL_ID
            tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            
            # Add special tokens if they don't exist
            special_tokens = ["<|im_start|>", "<|im_end|>"]
            for token in special_tokens:
                if token not in tokenizer.get_vocab():
                    tokenizer.add_special_tokens({"additional_special_tokens": [token]})
            
            # Configure quantization for training
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_use_double_quant=True
            )
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                quantization_config=bnb_config,
                device_map="auto"
            )
            
            # Setup LoRA config for parameter-efficient fine-tuning
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            )
            
            # Setup trainer
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                peft_config=peft_config,
                dataset_text_field="text",
                max_seq_length=512,
                tokenizer=tokenizer,
                args=transformers.TrainingArguments(
                    output_dir=f"./results/{job_id}",
                    num_train_epochs=epochs,
                    per_device_train_batch_size=4,
                    gradient_accumulation_steps=4,
                    learning_rate=learning_rate,
                    logging_steps=10,
                    save_strategy="epoch",
                    optim="paged_adamw_8bit"
                ),
            )
            
            # Train the model
            trainer.train()
            
            # Save model to Hugging Face Hub
            output_dir = f"./results/{job_id}/final"
            trainer.model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            
            # Push to Hub if token available
            if settings.HF_TOKEN:
                model_name = f"cybersec-qwen-{job_id}"
                trainer.push_to_hub(model_name)
                logger.info(f"Model pushed to hub: {model_name}")
            
            self.training_in_progress = False
            return {
                "status": "success", 
                "message": f"Fine-tuning completed for job {job_id}",
                "model_path": output_dir
            }
            
        except Exception as e:
            self.training_in_progress = False
            logger.error(f"Error during fine-tuning: {e}")
            return {"status": "error", "message": str(e)}

# Dependency injection
def get_model_service():
    return ModelService()
