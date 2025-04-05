from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, 
                      description="The cybersecurity query to process")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, 
                                       description="Temperature for controlling randomness")
    max_length: Optional[int] = Field(500, ge=10, le=2048,
                                    description="Maximum length of generated response")

    class Config:
        schema_extra = {
            "example": {
                "query": "Explain how cross-site scripting (XSS) attacks work and how to prevent them.",
                "temperature": 0.7,
                "max_length": 500
            }
        }

class TrainingExample(BaseModel):
    input: str = Field(..., min_length=1, description="Input prompt or question")
    output: str = Field(..., min_length=1, description="Expected output or answer")

class FineTuneRequest(BaseModel):
    training_data: List[TrainingExample] = Field(..., min_items=10,
                                              description="List of training examples")
    epochs: Optional[int] = Field(3, ge=1, le=10, 
                                description="Number of training epochs")
    learning_rate: Optional[float] = Field(5e-5, description="Learning rate for fine-tuning")
    
    class Config:
        schema_extra = {
            "example": {
                "training_data": [
                    {
                        "input": "What is a buffer overflow attack?",
                        "output": "A buffer overflow attack occurs when a program writes more data to a buffer than it can hold, causing memory corruption. This can lead to code execution, crashes, or data theft."
                    },
                    {
                        "input": "How can I protect against SQL injection?",
                        "output": "To protect against SQL injection: use parameterized queries, validate user input, implement least privilege access, use ORMs, enable prepared statements, and regularly update your database software."
                    }
                ],
                "epochs": 3,
                "learning_rate": 5e-5
            }
        }
