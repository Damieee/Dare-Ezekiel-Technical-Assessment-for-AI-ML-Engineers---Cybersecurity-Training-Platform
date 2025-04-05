from typing import Dict, List, Union, Callable, Any
import os

from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizer

from ..config import settings


class DataPreprocessor:
    """Class for data preprocessing tasks"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        """Initialize preprocessor with tokenizer
        
        Args:
            tokenizer: The tokenizer to use for preprocessing
        """
        self.tokenizer = tokenizer
        self.max_length = settings.MAX_SEQ_LENGTH
    
    def load_dataset(
        self, 
        dataset_name: str = None, 
        data_files: Union[str, Dict[str, str]] = None
    ) -> Union[Dataset, DatasetDict]:
        """Load dataset from Hugging Face or local files
        
        Args:
            dataset_name: Name of HF dataset
            data_files: Path to local data files
            
        Returns:
            Loaded dataset
        """
        if dataset_name:
            return load_dataset(dataset_name)
        elif data_files:
            if isinstance(data_files, str) and os.path.isfile(data_files):
                # Auto-detect file format from extension
                file_extension = os.path.splitext(data_files)[1].lower()
                if file_extension == '.csv':
                    format = 'csv'
                elif file_extension == '.json':
                    format = 'json'
                elif file_extension == '.jsonl':
                    format = 'json'
                else:
                    format = 'text'
                return load_dataset(format, data_files=data_files)
            else:
                # Handle dict of splits
                return load_dataset('csv', data_files=data_files)
        else:
            raise ValueError("Either dataset_name or data_files must be provided")
    
    def tokenize_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Tokenize text examples
        
        Args:
            examples: Dictionary of examples with text column
            
        Returns:
            Dictionary with tokenized inputs
        """
        text_column = "text"  # Default column name, can be made configurable
        
        # Handle different possible text column names
        if text_column not in examples:
            for possible_column in ["content", "input", "prompt", "instruction"]:
                if possible_column in examples:
                    text_column = possible_column
                    break
        
        if text_column not in examples:
            raise ValueError(f"Could not find text column in dataset. Available columns: {list(examples.keys())}")
        
        return self.tokenizer(
            examples[text_column],
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )
    
    def prepare_dataset(
        self, 
        dataset: Union[Dataset, DatasetDict],
        text_column: str = "text",
        train_test_split: float = 0.1
    ) -> Dict[str, Dataset]:
        """Prepare dataset for training
        
        Args:
            dataset: Input dataset
            text_column: Column containing text data
            train_test_split: Fraction of data to use for evaluation
            
        Returns:
            Dict with train and validation datasets
        """
        # Handle different dataset structures
        if isinstance(dataset, DatasetDict):
            if "train" in dataset and "validation" not in dataset:
                # Create validation split if not present
                train_val = dataset["train"].train_test_split(test_size=train_test_split)
                dataset = DatasetDict({
                    "train": train_val["train"],
                    "validation": train_val["test"]
                })
            prepared_dataset = dataset
        else:
            # Single dataset, split into train/validation
            splits = dataset.train_test_split(test_size=train_test_split)
            prepared_dataset = {
                "train": splits["train"],
                "validation": splits["test"]
            }
        
        # Verify text column exists and map
        for split in prepared_dataset:
            if text_column not in prepared_dataset[split].column_names:
                available_columns = prepared_dataset[split].column_names
                raise ValueError(f"Text column '{text_column}' not found. Available columns: {available_columns}")
        
        # Tokenize datasets
        tokenized_datasets = {}
        for split_name, split_dataset in prepared_dataset.items():
            tokenized_datasets[split_name] = split_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=[col for col in split_dataset.column_names if col != text_column]
            )
        
        return tokenized_datasets
