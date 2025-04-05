import os
import json
import logging
from typing import Dict, Any, List
import pandas as pd
from tqdm import tqdm

from app.kb.graph_db import GraphDatabase
from app.kb.vector_db import VectorDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_json_data(file_path: str) -> Dict[str, Any]:
    """Load data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return {}

def initialize_knowledge_base():
    """Initialize the knowledge base with cybersecurity data"""
    logger.info("Starting knowledge base initialization")
    
    # Initialize databases
    graph_db = GraphDatabase()
    vector_db = VectorDatabase()
    
    # Data file paths
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    concepts_file = os.path.join(data_dir, 'cybersec_concepts.json')
    relations_file = os.path.join(data_dir, 'cybersec_relations.json')
    
    # Load data
    concepts_data = load_json_data(concepts_file)
    relations_data = load_json_data(relations_file)
    
    if not concepts_data.get('concepts'):
        logger.error("No concepts found in the data file")
        return False
    
    # Process concepts
    logger.info(f"Processing {len(concepts_data['concepts'])} concepts")
    for concept in tqdm(concepts_data['concepts'], desc="Creating concept nodes"):
        # Create concept in graph database
        success = graph_db.create_concept(
            name=concept['name'],
            description=concept['description'],
            category=concept['category']
        )
        
        # Add to vector database for semantic search
        if success:
            vector_db.add_texts(
                texts=[concept['description']],
                metadatas=[{
                    'id': concept['name'],
                    'concept_name': concept['name'],
                    'category': concept['category'],
                    'description': concept['description']
                }]
            )
    
    # Process relationships
    if relations_data.get('relationships'):
        logger.info(f"Processing {len(relations_data['relationships'])} relationships")
        for relation in tqdm(relations_data['relationships'], desc="Creating relationships"):
            graph_db.create_relationship(
                source_name=relation['source'],
                target_name=relation['target'],
                rel_type=relation['type'],
                properties=relation.get('properties', {})
            )
    
    logger.info("Knowledge base initialization complete")
    return True

if __name__ == "__main__":
    initialize_knowledge_base()
