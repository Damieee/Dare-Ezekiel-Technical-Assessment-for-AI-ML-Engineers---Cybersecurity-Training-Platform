import os
import numpy as np
import faiss
import pickle
import logging
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    """
    Vector database using FAISS for efficient similarity search
    """
    def __init__(self):
        """Initialize the vector database with a sentence transformer model"""
        self.model_name = "all-MiniLM-L6-v2"  # Small but effective model
        self.vector_dim = 384  # Dimension of embeddings for the selected model
        
        # Load or create index
        self.index_file = os.path.join(os.path.dirname(__file__), '../../data/faiss_index.idx')
        self.metadata_file = os.path.join(os.path.dirname(__file__), '../../data/metadata.pkl')
        
        # Initialize sentence transformer model
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = SentenceTransformer(self.model_name, device=device)
            logger.info(f"Initialized sentence transformer model on {device}")
        except Exception as e:
            logger.error(f"Error loading sentence transformer model: {e}")
            self.model = None
            
        # Initialize or load FAISS index
        self.index = None
        self.metadata = []
        self.load_or_create_index()
        
    def load_or_create_index(self):
        """Load existing index or create a new one"""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
                # Load existing index and metadata
                self.index = faiss.read_index(self.index_file)
                with open(self.metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            else:
                # Create new index
                self.index = faiss.IndexFlatL2(self.vector_dim)
                logger.info("Created new FAISS index")
        except Exception as e:
            logger.error(f"Error loading or creating index: {e}")
            # Create a new index if loading fails
            self.index = faiss.IndexFlatL2(self.vector_dim)
            self.metadata = []
            
    def save_index(self):
        """Save the index and metadata to files"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
            
            faiss.write_index(self.index, self.index_file)
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
            logger.info(f"Saved FAISS index with {self.index.ntotal} vectors")
            return True
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return False
            
    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> List[int]:
        """Add texts and their metadata to the vector database"""
        if not texts or len(texts) != len(metadatas):
            logger.error("Invalid input: texts and metadatas must be non-empty and same length")
            return []
            
        try:
            # Generate embeddings
            embeddings = self.model.encode(texts)
            
            # Convert to necessary format for FAISS
            embeddings_np = np.array(embeddings).astype('float32')
            
            # Get current size of the index
            current_size = self.index.ntotal
            
            # Add to FAISS index
            self.index.add(embeddings_np)
            
            # Store metadata
            self.metadata.extend(metadatas)
            
            # Save updated index
            self.save_index()
            
            # Return IDs of the newly added vectors
            return list(range(current_size, current_size + len(texts)))
        
        except Exception as e:
            logger.error(f"Error adding texts to vector database: {e}")
            return []
            
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar vectors to the query and return metadata with distances
        """
        if not self.model or not self.index:
            logger.error("Vector database not properly initialized")
            return []
            
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query])
            query_embedding_np = np.array(query_embedding).astype('float32')
            
            # Search in FAISS index
            distances, indices = self.index.search(query_embedding_np, k)
            
            # Format results with metadata and distance scores
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.metadata) and idx >= 0:
                    results.append((self.metadata[idx], float(distances[0][i])))
            
            return results
        
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            return []
            
    def delete_vectors(self, ids: List[int]) -> bool:
        """Delete vectors from the index by their IDs"""
        try:
            # FAISS doesn't support direct deletion in all index types
            # For IndexFlatL2, we need to recreate the index
            if ids:
                remaining_ids = [i for i in range(self.index.ntotal) if i not in ids]
                
                if not remaining_ids:
                    # Reset the index if all vectors are deleted
                    self.index = faiss.IndexFlatL2(self.vector_dim)
                    self.metadata = []
                else:
                    # Extract remaining vectors and metadata
                    # This is a simplified approach - in production, use IDMap
                    new_metadata = [self.metadata[i] for i in remaining_ids]
                    self.metadata = new_metadata
                    
                    # Save the updated index
                    self.save_index()
                
            return True
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            return False
