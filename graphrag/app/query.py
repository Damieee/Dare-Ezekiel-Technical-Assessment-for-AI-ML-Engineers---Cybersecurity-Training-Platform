import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from app.kb.graph_db import GraphDatabase
from app.kb.vector_db import VectorDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphRAGQuery:
    """
    Graph Retrieval Augmented Generation query execution
    Combines graph traversal with vector similarity search
    """
    def __init__(self, graph_db: GraphDatabase, vector_db: VectorDatabase):
        """Initialize the query engine with graph and vector databases"""
        self.graph_db = graph_db
        self.vector_db = vector_db
        logger.info("GraphRAG query engine initialized")
        
    def execute_query(self, query: str, top_k: int = 5, include_graph: bool = True) -> Dict[str, Any]:
        """
        Execute a GraphRAG query against the knowledge base
        
        Args:
            query: User query string
            top_k: Number of relevant results to retrieve
            include_graph: Whether to include graph visualization data
            
        Returns:
            Dictionary containing the answer and supporting evidence
        """
        start_time = time.time()
        logger.info(f"Processing query: {query}")
        
        try:
            # Step 1: Get vector similarity results
            vector_results = self.vector_db.similarity_search(query, k=top_k)
            
            # Step 2: Extract concepts from vector results to use in graph search
            concepts = set()
            for metadata, _ in vector_results:
                if 'concept_name' in metadata:
                    concepts.add(metadata['concept_name'])
                    
            # Step 3: Expand knowledge with graph relationships
            graph_results = []
            for concept in concepts:
                # Get the concept itself
                concept_node = self.graph_db.query_concept(concept)
                if concept_node:
                    graph_results.append(concept_node)
                
                # Get related concepts through graph traversal
                related = self.graph_db.get_related_concepts(concept, max_depth=1)
                if related:
                    graph_results.extend(related)
                
            # Step 4: Merge and deduplicate results
            all_sources = self._merge_results(vector_results, graph_results)
            
            # Step 5: Generate graph visualization data if requested
            graph_data = None
            if include_graph and concepts:
                # Use the first concept as the center of our graph visualization
                center_concept = next(iter(concepts))
                graph_data = self.graph_db.get_concept_neighborhood(center_concept, depth=2)
            
            # Step 6: Create a summary response
            answer = self._generate_answer(query, all_sources)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            logger.info(f"Query processed in {processing_time:.2f} seconds")
            
            # Return formatted results
            return {
                "answer": answer,
                "sources": all_sources[:top_k],  # Limit sources to top_k
                "graph_data": graph_data,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "graph_data": None,
                "processing_time": time.time() - start_time
            }
    
    def _merge_results(self, vector_results: List[Tuple[Dict[str, Any], float]], 
                      graph_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge and deduplicate results from vector and graph searches
        """
        # Process vector results
        seen_ids = set()
        merged_results = []
        
        # Add vector results with similarity scores
        for metadata, score in vector_results:
            if metadata.get('id') not in seen_ids:
                seen_ids.add(metadata.get('id'))
                metadata['relevance_score'] = 1.0 - (score / 10.0)  # Normalize score
                merged_results.append(metadata)
        
        # Add unique graph results
        for result in graph_results:
            if result.get('id') not in seen_ids:
                seen_ids.add(result.get('id'))
                # No direct similarity score, so use lower base relevance
                result['relevance_score'] = 0.6  
                merged_results.append(result)
        
        # Sort by relevance score descending
        merged_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return merged_results
            
    def _generate_answer(self, query: str, sources: List[Dict[str, Any]]) -> str:
        """
        Generate an answer based on retrieved sources
        
        In a full implementation, this would call an LLM with the retrieved context.
        For this example, we'll create a simple aggregated response.
        """
        if not sources:
            return "I couldn't find relevant information about this topic in the knowledge base."
            
        # Categorize sources by type
        concepts = [s for s in sources if s.get('category')]
        
        # Generate a simple answer from top sources
        answer_parts = [
            f"Here's what I found about '{query}':",
            "\n\n"
        ]
        
        # Add concept information
        if concepts:
            main_concept = concepts[0]
            answer_parts.append(f"{main_concept.get('name', 'This concept')}: {main_concept.get('description', '')}")
            
            # Add related concepts
            if len(concepts) > 1:
                answer_parts.append("\n\nRelated concepts:")
                for concept in concepts[1:3]:  # Limit to 3 related concepts
                    answer_parts.append(f"- {concept.get('name')}: {concept.get('description', '')}")
        
        return "".join(answer_parts)
