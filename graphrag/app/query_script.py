import argparse
import json
import logging
import os
import sys
from typing import Dict, Any

from app.kb.graph_db import GraphDatabase
from app.kb.vector_db import VectorDatabase
from app.query import GraphRAGQuery

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def format_result(result: Dict[str, Any]) -> str:
    """Format the result for display"""
    output = []
    
    # Add the answer
    output.append("\n" + "=" * 80)
    output.append("ANSWER:")
    output.append(result["answer"])
    output.append("=" * 80)
    
    # Add sources
    output.append("\nSOURCES:")
    for i, source in enumerate(result["sources"]):
        output.append(f"\n[{i+1}] {source.get('concept_name', 'Concept')}")
        output.append(f"    Category: {source.get('category', 'Unknown')}")
        if source.get('description'):
            output.append(f"    {source.get('description')[:150]}...")
        if 'relevance_score' in source:
            output.append(f"    Relevance: {source['relevance_score']:.2f}")
    
    # Add processing time
    if "processing_time" in result:
        output.append(f"\nQuery processed in {result['processing_time']:.2f} seconds")
    
    return "\n".join(output)

def save_result(result: Dict[str, Any], output_file: str) -> bool:
    """Save the result to a JSON file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving result: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Query the cybersecurity knowledge graph")
    parser.add_argument("query", type=str, help="The query to process")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--output", "-o", type=str, help="Output file for JSON results")
    parser.add_argument("--graph", "-g", action="store_true", help="Include graph data in results")
    
    args = parser.parse_args()
    
    try:
        # Initialize components
        graph_db = GraphDatabase()
        vector_db = VectorDatabase()
        query_engine = GraphRAGQuery(graph_db, vector_db)
        
        # Execute query
        result = query_engine.execute_query(
            args.query,
            top_k=args.top_k,
            include_graph=args.graph
        )
        
        # Print formatted result
        print(format_result(result))
        
        # Save to file if requested
        if args.output:
            if save_result(result, args.output):
                print(f"\nResults saved to {args.output}")
            else:
                print(f"\nFailed to save results to {args.output}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")
        return 1
    
if __name__ == "__main__":
    sys.exit(main())
