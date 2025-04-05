import os
from neo4j import GraphDatabase as Neo4jDriver
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphDatabase:
    """
    Wrapper for Neo4j graph database operations
    """
    def __init__(self):
        """Initialize Neo4j connection using environment variables"""
        self.uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        
        try:
            self.driver = Neo4jDriver(self.uri, auth=(self.username, self.password))
            logger.info("Connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            # Fall back to in-memory mock if connection fails
            self.driver = None
            logger.warning("Using in-memory mock database instead")
    
    def close(self):
        """Close the Neo4j driver connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def create_concept(self, name: str, description: str, category: str) -> bool:
        """Create a cybersecurity concept node in the graph"""
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MERGE (c:Concept {name: $name})
                    SET c.description = $description,
                        c.category = $category,
                        c.created = timestamp()
                    RETURN c
                    """,
                    name=name,
                    description=description,
                    category=category
                )
                return result.single() is not None
        except Exception as e:
            logger.error(f"Error creating concept: {e}")
            return False
    
    def create_relationship(self, source_name: str, target_name: str, rel_type: str, properties: Dict = None) -> bool:
        """Create a relationship between two concept nodes"""
        if properties is None:
            properties = {}
            
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (source:Concept {name: $source_name})
                    MATCH (target:Concept {name: $target_name})
                    MERGE (source)-[r:%s]->(target)
                    SET r += $properties
                    RETURN r
                    """ % rel_type,
                    source_name=source_name,
                    target_name=target_name,
                    properties=properties
                )
                return result.single() is not None
        except Exception as e:
            logger.error(f"Error creating relationship: {e}")
            return False
    
    def query_concept(self, name: str) -> Optional[Dict]:
        """Query a concept by name"""
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (c:Concept {name: $name})
                    RETURN c {.*, id: id(c)} as concept
                    """,
                    name=name
                )
                record = result.single()
                return record["concept"] if record else None
        except Exception as e:
            logger.error(f"Error querying concept: {e}")
            return None
    
    def get_related_concepts(self, concept_name: str, rel_type: Optional[str] = None, 
                            max_depth: int = 2) -> List[Dict]:
        """Get concepts related to a given concept"""
        rel_clause = f"[r:{rel_type}]" if rel_type else "[r]"
        
        try:
            with self.driver.session() as session:
                result = session.run(
                    f"""
                    MATCH (c:Concept {{name: $concept_name}})-{rel_clause}*1..{max_depth}-(related:Concept)
                    RETURN DISTINCT related {{.*, id: id(related)}} as concept
                    """,
                    concept_name=concept_name
                )
                return [record["concept"] for record in result]
        except Exception as e:
            logger.error(f"Error getting related concepts: {e}")
            return []
    
    def execute_custom_query(self, query: str, params: Dict = None) -> List[Dict]:
        """Execute a custom Cypher query"""
        if params is None:
            params = {}
            
        try:
            with self.driver.session() as session:
                result = session.run(query, params)
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error executing custom query: {e}")
            return []
    
    def get_all_concepts(self) -> List[Dict]:
        """Get all concepts in the knowledge base"""
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (c:Concept)
                    RETURN c {.*, id: id(c)} as concept
                    """
                )
                return [record["concept"] for record in result]
        except Exception as e:
            logger.error(f"Error retrieving concepts: {e}")
            return []
    
    def get_concept_neighborhood(self, concept_name: str, depth: int = 1) -> Dict[str, Any]:
        """Get a concept and its neighborhood for visualization"""
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (c:Concept {name: $name})
                    CALL apoc.path.subgraphAll(c, {maxLevel: $depth}) 
                    YIELD nodes, relationships
                    RETURN nodes, relationships
                    """,
                    name=concept_name,
                    depth=depth
                )
                record = result.single()
                if not record:
                    return {"nodes": [], "relationships": []}
                    
                nodes = [dict(n.items()) for n in record["nodes"]]
                rels = [
                    {
                        "source": r.start_node.get("name"),
                        "target": r.end_node.get("name"),
                        "type": r.type
                    }
                    for r in record["relationships"]
                ]
                
                return {"nodes": nodes, "relationships": rels}
        except Exception as e:
            logger.error(f"Error retrieving concept neighborhood: {e}")
            return {"nodes": [], "relationships": []}
