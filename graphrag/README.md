# GraphRAG Cybersecurity Knowledge Base

This application implements a GraphRAG (Graph Retrieval-Augmented Generation) pipeline using Neo4j for knowledge graph storage and FAISS for vector embeddings.

## Features

- Knowledge graph of cybersecurity concepts, vulnerabilities, and mitigations
- Vector embeddings for semantic search capabilities
- Hybrid search combining graph traversal and vector similarity
- Python API for querying the knowledge graph
- Sample data focused on common cybersecurity topics

## Setup

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Set up environment variables:
```
export NEO4J_URI=neo4j+s://<your-instance-id>.databases.neo4j.io
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=your_password
```

3. Initialize the knowledge base:
```
python -m app.initialize_kb
```

4. Run a query:
```
python -m app.query "What are mitigation strategies for XSS attacks?"
```

## Project Structure

- `app/`: Application code
  - `main.py`: Entry point for the application
  - `kb/`: Knowledge base components
  - `query.py`: Query execution module
  - `initialize_kb.py`: Knowledge base initialization script
- `data/`: Sample data for the knowledge base
- `config/`: Configuration files
- `utils/`: Utility functions
