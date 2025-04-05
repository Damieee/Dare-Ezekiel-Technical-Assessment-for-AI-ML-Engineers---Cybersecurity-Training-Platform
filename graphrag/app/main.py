from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from app.kb.graph_db import GraphDatabase
from app.kb.vector_db import VectorDatabase
from app.query import GraphRAGQuery

app = FastAPI(
    title="GraphRAG Cybersecurity Knowledge Base",
    description="Query cybersecurity knowledge using a graph-based RAG pipeline",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize databases
graph_db = None
vector_db = None
query_engine = None

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    include_graph: Optional[bool] = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    graph_data: Optional[Dict[str, Any]] = None

@app.on_event("startup")
async def startup_event():
    global graph_db, vector_db, query_engine
    try:
        graph_db = GraphDatabase()
        vector_db = VectorDatabase()
        query_engine = GraphRAGQuery(graph_db, vector_db)
    except Exception as e:
        print(f"Error initializing databases: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    if graph_db:
        graph_db.close()

@app.get("/")
async def root():
    return {"message": "GraphRAG Cybersecurity Knowledge Base API"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not query_engine:
        raise HTTPException(status_code=500, detail="Query engine not initialized")
    
    try:
        result = query_engine.execute_query(
            request.query, 
            top_k=request.top_k,
            include_graph=request.include_graph
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing query: {str(e)}")

@app.get("/concepts")
async def get_concepts():
    if not graph_db:
        raise HTTPException(status_code=500, detail="Graph database not initialized")
    
    try:
        concepts = graph_db.get_all_concepts()
        return {"concepts": concepts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving concepts: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)
