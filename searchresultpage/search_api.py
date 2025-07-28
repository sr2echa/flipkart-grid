# search_api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import os
from contextlib import asynccontextmanager

# Import the searcher class from your existing inference script
from search_inference import FAISSProductSearcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store the searcher instance
searcher = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    This ensures the model is loaded once at startup and cleaned up on shutdown.
    """
    global searcher
    
    # Startup: Load the model once
    logger.info("🚀 Starting up the API server...")
    logger.info("📚 Loading FAISS search model...")
    
    try:
        searcher = FAISSProductSearcher(index_dir="./faiss_index")
        logger.info("✅ Model loaded successfully! Ready to serve requests.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize FAISSProductSearcher: {e}")
        raise RuntimeError(f"Failed to initialize FAISSProductSearcher: {e}")
    
    yield  # This is where the application runs
    
    # Shutdown: Clean up resources if needed
    logger.info("🛑 Shutting down the API server...")
    searcher = None
    logger.info("✅ Cleanup completed.")

# Define the request body for the API endpoint
class SearchRequest(BaseModel):
    query: str
    top_k: int = 100

    class Config:
        # Example for API documentation
        schema_extra = {
            "example": {
                "query": "running shoes for men",
                "top_k": 10
            }
        }

# Create the FastAPI app with lifespan management
app = FastAPI(
    title="Grid 7.0 Semantic Search API",
    description="An API for semantic product search using FAISS and SBERT.",
    version="1.0.0",
    lifespan=lifespan  # This ensures proper startup/shutdown handling
)

@app.post("/search/", response_model=List[Dict[str, Any]])
def perform_search(request: SearchRequest):
    """
    Performs semantic search and returns the top k results.
    This prediction is very fast as the model is already in memory.
    """
    global searcher
    
    if searcher is None:
        raise HTTPException(status_code=503, detail="Search model is not loaded. Please try again later.")
    
    try:
        results = searcher.search(query=request.query, top_k=request.top_k)
        logger.info(f"🔍 Processed search query: '{request.query}' - Found {len(results)} results")
        return results
    except ValueError as e:
        logger.warning(f"⚠️ Invalid search query: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Internal search error: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

@app.get("/health/")
def health_check():
    """Simple health check endpoint to confirm the API is running."""
    global searcher
    
    model_status = "loaded" if searcher is not None else "not_loaded"
    
    return {
        "status": "ok",
        "model_status": model_status,
        "message": "API is running and ready to serve requests"
    }

@app.get("/stats/")
def get_model_stats():
    """Get information about the loaded model and index."""
    global searcher
    
    if searcher is None:
        raise HTTPException(status_code=503, detail="Search model is not loaded.")
    
    return {
        "model_name": searcher.model_name,
        "total_products": len(searcher.product_ids),
        "index_dimension": searcher.index.d if hasattr(searcher.index, 'd') else "unknown",
        "stats": searcher.stats if searcher.stats else "No build stats available"
    }

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "Grid 7.0 Semantic Search API",
        "version": "1.0.0",
        "endpoints": {
            "search": "/search/ (POST)",
            "health": "/health/ (GET)",
            "stats": "/stats/ (GET)",
            "docs": "/docs (GET)"
        }
    }