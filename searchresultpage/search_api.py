# search_api.py

"""
Grid 7.0 - Updated Search API with Semantic Search Integration
=============================================================

This FastAPI application integrates the HybridSearcher system that combines
spaCy NER entity extraction with FAISS semantic search for intelligent
product discovery.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import os
from contextlib import asynccontextmanager
import time
import traceback

# Import the hybrid searcher for semantic search
from hybrid_search import HybridSearcher
from reranking_model import RerankingModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for loaded models
hybrid_searcher = None
reranking_model = None

# Configuration paths
SPACY_MODEL_PATH = "./spacy_ner_model"
FAISS_INDEX_DIR = "./faiss_index"
PRODUCT_CATALOG_PATH = "../dataset/product_catalog_merged.csv"
RERANKING_MODEL_PATH = "lgbm_rerank_model_with_label_fix.txt"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    This ensures all models are loaded once at startup and cleaned up on shutdown.
    """
    global hybrid_searcher, reranking_model
    
    # Startup: Load all models once
    logger.info("🚀 Starting up the Grid 7.0 Semantic Search API server...")
    logger.info("=" * 60)
    
    try:
        # Initialize HybridSearcher (loads both spaCy NER and FAISS)
        logger.info("🧠 Initializing Semantic Search System...")
        hybrid_searcher = HybridSearcher(
            spacy_model_path=SPACY_MODEL_PATH,
            faiss_index_dir=FAISS_INDEX_DIR,
            product_catalog_path=PRODUCT_CATALOG_PATH
        )
        logger.info("✅ Semantic Search System loaded successfully!")
        
        # Initialize Reranking Model
        logger.info("🎯 Initializing Reranking Model...")
        reranking_model = RerankingModel(model_path=RERANKING_MODEL_PATH)
        logger.info("✅ Reranking Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}")
        raise RuntimeError(f"Failed to initialize models: {e}")
    
    logger.info("🎉 All systems ready! API is now serving requests.")
    logger.info("=" * 60)
    
    yield  # This is where the application runs
    
    # Shutdown: Clean up resources
    logger.info("🛑 Shutting down the API server...")
    hybrid_searcher = None
    reranking_model = None
    logger.info("✅ Cleanup completed.")

# Request models
class SearchRequest(BaseModel):
    query: str
    context: Dict[str, Any]  # e.g., {"location": "Surat", "user_preferences": {...}}
    top_k: int = 100

    class Config:
        schema_extra = {
            "example": {
                "query": "red nike running shoes for men",
                "context": {"location": "Mumbai", "price_range": "under_5000"},
                "top_k": 10
            }
        }

# Create the FastAPI app
app = FastAPI(
    title="Grid 7.0 Enhanced Semantic Search API",
    description="An intelligent product search API with enhanced NER entity extraction, FAISS semantic search, and LightGBM reranking for optimal results.",
    version="2.2.0",
    lifespan=lifespan
)

@app.post("/hybrid_search/", response_model=List[Dict[str, Any]])
def hybrid_search(request: SearchRequest):
    """
    Performs hybrid search using both NER entity extraction and semantic search.
    
    The system:
    1. First extracts entities (brand, color, category, etc.) using spaCy NER
    2. If entities are found, performs rule-based filtering
    3. If no entities or insufficient results, falls back to FAISS semantic search
    """
    global hybrid_searcher
    
    if hybrid_searcher is None:
        raise HTTPException(status_code=503, detail="Semantic search system is not loaded. Please try again later.")
    
    try:
        results = hybrid_searcher.search(query=request.query, top_k=request.top_k)
        logger.info(f"🔍 Processed hybrid search query: '{request.query}' - Found {len(results)} results")
        return results
        
    except ValueError as e:
        logger.warning(f"⚠️ Invalid search query: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Internal hybrid search error: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

@app.post("/search/", response_model=List[Dict[str, Any]])
async def search_endpoint(request: SearchRequest):
    """
    Main search endpoint with complete pipeline: NER filtering, semantic search, and reranking.
    
    This endpoint implements the full search pipeline:
    1. NER entity extraction and filtering
    2. Semantic search using FAISS
    3. Feature enrichment
    4. LightGBM reranking for optimal results
    """
    try:
        logger.info(f"🔍 Search request: '{request.query}' (top {request.top_k})")
        
        if hybrid_searcher is None:
            logger.warning("⚠️ Search system not available - returning fallback response")
            return [{
                'product_id': 'fallback_1',
                'title': 'System temporarily unavailable',
                'brand': 'System',
                'category': 'Service',
                'price': 0,
                'rating': 0,
                'is_f_assured': False,
                'search_method': 'fallback',
                'message': 'Search system is initializing. Please try again in a moment.'
            }]
        
        # Step 1: Hybrid Search with Feature Enrichment
        start_time = time.time()
        search_results = hybrid_searcher.search(
            query=request.query,
            top_k=request.top_k,
            user_context=request.context
        )
        search_time = time.time() - start_time
        
        logger.info(f"✅ Search completed in {search_time:.3f}s - Found {len(search_results)} products")
        
        # Step 2: Validate features
        if hybrid_searcher and hybrid_searcher.feature_extractor and search_results:
            validation_start = time.time()
            is_valid = hybrid_searcher.feature_extractor.validate_features(search_results)
            validation_time = time.time() - validation_start
            
            if is_valid:
                logger.info(f"✅ Feature validation passed in {validation_time:.3f}s")
                logger.info(f"✅ All {len(search_results)} products contain required features")
            else:
                logger.warning(f"⚠️ Feature validation failed in {validation_time:.3f}s")
        else:
            logger.warning("⚠️ Feature extractor not available - features may be incomplete")
        
        # Step 3: Reranking (NEW STEP)
        if reranking_model and search_results:
            rerank_start = time.time()
            logger.info("🎯 Applying LightGBM reranking to optimize results...")
            
            reranked_results = reranking_model.rerank_results(search_results)
            rerank_time = time.time() - rerank_start
            
            logger.info(f"✅ Reranking completed in {rerank_time:.3f}s")
            logger.info(f"🎯 Reranked {len(reranked_results)} products for optimal relevance")
            
            # Update final results with reranked ones
            search_results = reranked_results
        else:
            logger.warning("⚠️ Reranking model not available - returning original results")
        
        total_time = time.time() - start_time
        logger.info(f"⚡ Total processing time: {total_time:.3f}s")
        logger.info(f"📊 Final pipeline: NER → Semantic Search → Feature Enrichment → Reranking")
        
        return search_results
        
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/health/")
def health_check():
    """Enhanced health check endpoint."""
    global hybrid_searcher, reranking_model
    
    hybrid_status = "loaded" if hybrid_searcher is not None else "not_loaded"
    reranking_status = "loaded" if reranking_model is not None else "not_loaded"
    
    return {
        "status": "ok" if hybrid_searcher is not None else "degraded",
        "semantic_search_status": hybrid_status,
        "reranking_status": reranking_status,
        "message": "API is running and ready to serve requests",
        "capabilities": {
            "semantic_search": hybrid_status == "loaded",
            "reranking": reranking_status == "loaded"
        }
    }

@app.get("/stats/")
def get_system_stats():
    """Get detailed information about the loaded models and systems."""
    global hybrid_searcher, reranking_model
    
    if hybrid_searcher is None:
        return {
            "status": "not_loaded",
            "message": "Search systems are not loaded",
            "models": {}
        }
    
    stats = {
        "status": "loaded",
        "message": "All systems operational",
        "models": {
            "hybrid_searcher": {
                "status": "loaded",
                "product_count": len(hybrid_searcher.product_ids) if hybrid_searcher.product_ids else 0,
                "model_name": hybrid_searcher.model_name,
                "has_spacy_ner": hybrid_searcher.nlp is not None,
                "has_faiss_index": hybrid_searcher.faiss_index is not None,
                "has_sbert_model": hybrid_searcher.sbert_model is not None,
                "has_product_catalog": hybrid_searcher.product_catalog is not None
            },
            "reranking_model": {
                "status": "loaded" if reranking_model is not None else "not_loaded",
                "model_available": reranking_model is not None,
                "model_path": RERANKING_MODEL_PATH if reranking_model else None
            }
        }
    }
    
    # Add detailed stats if available
    if hybrid_searcher.stats:
        stats["models"]["hybrid_searcher"]["detailed_stats"] = hybrid_searcher.stats
    
    return stats

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "Grid 7.0 Semantic Search API with Reranking",
        "version": "2.2.0",
        "description": "Intelligent product search combining NER, semantic search, and LightGBM reranking",
        "pipeline": "NER Filtering → Semantic Search → Feature Enrichment → Reranking",
        "endpoints": {
            "search": "/search/",
            "hybrid_search": "/hybrid_search/",
            "health": "/health/",
            "stats": "/stats/"
        },
        "documentation": "/docs"
    }   