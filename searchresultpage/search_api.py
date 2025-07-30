# search_api.py

"""
Grid 7.0 - Updated Search API with Hybrid Search Integration
===========================================================

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

# Import the hybrid searcher and existing components
from hybrid_search import HybridSearcher
from process_and_rank import ProductRanker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for loaded models
hybrid_searcher = None
product_ranker = None

# Configuration paths
SPACY_MODEL_PATH = "R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\searchresultpage\\spacy_ner_model"
FAISS_INDEX_DIR = "./faiss_index"
REALTIME_DATA_PATH = "R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\dataset\\realtime_product_info.csv"
PRODUCT_CATALOG_PATH = "R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\dataset\\product_catalog_merged.csv"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    This ensures all models are loaded once at startup and cleaned up on shutdown.
    """
    global hybrid_searcher, product_ranker
    
    # Startup: Load all models once
    logger.info("🚀 Starting up the Grid 7.0 Search API server...")
    logger.info("=" * 60)
    
    try:
        # Initialize HybridSearcher (loads both spaCy NER and FAISS)
        logger.info("🧠 Initializing Hybrid Search System...")
        hybrid_searcher = HybridSearcher(
            spacy_model_path=SPACY_MODEL_PATH,
            faiss_index_dir=FAISS_INDEX_DIR,
            product_catalog_path=PRODUCT_CATALOG_PATH
        )
        logger.info("✅ Hybrid Search System loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize HybridSearcher: {e}")
        raise RuntimeError(f"Failed to initialize HybridSearcher: {e}")
    
    try:
        # Initialize ProductRanker
        logger.info("📊 Initializing Product Ranker...")
        product_ranker = ProductRanker(realtime_data_path=REALTIME_DATA_PATH)
        logger.info("✅ ProductRanker initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize ProductRanker: {e}")
        product_ranker = None
    
    logger.info("🎉 All systems ready! API is now serving requests.")
    logger.info("=" * 60)
    
    yield  # This is where the application runs
    
    # Shutdown: Clean up resources
    logger.info("🛑 Shutting down the API server...")
    hybrid_searcher = None
    product_ranker = None
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
    title="Grid 7.0 Hybrid Search API",
    description="An intelligent product search API combining spaCy NER entity extraction with FAISS semantic search.",
    version="2.0.0",
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
        raise HTTPException(status_code=503, detail="Hybrid search system is not loaded. Please try again later.")
    
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
    Main search endpoint with end-to-end integration.
    
    This endpoint:
    1. Performs hybrid search (NER + Semantic)
    2. Extracts comprehensive features (17 required features)
    3. Applies real-time ranking
    4. Returns enriched results with all features
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
        
        # Step 2: Additional Ranking (if available)
        if product_ranker and search_results:
            rank_start = time.time()
            search_results = product_ranker.rank_products(search_results, request.context)
            rank_time = time.time() - rank_start
            logger.info(f"📊 Ranking completed in {rank_time:.3f}s")
        
        # Step 3: Validate features
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
        
        total_time = time.time() - start_time
        logger.info(f"⚡ Total processing time: {total_time:.3f}s")
        
        return search_results
        
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/health/")
def health_check():
    """Enhanced health check endpoint."""
    global hybrid_searcher, product_ranker
    
    hybrid_status = "loaded" if hybrid_searcher is not None else "not_loaded"
    ranker_status = "loaded" if product_ranker is not None else "not_loaded"
    
    return {
        "status": "ok" if hybrid_searcher is not None else "degraded",
        "hybrid_searcher_status": hybrid_status,
        "product_ranker_status": ranker_status,
        "message": "API is running and ready to serve requests",
        "capabilities": {
            "hybrid_search_with_ranking": hybrid_status == "loaded" and ranker_status == "loaded"
        }
    }

@app.get("/stats/")
def get_system_stats():
    """Get detailed information about the loaded models and systems."""
    global hybrid_searcher, product_ranker
    
    if hybrid_searcher is None:
        raise HTTPException(status_code=503, detail="Hybrid search system is not loaded.")
    
    stats = {
        "hybrid_searcher": {
            "spacy_model_path": hybrid_searcher.spacy_model_path,
            "faiss_model_name": hybrid_searcher.model_name,
            "total_products_indexed": len(hybrid_searcher.product_ids),
            "faiss_index_dimension": hybrid_searcher.faiss_index.d if hasattr(hybrid_searcher.faiss_index, 'd') else "unknown",
            "has_product_catalog": hybrid_searcher.product_catalog is not None,
            "catalog_size": len(hybrid_searcher.product_catalog) if hybrid_searcher.product_catalog is not None else 0
        },
        "build_stats": hybrid_searcher.stats if hybrid_searcher.stats else "No build stats available"
    }
    
    if product_ranker is not None:
        stats["product_ranker"] = {
            "status": "loaded",
            "realtime_data_path": REALTIME_DATA_PATH
        }
    
    return stats

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "Grid 7.0 Hybrid Search API",
        "version": "2.0.0",
        "description": "Intelligent product search combining spaCy NER with FAISS semantic search and advanced ranking",
        "endpoints": {
            "search": "/search/ (POST) - Main hybrid search endpoint with ranking",
            "health": "/health/ (GET) - System health check",
            "stats": "/stats/ (GET) - Detailed system statistics",
            "docs": "/docs (GET) - Interactive API documentation"
        },
        "features": [
            "spaCy NER entity extraction",
            "FAISS semantic search",
            "Hybrid search strategy",
            "Rule-based filtering",
            "Product ranking and enrichment with context",
            "Real-time performance optimization"
        ]
    }   