# run_server.py

"""
Production server runner for Grid 7.0 Search API
================================================

This script runs the FastAPI server in production mode without auto-reload,
ensuring the model stays loaded in memory for consistent performance.
"""

import uvicorn
import os
import sys
from pathlib import Path

def run_production_server():
    """Run the server in production mode."""
    print("🚀 Starting Grid 7.0 Search API in Production Mode")
    print("=" * 60)
    print("✅ Auto-reload is DISABLED - Model will stay in memory")
    print("✅ Optimized for consistent performance")
    print("=" * 60)
    
    # Check if the FAISS index directory exists
    index_dir = Path("./faiss_index")
    if not index_dir.exists():
        print(f"❌ Error: Index directory not found at {index_dir}")
        print("Please ensure you have built the FAISS index first.")
        sys.exit(1)
    
    # Run the server
    uvicorn.run(
        "search_api:app",  # Module:app
        host="127.0.0.1",
        port=8000,
        reload=False,  # CRITICAL: Disable auto-reload for production
        workers=1,  # Single worker to keep model in memory
        log_level="info",
        access_log=True
    )

def run_development_server():
    """Run the server in development mode with auto-reload."""
    print("🛠️ Starting Grid 7.0 Search API in Development Mode")
    print("=" * 60)
    print("⚠️ Auto-reload is ENABLED - Model may reload on file changes")
    print("⚠️ Use production mode for consistent performance")
    print("=" * 60)
    
    uvicorn.run(
        "search_api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Grid 7.0 Search API Server")
    parser.add_argument(
        "--mode", 
        choices=["production", "development", "prod", "dev"], 
        default="production",
        help="Server mode (default: production)"
    )
    parser.add_argument(
        "--host", 
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    
    args = parser.parse_args()
    
    # Normalize mode argument
    mode = "production" if args.mode in ["production", "prod"] else "development"
    
    if mode == "production":
        print("🚀 Running in PRODUCTION mode...")
        uvicorn.run(
            "search_api:app",
            host=args.host,
            port=args.port,
            reload=False,  # Disable auto-reload
            workers=1,
            log_level="info",
            access_log=True
        )
    else:
        print("🛠️ Running in DEVELOPMENT mode...")
        uvicorn.run(
            "search_api:app",
            host=args.host,
            port=args.port,
            reload=True,  # Enable auto-reload
            log_level="info",
            access_log=True
        )