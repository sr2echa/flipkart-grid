# run_server.py

"""
Production server runner for Grid 7.0 Hybrid Search API
======================================================

This script runs the FastAPI server with the integrated hybrid search system
that combines spaCy NER and FAISS semantic search for optimal performance.

The server ensures that both models (spaCy NER and FAISS) are loaded once
at startup and remain in memory for consistent, fast performance.
"""

import uvicorn
import os
import sys
from pathlib import Path

def check_required_files():
    """Check if all required files and directories exist before starting the server."""
    required_paths = {
        "FAISS Index Directory": "./faiss_index",
        "spaCy NER Model": "R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\searchresultpage\\spacy_ner_model",
        "Realtime Data": "R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\dataset\\realtime_product_info.csv"
    }
    
    missing_paths = []
    
    for name, path in required_paths.items():
        if not os.path.exists(path):
            missing_paths.append(f"❌ {name}: {path}")
        else:
            print(f"✅ {name}: {path}")
    
    if missing_paths:
        print("\n🚨 Missing required files/directories:")
        for missing in missing_paths:
            print(f"  {missing}")
        print("\nPlease ensure all required files are available before starting the server.")
        return False
    
    return True

def run_production_server(host="127.0.0.1", port=8000):
    """Run the server in production mode."""
    print("🚀 Starting Grid 7.0 Semantic Search API in Production Mode")
    print("=" * 70)
    print("✅ Auto-reload is DISABLED - All models will stay in memory")
    print("✅ Optimized for consistent performance")
    print("🧠 Loading: spaCy NER + FAISS Semantic Search")
    print("=" * 70)
    
    # Check required files
    if not check_required_files():
        sys.exit(1)
    
    print("\n🔧 All dependencies verified. Starting server...")
    
    # Run the server
    uvicorn.run(
        "search_api:app",  # Module:app
        host=host,
        port=port,
        reload=False,  # CRITICAL: Disable auto-reload for production
        workers=1,  # Single worker to keep models in memory
        log_level="info",
        access_log=True
    )

def run_development_server(host="127.0.0.1", port=8000):
    """Run the server in development mode with auto-reload."""
    print("🛠️ Starting Grid 7.0 Semantic Search API in Development Mode")
    print("=" * 70)
    print("⚠️ Auto-reload is ENABLED - Models may reload on file changes")
    print("⚠️ Use production mode for consistent performance")
    print("🧠 Loading: spaCy NER + FAISS Semantic Search")
    print("=" * 70)
    
    # Check required files
    if not check_required_files():
        print("\n⚠️ Some files are missing, but continuing in development mode...")
    
    uvicorn.run(
        "search_api:app",
        host=host,
        port=port,
        reload=True,  # Enable auto-reload for development
        log_level="info",
        access_log=True
    )

def main():
    """Main function with command line argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run Grid 7.0 Hybrid Search API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_server.py --mode production
  python run_server.py --mode dev --port 8080
  python run_server.py --host 0.0.0.0 --port 8000
        """
    )
    
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
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check if required files exist, don't start server"
    )
    
    args = parser.parse_args()
    
    # If only checking files
    if args.check_only:
        print("🔍 Checking required files and directories...")
        print("=" * 50)
        success = check_required_files()
        if success:
            print("\n✅ All required files are available!")
            print("🚀 Ready to start the server.")
        else:
            print("\n❌ Some required files are missing.")
            print("📝 Please ensure all dependencies are properly set up.")
        sys.exit(0 if success else 1)
    
    # Normalize mode argument
    mode = "production" if args.mode in ["production", "prod"] else "development"
    
    print(f"🎯 Mode: {mode.upper()}")
    print(f"🌐 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print()
    
    try:
        if mode == "production":
            run_production_server(host=args.host, port=args.port)
        else:
            run_development_server(host=args.host, port=args.port)
            
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()