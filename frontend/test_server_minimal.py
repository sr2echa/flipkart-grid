#!/usr/bin/env python3
"""
Clean Server with Real Search Integration
"""

import os
import sys
import time
import traceback
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from simple_autosuggest import SimpleAutosuggestSystem
    from data_preprocessing import DataPreprocessor
    AUTOSUGGEST_AVAILABLE = True
    print("✅ Autosuggest components loaded successfully")
except ImportError as e:
    print(f"❌ Autosuggest imports failed: {e}")
    AUTOSUGGEST_AVAILABLE = False

class CleanFrontendServer:
    """Clean Flask server with real search integration."""
    
    def __init__(self):
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        CORS(self.app)
        
        # Load personas from dataset
        self.personas = self._load_personas_from_dataset()
        
        # Location and event data
        self.locations = [
            {'id': 'Mumbai', 'name': 'Mumbai'},
            {'id': 'Delhi', 'name': 'Delhi'},
            {'id': 'Bangalore', 'name': 'Bangalore'},
            {'id': 'Chennai', 'name': 'Chennai'},
            {'id': 'Pune', 'name': 'Pune'}
        ]
        
        self.events = [
            {'id': 'none', 'name': 'None'},
            {'id': 'diwali', 'name': 'Diwali'},
            {'id': 'summer', 'name': 'Summer'},
            {'id': 'sale', 'name': 'Sale Season'}
        ]
        
        self.setup_routes()
        self.load_autosuggest_system()
    
    def _load_personas_from_dataset(self):
        """Load persona information from the dataset."""
        try:
            # Use robust relative path
            current_file_dir = os.path.dirname(os.path.abspath(__file__))  # frontend/
            project_root = os.path.dirname(current_file_dir)  # project root
            dataset_path = os.path.join(project_root, 'dataset', 'query_product_training_features_only.csv')
            
            df = pd.read_csv(dataset_path, nrows=1000)
            unique_personas = list(set(df['persona_tag'].tolist()))
            print(f"🎭 Found {len(unique_personas)} unique personas: {sorted(unique_personas)}")
            
            personas = {}
            for persona_tag in unique_personas:
                if persona_tag == 'brand_lover':
                    personas['brand_lover'] = {
                        'id': 'brand_lover',
                        'name': 'Premium Brand Enthusiast',
                        'description': 'Prefers well-known, premium brands and quality products with strong brand reputation'
                    }
                elif persona_tag == 'value_seeker':
                    personas['value_seeker'] = {
                        'id': 'value_seeker',
                        'name': 'Value-Conscious Shopper', 
                        'description': 'Seeks the best deals, compares prices, and looks for cost-effective options'
                    }
                elif persona_tag == 'newbie':
                    personas['newbie'] = {
                        'id': 'newbie',
                        'name': 'First-time Shopper',
                        'description': 'New to online shopping, needs guidance and prefers popular, highly-rated products'
                    }
                elif persona_tag == 'quality_hunter':
                    personas['quality_hunter'] = {
                        'id': 'quality_hunter',
                        'name': 'Quality-Focused Buyer',
                        'description': 'Prioritizes product quality, reviews, and specifications over price considerations'
                    }
            
            if 'tech_enthusiast' not in personas:
                personas['tech_enthusiast'] = {
                    'id': 'tech_enthusiast',
                    'name': 'Technology Enthusiast',
                    'description': 'Passionate about the latest technology trends and cutting-edge gadgets'
                }
            
            return personas
                
        except Exception as e:
            print(f"⚠️ Failed to load personas: {e}")
            return {
                'brand_lover': {'id': 'brand_lover', 'name': 'Premium Brand Enthusiast', 'description': 'Prefers premium brands'},
                'tech_enthusiast': {'id': 'tech_enthusiast', 'name': 'Technology Enthusiast', 'description': 'Loves technology'}
            }
    
    def load_autosuggest_system(self):
        """Load autosuggest system."""
        if AUTOSUGGEST_AVAILABLE:
            try:
                preprocessor = DataPreprocessor()
                preprocessor.run_all_preprocessing()
                self.data = preprocessor.get_processed_data()
                
                self.autosuggest = SimpleAutosuggestSystem()
                self.autosuggest.build_system(self.data)
                
                self.autosuggest_loaded = True
                print("✅ Autosuggest loaded successfully!")
                
            except Exception as e:
                print(f"❌ Failed to load autosuggest: {e}")
                self.autosuggest_loaded = False
        else:
            self.autosuggest_loaded = False
    
    def setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/api/health')
        def health():
            return jsonify({
                'status': 'healthy',
                'components': {
                    'autosuggest': 'loaded' if hasattr(self, 'autosuggest_loaded') and self.autosuggest_loaded else 'failed',
                    'real_search': 'available'
                },
                'timestamp': time.time()
            })
        
        @self.app.route('/api/config')
        def config():
            return jsonify({
                'personas': list(self.personas.values()),
                'locations': self.locations,
                'events': self.events
            })
        
        @self.app.route('/api/suggest', methods=['POST'])
        def suggest():
            try:
                data = request.json
                query = data.get('query', '').strip()
                
                if not query or not hasattr(self, 'autosuggest_loaded') or not self.autosuggest_loaded:
                    return jsonify({'suggestions': [], 'metadata': {'system': 'none'}})
                
                suggestions = self.autosuggest.get_suggestions(query, 5)
                formatted_suggestions = [[suggestion, score] for suggestion, score in suggestions]
                
                return jsonify({
                    'suggestions': formatted_suggestions,
                    'metadata': {'system': 'basic', 'total_suggestions': len(formatted_suggestions)},
                    'response_time_ms': 50
                })
                
            except Exception as e:
                print(f"❌ Suggest error: {e}")
                return jsonify({'suggestions': [], 'error': str(e)}), 500
        
        @self.app.route('/api/search', methods=['POST'])
        def search():
            """Search endpoint using real search system."""
            try:
                data = request.json
                query = data.get('query', '').strip()
                context = data.get('context', {})
                top_k = data.get('top_k', 10)
                
                location = context.get('location', 'Mumbai')
                persona_tag = context.get('persona_tag', 'brand_lover')
                
                print(f"🔍 Search request: query='{query}', location={location}, persona={persona_tag}, top_k={top_k}")
                
                # Use actual searchresultpage API
                try:
                    # Add searchresultpage to path
                    import sys
                    import os
                    
                    # Get absolute paths for robustness
                    current_file_dir = os.path.dirname(os.path.abspath(__file__))  # frontend/
                    project_root = os.path.dirname(current_file_dir)  # project root
                    searchresultpage_path = os.path.join(project_root, 'searchresultpage')
                    
                    if searchresultpage_path not in sys.path:
                        sys.path.append(searchresultpage_path)
                    
                    from hybrid_search import HybridSearcher
                    
                    # Initialize real search components with proper paths
                    spacy_model_path = os.path.join(searchresultpage_path, "spacy_ner_model")
                    faiss_index_dir = os.path.join(searchresultpage_path, "faiss_index")
                    product_catalog_path = os.path.join(project_root, "dataset", "product_catalog_merged.csv")
                    
                    print(f"📂 Using paths:")
                    print(f"   SpaCy Model: {spacy_model_path}")
                    print(f"   FAISS Index: {faiss_index_dir}")
                    print(f"   Product Catalog: {product_catalog_path}")
                    
                    # Check for missing metadata and create if needed
                    metadata_pkl = os.path.join(faiss_index_dir, "product_metadata.pkl")
                    metadata_json = os.path.join(faiss_index_dir, "product_metadata.json")
                    
                    if not os.path.exists(metadata_pkl) and os.path.exists(metadata_json):
                        print("📝 Converting JSON metadata to pickle format...")
                        import json
                        import pickle
                        with open(metadata_json, 'r') as f:
                            metadata = json.load(f)
                        with open(metadata_pkl, 'wb') as f:
                            pickle.dump(metadata, f)
                        print("✅ Metadata conversion completed")
                    
                    searcher = HybridSearcher(
                        spacy_model_path=spacy_model_path,
                        faiss_index_dir=faiss_index_dir,
                        product_catalog_path=product_catalog_path
                    )
                    
                    # Execute real search (correct method signature) 
                    print("🔍 Executing HybridSearcher (spaCy NER + FAISS semantic search)...")
                    search_results = searcher.search(query=query, top_k=top_k, user_context=context)
                    
                    if search_results and len(search_results) > 0:
                        print(f"✅ Real HybridSearcher returned {len(search_results)} results")
                        
                        # Try to enable reranker with fallback
                        try:
                            print("🔄 Attempting to load LightGBM reranker...")
                            
                            reranking_model_path = os.path.join(searchresultpage_path, "lgbm_rerank_model_with_label_fix.txt")
                            
                            from reranking_model import RerankingModel
                            reranker = RerankingModel(model_path=reranking_model_path)
                            reranked_results = reranker.rerank_results(search_results[:20])  # Limit to 20 for reranking
                            print(f"✅ LightGBM reranking applied - {len(reranked_results)} results")
                            search_results = reranked_results
                        except Exception as rerank_error:
                            print(f"⚠️ Reranking failed, using original results: {rerank_error}")
                            # Continue with original results
                        
                        # Fix JSON serialization issues and ensure proper image URLs
                        def convert_to_json_serializable(obj):
                            """Convert numpy/pandas types to JSON serializable types."""
                            import numpy as np
                            import pandas as pd
                            
                            if isinstance(obj, dict):
                                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                            elif isinstance(obj, list):
                                return [convert_to_json_serializable(v) for v in obj]
                            elif isinstance(obj, (np.int32, np.int64)):
                                return int(obj)
                            elif isinstance(obj, (np.float32, np.float64)):
                                return float(obj)
                            elif pd.isna(obj):
                                return None
                            else:
                                return obj
                        
                        # Convert results to JSON serializable format
                        serializable_results = convert_to_json_serializable(search_results)
                        
                        # Ensure results have proper image URLs from dataset
                        for result in serializable_results:
                            if not result.get('image_url') or result.get('image_url') == 'placeholder.jpg':
                                # Use dataset image_url if available, otherwise generate a proper placeholder
                                if result.get('brand'):
                                    result['image_url'] = f"https://via.placeholder.com/300x300/2874f0/ffffff?text={result['brand'].replace(' ', '+')}"
                                else:
                                    result['image_url'] = "https://via.placeholder.com/300x300/f0f0f0/333333?text=Product"
                        
                        return jsonify(serializable_results)
                    else:
                        print("⚠️ Real HybridSearcher returned no results")
                        
                except Exception as e:
                    print(f"⚠️ Real API failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Fallback response with better info
                return jsonify([{
                    "rank": 1,
                    "product_id": "SEARCH_001", 
                    "title": f"Search for '{query}' - Real API temporarily unavailable",
                    "brand": "Generic",
                    "category": "Electronics",
                    "price": 25999.0,
                    "similarity_score": 0.7,
                    "search_method": "fallback",
                    "original_price": 25999.0,
                    "parsed_price": 25999.0,
                    "rating": 4.0,
                    "is_f_assured": False,
                    "persona_tag": persona_tag,
                    "avg_price_last_k_clicks": 25000.0,
                    "preferred_brands_count": 2,
                    "session_length": 10,
                    "query_frequency": 30,
                    "brand_match": 0.5,
                    "price_gap_to_avg": 0.04,
                    "offer_preference_match": 0.7,
                    "semantic_similarity": 0.7,
                    "query_intent_similarity": 0.1,
                    "product_embedding_mean": 1.5,
                    "event": "general",
                    "click_count": 50,
                    "relevance_score": 5.0,
                    "reranked": True
                }])
                
            except Exception as e:
                print(f"❌ Search error: {e}")
                traceback.print_exc()
                return jsonify([]), 500
    
    def run(self, debug=True, host='127.0.0.1', port=3000):
        print("🚀 Starting Clean Flipkart Search Server...")
        print(f"📍 URL: http://{host}:{port}")
        print("=" * 50)
        self.app.run(debug=debug, host=host, port=port, threaded=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean Flipkart Search Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=3000, help="Port to bind to")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug mode")
    
    args = parser.parse_args()
    
    server = CleanFrontendServer()
    server.run(debug=not args.no_debug, host=args.host, port=args.port)