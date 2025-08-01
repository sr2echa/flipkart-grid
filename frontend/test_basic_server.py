#!/usr/bin/env python3
"""
Basic Server Test - Minimal Version for Debugging
================================================

A minimal version to test configuration loading without heavy ML dependencies.
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

# Try basic imports only
try:
    from simple_autosuggest import SimpleAutosuggestSystem
    from data_preprocessing import DataPreprocessor
    AUTOSUGGEST_AVAILABLE = True
    print("✅ Basic autosuggest components loaded successfully")
except ImportError as e:
    print(f"❌ Autosuggest imports failed: {e}")
    AUTOSUGGEST_AVAILABLE = False

class BasicFrontendServer:
    """Basic Flask server for testing configuration loading."""
    
    def __init__(self):
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        CORS(self.app)
        
        # System components
        self.autosuggest = None
        self.data = None
        self.autosuggest_loaded = False
        
        # Load personas from dataset
        self.personas = self._load_personas_from_dataset()
        
        # Fallback personas if dataset loading fails
        if not self.personas:
            self.personas = {
                'tech_enthusiast': {
                    'id': 'tech_enthusiast',
                    'name': 'Tech Enthusiast',
                    'description': 'Loves latest gadgets and technology',
                    'previous_queries': ['laptop', 'smartphone', 'gaming', 'tech'],
                    'clicked_categories': ['Electronics', 'Computers', 'Mobiles'],
                    'clicked_brands': ['Apple', 'Samsung', 'Dell', 'HP'],
                    'persona': 'tech_enthusiast'
                },
                'fashion_lover': {
                    'id': 'fashion_lover',
                    'name': 'Fashion Lover',
                    'description': 'Passionate about style and trends',
                    'previous_queries': ['shoes', 'dress', 'fashion', 'style'],
                    'clicked_categories': ['Clothing', 'Footwear', 'Fashion'],
                    'clicked_brands': ['Nike', 'Adidas', 'Zara', 'H&M'],
                    'persona': 'fashion_lover'
                },
                'budget_shopper': {
                    'id': 'budget_shopper',
                    'name': 'Budget Shopper',
                    'description': 'Value-conscious, seeks best deals',
                    'previous_queries': ['cheap', 'discount', 'sale', 'offer'],
                    'clicked_categories': ['Budget Items', 'Sale Items'],
                    'clicked_brands': ['Generic', 'Local'],
                    'persona': 'budget_shopper'
                },
                'sports_enthusiast': {
                    'id': 'sports_enthusiast',
                    'name': 'Sports Enthusiast',
                    'description': 'Active lifestyle, sports equipment',
                    'previous_queries': ['sports', 'fitness', 'gym', 'running'],
                    'clicked_categories': ['Sports', 'Fitness', 'Athletic'],
                    'clicked_brands': ['Nike', 'Adidas', 'Puma', 'Reebok'],
                    'persona': 'sports_enthusiast'
                }
            }
        
        # Location data
        self.locations = [
            {'id': 'Mumbai', 'name': 'Mumbai'},
            {'id': 'Delhi', 'name': 'Delhi'},
            {'id': 'Bangalore', 'name': 'Bangalore'},
            {'id': 'Chennai', 'name': 'Chennai'},
            {'id': 'Kolkata', 'name': 'Kolkata'},
            {'id': 'Hyderabad', 'name': 'Hyderabad'},
            {'id': 'Pune', 'name': 'Pune'},
            {'id': 'Ahmedabad', 'name': 'Ahmedabad'}
        ]
        
        # Event data
        self.events = [
            {'id': 'none', 'name': 'None', 'boost_keywords': []},
            {'id': 'diwali', 'name': 'Diwali', 'boost_keywords': ['lights', 'decoration', 'sweets', 'gift']},
            {'id': 'summer', 'name': 'Summer', 'boost_keywords': ['cooling', 'ac', 'fan', 'summer']},
            {'id': 'monsoon', 'name': 'Monsoon', 'boost_keywords': ['umbrella', 'raincoat', 'waterproof']},
            {'id': 'sale', 'name': 'Sale Season', 'boost_keywords': ['discount', 'offer', 'sale', 'deal']}
        ]
        
        self.setup_routes()
        self.load_autosuggest_system()
    
    def _load_personas_from_dataset(self):
        """Load persona information from the dataset."""
        try:
            # Load persona training data
            df = pd.read_csv('../dataset/query_product_training_features_only.csv', nrows=1000)
            
            print(f"📊 Loaded persona dataset with {len(df)} rows")
            print(f"📋 Found persona_tag column!")
            
            # Extract unique persona tags using set for all 4 tags
            unique_personas = list(set(df['persona_tag'].tolist()))
            print(f"🎭 Found {len(unique_personas)} unique personas: {sorted(unique_personas)}")
            
            # Create personas based on actual dataset tags
            personas = {}
            
            for persona_tag in unique_personas:
                if persona_tag == 'brand_lover':
                    personas['brand_lover'] = {
                        'id': 'brand_lover',
                        'name': 'Brand Lover',
                        'description': 'Prefers premium and popular brands',
                        'previous_queries': ['apple', 'samsung', 'nike', 'adidas'],
                        'clicked_categories': ['Electronics', 'Fashion', 'Sports'],
                        'clicked_brands': ['Apple', 'Samsung', 'Nike', 'Adidas'],
                        'persona': 'brand_lover'
                    }
                elif persona_tag == 'value_seeker':
                    personas['value_seeker'] = {
                        'id': 'value_seeker',
                        'name': 'Value Seeker',
                        'description': 'Looks for best value and deals',
                        'previous_queries': ['cheap', 'discount', 'sale', 'best price'],
                        'clicked_categories': ['Budget Items', 'Sale Items'],
                        'clicked_brands': ['Local', 'Generic'],
                        'persona': 'value_seeker'
                    }
                elif persona_tag == 'newbie':
                    personas['newbie'] = {
                        'id': 'newbie',
                        'name': 'Shopping Newbie',
                        'description': 'New to online shopping, needs guidance',
                        'previous_queries': ['popular', 'best', 'recommended'],
                        'clicked_categories': ['General', 'Popular Items'],
                        'clicked_brands': ['Popular', 'Recommended'],
                        'persona': 'newbie'
                    }
                elif persona_tag == 'quality_hunter':
                    personas['quality_hunter'] = {
                        'id': 'quality_hunter',
                        'name': 'Quality Hunter',
                        'description': 'Prioritizes quality and ratings over price',
                        'previous_queries': ['high quality', 'best rated', 'premium', 'top quality'],
                        'clicked_categories': ['Premium', 'High Rated', 'Quality'],
                        'clicked_brands': ['Premium Brands', 'Quality'],
                        'persona': 'quality_hunter'
                    }
            
            # Add fallback personas for compatibility
            if 'tech_enthusiast' not in personas:
                personas['tech_enthusiast'] = {
                    'id': 'tech_enthusiast',
                    'name': 'Tech Enthusiast',
                    'description': 'Loves latest gadgets and technology',
                    'previous_queries': ['laptop', 'smartphone', 'gaming', 'tech'],
                    'clicked_categories': ['Electronics', 'Computers', 'Mobiles'],
                    'clicked_brands': ['Apple', 'Samsung', 'Dell', 'HP'],
                    'persona': 'tech_enthusiast'
                }
            
            print(f"✅ Created {len(personas)} personas from dataset")
            return personas
                
        except Exception as e:
            print(f"⚠️ Failed to load personas from dataset: {e}")
            return None
    
    def load_autosuggest_system(self):
        """Load the basic autosuggest system."""
        print("🚀 Loading basic autosuggest system...")
        
        if AUTOSUGGEST_AVAILABLE:
            try:
                # Load and preprocess data
                preprocessor = DataPreprocessor()
                preprocessor.run_all_preprocessing()
                self.data = preprocessor.get_processed_data()
                
                # Initialize autosuggest system
                self.autosuggest = SimpleAutosuggestSystem()
                self.autosuggest.build_system(self.data)
                
                self.autosuggest_loaded = True
                print("✅ Basic autosuggest loaded successfully!")
                
            except Exception as e:
                print(f"❌ Failed to load autosuggest: {e}")
                print(f"Error details: {traceback.format_exc()}")
                self.autosuggest_loaded = False
        else:
            print("❌ Autosuggest components not available")
            self.autosuggest_loaded = False
    
    def setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Serve the frontend."""
            return render_template('index.html')
        
        @self.app.route('/api/health')
        def health():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'components': {
                    'autosuggest': 'loaded' if self.autosuggest_loaded else 'failed'
                },
                'timestamp': time.time(),
                'version': '1.0.0 - Basic'
            })
        
        @self.app.route('/api/config')
        def config():
            """Get system configuration."""
            try:
                return jsonify({
                    'personas': list(self.personas.values()),
                    'locations': self.locations,
                    'events': self.events
                })
            except Exception as e:
                print(f"❌ Config error: {e}")
                return jsonify({'error': 'Configuration failed'}), 500
        
        @self.app.route('/api/suggest', methods=['POST'])
        def suggest():
            """Get autosuggest suggestions."""
            start_time = time.time()
            
            try:
                data = request.json
                query = data.get('query', '').strip()
                
                if not query:
                    return jsonify({
                        'suggestions': [],
                        'response_time_ms': 0,
                        'metadata': {'query_length': 0}
                    })
                
                if not self.autosuggest_loaded:
                    return jsonify({
                        'error': 'Autosuggest system not loaded',
                        'suggestions': [],
                        'response_time_ms': 0
                    }), 503
                
                # Get basic suggestions
                suggestions = self.autosuggest.get_suggestions(query, 5)
                
                # Format suggestions - ensure proper format for JavaScript
                formatted_suggestions = []
                for suggestion, score in suggestions:
                    formatted_suggestions.append([suggestion, score])  # Return as [text, score] array
                
                response_time = (time.time() - start_time) * 1000
                
                return jsonify({
                    'suggestions': formatted_suggestions,
                    'metadata': {
                        'query_length': len(query),
                        'system': 'basic',
                        'total_suggestions': len(formatted_suggestions)
                    },
                    'response_time_ms': round(response_time, 2)
                })
                
            except Exception as e:
                print(f"ERROR in /api/suggest: {e}")
                traceback.print_exc()
                return jsonify({
                    'error': str(e),
                    'suggestions': [],
                    'response_time_ms': (time.time() - start_time) * 1000
                }), 500
        
        @self.app.route('/api/search', methods=['POST'])
        def search():
            """Advanced search endpoint matching searchresultpage format."""
            try:
                # Parse input matching searchresultpage format
                data = request.json
                query = data.get('query', '').strip()
                context = data.get('context', {})
                top_k = data.get('top_k', 10)
                
                location = context.get('location', 'Mumbai')
                persona_tag = context.get('persona_tag', 'brand_lover')
                
                print(f"🔍 Search request: query='{query}', location={location}, persona={persona_tag}, top_k={top_k}")
                
                # Generate advanced search results matching the format
                results = self._generate_advanced_search_results(query, context, top_k)
                
                return jsonify(results)
                
            except Exception as e:
                print(f"❌ Search error: {e}")
                traceback.print_exc()
                return jsonify([]), 500
        
    def _extract_price_filter(self, query):
        """Extract price filter from query."""
        import re
        
        # Look for patterns like "under 50000", "below 30000", etc.
        under_pattern = r'(?:under|below|less than|<)\s*(\d+)'
        over_pattern = r'(?:over|above|more than|>)\s*(\d+)'
        
        under_match = re.search(under_pattern, query.lower())
        over_match = re.search(over_pattern, query.lower())
        
        if under_match:
            return {"max_price": float(under_match.group(1)), "price_type": "under"}
        elif over_match:
            return {"min_price": float(over_match.group(1)), "price_type": "over"}
        
        return None
        
    def _generate_advanced_search_results(self, query, context, top_k):
        """Generate search results using real datasets."""
        try:
            # Import and use real search system
            from real_search_system import get_real_search_system
            
            search_system = get_real_search_system()
            if search_system and search_system.search_ready:
                return search_system.search_products(query, context, top_k)
            else:
                # Fallback to basic search if real system not available
                return self._fallback_search(query, context, top_k)
                
        except Exception as e:
            print(f"❌ Real search failed, using fallback: {e}")
            return self._fallback_search(query, context, top_k)
    
    def _fallback_search(self, query, context, top_k):
        """Fallback search when real system is not available."""
        import random
        
        # Extract search context
        location = context.get('location', 'Mumbai')
        persona_tag = context.get('persona_tag', 'brand_lover')
        
        # Extract price filter
        price_filter = self._extract_price_filter(query)
        
        # Simple fallback results
        return [{
            "rank": 1,
            "product_id": "FALLBACK_001",
            "title": f"Fallback result for '{query}'",
            "brand": "Generic",
            "category": "Electronics",
            "price": 25999.0,
            "similarity_score": 0.5,
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
            "semantic_similarity": 0.5,
            "query_intent_similarity": 0.1,
            "product_embedding_mean": 1.5,
            "event": "general",
            "click_count": 50,
            "relevance_score": 5.0,
            "reranked": True
        }]
                {'title': 'Samsung Galaxy S24', 'brand': 'Samsung', 'base_price': 79999},
                {'title': 'Samsung Galaxy A54', 'brand': 'Samsung', 'base_price': 38999},
                {'title': 'Samsung Galaxy M34', 'brand': 'Samsung', 'base_price': 18999},
            ]
        elif 'laptop' in query.lower():
            base_products = [
                {'title': 'Dell XPS 13', 'brand': 'Dell', 'base_price': 179990},
                {'title': 'HP Pavilion 15', 'brand': 'HP', 'base_price': 65999},
                {'title': 'Lenovo ThinkPad E14', 'brand': 'Lenovo', 'base_price': 55999},
                {'title': 'ASUS VivoBook 15', 'brand': 'ASUS', 'base_price': 45999},
            ]
        else:
            # Generic electronics
            base_products = [
                {'title': f'Electronics Product for {query}', 'brand': 'Brand A', 'base_price': 25999},
                {'title': f'Tech Item {query}', 'brand': 'Brand B', 'base_price': 35999},
                {'title': f'Gadget {query}', 'brand': 'Brand C', 'base_price': 15999},
            ]
        
        # Generate advanced results
        results = []
        for i, product in enumerate(base_products[:top_k]):
            # Simulate price variations
            price_variation = random.uniform(0.8, 1.2)
            final_price = product['base_price'] * price_variation
            
            # Apply price filter
            if price_filter:
                if price_filter.get('price_type') == 'under' and final_price > price_filter['max_price']:
                    continue
                elif price_filter.get('price_type') == 'over' and final_price < price_filter.get('min_price', 0):
                    continue
            
            # Generate persona-based features
            if persona_tag == 'brand_lover':
                avg_price_clicks = random.uniform(20000, 30000)
                preferred_brands = 3
                brand_match = 1.0 if product['brand'] in ['Apple', 'Samsung'] else 0.5
            elif persona_tag == 'value_seeker':
                avg_price_clicks = random.uniform(10000, 20000)
                preferred_brands = 1
                brand_match = 0.3
            elif persona_tag == 'quality_hunter':
                avg_price_clicks = random.uniform(25000, 40000)
                preferred_brands = 2
                brand_match = 0.8
            else:  # newbie
                avg_price_clicks = random.uniform(15000, 25000)
                preferred_brands = 2
                brand_match = 0.6
            
            result = {
                "rank": i + 1,
                "product_id": f"P{random.randint(10000, 19999)}",
                "title": product['title'],
                "brand": product['brand'],
                "category": "Electronics",
                "price": round(final_price, 2),
                "similarity_score": round(random.uniform(0.7, 0.75), 4),
                "search_method": "semantic_search",
                "original_price": round(final_price, 2),
                "parsed_price": round(final_price, 2),
                "rating": round(random.uniform(3.8, 4.5), 1),
                "is_f_assured": random.choice([True, False]),
                "persona_tag": persona_tag,
                "avg_price_last_k_clicks": round(avg_price_clicks, 2),
                "preferred_brands_count": preferred_brands,
                "session_length": random.randint(3, 15),
                "query_frequency": random.randint(5, 50),
                "brand_match": brand_match,
                "price_gap_to_avg": round((final_price - avg_price_clicks) / avg_price_clicks, 6),
                "offer_preference_match": round(random.uniform(0.5, 0.9), 1),
                "semantic_similarity": round(random.uniform(0.72, 0.74), 4),
                "query_intent_similarity": round(random.uniform(0.1, 0.15), 4),
                "product_embedding_mean": round(random.uniform(1.1, 1.9), 4),
                "event": "tech_sale" if 'tech' in query.lower() or any(brand in query.lower() for brand in ['apple', 'samsung']) else "general",
                "click_count": random.randint(30, 100),
                "relevance_score": round(random.uniform(7.5, 8.5), 6),
                "reranked": True
            }
            
            # Add price filter info if applied
            if price_filter:
                result["price_filter_applied"] = price_filter
            
            results.append(result)
        
        # Sort by relevance score (descending)
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Update ranks after sorting
        for i, result in enumerate(results):
            result['rank'] = i + 1
        
        print(f"✅ Generated {len(results)} advanced search results")
        return results
    
    def run(self, debug=True, host='127.0.0.1', port=3000):
        """Run the basic frontend server."""
        print("🚀 Starting Basic Flipkart Search Server...")
        print("=" * 60)
        print(f"📍 Application URL: http://{host}:{port}")
        print(f"🎨 Design: Original autosuggest design")
        print(f"🏗️  Architecture: Single Flask server")
        print(f"📦 Dependencies: Basic (pandas, flask)")
        print(f"🎯 Debug mode: {debug}")
        print("=" * 60)
        
        print("📊 Component Status:")
        print(f"   Basic Autosuggest: {'✅ Loaded' if self.autosuggest_loaded else '❌ Failed'}")
        
        if self.autosuggest_loaded:
            print("🎉 Basic system ready!")
        else:
            print("⚠️ Limited functionality - check component status above")
        
        print(f"\n💻 Access the application at: http://{host}:{port}")
        print(f"🔧 Health check: http://{host}:{port}/api/health")
        print("=" * 60)
        
        self.app.run(debug=debug, host=host, port=port, threaded=True)

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Basic Flipkart Search Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=3000, help="Port to bind to")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug mode")
    
    args = parser.parse_args()
    
    server = BasicFrontendServer()
    server.run(
        debug=not args.no_debug,
        host=args.host,
        port=args.port
    )

if __name__ == "__main__":
    main()