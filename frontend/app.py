#!/usr/bin/env python3
"""
Minimal Integrated Frontend Server for Flipkart Autosuggest
==========================================================

This Flask server integrates autosuggest functionality with minimal dependencies.
Uses the exact original design from the autosuggest system.
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import time
import json
import os
import sys
from typing import Dict, List, Any
import traceback

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import autosuggest components
try:
    from simple_autosuggest import SimpleAutosuggestSystem
    from data_preprocessing import DataPreprocessor
    AUTOSUGGEST_AVAILABLE = True
    print("✅ Autosuggest components loaded successfully")
except ImportError as e:
    print(f"❌ Autosuggest imports failed: {e}")
    AUTOSUGGEST_AVAILABLE = False

# Import search components with graceful fallback
try:
    from search_system import SimplifiedSearcher, search_products
    BASIC_SEARCH_AVAILABLE = True
    print("✅ Basic search components loaded successfully")
except ImportError as e:
    print(f"❌ Basic search imports failed: {e}")
    BASIC_SEARCH_AVAILABLE = False

# Try to import enhanced components (graceful fallback)
try:
    from persona_search import PersonaAwareSearcher
    PERSONA_SEARCH_AVAILABLE = True
    print("✅ Persona search loaded successfully")
except ImportError as e:
    print(f"⚠️ Persona search unavailable: {e}")
    PERSONA_SEARCH_AVAILABLE = False

try:
    from enhanced_autosuggest import EnhancedAutosuggestSystem
    ENHANCED_AUTOSUGGEST_AVAILABLE = True
    print("✅ Enhanced autosuggest loaded successfully")
except ImportError as e:
    print(f"⚠️ Enhanced autosuggest unavailable: {e}")
    ENHANCED_AUTOSUGGEST_AVAILABLE = False

SEARCH_AVAILABLE = BASIC_SEARCH_AVAILABLE

class IntegratedFrontendServer:
    """Integrated Flask server with autosuggest and search functionality."""
    
    def __init__(self):
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        CORS(self.app)
        
        # System components
        self.autosuggest = None
        self.enhanced_autosuggest = None
        self.searcher = None
        self.persona_searcher = None
        self.data = None
        self.autosuggest_loaded = False
        self.enhanced_autosuggest_loaded = False
        self.search_loaded = False
        
        # Load personas from dataset
        self.personas = self._load_personas_from_dataset()
        
        # Fallback personas if dataset loading fails
        if not self.personas:
            self.personas = {
            'tech_enthusiast': {
                'name': 'Tech Enthusiast',
                'description': 'Loves latest gadgets and technology',
                'previous_queries': ['laptop', 'smartphone', 'gaming', 'apple', 'samsung'],
                'clicked_categories': ['Electronics', 'Computers', 'Mobiles & Accessories'],
                'clicked_brands': ['Apple', 'Samsung', 'Dell', 'HP', 'Sony'],
                'persona': 'tech_enthusiast'
            },
            'fashion_lover': {
                'name': 'Fashion Lover',
                'description': 'Passionate about style and trends',
                'previous_queries': ['shoes', 'dress', 'shirt', 'jeans', 'nike'],
                'clicked_categories': ['Clothing', 'Footwear', 'Fashion'],
                'clicked_brands': ['Nike', 'Adidas', 'Zara', 'H&M', 'Puma'],
                'persona': 'fashion_lover'
            },
            'budget_shopper': {
                'name': 'Budget Shopper',
                'description': 'Value-conscious, seeks best deals',
                'previous_queries': ['under 1000', 'cheap', 'offer', 'discount', 'sale'],
                'clicked_categories': ['Budget Items', 'Sale Items'],
                'clicked_brands': ['Generic', 'Local', 'Affordable'],
                'persona': 'budget_shopper'
            },
            'sports_enthusiast': {
                'name': 'Sports Enthusiast', 
                'description': 'Active lifestyle, fitness focused',
                'previous_queries': ['sports', 'running', 'fitness', 'nike', 'shoes'],
                'clicked_categories': ['Sports & Fitness', 'Footwear', 'Health'],
                'clicked_brands': ['Nike', 'Adidas', 'Puma', 'Reebok', 'Fitbit'],
                'persona': 'sports_enthusiast'
            },
            'luxury_shopper': {
                'name': 'Luxury Shopper',
                'description': 'Premium quality seeker',
                'previous_queries': ['premium', 'luxury', 'apple', 'samsung', 'above 50000'],
                'clicked_categories': ['Premium', 'Luxury', 'High-end'],
                'clicked_brands': ['Apple', 'Samsung', 'Sony', 'LG', 'Canon'],
                'persona': 'luxury_shopper'
            }
        }
        
        # Locations
        self.locations = [
            'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata',
            'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow'
        ]
        
        # Events/seasons
        self.events = [
            {'id': 'none', 'name': 'None', 'boost_keywords': []},
            {'id': 'diwali', 'name': 'Diwali', 'boost_keywords': ['lights', 'decor', 'gifts', 'sweets', 'traditional']},
            {'id': 'holi', 'name': 'Holi', 'boost_keywords': ['colors', 'festival', 'celebration', 'traditional']},
            {'id': 'christmas', 'name': 'Christmas', 'boost_keywords': ['gifts', 'decoration', 'celebration', 'winter']},
            {'id': 'ipl', 'name': 'IPL Season', 'boost_keywords': ['jersey', 'cricket', 'sports', 'team', 'match']},
            {'id': 'wedding', 'name': 'Wedding Season', 'boost_keywords': ['formal', 'traditional', 'gifts', 'jewelry', 'dress']},
            {'id': 'summer', 'name': 'Summer', 'boost_keywords': ['cooling', 'ac', 'fan', 'summer', 'light']},
            {'id': 'monsoon', 'name': 'Monsoon', 'boost_keywords': ['umbrella', 'raincoat', 'waterproof', 'monsoon']},
            {'id': 'sale', 'name': 'Sale Season', 'boost_keywords': ['discount', 'offer', 'sale', 'deal', 'cheap']}
        ]
        
        self.setup_routes()
        self.load_autosuggest_system()
        self.load_search_system()
    
    def load_autosuggest_system(self):
        """Load both traditional and enhanced autosuggest systems."""
        print("🚀 Loading autosuggest systems...")
        
        if AUTOSUGGEST_AVAILABLE:
            try:
                # Load and preprocess data
                preprocessor = DataPreprocessor()
                preprocessor.run_all_preprocessing()
                self.data = preprocessor.get_processed_data()
                
                # Initialize traditional autosuggest system (fallback)
                self.autosuggest = SimpleAutosuggestSystem()
                self.autosuggest.build_system(self.data)
                self.autosuggest_loaded = True
                print("✅ Traditional autosuggest loaded successfully!")
                
                # Initialize enhanced autosuggest system (if available)
                if ENHANCED_AUTOSUGGEST_AVAILABLE:
                    try:
                        self.enhanced_autosuggest = EnhancedAutosuggestSystem()
                        self.enhanced_autosuggest.build_system(self.data)
                        self.enhanced_autosuggest_loaded = True
                        print("✅ Enhanced autosuggest loaded successfully!")
                    except Exception as e:
                        print(f"⚠️ Enhanced autosuggest failed, using traditional: {e}")
                        self.enhanced_autosuggest_loaded = False
                else:
                    print("ℹ️ Enhanced autosuggest not available, using traditional only")
                    self.enhanced_autosuggest_loaded = False
                
            except Exception as e:
                print(f"❌ Failed to load autosuggest: {e}")
                print(f"Error details: {traceback.format_exc()}")
                self.autosuggest_loaded = False
                self.enhanced_autosuggest_loaded = False
        else:
            print("❌ Autosuggest components not available")
            self.autosuggest_loaded = False
            self.enhanced_autosuggest_loaded = False
    
    def load_search_system(self):
        """Load both traditional and persona-aware search systems."""
        print("🚀 Loading search systems...")
        
        if SEARCH_AVAILABLE:
            try:
                # Initialize traditional search system (fallback)
                self.searcher = SimplifiedSearcher()
                basic_loaded = self.searcher.search_ready
                
                # Initialize persona-aware search system (if available)
                if basic_loaded and PERSONA_SEARCH_AVAILABLE:
                    try:
                        self.persona_searcher = PersonaAwareSearcher()
                        self.search_loaded = True
                        print("✅ Persona-aware search system loaded successfully!")
                    except Exception as e:
                        print(f"⚠️ Persona search failed, using basic search: {e}")
                        self.search_loaded = basic_loaded
                elif basic_loaded:
                    print("ℹ️ Using basic search system (persona search not available)")
                    self.search_loaded = True
                else:
                    print("⚠️ Search system not ready (missing dependencies)")
                    self.search_loaded = False
                    
            except Exception as e:
                print(f"❌ Failed to load search system: {e}")
                print(f"Error details: {traceback.format_exc()}")
                self.search_loaded = False
        else:
            print("❌ Search components not available")
            self.search_loaded = False
    
    def setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Serve the frontend with original design."""
            return render_template('index.html')
        
        @self.app.route('/api/health')
        def health():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'components': {
                    'autosuggest': 'loaded' if self.autosuggest_loaded else 'failed',
                    'enhanced_autosuggest': 'loaded' if self.enhanced_autosuggest_loaded else 'failed',
                    'search': 'loaded' if self.search_loaded else 'failed'
                },
                'imports': {
                    'autosuggest_available': AUTOSUGGEST_AVAILABLE,
                    'search_available': SEARCH_AVAILABLE
                },
                'timestamp': time.time(),
                'version': '2.0.0 - Integrated'
            })
        
        @self.app.route('/api/config')
        def config():
            """Get system configuration."""
            return jsonify({
                'personas': self.personas,
                'locations': self.locations,
                'events': self.events
            })
        
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
                
                # Extract context
                persona_id = data.get('persona', 'tech_enthusiast')
                location = data.get('location', 'Mumbai')
                event_id = data.get('event', 'none')
                max_suggestions = data.get('max_suggestions', 5)
                
                # Get persona context
                persona_context = self.personas.get(persona_id, self.personas['tech_enthusiast'])
                
                # Get event context
                event_context = next((e for e in self.events if e['id'] == event_id), self.events[0])
                event_name = event_context['name'] if event_context['id'] != 'none' else None
                
                # Prepare context for autosuggest
                context = {
                    "persona": persona_id,
                    "location": location,
                    "event": event_name,
                    "persona_details": persona_context
                }
                
                # Get suggestions (use enhanced if available, fallback to traditional)
                if self.enhanced_autosuggest_loaded and ENHANCED_AUTOSUGGEST_AVAILABLE:
                    suggestions = self.enhanced_autosuggest.get_suggestions(query, context, max_suggestions)
                else:
                    suggestions = self.autosuggest.get_suggestions(
                        query,
                        max_suggestions=max_suggestions,
                        context=context
                    )
                
                # Format suggestions
                formatted_suggestions = []
                for suggestion, score in suggestions:
                    formatted_suggestions.append({
                        'text': suggestion,
                        'score': round(float(score), 4),
                        'confidence': self._get_confidence_level(score)
                    })
                
                end_time = time.time()
                response_time = (end_time - start_time) * 1000
                
                return jsonify({
                    'suggestions': formatted_suggestions,
                    'response_time_ms': round(response_time, 2),
                    'metadata': {
                        'query': query,
                        'query_length': len(query),
                        'persona': persona_context['name'],
                        'location': location,
                        'event': event_context['name'],
                        'total_suggestions': len(suggestions)
                    }
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
            """Perform product search using real search system."""
            start_time = time.time()
            
            try:
                data = request.json
                query = data.get('query', '').strip()
                
                if not query:
                    return jsonify([])
                
                context = data.get('context', {})
                top_k = data.get('top_k', 50)  # Increased to 50 as requested
                
                # Use persona-aware search if available, fallback to traditional
                if self.search_loaded and self.persona_searcher and PERSONA_SEARCH_AVAILABLE:
                    try:
                        # Extract persona details for enhanced context
                        persona_id = context.get('persona', 'tech_enthusiast')
                        enhanced_context = {
                            **context,
                            'persona_details': self.personas.get(persona_id, self.personas.get('tech_enthusiast', {}))
                        }
                        results = self.persona_searcher.search_with_persona(query, enhanced_context, top_k)
                    except Exception as e:
                        print(f"⚠️ Persona search failed, using basic: {e}")
                        results = self.searcher.search(query, top_k, context) if self.searcher else []
                elif self.search_loaded and self.searcher:
                    results = self.searcher.search(query, top_k, context)
                else:
                    # Fallback response if search system not available
                    results = [{
                        'product_id': 'search_unavailable',
                        'title': 'Search system not available',
                        'brand': 'System',
                        'category': 'Service',
                        'price': 0,
                        'rating': 0,
                        'is_f_assured': False,
                        'search_method': 'fallback',
                        'message': 'Install search dependencies: pip install sentence-transformers faiss-cpu'
                    }]
                
                end_time = time.time()
                response_time = (end_time - start_time) * 1000
                
                # Add response time to each result
                for result in results:
                    result['response_time_ms'] = round(response_time, 2)
                
                return jsonify(results)
                
            except Exception as e:
                print(f"ERROR in /api/search: {e}")
                traceback.print_exc()
                return jsonify([{
                    'product_id': 'search_error',
                    'title': 'Search failed',
                    'brand': 'System',
                    'category': 'Error',
                    'price': 0,
                    'rating': 0,
                    'is_f_assured': False,
                    'error': str(e)
                }]), 500
        
        @self.app.route('/api/analytics')
        def analytics():
            """Get system analytics."""
            components = []
            features = []
            
            if self.autosuggest_loaded:
                components.extend([
                    'Simple Trie Matching',
                    'Fuzzy String Matching', 
                    'Frequency-based Ranking',
                    'Brand & Category Keywords'
                ])
                features.extend([
                    'Location-aware suggestions',
                    'Event-based contextual boosting',
                    'Session-aware personalization'
                ])
            
            if self.search_loaded:
                components.extend([
                    'FAISS Semantic Search',
                    'SBERT Embeddings',
                    'Price Constraint Extraction'
                ])
                features.extend([
                    'Semantic product search',
                    'Price filtering',
                    'Multi-modal search'
                ])
            
            return jsonify({
                'data_stats': {
                    'user_queries': len(self.data.get('user_queries', [])) if self.data else 0,
                    'product_catalog': len(self.data.get('product_catalog', [])) if self.data else 0,
                    'locations': len(self.data.get('locations', [])) if self.data else 0,
                    'categories': len(self.data.get('major_categories', [])) if self.data else 0
                },
                'system_info': {
                    'components': components,
                    'features': features
                },
                'integration_info': {
                    'version': '2.0.0 - Integrated',
                    'architecture': 'Single Flask server',
                    'dependencies': 'pandas, sentence-transformers, faiss-cpu',
                    'autosuggest_loaded': self.autosuggest_loaded,
                    'search_loaded': self.search_loaded
                }
            })
        
        @self.app.route('/static/<path:filename>')
        def static_files(filename):
            """Serve static files."""
            return send_from_directory('static', filename)
        
        @self.app.errorhandler(404)
        def not_found(error):
            """Handle 404 errors."""
            return jsonify({
                'error': 'Endpoint not found',
                'available_endpoints': [
                    '/',
                    '/api/health',
                    '/api/config',
                    '/api/suggest',
                    '/api/analytics'
                ]
            }), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            """Handle 500 errors."""
            return jsonify({
                'error': 'Internal server error',
                'message': 'Something went wrong on our end'
            }), 500
    

    
    def _get_confidence_level(self, score: float) -> str:
        """Get confidence level based on score."""
        if score > 0.5:
            return 'high'
        elif score > 0.1:
            return 'medium'
        else:
            return 'low'
    
    def run(self, debug=True, host='127.0.0.1', port=3000):
        """Run the integrated frontend server."""
        print("🚀 Starting Integrated Flipkart Search & Autosuggest Server...")
        print("=" * 70)
        print(f"📍 Application URL: http://{host}:{port}")
        print(f"🎨 Design: Original autosuggest with side panel")
        print(f"🏗️  Architecture: Single Flask server")
        print(f"📦 Dependencies: pandas, flask, sentence-transformers, faiss-cpu")
        print(f"🎯 Debug mode: {debug}")
        print("=" * 70)
        
        # Show component status
        print("📊 Component Status:")
        print(f"   Traditional Autosuggest: {'✅ Loaded' if self.autosuggest_loaded else '❌ Failed'}")
        print(f"   Enhanced Autosuggest: {'✅ Loaded' if self.enhanced_autosuggest_loaded else '❌ Failed'}")
        print(f"   Persona-Aware Search: {'✅ Loaded' if self.search_loaded else '❌ Failed'}")
        
        if not self.autosuggest_loaded:
            print("⚠️  Autosuggest system failed to load!")
        if not self.search_loaded:
            print("⚠️  Search system failed to load!")
            print("💡 Install dependencies: pip install sentence-transformers faiss-cpu")
        
        all_loaded = self.autosuggest_loaded and self.search_loaded
        enhanced_loaded = self.enhanced_autosuggest_loaded
        
        if all_loaded and enhanced_loaded:
            print("🎉 All enhanced systems loaded successfully!")
        elif all_loaded:
            print("🎉 Basic systems ready - Enhanced features may be limited")
        elif self.autosuggest_loaded:
            print("🎉 Autosuggest ready - Search will show fallback message")
        else:
            print("⚠️  Limited functionality - check component status above")
        
        print(f"\n💻 Access the application at: http://{host}:{port}")
        print(f"🔧 Health check: http://{host}:{port}/api/health")
        print("=" * 70)
        
        self.app.run(debug=debug, host=host, port=port, threaded=True)

def main():
    """Main function with command line argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Integrated Flipkart Search & Autosuggest Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py --port 3000
  python app.py --host 0.0.0.0 --port 8080 --no-debug
  
Features:
- Original autosuggest design with side panel
- Real semantic search with FAISS + SBERT
- Grid and list view for search results
- Mobile responsive design
- Price filtering and constraints
        """
    )
    
    parser.add_argument(
        "--host", 
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=3000,
        help="Port to bind to (default: 3000)"
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug mode"
    )
    
    args = parser.parse_args()
    
    server = IntegratedFrontendServer()
    server.run(
        debug=not args.no_debug,
        host=args.host,
        port=args.port
    )

if __name__ == "__main__":
    main()