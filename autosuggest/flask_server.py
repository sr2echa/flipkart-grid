#!/usr/bin/env python3
"""
Flask server for the Flipkart Autosuggest System.
Provides modern API endpoints with real-time suggestions.
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import time
import json
import os
import sys
from typing import Dict, List, Tuple

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import autosuggest components
from data_preprocessing import DataPreprocessor
from integrated_autosuggest_v3 import IntegratedAutosuggestV3

class FlaskAutosuggestServer:
    """Flask server for autosuggest system with modern API design."""
    
    def __init__(self):
        self.app = Flask(__name__, 
                        template_folder='templates',
                        static_folder='static')
        CORS(self.app)
        
        # System components
        self.autosuggest = None
        self.data = None
        self.is_loaded = False
        
        # User personas
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
    
    def load_system(self):
        """Load the autosuggest system."""
        if not self.is_loaded:
            print("🚀 Loading autosuggest system...")
            
            # Load and preprocess data
            preprocessor = DataPreprocessor()
            preprocessor.run_all_preprocessing()
            self.data = preprocessor.get_processed_data()
            
            # Initialize autosuggest system
            self.autosuggest = IntegratedAutosuggestV3()
            self.autosuggest.build_system(self.data)
            
            self.is_loaded = True
            print("✅ Autosuggest system loaded successfully!")
    
    def setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Serve the main UI."""
            return render_template('index_minimal.html')
        
        @self.app.route('/api/health')
        def health():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'loaded': self.is_loaded,
                'timestamp': time.time()
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
            
            # Ensure system is loaded
            if not self.is_loaded:
                self.load_system()
            
            data = request.json
            query = data.get('query', '').strip()
            
            if not query:
                return jsonify({
                    'suggestions': [],
                    'response_time_ms': 0,
                    'metadata': {'query_length': 0}
                })
            
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
            
            # Consolidate context into a single dictionary
            context = {
                "persona": persona_id,
                "location": location,
                "event": event_name,
                "persona_details": persona_context
            }
            
            try:
                # Get contextual suggestions
                suggestions = self.autosuggest.get_contextual_suggestions(
                    query,
                    context=context
                )
                
                # Format suggestions
                formatted_suggestions = []
                for suggestion, score in suggestions[:max_suggestions]:
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
                import traceback
                print(f"ERROR in /api/suggest: {e}")
                traceback.print_exc()
                return jsonify({
                    'error': str(e),
                    'suggestions': [],
                    'response_time_ms': (time.time() - start_time) * 1000
                }), 500
        
        @self.app.route('/api/analytics')
        def analytics():
            """Get system analytics."""
            if not self.is_loaded:
                return jsonify({'error': 'System not loaded'}), 503
            
            return jsonify({
                'data_stats': {
                    'user_queries': len(self.data.get('user_queries', [])),
                    'product_catalog': len(self.data.get('product_catalog', [])),
                    'session_log': len(self.data.get('session_log', [])),
                    'locations': len(self.data.get('locations', [])),
                    'categories': len(self.data.get('major_categories', []))
                },
                'system_info': {
                    'components': [
                        'Trie Matching',
                        'Semantic Correction (SBERT + FAISS)',
                        'BERT Completion',
                        'XGBoost Reranking'
                    ],
                    'features': [
                        'Location-aware suggestions',
                        'Event-based contextual boosting', 
                        'Session-aware personalization',
                        'Real-time performance optimization'
                    ]
                }
            })
        
        @self.app.route('/static/<path:filename>')
        def static_files(filename):
            """Serve static files."""
            return send_from_directory('static', filename)
    
    def _get_confidence_level(self, score: float) -> str:
        """Get confidence level based on score."""
        if score > 0.5:
            return 'high'
        elif score > 0.1:
            return 'medium'
        else:
            return 'low'
    
    def run(self, debug=True, host='127.0.0.1', port=5000):
        """Run the Flask server."""
        print(f"🌐 Starting Flask autosuggest server...")
        print(f"📍 Server will be available at: http://{host}:{port}")
        print(f"🎯 Debug mode: {debug}")
        
        # Load system on startup
        self.load_system()
        
        self.app.run(debug=debug, host=host, port=port, threaded=True)

if __name__ == "__main__":
    server = FlaskAutosuggestServer()
    server.run(debug=True, host='0.0.0.0', port=5000)