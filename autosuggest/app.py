"""
Simple, High-Quality Autosuggest Flask App
Focus: Clean UI, Persistent suggestions, Quality results
"""
import sys
import os
import time
import traceback
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import autosuggest system
from data_preprocessing import DataPreprocessor
from enhanced_autosuggest_v5 import EnhancedAutosuggestV5

class AutosuggestApp:
    """Clean Flask app for autosuggest with persistent suggestions."""
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # System state
        self.autosuggest = None
        self.is_loaded = False
        self.data = None
        
        # Configuration
        self.personas = {
            'tech_enthusiast': {
                'name': 'Tech Enthusiast',
                'description': 'Loves gadgets, laptops, phones, gaming',
                'keywords': ['laptop', 'mobile', 'gaming', 'tech', 'gadget']
            },
            'sports_enthusiast': {
                'name': 'Sports Enthusiast', 
                'description': 'Into cricket, football, fitness, sports gear',
                'keywords': ['cricket', 'football', 'jersey', 'sports', 'fitness']
            },
            'fashion_lover': {
                'name': 'Fashion Lover',
                'description': 'Loves clothing, accessories, style',
                'keywords': ['dress', 'jeans', 'shirt', 'fashion', 'style']
            },
            'home_maker': {
                'name': 'Home Maker',
                'description': 'Interested in home decor, kitchen, furniture',
                'keywords': ['home', 'kitchen', 'furniture', 'decor', 'appliance']
            },
            'budget_conscious': {
                'name': 'Budget Conscious',
                'description': 'Looks for deals, discounts, value products',
                'keywords': ['under', 'cheap', 'discount', 'offer', 'budget']
            }
        }
        
        self.events = [
            {'id': 'none', 'name': 'None'},
            {'id': 'diwali', 'name': 'Diwali'},
            {'id': 'ipl', 'name': 'IPL'},
            {'id': 'summer', 'name': 'Summer'},
            {'id': 'winter', 'name': 'Winter'},
            {'id': 'wedding', 'name': 'Wedding Season'}
        ]
        
        self.locations = [
            'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Pune', 
            'Kolkata', 'Hyderabad', 'Ahmedabad', 'Jaipur', 'Lucknow'
        ]
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            """Serve the main UI."""
            return render_template('index_simple.html')
        
        @self.app.route('/api/config')
        def get_config():
            """Get configuration for UI."""
            return jsonify({
                'personas': self.personas,
                'events': self.events,
                'locations': self.locations
            })
        
        @self.app.route('/api/health')
        def health():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'loaded': self.is_loaded,
                'timestamp': time.time()
            })
        
        @self.app.route('/api/suggest', methods=['POST'])
        def suggest():
            """Get autosuggest suggestions."""
            start_time = time.time()
            
            # Load system if not loaded
            if not self.is_loaded:
                self.load_system()
            
            try:
                data = request.json
                query = data.get('query', '').strip()
                persona_id = data.get('persona', 'tech_enthusiast')
                location = data.get('location', 'Mumbai')
                event_id = data.get('event', 'none')
                max_suggestions = data.get('max_suggestions', 5)
                
                # Return empty for empty query
                if not query:
                    return jsonify({
                        'suggestions': [],
                        'response_time_ms': 0,
                        'metadata': {
                            'query_length': 0,
                            'context': {
                                'persona': persona_id,
                                'location': location,
                                'event': event_id
                            }
                        }
                    })
                
                # Prepare context
                persona_context = self.personas.get(persona_id, self.personas['tech_enthusiast'])
                event_context = next((e for e in self.events if e['id'] == event_id), self.events[0])
                event_name = event_context['name'] if event_context['id'] != 'none' else None
                
                context = {
                    "persona": persona_id,
                    "location": location,
                    "event": event_name,
                    "persona_details": persona_context
                }
                
                # Get suggestions using the Enhanced V5 system
                suggestions = self.autosuggest.get_intelligent_suggestions(query, context, max_suggestions)
                
                # Format response
                formatted_suggestions = []
                for suggestion, score in suggestions:
                    formatted_suggestions.append({
                        'text': suggestion,
                        'score': round(score, 3)
                    })
                
                response_time = (time.time() - start_time) * 1000
                
                return jsonify({
                    'suggestions': formatted_suggestions,
                    'response_time_ms': round(response_time, 2),
                    'metadata': {
                        'query_length': len(query),
                        'total_suggestions': len(formatted_suggestions),
                        'context': {
                            'persona': persona_context['name'],
                            'location': location,
                            'event': event_name or 'None'
                        }
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

    def load_system(self):
        """Load the autosuggest system."""
        if self.is_loaded:
            return
            
        print("🚀 Loading autosuggest system...")
        
        try:
            # Load data
            preprocessor = DataPreprocessor()
            preprocessor.run_all_preprocessing()
            self.data = preprocessor.get_processed_data()
            
            # Initialize Enhanced V5 autosuggest system
            self.autosuggest = EnhancedAutosuggestV5()
            self.autosuggest.build_system(self.data)
            
            self.is_loaded = True
            print("✅ Autosuggest system loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading system: {e}")
            traceback.print_exc()
            raise
    
    def run(self, debug=True, host='0.0.0.0', port=5000):
        """Run the Flask app."""
        print(f"🚀 Starting Autosuggest App on http://{host}:{port}")
        
        # Preload system for faster first request
        self.load_system()
        
        self.app.run(debug=debug, host=host, port=port, threaded=True)

if __name__ == "__main__":
    app = AutosuggestApp()
    app.run(debug=True, host='0.0.0.0', port=5000)