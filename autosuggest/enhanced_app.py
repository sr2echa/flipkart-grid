#!/usr/bin/env python3
"""
Enhanced Flask App for the Flipkart Autosuggest System
Features advanced autosuggest with multi-task learning and contextual understanding.
"""

from flask import Flask, render_template, request, jsonify
import time
import sys
import os
import json
import logging
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_data_preprocessing import EnhancedDataPreprocessor
from enhanced_integrated_autosuggest import EnhancedIntegratedAutosuggest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for the enhanced autosuggest system
enhanced_autosuggest_system = None
system_ready = False
system_stats = {}

def initialize_enhanced_system():
    """Initialize the enhanced autosuggest system."""
    global enhanced_autosuggest_system, system_ready
    try:
        logger.info("Initializing enhanced autosuggest system...")
        
        # Initialize enhanced preprocessor
        preprocessor = EnhancedDataPreprocessor()
        
        # Check if synthetic data exists, if not generate it
        synthetic_data_paths = [
            'dataset/synthetic_product_catalog.csv',
            'dataset/synthetic_user_queries.csv',
            'dataset/synthetic_realtime_info.csv',
            'dataset/synthetic_session_log.csv'
        ]
        
        # Check if all synthetic data files exist
        missing_files = [path for path in synthetic_data_paths if not os.path.exists(path)]
        
        if missing_files:
            logger.info("Synthetic data files missing. Generating them...")
            from data_generator import EnhancedDataGenerator
            generator = EnhancedDataGenerator(seed=42)
            generator.generate_all_datasets('dataset')
            logger.info("Synthetic data generated successfully!")
        
        # Process data with enhanced preprocessor
        data = preprocessor.process_all_data(
            product_catalog_path='dataset/synthetic_product_catalog.csv',
            user_queries_path='dataset/synthetic_user_queries.csv',
            realtime_info_path='dataset/synthetic_realtime_info.csv',
            session_log_path='dataset/synthetic_session_log.csv'
        )
        
        # Initialize enhanced autosuggest system
        enhanced_autosuggest_system = EnhancedIntegratedAutosuggest(
            enable_multi_task=True,
            enable_feature_store=True,
            cache_dir='cache'
        )
        
        # Build enhanced system
        enhanced_autosuggest_system.build_system(data, use_enhanced_preprocessing=True)
        
        system_ready = True
        logger.info("✅ Enhanced system initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Error initializing enhanced system: {e}")
        system_ready = False
        return False

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/api/enhanced-suggestions', methods=['POST'])
def get_enhanced_suggestions():
    """API endpoint for getting enhanced suggestions."""
    if not system_ready:
        return jsonify({'error': 'Enhanced system not ready'}), 500
    
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        location = data.get('location', '')
        event = data.get('event', '')
        session_context = data.get('session_context', {})
        user_id = data.get('user_id', None)
        max_suggestions = data.get('max_suggestions', 10)
        
        if not query:
            return jsonify({'suggestions': [], 'response_time': 0})
        
        start_time = time.time()
        
        # Prepare context
        context = {
            'location': location,
            'event': event,
            'session_context': session_context
        }
        
        # Get enhanced suggestions
        suggestions = enhanced_autosuggest_system.get_enhanced_suggestions(
            query=query,
            max_suggestions=max_suggestions,
            context=context,
            user_id=user_id
        )
        
        response_time = (time.time() - start_time) * 1000
        
        # Update system stats
        update_system_stats(response_time, len(suggestions))
        
        return jsonify({
            'suggestions': [{'text': suggestion, 'score': score} for suggestion, score in suggestions],
            'response_time': round(response_time, 1),
            'query': query,
            'context': context,
            'system_stats': get_system_stats()
        })
        
    except Exception as e:
        logger.error(f"Error getting enhanced suggestions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system-status')
def system_status():
    """Get system status and statistics."""
    if not system_ready:
        return jsonify({
            'status': 'not_ready',
            'message': 'Enhanced system is not ready'
        })
    
    try:
        # Get performance stats from enhanced system
        performance_stats = enhanced_autosuggest_system.get_performance_stats()
        
        return jsonify({
            'status': 'ready',
            'message': 'Enhanced system is ready',
            'performance_stats': performance_stats,
            'system_stats': get_system_stats(),
            'features': {
                'multi_task_learning': enhanced_autosuggest_system.enable_multi_task,
                'feature_store': enhanced_autosuggest_system.enable_feature_store,
                'contextual_understanding': True,
                'personalization': True
            }
        })
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance-metrics')
def performance_metrics():
    """Get detailed performance metrics."""
    if not system_ready:
        return jsonify({'error': 'System not ready'}), 500
    
    try:
        performance_stats = enhanced_autosuggest_system.get_performance_stats()
        
        return jsonify({
            'performance_stats': performance_stats,
            'system_stats': get_system_stats(),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/test-queries')
def test_queries():
    """Test endpoint with predefined queries."""
    if not system_ready:
        return jsonify({'error': 'System not ready'}), 500
    
    test_queries = [
        'samsung',
        'gaming laptop',
        'nike shoes',
        'wireless headphones',
        'budget mobile',
        'formal shirt',
        'camera for beginners',
        'smart watch'
    ]
    
    results = {}
    
    for query in test_queries:
        try:
            start_time = time.time()
            suggestions = enhanced_autosuggest_system.get_enhanced_suggestions(
                query=query,
                max_suggestions=5
            )
            response_time = (time.time() - start_time) * 1000
            
            results[query] = {
                'suggestions': [{'text': suggestion, 'score': score} for suggestion, score in suggestions],
                'response_time': round(response_time, 1),
                'num_suggestions': len(suggestions)
            }
        except Exception as e:
            results[query] = {'error': str(e)}
    
    return jsonify({
        'test_results': results,
        'timestamp': datetime.now().isoformat()
    })

def update_system_stats(response_time: float, num_suggestions: int):
    """Update system statistics."""
    global system_stats
    
    if 'total_requests' not in system_stats:
        system_stats = {
            'total_requests': 0,
            'total_response_time': 0.0,
            'avg_response_time': 0.0,
            'total_suggestions': 0,
            'avg_suggestions_per_query': 0.0,
            'start_time': datetime.now().isoformat()
        }
    
    system_stats['total_requests'] += 1
    system_stats['total_response_time'] += response_time
    system_stats['total_suggestions'] += num_suggestions
    
    # Update averages
    system_stats['avg_response_time'] = system_stats['total_response_time'] / system_stats['total_requests']
    system_stats['avg_suggestions_per_query'] = system_stats['total_suggestions'] / system_stats['total_requests']

def get_system_stats() -> dict:
    """Get current system statistics."""
    global system_stats
    
    if not system_stats:
        return {
            'total_requests': 0,
            'avg_response_time': 0.0,
            'avg_suggestions_per_query': 0.0,
            'uptime': '0:00:00'
        }
    
    # Calculate uptime
    start_time = datetime.fromisoformat(system_stats['start_time'])
    uptime = datetime.now() - start_time
    uptime_str = str(uptime).split('.')[0]  # Remove microseconds
    
    return {
        'total_requests': system_stats['total_requests'],
        'avg_response_time': round(system_stats['avg_response_time'], 2),
        'avg_suggestions_per_query': round(system_stats['avg_suggestions_per_query'], 2),
        'uptime': uptime_str
    }

@app.route('/api/contextual-test')
def contextual_test():
    """Test contextual suggestions with different scenarios."""
    if not system_ready:
        return jsonify({'error': 'System not ready'}), 500
    
    test_scenarios = [
        {
            'name': 'Mumbai Electronics',
            'query': 'mobile',
            'context': {'location': 'Mumbai', 'event': '', 'session_context': {}}
        },
        {
            'name': 'Delhi Fashion',
            'query': 'shoes',
            'context': {'location': 'Delhi', 'event': '', 'session_context': {}}
        },
        {
            'name': 'Diwali Shopping',
            'query': 'gift',
            'context': {'location': '', 'event': 'Diwali', 'session_context': {}}
        },
        {
            'name': 'IPL Season',
            'query': 'jersey',
            'context': {'location': '', 'event': 'IPL', 'session_context': {}}
        },
        {
            'name': 'Wedding Season',
            'query': 'formal',
            'context': {'location': '', 'event': 'Wedding', 'session_context': {}}
        }
    ]
    
    results = {}
    
    for scenario in test_scenarios:
        try:
            start_time = time.time()
            suggestions = enhanced_autosuggest_system.get_enhanced_suggestions(
                query=scenario['query'],
                max_suggestions=5,
                context=scenario['context']
            )
            response_time = (time.time() - start_time) * 1000
            
            results[scenario['name']] = {
                'query': scenario['query'],
                'context': scenario['context'],
                'suggestions': [{'text': suggestion, 'score': score} for suggestion, score in suggestions],
                'response_time': round(response_time, 1)
            }
        except Exception as e:
            results[scenario['name']] = {'error': str(e)}
    
    return jsonify({
        'contextual_test_results': results,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/save-models')
def save_models():
    """Save all models to disk."""
    if not system_ready:
        return jsonify({'error': 'System not ready'}), 500
    
    try:
        model_path = 'saved_models'
        enhanced_autosuggest_system.save_models(model_path)
        
        return jsonify({
            'message': 'Models saved successfully',
            'model_path': model_path,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error saving models: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/load-models')
def load_models():
    """Load models from disk."""
    if not system_ready:
        return jsonify({'error': 'System not ready'}), 500
    
    try:
        model_path = 'saved_models'
        enhanced_autosuggest_system.load_models(model_path)
        
        return jsonify({
            'message': 'Models loaded successfully',
            'model_path': model_path,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize the enhanced system
    if initialize_enhanced_system():
        logger.info("🚀 Enhanced Flask app starting...")
        logger.info("📊 System features:")
        logger.info("   - Multi-task learning: Enabled")
        logger.info("   - Feature store: Enabled")
        logger.info("   - Contextual understanding: Enabled")
        logger.info("   - Personalization: Enabled")
        logger.info("🌐 Web interface available at: http://localhost:5000")
        
        # Run the Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("❌ Failed to initialize enhanced system. Exiting.")
        sys.exit(1) 