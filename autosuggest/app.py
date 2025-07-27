#!/usr/bin/env python3
"""
Flask app for the Flipkart Autosuggest System - Unified Interface
"""

from flask import Flask, render_template, request, jsonify
import time
import sys
import os
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import DataPreprocessor
from integrated_autosuggest import IntegratedAutosuggest

app = Flask(__name__)

# Global variables for the autosuggest system
autosuggest_system = None
system_ready = False

def initialize_system():
    """Initialize the autosuggest system."""
    global autosuggest_system, system_ready
    try:
        print("Initializing autosuggest system...")
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        preprocessor.run_all_preprocessing()
        data = preprocessor.get_processed_data()
        
        # Initialize autosuggest system
        autosuggest_system = IntegratedAutosuggest()
        autosuggest_system.build_system(data)
        
        system_ready = True
        print("✅ System initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        system_ready = False
        return False

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/api/suggestions', methods=['POST'])
def get_suggestions():
    """API endpoint for getting suggestions."""
    if not system_ready:
        return jsonify({'error': 'System not ready'}), 500
    
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        location = data.get('location', '')
        event = data.get('event', '')
        session_context = data.get('session_context', {})
        
        if not query:
            return jsonify({'suggestions': [], 'response_time': 0})
        
        start_time = time.time()
        
        # Unified suggestion logic - combines all approaches
        suggestions = get_unified_suggestions(query, location, event, session_context)
        
        response_time = (time.time() - start_time) * 1000
        
        return jsonify({
            'suggestions': suggestions,
            'response_time': round(response_time, 1),
            'query': query,
            'context': {
                'location': location,
                'event': event,
                'session_context': session_context
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_unified_suggestions(query, location, event, session_context):
    """Get unified suggestions combining all approaches."""
    all_suggestions = {}
    
    # 1. Get suggestions from integrated system
    context = {
        'location': location,
        'event': event,
        'session_context': session_context
    }
    
    integrated_suggestions = autosuggest_system.get_suggestions(query, max_suggestions=20, context=context)
    for suggestion, score in integrated_suggestions:
        if suggestion and len(suggestion.strip()) > 0:
            all_suggestions[suggestion] = score
    
    # 2. Additional typo correction for short queries
    if len(query) <= 8:
        typo_corrections = autosuggest_system.semantic_correction.get_semantic_suggestions(query, top_k=5)
        for correction, similarity in typo_corrections:
            if correction and len(correction.strip()) > 0:
                # Boost typo corrections slightly
                boosted_score = min(similarity * 1.1, 1.0)
                if correction not in all_suggestions or boosted_score > all_suggestions[correction]:
                    all_suggestions[correction] = boosted_score
    
    # 3. Contextual completions for very short queries
    if len(query) <= 4:
        bert_completions = autosuggest_system.bert_completion.complete_query(query, max_suggestions=5)
        for completion in bert_completions:
            if completion and len(completion.strip()) > 0:
                if completion not in all_suggestions:
                    all_suggestions[completion] = 0.6  # Moderate score for completions
    
    # 4. Apply final filtering and ranking
    final_suggestions = []
    for suggestion, score in all_suggestions.items():
        # Filter out poor suggestions
        if _is_poor_suggestion(suggestion):
            continue
        
        # Ensure score is within 0-1 range
        final_score = max(0.0, min(1.0, score))
        final_suggestions.append((suggestion, final_score))
    
    # Sort by score and return top suggestions
    final_suggestions.sort(key=lambda x: x[1], reverse=True)
    return final_suggestions[:10]

def _is_poor_suggestion(suggestion):
    """Check if a suggestion is poor quality and should be filtered out."""
    if not suggestion or len(suggestion.strip()) == 0:
        return True
    
    suggestion_lower = suggestion.lower().strip()
    
    # Filter out suggestions with punctuation or special characters
    if any(char in suggestion_lower for char in [';', '.', ',', '!', '?', ':', '"', "'", '##']):
        return True
    
    # Filter out very short suggestions (less than 2 characters)
    if len(suggestion_lower) < 2:
        return True
    
    # Filter out suggestions that are just single letters or numbers
    if len(suggestion_lower) == 1 and suggestion_lower.isalnum():
        return True
    
    # Filter out suggestions that are just common words without context
    common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
    if suggestion_lower in common_words:
        return True
    
    return False

def is_likely_typo(query):
    """Check if a query is likely a typo."""
    # Common typo patterns
    typo_patterns = [
        'samsng', 'nkie', 'addidas', 'xiomi', 'laptap', 'headphons',
        'camra', 'mobil', 'sho', 'phon', 'soney', 'onepls', 'vvo',
        'opo', 'realmi', 'del', 'lenvo', 'asuss', 'bot', 'jb', 'pma',
        'rebbok', 'zarra', 'ikia', 'prestig', 'h p', 'l g', 'bta', 'h m'
    ]
    return query.lower() in typo_patterns

@app.route('/api/system-status')
def system_status():
    """Check system status."""
    return jsonify({
        'ready': system_ready,
        'components': {
            'trie': True,
            'semantic': True,
            'bert': True,
            'reranker': True
        }
    })

if __name__ == '__main__':
    # Initialize system before starting Flask
    if initialize_system():
        print("🚀 Starting Flask app...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize system. Exiting.")
        sys.exit(1) 