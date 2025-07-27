#!/usr/bin/env python3
"""
Enhanced Integrated Autosuggest System for Flipkart
Combines all enhanced components with advanced features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import logging
import pickle
import os
from datetime import datetime, timedelta
import json

# Import enhanced components
from enhanced_trie_autosuggest import EnhancedTrieAutosuggest
from enhanced_semantic_correction import EnhancedSemanticCorrection
from enhanced_bert_completion import EnhancedBERTCompletion
from enhanced_reranker import EnhancedReranker
from enhanced_data_preprocessing import EnhancedDataPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedIntegratedAutosuggest:
    """
    Enhanced integrated autosuggest system combining all advanced components.
    Features multi-task learning, feature store, and improved contextual understanding.
    """
    
    def __init__(self, 
                 enable_multi_task: bool = True,
                 enable_feature_store: bool = True,
                 cache_dir: str = 'cache'):
        """
        Initialize the enhanced integrated autosuggest system.
        
        Args:
            enable_multi_task: Enable multi-task learning for relevance and click prediction
            enable_feature_store: Enable feature store for caching
            cache_dir: Directory for caching models and features
        """
        self.enable_multi_task = enable_multi_task
        self.enable_feature_store = enable_feature_store
        self.cache_dir = cache_dir
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize enhanced components
        self.trie_autosuggest = EnhancedTrieAutosuggest()
        self.semantic_correction = EnhancedSemanticCorrection()
        self.bert_completion = EnhancedBERTCompletion()
        self.reranker = EnhancedReranker(enable_multi_task=enable_multi_task, cache_dir=cache_dir)
        
        # Data storage
        self.user_queries = None
        self.session_log = None
        self.product_catalog = None
        self.realtime_product_info = None
        
        # Feature store
        self.feature_store = {}
        self.session_contexts = {}
        
        # Performance tracking
        self.performance_metrics = {
            'total_queries': 0,
            'avg_response_time': 0.0,
            'cache_hit_rate': 0.0,
            'suggestion_quality_scores': []
        }
        
    def build_system(self, data: Dict, use_enhanced_preprocessing: bool = True):
        """
        Build the complete enhanced autosuggest system.
        
        Args:
            data: Dictionary containing all datasets
            use_enhanced_preprocessing: Whether to use enhanced preprocessing
        """
        logger.info("Building enhanced integrated autosuggest system...")
        start_time = time.time()
        
        # Store data
        self.user_queries = data['user_queries']
        self.session_log = data['session_log']
        self.product_catalog = data['product_catalog']
        self.realtime_product_info = data['realtime_product_info']
        
        # Build enhanced Trie component
        logger.info("Building Enhanced Trie component...")
        self.trie_autosuggest.build_trie(data['user_queries'])
        
        # Build enhanced Semantic Correction component
        logger.info("Building Enhanced Semantic Correction component...")
        self.semantic_correction.build_semantic_index(data['user_queries'])
        
        # Build enhanced BERT Completion component
        logger.info("Building Enhanced BERT Completion component...")
        self.bert_completion.build_completion_patterns(data['user_queries'])
        
        # Build enhanced Reranker
        logger.info("Building Enhanced XGBoost Reranker...")
        self._build_enhanced_reranker(data)
        
        # Build session contexts
        logger.info("Building session contexts...")
        self._build_session_contexts()
        
        # Initialize feature store
        if self.enable_feature_store:
            logger.info("Initializing feature store...")
            self._initialize_feature_store()
        
        build_time = time.time() - start_time
        logger.info(f"✅ Enhanced autosuggest system built successfully in {build_time:.2f}s!")
        
    def _build_enhanced_reranker(self, data: Dict):
        """Build enhanced reranker with multi-task learning."""
        # Prepare training data
        training_data = self._prepare_enhanced_training_data(data)
        
        # Train models
        if self.enable_multi_task:
            self.reranker.train_models(
                training_data=training_data['features'],
                relevance_labels=training_data['relevance_labels'],
                click_labels=training_data['click_labels']
            )
        else:
            self.reranker.train_models(
                training_data=training_data['features'],
                relevance_labels=training_data['relevance_labels']
            )
        
        logger.info("Enhanced reranker trained successfully!")
    
    def _prepare_enhanced_training_data(self, data: Dict) -> Dict:
        """Prepare enhanced training data with interaction features."""
        training_features = []
        relevance_labels = []
        click_labels = []
        
        # Generate training examples from user queries and session data
        for _, query_row in data['user_queries'].iterrows():
            query = query_row['original_query']
            
            # Get suggestions from all components
            trie_suggestions = self.trie_autosuggest.get_suggestions(query, max_suggestions=10)
            semantic_suggestions = self.semantic_correction.get_semantic_suggestions(query, top_k=10)
            bert_suggestions = self.bert_completion.get_completions(query, max_suggestions=10)
            
            # Combine suggestions
            all_suggestions = []
            for suggestion, score in trie_suggestions:
                all_suggestions.append({'suggestion': suggestion, 'score': score, 'source': 'trie'})
            for suggestion, score in semantic_suggestions:
                all_suggestions.append({'suggestion': suggestion, 'score': score, 'source': 'semantic'})
            for suggestion, score in bert_suggestions:
                all_suggestions.append({'suggestion': suggestion, 'score': score, 'source': 'bert'})
            
            # Create training examples
            for suggestion_data in all_suggestions:
                features = self._extract_enhanced_features(query, suggestion_data['suggestion'], 
                                                         suggestion_data['score'], data)
                training_features.append(features)
                
                # Generate labels (simplified for demo)
                relevance_score = suggestion_data['score']
                relevance_labels.append(relevance_score)
                
                # Click label based on frequency and predicted purchase
                click_prob = 0.5
                if query_row['predicted_purchase'] == 'yes':
                    click_prob = 0.8
                elif query_row['predicted_purchase'] == 'maybe':
                    click_prob = 0.6
                click_labels.append(1 if np.random.random() < click_prob else 0)
        
        return {
            'features': training_features,
            'relevance_labels': relevance_labels,
            'click_labels': click_labels
        }
    
    def _extract_enhanced_features(self, query: str, suggestion: str, initial_score: float, 
                                 data: Dict) -> Dict[str, float]:
        """Extract enhanced features for training."""
        features = {}
        
        # Basic features
        features['query_length'] = len(query)
        features['suggestion_length'] = len(suggestion)
        features['initial_score'] = initial_score
        features['exact_match'] = float(query.lower() == suggestion.lower())
        features['query_in_suggestion'] = float(query.lower() in suggestion.lower())
        features['suggestion_in_query'] = float(suggestion.lower() in query.lower())
        
        # Semantic features
        features['word_overlap'] = len(set(query.lower().split()) & set(suggestion.lower().split()))
        features['word_overlap_ratio'] = features['word_overlap'] / max(len(query.split()), 1)
        
        # Contextual features
        features['hour_of_day'] = datetime.now().hour
        features['day_of_week'] = datetime.now().weekday()
        features['is_weekend'] = float(datetime.now().weekday() >= 5)
        
        # Product-specific features
        product_features = self._extract_product_features(query, suggestion, data)
        features.update(product_features)
        
        # Session features
        session_features = self._extract_session_features(query, suggestion, data)
        features.update(session_features)
        
        return features
    
    def _extract_product_features(self, query: str, suggestion: str, data: Dict) -> Dict[str, float]:
        """Extract product-specific features."""
        features = {}
        
        # Check if suggestion matches any product
        matching_products = self.product_catalog[
            self.product_catalog['title'].str.contains(suggestion, case=False, na=False) |
            self.product_catalog['description'].str.contains(suggestion, case=False, na=False)
        ]
        
        if len(matching_products) > 0:
            features['product_match_count'] = len(matching_products)
            features['avg_product_rating'] = matching_products['rating'].mean()
            features['avg_product_price'] = matching_products['price'].mean()
            features['product_availability'] = 1.0
        else:
            features['product_match_count'] = 0
            features['avg_product_rating'] = 0.0
            features['avg_product_price'] = 0.0
            features['product_availability'] = 0.0
        
        return features
    
    def _extract_session_features(self, query: str, suggestion: str, data: Dict) -> Dict[str, float]:
        """Extract session-based features."""
        features = {}
        
        # Get recent session data
        recent_sessions = self.session_log[
            self.session_log['event_type'] == 'search'
        ].tail(100)  # Last 100 searches
        
        # Check if suggestion appears in recent searches
        suggestion_in_recent = recent_sessions[
            recent_sessions['event_data'].str.contains(suggestion, case=False, na=False)
        ]
        
        features['recent_suggestion_frequency'] = len(suggestion_in_recent)
        features['suggestion_popularity'] = len(suggestion_in_recent) / max(len(recent_sessions), 1)
        
        return features
    
    def _build_session_contexts(self):
        """Build session contexts for personalization."""
        # Group by session_id
        session_groups = self.session_log.groupby('session_id')
        
        for session_id, session_data in session_groups:
            context = {
                'queries': session_data[session_data['event_type'] == 'search']['event_data'].tolist(),
                'viewed_products': session_data[session_data['event_type'] == 'view']['event_data'].tolist(),
                'clicked_products': session_data[session_data['event_type'] == 'click']['event_data'].tolist(),
                'session_start': session_data['session_start'].iloc[0],
                'session_end': session_data['session_end'].iloc[0]
            }
            self.session_contexts[session_id] = context
    
    def _initialize_feature_store(self):
        """Initialize feature store with precomputed features."""
        # Precompute common features
        for _, product in self.product_catalog.iterrows():
            key = f"product_{product['product_id']}"
            self.feature_store[key] = {
                'rating': product['rating'],
                'price': product['price'],
                'review_count': product['num_reviews']
            }
        
        # Precompute brand features
        brand_stats = self.product_catalog.groupby('brand').agg({
            'rating': 'mean',
            'price': 'mean',
            'num_reviews': 'sum'
        }).to_dict('index')
        
        for brand, stats in brand_stats.items():
            key = f"brand_{brand}"
            self.feature_store[key] = stats
    
    def get_enhanced_suggestions(self, query: str, max_suggestions: int = 10, 
                               context: Dict = None, user_id: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Get enhanced suggestions combining all components with advanced features.
        
        Args:
            query: User query
            max_suggestions: Maximum number of suggestions
            context: Contextual information (location, event, session)
            user_id: User ID for personalization
            
        Returns:
            List of (suggestion, score) tuples
        """
        start_time = time.time()
        
        # Get suggestions from all components
        trie_suggestions = self.trie_autosuggest.get_suggestions(query, max_suggestions=max_suggestions)
        semantic_suggestions = self.semantic_correction.get_semantic_suggestions(query, top_k=max_suggestions)
        bert_suggestions = self.bert_completion.get_completions(query, max_suggestions=max_suggestions)
        
        # Combine and deduplicate suggestions
        all_suggestions = {}
        
        # Add Trie suggestions
        for suggestion, score in trie_suggestions:
            if suggestion not in all_suggestions:
                all_suggestions[suggestion] = {'score': score, 'sources': ['trie']}
            else:
                all_suggestions[suggestion]['score'] = max(all_suggestions[suggestion]['score'], score)
                all_suggestions[suggestion]['sources'].append('trie')
        
        # Add Semantic suggestions
        for suggestion, score in semantic_suggestions:
            if suggestion not in all_suggestions:
                all_suggestions[suggestion] = {'score': score, 'sources': ['semantic']}
            else:
                all_suggestions[suggestion]['score'] = max(all_suggestions[suggestion]['score'], score)
                all_suggestions[suggestion]['sources'].append('semantic')
        
        # Add BERT suggestions
        for suggestion, score in bert_suggestions:
            if suggestion not in all_suggestions:
                all_suggestions[suggestion] = {'score': score, 'sources': ['bert']}
            else:
                all_suggestions[suggestion]['score'] = max(all_suggestions[suggestion]['score'], score)
                all_suggestions[suggestion]['sources'].append('bert')
        
        # Apply contextual boosting
        if context:
            all_suggestions = self._apply_contextual_boosting(all_suggestions, context)
        
        # Apply personalization
        if user_id:
            all_suggestions = self._apply_personalization(all_suggestions, user_id)
        
        # Convert to list format for reranker
        suggestion_list = []
        for suggestion, data in all_suggestions.items():
            suggestion_list.append({
                'suggestion': suggestion,
                'score': data['score'],
                'sources': data['sources']
            })
        
        # Apply enhanced reranking
        if self.enable_multi_task:
            ranked_suggestions = self.reranker.predict_multi_objective(query, suggestion_list, user_id)
        else:
            ranked_suggestions = self.reranker.predict_relevance(query, suggestion_list, user_id)
        
        # Format results
        results = []
        for suggestion_data in ranked_suggestions[:max_suggestions]:
            results.append((suggestion_data['suggestion'], suggestion_data['final_score']))
        
        # Update performance metrics
        response_time = time.time() - start_time
        self._update_performance_metrics(response_time)
        
        return results
    
    def _apply_contextual_boosting(self, suggestions: Dict, context: Dict) -> Dict:
        """Apply contextual boosting to suggestions."""
        location = context.get('location', '')
        event = context.get('event', '')
        session_context = context.get('session_context', {})
        
        for suggestion, data in suggestions.items():
            boost = 1.0
            
            # Location-based boosting
            if location:
                location_boost = self._get_location_boost(suggestion, location)
                boost *= location_boost
            
            # Event-based boosting
            if event:
                event_boost = self._get_event_boost(suggestion, event)
                boost *= event_boost
            
            # Session-based boosting
            if session_context:
                session_boost = self._get_session_boost(suggestion, session_context)
                boost *= session_boost
            
            data['score'] *= boost
        
        return suggestions
    
    def _get_location_boost(self, suggestion: str, location: str) -> float:
        """Get location-based boost for suggestion."""
        # Mumbai-specific boosts
        if location.lower() == 'mumbai':
            if 'mobile' in suggestion.lower() or 'phone' in suggestion.lower():
                return 1.2
            if 'laptop' in suggestion.lower():
                return 1.1
        
        # Delhi-specific boosts
        elif location.lower() == 'delhi':
            if 'fashion' in suggestion.lower() or 'clothing' in suggestion.lower():
                return 1.2
            if 'electronics' in suggestion.lower():
                return 1.1
        
        return 1.0
    
    def _get_event_boost(self, suggestion: str, event: str) -> float:
        """Get event-based boost for suggestion."""
        # Diwali boosts
        if event.lower() == 'diwali':
            if 'gift' in suggestion.lower() or 'electronics' in suggestion.lower():
                return 1.3
            if 'clothing' in suggestion.lower() or 'fashion' in suggestion.lower():
                return 1.2
        
        # IPL boosts
        elif event.lower() == 'ipl':
            if 'jersey' in suggestion.lower() or 'sports' in suggestion.lower():
                return 1.4
            if 'electronics' in suggestion.lower():
                return 1.1
        
        # Wedding boosts
        elif event.lower() == 'wedding':
            if 'formal' in suggestion.lower() or 'dress' in suggestion.lower():
                return 1.3
            if 'jewelry' in suggestion.lower():
                return 1.2
        
        return 1.0
    
    def _get_session_boost(self, suggestion: str, session_context: Dict) -> float:
        """Get session-based boost for suggestion."""
        boost = 1.0
        
        # Check if suggestion is related to previous queries
        previous_queries = session_context.get('queries', [])
        for prev_query in previous_queries[-3:]:  # Last 3 queries
            if any(word in suggestion.lower() for word in prev_query.lower().split()):
                boost *= 1.1
        
        # Check if suggestion is related to viewed products
        viewed_products = session_context.get('viewed_products', [])
        for product_id in viewed_products[-5:]:  # Last 5 viewed products
            if product_id in self.product_catalog['product_id'].values:
                product = self.product_catalog[self.product_catalog['product_id'] == int(product_id)].iloc[0]
                if any(word in suggestion.lower() for word in product['title'].lower().split()):
                    boost *= 1.05
        
        return boost
    
    def _apply_personalization(self, suggestions: Dict, user_id: str) -> Dict:
        """Apply personalization based on user preferences."""
        # Get user preferences from feature store
        user_key = f"user_{user_id}_preference"
        user_pref = self.feature_store.get(user_key, 0.5)
        
        # Apply user preference boost
        for suggestion, data in suggestions.items():
            data['score'] *= (1 + user_pref * 0.2)
        
        return suggestions
    
    def _update_performance_metrics(self, response_time: float):
        """Update performance metrics."""
        self.performance_metrics['total_queries'] += 1
        
        # Update average response time
        current_avg = self.performance_metrics['avg_response_time']
        total_queries = self.performance_metrics['total_queries']
        self.performance_metrics['avg_response_time'] = (
            (current_avg * (total_queries - 1) + response_time) / total_queries
        )
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        return {
            'total_queries': self.performance_metrics['total_queries'],
            'avg_response_time_ms': self.performance_metrics['avg_response_time'] * 1000,
            'cache_hit_rate': self.performance_metrics['cache_hit_rate'],
            'reranker_stats': self.reranker.get_statistics()
        }
    
    def save_models(self, model_path: str):
        """Save all models to disk."""
        os.makedirs(model_path, exist_ok=True)
        
        # Save reranker models
        self.reranker.save_models(model_path)
        
        # Save feature store
        feature_store_path = os.path.join(model_path, 'feature_store.pkl')
        with open(feature_store_path, 'wb') as f:
            pickle.dump(self.feature_store, f)
        
        # Save session contexts
        session_contexts_path = os.path.join(model_path, 'session_contexts.pkl')
        with open(session_contexts_path, 'wb') as f:
            pickle.dump(self.session_contexts, f)
        
        logger.info(f"Models saved to {model_path}")
    
    def load_models(self, model_path: str):
        """Load models from disk."""
        # Load reranker models
        self.reranker.load_models(model_path)
        
        # Load feature store
        feature_store_path = os.path.join(model_path, 'feature_store.pkl')
        if os.path.exists(feature_store_path):
            with open(feature_store_path, 'rb') as f:
                self.feature_store = pickle.load(f)
        
        # Load session contexts
        session_contexts_path = os.path.join(model_path, 'session_contexts.pkl')
        if os.path.exists(session_contexts_path):
            with open(session_contexts_path, 'rb') as f:
                self.session_contexts = pickle.load(f)
        
        logger.info(f"Models loaded from {model_path}")

def main():
    """Main function for testing the enhanced system."""
    # Initialize enhanced preprocessor
    preprocessor = EnhancedDataPreprocessor()
    
    # Load and process data
    data = preprocessor.process_all_data(
        product_catalog_path='dataset/synthetic_product_catalog.csv',
        user_queries_path='dataset/synthetic_user_queries.csv',
        realtime_info_path='dataset/synthetic_realtime_info.csv',
        session_log_path='dataset/synthetic_session_log.csv'
    )
    
    # Initialize enhanced autosuggest system
    autosuggest = EnhancedIntegratedAutosuggest(
        enable_multi_task=True,
        enable_feature_store=True
    )
    
    # Build system
    autosuggest.build_system(data)
    
    # Test queries
    test_queries = ['samsung', 'gaming laptop', 'nike shoes', 'wireless headphones']
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        suggestions = autosuggest.get_enhanced_suggestions(query, max_suggestions=5)
        for i, (suggestion, score) in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion} (score: {score:.3f})")
    
    # Print performance stats
    stats = autosuggest.get_performance_stats()
    print(f"\nPerformance Stats:")
    print(f"  Total Queries: {stats['total_queries']}")
    print(f"  Avg Response Time: {stats['avg_response_time_ms']:.2f}ms")

if __name__ == "__main__":
    main() 