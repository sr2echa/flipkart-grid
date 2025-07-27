import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
import pickle
import os
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureStore:
    """Feature store for caching frequently accessed features."""
    
    def __init__(self, cache_dir: str = 'cache'):
        self.cache_dir = cache_dir
        self.feature_cache = {}
        os.makedirs(cache_dir, exist_ok=True)
        self.load_cached_features()
    
    def load_cached_features(self):
        cache_file = os.path.join(self.cache_dir, 'feature_store.pkl')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.feature_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.feature_cache)} cached features")
            except Exception as e:
                logger.warning(f"Failed to load feature store: {e}")
    
    def save_cached_features(self):
        cache_file = os.path.join(self.cache_dir, 'feature_store.pkl')
        with open(cache_file, 'wb') as f:
            pickle.dump(self.feature_cache, f)
        logger.info("Saved feature store")
    
    def get_feature(self, key: str) -> Optional[float]:
        return self.feature_cache.get(key)
    
    def set_feature(self, key: str, value: float):
        self.feature_cache[key] = value

class AdvancedFeatureEngineering:
    """Advanced feature engineering with interaction features."""
    
    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store
        self.label_encoders = {}
        self.scalers = {}
    
    def create_features(self, query: str, suggestion: str, suggestion_data: Dict,
                       user_id: Optional[str] = None) -> Dict[str, float]:
        """Create comprehensive features for query-suggestion pair."""
        features = {}
        
        # Basic features
        features['query_length'] = len(query)
        features['suggestion_length'] = len(suggestion)
        features['suggestion_frequency'] = suggestion_data.get('frequency', 0)
        features['suggestion_score'] = suggestion_data.get('score', 0.0)
        features['exact_match'] = float(query.lower() == suggestion.lower())
        features['query_in_suggestion'] = float(query.lower() in suggestion.lower())
        
        # Interaction features
        features['frequency_score_interaction'] = (
            features['suggestion_frequency'] * features['suggestion_score']
        )
        features['length_ratio'] = (
            features['query_length'] / max(features['suggestion_length'], 1)
        )
        
        # Temporal features
        now = datetime.now()
        features['hour_of_day'] = now.hour
        features['day_of_week'] = now.weekday()
        features['is_weekend'] = float(now.weekday() >= 5)
        
        # Personalization features
        if user_id:
            user_key = f"user_{user_id}_preference"
            user_pref = self.feature_store.get_feature(user_key)
            features['user_preference'] = user_pref if user_pref else 0.5
        else:
            features['user_preference'] = 0.5
        
        return features

class EnhancedReranker:
    """Enhanced reranker with multi-task learning and feature store."""
    
    def __init__(self, enable_multi_task: bool = True, cache_dir: str = 'cache'):
        self.enable_multi_task = enable_multi_task
        self.feature_store = FeatureStore(cache_dir)
        self.feature_engineering = AdvancedFeatureEngineering(self.feature_store)
        
        # Models
        self.relevance_model = None
        self.click_model = None
        
        # Statistics
        self.total_predictions = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def train_models(self, training_data: List[Dict], relevance_labels: List[float],
                    click_labels: Optional[List[int]] = None):
        """Train reranker models."""
        logger.info(f"Training reranker with {len(training_data)} examples...")
        
        # Extract features
        X = []
        for example in training_data:
            features = self.feature_engineering.create_features(
                example['query'], example['suggestion'], example['suggestion_data'],
                example.get('user_id')
            )
            X.append(list(features.values()))
        
        X = np.array(X)
        y_relevance = np.array(relevance_labels)
        
        # Train relevance model
        self.relevance_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.relevance_model.fit(X, y_relevance)
        
        # Train click model if enabled
        if self.enable_multi_task and click_labels:
            y_click = np.array(click_labels)
            self.click_model = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            self.click_model.fit(X, y_click)
        
        logger.info("Reranker training completed")
    
    def predict_relevance(self, query: str, suggestions: List[Dict],
                         user_id: Optional[str] = None) -> List[Dict]:
        """Predict relevance scores for suggestions."""
        self.total_predictions += 1
        
        if not self.relevance_model:
            return suggestions
        
        # Create features and make predictions
        X = []
        valid_suggestions = []
        
        for suggestion_data in suggestions:
            suggestion = suggestion_data.get('suggestion', '')
            
            # Check cache
            cache_key = f"relevance_{query}_{suggestion}_{user_id or 'anonymous'}"
            cached_score = self.feature_store.get_feature(cache_key)
            
            if cached_score is not None:
                self.cache_hits += 1
                suggestion_data['relevance_score'] = cached_score
                valid_suggestions.append(suggestion_data)
            else:
                self.cache_misses += 1
                features = self.feature_engineering.create_features(
                    query, suggestion, suggestion_data, user_id
                )
                X.append(list(features.values()))
                valid_suggestions.append(suggestion_data)
        
        # Make predictions
        if X:
            X = np.array(X)
            relevance_scores = self.relevance_model.predict(X)
            
            for i, suggestion_data in enumerate(valid_suggestions):
                score = float(relevance_scores[i])
                suggestion_data['relevance_score'] = score
                
                # Cache score
                suggestion = suggestion_data.get('suggestion', '')
                cache_key = f"relevance_{query}_{suggestion}_{user_id or 'anonymous'}"
                self.feature_store.set_feature(cache_key, score)
        
        # Sort by relevance
        valid_suggestions.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return valid_suggestions
    
    def predict_multi_objective(self, query: str, suggestions: List[Dict],
                              user_id: Optional[str] = None) -> List[Dict]:
        """Predict multiple objectives."""
        if not self.enable_multi_task:
            return self.predict_relevance(query, suggestions, user_id)
        
        # Create features
        X = []
        for suggestion_data in suggestions:
            suggestion = suggestion_data.get('suggestion', '')
            features = self.feature_engineering.create_features(
                query, suggestion, suggestion_data, user_id
            )
            X.append(list(features.values()))
        
        if not X:
            return suggestions
        
        X = np.array(X)
        
        # Make predictions
        if self.relevance_model:
            relevance_scores = self.relevance_model.predict(X)
        
        if self.click_model:
            click_probs = self.click_model.predict_proba(X)[:, 1]
        
        # Combine predictions
        for i, suggestion_data in enumerate(suggestions):
            if self.relevance_model:
                suggestion_data['relevance_score'] = float(relevance_scores[i])
            
            if self.click_model:
                suggestion_data['click_probability'] = float(click_probs[i])
            
            # Combined score
            relevance = suggestion_data.get('relevance_score', 0)
            click_prob = suggestion_data.get('click_probability', 0)
            combined = 0.7 * relevance + 0.3 * click_prob
            suggestion_data['combined_score'] = combined
        
        # Sort by combined score
        suggestions.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        return suggestions
    
    def get_statistics(self) -> Dict:
        """Get reranker statistics."""
        return {
            'total_predictions': self.total_predictions,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'multi_task_enabled': self.enable_multi_task
        }
    
    def save_models(self, model_path: str):
        """Save trained models."""
        os.makedirs(model_path, exist_ok=True)
        
        if self.relevance_model:
            self.relevance_model.save_model(os.path.join(model_path, 'relevance_model.json'))
        
        if self.click_model:
            self.click_model.save_model(os.path.join(model_path, 'click_model.json'))
        
        self.feature_store.save_cached_features()
        logger.info(f"Saved models to {model_path}")

if __name__ == "__main__":
    # Example usage
    reranker = EnhancedReranker(enable_multi_task=True)
    
    # Sample training data
    training_data = [
        {'query': 'samsung', 'suggestion': 'samsung galaxy', 'suggestion_data': {'frequency': 1000, 'score': 0.9}},
        {'query': 'samsung', 'suggestion': 'samsung phone', 'suggestion_data': {'frequency': 800, 'score': 0.8}},
        {'query': 'apple', 'suggestion': 'apple iphone', 'suggestion_data': {'frequency': 1200, 'score': 0.95}}
    ]
    
    relevance_labels = [0.9, 0.7, 0.95]
    click_labels = [1, 0, 1]
    
    # Train models
    reranker.train_models(training_data, relevance_labels, click_labels)
    
    # Test predictions
    test_suggestions = [
        {'suggestion': 'samsung galaxy', 'frequency': 1000, 'score': 0.9},
        {'suggestion': 'samsung phone', 'frequency': 800, 'score': 0.8},
        {'suggestion': 'samsung tv', 'frequency': 600, 'score': 0.6}
    ]
    
    print("Testing Enhanced Reranker:")
    results = reranker.predict_multi_objective('samsung', test_suggestions)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. '{result['suggestion']}' "
              f"(relevance: {result.get('relevance_score', 0):.3f}, "
              f"click: {result.get('click_probability', 0):.3f}, "
              f"combined: {result.get('combined_score', 0):.3f})")
    
    print(f"\nStatistics: {reranker.get_statistics()}")
    reranker.save_models('models') 