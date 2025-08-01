#!/usr/bin/env python3
"""
Grid 7.0 - LightGBM Reranking Model Integration
===============================================

This module integrates the trained LightGBM model for reranking search results
based on comprehensive features extracted from the hybrid search system.
"""

import pandas as pd
import lightgbm as lgb
import ast
import logging
from typing import List, Dict, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RerankingModel:
    """
    LightGBM-based reranking model for search results.
    """
    
    def __init__(self, model_path: str = "lgbm_rerank_model_with_label_fix.txt"):
        """
        Initialize the reranking model.
        
        Args:
            model_path: Path to the trained LightGBM model file
        """
        self.model_path = model_path
        self.model = None
        self.feature_columns = [
            'persona_tag', 'avg_price_last_k_clicks', 'preferred_brands_count',
            'session_length', 'query_frequency', 'brand', 'price', 'rating',
            'click_count', 'is_f_assured', 'brand_match', 'price_gap_to_avg',
            'offer_preference_match', 'event', 'brand_lover'
        ]
        
        self._load_model()
        logger.info("✅ Reranking Model initialized successfully")
    
    def _load_model(self):
        """Load the trained LightGBM model."""
        try:
            if os.path.exists(self.model_path):
                self.model = lgb.Booster(model_file=self.model_path)
                logger.info(f"✅ Loaded LightGBM model from: {self.model_path}")
            else:
                logger.warning(f"⚠️ Model file not found: {self.model_path}")
                self.model = None
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.model = None
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_features = df.copy()

        # Factorize categorical columns exactly as training
        for col in ['persona_tag', 'brand', 'event']:
            if col in df_features.columns:
                df_features[col] = df_features[col].fillna('unknown')
                df_features[col], _ = pd.factorize(df_features[col])
            else:
                df_features[col] = 0

        # Convert numeric features, fill NaNs with 0
        numeric_cols = [
            'avg_price_last_k_clicks', 'preferred_brands_count', 'session_length',
            'query_frequency', 'price', 'rating', 'click_count',
            'brand_match', 'price_gap_to_avg', 'offer_preference_match'
        ]
        for col in numeric_cols:
            if col in df_features.columns:
                df_features[col] = pd.to_numeric(df_features[col], errors='coerce').fillna(0)
            else:
                df_features[col] = 0

        # Derive brand_lover from brand_match (1 if brand_match == 1 else 0)
        if 'brand_match' in df_features.columns:
            df_features['brand_lover'] = df_features['brand_match'].apply(lambda x: 1 if x == 1 else 0)
        else:
            df_features['brand_lover'] = 0

        # Convert boolean to int, fill NaNs with 0
        if 'is_f_assured' in df_features.columns:
            df_features['is_f_assured'] = df_features['is_f_assured'].fillna(False).astype(int)
        else:
            df_features['is_f_assured'] = 0

        # Fill any remaining NaNs in the feature columns with 0
        df_features = df_features.fillna(0)

        return df_features
    
    def rerank_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank search results using the LightGBM model.
        
        Args:
            search_results: List of search result dictionaries
            
        Returns:
            Reranked list of search results
        """
        if not self.model:
            logger.warning("⚠️ Reranking model not available - returning original results")
            return search_results
        
        if not search_results:
            return search_results
        
        try:
            # Convert to DataFrame
            df_candidates = pd.DataFrame(search_results)
            
            # Prepare features
            df_features = self.prepare_features(df_candidates)
            
            # Check for missing features
            missing_features = [col for col in self.feature_columns if col not in df_features.columns]
            if missing_features:
                logger.warning(f"⚠️ Missing features: {missing_features} - using default values")
                for col in missing_features:
                    df_features[col] = 0
            
            # Select model feature columns
            X = df_features[self.feature_columns]
            
            # Predict relevance scores
            relevance_scores = self.model.predict(X)
            
            # Add relevance scores to results
            for i, result in enumerate(search_results):
                result['relevance_score'] = float(relevance_scores[i])
                result['reranked'] = True
            
            # Sort by relevance score (descending)
            reranked_results = sorted(search_results, key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            # Update ranks
            for i, result in enumerate(reranked_results):
                result['rank'] = i + 1
            
            logger.info(f"✅ Reranked {len(reranked_results)} results using LightGBM model")
            return reranked_results
            
        except Exception as e:
            logger.error(f"❌ Error during reranking: {e}")
            return search_results

def main():
    """Test the reranking model with sample data."""
    # Sample search results (you can replace this with actual API results)
    sample_results = [
        {
            "rank": 1,
            "product_id": "P10924",
            "title": "Samsung Galaxy A54",
            "brand": "Samsung",
            "category": "Electronics",
            "price": 83726.59,
            "rating": 0.0,
            "is_f_assured": True,
            "similarity_score": 0.5458,
            "search_method": "semantic_search",
            "persona_tag": "general_shopper",
            "avg_price_last_k_clicks": 25000.0,
            "preferred_brands_count": 3,
            "session_length": 5,
            "query_frequency": 10,
            "brand_match": 1.0,
            "price_gap_to_avg": 2.0,
            "offer_preference_match": 0.5,
            "semantic_similarity": 0.5458,
            "query_intent_similarity": 0.0,
            "product_embedding_mean": 0.4457553,
            "event": "general",
            "click_count": 0
        },
        {
            "rank": 2,
            "product_id": "P10273",
            "title": "Xiaomi Redmi Note 12",
            "brand": "Xiaomi",
            "category": "Electronics",
            "price": 28970.4,
            "rating": 0.0,
            "is_f_assured": True,
            "similarity_score": 0.516,
            "search_method": "semantic_search",
            "persona_tag": "general_shopper",
            "avg_price_last_k_clicks": 25000.0,
            "preferred_brands_count": 3,
            "session_length": 5,
            "query_frequency": 10,
            "brand_match": 0.0,
            "price_gap_to_avg": 0.15881600000000007,
            "offer_preference_match": 0.5,
            "semantic_similarity": 0.516,
            "query_intent_similarity": 0.0,
            "product_embedding_mean": 0.2632346666666667,
            "event": "general",
            "click_count": 0
        }
    ]
    
    # Initialize reranking model
    reranker = RerankingModel()
    
    # Rerank results
    reranked_results = reranker.rerank_results(sample_results)
    
    # Print results
    print("\n🔍 Original Results:")
    for result in sample_results:
        print(f"  Rank {result['rank']}: {result['title']} (Score: {result['similarity_score']:.3f})")
    
    print("\n🎯 Reranked Results:")
    for result in reranked_results:
        print(f"  Rank {result['rank']}: {result['title']} (Relevance: {result.get('relevance_score', 0):.3f})")

if __name__ == "__main__":
    main() 