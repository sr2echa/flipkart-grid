# feature_extraction.py
"""
Grid 7.0 - Feature Extraction System for Robust Dataset
=======================================================

This module extracts comprehensive features for products using the new robust dataset.
It ensures all products have the required 17 features for optimal search results.

Required Features:
- persona_tag, avg_price_last_k_clicks, preferred_brands_count
- session_length, query_frequency, brand, price, rating
- click_count, is_f_assured, brand_match, price_gap_to_avg
- offer_preference_match, semantic_similarity
- query_intent_similarity, product_embedding_mean, event
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import os
from typing import List, Dict, Any

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Feature extraction system for comprehensive product analysis
    using the new robust dataset structure.
    """
    
    def __init__(self, session_log_path: str = None, user_queries_path: str = None, realtime_data_path: str = None):
        """
        Initialize the feature extractor with new dataset paths.
        
        Args:
            session_log_path: Path to session log CSV
            user_queries_path: Path to user queries CSV
            realtime_data_path: Path to realtime product info CSV
        """
        self.session_log_path = session_log_path
        self.user_queries_path = user_queries_path
        self.realtime_data_path = realtime_data_path
        
        # Data storage
        self.session_log = None
        self.user_queries = None
        self.realtime_data = None
        self.user_profiles = {}
        
        # Load datasets
        self._load_datasets()
        self._build_user_profiles()
        
        logger.info("✅ Feature Extractor initialized successfully")
    
    def _load_datasets(self):
        """Load all necessary datasets with robust error handling."""
        logger.info("📂 Loading datasets for feature extraction...")
        
        try:
            # Load session log
            if os.path.exists(self.session_log_path):
                self.session_log = pd.read_csv(self.session_log_path)
                logger.info(f"✅ Loaded session log: {len(self.session_log)} records")
                logger.info(f"📋 Session log columns: {list(self.session_log.columns)}")
            else:
                logger.warning(f"⚠️ Session log not found: {self.session_log_path}")
                self.session_log = pd.DataFrame()
            
            # Load user queries
            if os.path.exists(self.user_queries_path):
                self.user_queries = pd.read_csv(self.user_queries_path)
                logger.info(f"✅ Loaded user queries: {len(self.user_queries)} records")
                logger.info(f"📋 User queries columns: {list(self.user_queries.columns)}")
            else:
                logger.warning(f"⚠️ User queries not found: {self.user_queries_path}")
                self.user_queries = pd.DataFrame()
            
            # Load realtime data
            if os.path.exists(self.realtime_data_path):
                self.realtime_data = pd.read_csv(self.realtime_data_path)
                logger.info(f"✅ Loaded realtime data: {len(self.realtime_data)} records")
            else:
                logger.warning(f"⚠️ Realtime data not found: {self.realtime_data_path}")
                self.realtime_data = pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Failed to load datasets: {e}")
            # Create empty DataFrames as fallback
            self.session_log = pd.DataFrame()
            self.user_queries = pd.DataFrame()
            self.realtime_data = pd.DataFrame()
    
    def _build_user_profiles(self):
        """Build user profiles from session data with robust handling."""
        logger.info("👥 Building user profiles...")
        
        if self.session_log.empty:
            logger.warning("⚠️ No session log data available - using default profiles")
            self.user_profiles = self._create_default_profiles()
            return
        
        # Check if user_id column exists
        if 'user_id' not in self.session_log.columns:
            logger.warning("⚠️ No user_id column in session log - using session_id as user_id")
            # Use session_id as user_id if user_id doesn't exist
            self.session_log['user_id'] = self.session_log['session_id']
        
        # Group by user_id to build profiles
        try:
            user_sessions = self.session_log.groupby('user_id')
            
            self.user_profiles = {}
            for user_id, user_data in user_sessions:
                profile = self._create_user_profile(user_id, user_data)
                self.user_profiles[user_id] = profile
            
            logger.info(f"✅ Built profiles for {len(self.user_profiles)} users")
            
        except Exception as e:
            logger.error(f"❌ Error building user profiles: {e}")
            logger.info("🔄 Using default profiles instead")
            self.user_profiles = self._create_default_profiles()
    
    def _create_default_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Create default user profiles when data is not available."""
        default_profiles = {
            'default_user': {
                'user_id': 'default_user',
                'persona_tag': 'general_shopper',
                'preferred_brands': ['Samsung', 'Apple', 'Nike'],
                'avg_price_last_k_clicks': 25000.0,
                'session_length': 5,
                'query_frequency': 10,
                'event_preferences': ['general']
            },
            'premium_user': {
                'user_id': 'premium_user',
                'persona_tag': 'premium_shopper',
                'preferred_brands': ['Apple', 'Samsung', 'Sony'],
                'avg_price_last_k_clicks': 75000.0,
                'session_length': 8,
                'query_frequency': 15,
                'event_preferences': ['tech_sale']
            },
            'budget_user': {
                'user_id': 'budget_user',
                'persona_tag': 'budget_conscious',
                'preferred_brands': ['Xiaomi', 'Realme', 'Puma'],
                'avg_price_last_k_clicks': 8000.0,
                'session_length': 3,
                'query_frequency': 8,
                'event_preferences': ['fashion_sale']
            }
        }
        
        logger.info(f"✅ Created {len(default_profiles)} default user profiles")
        return default_profiles
    
    def _create_user_profile(self, user_id: str, user_data: pd.DataFrame) -> Dict[str, Any]:
        """Create a single user profile."""
        return {
            'user_id': user_id,
            'persona_tag': self._determine_persona(user_data),
            'preferred_brands': self._get_preferred_brands(user_data),
            'avg_price_last_k_clicks': self._calculate_avg_price_last_k_clicks(user_data),
            'session_length': self._calculate_session_length(user_data),
            'query_frequency': self._calculate_query_frequency(user_id),
            'event_preferences': self._get_event_preferences(user_data)
        }
    
    def _determine_persona(self, user_data: pd.DataFrame) -> str:
        """Determine user persona based on behavior."""
        if user_data.empty:
            return 'general_shopper'
        
        # Analyze user behavior patterns
        avg_price = user_data['price'].mean() if 'price' in user_data.columns else 25000.0
        
        if avg_price > 50000:
            return 'premium_shopper'
        elif avg_price < 15000:
            return 'budget_conscious'
        else:
            return 'general_shopper'
    
    def _get_preferred_brands(self, user_data: pd.DataFrame) -> List[str]:
        """Get user's preferred brands."""
        if 'brand' not in user_data.columns:
            return ['Samsung', 'Apple', 'Nike']  # Default brands
        
        brand_counts = user_data['brand'].value_counts()
        # Get top 3 brands
        return brand_counts.head(3).index.tolist()
    
    def _calculate_avg_price_last_k_clicks(self, user_data: pd.DataFrame, k: int = 10) -> float:
        """Calculate average price of last k clicks."""
        if 'price' not in user_data.columns:
            return 25000.0  # Default average price
        
        # Sort by timestamp if available, otherwise use index
        sorted_data = user_data.sort_values(by='timestamp', ascending=False) if 'timestamp' in user_data.columns else user_data
        last_k_prices = sorted_data.head(k)['price'].dropna()
        return float(last_k_prices.mean()) if len(last_k_prices) > 0 else 25000.0
    
    def _calculate_session_length(self, user_data: pd.DataFrame) -> int:
        """Calculate average session length for user."""
        if 'session_id' not in user_data.columns:
            return 5  # Default session length
        
        session_counts = user_data.groupby('session_id').size()
        return int(session_counts.mean()) if len(session_counts) > 0 else 5
    
    def _calculate_query_frequency(self, user_id: str) -> int:
        """Calculate query frequency for user."""
        if self.user_queries is None:
            return 10  # Default query frequency
        
        if 'user_id' not in self.user_queries.columns:
            return 10
        
        user_queries = self.user_queries[self.user_queries['user_id'] == user_id]
        return len(user_queries) if len(user_queries) > 0 else 10
    
    def _get_event_preferences(self, user_data: pd.DataFrame) -> List[str]:
        """Get user's event preferences based on clicked products."""
        if 'event' not in user_data.columns:
            return ['general']
        
        events = user_data['event'].value_counts().head(3).index.tolist()
        return events if events else ['general']
    
    def extract_features_for_products(self, 
                                    products: List[Dict[str, Any]], 
                                    query: str, 
                                    user_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Extract comprehensive features for products using new dataset structure.
        
        Args:
            products: List of product dictionaries
            query: Search query
            user_context: User context information
            
        Returns:
            List of products enriched with all required features
        """
        if not products:
            return []
        
        logger.info(f"🔧 Extracting features for {len(products)} products")
        
        # Get user profile - prioritize persona_tag from user context
        user_id = user_context.get('user_id', 'default_user') if user_context else 'default_user'
        user_profile = self.user_profiles.get(user_id, self._get_default_user_profile())
        
        # Override persona_tag with user context if provided
        if user_context and 'persona_tag' in user_context:
            user_profile['persona_tag'] = user_context['persona_tag']
            logger.info(f"🎭 Using persona_tag from user context: {user_context['persona_tag']}")
        else:
            logger.info(f"🎭 Using default persona_tag: {user_profile['persona_tag']}")
        
        enriched_products = []
        
        for i, product in enumerate(products):
            enriched = product.copy()
            
            # Add rank
            enriched['rank'] = i + 1
            
            # Basic product features (from new dataset structure)
            enriched['brand'] = product.get('brand', 'N/A')
            enriched['price'] = product.get('price', 0)
            enriched['rating'] = product.get('rating', product.get('seller_rating', 4.0))  # Use seller_rating as fallback
            enriched['is_f_assured'] = product.get('is_f_assured', False)
            
            # User-specific features
            enriched['persona_tag'] = user_profile['persona_tag']
            enriched['avg_price_last_k_clicks'] = user_profile['avg_price_last_k_clicks']
            enriched['preferred_brands_count'] = len(user_profile['preferred_brands'])
            enriched['session_length'] = user_profile['session_length']
            enriched['query_frequency'] = user_profile['query_frequency']
            
            # Matching features
            enriched['brand_match'] = self._calculate_brand_match(product, user_profile)
            enriched['price_gap_to_avg'] = self._calculate_price_gap(product, user_profile)
            
            # Offer preference
            enriched['offer_preference_match'] = self._calculate_offer_preference_match(product, user_context)
            
            # Semantic features
            enriched['semantic_similarity'] = product.get('similarity_score', 0.0)
            enriched['query_intent_similarity'] = self._calculate_query_intent_similarity(query, product)
            enriched['product_embedding_mean'] = self._calculate_product_embedding_mean(product)
            
            # Event and click features
            enriched['event'] = self._determine_event_type(product, user_context)
            enriched['click_count'] = self._get_click_count(product)
            
            enriched_products.append(enriched)
        
        logger.info(f"✅ Feature extraction completed for {len(enriched_products)} products")
        return enriched_products
    
    def _calculate_brand_match(self, product: Dict[str, Any], user_profile: Dict[str, Any]) -> float:
        """Calculate brand match score."""
        product_brand = product.get('brand', '').lower()
        preferred_brands = [brand.lower() for brand in user_profile.get('preferred_brands', [])]
        
        if product_brand in preferred_brands:
            return 1.0
        elif any(brand in product_brand or product_brand in brand for brand in preferred_brands):
            return 0.7
        else:
            return 0.0
    
    def _calculate_price_gap(self, product: Dict[str, Any], user_profile: Dict[str, Any]) -> float:
        """Calculate price gap to user's average."""
        product_price = product.get('price', 0)
        avg_price = user_profile.get('avg_price_last_k_clicks', 25000.0)
        
        if avg_price == 0:
            return 0.0
        
        return abs(product_price - avg_price) / avg_price
    
    def _calculate_offer_preference_match(self, product: Dict[str, Any], user_context: Dict[str, Any]) -> float:
        """Calculate offer preference match."""
        if not user_context:
            return 0.5
        
        # Check if user prefers premium products
        price_range = user_context.get('price_range', 'general')
        product_price = product.get('price', 0)
        
        if price_range == 'premium' and product_price > 50000:
            return 1.0
        elif price_range == 'budget' and product_price < 15000:
            return 1.0
        elif price_range == 'general':
            return 0.7
        else:
            return 0.3
    
    def _calculate_query_intent_similarity(self, query: str, product: Dict[str, Any]) -> float:
        """Calculate query intent similarity."""
        query_words = set(query.lower().split())
        product_title = product.get('title', '').lower()
        product_words = set(product_title.split())
        
        if not query_words or not product_words:
            return 0.0
        
        intersection = query_words.intersection(product_words)
        union = query_words.union(product_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_product_embedding_mean(self, product: Dict[str, Any]) -> float:
        """Calculate product embedding mean (simplified)."""
        # This would normally use actual embeddings
        # For now, use a simple heuristic based on product features
        features = [
            product.get('price', 0) / 10000,  # Normalized price
            product.get('rating', 4.0) / 5.0,  # Normalized rating
            float(product.get('is_f_assured', False)),  # F-assured
        ]
        
        return np.mean(features) if features else 0.5
    
    def _determine_event_type(self, product: Dict[str, Any], user_context: Dict[str, Any]) -> str:
        """Determine event type based on product and context."""
        category = product.get('category', '').lower()
        
        if 'electronics' in category or 'mobile' in category:
            return 'tech_sale'
        elif 'fashion' in category or 'clothing' in category:
            return 'fashion_sale'
        elif 'home' in category or 'furniture' in category:
            return 'home_decor'
        else:
            return 'general'
    
    def _get_click_count(self, product: Dict[str, Any]) -> int:
        """Get click count for product (simplified)."""
        # This would normally come from real-time data
        # For now, use a simple heuristic
        price = product.get('price', 0)
        rating = product.get('rating', 4.0)
        
        # Higher rating and moderate price = more clicks
        if rating > 4.5 and 10000 < price < 50000:
            return 150
        elif rating > 4.0:
            return 100
        else:
            return 50
    
    def validate_features(self, products: List[Dict[str, Any]]) -> bool:
        """
        Validate that all products have the required features.
        
        Args:
            products: List of product dictionaries
            
        Returns:
            True if all products have required features, False otherwise
        """
        if not products:
            return False
        
        required_features = [
            'persona_tag', 'avg_price_last_k_clicks', 'preferred_brands_count',
            'session_length', 'query_frequency', 'brand', 'price', 'rating',
            'click_count', 'is_f_assured', 'brand_match', 'price_gap_to_avg',
            'offer_preference_match', 'semantic_similarity',
            'query_intent_similarity', 'product_embedding_mean', 'event'
        ]
        
        for i, product in enumerate(products):
            missing_features = [f for f in required_features if f not in product]
            if missing_features:
                logger.warning(f"⚠️ Product {i} missing features: {missing_features}")
                return False
        
        logger.info(f"✅ All {len(products)} products have required features")
        return True
    
    def _get_default_user_profile(self) -> Dict[str, Any]:
        """Returns a default user profile."""
        return {
            'persona_tag': 'general_shopper',
            'preferred_brands': ['Samsung', 'Apple', 'Nike'],
            'avg_price_last_k_clicks': 25000.0,
            'session_length': 5,
            'query_frequency': 10,
            'event_preferences': ['general']
        }

# Test the feature extractor
if __name__ == "__main__":
    # Initialize feature extractor
    extractor = FeatureExtractor(
        session_log_path="R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\dataset\\session_log.csv",
        user_queries_path="R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\dataset\\user_queries.csv",
        realtime_data_path="R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\dataset\\realtime_product_info.csv"
    )
    
    # Test with sample products
    sample_products = [
        {
            'product_id': 'test_1',
            'title': 'Samsung Galaxy Smartphone',
            'brand': 'Samsung',
            'price': 45000,
            'rating': 4.5,
            'is_f_assured': True
        },
        {
            'product_id': 'test_2',
            'title': 'Nike Running Shoes',
            'brand': 'Nike',
            'price': 2500,
            'rating': 4.2,
            'is_f_assured': False
        }
    ]
    
    user_context = {
        'user_id': 'test_user',
        'location': 'Mumbai',
        'price_range': 'premium'
    }
    
    # Extract features
    enriched_products = extractor.extract_features_for_products(
        sample_products, 
        "samsung smartphone", 
        user_context
    )
    
    # Validate features
    is_valid = extractor.validate_features(enriched_products)
    
    print(f"✅ Feature extraction test completed: {is_valid}")
    print(f"📊 Enriched {len(enriched_products)} products")
    
    if enriched_products:
        print("\nSample enriched product:")
        for key, value in enriched_products[0].items():
            print(f"  {key}: {value}") 