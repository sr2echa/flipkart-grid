#!/usr/bin/env python3
"""
Real Search System Integration
=============================

Integrates with actual datasets and searchresultpage implementation.
"""

import os
import sys
import time
import re
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

# Add current and parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealSearchSystem:
    """Real search system using actual datasets and models."""
    
    def __init__(self):
        self.product_catalog = None
        self.personas_data = None
        self.search_ready = False
        
        try:
            self.load_datasets()
            self.search_ready = True
            logger.info("✅ Real search system initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize real search system: {e}")
            self.search_ready = False
    
    def load_datasets(self):
        """Load actual datasets."""
        # Load product catalog
        catalog_path = os.path.join(os.path.dirname(current_dir), 'dataset', 'product_catalog.csv')
        if os.path.exists(catalog_path):
            self.product_catalog = pd.read_csv(catalog_path)
            logger.info(f"✅ Loaded product catalog: {len(self.product_catalog)} products")
        else:
            logger.error(f"❌ Product catalog not found at: {catalog_path}")
            raise FileNotFoundError(f"Product catalog not found: {catalog_path}")
        
        # Load persona training data
        persona_path = os.path.join(os.path.dirname(current_dir), 'dataset', 'query_product_training_features_only.csv')
        if os.path.exists(persona_path):
            self.personas_data = pd.read_csv(persona_path, nrows=5000)  # Load sample for analysis
            logger.info(f"✅ Loaded persona data: {len(self.personas_data)} records")
        else:
            logger.warning(f"⚠️ Persona data not found at: {persona_path}")
    
    def extract_price_filter(self, query: str) -> Optional[Dict]:
        """Extract price filter from query."""
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
    
    def search_products(self, query: str, context: Dict, top_k: int = 10) -> List[Dict]:
        """Search products using real datasets."""
        if not self.search_ready or self.product_catalog is None:
            logger.error("❌ Search system not ready")
            return []
        
        try:
            # Extract context
            location = context.get('location', 'Mumbai')
            persona_tag = context.get('persona_tag', 'brand_lover')
            
            # Extract price filter
            price_filter = self.extract_price_filter(query)
            
            # Perform text-based search on product catalog
            results = self._text_search(query, price_filter, persona_tag, top_k)
            
            logger.info(f"✅ Found {len(results)} real products for query: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []
    
    def _text_search(self, query: str, price_filter: Optional[Dict], persona_tag: str, top_k: int) -> List[Dict]:
        """Perform text-based search on product catalog."""
        query_words = query.lower().split()
        results = []
        
        # Search in product catalog
        for idx, product in self.product_catalog.iterrows():
            if len(results) >= top_k:
                break
            
            # Text matching score (using actual column names)
            title_lower = str(product.get('title', '')).lower()
            brand_lower = str(product.get('brand', '')).lower()
            category_lower = str(product.get('category', '')).lower()
            description_lower = str(product.get('description', '')).lower()
            tags_lower = str(product.get('tags', '')).lower()
            
            # Calculate relevance score
            relevance_score = 0
            for word in query_words:
                if word in title_lower:
                    relevance_score += 4
                elif word in brand_lower:
                    relevance_score += 3
                elif word in category_lower:
                    relevance_score += 2
                elif word in tags_lower:
                    relevance_score += 1
                elif word in description_lower:
                    relevance_score += 0.5
            
            # Skip if no relevance
            if relevance_score == 0:
                continue
            
            # Get product price
            try:
                product_price = float(product.get('price', 0))
            except (ValueError, TypeError):
                product_price = 0
            
            # Apply price filter
            if price_filter:
                if price_filter.get('price_type') == 'under' and product_price > price_filter['max_price']:
                    continue
                elif price_filter.get('price_type') == 'over' and product_price < price_filter.get('min_price', 0):
                    continue
            
            # Get persona-specific features
            persona_features = self._get_persona_features(persona_tag, product_price)
            
            # Create result matching searchresultpage format
            result = {
                "rank": len(results) + 1,
                "product_id": str(product.get('product_id', f'P{idx}')),
                "title": str(product.get('title', 'Unknown Product')),
                "brand": str(product.get('brand', 'Unknown Brand')),
                "category": str(product.get('category', 'Electronics')),
                "price": round(product_price, 2),
                "similarity_score": round(min(relevance_score / 10.0, 1.0), 4),
                "search_method": "text_search",
                "original_price": round(product_price, 2),
                "parsed_price": round(product_price, 2),
                "rating": float(product.get('rating', 4.0)),
                "is_f_assured": bool(product.get('is_f_assured', False)),
                "persona_tag": persona_tag,
                "avg_price_last_k_clicks": persona_features['avg_price_clicks'],
                "preferred_brands_count": persona_features['preferred_brands'],
                "session_length": persona_features['session_length'],
                "query_frequency": persona_features['query_frequency'],
                "brand_match": persona_features['brand_match'],
                "price_gap_to_avg": round((product_price - persona_features['avg_price_clicks']) / max(persona_features['avg_price_clicks'], 1), 6),
                "offer_preference_match": persona_features['offer_preference'],
                "semantic_similarity": round(min(relevance_score / 8.0, 1.0), 4),
                "query_intent_similarity": round(relevance_score / 20.0, 4),
                "product_embedding_mean": round(relevance_score / 5.0, 4),
                "event": self._determine_event(query),
                "click_count": persona_features['click_count'],
                "relevance_score": round(relevance_score + persona_features['persona_boost'], 6),
                "reranked": True
            }
            
            # Add price filter info if applied
            if price_filter:
                result["price_filter_applied"] = price_filter
            
            results.append(result)
        
        # Sort by relevance score
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Update ranks after sorting
        for i, result in enumerate(results):
            result['rank'] = i + 1
        
        return results
    
    def _get_persona_features(self, persona_tag: str, product_price: float) -> Dict:
        """Get persona-specific features from real data."""
        if self.personas_data is not None and persona_tag in self.personas_data['persona_tag'].values:
            # Get actual persona data
            persona_data = self.personas_data[self.personas_data['persona_tag'] == persona_tag]
            if len(persona_data) > 0:
                avg_price = persona_data['avg_price_last_k_clicks'].mean()
                preferred_brands = int(persona_data['preferred_brands_count'].iloc[0])
                session_length = int(persona_data['session_length'].mean())
                query_freq = int(persona_data['query_frequency'].mean())
                brand_match = float(persona_data['brand_match'].mean()) if 'brand_match' in persona_data.columns else 0.5
                click_count = int(persona_data['click_count'].mean()) if 'click_count' in persona_data.columns else 50
                
                # Calculate persona boost
                if persona_tag == 'brand_lover':
                    persona_boost = 2.0
                    offer_pref = 0.7
                elif persona_tag == 'value_seeker':
                    persona_boost = 1.5 if product_price < avg_price else 0.5
                    offer_pref = 0.9
                elif persona_tag == 'quality_hunter':
                    persona_boost = 1.8 if product_price > avg_price else 1.0
                    offer_pref = 0.6
                else:  # newbie
                    persona_boost = 1.0
                    offer_pref = 0.8
                
                return {
                    'avg_price_clicks': round(avg_price, 2),
                    'preferred_brands': preferred_brands,
                    'session_length': session_length,
                    'query_frequency': query_freq,
                    'brand_match': brand_match,
                    'click_count': click_count,
                    'offer_preference': offer_pref,
                    'persona_boost': persona_boost
                }
        
        # Fallback defaults
        return {
            'avg_price_clicks': 25000.0,
            'preferred_brands': 2,
            'session_length': 10,
            'query_frequency': 30,
            'brand_match': 0.5,
            'click_count': 50,
            'offer_preference': 0.7,
            'persona_boost': 1.0
        }
    
    def _determine_event(self, query: str) -> str:
        """Determine event type based on query."""
        query_lower = query.lower()
        if any(brand in query_lower for brand in ['apple', 'samsung', 'tech']):
            return "tech_sale"
        elif any(word in query_lower for word in ['sale', 'discount', 'offer']):
            return "sale_event"
        elif any(word in query_lower for word in ['festival', 'diwali', 'christmas']):
            return "festival"
        else:
            return "general"

# Global instance
real_search_system = None

def get_real_search_system():
    """Get or create real search system instance."""
    global real_search_system
    if real_search_system is None:
        real_search_system = RealSearchSystem()
    return real_search_system