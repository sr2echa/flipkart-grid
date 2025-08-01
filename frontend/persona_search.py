#!/usr/bin/env python3
"""
Persona-Aware Search System
============================

Enhanced search system that integrates persona preferences, context awareness,
and intelligent reranking for personalized product discovery.
"""

import logging
from typing import List, Dict, Any, Optional
from search_system import SimplifiedSearcher
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonaAwareSearcher:
    """
    Enhanced search system with persona integration and context awareness.
    """
    
    def __init__(self):
        self.simplified_searcher = SimplifiedSearcher()
        
        # Persona-specific boosting rules
        self.persona_boost_rules = {
            'tech_enthusiast': {
                'brand_boost': {'Apple': 1.3, 'Samsung': 1.2, 'Sony': 1.2, 'Dell': 1.1, 'HP': 1.1},
                'category_boost': {'Electronics': 1.3, 'Computers': 1.3, 'Mobiles & Accessories': 1.2},
                'feature_keywords': ['latest', 'premium', 'pro', 'advanced', 'smart', 'wireless', '5g', '4k', 'hd'],
                'price_preference': 'premium'  # Higher end products
            },
            'fashion_lover': {
                'brand_boost': {'Nike': 1.3, 'Adidas': 1.3, 'Zara': 1.2, 'H&M': 1.2, 'Puma': 1.1},
                'category_boost': {'Clothing': 1.3, 'Footwear': 1.3, 'Fashion': 1.3, 'Sports Wear': 1.1},
                'feature_keywords': ['trendy', 'stylish', 'designer', 'fashionable', 'elegant', 'chic', 'new', 'collection'],
                'price_preference': 'mid_to_high'
            },
            'budget_shopper': {
                'brand_boost': {'Generic': 1.2, 'Local': 1.1},  # Boost affordable brands
                'category_boost': {'Budget Items': 1.3, 'Sale Items': 1.3},
                'feature_keywords': ['cheap', 'affordable', 'budget', 'discount', 'sale', 'offer', 'deal', 'value'],
                'price_preference': 'low',
                'price_boost': {  # Boost products under certain price ranges
                    'max_1000': 1.4,
                    'max_2000': 1.2,
                    'max_5000': 1.1
                }
            },
            'sports_enthusiast': {
                'brand_boost': {'Nike': 1.4, 'Adidas': 1.4, 'Puma': 1.3, 'Reebok': 1.2, 'Under Armour': 1.2},
                'category_boost': {'Sports Shoes': 1.4, 'Sports Wear': 1.3, 'Fitness': 1.3, 'Athletic': 1.2},
                'feature_keywords': ['sports', 'fitness', 'athletic', 'performance', 'durable', 'professional', 'training', 'running'],
                'price_preference': 'mid_to_high'
            }
        }
        
        # Location-specific preferences
        self.location_preferences = {
            'Mumbai': {'delivery_boost': 1.1, 'local_brands': []},
            'Delhi': {'delivery_boost': 1.1, 'local_brands': []},
            'Bangalore': {'delivery_boost': 1.1, 'tech_boost': 1.1},
            'Chennai': {'delivery_boost': 1.1},
            'Kolkata': {'delivery_boost': 1.1},
            'Hyderabad': {'delivery_boost': 1.1, 'tech_boost': 1.05}
        }
        
        logger.info("🎯 Persona-Aware Search System initialized")
    
    def search_with_persona(self, query: str, context: Dict, top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Perform search with full persona integration and context awareness.
        
        Args:
            query: Search query
            context: Context including persona, location, preferences
            top_k: Number of results to return (default 50)
            
        Returns:
            List of personalized and reranked search results
        """
        logger.info(f"🔍 Persona-aware search: '{query}' (top {top_k})")
        
        # Extract persona and context
        persona_id = context.get('persona', 'tech_enthusiast')
        persona_details = context.get('persona_details', {})
        location = context.get('location', 'Mumbai')
        
        # Get base search results (request more for better reranking)
        base_results = self.simplified_searcher.search(query, min(top_k * 2, 100), context)
        
        if not base_results:
            logger.info("❌ No base results found")
            return []
        
        logger.info(f"📊 Got {len(base_results)} base results")
        
        # Apply persona-based reranking
        enhanced_results = self._apply_persona_reranking(
            base_results, persona_id, persona_details, query
        )
        
        # Apply location-based adjustments
        enhanced_results = self._apply_location_boost(enhanced_results, location)
        
        # Apply query-specific boosts
        enhanced_results = self._apply_query_specific_boost(enhanced_results, query, persona_id)
        
        # Final ranking and filtering
        final_results = self._final_ranking(enhanced_results, top_k)
        
        logger.info(f"✅ Returning {len(final_results)} persona-enhanced results")
        return final_results
    
    def _apply_persona_reranking(self, results: List[Dict], persona_id: str, persona_details: Dict, query: str) -> List[Dict]:
        """Apply persona-based reranking to search results."""
        
        persona_rules = self.persona_boost_rules.get(persona_id, {})
        brand_boosts = persona_rules.get('brand_boost', {})
        category_boosts = persona_rules.get('category_boost', {})
        feature_keywords = persona_rules.get('feature_keywords', [])
        price_preference = persona_rules.get('price_preference', 'mid')
        
        enhanced_results = []
        
        for result in results:
            enhanced_result = result.copy()
            original_score = result.get('search_score', 0.5)
            persona_boost = 1.0
            
            # Brand preference boost
            brand = result.get('brand', '').strip()
            if brand in brand_boosts:
                persona_boost *= brand_boosts[brand]
                logger.debug(f"Brand boost for {brand}: {brand_boosts[brand]}")
            
            # Category preference boost
            category = result.get('category', '').strip()
            for cat, boost in category_boosts.items():
                if cat.lower() in category.lower():
                    persona_boost *= boost
                    logger.debug(f"Category boost for {category}: {boost}")
                    break
            
            # Feature keyword boost
            title = result.get('title', '').lower()
            description = result.get('description', '').lower()
            tags = result.get('tags', '').lower()
            
            text_content = f"{title} {description} {tags}"
            keyword_matches = sum(1 for keyword in feature_keywords if keyword in text_content)
            if keyword_matches > 0:
                keyword_boost = 1.0 + (keyword_matches * 0.1)  # 10% boost per keyword
                persona_boost *= min(keyword_boost, 1.5)  # Cap at 50% boost
                logger.debug(f"Keyword boost: {keyword_boost} (matches: {keyword_matches})")
            
            # Price preference boost
            price = float(result.get('price', 0))
            price_boost = self._calculate_price_boost(price, price_preference, persona_rules)
            persona_boost *= price_boost
            
            # Previous query/brand history boost
            if persona_details:
                history_boost = self._calculate_history_boost(result, persona_details)
                persona_boost *= history_boost
            
            # Apply F-Assured boost for premium personas
            if result.get('is_f_assured') and persona_id in ['tech_enthusiast', 'fashion_lover']:
                persona_boost *= 1.1
            
            # Calculate final score
            enhanced_result['original_score'] = original_score
            enhanced_result['persona_boost'] = persona_boost
            enhanced_result['persona_score'] = original_score * persona_boost
            enhanced_result['persona_applied'] = persona_id
            
            enhanced_results.append(enhanced_result)
        
        # Sort by persona-enhanced score
        enhanced_results.sort(key=lambda x: x['persona_score'], reverse=True)
        
        return enhanced_results
    
    def _calculate_price_boost(self, price: float, preference: str, persona_rules: Dict) -> float:
        """Calculate price-based boost based on persona preference."""
        
        if preference == 'low':  # Budget shopper
            price_boosts = persona_rules.get('price_boost', {})
            if price <= 1000:
                return price_boosts.get('max_1000', 1.4)
            elif price <= 2000:
                return price_boosts.get('max_2000', 1.2)
            elif price <= 5000:
                return price_boosts.get('max_5000', 1.1)
            else:
                return 0.9  # Penalize expensive items for budget shoppers
        
        elif preference == 'premium':  # Tech enthusiast
            if price >= 50000:
                return 1.2  # Boost very expensive items
            elif price >= 20000:
                return 1.1  # Boost expensive items
            elif price <= 5000:
                return 0.9  # Slightly penalize very cheap items
        
        elif preference == 'mid_to_high':  # Fashion lover, sports enthusiast
            if price >= 10000 and price <= 50000:
                return 1.1  # Boost mid-to-high range
            elif price <= 2000:
                return 0.95  # Slightly penalize very cheap items
        
        return 1.0  # No boost
    
    def _calculate_history_boost(self, result: Dict, persona_details: Dict) -> float:
        """Calculate boost based on user's previous queries and clicks."""
        
        boost = 1.0
        
        # Previous queries boost
        previous_queries = persona_details.get('previous_queries', [])
        title_lower = result.get('title', '').lower()
        
        for prev_query in previous_queries:
            if prev_query.lower() in title_lower:
                boost += 0.1
        
        # Clicked brands boost
        clicked_brands = persona_details.get('clicked_brands', [])
        result_brand = result.get('brand', '').lower()
        
        for brand in clicked_brands:
            if brand.lower() == result_brand:
                boost += 0.15
                break
        
        # Clicked categories boost
        clicked_categories = persona_details.get('clicked_categories', [])
        result_category = result.get('category', '').lower()
        
        for category in clicked_categories:
            if category.lower() in result_category:
                boost += 0.1
                break
        
        return min(boost, 1.5)  # Cap the boost
    
    def _apply_location_boost(self, results: List[Dict], location: str) -> List[Dict]:
        """Apply location-based boosts."""
        
        location_prefs = self.location_preferences.get(location, {})
        delivery_boost = location_prefs.get('delivery_boost', 1.0)
        tech_boost = location_prefs.get('tech_boost', 1.0)
        
        for result in results:
            location_boost = 1.0
            
            # Delivery boost for all products
            location_boost *= delivery_boost
            
            # Tech boost for tech cities
            if tech_boost > 1.0:
                category = result.get('category', '').lower()
                if any(tech_term in category for tech_term in ['electronics', 'computer', 'mobile']):
                    location_boost *= tech_boost
            
            # Apply location boost
            result['location_boost'] = location_boost
            result['persona_score'] = result.get('persona_score', 0) * location_boost
        
        return results
    
    def _apply_query_specific_boost(self, results: List[Dict], query: str, persona_id: str) -> List[Dict]:
        """Apply query-specific boosts based on query intent."""
        
        query_lower = query.lower()
        
        for result in results:
            query_boost = 1.0
            
            # Exact brand match in query
            brand = result.get('brand', '').lower()
            if brand and brand in query_lower:
                query_boost *= 1.3
            
            # Exact model/title match
            title_words = result.get('title', '').lower().split()
            query_words = query_lower.split()
            
            # Count exact word matches
            exact_matches = sum(1 for word in query_words if word in title_words)
            if exact_matches > 0:
                match_boost = 1.0 + (exact_matches * 0.05)  # 5% per exact word match
                query_boost *= min(match_boost, 1.3)
            
            # Price range detection in query
            price_patterns = [
                r'under\s+(\d+)', r'below\s+(\d+)', r'less\s+than\s+(\d+)',
                r'above\s+(\d+)', r'over\s+(\d+)', r'more\s+than\s+(\d+)'
            ]
            
            for pattern in price_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    price_limit = int(match.group(1))
                    product_price = float(result.get('price', 0))
                    
                    if 'under' in pattern or 'below' in pattern or 'less' in pattern:
                        if product_price <= price_limit:
                            query_boost *= 1.2
                    elif 'above' in pattern or 'over' in pattern or 'more' in pattern:
                        if product_price >= price_limit:
                            query_boost *= 1.2
            
            # Apply query boost
            result['query_boost'] = query_boost
            result['persona_score'] = result.get('persona_score', 0) * query_boost
        
        return results
    
    def _final_ranking(self, results: List[Dict], top_k: int) -> List[Dict]:
        """Final ranking and filtering of results."""
        
        # Sort by final persona score
        results.sort(key=lambda x: x.get('persona_score', 0), reverse=True)
        
        # Add final ranking information
        for i, result in enumerate(results[:top_k]):
            result['final_rank'] = i + 1
            result['personalization_applied'] = True
            
            # Add explanation of boosts applied
            boosts_applied = []
            if result.get('persona_boost', 1.0) > 1.0:
                boosts_applied.append('persona')
            if result.get('location_boost', 1.0) > 1.0:
                boosts_applied.append('location')
            if result.get('query_boost', 1.0) > 1.0:
                boosts_applied.append('query_match')
            
            result['boosts_applied'] = boosts_applied
        
        return results[:top_k]

def main():
    """Test the persona-aware search system."""
    searcher = PersonaAwareSearcher()
    
    test_cases = [
        {
            'query': 'apple watch ultra',
            'context': {
                'persona': 'tech_enthusiast',
                'persona_details': {
                    'previous_queries': ['smartwatch', 'apple', 'premium'],
                    'clicked_brands': ['Apple', 'Samsung'],
                    'clicked_categories': ['Electronics', 'Wearables']
                },
                'location': 'Bangalore'
            }
        },
        {
            'query': 'nike running shoes',
            'context': {
                'persona': 'sports_enthusiast', 
                'persona_details': {
                    'previous_queries': ['running', 'athletic', 'sports'],
                    'clicked_brands': ['Nike', 'Adidas'],
                    'clicked_categories': ['Sports Shoes', 'Athletic Wear']
                },
                'location': 'Mumbai'
            }
        }
    ]
    
    for test in test_cases:
        print(f"\n🔍 Query: '{test['query']}' | Persona: {test['context']['persona']}")
        results = searcher.search_with_persona(test['query'], test['context'], top_k=5)
        
        for result in results:
            print(f"   • {result.get('title', 'N/A')} - {result.get('brand', 'N/A')}")
            print(f"     Score: {result.get('persona_score', 0):.3f} | Boosts: {result.get('boosts_applied', [])}")

if __name__ == "__main__":
    main()