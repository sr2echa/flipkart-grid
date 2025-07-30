# hybrid_search.py

"""
Grid 7.0 - Hybrid Search System
===============================

This module combines spaCy NER-based entity extraction with FAISS semantic search
to provide intelligent product search capabilities.

The hybrid approach:
1. First attempts to extract entities (Brand, Color, Category, etc.) using spaCy NER
2. If entities are found, performs rule-based filtering on the product catalog
3. If no entities are found, falls back to FAISS semantic search
"""

import os
import json
import pickle
import time
import logging
import spacy
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required packages are available."""
    required_packages = {
        'sentence_transformers': 'sentence-transformers',
        'faiss': 'faiss-cpu',
        'numpy': 'numpy',
        'spacy': 'spacy',
        'pandas': 'pandas'
    }
    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            if package == 'faiss':
                import faiss
            else:
                __import__(package)
            logger.info(f"✅ {package} is available")
        except ImportError:
            logger.error(f"❌ {package} is missing. Please install: pip install {pip_name}")
            missing_packages.append(pip_name)
    
    if missing_packages:
        return False
    return True

class HybridSearcher:
    """
    Semantic search system that uses FAISS for optimal product discovery.
    Now primarily uses semantic search for better results.
    """
    
    def __init__(self, 
                 spacy_model_path: str,
                 faiss_index_dir: str = "./faiss_index",
                 product_catalog_path: Optional[str] = None):
        """
        Initialize the hybrid search system.
        
        Args:
            spacy_model_path: Path to trained spaCy NER model
            faiss_index_dir: Directory containing FAISS index files
            product_catalog_path: Path to product catalog CSV (optional)
        """
        self.spacy_model_path = spacy_model_path
        self.faiss_index_dir = faiss_index_dir
        self.product_catalog_path = product_catalog_path
        
        # Model instances
        self.nlp = None
        self.faiss_index = None
        self.sbert_model = None
        self.product_ids = None
        self.metadata = None
        self.product_catalog = None
        self.stats = None
        self.model_name = 'all-MiniLM-L6-v2'
        
        logger.info("🔧 Grid 7.0 - Semantic Search System Initializing")
        logger.info("=" * 60)
        
        if not check_dependencies():
            raise RuntimeError("Required dependencies are missing. Please install them.")
        
        self._load_all_components()
        
        logger.info("✅ Hybrid Search System Ready!")
        logger.info(f"📊 Loaded {len(self.product_ids):,} products for semantic search")
        if self.product_catalog is not None:
            logger.info(f"📊 Loaded {len(self.product_catalog):,} products for rule-based filtering")
        logger.info("=" * 60)
    
    def _load_spacy_model(self):
        """Load the trained spaCy NER model."""
        logger.info(f"🧠 Loading spaCy NER model from: {self.spacy_model_path}")
        try:
            self.nlp = spacy.load(self.spacy_model_path)
            logger.info("✅ spaCy NER model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load spaCy model: {e}")
            raise RuntimeError(f"Failed to load spaCy model: {e}")
    
    def _load_faiss_components(self):
        """Load FAISS index, mappings, metadata, and SBERT model."""
        logger.info(f"📂 Loading FAISS components from: {self.faiss_index_dir}")
        
        if not os.path.exists(self.faiss_index_dir):
            raise FileNotFoundError(f"FAISS index directory not found: {self.faiss_index_dir}")

        # Load stats to get model name
        stats_path = os.path.join(self.faiss_index_dir, "embedding_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, 'r', encoding='utf-8') as f:
                self.stats = json.load(f)
            self.model_name = self.stats.get('model_name', self.model_name)
            logger.info(f"✅ Build stats loaded - Model: {self.model_name}")

        # Load SBERT model
        from sentence_transformers import SentenceTransformer
        logger.info(f"🧠 Loading SBERT model: {self.model_name}")
        self.sbert_model = SentenceTransformer(self.model_name)

        # Load FAISS index
        import faiss
        index_path = os.path.join(self.faiss_index_dir, "product_index.faiss")
        self.faiss_index = faiss.read_index(index_path)
        logger.info(f"✅ FAISS index loaded with {self.faiss_index.ntotal:,} vectors")

        # Load product ID mapping
        mapping_path = os.path.join(self.faiss_index_dir, "product_id_mapping.json")
        with open(mapping_path, 'r', encoding='utf-8') as f:
            self.product_ids = json.load(f)
        logger.info(f"✅ Product ID mapping loaded for {len(self.product_ids):,} products")

        # Load metadata
        metadata_path = os.path.join(self.faiss_index_dir, "product_metadata.pkl")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        logger.info(f"✅ Product metadata loaded for {len(self.metadata):,} products")
    
    def _load_product_catalog(self):
        """Load product catalog for rule-based filtering (optional)."""
        if self.product_catalog_path and os.path.exists(self.product_catalog_path):
            logger.info(f"📊 Loading product catalog from: {self.product_catalog_path}")
            try:
                self.product_catalog = pd.read_csv(self.product_catalog_path)
                logger.info(f"✅ Product catalog loaded with {len(self.product_catalog):,} products")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load product catalog: {e}")
                self.product_catalog = None
        else:
            logger.info("ℹ️ No product catalog provided - rule-based filtering disabled")
    
    def _load_feature_extractor(self):
        """Load the feature extraction system."""
        logger.info("🔧 Loading Feature Extractor...")
        try:
            from feature_extraction import FeatureExtractor
            self.feature_extractor = FeatureExtractor(
                session_log_path="R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\dataset\\session_log.csv",
                user_queries_path="R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\dataset\\user_queries.csv",
                realtime_data_path="R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\dataset\\realtime_product_info.csv"
            )
            logger.info("✅ Feature Extractor loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load Feature Extractor: {e}")
            self.feature_extractor = None
    
    def _load_all_components(self):
        """Load all components: spaCy NER, FAISS, and product catalog."""
        self._load_spacy_model()
        self._load_faiss_components()
        self._load_product_catalog()
        self._load_feature_extractor()
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text using spaCy NER model."""
        if not self.nlp:
            raise RuntimeError("spaCy model not loaded")
        
        doc = self.nlp(text)
        entities = {}
        for ent in doc.ents:
            entities.setdefault(ent.label_, []).append(ent.text)
        
        return entities
    
    def _enhance_entities_with_categories(self, text: str, entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Enhance entity extraction by detecting category-like terms that the NER model might miss.
        """
        category_keywords = {
            'clothing': ['clothing', 'apparel', 'fashion', 'wear', 'dress', 'shirt', 'pants', 'jeans', 't-shirt', 'tshirt'],
            'footwear': ['shoes', 'footwear', 'sneakers', 'boots', 'sandals', 'flip-flops'],
            'electronics': ['electronics', 'mobile', 'phone', 'smartphone', 'laptop', 'computer', 'tv', 'television'],
            'accessories': ['accessories', 'jewelry', 'watch', 'bag', 'wallet', 'belt'],
            'sports': ['sports', 'fitness', 'gym', 'exercise', 'athletic'],
            'home': ['home', 'kitchen', 'furniture', 'decor', 'appliances'],
            'beauty': ['beauty', 'cosmetics', 'makeup', 'skincare', 'personal care'],
            'books': ['books', 'stationery', 'pens', 'pencils', 'notebooks'],
            'toys': ['toys', 'games', 'puzzles', 'educational'],
            'automotive': ['automotive', 'car', 'bike', 'vehicle', 'accessories']
        }
        
        text_lower = text.lower()
        enhanced_entities = entities.copy()
        
        # Check for category keywords in the text
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # If this keyword wasn't already extracted as an entity
                    if not any(keyword in entity.lower() for entity_list in entities.values() for entity in entity_list):
                        if 'CATEGORY' not in enhanced_entities:
                            enhanced_entities['CATEGORY'] = []
                        enhanced_entities['CATEGORY'].append(category)
                        logger.info(f"🎯 Enhanced entity extraction: detected '{category}' as CATEGORY")
                        break
        
        return enhanced_entities
    
    def _filter_by_entities(self, entities: Dict[str, List[str]], top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Filter products using extracted entities (rule-based approach).
        This is a simplified implementation - customize based on your catalog structure.
        """
        if self.product_catalog is None:
            logger.warning("⚠️ Product catalog not available for rule-based filtering")
            return []
        
        logger.info(f"🔍 Filtering products using entities: {entities}")
        
        # Start with all products
        filtered_products = self.product_catalog.copy()
        
        # Apply filters based on entities
        if 'BRAND' in entities:
            brands = [brand.lower() for brand in entities['BRAND']]
            if 'brand' in filtered_products.columns:
                filtered_products = filtered_products[
                    filtered_products['brand'].str.lower().isin(brands)
                ]
        
        if 'COLOR' in entities:
            colors = [color.lower() for color in entities['COLOR']]
            # Assuming color info is in title or a color column
            color_filter = filtered_products['title'].str.lower().str.contains('|'.join(colors), na=False)
            filtered_products = filtered_products[color_filter]
        
        # Handle FEATURE entities that might actually be categories
        if 'FEATURE' in entities:
            features = [feature.lower() for feature in entities['FEATURE']]
            
            # Check if any features match category terms
            category_keywords = ['clothing', 'apparel', 'fashion', 'wear', 'dress', 'shirt', 'pants', 'jeans', 'shoes', 'footwear', 'accessories', 'electronics', 'mobile', 'laptop', 'tv', 'headphones', 'speakers', 'camera', 'gaming', 'sports', 'fitness', 'home', 'kitchen', 'beauty', 'health', 'books', 'toys', 'automotive']
            
            category_matches = []
            feature_matches = []
            
            for feature in features:
                if feature in category_keywords:
                    category_matches.append(feature)
                else:
                    feature_matches.append(feature)
            
            # Apply category filtering if category matches found
            if category_matches and 'category' in filtered_products.columns:
                category_filter = filtered_products['category'].str.lower().str.contains('|'.join(category_matches), na=False)
                filtered_products = filtered_products[category_filter]
                logger.info(f"🎯 Applied category filtering for: {category_matches}")
            
            # Apply feature filtering for remaining features
            if feature_matches:
                feature_filter = filtered_products['title'].str.lower().str.contains('|'.join(feature_matches), na=False)
                filtered_products = filtered_products[feature_filter]
                logger.info(f"🔧 Applied feature filtering for: {feature_matches}")
        
        # Handle CATEGORY entities (in case model is retrained later)
        if 'CATEGORY' in entities:
            categories = [cat.lower() for cat in entities['CATEGORY']]
            if 'category' in filtered_products.columns:
                category_filter = filtered_products['category'].str.lower().str.contains('|'.join(categories), na=False)
                filtered_products = filtered_products[category_filter]
        
        # Sort by rating or popularity (customize based on available columns)
        if 'rating' in filtered_products.columns:
            filtered_products = filtered_products.sort_values('rating', ascending=False)
        elif 'price' in filtered_products.columns:
            filtered_products = filtered_products.sort_values('price', ascending=True)
        
        # Convert to results format
        results = []
        for i, (_, row) in enumerate(filtered_products.head(top_k).iterrows()):
            results.append({
                'rank': i + 1,
                'product_id': row.get('product_id', row.get('id', f'rule_based_{i}')),
                'title': row.get('title', 'N/A'),
                'brand': row.get('brand', 'N/A'),
                'category': row.get('category', 'N/A'),
                'price': row.get('price', 0),
                'search_method': 'rule_based_filtering',
                'entities_used': entities
            })
        
        logger.info(f"📋 Rule-based filtering found {len(results)} products")

        # logger.info(f"Filtered products after NER: {filtered_products[['product_id', 'title', 'price']].head(10)}")
        return results

    def _extract_product_price(self, product: Dict[str, Any]) -> Optional[float]:
        """Enhanced price extraction from product data."""
        # Try multiple price field names
        price_fields = ['price', 'final_price', 'selling_price', 'mrp', 'cost', 'amount']
        
        for field in price_fields:
            if field in product:
                price_value = product[field]
                
                # Handle numeric prices
                if isinstance(price_value, (int, float)):
                    if price_value > 0:
                        return float(price_value)
                
                # Handle string prices
                elif isinstance(price_value, str) and price_value.strip():
                    try:
                        # Remove currency symbols and commas
                        cleaned_price = re.sub(r'[₹,\s]', '', price_value.strip())
                        cleaned_price = re.sub(r'rs\.?', '', cleaned_price, flags=re.IGNORECASE)
                        
                        if cleaned_price:
                            price = float(cleaned_price)
                            if price > 0:
                                return price
                    except (ValueError, TypeError):
                        continue
        
        # Try to extract from title as last resort
        title = product.get('title', '')
        if title:
            price_patterns = [
                r'₹\s*(\d+(?:,\d{3})*(?:\.\d+)?)',
                r'rs\.?\s*(\d+(?:,\d{3})*(?:\.\d+)?)',
                r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*rs',
            ]
            
            for pattern in price_patterns:
                matches = re.findall(pattern, title.lower())
                if matches:
                    try:
                        price = float(matches[0].replace(',', ''))
                        if price > 0:
                            return price
                    except (ValueError, IndexError):
                        continue
        
        return None

    
    def _extract_price_constraints(self, query: str) -> Dict[str, Any]:
        """
        Enhanced price constraint extraction with better patterns and debugging.
        """
        query_lower = query.lower().strip()
        logger.info(f"🔍 Extracting price constraints from: '{query_lower}'")
        
        # Improved regex patterns with better matching
        price_patterns = [
            # Under/Below patterns - more comprehensive
            (r'under\s+(?:rs\.?\s*|₹\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)', 'under'),
            (r'below\s+(?:rs\.?\s*|₹\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)', 'under'),
            (r'less\s+than\s+(?:rs\.?\s*|₹\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)', 'under'),
            (r'upto\s+(?:rs\.?\s*|₹\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)', 'under'),
            (r'up\s+to\s+(?:rs\.?\s*|₹\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)', 'under'),
            (r'within\s+(?:rs\.?\s*|₹\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)', 'under'),
            
            # Over/Above patterns
            (r'over\s+(?:rs\.?\s*|₹\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)', 'over'),
            (r'above\s+(?:rs\.?\s*|₹\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)', 'over'),
            (r'more\s+than\s+(?:rs\.?\s*|₹\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)', 'over'),
            
            # Range patterns
            (r'between\s+(?:rs\.?\s*|₹\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:and|to|-)\s*(?:rs\.?\s*|₹\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)', 'range'),
            (r'(?:rs\.?\s*|₹\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:to|-)\s*(?:rs\.?\s*|₹\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)', 'range'),
        ]
        
        # Remove interfering words
        query_cleaned = re.sub(r'\b(price|cost|budget|affordable|cheap)\b', '', query_lower)
        
        for pattern, constraint_type in price_patterns:
            matches = re.findall(pattern, query_cleaned, re.IGNORECASE)
            
            if matches:
                logger.info(f"🎯 Pattern matched: {constraint_type} -> {matches}")
                
                try:
                    if constraint_type == 'under':
                        price = float(matches[0].replace(',', ''))
                        constraints = {'max_price': price, 'price_type': 'under'}
                        logger.info(f"💰 Under constraint extracted: {constraints}")
                        return constraints
                    
                    elif constraint_type == 'over':
                        price = float(matches[0].replace(',', ''))
                        constraints = {'min_price': price, 'price_type': 'over'}
                        logger.info(f"💰 Over constraint extracted: {constraints}")
                        return constraints
                    
                    elif constraint_type == 'range' and len(matches[0]) == 2:
                        min_price = float(matches[0][0].replace(',', ''))
                        max_price = float(matches[0][1].replace(',', ''))
                        constraints = {
                            'min_price': min_price, 
                            'max_price': max_price, 
                            'price_type': 'range'
                        }
                        logger.info(f"💰 Range constraint extracted: {constraints}")
                        return constraints
                
                except (ValueError, IndexError) as e:
                    logger.warning(f"⚠️ Failed to parse price from matches {matches}: {e}")
                    continue
        
        logger.info("ℹ️ No price constraints found in query")
        return {}
    
    def _filter_by_price(self, results: List[Dict[str, Any]], price_constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhanced price filtering with better price extraction and debugging."""
        if not price_constraints:
            logger.info("ℹ️ No price constraints to apply")
            return results
        
        logger.info(f"💰 Applying price filter: {price_constraints}")
        logger.info(f"📊 Starting with {len(results)} products")
        
        filtered_results = []
        debug_info = []
        
        for i, result in enumerate(results):
            # Enhanced price extraction
            price = self._extract_product_price(result)
            
            if price is None or price <= 0:
                debug_info.append(f"Product {i+1}: Invalid price - skipped")
                continue
            
            # Apply filtering based on constraint type
            constraint_type = price_constraints.get('price_type')
            include_product = False
            
            if constraint_type == 'under':
                max_price = price_constraints.get('max_price', float('inf'))
                include_product = price <= max_price
                debug_info.append(f"Product {i+1}: ₹{price} <= ₹{max_price} = {include_product}")
            
            elif constraint_type == 'over':
                min_price = price_constraints.get('min_price', 0)
                include_product = price >= min_price
                debug_info.append(f"Product {i+1}: ₹{price} >= ₹{min_price} = {include_product}")
            
            elif constraint_type == 'range':
                min_price = price_constraints.get('min_price', 0)
                max_price = price_constraints.get('max_price', float('inf'))
                include_product = min_price <= price <= max_price
                debug_info.append(f"Product {i+1}: ₹{min_price} <= ₹{price} <= ₹{max_price} = {include_product}")
            
            if include_product:
                # Add debugging info to result
                result['original_price'] = result.get('price')
                result['parsed_price'] = price
                result['price_filter_applied'] = price_constraints
                filtered_results.append(result)
        
        # Log first few comparisons for debugging
        for info in debug_info[:10]:
            logger.info(f"🔍 {info}")
        
        if len(debug_info) > 10:
            logger.info(f"... and {len(debug_info) - 10} more products checked")
        
        logger.info(f"✅ Price filtering completed: {len(results)} -> {len(filtered_results)} products")
        
        # If no results, log sample prices for debugging
        if len(filtered_results) == 0:
            logger.warning("⚠️ No products matched the price criteria!")
            sample_prices = []
            for result in results[:5]:
                price = self._extract_product_price(result)
                title = result.get('title', 'Unknown')[:30]
                if price:
                    sample_prices.append(f"{title}: ₹{price}")
            
            if sample_prices:
                logger.info("📊 Sample prices from original results:")
                for sample in sample_prices:
                    logger.info(f"   - {sample}")
        
        return filtered_results
    
    def _semantic_search(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Perform semantic search using FAISS with price filtering."""
        if not self.faiss_index or not self.sbert_model:
            raise RuntimeError("FAISS components not loaded")
        
        logger.info(f"🔍 Performing semantic search for: '{query}'")
        
        # Extract price constraints from query
        price_constraints = self._extract_price_constraints(query)
        
        # Enhance query for better semantic search
        enhanced_query = self._enhance_query_for_semantic_search(query)
        
        # Encode enhanced query and search FAISS (search more to account for filtering)
        search_k = top_k * 3 if price_constraints else top_k  # Search more if we need to filter
        query_embedding = self.sbert_model.encode([enhanced_query.strip()], normalize_embeddings=True)
        distances, indices = self.faiss_index.search(query_embedding.astype('float32'), search_k)
        
        # Prepare results
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx == -1:
                continue
            
            product_id = self.product_ids[idx]
            product_data = self.metadata.get(product_id, {})
            
            # Convert L2 distance to similarity score (0-1)
            similarity_score = max(0, 1 - (dist / 2))
            
            results.append({
                'rank': i + 1,
                'product_id': product_id,
                'title': product_data.get('title', 'N/A'),
                'brand': product_data.get('brand', 'N/A'),
                'category': product_data.get('category', 'N/A'),
                'price': product_data.get('price', 0),
                'similarity_score': round(similarity_score, 4),
                'search_method': 'semantic_search'
            })
        
        # Apply price filtering if constraints found
        if price_constraints:
            results = self._filter_by_price(results, price_constraints)
        
        # Limit to top_k results
        results = results[:top_k]
        
        logger.info(f"📋 Semantic search found {len(results)} products")
        return results
    
    def _enhance_query_for_semantic_search(self, query: str) -> str:
        """
        Enhance query with additional keywords for better semantic search results.
        """
        query_lower = query.lower()
        enhanced_query = query
        
        # Add mobile phone related keywords if query mentions mobile/phone
        if any(word in query_lower for word in ['mobile', 'phone', 'smartphone']):
            if 'children' in query_lower or 'kids' in query_lower or 'child' in query_lower:
                enhanced_query += " affordable budget cheap redmi xiaomi poco"
            else:
                enhanced_query += " smartphone mobile phone"
        
        # Add price-related keywords if price constraint is mentioned
        if any(word in query_lower for word in ['under', 'below', 'less than', 'cheap', 'budget', 'affordable']):
            enhanced_query += " affordable budget cheap low price"
        
        # Add brand keywords for better matching
        if 'redmi' in query_lower:
            enhanced_query += " xiaomi redmi"
        elif 'samsung' in query_lower:
            enhanced_query += " samsung galaxy"
        elif 'apple' in query_lower or 'iphone' in query_lower:
            enhanced_query += " apple iphone"
        
        if enhanced_query != query:
            logger.info(f"🔍 Enhanced query: '{query}' -> '{enhanced_query}'")
        
        return enhanced_query

    def search(self, query: str, top_k: int = 100, user_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Main hybrid search function that prioritizes semantic search with optional feature enrichment.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            user_context: User context for feature enrichment
            
        Returns:
            List of product results with comprehensive features
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")
        
        logger.info(f"🚀 Starting semantic search for: '{query}' (top {top_k})")
        start_time = time.time()
        
        # Step A: Extract price constraints first (fast operation)
        price_constraints = self._extract_price_constraints(query)
    
        # Step 2: Perform semantic search (get more results if we need to filter)
        search_k = top_k * 2 if price_constraints else top_k
        logger.info(f"🔍 Performing semantic search (retrieving {search_k} results for filtering)")
        
        # Step B: Extract entities using spaCy NER (for logging only)
        entities = self.extract_entities(query)
        if entities:
            logger.info(f"ℹ️ Entities detected (for reference): {entities}")
        
        # Step C: Use semantic search as primary method
        logger.info("🔍 Using semantic search as primary method")
        results = self._semantic_search(query, top_k)
        
        # Step D: Apply price filtering if constraints found
        if price_constraints and results:
            logger.info("💰 Applying price filtering...")
            filtered_results = self._filter_by_price(results, price_constraints)
            if filtered_results:
                results = filtered_results
                logger.info(f"✅ Price filtering applied: {len(results)} results remain")
            else:
                logger.warning("⚠️ No results match price constraints")
        
        # Step E: Enrich with features only if needed (performance optimization)
        if self.feature_extractor and len(results) > 0:
            # Only enrich if we have results and feature extraction is enabled
            logger.info("🔧 Enriching products with essential features...")
            
            # Limit feature extraction to top results for performance
            top_results = results[:min(20, len(results))]  # Only enrich top 20 results
            
            enriched_results = self.feature_extractor.extract_features_for_products(
                top_results, query, user_context
            )
            
            # Validate that essential features are present
            if not self.feature_extractor.validate_features(enriched_results):
                logger.warning("⚠️ Some products are missing required features")
            
            # Replace top results with enriched ones
            results[:len(enriched_results)] = enriched_results
        else:
            logger.info("⚡ Skipping feature enrichment for performance")
        
        search_time = time.time() - start_time
        logger.info(f"⚡ Semantic search completed in {search_time:.3f}s - Found {len(results)} products")
        
        return results

def main():
    """Demonstration of the hybrid search system."""
    print("🛒 Grid 7.0 - Hybrid Search System Demo")
    print("=" * 60)
    
    # Configuration
    SPACY_MODEL_PATH = "R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\searchresultpage\\spacy_ner_model"
    FAISS_INDEX_DIR = "./faiss_index"
    PRODUCT_CATALOG_PATH = "R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\dataset\\product_catalog_merged.csv"
    
    try:
        # Initialize hybrid searcher
        searcher = HybridSearcher(
            spacy_model_path=SPACY_MODEL_PATH,
            faiss_index_dir=FAISS_INDEX_DIR,
            product_catalog_path=PRODUCT_CATALOG_PATH
        )
        
        # Test queries
        test_queries = [
            "red running shoes size 9",  # Should extract entities: COLOR, CATEGORY, SIZE
            "bluetooth headphones noise cancelling",  # Should extract entities: CATEGORY, FEATURE
            "laptop for gaming",  # May not extract clear entities - fallback to semantic
            "mens leather wallet bifold",  # Should extract entities: GENDER, MATERIAL, CATEGORY
            "artificial intelligence machine learning",  # No product entities - semantic search
        ]
        
        print("🧪 Running test queries:")
        print("-" * 60)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test {i}: '{query}'")
            print("-" * 40)
            
            # Perform search
            results = searcher.search(query, top_k=5)
            
            if results:
                for result in results:
                    print(f"  {result['rank']}. {result['title']}")
                    print(f"     Brand: {result['brand']} | Price: ₹{result['price']}")
                    print(f"     Method: {result['search_method']}")
                    if 'similarity_score' in result:
                        print(f"     Similarity: {result['similarity_score']:.3f}")
                    if 'entities_used' in result:
                        print(f"     Entities: {result['entities_used']}")
                print()
            else:
                print("  No results found.")
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())