#!/usr/bin/env python3
"""
Simplified Search System for Frontend Integration
================================================

A lightweight version of the HybridSearcher that works with the frontend.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_search_dependencies():
    """Check if search dependencies are available."""
    required_packages = {
        'sentence_transformers': 'sentence-transformers',
        'faiss': 'faiss-cpu', 
        'numpy': 'numpy',
        'pandas': 'pandas'
    }
    
    missing_packages = []
    available_packages = []
    
    for package, pip_name in required_packages.items():
        try:
            if package == 'faiss':
                import faiss
                available_packages.append(package)
            elif package == 'sentence_transformers':
                from sentence_transformers import SentenceTransformer
                available_packages.append(package)
            else:
                __import__(package)
                available_packages.append(package)
        except ImportError:
            missing_packages.append(pip_name)
    
    return len(missing_packages) == 0, available_packages, missing_packages

class SimplifiedSearcher:
    """
    Simplified search system that uses FAISS for semantic product search.
    """
    
    def __init__(self):
        self.faiss_index = None
        self.sbert_model = None
        self.product_ids = None
        self.metadata = None
        self.model_name = 'all-MiniLM-L6-v2'
        self.search_ready = False
        
        # Paths relative to project root
        self.faiss_index_dir = "../searchresultpage/faiss_index"
        self.product_catalog_path = "../dataset/product_catalog_merged.csv"
        
        logger.info("🔧 Initializing Simplified Search System...")
        
        # Check dependencies
        deps_ok, available, missing = check_search_dependencies()
        if not deps_ok:
            logger.warning(f"⚠️ Missing dependencies: {', '.join(missing)}")
            logger.warning("💡 Install with: pip install " + " ".join(missing))
            return
        
        logger.info(f"✅ Dependencies available: {', '.join(available)}")
        
        # Load components
        self._load_components()
    
    def _load_components(self):
        """Load FAISS index and SBERT model."""
        try:
            self._load_faiss_index()
            self._load_sbert_model()
            self._load_metadata()
            self.search_ready = True
            logger.info("✅ Simplified Search System ready!")
        except Exception as e:
            logger.error(f"❌ Failed to load search components: {e}")
            self.search_ready = False
    
    def _load_faiss_index(self):
        """Load FAISS index."""
        try:
            import faiss
            
            index_path = os.path.join(self.faiss_index_dir, "product_index.faiss")
            mapping_path = os.path.join(self.faiss_index_dir, "product_id_mapping.json")
            
            if not os.path.exists(index_path) or not os.path.exists(mapping_path):
                raise FileNotFoundError(f"FAISS index files not found in {self.faiss_index_dir}")
            
            # Load FAISS index
            self.faiss_index = faiss.read_index(index_path)
            logger.info(f"✅ FAISS index loaded: {self.faiss_index.ntotal:,} products")
            
            # Load product ID mapping
            with open(mapping_path, 'r', encoding='utf-8') as f:
                self.product_ids = json.load(f)
            logger.info(f"✅ Product ID mapping loaded: {len(self.product_ids):,} products")
            
        except Exception as e:
            logger.error(f"❌ Failed to load FAISS index: {e}")
            raise
    
    def _load_sbert_model(self):
        """Load SBERT model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            self.sbert_model = SentenceTransformer(self.model_name)
            logger.info(f"✅ SBERT model loaded: {self.model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load SBERT model: {e}")
            raise
    
    def _load_metadata(self):
        """Load product metadata."""
        try:
            metadata_path = os.path.join(self.faiss_index_dir, "product_metadata.json")
            
            if not os.path.exists(metadata_path):
                logger.warning(f"⚠️ Metadata file not found: {metadata_path}")
                self.metadata = {}
                return
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info(f"✅ Product metadata loaded: {len(self.metadata):,} products")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load metadata: {e}")
            self.metadata = {}
    
    def _extract_price_constraints(self, query: str) -> Dict[str, Optional[float]]:
        """Extract price constraints from query."""
        constraints = {'min_price': None, 'max_price': None}
        
        # Patterns for price extraction
        patterns = [
            r'under\s+(?:rs\.?\s*|₹\s*)?(\d+(?:,\d+)*)',
            r'below\s+(?:rs\.?\s*|₹\s*)?(\d+(?:,\d+)*)',
            r'less\s+than\s+(?:rs\.?\s*|₹\s*)?(\d+(?:,\d+)*)',
            r'above\s+(?:rs\.?\s*|₹\s*)?(\d+(?:,\d+)*)',
            r'over\s+(?:rs\.?\s*|₹\s*)?(\d+(?:,\d+)*)',
            r'more\s+than\s+(?:rs\.?\s*|₹\s*)?(\d+(?:,\d+)*)',
            r'between\s+(?:rs\.?\s*|₹\s*)?(\d+(?:,\d+)*)\s+(?:and|to)\s+(?:rs\.?\s*|₹\s*)?(\d+(?:,\d+)*)',
            r'(?:rs\.?\s*|₹\s*)?(\d+(?:,\d+)*)\s*(?:to|-)\s*(?:rs\.?\s*|₹\s*)?(\d+(?:,\d+)*)'
        ]
        
        query_lower = query.lower()
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                if 'under' in pattern or 'below' in pattern or 'less' in pattern:
                    constraints['max_price'] = float(match.group(1).replace(',', ''))
                elif 'above' in pattern or 'over' in pattern or 'more' in pattern:
                    constraints['min_price'] = float(match.group(1).replace(',', ''))
                elif 'between' in pattern or 'to' in pattern or '-' in pattern:
                    price1 = float(match.group(1).replace(',', ''))
                    price2 = float(match.group(2).replace(',', ''))
                    constraints['min_price'] = min(price1, price2)
                    constraints['max_price'] = max(price1, price2)
                break
        
        return constraints
    
    def _semantic_search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Perform semantic search using FAISS."""
        if not self.search_ready:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.sbert_model.encode([query], normalize_embeddings=True)
            
            # Search FAISS index
            scores, indices = self.faiss_index.search(
                query_embedding.astype('float32'), 
                min(top_k * 2, len(self.product_ids))  # Get more for filtering
            )
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # Invalid index
                    continue
                
                product_id = self.product_ids[idx]
                
                # Get metadata
                product_data = self.metadata.get(product_id, {})
                
                result = {
                    'product_id': product_id,
                    'title': product_data.get('title', 'Unknown Product'),
                    'brand': product_data.get('brand', 'Unknown Brand'),
                    'category': product_data.get('category', 'General'),
                    'subcategory': product_data.get('subcategory', ''),
                    'price': float(product_data.get('price', 0)),
                    'rating': float(product_data.get('rating', 0)),
                    'color': product_data.get('color', ''),
                    'is_f_assured': bool(product_data.get('is_f_assured', False)),
                    'description': product_data.get('description', '')[:200],
                    'image_url': product_data.get('image_url', ''),
                    'tags': product_data.get('tags', ''),
                    'search_score': float(score),
                    'search_method': 'semantic_search',
                    'rank': i + 1
                }
                
                results.append(result)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            return []
    
    def _apply_price_filter(self, results: List[Dict[str, Any]], constraints: Dict[str, Optional[float]]) -> List[Dict[str, Any]]:
        """Apply price filtering to results."""
        if not constraints['min_price'] and not constraints['max_price']:
            return results
        
        filtered_results = []
        for result in results:
            price = result.get('price', 0)
            
            # Apply price constraints
            if constraints['min_price'] and price < constraints['min_price']:
                continue
            if constraints['max_price'] and price > constraints['max_price']:
                continue
            
            filtered_results.append(result)
        
        logger.info(f"🔍 Price filter: {len(results)} → {len(filtered_results)} results")
        return filtered_results
    
    def search(self, query: str, top_k: int = 20, user_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Main search function.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            user_context: Optional user context (not used in simplified version)
        
        Returns:
            List of product results
        """
        if not query or not query.strip():
            return []
        
        if not self.search_ready:
            logger.warning("⚠️ Search system not ready")
            return [{
                'product_id': 'system_error',
                'title': 'Search system not available',
                'brand': 'System',
                'category': 'Error',
                'price': 0,
                'rating': 0,
                'is_f_assured': False,
                'search_method': 'error',
                'message': 'Search dependencies not installed. Run: pip install sentence-transformers faiss-cpu'
            }]
        
        logger.info(f"🔍 Searching for: '{query}' (top {top_k})")
        
        try:
            # Extract price constraints
            price_constraints = self._extract_price_constraints(query)
            
            # Perform semantic search (get more results if we need to filter by price)
            search_k = top_k * 3 if any(price_constraints.values()) else top_k
            results = self._semantic_search(query, search_k)
            
            # Apply price filtering if constraints found
            if any(price_constraints.values()):
                results = self._apply_price_filter(results, price_constraints)
            
            # Limit to requested number
            final_results = results[:top_k]
            
            logger.info(f"✅ Found {len(final_results)} results for '{query}'")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return [{
                'product_id': 'search_error',
                'title': 'Search failed',
                'brand': 'System',
                'category': 'Error', 
                'price': 0,
                'rating': 0,
                'is_f_assured': False,
                'search_method': 'error',
                'error': str(e)
            }]

# Global searcher instance
_searcher_instance = None

def get_searcher():
    """Get or create the global searcher instance."""
    global _searcher_instance
    if _searcher_instance is None:
        _searcher_instance = SimplifiedSearcher()
    return _searcher_instance

def search_products(query: str, top_k: int = 20, user_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Convenience function for product search."""
    searcher = get_searcher()
    return searcher.search(query, top_k, user_context)