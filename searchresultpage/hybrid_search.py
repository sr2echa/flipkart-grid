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
    Hybrid search system that combines spaCy NER entity extraction 
    with FAISS semantic search for optimal product discovery.
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
        
        logger.info("🔧 Grid 7.0 - Hybrid Search System Initializing")
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
    
    def _load_all_components(self):
        """Load all components: spaCy NER, FAISS, and product catalog."""
        self._load_spacy_model()
        self._load_faiss_components()
        self._load_product_catalog()
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text using spaCy NER model."""
        if not self.nlp:
            raise RuntimeError("spaCy model not loaded")
        
        doc = self.nlp(text)
        entities = {}
        for ent in doc.ents:
            entities.setdefault(ent.label_, []).append(ent.text)
        
        return entities
    
    def _filter_by_entities(self, entities: Dict[str, List[str]], top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Filter products using extracted entities (rule-based approach).
        This is a simplified implementation - customize based on your catalog structure.
        """
        if not self.product_catalog is not None:
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
        return results
    
    def _semantic_search(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Perform semantic search using FAISS."""
        if not self.faiss_index or not self.sbert_model:
            raise RuntimeError("FAISS components not loaded")
        
        logger.info(f"🔍 Performing semantic search for: '{query}'")
        
        # Encode query and search FAISS
        query_embedding = self.sbert_model.encode([query.strip()], normalize_embeddings=True)
        distances, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        
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
        
        logger.info(f"📋 Semantic search found {len(results)} products")
        return results
    
    def search(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Main hybrid search function that combines NER and semantic search.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of product results with metadata
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")
        
        logger.info(f"🚀 Starting hybrid search for: '{query}' (top {top_k})")
        start_time = time.time()
        
        # Step A: Extract entities using spaCy NER
        entities = self.extract_entities(query)
        
        # Step B: Conditional search based on entity extraction
        if entities:
            # Entities found - use rule-based filtering
            logger.info(f"✅ Entities detected: {entities}")
            results = self._filter_by_entities(entities, top_k)
            
            # If rule-based filtering didn't find enough results, fall back to semantic search
            if len(results) < min(10, top_k):  # Minimum threshold
                logger.info(f"⚠️ Rule-based filtering found only {len(results)} products, falling back to semantic search")
                semantic_results = self._semantic_search(query, top_k)
                # Combine results (rule-based first, then semantic)
                combined_results = results + semantic_results
                # Remove duplicates and limit to top_k
                seen_ids = set()
                final_results = []
                for result in combined_results:
                    if result['product_id'] not in seen_ids:
                        seen_ids.add(result['product_id'])
                        final_results.append(result)
                        if len(final_results) >= top_k:
                            break
                results = final_results
        else:
            # No entities found - use semantic search
            logger.info("ℹ️ No entities detected, using semantic search")
            results = self._semantic_search(query, top_k)
        
        search_time = time.time() - start_time
        logger.info(f"⚡ Hybrid search completed in {search_time:.3f}s - Found {len(results)} products")
        
        return results

def main():
    """Demonstration of the hybrid search system."""
    print("🛒 Grid 7.0 - Hybrid Search System Demo")
    print("=" * 60)
    
    # Configuration
    SPACY_MODEL_PATH = "R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\searchresultpage\\spacy_ner_model"
    FAISS_INDEX_DIR = "./faiss_index"
    PRODUCT_CATALOG_PATH = None  # Add path to your product catalog CSV if available
    
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