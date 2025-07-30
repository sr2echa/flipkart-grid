#!/usr/bin/env python3
"""
Test script to verify price filtering functionality.
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from hybrid_search import HybridSearcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_price_filtering():
    """Test the price filtering functionality."""
    
    # Configuration
    SPACY_MODEL_PATH = "R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\searchresultpage\\spacy_ner_model"
    FAISS_INDEX_DIR = "./faiss_index"
    PRODUCT_CATALOG_PATH = "R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\dataset\\product_catalog_merged.csv"
    
    try:
        # Initialize hybrid searcher
        logger.info("🚀 Initializing Hybrid Searcher...")
        searcher = HybridSearcher(
            spacy_model_path=SPACY_MODEL_PATH,
            faiss_index_dir=FAISS_INDEX_DIR,
            product_catalog_path=PRODUCT_CATALOG_PATH
        )
        
        # Test queries
        test_queries = [
            "mobile phones for children under 10000",
            "redmi mobile under 10000",
            "smartphone for kids below 8000",
            "mobile phones between 5000 and 15000",
            "expensive mobile phones over 50000"
        ]
        
        for query in test_queries:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 Testing query: '{query}'")
            logger.info(f"{'='*60}")
            
            # Test price constraint extraction
            price_constraints = searcher._extract_price_constraints(query)
            logger.info(f"💰 Extracted price constraints: {price_constraints}")
            
            # Test enhanced query
            enhanced_query = searcher._enhance_query_for_semantic_search(query)
            logger.info(f"🔍 Enhanced query: '{enhanced_query}'")
            
            # Perform search
            results = searcher.search(query, top_k=10)
            
            # Display results
            logger.info(f"📋 Found {len(results)} results:")
            for i, result in enumerate(results[:5]):  # Show first 5 results
                price = result.get('price', 0)
                if isinstance(price, str):
                    try:
                        price = float(price.replace(',', '').replace('₹', '').strip())
                    except:
                        price = 0
                
                logger.info(f"  {i+1}. {result.get('title', 'N/A')} - ₹{price} (Score: {result.get('similarity_score', 'N/A')})")
            
            # Verify price filtering
            if price_constraints:
                max_price = price_constraints.get('max_price', float('inf'))
                min_price = price_constraints.get('min_price', 0)
                
                if price_constraints.get('price_type') == 'under':
                    expensive_results = [r for r in results if r.get('price', 0) > max_price]
                    if expensive_results:
                        logger.warning(f"⚠️ Found {len(expensive_results)} results above ₹{max_price}")
                    else:
                        logger.info(f"✅ All results are under ₹{max_price}")
                elif price_constraints.get('price_type') == 'over':
                    cheap_results = [r for r in results if r.get('price', 0) < min_price]
                    if cheap_results:
                        logger.warning(f"⚠️ Found {len(cheap_results)} results below ₹{min_price}")
                    else:
                        logger.info(f"✅ All results are over ₹{min_price}")
        
        logger.info(f"\n{'='*60}")
        logger.info("✅ Price filtering test completed!")
        logger.info(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"❌ Error during testing: {e}")
        raise

if __name__ == "__main__":
    test_price_filtering() 