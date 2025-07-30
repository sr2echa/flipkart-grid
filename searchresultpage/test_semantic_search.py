#!/usr/bin/env python3
"""
Test Semantic Search with Expanded Dataset
=========================================

This script tests the semantic search functionality with the merged dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hybrid_search import HybridSearcher

def test_semantic_search():
    """Test semantic search with various queries."""
    
    # Configuration
    SPACY_MODEL_PATH = "R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\searchresultpage\\spacy_ner_model"
    FAISS_INDEX_DIR = "./faiss_index"
    PRODUCT_CATALOG_PATH = "R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\dataset\\product_catalog_merged.csv"
    
    print("🧪 Testing Semantic Search with Expanded Dataset")
    print("=" * 60)
    
    try:
        # Initialize searcher
        print("🔧 Initializing search system...")
        searcher = HybridSearcher(
            spacy_model_path=SPACY_MODEL_PATH,
            faiss_index_dir=FAISS_INDEX_DIR,
            product_catalog_path=PRODUCT_CATALOG_PATH
        )
        
        # Test queries
        test_queries = [
            "Women's Clothing",
            "Men's Shoes",
            "Laptop for gaming",
            "Bluetooth headphones",
            "Smartphone under 20000",
            "Kitchen appliances",
            "Beauty products",
            "Sports equipment"
        ]
        
        print("\n🔍 Testing queries:")
        print("-" * 60)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            print("-" * 40)
            
            # Perform search
            results = searcher.search(query, top_k=5)
            
            if results:
                print(f"✅ Found {len(results)} results")
                for j, result in enumerate(results[:3], 1):  # Show top 3
                    print(f"   {j}. {result.get('title', 'N/A')}")
                    print(f"      Category: {result.get('category', 'N/A')}")
                    print(f"      Brand: {result.get('brand', 'N/A')}")
                    if 'similarity_score' in result:
                        print(f"      Similarity: {result['similarity_score']:.3f}")
                    print()
            else:
                print("❌ No results found")
        
        print("\n✅ Semantic search test completed!")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_semantic_search() 