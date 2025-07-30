# test_feature_extraction.py
"""
Grid 7.0 - Feature Extraction Test Script
=========================================

This script tests the feature extraction system to ensure all products
in search results contain the required features.
"""

import requests
import json
import time
from typing import List, Dict, Any

# Required features that should be present in all products
REQUIRED_FEATURES = [
    'persona_tag', 'avg_price_last_k_clicks', 'preferred_brands_count',
    'session_length', 'query_frequency', 'brand', 'price', 'rating',
    'click_count', 'is_f_assured', 'brand_match', 'price_gap_to_avg',
    'offer_preference_match', 'semantic_similarity',
    'query_intent_similarity', 'product_embedding_mean', 'event'
]

def test_feature_extraction():
    """Test the feature extraction system."""
    print("🧪 Testing Feature Extraction System")
    print("=" * 60)
    
    # API endpoint
    api_url = "http://localhost:8000/hybrid_search/"
    
    # Test queries
    test_queries = [
        "samsung smartphone",
        "nike running shoes",
        "laptop for gaming",
        "wireless headphones",
        "smart tv 4k"
    ]
    
    # User context for testing
    user_context = {
        "user_id": "test_user_123",
        "location": "Mumbai",
        "price_range": "premium",
        "session_id": "test_session_456"
    }
    
    total_tests = 0
    passed_tests = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: '{query}'")
        print("-" * 40)
        
        try:
            # Make API request
            request_data = {
                "query": query,
                "context": user_context,
                "top_k": 10
            }
            
            response = requests.post(api_url, json=request_data, timeout=30)
            
            if response.status_code == 200:
                results = response.json()
                total_tests += 1
                
                if results:
                    print(f"✅ Found {len(results)} products")
                    
                    # Check if all products have required features
                    missing_features_products = []
                    
                    for j, product in enumerate(results):
                        missing_features = []
                        for feature in REQUIRED_FEATURES:
                            if feature not in product:
                                missing_features.append(feature)
                        
                        if missing_features:
                            missing_features_products.append({
                                'product_id': product.get('product_id', f'product_{j}'),
                                'missing_features': missing_features
                            })
                    
                    if missing_features_products:
                        print(f"❌ {len(missing_features_products)} products missing features:")
                        for item in missing_features_products:
                            print(f"   {item['product_id']}: {item['missing_features']}")
                    else:
                        print("✅ All products have required features!")
                        passed_tests += 1
                        
                        # Show sample product features
                        sample_product = results[0]
                        print("\n📋 Sample product features:")
                        for feature in REQUIRED_FEATURES:
                            value = sample_product.get(feature, 'N/A')
                            print(f"   {feature}: {value}")
                    
                else:
                    print("⚠️ No results found")
                    
            else:
                print(f"❌ API request failed: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Test Summary:")
    print(f"   Total tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {total_tests - passed_tests}")
    print(f"   Success rate: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "   Success rate: 0%")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Feature extraction is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the feature extraction system.")

def test_api_health():
    """Test API health endpoint."""
    print("\n🏥 Testing API Health")
    print("-" * 30)
    
    try:
        response = requests.get("http://localhost:8000/health/", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API Health Check:")
            for key, value in health_data.items():
                print(f"   {key}: {value}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")

def test_api_stats():
    """Test API stats endpoint."""
    print("\n📊 Testing API Stats")
    print("-" * 30)
    
    try:
        response = requests.get("http://localhost:8000/stats/", timeout=10)
        if response.status_code == 200:
            stats_data = response.json()
            print("✅ API Stats:")
            print(f"   Hybrid Searcher: {stats_data.get('hybrid_searcher', {})}")
            print(f"   Product Ranker: {stats_data.get('product_ranker', {})}")
            print(f"   Required Features: {len(stats_data.get('required_features', []))}")
        else:
            print(f"❌ Stats check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Stats check error: {e}")

def main():
    """Main test function."""
    print("🚀 Starting Feature Extraction Tests")
    print("=" * 60)
    
    # Test API health first
    test_api_health()
    
    # Test API stats
    test_api_stats()
    
    # Test feature extraction
    test_feature_extraction()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main() 