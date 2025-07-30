#!/usr/bin/env python3
"""
Test script for the search API
"""

import requests
import json

def test_search():
    """Test the search API with the problematic query."""
    
    # Test the problematic query
    query_data = {
        "query": "Women's Clothing",
        "context": {"location": "Pune"},
        "top_k": 10
    }
    
    try:
        print("🔍 Testing search with query: 'Women's Clothing'")
        print("=" * 60)
        
        response = requests.post(
            "http://localhost:8000/search",
            json=query_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Search successful! Found {len(results)} results")
            print("\n📋 Top 5 results:")
            print("-" * 60)
            
            for i, result in enumerate(results[:5], 1):
                print(f"{i}. {result.get('title', 'N/A')}")
                print(f"   Category: {result.get('category', 'N/A')}")
                print(f"   Brand: {result.get('brand', 'N/A')}")
                print(f"   Search Method: {result.get('search_method', 'N/A')}")
                if 'entities_used' in result:
                    print(f"   Entities Used: {result['entities_used']}")
                print()
        else:
            print(f"❌ Search failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the search API. Make sure the server is running on localhost:8000")
    except Exception as e:
        print(f"❌ Error testing search: {e}")

if __name__ == "__main__":
    test_search() 