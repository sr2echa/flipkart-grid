#!/usr/bin/env python3
"""
Test the advanced search functionality directly
"""

import json
import requests

def test_search():
    """Test the search API with the advanced format."""
    url = "http://localhost:3000/api/search"
    
    # Test data matching your example
    test_data = {
        "query": "iphone under 50000",
        "context": {
            "location": "Pune", 
            "persona_tag": "brand_lover"
        },
        "top_k": 10
    }
    
    try:
        response = requests.post(url, json=test_data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Got {len(results)} results")
            
            if results:
                print("\n📊 First result:")
                first_result = results[0]
                for key, value in first_result.items():
                    print(f"  {key}: {value}")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        print("Make sure the server is running on port 3000")

if __name__ == "__main__":
    test_search()