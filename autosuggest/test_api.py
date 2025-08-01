#!/usr/bin/env python3
"""
Test script for the autosuggest API to verify it's working correctly.
"""

import requests
import json
import time

def test_api():
    """Test the /api/suggest endpoint with various scenarios."""
    base_url = "http://127.0.0.1:5000"
    
    # Wait for server to start
    print("🚀 Testing Autosuggest API...")
    
    # Test health check
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        print(f"✅ Health check: {response.json()}")
    except:
        print("❌ Server not ready yet...")
        return
    
    # Test cases
    test_cases = [
        {
            "name": "Sports Enthusiast in Chennai during IPL - 'jersy'",
            "payload": {
                "query": "jersy",
                "persona": "sports_enthusiast", 
                "location": "Chennai",
                "event": "ipl"
            }
        },
        {
            "name": "Tech Enthusiast in Bangalore - 'laptop'",
            "payload": {
                "query": "laptop",
                "persona": "tech_enthusiast",
                "location": "Bangalore", 
                "event": "none"
            }
        },
        {
            "name": "Fashion Lover - 'nike'",
            "payload": {
                "query": "nike",
                "persona": "fashion_lover",
                "location": "Mumbai",
                "event": "none"
            }
        },
        {
            "name": "Typo correction - 'samsng'",
            "payload": {
                "query": "samsng",
                "persona": "tech_enthusiast",
                "location": "Delhi",
                "event": "none"
            }
        },
        {
            "name": "Diwali context - 'lights'",
            "payload": {
                "query": "lights",
                "persona": "tech_enthusiast",
                "location": "Mumbai",
                "event": "diwali"
            }
        }
    ]
    
    print("\n=== API Test Results ===")
    
    for test_case in test_cases:
        print(f"\n🧪 Test: {test_case['name']}")
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/api/suggest",
                json=test_case['payload'],
                timeout=10
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                suggestions = [s['text'] for s in data['suggestions']]
                response_time = data.get('response_time_ms', 0)
                
                print(f"✅ Status: SUCCESS")
                print(f"📝 Query: '{test_case['payload']['query']}'")
                print(f"💡 Suggestions: {suggestions}")
                print(f"⚡ Response Time: {response_time}ms")
                print(f"🏷️ Context: {test_case['payload']['persona']}, {test_case['payload']['location']}, {test_case['payload']['event']}")
                
                if len(suggestions) > 0:
                    print(f"✅ Got {len(suggestions)} suggestions")
                else:
                    print(f"⚠️ No suggestions returned")
                    
            else:
                print(f"❌ Status: ERROR {response.status_code}")
                print(f"📄 Response: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"⏰ TIMEOUT - Server took too long to respond")
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print("\n🎉 API Testing Complete!")

if __name__ == "__main__":
    test_api()