#!/usr/bin/env python3
"""
Test script for the new simplified autosuggest API
"""

import requests
import json
import time

def test_new_api():
    """Test the new simplified /api/suggest endpoint."""
    base_url = "http://127.0.0.1:5000"
    
    print("🚀 Testing New Simplified Autosuggest API...")
    
    # Wait for server to be ready
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        health_data = response.json()
        print(f"✅ Health check: {health_data}")
        if not health_data.get('loaded', False):
            print("⏳ Waiting for system to load...")
            time.sleep(5)
    except:
        print("❌ Server not ready yet...")
        return
    
    # Test cases demonstrating quality improvements
    test_cases = [
        {
            "name": "Single letter 'j' - should give jersey, jeans",
            "payload": {
                "query": "j",
                "persona": "sports_enthusiast", 
                "location": "Chennai",
                "event": "ipl"
            },
            "expected": ["jersey", "jeans"]
        },
        {
            "name": "Prefix 'jer' - should give jersey suggestions",
            "payload": {
                "query": "jer",
                "persona": "sports_enthusiast",
                "location": "Chennai", 
                "event": "ipl"
            },
            "expected": ["jersey"]
        },
        {
            "name": "Typo 'jersy' - should correct to 'jersey'",
            "payload": {
                "query": "jersy",
                "persona": "sports_enthusiast",
                "location": "Chennai",
                "event": "ipl"
            },
            "expected": ["jersey"]
        },
        {
            "name": "Tech query 'laptop' - should give laptop variations",
            "payload": {
                "query": "laptop",
                "persona": "tech_enthusiast",
                "location": "Bangalore",
                "event": "none"
            },
            "expected": ["laptop", "gaming laptop"]
        },
        {
            "name": "Brand prefix 'sam' - should give Samsung suggestions",
            "payload": {
                "query": "sam",
                "persona": "tech_enthusiast",
                "location": "Delhi",
                "event": "none"
            },
            "expected": ["samsung"]
        },
        {
            "name": "Diwali context 'lights' - should suggest diwali lights",
            "payload": {
                "query": "lights",
                "persona": "home_maker",
                "location": "Mumbai",
                "event": "diwali"
            },
            "expected": ["lights", "diwali lights"]
        }
    ]
    
    print("\n=== Quality Test Results ===")
    
    passed_tests = 0
    total_tests = len(test_cases)
    
    for test_case in test_cases:
        print(f"\n🧪 Test: {test_case['name']}")
        print(f"📝 Query: '{test_case['payload']['query']}'")
        print(f"🏷️ Context: {test_case['payload']['persona']}, {test_case['payload']['location']}, {test_case['payload']['event']}")
        
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
                
                print(f"💡 Suggestions: {suggestions}")
                print(f"⚡ Response Time: {response_time}ms")
                
                # Check if expected suggestions are present
                expected = test_case['expected']
                found_expected = any(exp in suggestions for exp in expected)
                
                if suggestions and found_expected:
                    print(f"✅ PASS - Found expected suggestions")
                    passed_tests += 1
                elif suggestions:
                    print(f"⚠️ PARTIAL - Got suggestions but not exactly expected ones")
                else:
                    print(f"❌ FAIL - No suggestions returned")
                    
            else:
                print(f"❌ FAIL - HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"⏰ TIMEOUT - Server took too long")
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print(f"\n🎯 Summary: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! The autosuggest system is working excellently!")
    elif passed_tests >= total_tests * 0.8:
        print("✅ Most tests passed! System quality is good.")
    else:
        print("⚠️ Some tests failed. System needs improvement.")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    test_new_api()