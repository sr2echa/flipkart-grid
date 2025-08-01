#!/usr/bin/env python3
"""
Flipkart Frontend Demo Script
============================

This script demonstrates the integrated frontend functionality
by making sample API calls and showing the integration works.
"""

import requests
import json
import time
from typing import Dict, Any

class FrontendDemo:
    """Demo class to test the unified frontend functionality."""
    
    def __init__(self, base_url='http://localhost:3000'):
        self.base_url = base_url
    
    def test_health(self):
        """Test the health endpoint."""
        print("🏥 Testing Health Endpoint")
        print("-" * 30)
        
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print("✅ Health check passed")
                print(f"   Frontend status: {data.get('status', 'unknown')}")
                services = data.get('services', {})
                for service, status in services.items():
                    print(f"   {service.title()}: {status}")
            else:
                print(f"❌ Health check failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Health check error: {e}")
        print()
    
    def test_config(self):
        """Test the configuration endpoint."""
        print("⚙️  Testing Configuration Endpoint")
        print("-" * 35)
        
        try:
            response = requests.get(f"{self.base_url}/api/config", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print("✅ Configuration loaded successfully")
                print(f"   Personas: {len(data.get('personas', {}))}")
                print(f"   Locations: {len(data.get('locations', []))}")
                print(f"   Events: {len(data.get('events', []))}")
                
                # Show sample persona
                personas = data.get('personas', {})
                if personas:
                    first_persona = list(personas.keys())[0]
                    persona_info = personas[first_persona]
                    print(f"   Sample persona: {persona_info.get('name', 'Unknown')}")
            else:
                print(f"❌ Configuration failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Configuration error: {e}")
        print()
    
    def test_autosuggest(self):
        """Test the autosuggest functionality."""
        print("💡 Testing Autosuggest Functionality")
        print("-" * 37)
        
        test_queries = ['lap', 'phone', 'shoes', 'headphones']
        
        for query in test_queries:
            try:
                payload = {
                    "query": query,
                    "persona": "tech_enthusiast",
                    "location": "Mumbai",
                    "event": "none",
                    "max_suggestions": 5
                }
                
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/api/suggest",
                    json=payload,
                    timeout=10
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    data = response.json()
                    suggestions = data.get('suggestions', [])
                    response_time = data.get('response_time_ms', 0)
                    
                    print(f"✅ Query: '{query}' -> {len(suggestions)} suggestions ({response_time:.1f}ms)")
                    for i, suggestion in enumerate(suggestions[:3], 1):
                        print(f"   {i}. {suggestion.get('text', 'N/A')}")
                    if len(suggestions) > 3:
                        print(f"   ... and {len(suggestions) - 3} more")
                else:
                    print(f"❌ Query: '{query}' failed: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Query: '{query}' error: {e}")
            
            time.sleep(0.5)  # Small delay between requests
        print()
    
    def test_search(self):
        """Test the search functionality."""
        print("🔍 Testing Search Functionality")
        print("-" * 31)
        
        test_queries = [
            'samsung phone',
            'nike shoes',
            'laptop under 50000',
            'bluetooth headphones'
        ]
        
        for query in test_queries:
            try:
                payload = {
                    "query": query,
                    "context": {
                        "location": "Mumbai",
                        "persona": "tech_enthusiast"
                    },
                    "top_k": 10
                }
                
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/api/search",
                    json=payload,
                    timeout=30
                )
                end_time = time.time()
                response_time = (end_time - start_time) * 1000
                
                if response.status_code == 200:
                    results = response.json()
                    print(f"✅ Query: '{query}' -> {len(results)} results ({response_time:.1f}ms)")
                    
                    # Show top 3 results
                    for i, result in enumerate(results[:3], 1):
                        title = result.get('title', 'N/A')[:50]
                        brand = result.get('brand', 'N/A')
                        price = result.get('price', 0)
                        print(f"   {i}. {title}... - {brand} - ₹{price:,}")
                    
                    if len(results) > 3:
                        print(f"   ... and {len(results) - 3} more results")
                else:
                    print(f"❌ Query: '{query}' failed: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Query: '{query}' error: {e}")
            
            time.sleep(1)  # Delay between search requests
        print()
    
    def test_analytics(self):
        """Test the analytics endpoint."""
        print("📊 Testing Analytics Endpoint")
        print("-" * 29)
        
        try:
            response = requests.get(f"{self.base_url}/api/analytics", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print("✅ Analytics data retrieved")
                
                # Show data stats if available
                data_stats = data.get('data_stats', {})
                if data_stats:
                    print("   Data Statistics:")
                    for key, value in data_stats.items():
                        print(f"     {key}: {value:,}")
                
                # Show frontend info
                frontend_info = data.get('frontend_info', {})
                if frontend_info:
                    version = frontend_info.get('version', 'Unknown')
                    print(f"   Frontend Version: {version}")
                    
                    features = frontend_info.get('features', [])
                    if features:
                        print(f"   Features: {len(features)} available")
            else:
                print(f"❌ Analytics failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Analytics error: {e}")
        print()
    
    def run_full_demo(self):
        """Run the complete demo."""
        print("🎯 Flipkart Frontend Integration Demo")
        print("=" * 50)
        print(f"Base URL: {self.base_url}")
        print("=" * 50)
        print()
        
        # Run all tests
        self.test_health()
        self.test_config()
        self.test_autosuggest()
        self.test_search()
        self.test_analytics()
        
        print("🎉 Demo completed!")
        print()
        print("💻 Access the full application at: http://localhost:3000")
        print("🔧 Try searching for products and see the autosuggest in action!")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Flipkart Frontend Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--url",
        default="http://localhost:3000",
        help="Base URL for the frontend server (default: http://localhost:3000)"
    )
    
    args = parser.parse_args()
    
    demo = FrontendDemo(args.url)
    demo.run_full_demo()

if __name__ == "__main__":
    main()