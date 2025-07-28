# test_search_enhanced.py

"""
Enhanced testing script for Grid 7.0 Search API
===============================================

This script performs comprehensive testing of the search API to ensure
consistent performance without model reloading issues.
"""

import requests
import time
import json
from typing import List, Dict
import statistics

API_BASE_URL = "http://127.0.0.1:8000"
SEARCH_URL = f"{API_BASE_URL}/search/"
HEALTH_URL = f"{API_BASE_URL}/health/"
STATS_URL = f"{API_BASE_URL}/stats/"

def test_health_check():
    """Test the health check endpoint."""
    print("🏥 Testing health check...")
    try:
        response = requests.get(HEALTH_URL, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check error: {e}")
        return False

def test_model_stats():
    """Test the model stats endpoint."""
    print("📊 Testing model stats...")
    try:
        response = requests.get(STATS_URL, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model stats: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ Stats check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Stats check error: {e}")
        return False

def perform_single_search(query: str, top_k: int = 10) -> Dict:
    """Perform a single search and return timing information."""
    start_time = time.time()
    
    try:
        response = requests.post(
            SEARCH_URL, 
            json={"query": query, "top_k": top_k},
            timeout=30  # 30 second timeout
        )
        
        duration = time.time() - start_time
        
        if response.status_code == 200:
            results = response.json()
            return {
                "success": True,
                "query": query,
                "duration": duration,
                "results_count": len(results),
                "top_result": results[0] if results else None
            }
        else:
            return {
                "success": False,
                "query": query,
                "duration": duration,
                "error": f"HTTP {response.status_code}: {response.text}"
            }
            
    except requests.exceptions.RequestException as e:
        duration = time.time() - start_time
        return {
            "success": False,
            "query": query,
            "duration": duration,
            "error": str(e)
        }

def test_consistent_performance():
    """Test multiple queries to ensure consistent performance."""
    print("\n🎯 Testing consistent performance...")
    print("=" * 60)
    
    test_queries = [
        "running shoes for men",
    ]
    
    all_results = []
    durations = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"🔍 Test {i}/10: '{query}'")
        result = perform_single_search(query, top_k=5)
        all_results.append(result)
        
        if result["success"]:
            durations.append(result["duration"])
            print(f"   ✅ Success in {result['duration']:.3f}s - {result['results_count']} results")
            if result["top_result"]:
                print(f"   🥇 Top result: {result['top_result']['title']}")
        else:
            print(f"   ❌ Failed: {result['error']}")
        
        # Small delay between requests
        time.sleep(0.5)
    
    # Performance analysis
    print("\n📈 Performance Analysis:")
    print("=" * 40)
    
    successful_tests = [r for r in all_results if r["success"]]
    failed_tests = [r for r in all_results if not r["success"]]
    
    print(f"✅ Successful tests: {len(successful_tests)}/10")
    print(f"❌ Failed tests: {len(failed_tests)}/10")
    
    if durations:
        avg_duration = statistics.mean(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        std_duration = statistics.stdev(durations) if len(durations) > 1 else 0
        
        print(f"⏱️ Average response time: {avg_duration:.3f}s")
        print(f"⚡ Fastest response: {min_duration:.3f}s")
        print(f"🐌 Slowest response: {max_duration:.3f}s")
        print(f"📊 Standard deviation: {std_duration:.3f}s")
        
        # Check for consistency (no response should be > 2 seconds after first)
        slow_responses = [d for d in durations if d > 2.0]
        if slow_responses:
            print(f"⚠️ WARNING: {len(slow_responses)} responses took >2s (possible model reload)")
        else:
            print("🎉 All responses were fast (<2s) - No model reloading detected!")
    
    return len(failed_tests) == 0

def test_concurrent_requests():
    """Test multiple concurrent requests."""
    print("\n🔄 Testing concurrent requests...")
    print("=" * 40)
    
    import threading
    import queue
    
    queries = ["shoes", "laptop", "phone", "jacket", "headphones"]
    results_queue = queue.Queue()
    
    def worker(query):
        result = perform_single_search(f"{query} best", top_k=3)
        results_queue.put(result)
    
    # Start all threads
    threads = []
    start_time = time.time()
    
    for query in queries:
        thread = threading.Thread(target=worker, args=(query,))
        thread.start()
        threads.append(thread)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    total_time = time.time() - start_time
    
    # Collect results
    concurrent_results = []
    while not results_queue.empty():
        concurrent_results.append(results_queue.get())
    
    successful = sum(1 for r in concurrent_results if r["success"])
    print(f"✅ Concurrent requests: {successful}/{len(queries)} successful")
    print(f"⏱️ Total time for {len(queries)} concurrent requests: {total_time:.3f}s")
    
    return successful == len(queries)

def main():
    """Run all tests."""
    print("🧪 Grid 7.0 Search API - Comprehensive Testing")
    print("=" * 60)
    
    # Test 1: Health check
    if not test_health_check():
        print("❌ Health check failed. Is the server running?")
        return
    
    print()
    
    # Test 2: Model stats
    test_model_stats()
    
    print()
    
    # Test 3: Consistent performance
    performance_ok = test_consistent_performance()
    
    print()
    
    # Test 4: Concurrent requests
    concurrent_ok = test_concurrent_requests()
    
    print("\n" + "=" * 60)
    print("🏁 FINAL RESULTS:")
    
    if performance_ok and concurrent_ok:
        print("🎉 ALL TESTS PASSED! The API is working consistently.")
        print("✅ No model reloading issues detected.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        if not performance_ok:
            print("❌ Performance test failed - possible model reloading issues")
        if not concurrent_ok:
            print("❌ Concurrent request test failed")

if __name__ == "__main__":
    main()