#!/usr/bin/env python3
"""
Comprehensive test script to verify all autosuggest components work correctly with enhanced data.
"""

import time
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import DataPreprocessor
from integrated_autosuggest import IntegratedAutosuggest

def run_comprehensive_test():
    """Runs a comprehensive test suite for the Flipkart Autosuggest System."""
    print("🚀 Starting Comprehensive Test of Flipkart Autosuggest System...")
    
    try:
        # 1. Initialize and preprocess data
        print("\n📊 Initializing data preprocessor and loading datasets...")
        preprocessor = DataPreprocessor()
        preprocessor.run_all_preprocessing()
        data = preprocessor.get_processed_data()
        print("✅ Data preprocessing completed. Dataset sizes:")
        print(f"   - Product Catalog: {len(data['product_catalog'])} records")
        print(f"   - User Queries: {len(data['user_queries'])} records")
        print(f"   - Realtime Info: {len(data['realtime_product_info'])} records")
        print(f"   - Session Log: {len(data['session_log'])} records")
        
        # 2. Build autosuggest system
        print("\n🔧 Building autosuggest system components (Trie, Semantic, BERT, Reranker)...")
        autosuggest = IntegratedAutosuggest()
        autosuggest.build_system(data)
        print("✅ Autosuggest system built successfully!")
        
        # 3. Test Basic Autosuggestion (including the 'sam' scenario)
        print("\n🔍 Testing Basic Autosuggestions:")
        basic_test_queries = [
            "sam",           # Should now suggest Samsung products/queries
            "samsung",       # Should suggest Samsung products, e.g., "samsung galaxy s23"
            "xiaomi",        # Should suggest Xiaomi products, e.g., "xiaomi redmi note 12"
            "laptop",
            "gaming laptop",
            "nike shoes",
            "wireless headphones",
            "budget mobile",
            "camera for beginners"
        ]
        
        for query in basic_test_queries:
            start_time = time.time()
            suggestions = autosuggest.get_suggestions(query, max_suggestions=10)
            response_time = (time.time() - start_time) * 1000
            
            print(f"\nQuery: '{query}' (Response time: {response_time:.2f}ms)")
            print("  Suggestions:")
            if suggestions:
                for i, (suggestion, score) in enumerate(suggestions, 1):
                    print(f"    {i}. {suggestion} (score: {score:.3f})")
                    if not (0 <= score <= 1):
                        print(f"      ⚠️  WARNING: Score {score:.3f} is outside 0-1 range!")
            else:
                print("    No suggestions found.")

        # 4. Test Typo Correction (ensuring unique and relevant results)
        print("\n🔤 Testing Typo Correction:")
        typo_test_queries = [
            "samsng",    # Should correct to "samsung" or specific Samsung products
            "nkie",      # Should correct to "nike" or specific Nike products
            "addidas",   # Should correct to "adidas" or specific Adidas products
            "xiomi",     # Should correct to "xiaomi" or specific Xiaomi products
            "laptap",    # Should correct to "laptop"
            "headphons"  # Should correct to "headphones"
        ]
        
        for query in typo_test_queries:
            start_time = time.time()
            # Semantic correction now returns scores directly
            corrections = autosuggest.semantic_correction.get_semantic_suggestions(query, top_k=5)
            response_time = (time.time() - start_time) * 1000
            
            print(f"\nTypo Query: '{query}' (Response time: {response_time:.2f}ms)")
            print("  Corrections:")
            if corrections:
                seen_suggestions = set()
                unique_corrections = []
                for correction, similarity in corrections:
                    if correction not in seen_suggestions:
                        unique_corrections.append((correction, similarity))
                        seen_suggestions.add(correction)
                
                for i, (correction, similarity) in enumerate(unique_corrections[:3], 1): # Show top 3 unique
                    print(f"    {i}. {correction} (similarity: {similarity:.3f})")
                    if not (0 <= similarity <= 1):
                        print(f"      ⚠️  WARNING: Similarity {similarity:.3f} is outside 0-1 range!")
            else:
                print("    No corrections found.")

        # 5. Test Contextual Suggestions
        print("\n🎯 Testing Contextual Suggestions:")
        contextual_tests = [
            {"query": "lights", "location": "mumbai", "event": "diwali"},
            {"query": "jersey", "location": "mumbai", "event": "ipl"},
            {"query": "formal", "location": "delhi", "event": "wedding"},
            {"query": "gaming", "location": "bangalore", "session_context": {"previous_queries": ["laptop"]}}
        ]
        
        for test_case in contextual_tests:
            query = test_case["query"]
            location = test_case.get("location", None)
            event = test_case.get("event", None)
            session_context = test_case.get("session_context", None)
            
            start_time = time.time()
            suggestions = autosuggest.get_contextual_suggestions(
                query, session_context=session_context, location=location, event=event
            )
            response_time = (time.time() - start_time) * 1000
            
            context_str = f"location: {location}, event: {event}, session: {session_context}"
            print(f"\nQuery: '{query}' (Context: {context_str}) (Response time: {response_time:.2f}ms)")
            print("  Contextual Suggestions:")
            if suggestions:
                for i, (suggestion, score) in enumerate(suggestions[:5], 1): # Show top 5
                    print(f"    {i}. {suggestion} (score: {score:.3f})")
            else:
                print("    No suggestions found.")

        print("\n🎉 Comprehensive test completed!")
        print("If all tests passed and suggestions look relevant, you are ready to run the Streamlit app.")
        print("To run the Streamlit app:")
        print(f"  1. Navigate to the 'autosuggest' directory: cd {os.path.dirname(os.path.abspath(__file__))}")
        print("  2. Run the Streamlit app: streamlit run app.py")
        print("  3. Open your web browser and go to the URL provided by Streamlit (usually http://localhost:8501)")

        return True
        
    except Exception as e:
        print(f"❌ An unexpected error occurred during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    if not success:
        print("\n❌ Errors detected. Please review the output and logs to fix the issues before launching the UI.") 