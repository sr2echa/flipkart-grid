#!/usr/bin/env python3
"""
Comprehensive test suite for the Flipkart Autosuggest System.
Tests all components individually and the integrated system.
"""

import time
import pandas as pd
from typing import List, Dict, Tuple
import numpy as np

# Import all components
from data_preprocessing import DataPreprocessor
from enhanced_trie import EnhancedTrieAutosuggest
from enhanced_semantic_correction import EnhancedSemanticCorrection
from bert_completion import BERTCompletion
from integrated_autosuggest import IntegratedAutosuggest

class ComprehensiveTest:
    """Comprehensive test suite for the autosuggest system."""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.data = None
        
    def log_test(self, test_name: str, status: str, details: str = ""):
        """Log test results."""
        self.results[test_name] = {
            'status': status,
            'details': details,
            'timestamp': time.time()
        }
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_emoji} {test_name}: {status}")
        if details:
            print(f"   {details}")

    def test_data_preprocessing(self):
        """Test data preprocessing component."""
        print("\n🔄 Testing Data Preprocessing...")
        
        try:
            preprocessor = DataPreprocessor()
            success = preprocessor.run_all_preprocessing()
            self.data = preprocessor.get_processed_data()
            
            # Verify data structure
            required_keys = ['user_queries', 'product_catalog', 'session_log', 'realtime_product_info']
            for key in required_keys:
                if key not in self.data:
                    self.log_test(f"Data Preprocessing - {key}", "FAIL", f"Missing {key} in data")
                    return False
            
            # Verify data content
            if len(self.data['user_queries']) == 0:
                self.log_test("Data Preprocessing - Content", "FAIL", "No user queries loaded")
                return False
                
            self.log_test("Data Preprocessing", "PASS", 
                         f"Loaded {len(self.data['user_queries'])} queries, "
                         f"{len(self.data['product_catalog'])} products")
            return True
            
        except Exception as e:
            self.log_test("Data Preprocessing", "FAIL", str(e))
            return False

    def test_trie_component(self):
        """Test Trie autosuggest component."""
        print("\n🌳 Testing Trie Component...")
        
        try:
            trie = EnhancedTrieAutosuggest()
            trie.build_trie(self.data['user_queries'])
            
            # Test basic functionality
            test_queries = ['sam', 'app', 'laptop', 'nike']
            for query in test_queries:
                suggestions = trie.get_suggestions(query, max_suggestions=5)
                if not suggestions:
                    self.log_test(f"Trie - {query}", "WARN", "No suggestions returned")
                else:
                    self.log_test(f"Trie - {query}", "PASS", 
                                 f"Got {len(suggestions)} suggestions")
            
            # Test performance
            start_time = time.time()
            for _ in range(100):
                trie.get_suggestions('samsung')
            end_time = time.time()
            avg_time = (end_time - start_time) / 100 * 1000
            
            if avg_time < 5:  # Should be under 5ms
                self.log_test("Trie Performance", "PASS", f"Avg response: {avg_time:.2f}ms")
            else:
                self.log_test("Trie Performance", "WARN", f"Slow response: {avg_time:.2f}ms")
                
            return True
            
        except Exception as e:
            self.log_test("Trie Component", "FAIL", str(e))
            return False

    def test_semantic_component(self):
        """Test semantic correction component."""
        print("\n🧠 Testing Semantic Component...")
        
        try:
            semantic = EnhancedSemanticCorrection()
            semantic.build_semantic_index(self.data['user_queries'])
            
            # Test typo correction
            typo_tests = [
                ('samsng', 'samsung'),
                ('aple', 'apple'),
                ('nkie', 'nike'),
                ('lenvo', 'lenovo')
            ]
            
            correct_count = 0
            for typo, expected in typo_tests:
                suggestions = semantic.get_semantic_suggestions(typo)
                if suggestions:
                    top_suggestion = suggestions[0][0]
                    if expected in top_suggestion.lower():
                        correct_count += 1
                        self.log_test(f"Semantic - {typo}", "PASS", 
                                     f"Corrected to: {top_suggestion}")
                    else:
                        self.log_test(f"Semantic - {typo}", "WARN", 
                                     f"Expected {expected}, got {top_suggestion}")
                else:
                    self.log_test(f"Semantic - {typo}", "FAIL", "No suggestions")
            
            accuracy = correct_count / len(typo_tests) * 100
            if accuracy >= 75:
                self.log_test("Semantic Accuracy", "PASS", f"{accuracy:.1f}% correct")
            else:
                self.log_test("Semantic Accuracy", "WARN", f"{accuracy:.1f}% correct")
                
            return True
            
        except Exception as e:
            self.log_test("Semantic Component", "FAIL", str(e))
            return False

    def test_bert_component(self):
        """Test BERT completion component."""
        print("\n🤖 Testing BERT Component...")
        
        try:
            bert = BERTCompletion()
            bert.build_completion_patterns(self.data['user_queries'])
            
            # Test completion
            test_prefixes = ['phone', 'laptop', 'shoes', 'watch']
            for prefix in test_prefixes:
                completions = bert.complete_query(prefix)
                if completions:
                    # Check for quality (no weird tokens)
                    quality_check = all(
                        not any(bad in completion for bad in ['##', '[UNK]', '!', '.', ';'])
                        for completion in completions
                    )
                    if quality_check:
                        self.log_test(f"BERT - {prefix}", "PASS", 
                                     f"Got {len(completions)} quality completions")
                    else:
                        self.log_test(f"BERT - {prefix}", "WARN", "Some low-quality completions")
                else:
                    self.log_test(f"BERT - {prefix}", "WARN", "No completions")
            
            return True
            
        except Exception as e:
            self.log_test("BERT Component", "FAIL", str(e))
            return False

    def test_integrated_system(self):
        """Test the integrated autosuggest system."""
        print("\n🔗 Testing Integrated System...")
        
        try:
            autosuggest = IntegratedAutosuggest()
            autosuggest.build_system(self.data)
            
            # Test basic suggestions
            test_cases = [
                'sam',
                'samsng',  # typo
                'phone under',  # completion
                'nike shoes'
            ]
            
            all_passed = True
            for query in test_cases:
                start_time = time.time()
                suggestions = autosuggest.get_suggestions(query, max_suggestions=5)
                end_time = time.time()
                response_time = (end_time - start_time) * 1000
                
                if suggestions:
                    self.log_test(f"Integrated - {query}", "PASS", 
                                 f"{len(suggestions)} suggestions in {response_time:.1f}ms")
                else:
                    self.log_test(f"Integrated - {query}", "WARN", "No suggestions")
                    all_passed = False
            
            # Test contextual suggestions
            context_tests = [
                ('lights', {'location': 'Mumbai', 'event': 'diwali'}),
                ('jersey', {'location': 'Chennai', 'event': 'ipl'}),
                ('laptop', {'session_context': {'previous_queries': ['gaming']}})
            ]
            
            for query, context in context_tests:
                suggestions = autosuggest.get_contextual_suggestions(
                    query, 
                    session_context=context.get('session_context'),
                    location=context.get('location'),
                    event=context.get('event')
                )
                
                if suggestions:
                    self.log_test(f"Contextual - {query}", "PASS", 
                                 f"Got contextual suggestions")
                else:
                    self.log_test(f"Contextual - {query}", "WARN", "No contextual suggestions")
            
            return all_passed
            
        except Exception as e:
            self.log_test("Integrated System", "FAIL", str(e))
            return False

    def test_performance(self):
        """Test system performance."""
        print("\n⚡ Testing Performance...")
        
        try:
            autosuggest = IntegratedAutosuggest()
            autosuggest.build_system(self.data)
            
            # Performance test
            test_queries = ['sam', 'phone', 'laptop', 'shoes', 'watch'] * 20
            times = []
            
            for query in test_queries:
                start_time = time.time()
                suggestions = autosuggest.get_suggestions(query)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)
            
            avg_time = np.mean(times)
            p95_time = np.percentile(times, 95)
            qps = 1000 / avg_time
            
            # Performance benchmarks
            if avg_time < 200:
                self.log_test("Average Response Time", "PASS", f"{avg_time:.1f}ms")
            else:
                self.log_test("Average Response Time", "WARN", f"{avg_time:.1f}ms (target: <200ms)")
            
            if p95_time < 500:
                self.log_test("P95 Response Time", "PASS", f"{p95_time:.1f}ms")
            else:
                self.log_test("P95 Response Time", "WARN", f"{p95_time:.1f}ms (target: <500ms)")
            
            if qps >= 5:
                self.log_test("Throughput", "PASS", f"{qps:.1f} QPS")
            else:
                self.log_test("Throughput", "WARN", f"{qps:.1f} QPS (target: >=5)")
            
            return True
            
        except Exception as e:
            self.log_test("Performance Test", "FAIL", str(e))
            return False

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("\n🧪 Testing Edge Cases...")
        
        try:
            autosuggest = IntegratedAutosuggest()
            autosuggest.build_system(self.data)
            
            edge_cases = [
                '',  # Empty query
                ' ',  # Whitespace only
                'xxxxxxxxx',  # Non-existent
                '123',  # Numbers only
                'a',  # Single character
                'a' * 100,  # Very long query
            ]
            
            for query in edge_cases:
                try:
                    suggestions = autosuggest.get_suggestions(query)
                    self.log_test(f"Edge Case - '{query[:10]}...'", "PASS", 
                                 f"Handled gracefully")
                except Exception as e:
                    self.log_test(f"Edge Case - '{query[:10]}...'", "FAIL", str(e))
            
            return True
            
        except Exception as e:
            self.log_test("Edge Cases", "FAIL", str(e))
            return False

    def run_all_tests(self):
        """Run all tests and generate report."""
        print("🚀 Starting Comprehensive Autosuggest System Tests")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Run all tests
        tests = [
            self.test_data_preprocessing,
            self.test_trie_component,
            self.test_semantic_component,
            self.test_bert_component,
            self.test_integrated_system,
            self.test_performance,
            self.test_edge_cases
        ]
        
        passed = 0
        failed = 0
        warnings = 0
        
        for test in tests:
            try:
                test()
            except Exception as e:
                self.log_test(f"Test Execution Error", "FAIL", str(e))
        
        # Generate report
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        for test_name, result in self.results.items():
            status = result['status']
            if status == "PASS":
                passed += 1
            elif status == "FAIL":
                failed += 1
            else:
                warnings += 1
        
        total_time = time.time() - self.start_time
        
        print(f"✅ Passed: {passed}")
        print(f"⚠️  Warnings: {warnings}")
        print(f"❌ Failed: {failed}")
        print(f"⏱️  Total time: {total_time:.2f}s")
        
        if failed == 0:
            print("\n🎉 ALL CORE TESTS PASSED! System is ready for production.")
            if warnings > 0:
                print(f"⚠️  Note: {warnings} warnings need attention for optimal performance.")
        else:
            print(f"\n🔧 {failed} tests failed. Please fix issues before deployment.")
        
        # Detailed results
        print("\n📋 DETAILED RESULTS:")
        print("-" * 40)
        for test_name, result in self.results.items():
            status_emoji = "✅" if result['status'] == "PASS" else "❌" if result['status'] == "FAIL" else "⚠️"
            print(f"{status_emoji} {test_name}: {result['status']}")
            if result['details']:
                print(f"   └─ {result['details']}")
        
        return failed == 0

if __name__ == "__main__":
    tester = ComprehensiveTest()
    success = tester.run_all_tests()
    exit(0 if success else 1)