import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from datetime import datetime

# Import our components
from data_preprocessing import DataPreprocessor
from integrated_autosuggest import IntegratedAutosuggest

class AutosuggestDemo:
    """Comprehensive demonstration of the autosuggest system."""
    
    def __init__(self):
        self.autosuggest = None
        self.data = None
        
    def setup_system(self):
        """Set up the complete autosuggest system."""
        print("🚀 Setting up Flipkart Autosuggest System...")
        print("=" * 60)
        
        # Load and preprocess data
        preprocessor = DataPreprocessor()
        preprocessor.run_all_preprocessing()
        self.data = preprocessor.get_processed_data()
        
        # Initialize and build autosuggest system
        self.autosuggest = IntegratedAutosuggest()
        self.autosuggest.build_system(self.data)
        
        print("✅ Autosuggest system ready!")
        print("=" * 60)
    
    def demo_basic_suggestions(self):
        """Demonstrate basic autosuggest functionality."""
        print("\n📱 Basic Autosuggest Demo")
        print("-" * 40)
        
        test_cases = [
            ("sam", "Samsung products"),
            ("app", "Apple products"),
            ("nik", "Nike products"),
            ("smart", "Smart devices"),
            ("lap", "Laptops"),
            ("head", "Headphones"),
            ("sho", "Shoes"),
            ("tv", "Televisions"),
            ("phone", "Mobile phones"),
            ("ear", "Earbuds"),
            ("key", "Keyboards"),
            ("char", "Chargers"),
            ("watch", "Watches"),
            ("tab", "Tablets"),
            ("cam", "Cameras"),
            ("speak", "Speakers"),
            ("mous", "Mouse"),
            ("case", "Cases"),
            ("bag", "Bags"),
            ("wallet", "Wallets"),
            ("hood", "Hoodies"),
            ("jean", "Jeans"),
            ("shirt", "Shirts"),
            ("sneak", "Sneakers"),
        ]
        
        for query, description in test_cases:
            start_time = time.time()
            suggestions = self.autosuggest.get_suggestions(query)
            end_time = time.time()
            
            print(f"\n🔍 Query: '{query}' ({description})")
            print(f"⏱️  Response time: {(end_time - start_time)*1000:.1f}ms")
            print(f"💡 Top suggestions:")
            
            for i, (suggestion, score) in enumerate(suggestions[:5], 1):
                print(f"   {i}. {suggestion} (score: {score:.4f})")
    
    def demo_typo_correction(self):
        """Demonstrate typo correction capabilities."""
        print("\n🔧 Typo Correction Demo")
        print("-" * 40)
        
        typo_cases = [
            ("aple fon", "apple phone"),
            ("samsng", "samsung"),
            ("nkie", "nike"),
            ("addidas", "adidas"),
            ("soney", "sony"),
            ("onepls", "oneplus"),
            ("xiomi", "xiaomi"),
            ("vvo", "vivo"),
            ("opo", "oppo"),
            ("realmi", "realme"),
            ("del", "dell"),
            ("lenvo", "lenovo"),
            ("asuss", "asus"),
            ("bot", "boat"),
            ("jb", "jbl"),
            ("pma", "puma"),
            ("rebbok", "reebok"),
            ("zarra", "zara"),
            ("ikia", "ikea"),
            ("prestig", "prestige"),
            ("h p", "hp"),
            ("l g", "lg"),
            ("bta", "bata"),
            ("h m", "hm"),
        ]
        
        for typo, expected in typo_cases:
            start_time = time.time()
            suggestions = self.autosuggest.get_suggestions(typo)
            end_time = time.time()
            
            print(f"\n🔍 Typo: '{typo}' → Expected: '{expected}'")
            print(f"⏱️  Response time: {(end_time - start_time)*1000:.1f}ms")
            print(f"💡 Suggestions:")
            
            for i, (suggestion, score) in enumerate(suggestions[:3], 1):
                print(f"   {i}. {suggestion} (score: {score:.4f})")
    
    def demo_contextual_suggestions(self):
        """Demonstrate contextual suggestions with events and locations."""
        print("\n🎯 Contextual Suggestions Demo")
        print("-" * 40)
        
        # Diwali context
        print("\n🪔 Diwali Festival Context")
        print("Location: Mumbai, Event: Diwali")
        
        diwali_queries = ["lights", "decor", "gifts", "sweets", "traditional"]
        for query in diwali_queries:
            suggestions = self.autosuggest.get_contextual_suggestions(
                query, location="Mumbai", event="diwali"
            )
            print(f"\n🔍 Query: '{query}'")
            print(f"💡 Suggestions:")
            for i, (suggestion, score) in enumerate(suggestions[:3], 1):
                print(f"   {i}. {suggestion} (score: {score:.4f})")
        
        # IPL context
        print("\n🏏 IPL Cricket Season Context")
        print("Location: Mumbai, Event: ipl")
        
        ipl_queries = ["jersey", "sports", "cricket", "team"]
        for query in ipl_queries:
            suggestions = self.autosuggest.get_contextual_suggestions(
                query, location="Mumbai", event="ipl"
            )
            print(f"\n🔍 Query: '{query}'")
            print(f"💡 Suggestions:")
            for i, (suggestion, score) in enumerate(suggestions[:3], 1):
                print(f"   {i}. {suggestion} (score: {score:.4f})")
        
        # Wedding context
        print("\n💒 Wedding Season Context")
        print("Location: Delhi, Event: wedding")
        
        wedding_queries = ["formal", "traditional", "gifts", "jewelry"]
        for query in wedding_queries:
            suggestions = self.autosuggest.get_contextual_suggestions(
                query, location="Delhi", event="wedding"
            )
            print(f"\n🔍 Query: '{query}'")
            print(f"💡 Suggestions:")
            for i, (suggestion, score) in enumerate(suggestions[:3], 1):
                print(f"   {i}. {suggestion} (score: {score:.4f})")
    
    def demo_session_awareness(self):
        """Demonstrate session-aware suggestions."""
        print("\n🔄 Session-Aware Suggestions Demo")
        print("-" * 40)
        
        # Session 1: Electronics shopper
        print("\n👨‍💻 Session 1: Electronics Shopper")
        session1 = {
            'previous_queries': ['samsung', 'mobile', 'smartphone'],
            'clicked_categories': ['Electronics'],
            'clicked_brands': ['Samsung', 'Apple']
        }
        
        electronics_queries = ["phone", "laptop", "headphone", "camera"]
        for query in electronics_queries:
            suggestions = self.autosuggest.get_contextual_suggestions(
                query, session_context=session1
            )
            print(f"\n🔍 Query: '{query}'")
            print(f"💡 Session-aware suggestions:")
            for i, (suggestion, score) in enumerate(suggestions[:3], 1):
                print(f"   {i}. {suggestion} (score: {score:.4f})")
        
        # Session 2: Fashion shopper
        print("\n👗 Session 2: Fashion Shopper")
        session2 = {
            'previous_queries': ['nike', 'shoes', 'sneakers'],
            'clicked_categories': ['Fashion'],
            'clicked_brands': ['Nike', 'Adidas']
        }
        
        fashion_queries = ["shoes", "shirt", "jeans", "hoodie"]
        for query in fashion_queries:
            suggestions = self.autosuggest.get_contextual_suggestions(
                query, session_context=session2
            )
            print(f"\n🔍 Query: '{query}'")
            print(f"💡 Session-aware suggestions:")
            for i, (suggestion, score) in enumerate(suggestions[:3], 1):
                print(f"   {i}. {suggestion} (score: {score:.4f})")
    
    def demo_performance_metrics(self):
        """Demonstrate performance metrics."""
        print("\n⚡ Performance Metrics Demo")
        print("-" * 40)
        
        test_queries = [
            "sam", "app", "nik", "smart", "lap", "head", "sho", "tv", 
            "phone", "ear", "key", "char", "watch", "tab", "cam", "speak"
        ]
        
        total_time = 0
        total_queries = len(test_queries)
        
        print(f"Testing {total_queries} queries...")
        
        for query in test_queries:
            start_time = time.time()
            suggestions = self.autosuggest.get_suggestions(query)
            end_time = time.time()
            
            query_time = (end_time - start_time) * 1000
            total_time += query_time
            
            print(f"✅ '{query}': {query_time:.1f}ms")
        
        avg_time = total_time / total_queries
        qps = 1000 / avg_time
        
        print(f"\n📊 Performance Summary:")
        print(f"   Average response time: {avg_time:.1f}ms")
        print(f"   Queries per second: {qps:.0f}")
        print(f"   Total test time: {total_time/1000:.2f}s")
    
    def demo_real_world_scenarios(self):
        """Demonstrate real-world e-commerce scenarios."""
        print("\n🛒 Real-World E-commerce Scenarios")
        print("-" * 40)
        
        scenarios = [
            {
                "name": "Mobile Phone Shopper",
                "location": "Bangalore",
                "event": None,
                "session": {
                    'previous_queries': ['samsung', 'mobile'],
                    'clicked_categories': ['Electronics'],
                    'clicked_brands': ['Samsung']
                },
                "queries": ["phone", "smartphone", "camera", "charger"]
            },
            {
                "name": "Gaming Laptop Shopper",
                "location": "Mumbai",
                "event": None,
                "session": {
                    'previous_queries': ['asus', 'gaming', 'laptop'],
                    'clicked_categories': ['Electronics'],
                    'clicked_brands': ['Asus', 'Dell']
                },
                "queries": ["laptop", "gaming", "keyboard", "mouse"]
            },
            {
                "name": "Diwali Gift Shopper",
                "location": "Delhi",
                "event": "diwali",
                "session": {
                    'previous_queries': ['gifts', 'lights'],
                    'clicked_categories': ['Home', 'Electronics'],
                    'clicked_brands': []
                },
                "queries": ["lights", "decor", "gifts", "sweets"]
            },
            {
                "name": "Sports Shoe Shopper",
                "location": "Chennai",
                "event": None,
                "session": {
                    'previous_queries': ['nike', 'running', 'shoes'],
                    'clicked_categories': ['Fashion'],
                    'clicked_brands': ['Nike', 'Adidas']
                },
                "queries": ["shoes", "sneakers", "running", "sports"]
            }
        ]
        
        for scenario in scenarios:
            print(f"\n👤 {scenario['name']}")
            print(f"📍 Location: {scenario['location']}")
            if scenario['event']:
                print(f"🎉 Event: {scenario['event']}")
            
            for query in scenario['queries']:
                suggestions = self.autosuggest.get_contextual_suggestions(
                    query,
                    location=scenario['location'],
                    event=scenario['event'],
                    session_context=scenario['session']
                )
                
                print(f"\n🔍 Query: '{query}'")
                print(f"💡 Contextual suggestions:")
                for i, (suggestion, score) in enumerate(suggestions[:3], 1):
                    print(f"   {i}. {suggestion} (score: {score:.4f})")
    
    def run_complete_demo(self):
        """Run the complete demonstration."""
        print("🎯 Flipkart Autosuggest System - Complete Demo")
        print("=" * 60)
        
        # Setup system
        self.setup_system()
        
        # Run all demos
        self.demo_basic_suggestions()
        self.demo_typo_correction()
        self.demo_contextual_suggestions()
        self.demo_session_awareness()
        self.demo_performance_metrics()
        self.demo_real_world_scenarios()
        
        print("\n🎉 Demo completed successfully!")
        print("=" * 60)
        print("📋 Summary of Features:")
        print("   ✅ Trie-based prefix matching")
        print("   ✅ Semantic correction for typos")
        print("   ✅ BERT-based query completion")
        print("   ✅ XGBoost reranking")
        print("   ✅ Location-aware suggestions")
        print("   ✅ Event-based contextual boosting")
        print("   ✅ Session-aware personalization")
        print("   ✅ Real-time performance optimization")

# Run the demo
if __name__ == "__main__":
    demo = AutosuggestDemo()
    demo.run_complete_demo() 