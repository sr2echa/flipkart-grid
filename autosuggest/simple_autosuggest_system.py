"""
Simple, High-Quality Autosuggest System
Focus: Quality over complexity, reliable suggestions, contextual relevance
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from collections import defaultdict, Counter
import re
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SimpleAutosuggestSystem:
    """
    A simplified, high-quality autosuggest system focusing on reliable suggestions.
    """
    
    def __init__(self):
        self.query_frequencies = {}
        self.query_trie = {}
        self.popular_queries = []
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.query_vectors = None
        self.all_queries = []
        self.category_keywords = {}
        self.brand_keywords = {}
        
    def build_system(self, data: Dict):
        """Build the simplified autosuggest system."""
        print("🚀 Building Simple Autosuggest System...")
        
        # Process user queries
        user_queries = data.get('user_queries', pd.DataFrame())
        if not user_queries.empty:
            self._build_query_database(user_queries)
        
        # Process product catalog for keywords
        product_catalog = data.get('product_catalog', pd.DataFrame())
        if not product_catalog.empty:
            self._extract_product_keywords(product_catalog)
        
        print("✅ Simple Autosuggest System built successfully!")
    
    def _build_query_database(self, user_queries: pd.DataFrame):
        """Build query database with frequencies and trie structure."""
        print("-> Building query database...")
        
        # Extract queries and frequencies
        queries = []
        if 'corrected_query' in user_queries.columns:
            queries = user_queries['corrected_query'].dropna().tolist()
        elif 'raw_query' in user_queries.columns:
            queries = user_queries['raw_query'].dropna().tolist()
        else:
            print("Warning: No query columns found")
            return
        
        # Clean and process queries
        processed_queries = []
        for query in queries:
            if isinstance(query, str) and len(query.strip()) > 0:
                clean_query = query.lower().strip()
                processed_queries.append(clean_query)
        
        # Calculate frequencies
        self.query_frequencies = Counter(processed_queries)
        self.all_queries = list(self.query_frequencies.keys())
        
        # Get most popular queries
        self.popular_queries = [q for q, _ in self.query_frequencies.most_common(100)]
        
        # Build trie for prefix matching
        self._build_trie()
        
        # Build TF-IDF vectors for semantic similarity
        if self.all_queries:
            self.vectorizer.fit(self.all_queries)
            self.query_vectors = self.vectorizer.transform(self.all_queries)
        
        print(f"-> Processed {len(self.all_queries)} unique queries")
    
    def _build_trie(self):
        """Build a simple trie structure for fast prefix matching."""
        self.query_trie = {}
        
        for query in self.all_queries:
            current = self.query_trie
            for char in query:
                if char not in current:
                    current[char] = {}
                current = current[char]
            current['_end'] = query
    
    def _extract_product_keywords(self, product_catalog: pd.DataFrame):
        """Extract category and brand keywords from product catalog."""
        print("-> Extracting product keywords...")
        
        # Extract category keywords
        category_queries = []
        if 'category' in product_catalog.columns:
            categories = product_catalog['category'].dropna().unique()
            category_queries.extend([cat.lower() for cat in categories])
        
        if 'subcategory' in product_catalog.columns:
            subcategories = product_catalog['subcategory'].dropna().unique()
            category_queries.extend([sub.lower() for sub in subcategories])
        
        # Extract brand keywords
        brand_queries = []
        if 'brand' in product_catalog.columns:
            brands = product_catalog['brand'].dropna().unique()
            brand_queries.extend([brand.lower() for brand in brands])
        
        # Store keywords
        self.category_keywords = set(category_queries)
        self.brand_keywords = set(brand_queries)
        
        print(f"-> Extracted {len(self.category_keywords)} categories, {len(self.brand_keywords)} brands")
    
    def get_suggestions(self, query: str, context: Dict = None, max_suggestions: int = 5) -> List[Tuple[str, float]]:
        """Get autosuggest suggestions with multiple strategies."""
        if not query or len(query.strip()) == 0:
            # Return popular queries if no input
            return [(q, 1.0) for q in self.popular_queries[:max_suggestions]]
        
        query = query.lower().strip()
        suggestions = []
        
        # Strategy 1: Exact prefix matching
        prefix_matches = self._get_prefix_matches(query, max_suggestions * 2)
        suggestions.extend(prefix_matches)
        
        # Strategy 2: Fuzzy matching for typos
        fuzzy_matches = self._get_fuzzy_matches(query, max_suggestions)
        suggestions.extend(fuzzy_matches)
        
        # Strategy 3: Semantic similarity
        semantic_matches = self._get_semantic_matches(query, max_suggestions)
        suggestions.extend(semantic_matches)
        
        # Strategy 4: Category/Brand matching
        keyword_matches = self._get_keyword_matches(query, max_suggestions)
        suggestions.extend(keyword_matches)
        
        # Strategy 5: Contextual boosting
        if context:
            suggestions = self._apply_contextual_boosting(suggestions, context)
        
        # Deduplicate and rank
        final_suggestions = self._rank_and_deduplicate(suggestions, max_suggestions)
        
        return final_suggestions
    
    def _get_prefix_matches(self, query: str, max_results: int) -> List[Tuple[str, float]]:
        """Get exact prefix matches using the trie."""
        matches = []
        current = self.query_trie
        
        # Traverse the trie
        for char in query:
            if char in current:
                current = current[char]
            else:
                return matches  # No matches found
        
        # Collect all completions
        def collect_completions(node, current_matches):
            if len(current_matches) >= max_results:
                return
            
            if '_end' in node:
                completion = node['_end']
                frequency = self.query_frequencies.get(completion, 1)
                score = frequency / max(self.query_frequencies.values())
                current_matches.append((completion, score))
            
            for char, child_node in node.items():
                if char != '_end':
                    collect_completions(child_node, current_matches)
        
        prefix_matches = []
        collect_completions(current, prefix_matches)
        
        # Sort by frequency
        prefix_matches.sort(key=lambda x: x[1], reverse=True)
        return prefix_matches[:max_results]
    
    def _get_fuzzy_matches(self, query: str, max_results: int) -> List[Tuple[str, float]]:
        """Get fuzzy matches for typo correction."""
        matches = []
        
        for candidate in self.all_queries:
            # Skip if too different in length
            if abs(len(candidate) - len(query)) > 3:
                continue
            
            # Calculate similarity
            similarity = fuzz.ratio(query, candidate) / 100.0
            
            # Only consider good matches
            if similarity > 0.7:
                frequency = self.query_frequencies.get(candidate, 1)
                score = similarity * 0.7 + (frequency / max(self.query_frequencies.values())) * 0.3
                matches.append((candidate, score))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:max_results]
    
    def _get_semantic_matches(self, query: str, max_results: int) -> List[Tuple[str, float]]:
        """Get semantically similar queries."""
        if self.query_vectors is None or self.query_vectors.shape[0] == 0:
            return []
        
        try:
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.query_vectors).flatten()
            
            # Get top matches
            top_indices = np.argsort(similarities)[::-1][:max_results * 2]
            matches = []
            
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    candidate = self.all_queries[idx]
                    if candidate != query:  # Don't suggest the same query
                        score = similarities[idx] * 0.8
                        matches.append((candidate, score))
            
            return matches[:max_results]
            
        except Exception as e:
            print(f"Semantic matching error: {e}")
            return []
    
    def _get_keyword_matches(self, query: str, max_results: int) -> List[Tuple[str, float]]:
        """Get matches based on category/brand keywords."""
        matches = []
        
        # Check category keywords
        for category in self.category_keywords:
            if query in category or category in query:
                # Find queries containing this category
                for candidate in self.all_queries:
                    if category in candidate and candidate not in [m[0] for m in matches]:
                        frequency = self.query_frequencies.get(candidate, 1)
                        score = 0.6 + (frequency / max(self.query_frequencies.values())) * 0.4
                        matches.append((candidate, score))
        
        # Check brand keywords  
        for brand in self.brand_keywords:
            if query in brand or brand in query:
                for candidate in self.all_queries:
                    if brand in candidate and candidate not in [m[0] for m in matches]:
                        frequency = self.query_frequencies.get(candidate, 1)
                        score = 0.6 + (frequency / max(self.query_frequencies.values())) * 0.4
                        matches.append((candidate, score))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:max_results]
    
    def _apply_contextual_boosting(self, suggestions: List[Tuple[str, float]], context: Dict) -> List[Tuple[str, float]]:
        """Apply contextual boosting based on persona, location, event."""
        boosted_suggestions = []
        
        persona = context.get('persona', '')
        location = context.get('location', '')
        event = context.get('event', '')
        
        for suggestion, score in suggestions:
            boosted_score = score
            
            # Persona-based boosting
            if persona == 'sports_enthusiast':
                if any(word in suggestion for word in ['nike', 'adidas', 'puma', 'jersey', 'cricket', 'football', 'sports']):
                    boosted_score *= 1.5
            elif persona == 'tech_enthusiast':
                if any(word in suggestion for word in ['laptop', 'mobile', 'phone', 'samsung', 'apple', 'gaming']):
                    boosted_score *= 1.5
            elif persona == 'fashion_lover':
                if any(word in suggestion for word in ['dress', 'shirt', 'jeans', 'fashion', 'style', 'trendy']):
                    boosted_score *= 1.5
            
            # Event-based boosting
            if event == 'IPL' and any(word in suggestion for word in ['cricket', 'jersey', 'csk', 'rcb', 'mi']):
                boosted_score *= 1.3
            elif event == 'Diwali' and any(word in suggestion for word in ['lights', 'decoration', 'gift', 'diya']):
                boosted_score *= 1.3
            
            # Location-based boosting
            if location in ['Bangalore', 'Hyderabad'] and any(word in suggestion for word in ['tech', 'laptop', 'gaming']):
                boosted_score *= 1.2
            
            boosted_suggestions.append((suggestion, boosted_score))
        
        return boosted_suggestions
    
    def _rank_and_deduplicate(self, suggestions: List[Tuple[str, float]], max_results: int) -> List[Tuple[str, float]]:
        """Rank and deduplicate suggestions."""
        # Deduplicate by query text
        seen = set()
        unique_suggestions = []
        
        for suggestion, score in suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique_suggestions.append((suggestion, score))
        
        # Sort by score
        unique_suggestions.sort(key=lambda x: x[1], reverse=True)
        
        return unique_suggestions[:max_results]

# Test the system
if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    
    print("=== Testing Simple Autosuggest System ===")
    
    # Load data
    preprocessor = DataPreprocessor()
    preprocessor.run_all_preprocessing()
    data = preprocessor.get_processed_data()
    
    # Build system
    autosuggest = SimpleAutosuggestSystem()
    autosuggest.build_system(data)
    
    # Test cases
    test_cases = [
        ("j", {"persona": "sports_enthusiast", "location": "Chennai", "event": "IPL"}),
        ("jer", {"persona": "sports_enthusiast", "location": "Chennai", "event": "IPL"}),
        ("jersy", {"persona": "sports_enthusiast", "location": "Chennai", "event": "IPL"}),
        ("laptop", {"persona": "tech_enthusiast", "location": "Bangalore", "event": "None"}),
        ("sam", {"persona": "tech_enthusiast", "location": "Delhi", "event": "None"}),
        ("samsng", {"persona": "tech_enthusiast", "location": "Delhi", "event": "None"}),
        ("nike", {"persona": "sports_enthusiast", "location": "Pune", "event": "None"}),
        ("lights", {"persona": "home_maker", "location": "Mumbai", "event": "Diwali"}),
    ]
    
    print("\n=== Test Results ===")
    for query, context in test_cases:
        start_time = time.time()
        suggestions = autosuggest.get_suggestions(query, context)
        end_time = time.time()
        
        print(f"\nQuery: '{query}' | Context: {context}")
        print(f"Suggestions: {[s for s, score in suggestions]}")
        print(f"Scores: {[f'{score:.3f}' for s, score in suggestions]}")
        print(f"Response time: {(end_time - start_time)*1000:.2f}ms")
        
        if not suggestions:
            print("❌ NO SUGGESTIONS - NEED TO FIX!")
        else:
            print(f"✅ Got {len(suggestions)} suggestions")