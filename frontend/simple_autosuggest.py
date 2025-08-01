"""
Simple, High-Quality Autosuggest System - Minimal Dependencies Version
"""
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import re

class SimpleAutosuggestSystem:
    """
    A simplified, high-quality autosuggest system with minimal dependencies.
    """
    
    def __init__(self):
        self.query_frequencies = {}
        self.query_trie = {}
        self.popular_queries = []
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
        elif 'query' in user_queries.columns:
            queries = user_queries['query'].dropna().tolist()
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
        
        print(f"   -> Processed {len(self.all_queries)} unique queries")
    
    def _build_trie(self):
        """Build a simple trie for fast prefix matching."""
        self.query_trie = {}
        
        for query in self.all_queries:
            current = self.query_trie
            for char in query:
                if char not in current:
                    current[char] = {}
                current = current[char]
            current['$'] = query  # End marker with full query
    
    def _extract_product_keywords(self, product_catalog: pd.DataFrame):
        """Extract keywords from product catalog."""
        print("-> Extracting product keywords...")
        
        # Extract brand keywords
        if 'brand' in product_catalog.columns:
            brands = product_catalog['brand'].dropna().str.lower().unique()
            self.brand_keywords = {brand: brand for brand in brands if len(brand) > 1}
        
        # Extract category keywords
        if 'category' in product_catalog.columns:
            categories = product_catalog['category'].dropna().str.lower().unique()
            self.category_keywords = {cat: cat for cat in categories if len(cat) > 2}
        
        print(f"   -> Extracted {len(self.brand_keywords)} brands and {len(self.category_keywords)} categories")
    
    def get_suggestions(self, query: str, max_suggestions: int = 5, context: Optional[Dict] = None) -> List[Tuple[str, float]]:
        """Get autosuggest suggestions for a query."""
        if not query or len(query.strip()) == 0:
            return []
        
        query = query.lower().strip()
        suggestions = []
        
        # 1. Prefix matching from trie
        prefix_matches = self._get_prefix_matches(query, max_suggestions * 2)
        for match in prefix_matches:
            score = self.query_frequencies.get(match, 1) / 100.0  # Normalize frequency
            suggestions.append((match, score))
        
        # 2. Fuzzy matching for typo correction
        fuzzy_matches = self._get_fuzzy_matches(query, max_suggestions)
        for match in fuzzy_matches:
            if match not in [s[0] for s in suggestions]:
                score = self.query_frequencies.get(match, 1) / 200.0  # Lower score for fuzzy
                suggestions.append((match, score))
        
        # 3. Brand and category matching
        keyword_matches = self._get_keyword_matches(query, max_suggestions)
        for match in keyword_matches:
            if match not in [s[0] for s in suggestions]:
                suggestions.append((match, 0.3))  # Fixed score for keywords
        
        # 4. Popular queries as fallback
        if len(suggestions) < max_suggestions:
            for popular in self.popular_queries[:max_suggestions]:
                if popular not in [s[0] for s in suggestions] and query in popular:
                    suggestions.append((popular, 0.1))
        
        # Sort by score and return top suggestions
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:max_suggestions]
    
    def _get_prefix_matches(self, query: str, max_matches: int) -> List[str]:
        """Get queries that start with the given prefix."""
        current = self.query_trie
        
        # Navigate to the prefix
        for char in query:
            if char not in current:
                return []
            current = current[char]
        
        # Collect all queries with this prefix
        matches = []
        self._collect_queries(current, matches, max_matches)
        
        # Sort by frequency
        matches.sort(key=lambda x: self.query_frequencies.get(x, 0), reverse=True)
        return matches[:max_matches]
    
    def _collect_queries(self, node: dict, matches: List[str], max_matches: int):
        """Recursively collect queries from trie node."""
        if len(matches) >= max_matches:
            return
        
        if '$' in node:
            matches.append(node['$'])
        
        for char, child in node.items():
            if char != '$' and len(matches) < max_matches:
                self._collect_queries(child, matches, max_matches)
    
    def _get_fuzzy_matches(self, query: str, max_matches: int) -> List[str]:
        """Get fuzzy matches for typo correction."""
        matches = []
        
        for candidate in self.all_queries:
            if len(matches) >= max_matches:
                break
            
            # Simple fuzzy matching: check if most characters match
            if self._simple_fuzzy_score(query, candidate) > 0.7:
                matches.append(candidate)
        
        return matches
    
    def _simple_fuzzy_score(self, s1: str, s2: str) -> float:
        """Simple fuzzy matching score."""
        if len(s1) == 0 or len(s2) == 0:
            return 0.0
        
        # Count matching characters
        matches = 0
        for char in s1:
            if char in s2:
                matches += 1
        
        return matches / max(len(s1), len(s2))
    
    def _get_keyword_matches(self, query: str, max_matches: int) -> List[str]:
        """Get matches from brand and category keywords."""
        matches = []
        
        # Check brand keywords
        for brand in self.brand_keywords:
            if len(matches) >= max_matches:
                break
            if query in brand or brand in query:
                matches.append(brand)
        
        # Check category keywords
        for category in self.category_keywords:
            if len(matches) >= max_matches:
                break
            if query in category or category in query:
                matches.append(category)
        
        return matches