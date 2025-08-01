import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from collections import defaultdict
import re

class TrieNode:
    """Enhanced Trie node with frequency tracking and metadata."""
    
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.frequency = 0
        self.query = None
        self.metadata = {}  # Additional metadata like category, brand, etc.

class EnhancedTrieAutosuggest:
    """Enhanced Trie-based autosuggest with improved features and performance."""
    
    def __init__(self):
        self.root = TrieNode()
        self.user_queries = None
        self.query_metadata = {}
        self.category_boost = {}
        self.brand_boost = {}
        self.total_queries = 0
        
    def build_trie(self, user_queries_df: pd.DataFrame):
        """Build enhanced Trie from user queries DataFrame."""
        print("Building Enhanced Trie from user queries...")
        
        self.user_queries = user_queries_df
        self.total_queries = len(user_queries_df)
        
        # Build category and brand boost mappings
        self._build_boost_mappings()
        
        # Insert each corrected_query into the Trie
        for _, row in user_queries_df.iterrows():
            query = row['corrected_query']
            frequency = row['frequency']
            
            if pd.isna(query) or pd.isna(frequency):
                continue
            
            # Extract metadata
            metadata = self._extract_metadata(query, row)
            
            self._insert(query, frequency, metadata)
        
        print(f"Enhanced Trie built with {len(user_queries_df)} queries")
        print(f"Total unique prefixes: {self._count_nodes()}")
    
    def _build_boost_mappings(self):
        """Build boost mappings for categories and brands."""
        # Popular categories get higher boost
        category_keywords = {
            'phone': 1.5, 'smartphone': 1.5, 'mobile': 1.5,
            'laptop': 1.4, 'computer': 1.4,
            'headphones': 1.3, 'earbuds': 1.3,
            'shoes': 1.3, 'sneakers': 1.3,
            'watch': 1.2, 'smartwatch': 1.2,
            'tablet': 1.2, 'tv': 1.2
        }
        
        # Popular brands get higher boost
        brand_keywords = {
            'samsung': 1.5, 'apple': 1.5, 'nike': 1.4,
            'adidas': 1.4, 'sony': 1.3, 'oneplus': 1.3,
            'xiaomi': 1.2, 'dell': 1.2, 'hp': 1.2
        }
        
        self.category_boost = category_keywords
        self.brand_boost = brand_keywords
    
    def _extract_metadata(self, query: str, row: pd.Series) -> Dict:
        """Extract metadata from query and row data."""
        metadata = {}
        
        # Extract category information
        for category, boost in self.category_boost.items():
            if category in query.lower():
                metadata['category'] = category
                metadata['category_boost'] = boost
                break
        
        # Extract brand information
        for brand, boost in self.brand_boost.items():
            if brand in query.lower():
                metadata['brand'] = brand
                metadata['brand_boost'] = boost
                break
        
        # Extract price information
        price_patterns = [
            r'under (\d+)', r'above (\d+)', r'below (\d+)',
            r'between (\d+) and (\d+)'
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, query.lower())
            if match:
                metadata['has_price'] = True
                metadata['price_numbers'] = match.groups()
                break
        
        # Extract attributes
        attributes = ['wireless', 'bluetooth', 'gaming', 'smart', 'premium', 'budget']
        for attr in attributes:
            if attr in query.lower():
                metadata['attributes'] = metadata.get('attributes', []) + [attr]
        
        return metadata
    
    def _insert(self, query: str, frequency: int, metadata: Dict = None):
        """Insert a query into the Enhanced Trie with metadata."""
        node = self.root
        
        # Convert to lowercase for consistency
        query = query.lower().strip()
        
        for char in query:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_end_of_word = True
        node.frequency = frequency
        node.query = query
        node.metadata = metadata or {}
        
        # Store in query metadata dictionary for quick lookup
        self.query_metadata[query] = metadata or {}
    
    def _search_prefix(self, prefix: str) -> List[Tuple[str, int, Dict]]:
        """Search for all queries that start with the given prefix."""
        node = self.root
        prefix = prefix.lower().strip()
        
        # Navigate to the end of the prefix
        for char in prefix:
            if char not in node.children:
                return []  # Prefix not found
            node = node.children[char]
        
        # Collect all words that start with this prefix
        results = []
        self._collect_words(node, prefix, results)
        
        return results
    
    def _collect_words(self, node: TrieNode, prefix: str, results: List[Tuple[str, int, Dict]]):
        """Collect all words from the current node."""
        if node.is_end_of_word:
            results.append((node.query, node.frequency, node.metadata))
        
        for char, child_node in node.children.items():
            self._collect_words(child_node, prefix + char, results)
    
    def get_suggestions(self, prefix: str, max_suggestions: int = 10) -> List[str]:
        """Get autosuggestions for a given prefix."""
        suggestions_with_scores = self.get_suggestions_with_scores(prefix, max_suggestions)
        return [suggestion for suggestion, _ in suggestions_with_scores]
    
    def get_suggestions_with_scores(self, prefix: str, max_suggestions: int = 10, 
                                  context: Dict = None) -> List[Tuple[str, float]]:
        """Get autosuggestions with enhanced scoring."""
        if not prefix.strip():
            return []
        
        # Convert to lowercase for consistency
        prefix = prefix.lower().strip()
        
        # Search for matching queries
        matches = self._search_prefix(prefix)
        
        if not matches:
            return []
        
        # Enhanced scoring
        scored_matches = []
        for query, frequency, metadata in matches:
            score = self._calculate_enhanced_score(query, frequency, metadata, prefix, context)
            scored_matches.append((query, score))
        
        # Sort by enhanced score (descending)
        scored_matches.sort(key=lambda x: x[1], reverse=True)
        
        # Return top suggestions
        return scored_matches[:max_suggestions]
    
    def _calculate_enhanced_score(self, query: str, frequency: int, metadata: Dict, 
                                prefix: str, context: Dict = None) -> float:
        """Calculate enhanced score for a query."""
        base_score = frequency
        
        # Frequency normalization (log scale)
        normalized_frequency = np.log1p(frequency) / np.log1p(self.total_queries)
        score = normalized_frequency
        
        # Prefix match quality boost
        if query.startswith(prefix):
            prefix_quality = len(prefix) / len(query)
            score *= (1 + prefix_quality)
        
        # Category boost
        if 'category_boost' in metadata:
            score *= metadata['category_boost']
        
        # Brand boost
        if 'brand_boost' in metadata:
            score *= metadata['brand_boost']
        
        # Attribute boost
        if 'attributes' in metadata:
            attribute_boost = 1 + len(metadata['attributes']) * 0.1
            score *= attribute_boost
        
        # Context-based boosting
        if context:
            context_boost = self._get_context_boost(query, metadata, context)
            score *= context_boost
        
        # Price-related boost (price queries are often important)
        if 'has_price' in metadata:
            score *= 1.2
        
        # Recency boost (more recent queries get slight boost)
        # This could be implemented with timestamp data
        
        return float(score)
    
    def _get_context_boost(self, query: str, metadata: Dict, context: Dict) -> float:
        """Calculate context-based boost."""
        boost = 1.0
        
        # Session context boost
        if 'previous_queries' in context:
            for prev_query in context['previous_queries']:
                # If previous query shares category/brand, boost this query
                if metadata.get('category') and metadata['category'] in prev_query:
                    boost *= 1.3
                if metadata.get('brand') and metadata['brand'] in prev_query:
                    boost *= 1.3
        
        # Location context boost
        if 'location' in context:
            location = context['location'].lower()
            # Boost tech products in tech cities
            if location in ['bangalore', 'mumbai', 'delhi'] and metadata.get('category') in ['laptop', 'phone', 'smartphone']:
                boost *= 1.2
        
        # Event context boost
        if 'event' in context:
            event = context['event']
            if event == 'diwali' and any(word in query for word in ['lights', 'decor', 'gifts']):
                boost *= 1.5
            elif event == 'ipl' and any(word in query for word in ['jersey', 'cricket', 'sports']):
                boost *= 1.5
        
        return boost
    
    def _count_nodes(self) -> int:
        """Count total number of nodes in the Trie."""
        def count_recursive(node):
            count = 1
            for child in node.children.values():
                count += count_recursive(child)
            return count
        
        return count_recursive(self.root)
    
    def get_analytics(self) -> Dict:
        """Get analytics about the Trie structure."""
        total_nodes = self._count_nodes()
        total_queries = len([q for q in self.query_metadata.keys()])
        
        # Category distribution
        category_dist = defaultdict(int)
        brand_dist = defaultdict(int)
        
        for metadata in self.query_metadata.values():
            if 'category' in metadata:
                category_dist[metadata['category']] += 1
            if 'brand' in metadata:
                brand_dist[metadata['brand']] += 1
        
        return {
            'total_nodes': total_nodes,
            'total_queries': total_queries,
            'categories': dict(category_dist),
            'brands': dict(brand_dist),
            'memory_efficiency': total_queries / total_nodes if total_nodes > 0 else 0
        }
    
    def get_prefix_suggestions_fast(self, prefix: str, max_suggestions: int = 5) -> List[Tuple[str, float]]:
        """Get suggestions with optimized performance for real-time use."""
        if not prefix.strip() or len(prefix) < 2:
            return []
        
        # Use a simpler scoring for very fast response
        matches = self._search_prefix(prefix)
        
        if not matches:
            return []
        
        # Simple frequency-based scoring
        scored_matches = [(query, float(frequency)) for query, frequency, _ in matches]
        scored_matches.sort(key=lambda x: x[1], reverse=True)
        
        return scored_matches[:max_suggestions]

# Test the enhanced Trie
if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from data_preprocessing import DataPreprocessor
    
    # Load data
    preprocessor = DataPreprocessor()
    preprocessor.run_all_preprocessing()
    data = preprocessor.get_processed_data()
    
    # Test enhanced Trie
    enhanced_trie = EnhancedTrieAutosuggest()
    enhanced_trie.build_trie(data['user_queries'])
    
    print("\n=== Enhanced Trie Test ===")
    
    # Test basic functionality
    test_queries = ['sam', 'app', 'laptop', 'nike', 'phone']
    
    for query in test_queries:
        start_time = time.time()
        suggestions = enhanced_trie.get_suggestions_with_scores(query, max_suggestions=5)
        end_time = time.time()
        
        print(f"\nQuery: '{query}'")
        print(f"Response time: {(end_time - start_time) * 1000:.2f}ms")
        print("Suggestions:")
        for i, (suggestion, score) in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion} (score: {score:.4f})")
    
    # Test contextual suggestions
    print(f"\n=== Contextual Test ===")
    context = {
        'previous_queries': ['gaming', 'laptop'],
        'location': 'Bangalore',
        'event': 'diwali'
    }
    
    suggestions = enhanced_trie.get_suggestions_with_scores('laptop', context=context)
    print(f"Contextual suggestions for 'laptop':")
    for i, (suggestion, score) in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion} (score: {score:.4f})")
    
    # Show analytics
    analytics = enhanced_trie.get_analytics()
    print(f"\n=== Analytics ===")
    print(f"Total nodes: {analytics['total_nodes']}")
    print(f"Total queries: {analytics['total_queries']}")
    print(f"Memory efficiency: {analytics['memory_efficiency']:.3f}")
    print(f"Top categories: {list(analytics['categories'].items())[:5]}")
    print(f"Top brands: {list(analytics['brands'].items())[:5]}")