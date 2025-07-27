import pandas as pd
from typing import List, Dict, Tuple, Optional
import time

class TrieNode:
    """Node in the Trie data structure."""
    
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.frequency = 0
        self.query = None

class TrieAutosuggest:
    """Trie-based autosuggest system for prefix matching."""
    
    def __init__(self):
        self.root = TrieNode()
        self.user_queries = None
        
    def build_trie(self, user_queries_df: pd.DataFrame):
        """Build Trie from user queries DataFrame."""
        print("Building Trie from user queries...")
        
        self.user_queries = user_queries_df
        
        # Insert each corrected_query into the Trie
        for _, row in user_queries_df.iterrows():
            query = row['corrected_query']
            frequency = row['frequency']
            
            if pd.isna(query) or pd.isna(frequency):
                continue
                
            self._insert(query, frequency)
        
        print(f"Trie built with {len(user_queries_df)} queries")
    
    def _insert(self, query: str, frequency: int):
        """Insert a query into the Trie."""
        node = self.root
        
        for char in query:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_end_of_word = True
        node.frequency = frequency
        node.query = query
    
    def _search_prefix(self, prefix: str) -> List[Tuple[str, int]]:
        """Search for all queries that start with the given prefix."""
        node = self.root
        
        # Navigate to the end of the prefix
        for char in prefix:
            if char not in node.children:
                return []  # Prefix not found
            node = node.children[char]
        
        # Collect all words that start with this prefix
        results = []
        self._collect_words(node, prefix, results)
        
        return results
    
    def _collect_words(self, node: TrieNode, prefix: str, results: List[Tuple[str, int]]):
        """Recursively collect all words starting with the prefix."""
        if node.is_end_of_word:
            results.append((node.query, node.frequency))
        
        for char, child_node in node.children.items():
            self._collect_words(child_node, prefix + char, results)
    
    def get_suggestions(self, prefix: str, max_suggestions: int = 10) -> List[str]:
        """Get autosuggestions for a given prefix."""
        if not prefix.strip():
            return []
        
        # Convert to lowercase for consistency
        prefix = prefix.lower().strip()
        
        # Search for matching queries
        matches = self._search_prefix(prefix)
        
        # Sort by frequency (descending) and then alphabetically
        matches.sort(key=lambda x: (-x[1], x[0]))
        
        # Return top suggestions
        suggestions = [query for query, _ in matches[:max_suggestions]]
        
        return suggestions
    
    def get_suggestions_with_scores(self, prefix: str, max_suggestions: int = 10) -> List[Tuple[str, int]]:
        """Get autosuggestions with their frequency scores."""
        if not prefix.strip():
            return []
        
        # Convert to lowercase for consistency
        prefix = prefix.lower().strip()
        
        # Search for matching queries
        matches = self._search_prefix(prefix)
        
        # Sort by frequency (descending) and then alphabetically
        matches.sort(key=lambda x: (-x[1], x[0]))
        
        # Return top suggestions with scores
        return matches[:max_suggestions]

# Test the Trie autosuggest
if __name__ == "__main__":
    # Load and preprocess data
    from data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    preprocessor.run_all_preprocessing()
    data = preprocessor.get_processed_data()
    
    # Initialize and build Trie
    trie_autosuggest = TrieAutosuggest()
    trie_autosuggest.build_trie(data['user_queries'])
    
    # Test cases
    test_prefixes = [
        "sam",      # Should match "samsung"
        "app",      # Should match "apple"
        "nik",      # Should match "nike"
        "smart",    # Should match "smartphone", "smartwatch"
        "lap",      # Should match "laptop"
        "head",     # Should match "headphones"
        "sho",      # Should match "shoes"
        "tv",       # Should match "tv"
        "phone",    # Should match "mobile phone"
        "ear",      # Should match "earbuds"
        "key",      # Should match "keyboard"
        "char",     # Should match "charger"
        "watch",    # Should match "watch", "smartwatch"
        "tab",      # Should match "tablet"
        "cam",      # Should match "camera"
        "speak",    # Should match "speaker"
        "mous",     # Should match "mouse"
        "case",     # Should match "case"
        "bag",      # Should match "bag"
        "wallet",   # Should match "wallet"
        "hood",     # Should match "hoodie"
        "jean",     # Should match "jeans"
        "shirt",    # Should match "shirt", "t shirt"
        "sneak",    # Should match "sneakers"
        "notebook", # Should match "notebook"
        "televis",  # Should match "television"
        "mobil",    # Should match "mobile phone"
        "smartphon", # Should match "smartphone"
        "headphon", # Should match "headphones"
        "earbud",   # Should match "earbuds"
        "televisn", # Should match "television"
        "sneakr",   # Should match "sneakers"
        "smartwach", # Should match "smartwatch"
        "tablit",   # Should match "tablet"
        "camra",    # Should match "camera"
        "speakr",   # Should match "speaker"
        "keybord",  # Should match "keyboard"
        "chargr",   # Should match "charger"
        "hoodi",    # Should match "hoodie"
        "jens",     # Should match "jeans"
        "notbook",  # Should match "notebook"
        "shoos",    # Should match "shoes"
        "wach",     # Should match "watch"
        "shrt",     # Should match "shirt"
        "walet",    # Should match "wallet"
        "mous",     # Should match "mouse"
        "cas",      # Should match "case"
        "bg",       # Should match "bag"
    ]
    
    print("\n=== Trie Autosuggest Test Results ===")
    
    for prefix in test_prefixes:
        start_time = time.time()
        suggestions = trie_autosuggest.get_suggestions(prefix)
        end_time = time.time()
        
        print(f"\nPrefix: '{prefix}'")
        print(f"Suggestions ({len(suggestions)}): {suggestions}")
        print(f"Response time: {(end_time - start_time)*1000:.2f}ms")
        
        # Test with scores
        suggestions_with_scores = trie_autosuggest.get_suggestions_with_scores(prefix)
        print(f"Top suggestions with scores: {suggestions_with_scores[:3]}")
    
    # Test performance
    print(f"\n=== Performance Test ===")
    test_query = "smart"
    start_time = time.time()
    for _ in range(1000):
        trie_autosuggest.get_suggestions(test_query)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 1000 * 1000
    print(f"Average response time for '{test_query}': {avg_time:.2f}ms")
    print(f"QPS: {1000 / (end_time - start_time):.0f}") 