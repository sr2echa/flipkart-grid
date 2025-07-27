import time
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrieNode:
    """Enhanced Trie node with additional metadata."""
    children: Dict[str, 'TrieNode']
    is_end: bool
    frequency: int
    recency_score: float
    last_updated: datetime
    suggestions: List[str]
    metadata: Dict

class CompressedTrieNode:
    """Compressed Trie node for memory efficiency."""
    def __init__(self):
        self.children: Dict[str, 'CompressedTrieNode'] = {}
        self.is_end: bool = False
        self.frequency: int = 0
        self.recency_score: float = 0.0
        self.last_updated: datetime = datetime.now()
        self.suggestions: List[str] = []
        self.metadata: Dict = {}
        self.compressed_edge: str = ""
        self.parent: Optional['CompressedTrieNode'] = None

class EnhancedTrieAutosuggest:
    """
    Enhanced Trie-based autosuggest with optimizations:
    - Compressed Trie (Radix Tree) for memory efficiency
    - Recency scoring for suggestions
    - Partial word matching with Levenshtein distance
    - Incremental updates
    - Caching for performance
    """
    
    def __init__(self, 
                 enable_compression: bool = True,
                 enable_recency: bool = True,
                 enable_partial_matching: bool = True,
                 cache_dir: str = 'cache',
                 max_suggestions: int = 10,
                 recency_decay_days: int = 30):
        """
        Initialize enhanced Trie autosuggest.
        
        Args:
            enable_compression: Enable compressed Trie for memory efficiency
            enable_recency: Enable recency scoring
            enable_partial_matching: Enable partial word matching
            cache_dir: Directory for caching
            max_suggestions: Maximum number of suggestions to return
            recency_decay_days: Days for recency score decay
        """
        self.enable_compression = enable_compression
        self.enable_recency = enable_recency
        self.enable_partial_matching = enable_partial_matching
        self.cache_dir = cache_dir
        self.max_suggestions = max_suggestions
        self.recency_decay_days = recency_decay_days
        
        # Initialize Trie
        if enable_compression:
            self.root = CompressedTrieNode()
        else:
            self.root = TrieNode(children={}, is_end=False, frequency=0, 
                               recency_score=0.0, last_updated=datetime.now(),
                               suggestions=[], metadata={})
        
        # Cache for performance
        self.suggestion_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Statistics
        self.total_queries = 0
        self.total_suggestions = 0
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load cached Trie if available
        self.load_cached_trie()
    
    def load_cached_trie(self):
        """Load cached Trie from disk."""
        cache_file = os.path.join(self.cache_dir, 'trie_cache.pkl')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.root = cached_data.get('root', self.root)
                    self.suggestion_cache = cached_data.get('cache', {})
                logger.info("Loaded cached Trie")
            except Exception as e:
                logger.warning(f"Failed to load cached Trie: {e}")
    
    def save_cached_trie(self):
        """Save Trie to cache."""
        cache_file = os.path.join(self.cache_dir, 'trie_cache.pkl')
        cached_data = {
            'root': self.root,
            'cache': self.suggestion_cache
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_data, f)
        logger.info("Saved Trie to cache")
    
    def calculate_recency_score(self, last_updated: datetime) -> float:
        """
        Calculate recency score based on last update time.
        
        Args:
            last_updated: Last update timestamp
            
        Returns:
            Recency score (0-1, higher is more recent)
        """
        if not self.enable_recency:
            return 1.0
        
        days_since_update = (datetime.now() - last_updated).days
        decay_factor = max(0, 1 - (days_since_update / self.recency_decay_days))
        return decay_factor
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Levenshtein distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Levenshtein distance
        """
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def find_partial_matches(self, prefix: str, node: TrieNode, 
                           max_distance: int = 2) -> List[Tuple[str, int, float]]:
        """
        Find partial matches using Levenshtein distance.
        
        Args:
            prefix: Query prefix
            node: Current Trie node
            max_distance: Maximum edit distance allowed
            
        Returns:
            List of (suggestion, distance, score) tuples
        """
        if not self.enable_partial_matching:
            return []
        
        matches = []
        
        def dfs(current_node: TrieNode, current_path: str):
            if current_node.is_end:
                distance = self.levenshtein_distance(prefix, current_path)
                if distance <= max_distance:
                    recency_score = self.calculate_recency_score(current_node.last_updated)
                    total_score = current_node.frequency * recency_score
                    matches.append((current_path, distance, total_score))
            
            for char, child in current_node.children.items():
                dfs(child, current_path + char)
        
        dfs(node, "")
        return matches
    
    def compress_trie(self, node: CompressedTrieNode):
        """
        Compress Trie by merging single-child nodes.
        
        Args:
            node: Root node to compress
        """
        if not self.enable_compression:
            return
        
        # Find nodes with single children and compress them
        for char, child in list(node.children.items()):
            self._compress_node(node, char, child)
    
    def _compress_node(self, parent: CompressedTrieNode, edge: str, node: CompressedTrieNode):
        """
        Compress a single node by merging with its parent.
        
        Args:
            parent: Parent node
            edge: Edge label
            node: Node to compress
        """
        if len(node.children) == 1 and not node.is_end:
            # Merge with single child
            child_char, child_node = list(node.children.items())[0]
            new_edge = edge + child_char
            
            # Update parent's children
            del parent.children[edge]
            parent.children[new_edge] = child_node
            child_node.compressed_edge = new_edge
            child_node.parent = parent
            
            # Recursively compress the child
            self._compress_node(parent, new_edge, child_node)
        else:
            # Recursively compress children
            for char, child in list(node.children.items()):
                self._compress_node(node, char, child)
    
    def insert_query(self, query: str, frequency: int = 1, 
                    metadata: Optional[Dict] = None):
        """
        Insert a query into the Trie with enhanced features.
        
        Args:
            query: Query to insert
            frequency: Query frequency
            metadata: Additional metadata
        """
        query = query.lower().strip()
        if not query:
            return
        
        if self.enable_compression:
            self._insert_compressed(query, frequency, metadata)
        else:
            self._insert_standard(query, frequency, metadata)
        
        # Clear cache for this prefix
        self._clear_cache_for_prefix(query)
    
    def _insert_standard(self, query: str, frequency: int, metadata: Optional[Dict]):
        """Insert into standard Trie."""
        node = self.root
        for char in query:
            if char not in node.children:
                node.children[char] = TrieNode(
                    children={}, is_end=False, frequency=0,
                    recency_score=0.0, last_updated=datetime.now(),
                    suggestions=[], metadata={}
                )
            node = node.children[char]
        
        # Update node
        node.is_end = True
        node.frequency += frequency
        node.last_updated = datetime.now()
        node.recency_score = self.calculate_recency_score(node.last_updated)
        
        if metadata:
            node.metadata.update(metadata)
        
        # Update suggestions list
        if query not in node.suggestions:
            node.suggestions.append(query)
            node.suggestions.sort(key=lambda x: self._get_suggestion_score(x), reverse=True)
            node.suggestions = node.suggestions[:self.max_suggestions]
    
    def _insert_compressed(self, query: str, frequency: int, metadata: Optional[Dict]):
        """Insert into compressed Trie."""
        node = self.root
        remaining = query
        
        while remaining:
            # Find matching edge
            matched_edge = None
            matched_node = None
            
            for edge, child in node.children.items():
                if remaining.startswith(edge):
                    matched_edge = edge
                    matched_node = child
                    break
            
            if matched_edge:
                # Follow existing edge
                node = matched_node
                remaining = remaining[len(matched_edge):]
            else:
                # Create new edge
                new_node = CompressedTrieNode()
                new_node.compressed_edge = remaining
                new_node.parent = node
                node.children[remaining] = new_node
                node = new_node
                remaining = ""
        
        # Update node
        node.is_end = True
        node.frequency += frequency
        node.last_updated = datetime.now()
        node.recency_score = self.calculate_recency_score(node.last_updated)
        
        if metadata:
            node.metadata.update(metadata)
        
        # Update suggestions
        if query not in node.suggestions:
            node.suggestions.append(query)
            node.suggestions.sort(key=lambda x: self._get_suggestion_score(x), reverse=True)
            node.suggestions = node.suggestions[:self.max_suggestions]
    
    def _get_suggestion_score(self, suggestion: str) -> float:
        """
        Calculate score for a suggestion.
        
        Args:
            suggestion: Suggestion string
            
        Returns:
            Suggestion score
        """
        # Find the node for this suggestion
        node = self._find_node(suggestion)
        if not node:
            return 0.0
        
        frequency_score = node.frequency
        recency_score = self.calculate_recency_score(node.last_updated) if self.enable_recency else 1.0
        
        # Combine scores (frequency * recency)
        return frequency_score * recency_score
    
    def _find_node(self, query: str):
        """Find node for a given query."""
        if self.enable_compression:
            return self._find_node_compressed(query)
        else:
            return self._find_node_standard(query)
    
    def _find_node_standard(self, query: str):
        """Find node in standard Trie."""
        node = self.root
        for char in query:
            if char not in node.children:
                return None
            node = node.children[char]
        return node if node.is_end else None
    
    def _find_node_compressed(self, query: str):
        """Find node in compressed Trie."""
        node = self.root
        remaining = query
        
        while remaining and node:
            # Find matching edge
            matched_edge = None
            matched_node = None
            
            for edge, child in node.children.items():
                if remaining.startswith(edge):
                    matched_edge = edge
                    matched_node = child
                    break
            
            if matched_edge:
                node = matched_node
                remaining = remaining[len(matched_edge):]
            else:
                return None
        
        return node if node and node.is_end else None
    
    def get_suggestions(self, prefix: str, max_suggestions: Optional[int] = None) -> List[Dict]:
        """
        Get autosuggestions for a prefix with enhanced features.
        
        Args:
            prefix: Query prefix
            max_suggestions: Maximum number of suggestions (overrides default)
            
        Returns:
            List of suggestion dictionaries
        """
        self.total_queries += 1
        prefix = prefix.lower().strip()
        
        if not prefix:
            return []
        
        # Check cache first
        cache_key = f"{prefix}_{max_suggestions or self.max_suggestions}"
        if cache_key in self.suggestion_cache:
            self.cache_hits += 1
            return self.suggestion_cache[cache_key]
        
        self.cache_misses += 1
        
        # Get suggestions
        suggestions = []
        
        if self.enable_compression:
            suggestions = self._get_suggestions_compressed(prefix, max_suggestions)
        else:
            suggestions = self._get_suggestions_standard(prefix, max_suggestions)
        
        # Add partial matches if enabled
        if self.enable_partial_matching and len(suggestions) < (max_suggestions or self.max_suggestions):
            partial_suggestions = self._get_partial_suggestions(prefix, max_suggestions)
            suggestions.extend(partial_suggestions)
        
        # Sort by score and limit
        suggestions.sort(key=lambda x: x['score'], reverse=True)
        suggestions = suggestions[:max_suggestions or self.max_suggestions]
        
        # Cache results
        self.suggestion_cache[cache_key] = suggestions
        
        self.total_suggestions += len(suggestions)
        return suggestions
    
    def _get_suggestions_standard(self, prefix: str, max_suggestions: Optional[int]) -> List[Dict]:
        """Get suggestions from standard Trie."""
        suggestions = []
        node = self.root
        
        # Navigate to prefix node
        for char in prefix:
            if char not in node.children:
                return suggestions
            node = node.children[char]
        
        # Collect suggestions from this node and its descendants
        self._collect_suggestions_standard(node, prefix, suggestions, max_suggestions)
        return suggestions
    
    def _get_suggestions_compressed(self, prefix: str, max_suggestions: Optional[int]) -> List[Dict]:
        """Get suggestions from compressed Trie."""
        suggestions = []
        node = self.root
        current_path = ""
        
        # Navigate to prefix node
        remaining = prefix
        while remaining and node:
            matched_edge = None
            matched_node = None
            
            for edge, child in node.children.items():
                if remaining.startswith(edge):
                    matched_edge = edge
                    matched_node = child
                    break
            
            if matched_edge:
                current_path += matched_edge
                node = matched_node
                remaining = remaining[len(matched_edge):]
            else:
                return suggestions
        
        # Collect suggestions
        self._collect_suggestions_compressed(node, current_path, suggestions, max_suggestions)
        return suggestions
    
    def _collect_suggestions_standard(self, node: TrieNode, current_path: str, 
                                    suggestions: List[Dict], max_suggestions: Optional[int]):
        """Collect suggestions from standard Trie node."""
        if node.is_end:
            score = self._get_suggestion_score(current_path)
            suggestions.append({
                'suggestion': current_path,
                'score': score,
                'frequency': node.frequency,
                'recency_score': node.recency_score,
                'metadata': node.metadata
            })
        
        if len(suggestions) >= (max_suggestions or self.max_suggestions):
            return
        
        for char, child in node.children.items():
            self._collect_suggestions_standard(child, current_path + char, suggestions, max_suggestions)
            if len(suggestions) >= (max_suggestions or self.max_suggestions):
                break
    
    def _collect_suggestions_compressed(self, node: CompressedTrieNode, current_path: str,
                                      suggestions: List[Dict], max_suggestions: Optional[int]):
        """Collect suggestions from compressed Trie node."""
        if node.is_end:
            score = self._get_suggestion_score(current_path)
            suggestions.append({
                'suggestion': current_path,
                'score': score,
                'frequency': node.frequency,
                'recency_score': node.recency_score,
                'metadata': node.metadata
            })
        
        if len(suggestions) >= (max_suggestions or self.max_suggestions):
            return
        
        for edge, child in node.children.items():
            self._collect_suggestions_compressed(child, current_path + edge, suggestions, max_suggestions)
            if len(suggestions) >= (max_suggestions or self.max_suggestions):
                break
    
    def _get_partial_suggestions(self, prefix: str, max_suggestions: Optional[int]) -> List[Dict]:
        """Get partial word suggestions using Levenshtein distance."""
        if not self.enable_partial_matching:
            return []
        
        partial_matches = []
        
        if self.enable_compression:
            # For compressed Trie, we need to traverse all nodes
            self._collect_partial_matches_compressed(self.root, "", prefix, partial_matches)
        else:
            # For standard Trie, we can use the existing method
            partial_matches = self.find_partial_matches(prefix, self.root)
        
        # Convert to suggestion format
        suggestions = []
        for match, distance, score in partial_matches:
            suggestions.append({
                'suggestion': match,
                'score': score * (1 - distance * 0.1),  # Penalize by distance
                'frequency': 0,  # Will be filled by _get_suggestion_score
                'recency_score': 1.0,
                'metadata': {'edit_distance': distance, 'is_partial_match': True}
            })
        
        return suggestions[:max_suggestions or self.max_suggestions]
    
    def _collect_partial_matches_compressed(self, node: CompressedTrieNode, current_path: str,
                                          prefix: str, matches: List[Tuple[str, int, float]]):
        """Collect partial matches from compressed Trie."""
        if node.is_end:
            distance = self.levenshtein_distance(prefix, current_path)
            if distance <= 2:  # Max distance of 2
                recency_score = self.calculate_recency_score(node.last_updated)
                total_score = node.frequency * recency_score
                matches.append((current_path, distance, total_score))
        
        for edge, child in node.children.items():
            self._collect_partial_matches_compressed(child, current_path + edge, prefix, matches)
    
    def _clear_cache_for_prefix(self, prefix: str):
        """Clear cache entries that start with the given prefix."""
        keys_to_remove = [key for key in self.suggestion_cache.keys() if key.startswith(prefix)]
        for key in keys_to_remove:
            del self.suggestion_cache[key]
    
    def build_from_queries(self, queries: List[Tuple[str, int, Optional[Dict]]]):
        """
        Build Trie from a list of queries.
        
        Args:
            queries: List of (query, frequency, metadata) tuples
        """
        logger.info(f"Building Trie from {len(queries)} queries...")
        
        for query, frequency, metadata in queries:
            self.insert_query(query, frequency, metadata)
        
        # Compress Trie if enabled
        if self.enable_compression:
            self.compress_trie(self.root)
        
        # Save to cache
        self.save_cached_trie()
        
        logger.info("Trie construction completed")
    
    def get_statistics(self) -> Dict:
        """Get Trie statistics."""
        return {
            'total_queries': self.total_queries,
            'total_suggestions': self.total_suggestions,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'compression_enabled': self.enable_compression,
            'recency_enabled': self.enable_recency,
            'partial_matching_enabled': self.enable_partial_matching
        }
    
    def clear_cache(self):
        """Clear suggestion cache."""
        self.suggestion_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Cache cleared")

# Example usage and testing
if __name__ == "__main__":
    # Create enhanced Trie autosuggest
    trie = EnhancedTrieAutosuggest(
        enable_compression=True,
        enable_recency=True,
        enable_partial_matching=True,
        max_suggestions=10
    )
    
    # Sample queries with frequencies
    sample_queries = [
        ("samsung", 1500, {"category": "Electronics"}),
        ("samsung galaxy", 800, {"category": "Electronics"}),
        ("samsung phone", 600, {"category": "Electronics"}),
        ("apple", 1200, {"category": "Electronics"}),
        ("apple iphone", 900, {"category": "Electronics"}),
        ("nike", 1000, {"category": "Fashion"}),
        ("nike shoes", 700, {"category": "Fashion"}),
        ("adidas", 800, {"category": "Fashion"}),
        ("laptop", 1200, {"category": "Electronics"}),
        ("smartphone", 1100, {"category": "Electronics"}),
        ("samsng", 200, {"category": "Electronics", "is_typo": True}),  # Typo
        ("aple", 150, {"category": "Electronics", "is_typo": True}),   # Typo
    ]
    
    # Build Trie
    trie.build_from_queries(sample_queries)
    
    # Test suggestions
    test_prefixes = ["sam", "sams", "samsng", "ap", "nike", "lap"]
    
    print("Testing Enhanced Trie Autosuggest:")
    print("=" * 50)
    
    for prefix in test_prefixes:
        suggestions = trie.get_suggestions(prefix)
        print(f"\nPrefix: '{prefix}'")
        print(f"Suggestions ({len(suggestions)}):")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion['suggestion']} (score: {suggestion['score']:.2f}, "
                  f"freq: {suggestion['frequency']}, recency: {suggestion['recency_score']:.2f})")
    
    # Print statistics
    print("\n" + "=" * 50)
    print("Statistics:")
    stats = trie.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}") 