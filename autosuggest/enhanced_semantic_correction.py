import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import pickle
import os
import time
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import re
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastSpellChecker:
    """
    Fast edit-distance based spell checker using BK-Tree and SymSpell-inspired approach.
    """
    
    def __init__(self, max_distance: int = 2):
        """
        Initialize fast spell checker.
        
        Args:
            max_distance: Maximum edit distance for corrections
        """
        self.max_distance = max_distance
        self.vocabulary = set()
        self.word_frequencies = {}
        self.bk_tree = None
        
    def add_words(self, words: List[str], frequencies: Optional[List[int]] = None):
        """
        Add words to the vocabulary.
        
        Args:
            words: List of words to add
            frequencies: Optional list of word frequencies
        """
        for i, word in enumerate(words):
            self.vocabulary.add(word.lower())
            if frequencies and i < len(frequencies):
                self.word_frequencies[word.lower()] = frequencies[i]
            else:
                self.word_frequencies[word.lower()] = self.word_frequencies.get(word.lower(), 0) + 1
        
        # Build BK-Tree for efficient search
        self._build_bk_tree()
    
    def _build_bk_tree(self):
        """Build BK-Tree for efficient edit distance search."""
        if not self.vocabulary:
            return
        
        words = list(self.vocabulary)
        self.bk_tree = BKTreeNode(words[0])
        
        for word in words[1:]:
            self.bk_tree.insert(word)
    
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
    
    def find_corrections(self, word: str, max_distance: Optional[int] = None) -> List[Tuple[str, int, int]]:
        """
        Find corrections for a word using BK-Tree.
        
        Args:
            word: Word to correct
            max_distance: Maximum edit distance (overrides default)
            
        Returns:
            List of (corrected_word, distance, frequency) tuples
        """
        if not self.bk_tree:
            return []
        
        max_dist = max_distance or self.max_distance
        corrections = []
        
        def search_bk_tree(node, target, max_dist):
            if not node:
                return
            
            distance = self.levenshtein_distance(node.word, target)
            if distance <= max_dist:
                frequency = self.word_frequencies.get(node.word, 0)
                corrections.append((node.word, distance, frequency))
            
            # Search children within distance range
            for child_distance, child in node.children.items():
                if abs(distance - child_distance) <= max_dist:
                    search_bk_tree(child, target, max_dist)
        
        search_bk_tree(self.bk_tree, word.lower(), max_dist)
        
        # Sort by distance, then by frequency
        corrections.sort(key=lambda x: (x[1], -x[2]))
        return corrections

class BKTreeNode:
    """BK-Tree node for efficient edit distance search."""
    
    def __init__(self, word: str):
        self.word = word
        self.children = {}  # distance -> child_node
    
    def insert(self, word: str):
        """Insert a word into the BK-Tree."""
        distance = self._levenshtein_distance(self.word, word)
        
        if distance == 0:
            return  # Word already exists
        
        if distance in self.children:
            self.children[distance].insert(word)
        else:
            self.children[distance] = BKTreeNode(word)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
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

class UserCorrectionHistory:
    """
    Track user-specific correction history for personalized suggestions.
    """
    
    def __init__(self, cache_dir: str = 'cache'):
        """
        Initialize user correction history.
        
        Args:
            cache_dir: Directory for caching history
        """
        self.cache_dir = cache_dir
        self.user_corrections = defaultdict(list)  # user_id -> [(typo, correction, timestamp)]
        self.global_corrections = Counter()  # (typo, correction) -> count
        
        os.makedirs(cache_dir, exist_ok=True)
        self.load_history()
    
    def add_correction(self, user_id: str, typo: str, correction: str, accepted: bool = True):
        """
        Add a correction to user history.
        
        Args:
            user_id: User identifier
            typo: Original typo
            correction: Suggested correction
            accepted: Whether user accepted the correction
        """
        timestamp = datetime.now()
        
        # Add to user history
        self.user_corrections[user_id].append({
            'typo': typo.lower(),
            'correction': correction.lower(),
            'timestamp': timestamp,
            'accepted': accepted
        })
        
        # Add to global corrections if accepted
        if accepted:
            self.global_corrections[(typo.lower(), correction.lower())] += 1
        
        # Keep only recent history (last 100 corrections per user)
        if len(self.user_corrections[user_id]) > 100:
            self.user_corrections[user_id] = self.user_corrections[user_id][-100:]
    
    def get_user_corrections(self, user_id: str, typo: str) -> List[Tuple[str, float]]:
        """
        Get user-specific corrections for a typo.
        
        Args:
            user_id: User identifier
            typo: Typo to correct
            
        Returns:
            List of (correction, confidence) tuples
        """
        if user_id not in self.user_corrections:
            return []
        
        corrections = []
        typo_lower = typo.lower()
        
        for correction_data in self.user_corrections[user_id]:
            if correction_data['typo'] == typo_lower and correction_data['accepted']:
                # Calculate confidence based on recency and frequency
                days_ago = (datetime.now() - correction_data['timestamp']).days
                recency_factor = max(0, 1 - (days_ago / 30))  # Decay over 30 days
                
                corrections.append((correction_data['correction'], recency_factor))
        
        # Sort by confidence
        corrections.sort(key=lambda x: x[1], reverse=True)
        return corrections
    
    def get_global_corrections(self, typo: str) -> List[Tuple[str, int]]:
        """
        Get global corrections for a typo.
        
        Args:
            typo: Typo to correct
            
        Returns:
            List of (correction, frequency) tuples
        """
        typo_lower = typo.lower()
        corrections = []
        
        for (t, c), count in self.global_corrections.items():
            if t == typo_lower:
                corrections.append((c, count))
        
        # Sort by frequency
        corrections.sort(key=lambda x: x[1], reverse=True)
        return corrections
    
    def save_history(self):
        """Save correction history to disk."""
        history_file = os.path.join(self.cache_dir, 'correction_history.pkl')
        history_data = {
            'user_corrections': dict(self.user_corrections),
            'global_corrections': dict(self.global_corrections)
        }
        with open(history_file, 'wb') as f:
            pickle.dump(history_data, f)
        logger.info("Saved correction history")
    
    def load_history(self):
        """Load correction history from disk."""
        history_file = os.path.join(self.cache_dir, 'correction_history.pkl')
        if os.path.exists(history_file):
            try:
                with open(history_file, 'rb') as f:
                    history_data = pickle.load(f)
                    self.user_corrections = defaultdict(list, history_data.get('user_corrections', {}))
                    self.global_corrections = Counter(history_data.get('global_corrections', {}))
                logger.info("Loaded correction history")
            except Exception as e:
                logger.warning(f"Failed to load correction history: {e}")

class EnhancedSemanticCorrection:
    """
    Enhanced semantic correction combining SBERT with fast spell checking.
    Implements quantization, hybrid approach, and user-specific corrections.
    """
    
    def __init__(self, 
                 sbert_model_name: str = 'all-MiniLM-L6-v2',
                 enable_quantization: bool = True,
                 enable_spell_check: bool = True,
                 enable_user_history: bool = True,
                 cache_dir: str = 'cache',
                 similarity_threshold: float = 0.7,
                 max_corrections: int = 5):
        """
        Initialize enhanced semantic correction.
        
        Args:
            sbert_model_name: SBERT model name
            enable_quantization: Enable model quantization
            enable_spell_check: Enable fast spell checking
            enable_user_history: Enable user-specific corrections
            cache_dir: Directory for caching
            similarity_threshold: Minimum similarity for semantic correction
            max_corrections: Maximum number of corrections to return
        """
        self.enable_quantization = enable_quantization
        self.enable_spell_check = enable_spell_check
        self.enable_user_history = enable_user_history
        self.cache_dir = cache_dir
        self.similarity_threshold = similarity_threshold
        self.max_corrections = max_corrections
        
        # Initialize components
        logger.info(f"Loading SBERT model: {sbert_model_name}")
        self.sbert_model = SentenceTransformer(sbert_model_name)
        
        if enable_quantization:
            self._quantize_model()
        
        if enable_spell_check:
            self.spell_checker = FastSpellChecker()
        
        if enable_user_history:
            self.user_history = UserCorrectionHistory(cache_dir)
        
        # Cache for embeddings and corrections
        self.embedding_cache = {}
        self.correction_cache = {}
        
        # Statistics
        self.total_queries = 0
        self.spell_check_corrections = 0
        self.semantic_corrections = 0
        self.user_history_corrections = 0
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load cached data
        self.load_cached_data()
    
    def _quantize_model(self):
        """Apply quantization to SBERT model for faster inference."""
        try:
            # Convert model to float16 for quantization
            self.sbert_model.half()
            logger.info("Applied quantization to SBERT model")
        except Exception as e:
            logger.warning(f"Failed to quantize model: {e}")
    
    def load_cached_data(self):
        """Load cached embeddings and corrections."""
        cache_file = os.path.join(self.cache_dir, 'semantic_correction_cache.pkl')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.embedding_cache = cached_data.get('embeddings', {})
                    self.correction_cache = cached_data.get('corrections', {})
                logger.info(f"Loaded cached data: {len(self.embedding_cache)} embeddings, "
                          f"{len(self.correction_cache)} corrections")
            except Exception as e:
                logger.warning(f"Failed to load cached data: {e}")
    
    def save_cached_data(self):
        """Save embeddings and corrections to cache."""
        cache_file = os.path.join(self.cache_dir, 'semantic_correction_cache.pkl')
        cached_data = {
            'embeddings': self.embedding_cache,
            'corrections': self.correction_cache
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_data, f)
        logger.info("Saved cached data")
    
    def build_vocabulary(self, queries: List[str], frequencies: Optional[List[int]] = None):
        """
        Build vocabulary for spell checking.
        
        Args:
            queries: List of queries to build vocabulary from
            frequencies: Optional list of query frequencies
        """
        if not self.enable_spell_check:
            return
        
        logger.info(f"Building vocabulary from {len(queries)} queries...")
        
        # Extract individual words
        words = []
        word_frequencies = []
        
        for i, query in enumerate(queries):
            query_words = query.lower().split()
            for word in query_words:
                if len(word) > 2:  # Only words longer than 2 characters
                    words.append(word)
                    if frequencies and i < len(frequencies):
                        word_frequencies.append(frequencies[i])
                    else:
                        word_frequencies.append(1)
        
        # Add words to spell checker
        self.spell_checker.add_words(words, word_frequencies)
        logger.info(f"Built vocabulary with {len(self.spell_checker.vocabulary)} words")
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for texts with caching.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Embeddings array
        """
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            if text in self.embedding_cache:
                embeddings.append(self.embedding_cache[text])
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                embeddings.append(None)  # Placeholder
        
        # Get embeddings for uncached texts
        if uncached_texts:
            try:
                new_embeddings = self.sbert_model.encode(uncached_texts)
                
                # Cache new embeddings
                for i, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                    self.embedding_cache[text] = embedding
                    embeddings[uncached_indices[i]] = embedding
                    
            except Exception as e:
                logger.error(f"Failed to get embeddings: {e}")
                return np.array([])
        
        return np.array(embeddings)
    
    def correct_query(self, query: str, user_id: Optional[str] = None, 
                     candidate_queries: Optional[List[str]] = None) -> List[Dict]:
        """
        Correct a query using hybrid approach.
        
        Args:
            query: Query to correct
            user_id: Optional user identifier for personalized corrections
            candidate_queries: Optional list of candidate queries for semantic search
            
        Returns:
            List of correction dictionaries
        """
        self.total_queries += 1
        query = query.lower().strip()
        
        if not query:
            return []
        
        # Check cache first
        cache_key = f"{query}_{user_id or 'anonymous'}"
        if cache_key in self.correction_cache:
            return self.correction_cache[cache_key]
        
        corrections = []
        
        # Step 1: Fast spell checking for obvious typos
        if self.enable_spell_check:
            spell_corrections = self._get_spell_check_corrections(query)
            corrections.extend(spell_corrections)
            self.spell_check_corrections += len(spell_corrections)
        
        # Step 2: User-specific corrections
        if self.enable_user_history and user_id:
            user_corrections = self._get_user_corrections(user_id, query)
            corrections.extend(user_corrections)
            self.user_history_corrections += len(user_corrections)
        
        # Step 3: Semantic correction using SBERT
        if candidate_queries:
            semantic_corrections = self._get_semantic_corrections(query, candidate_queries)
            corrections.extend(semantic_corrections)
            self.semantic_corrections += len(semantic_corrections)
        
        # Remove duplicates and sort by confidence
        unique_corrections = self._deduplicate_corrections(corrections)
        unique_corrections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Limit results
        result = unique_corrections[:self.max_corrections]
        
        # Cache results
        self.correction_cache[cache_key] = result
        
        return result
    
    def _get_spell_check_corrections(self, query: str) -> List[Dict]:
        """Get spell check corrections."""
        if not self.enable_spell_check:
            return []
        
        corrections = []
        words = query.split()
        
        for i, word in enumerate(words):
            if len(word) <= 2:
                continue
            
            word_corrections = self.spell_checker.find_corrections(word, max_distance=2)
            
            for corrected_word, distance, frequency in word_corrections[:3]:  # Top 3 corrections
                if corrected_word != word:
                    # Create corrected query
                    corrected_words = words.copy()
                    corrected_words[i] = corrected_word
                    corrected_query = ' '.join(corrected_words)
                    
                    # Calculate confidence based on distance and frequency
                    distance_penalty = 1 - (distance * 0.2)  # 20% penalty per edit distance
                    frequency_factor = min(frequency / 100, 1.0)  # Normalize frequency
                    confidence = distance_penalty * frequency_factor
                    
                    corrections.append({
                        'original': query,
                        'corrected': corrected_query,
                        'confidence': confidence,
                        'method': 'spell_check',
                        'metadata': {
                            'corrected_word': corrected_word,
                            'original_word': word,
                            'edit_distance': distance,
                            'word_frequency': frequency
                        }
                    })
        
        return corrections
    
    def _get_user_corrections(self, user_id: str, query: str) -> List[Dict]:
        """Get user-specific corrections."""
        if not self.enable_user_history:
            return []
        
        corrections = []
        
        # Check for exact matches in user history
        user_corrections = self.user_history.get_user_corrections(user_id, query)
        
        for corrected_query, confidence in user_corrections:
            corrections.append({
                'original': query,
                'corrected': corrected_query,
                'confidence': confidence,
                'method': 'user_history',
                'metadata': {
                    'user_id': user_id,
                    'personalized': True
                }
            })
        
        # Check for word-level corrections
        words = query.split()
        for i, word in enumerate(words):
            if len(word) <= 2:
                continue
            
            word_corrections = self.user_history.get_user_corrections(user_id, word)
            
            for corrected_word, confidence in word_corrections:
                corrected_words = words.copy()
                corrected_words[i] = corrected_word
                corrected_query = ' '.join(corrected_words)
                
                corrections.append({
                    'original': query,
                    'corrected': corrected_query,
                    'confidence': confidence * 0.8,  # Slight penalty for word-level
                    'method': 'user_history_word',
                    'metadata': {
                        'user_id': user_id,
                        'corrected_word': corrected_word,
                        'original_word': word,
                        'personalized': True
                    }
                })
        
        return corrections
    
    def _get_semantic_corrections(self, query: str, candidate_queries: List[str]) -> List[Dict]:
        """Get semantic corrections using SBERT."""
        if not candidate_queries:
            return []
        
        try:
            # Get embeddings
            all_texts = [query] + candidate_queries
            embeddings = self.get_embeddings(all_texts)
            
            if len(embeddings) == 0:
                return []
            
            query_embedding = embeddings[0:1]
            candidate_embeddings = embeddings[1:]
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
            
            corrections = []
            for i, (candidate, similarity) in enumerate(zip(candidate_queries, similarities)):
                if similarity >= self.similarity_threshold and candidate != query:
                    corrections.append({
                        'original': query,
                        'corrected': candidate,
                        'confidence': similarity,
                        'method': 'semantic',
                        'metadata': {
                            'similarity_score': similarity,
                            'candidate_index': i
                        }
                    })
            
            return corrections
            
        except Exception as e:
            logger.error(f"Failed to get semantic corrections: {e}")
            return []
    
    def _deduplicate_corrections(self, corrections: List[Dict]) -> List[Dict]:
        """Remove duplicate corrections and merge metadata."""
        unique_corrections = {}
        
        for correction in corrections:
            corrected_query = correction['corrected']
            
            if corrected_query in unique_corrections:
                # Merge with existing correction
                existing = unique_corrections[corrected_query]
                existing['confidence'] = max(existing['confidence'], correction['confidence'])
                existing['methods'] = existing.get('methods', [existing['method']]) + [correction['method']]
                existing['metadata'].update(correction['metadata'])
            else:
                # Add new correction
                correction['methods'] = [correction['method']]
                unique_corrections[corrected_query] = correction
        
        return list(unique_corrections.values())
    
    def update_user_correction(self, user_id: str, original: str, correction: str, accepted: bool = True):
        """
        Update user correction history.
        
        Args:
            user_id: User identifier
            original: Original query
            correction: Suggested correction
            accepted: Whether user accepted the correction
        """
        if self.enable_user_history:
            self.user_history.add_correction(user_id, original, correction, accepted)
    
    def get_statistics(self) -> Dict:
        """Get correction statistics."""
        return {
            'total_queries': self.total_queries,
            'spell_check_corrections': self.spell_check_corrections,
            'semantic_corrections': self.semantic_corrections,
            'user_history_corrections': self.user_history_corrections,
            'cache_size': len(self.correction_cache),
            'embedding_cache_size': len(self.embedding_cache),
            'quantization_enabled': self.enable_quantization,
            'spell_check_enabled': self.enable_spell_check,
            'user_history_enabled': self.enable_user_history
        }
    
    def clear_cache(self):
        """Clear correction cache."""
        self.correction_cache.clear()
        logger.info("Cleared correction cache")
    
    def save_all_data(self):
        """Save all cached data and user history."""
        self.save_cached_data()
        if self.enable_user_history:
            self.user_history.save_history()

# Example usage and testing
if __name__ == "__main__":
    # Create enhanced semantic correction
    semantic_corrector = EnhancedSemanticCorrection(
        enable_quantization=True,
        enable_spell_check=True,
        enable_user_history=True,
        similarity_threshold=0.7
    )
    
    # Sample queries for vocabulary building
    sample_queries = [
        "samsung", "samsung galaxy", "apple iphone", "nike shoes", "laptop",
        "smartphone", "headphones", "television", "camera", "watch"
    ]
    
    # Build vocabulary
    semantic_corrector.build_vocabulary(sample_queries)
    
    # Test corrections
    test_queries = [
        "samsng",  # Obvious typo
        "aple",    # Obvious typo
        "smartphon",  # Partial word
        "nike shos",  # Multiple typos
        "laptap",  # Common typo
    ]
    
    print("Testing Enhanced Semantic Correction:")
    print("=" * 60)
    
    for query in test_queries:
        corrections = semantic_corrector.correct_query(
            query, 
            user_id="test_user",
            candidate_queries=sample_queries
        )
        
        print(f"\nQuery: '{query}'")
        print(f"Corrections ({len(corrections)}):")
        for i, correction in enumerate(corrections, 1):
            print(f"  {i}. '{correction['corrected']}' "
                  f"(confidence: {correction['confidence']:.3f}, "
                  f"method: {correction['method']})")
    
    # Test user history
    print("\n" + "=" * 60)
    print("Testing User History:")
    
    # Add some user corrections
    semantic_corrector.update_user_correction("test_user", "samsng", "samsung", True)
    semantic_corrector.update_user_correction("test_user", "aple", "apple", True)
    
    # Test with user history
    corrections = semantic_corrector.correct_query("samsng", user_id="test_user")
    print(f"\nUser-specific corrections for 'samsng':")
    for correction in corrections:
        print(f"  '{correction['corrected']}' (confidence: {correction['confidence']:.3f})")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Statistics:")
    stats = semantic_corrector.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save data
    semantic_corrector.save_all_data() 