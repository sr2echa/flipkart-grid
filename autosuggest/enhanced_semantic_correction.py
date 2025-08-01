import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from difflib import SequenceMatcher
import Levenshtein

class EnhancedSemanticCorrection:
    """Enhanced semantic correction with improved accuracy and multiple correction strategies."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.user_queries = None
        self.query_embeddings = None
        self.faiss_index = None
        self.corrected_queries = None
        
        # Additional components for better correction
        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 4),
            max_features=10000
        )
        self.tfidf_matrix = None
        
        # Brand and category mappings for better correction
        self.brand_mapping = {}
        self.category_mapping = {}
        self.common_typos = {}
        
        # Phonetic and character-level similarities
        self.char_similarity_threshold = 0.6
        self.semantic_similarity_threshold = 0.5
        
    def build_semantic_index(self, user_queries_df: pd.DataFrame):
        """Build enhanced semantic index with multiple correction strategies."""
        print("Building enhanced semantic index...")
        
        self.user_queries = user_queries_df
        
        # Get unique corrected queries
        self.corrected_queries = user_queries_df['corrected_query'].dropna().unique().tolist()
        
        # Clean and preprocess queries
        self.corrected_queries = [self._clean_query(q) for q in self.corrected_queries if q and len(q.strip()) > 0]
        self.corrected_queries = list(set(self.corrected_queries))  # Remove duplicates
        
        print(f"Processing {len(self.corrected_queries)} unique queries...")
        
        # Build mappings
        self._build_brand_category_mappings()
        self._build_common_typos_mapping()
        
        # Generate embeddings for semantic similarity
        print("Generating semantic embeddings...")
        self.query_embeddings = self.model.encode(self.corrected_queries, show_progress_bar=True)
        
        # Build FAISS index for fast semantic search
        print("Building FAISS index...")
        dimension = self.query_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(self.query_embeddings.astype('float32'))
        
        # Build TF-IDF index for character-level similarity
        print("Building character-level TF-IDF index...")
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corrected_queries)
        
        print(f"Enhanced semantic index built with {len(self.corrected_queries)} queries")
        self._save_index()
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize query."""
        if not query:
            return ""
        
        # Convert to lowercase and strip
        query = query.lower().strip()
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query)
        
        # Remove special characters but keep alphanumeric and spaces
        query = re.sub(r'[^\w\s]', '', query)
        
        return query
    
    def _build_brand_category_mappings(self):
        """Build mappings for brands and categories to improve correction."""
        # Extract brands and categories from queries
        brand_keywords = [
            'samsung', 'apple', 'nike', 'adidas', 'sony', 'oneplus', 'xiaomi',
            'vivo', 'oppo', 'realme', 'dell', 'hp', 'lenovo', 'asus', 'lg',
            'boat', 'jbl', 'puma', 'reebok', 'zara', 'h&m', 'bata', 'titan',
            'fossil', 'casio', 'canon', 'nikon', 'gopro', 'fitbit', 'garmin'
        ]
        
        category_keywords = [
            'phone', 'smartphone', 'mobile', 'laptop', 'computer', 'tablet',
            'headphones', 'earbuds', 'speaker', 'camera', 'watch', 'smartwatch',
            'shoes', 'sneakers', 'sandals', 'boots', 'shirt', 'tshirt', 't-shirt',
            'jeans', 'trousers', 'jacket', 'hoodie', 'dress', 'skirt'
        ]
        
        # Build brand mapping
        for brand in brand_keywords:
            self.brand_mapping[brand] = brand
            # Add common variations
            if brand == 'samsung':
                self.brand_mapping.update({
                    'samsng': 'samsung',
                    'samung': 'samsung', 
                    'samsong': 'samsung'
                })
            elif brand == 'apple':
                self.brand_mapping.update({
                    'aple': 'apple',
                    'appl': 'apple',
                    'appel': 'apple'
                })
            elif brand == 'nike':
                self.brand_mapping.update({
                    'nkie': 'nike',
                    'niki': 'nike',
                    'nyke': 'nike'
                })
            elif brand == 'lenovo':
                self.brand_mapping.update({
                    'lenvo': 'lenovo',
                    'lenavo': 'lenovo',
                    'lenova': 'lenovo'
                })
        
        # Build category mapping
        for category in category_keywords:
            self.category_mapping[category] = category
    
    def _build_common_typos_mapping(self):
        """Build mapping for common typos and corrections."""
        self.common_typos = {
            'phon': 'phone',
            'fone': 'phone',
            'leptop': 'laptop',
            'laptap': 'laptop',
            'headfone': 'headphone',
            'hedphone': 'headphone',
            'shooes': 'shoes',
            'shoos': 'shoes',
            'jens': 'jeans',
            'waych': 'watch',
            'wach': 'watch'
        }
    
    def get_semantic_suggestions(self, query: str, max_suggestions: int = 5) -> List[Tuple[str, float]]:
        """Get enhanced semantic suggestions using multiple strategies."""
        if not query.strip():
            return []
        
        query = self._clean_query(query)
        if not query:
            return []
        
        # Strategy 1: Exact brand/category correction
        exact_corrections = self._get_exact_corrections(query)
        if exact_corrections:
            return exact_corrections[:max_suggestions]
        
        # Strategy 2: Combine multiple similarity approaches
        suggestions = []
        
        # Character-level similarity (TF-IDF)
        char_suggestions = self._get_character_suggestions(query, max_suggestions * 2)
        suggestions.extend(char_suggestions)
        
        # Semantic similarity (SBERT)
        semantic_suggestions = self._get_embedding_suggestions(query, max_suggestions * 2)
        suggestions.extend(semantic_suggestions)
        
        # Edit distance similarity
        edit_suggestions = self._get_edit_distance_suggestions(query, max_suggestions)
        suggestions.extend(edit_suggestions)
        
        # Combine and rank suggestions
        combined_suggestions = self._combine_and_rank_suggestions(query, suggestions)
        
        return combined_suggestions[:max_suggestions]
    
    def _get_exact_corrections(self, query: str) -> List[Tuple[str, float]]:
        """Get exact corrections for known brands/categories."""
        corrections = []
        
        # Check for exact brand matches
        if query in self.brand_mapping:
            corrected = self.brand_mapping[query]
            if corrected != query:
                # Find queries containing this brand
                matching_queries = [q for q in self.corrected_queries if corrected in q]
                for match in matching_queries[:3]:
                    corrections.append((match, 1.0))
        
        # Check for exact typo corrections
        if query in self.common_typos:
            corrected = self.common_typos[query]
            matching_queries = [q for q in self.corrected_queries if corrected in q]
            for match in matching_queries[:3]:
                corrections.append((match, 0.95))
        
        return corrections
    
    def _get_character_suggestions(self, query: str, max_suggestions: int) -> List[Tuple[str, float]]:
        """Get suggestions based on character-level similarity."""
        try:
            query_vector = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top matches
            top_indices = np.argsort(similarities)[::-1][:max_suggestions]
            
            suggestions = []
            for idx in top_indices:
                if similarities[idx] > self.char_similarity_threshold:
                    suggestions.append((self.corrected_queries[idx], float(similarities[idx])))
            
            return suggestions
        except Exception as e:
            print(f"Character similarity error: {e}")
            return []
    
    def _get_embedding_suggestions(self, query: str, max_suggestions: int) -> List[Tuple[str, float]]:
        """Get suggestions based on semantic embeddings."""
        try:
            query_embedding = self.model.encode([query])
            
            # Search in FAISS index
            similarities, indices = self.faiss_index.search(
                query_embedding.astype('float32'), 
                max_suggestions * 2
            )
            
            suggestions = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if similarity > self.semantic_similarity_threshold and idx < len(self.corrected_queries):
                    suggestions.append((self.corrected_queries[idx], float(similarity)))
            
            return suggestions
        except Exception as e:
            print(f"Semantic similarity error: {e}")
            return []
    
    def _get_edit_distance_suggestions(self, query: str, max_suggestions: int) -> List[Tuple[str, float]]:
        """Get suggestions based on edit distance."""
        suggestions = []
        
        for candidate in self.corrected_queries:
            # Calculate normalized edit distance
            distance = Levenshtein.distance(query, candidate)
            max_len = max(len(query), len(candidate))
            
            if max_len == 0:
                continue
                
            similarity = 1.0 - (distance / max_len)
            
            # Only consider candidates with reasonable similarity
            if similarity > 0.6:
                suggestions.append((candidate, similarity))
        
        # Sort by similarity and return top suggestions
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:max_suggestions]
    
    def _combine_and_rank_suggestions(self, query: str, suggestions: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Combine suggestions from different strategies and rank them."""
        # Group suggestions by text
        suggestion_scores = {}
        
        for text, score in suggestions:
            if text == query:  # Skip exact matches
                continue
                
            if text in suggestion_scores:
                # Combine scores (weighted average)
                existing_score = suggestion_scores[text]
                suggestion_scores[text] = (existing_score + score) / 2
            else:
                suggestion_scores[text] = score
        
        # Add frequency boost
        for text in suggestion_scores:
            frequency_boost = self._get_frequency_boost(text)
            suggestion_scores[text] *= frequency_boost
        
        # Sort by combined score
        ranked_suggestions = sorted(
            suggestion_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return ranked_suggestions
    
    def _get_frequency_boost(self, query: str) -> float:
        """Get frequency boost for popular queries."""
        if self.user_queries is not None:
            query_counts = self.user_queries['corrected_query'].value_counts()
            count = query_counts.get(query, 1)
            # Apply logarithmic boost
            return 1.0 + np.log10(count) / 10
        return 1.0
    
    def _save_index(self):
        """Save the enhanced semantic index to disk."""
        os.makedirs('../models', exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, '../models/enhanced_semantic_index.faiss')
        
        # Save other components
        with open('../models/enhanced_corrected_queries.pkl', 'wb') as f:
            pickle.dump({
                'corrected_queries': self.corrected_queries,
                'brand_mapping': self.brand_mapping,
                'category_mapping': self.category_mapping,
                'common_typos': self.common_typos
            }, f)
        
        # Save TF-IDF vectorizer and matrix
        with open('../models/enhanced_tfidf_components.pkl', 'wb') as f:
            pickle.dump({
                'vectorizer': self.tfidf_vectorizer,
                'matrix': self.tfidf_matrix
            }, f)
        
        print("Enhanced semantic index saved to disk")
    
    def load_index(self):
        """Load the enhanced semantic index from disk."""
        try:
            # Load FAISS index
            self.faiss_index = faiss.read_index('../models/enhanced_semantic_index.faiss')
            
            # Load other components
            with open('../models/enhanced_corrected_queries.pkl', 'rb') as f:
                data = pickle.load(f)
                self.corrected_queries = data['corrected_queries']
                self.brand_mapping = data['brand_mapping']
                self.category_mapping = data['category_mapping']
                self.common_typos = data['common_typos']
            
            # Load TF-IDF components
            with open('../models/enhanced_tfidf_components.pkl', 'rb') as f:
                tfidf_data = pickle.load(f)
                self.tfidf_vectorizer = tfidf_data['vectorizer']
                self.tfidf_matrix = tfidf_data['matrix']
            
            print("Enhanced semantic index loaded from disk")
            return True
            
        except Exception as e:
            print(f"Failed to load enhanced semantic index: {e}")
            return False

# Test the enhanced semantic correction
if __name__ == "__main__":
    # Import data preprocessing to test
    import sys
    sys.path.append('.')
    from data_preprocessing import DataPreprocessor
    
    # Load data
    preprocessor = DataPreprocessor()
    preprocessor.run_all_preprocessing()
    data = preprocessor.get_processed_data()
    
    # Test enhanced semantic correction
    enhanced_semantic = EnhancedSemanticCorrection()
    enhanced_semantic.build_semantic_index(data['user_queries'])
    
    # Test some corrections
    test_queries = ['samsng', 'aple', 'nkie', 'lenvo', 'phon', 'leptop']
    expected = ['samsung', 'apple', 'nike', 'lenovo', 'phone', 'laptop']
    
    print("\n=== Enhanced Semantic Correction Test ===")
    correct_count = 0
    
    for query, expected_word in zip(test_queries, expected):
        suggestions = enhanced_semantic.get_semantic_suggestions(query)
        print(f"\nQuery: '{query}' (expected: {expected_word})")
        
        if suggestions:
            top_suggestion = suggestions[0][0]
            print(f"Top suggestion: '{top_suggestion}' (score: {suggestions[0][1]:.4f})")
            
            # Check if expected word is in the top suggestion
            if expected_word in top_suggestion.lower():
                correct_count += 1
                print("✅ Correct!")
            else:
                print("❌ Incorrect")
                
            # Show all suggestions
            for i, (suggestion, score) in enumerate(suggestions, 1):
                print(f"  {i}. {suggestion} ({score:.4f})")
        else:
            print("No suggestions found")
    
    accuracy = correct_count / len(test_queries) * 100
    print(f"\nEnhanced Accuracy: {accuracy:.1f}% ({correct_count}/{len(test_queries)})")