"""
Advanced Autosuggest System V4 with SBERT + FAISS + NLTK
Features:
- Real brand names and product data
- SBERT semantic embeddings for similarity search
- FAISS for fast similarity indexing
- NLTK for spell correction and text processing
- Proper typo correction (xiomi -> xiaomi)
- Context-aware suggestions based on persona
"""

import pandas as pd
import numpy as np
import nltk
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import pickle
import os
import re
from collections import Counter, defaultdict
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# Download required NLTK data
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.corpus import words
from nltk.tokenize import word_tokenize

class AdvancedAutosuggestV4:
    """
    Advanced autosuggest system with SBERT, FAISS, NLTK, and real product data.
    """
    
    def __init__(self):
        print("🚀 Initializing Advanced Autosuggest V4...")
        
        # Core components
        self.sbert_model = None
        self.faiss_index = None
        self.query_embeddings = None
        self.all_queries = []
        self.query_metadata = []
        
        # NLTK components
        self.english_words = set(words.words())
        self.nltk_initialized = False
        
        # Spell correction
        self.typo_corrections = {
            'xiomi': 'xiaomi',
            'samsng': 'samsung', 
            'samung': 'samsung',
            'jersy': 'jersey',
            'jesery': 'jersey',
            'lapto': 'laptop',
            'leptop': 'laptop',
            'mobil': 'mobile',
            'moble': 'mobile', 
            'headphone': 'headphones',
            'hedphones': 'headphones',
            'aple': 'apple',
            'ifone': 'iphone',
            'criket': 'cricket',
            'footbal': 'football',
            'cloths': 'clothes',
            'tshrt': 'tshirt',
            'shooe': 'shoe',
            'shose': 'shoes'
        }
        
        # Brand mapping for contextual suggestions
        self.brand_context = {
            'tech_enthusiast': ['Samsung', 'Apple', 'Xiaomi', 'OnePlus', 'HP', 'Dell', 'Lenovo'],
            'sports_enthusiast': ['Nike', 'Adidas', 'Puma', 'Reebok', 'MRF', 'SG'],
            'fashion_lover': ['Zara', 'H&M', 'Nike', 'Adidas', 'Levis', 'Tommy Hilfiger'],
            'home_maker': ['LG', 'Samsung', 'Whirlpool', 'Godrej', 'Prestige', 'Hawkins'],
            'budget_conscious': ['Xiaomi', 'Realme', 'Decathlon', 'Nilkamal', 'Pigeon']
        }
        
        # Category keywords for intent understanding
        self.category_keywords = {
            'electronics': ['phone', 'mobile', 'laptop', 'tv', 'tablet', 'headphones', 'earbuds'],
            'fashion': ['shirt', 'jeans', 'dress', 'shoes', 'bag', 'watch', 'sunglasses'],
            'sports': ['cricket', 'football', 'jersey', 'bat', 'ball', 'fitness', 'gym'],
            'home': ['furniture', 'sofa', 'bed', 'table', 'chair', 'appliance', 'kitchen']
        }
        
        # Cache for frequent operations
        self.suggestion_cache = {}
        self.spell_cache = {}
        
    def initialize_nltk(self):
        """Initialize NLTK components."""
        try:
            # Try to use NLTK words corpus
            test_words = list(self.english_words)[:10]
            self.nltk_initialized = True
            print("✅ NLTK initialized successfully")
        except Exception as e:
            print(f"⚠️ NLTK initialization warning: {e}")
            # Fallback to basic functionality
            self.english_words = set(['phone', 'mobile', 'laptop', 'tablet', 'shoes', 'shirt'])
            self.nltk_initialized = False
    
    def load_sbert_model(self):
        """Load SBERT model for semantic embeddings."""
        try:
            print("🤖 Loading SBERT model...")
            self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ SBERT model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading SBERT model: {e}")
            raise
    
    def build_system(self, data: Dict):
        """Build the complete autosuggest system."""
        print("🏗️ Building Advanced Autosuggest System V4...")
        
        # Initialize components
        self.initialize_nltk()
        self.load_sbert_model()
        
        # Load and process data
        self._load_realistic_queries()
        self._build_semantic_index()
        self._prepare_spell_correction()
        
        print("✅ Advanced Autosuggest System V4 built successfully!")
    
    def _load_realistic_queries(self):
        """Load realistic queries with proper brand names."""
        try:
            # Try to load the enhanced realistic dataset
            queries_df = pd.read_csv('../dataset/user_queries_realistic_v4.csv')
            print(f"✅ Loaded realistic queries: {len(queries_df)} entries")
        except FileNotFoundError:
            # Fallback to comprehensive dataset
            queries_df = pd.read_csv('../dataset/user_queries_comprehensive.csv')
            print(f"⚠️ Using fallback dataset: {len(queries_df)} entries")
        
        # Process queries
        self.all_queries = queries_df['corrected_query'].dropna().tolist()
        
        # Create metadata for each query
        self.query_metadata = []
        for _, row in queries_df.iterrows():
            metadata = {
                'query': row['corrected_query'],
                'frequency': row.get('frequency', 1),
                'category': self._infer_category(row['corrected_query']),
                'brand': self._extract_brand(row['corrected_query']),
                'is_typo': False
            }
            self.query_metadata.append(metadata)
        
        print(f"📊 Processed {len(self.all_queries)} unique queries")
        print(f"🏷️ Identified {len(set([m['brand'] for m in self.query_metadata if m['brand']]))} unique brands")
    
    def _build_semantic_index(self):
        """Build FAISS index with SBERT embeddings."""
        print("🔍 Building semantic index with SBERT + FAISS...")
        
        if not self.sbert_model:
            raise ValueError("SBERT model not loaded")
        
        # Generate embeddings for all queries
        print("   Generating embeddings...")
        self.query_embeddings = self.sbert_model.encode(
            self.all_queries,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Build FAISS index
        print("   Building FAISS index...")
        dimension = self.query_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.query_embeddings)
        self.faiss_index.add(self.query_embeddings)
        
        print(f"✅ FAISS index built with {self.faiss_index.ntotal} vectors")
        
        # Save index for faster loading
        try:
            index_path = '../models/faiss_index_v4.bin'
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            faiss.write_index(self.faiss_index, index_path)
            
            # Save query mapping
            with open('../models/query_mapping_v4.pkl', 'wb') as f:
                pickle.dump({
                    'queries': self.all_queries,
                    'metadata': self.query_metadata
                }, f)
            
            print("💾 Index and mappings saved to disk")
        except Exception as e:
            print(f"⚠️ Could not save index: {e}")
    
    def _prepare_spell_correction(self):
        """Prepare spell correction using NLTK and custom typo mappings."""
        print("🔤 Preparing spell correction...")
        
        # Extend typo corrections with learned patterns
        query_words = set()
        for query in self.all_queries:
            words_in_query = re.findall(r'\b\w+\b', query.lower())
            query_words.update(words_in_query)
        
        # Add common product-related words to our vocabulary
        self.product_vocabulary = query_words.union(self.english_words)
        
        print(f"✅ Spell correction prepared with {len(self.product_vocabulary)} words")
    
    def _infer_category(self, query: str) -> str:
        """Infer category from query text."""
        query_lower = query.lower()
        
        for category, keywords in self.category_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def _extract_brand(self, query: str) -> Optional[str]:
        """Extract brand name from query."""
        query_lower = query.lower()
        
        # Check all known brands
        all_brands = set()
        for brands in self.brand_context.values():
            all_brands.update([b.lower() for b in brands])
        
        for brand in all_brands:
            if brand in query_lower:
                return brand.title()
        
        return None
    
    def correct_spelling(self, query: str) -> str:
        """Correct spelling using NLTK and custom mappings."""
        # Check cache first
        if query in self.spell_cache:
            return self.spell_cache[query]
        
        # Apply direct typo corrections first
        corrected = query.lower()
        for typo, correction in self.typo_corrections.items():
            if typo in corrected:
                corrected = corrected.replace(typo, correction)
                self.spell_cache[query] = corrected
                return corrected
        
        # For unknown words, try to find closest match
        words_in_query = re.findall(r'\b\w+\b', query.lower())
        corrected_words = []
        
        for word in words_in_query:
            if word in self.product_vocabulary:
                corrected_words.append(word)
            else:
                # Find closest match using difflib
                close_matches = difflib.get_close_matches(
                    word, self.product_vocabulary, n=1, cutoff=0.7
                )
                if close_matches:
                    corrected_words.append(close_matches[0])
                else:
                    corrected_words.append(word)  # Keep original if no good match
        
        corrected = ' '.join(corrected_words)
        self.spell_cache[query] = corrected
        return corrected
    
    def get_semantic_suggestions(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get semantically similar suggestions using SBERT + FAISS."""
        if not self.faiss_index or not self.sbert_model:
            return []
        
        try:
            # Correct spelling first
            corrected_query = self.correct_spelling(query)
            
            # Generate embedding for query
            query_embedding = self.sbert_model.encode([corrected_query])
            faiss.normalize_L2(query_embedding)
            
            # Search in FAISS index
            scores, indices = self.faiss_index.search(query_embedding, top_k)
            
            # Convert to suggestions with scores
            suggestions = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.all_queries):
                    suggestion = self.all_queries[idx]
                    # Skip exact matches
                    if suggestion.lower() != query.lower():
                        suggestions.append((suggestion, float(score)))
            
            return suggestions
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    def get_contextual_suggestions(self, query: str, context: Dict = None, max_suggestions: int = 5) -> List[Tuple[str, float]]:
        """Get contextual suggestions with persona-based ranking."""
        if not query.strip():
            return []
        
        # Check cache
        cache_key = f"{query}_{context.get('persona', '')}"
        if cache_key in self.suggestion_cache:
            return self.suggestion_cache[cache_key]
        
        query = query.lower().strip()
        
        # Get semantic suggestions
        semantic_suggestions = self.get_semantic_suggestions(query, top_k=15)
        
        # Apply contextual boosting
        if context:
            semantic_suggestions = self._apply_contextual_boosting(semantic_suggestions, context)
        
        # Sort by score and return top suggestions
        semantic_suggestions.sort(key=lambda x: x[1], reverse=True)
        result = semantic_suggestions[:max_suggestions]
        
        # Cache result
        self.suggestion_cache[cache_key] = result
        
        return result
    
    def _apply_contextual_boosting(self, suggestions: List[Tuple[str, float]], context: Dict) -> List[Tuple[str, float]]:
        """Apply contextual boosting based on persona, location, event."""
        boosted_suggestions = []
        
        persona = context.get('persona', '')
        location = context.get('location', '')
        event = context.get('event', '')
        
        # Get preferred brands for this persona
        preferred_brands = self.brand_context.get(persona, [])
        
        for suggestion, score in suggestions:
            boosted_score = score
            suggestion_lower = suggestion.lower()
            
            # Persona-based brand boosting
            for brand in preferred_brands:
                if brand.lower() in suggestion_lower:
                    boosted_score *= 1.5
                    break
            
            # Event-based boosting
            if event:
                event_lower = event.lower()
                if 'ipl' in event_lower and any(word in suggestion_lower for word in ['cricket', 'jersey', 'csk', 'rcb', 'mi']):
                    boosted_score *= 1.4
                elif 'diwali' in event_lower and any(word in suggestion_lower for word in ['lights', 'decoration', 'gift']):
                    boosted_score *= 1.4
                elif 'summer' in event_lower and any(word in suggestion_lower for word in ['ac', 'cooler', 'summer']):
                    boosted_score *= 1.3
            
            # Location-based boosting  
            if location:
                location_lower = location.lower()
                if location_lower in ['bangalore', 'hyderabad'] and any(word in suggestion_lower for word in ['laptop', 'tech', 'gaming']):
                    boosted_score *= 1.2
                elif location_lower in ['mumbai', 'delhi'] and any(word in suggestion_lower for word in ['fashion', 'brand']):
                    boosted_score *= 1.2
            
            boosted_suggestions.append((suggestion, boosted_score))
        
        return boosted_suggestions
    
    def get_suggestions(self, query: str, context: Dict = None, max_suggestions: int = 5) -> List[Tuple[str, float]]:
        """Main method to get suggestions."""
        return self.get_contextual_suggestions(query, context, max_suggestions)

# Test the system
if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    
    print("=== Testing Advanced Autosuggest System V4 ===")
    
    # Load data
    preprocessor = DataPreprocessor()
    preprocessor.run_all_preprocessing()
    data = preprocessor.get_processed_data()
    
    # Build system
    autosuggest = AdvancedAutosuggestV4()
    autosuggest.build_system(data)
    
    # Test cases
    test_cases = [
        ("xiomi", {"persona": "tech_enthusiast", "location": "Delhi", "event": "none"}),
        ("samsng", {"persona": "tech_enthusiast", "location": "Mumbai", "event": "none"}),
        ("jersy", {"persona": "sports_enthusiast", "location": "Chennai", "event": "ipl"}),
        ("nike sho", {"persona": "sports_enthusiast", "location": "Pune", "event": "none"}),
        ("lapto", {"persona": "tech_enthusiast", "location": "Bangalore", "event": "none"}),
        ("iphon", {"persona": "tech_enthusiast", "location": "Mumbai", "event": "none"}),
        ("apl", {"persona": "tech_enthusiast", "location": "Delhi", "event": "none"}),
        ("lights", {"persona": "home_maker", "location": "Mumbai", "event": "diwali"}),
    ]
    
    print("\n=== Advanced Test Results ===")
    for query, context in test_cases:
        start_time = time.time()
        suggestions = autosuggest.get_suggestions(query, context)
        end_time = time.time()
        
        # Check spelling correction
        corrected = autosuggest.correct_spelling(query)
        
        print(f"\n🔍 Query: '{query}' → Corrected: '{corrected}'")
        print(f"🎯 Context: {context['persona']}, {context['location']}, {context['event']}")
        print(f"💡 Suggestions: {[s for s, score in suggestions]}")
        print(f"📊 Scores: {[f'{score:.3f}' for s, score in suggestions]}")
        print(f"⚡ Response time: {(end_time - start_time)*1000:.2f}ms")
        
        if suggestions:
            print(f"✅ Got {len(suggestions)} quality suggestions")
        else:
            print(f"❌ No suggestions found - needs improvement")
