"""
Enhanced Autosuggest V5 with Intelligent Brand Recognition and Continuous Improvement
Fixes critical issues like 'sam' suggesting cricket equipment instead of Samsung
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

class EnhancedAutosuggestV5:
    """
    Enhanced autosuggest system V5 with intelligent brand recognition and continuous improvement.
    """
    
    def __init__(self):
        print("🚀 Initializing Enhanced Autosuggest V5...")
        
        # Core components
        self.sbert_model = None
        self.faiss_index = None
        self.query_embeddings = None
        self.all_queries = []
        self.query_metadata = []
        
        # Enhanced brand recognition
        self.brand_prefixes = {
            'sam': ['samsung', 'samsung galaxy', 'samsung phone', 'samsung tv', 'samsung laptop'],
            'xiaomi': ['xiaomi', 'xiaomi phone', 'mi phone', 'redmi'],
            'xi': ['xiaomi', 'xiaomi phone'],
            'xiomi': ['xiaomi', 'xiaomi phone'],
            'nike': ['nike shoes', 'nike sneakers', 'nike running', 'nike sports'],
            'nik': ['nike shoes', 'nike sneakers'],
            'adidas': ['adidas shoes', 'adidas sports', 'adidas running'],
            'adi': ['adidas shoes', 'adidas sports'],
            'apple': ['apple iphone', 'apple macbook', 'apple watch', 'iphone'],
            'app': ['apple iphone', 'apple macbook'],
            'hp': ['hp laptop', 'hp printer', 'hp computer'],
            'dell': ['dell laptop', 'dell computer', 'dell inspiron'],
            'lenovo': ['lenovo laptop', 'lenovo thinkpad', 'lenovo computer'],
            'lg': ['lg tv', 'lg refrigerator', 'lg washing machine'],
            'sony': ['sony tv', 'sony headphones', 'sony camera'],
            'one': ['oneplus', 'oneplus phone'],
            'oneplus': ['oneplus phone', 'oneplus mobile'],
            'realme': ['realme phone', 'realme mobile'],
            'vivo': ['vivo phone', 'vivo mobile'],
            'oppo': ['oppo phone', 'oppo mobile']
        }
        
        # Product categories for better matching
        self.category_patterns = {
            'phone': ['mobile', 'smartphone', 'phone', 'galaxy', 'iphone'],
            'laptop': ['laptop', 'notebook', 'computer', 'macbook'],
            'tv': ['television', 'tv', 'smart tv', 'led tv'],
            'shoes': ['shoes', 'sneakers', 'footwear', 'running shoes'],
            'clothing': ['shirt', 'jeans', 'dress', 'jacket', 'hoodie'],
            'electronics': ['headphones', 'earbuds', 'speaker', 'charger']
        }
        
        # Advanced spell correction
        self.typo_corrections = {
            'xiomi': 'xiaomi',
            'samsng': 'samsung', 
            'samung': 'samsung',
            'samsunq': 'samsung',
            'jersy': 'jersey',
            'jesery': 'jersey',
            'lapto': 'laptop',
            'leptop': 'laptop',
            'mobil': 'mobile',
            'moble': 'mobile',
            'headphone': 'headphones',
            'aple': 'apple',
            'ifone': 'iphone',
            'ipone': 'iphone',
            'criket': 'cricket',
            'footbal': 'football'
        }
        
        # Context boosting weights
        self.context_weights = {
            'exact_brand_match': 3.0,
            'brand_prefix_match': 2.5,
            'category_match': 2.0,
            'persona_match': 1.5,
            'event_match': 1.4,
            'location_match': 1.2,
            'semantic_similarity': 1.0
        }
        
        # Cache for performance
        self.suggestion_cache = {}
        self.brand_cache = {}
        
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
        """Build the enhanced autosuggest system."""
        print("🏗️ Building Enhanced Autosuggest System V5...")
        
        # Initialize components
        self.load_sbert_model()
        
        # Load and process data
        self._load_realistic_queries()
        self._build_semantic_index()
        self._prepare_enhanced_spell_correction()
        
        print("✅ Enhanced Autosuggest System V5 built successfully!")
    
    def _load_realistic_queries(self):
        """Load realistic queries with enhanced brand processing."""
        try:
            queries_df = pd.read_csv('../dataset/user_queries_realistic_v4.csv')
            print(f"✅ Loaded realistic queries: {len(queries_df)} entries")
        except FileNotFoundError:
            queries_df = pd.read_csv('../dataset/user_queries_comprehensive.csv')
            print(f"⚠️ Using fallback dataset: {len(queries_df)} entries")
        
        # Process queries with enhanced metadata
        self.all_queries = queries_df['corrected_query'].dropna().tolist()
        
        # Create enhanced metadata
        self.query_metadata = []
        for _, row in queries_df.iterrows():
            query = row['corrected_query']
            metadata = {
                'query': query,
                'frequency': row.get('frequency', 1),
                'category': self._infer_category(query),
                'brand': self._extract_brand_advanced(query),
                'query_type': self._classify_query_type(query),
                'popularity_score': self._calculate_popularity_score(query),
                'brand_strength': self._calculate_brand_strength(query)
            }
            self.query_metadata.append(metadata)
        
        print(f"📊 Processed {len(self.all_queries)} unique queries")
        print(f"🏷️ Enhanced brand recognition for {len(set([m['brand'] for m in self.query_metadata if m['brand']]))} brands")
    
    def _extract_brand_advanced(self, query: str) -> Optional[str]:
        """Advanced brand extraction with better recognition."""
        query_lower = query.lower()
        
        # Check for exact brand matches first
        known_brands = ['samsung', 'xiaomi', 'nike', 'adidas', 'apple', 'hp', 'dell', 'lenovo', 'sony', 'lg']
        for brand in known_brands:
            if brand in query_lower:
                return brand.title()
        
        # Check for common brand variations
        brand_variations = {
            'galaxy': 'Samsung',
            'iphone': 'Apple',
            'macbook': 'Apple',
            'thinkpad': 'Lenovo',
            'inspiron': 'Dell'
        }
        
        for variation, brand in brand_variations.items():
            if variation in query_lower:
                return brand
        
        return None
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query for better handling."""
        query_lower = query.lower()
        
        if any(brand in query_lower for brand in ['samsung', 'xiaomi', 'apple', 'nike']):
            return 'brand_specific'
        elif any(cat in query_lower for cat in ['phone', 'laptop', 'tv', 'shoes']):
            return 'category_specific'
        elif len(query.split()) == 1:
            return 'single_word'
        else:
            return 'multi_word'
    
    def _calculate_popularity_score(self, query: str) -> float:
        """Calculate popularity score based on brand and product type."""
        score = 1.0
        query_lower = query.lower()
        
        # Popular brands get higher scores
        popular_brands = {
            'samsung': 2.0, 'xiaomi': 1.8, 'apple': 2.0, 'nike': 1.7,
            'adidas': 1.6, 'hp': 1.5, 'dell': 1.5, 'sony': 1.4
        }
        
        for brand, boost in popular_brands.items():
            if brand in query_lower:
                score *= boost
                break
        
        # Popular categories get boosts
        if any(cat in query_lower for cat in ['phone', 'mobile', 'smartphone']):
            score *= 1.8
        elif any(cat in query_lower for cat in ['laptop', 'computer']):
            score *= 1.6
        elif any(cat in query_lower for cat in ['shoes', 'sneakers']):
            score *= 1.4
        
        return score
    
    def _infer_category(self, query: str) -> str:
        """Infer category from query text."""
        query_lower = query.lower()
        
        for category, keywords in self.category_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def _calculate_brand_strength(self, query: str) -> float:
        """Calculate how strongly the query is associated with a brand."""
        query_lower = query.lower()
        brand = self._extract_brand_advanced(query)
        
        if not brand:
            return 0.0
        
        brand_lower = brand.lower()
        
        # Exact brand name match
        if brand_lower in query_lower:
            if query_lower.startswith(brand_lower):
                return 1.0  # Strong brand association
            else:
                return 0.8  # Brand mentioned but not primary
        
        return 0.5  # Weak brand association
    
    def _build_semantic_index(self):
        """Build FAISS index with enhanced embeddings."""
        print("🔍 Building enhanced semantic index...")
        
        # Generate embeddings
        print("   Generating embeddings...")
        self.query_embeddings = self.sbert_model.encode(
            self.all_queries,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Build FAISS index
        print("   Building FAISS index...")
        dimension = self.query_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        
        faiss.normalize_L2(self.query_embeddings)
        self.faiss_index.add(self.query_embeddings)
        
        print(f"✅ Enhanced FAISS index built with {self.faiss_index.ntotal} vectors")
    
    def _prepare_enhanced_spell_correction(self):
        """Enhanced spell correction with better brand recognition."""
        print("🔤 Preparing enhanced spell correction...")
        
        # Build comprehensive vocabulary
        query_words = set()
        for query in self.all_queries:
            words_in_query = re.findall(r'\b\w+\b', query.lower())
            query_words.update(words_in_query)
        
        # Add important brand names and products
        important_words = set([
            'samsung', 'xiaomi', 'apple', 'nike', 'adidas', 'hp', 'dell', 'lenovo',
            'phone', 'mobile', 'smartphone', 'laptop', 'computer', 'tv', 'shoes',
            'galaxy', 'iphone', 'macbook', 'thinkpad', 'inspiron'
        ])
        
        self.enhanced_vocabulary = query_words.union(important_words)
        print(f"✅ Enhanced spell correction with {len(self.enhanced_vocabulary)} words")
    
    def correct_spelling_advanced(self, query: str) -> str:
        """Advanced spelling correction with brand focus."""
        # Check cache first
        if query in self.brand_cache:
            return self.brand_cache[query]
        
        # Direct typo corrections
        corrected = query.lower()
        for typo, correction in self.typo_corrections.items():
            if typo in corrected:
                corrected = corrected.replace(typo, correction)
                self.brand_cache[query] = corrected
                return corrected
        
        # Check for partial brand matches
        for prefix, completions in self.brand_prefixes.items():
            if query.lower().startswith(prefix):
                # Return the most likely completion
                best_completion = completions[0]
                self.brand_cache[query] = best_completion
                return best_completion
        
        # Fallback to original query
        return query.lower()
    
    def get_intelligent_suggestions(self, query: str, context: Dict = None, max_suggestions: int = 5) -> List[Tuple[str, float]]:
        """Get intelligent suggestions with enhanced brand recognition."""
        if not query.strip():
            return []
        
        original_query = query
        query = query.lower().strip()
        
        # Step 1: Advanced spell correction and brand recognition
        corrected_query = self.correct_spelling_advanced(query)
        
        # Step 2: Check for exact brand prefix matches first
        prefix_suggestions = self._get_brand_prefix_suggestions(query)
        
        # Step 3: Get semantic suggestions
        semantic_suggestions = self._get_enhanced_semantic_suggestions(corrected_query, max_suggestions * 2)
        
        # Step 4: Combine and rank suggestions
        all_suggestions = self._combine_and_rank_suggestions(
            prefix_suggestions, semantic_suggestions, query, corrected_query, context
        )
        
        # Step 5: Apply contextual boosting
        if context:
            all_suggestions = self._apply_enhanced_contextual_boosting(all_suggestions, context)
        
        # Step 6: Sort and return top suggestions
        all_suggestions.sort(key=lambda x: x[1], reverse=True)
        return all_suggestions[:max_suggestions]
    
    def _get_brand_prefix_suggestions(self, query: str) -> List[Tuple[str, float]]:
        """Get suggestions based on brand prefix matching."""
        suggestions = []
        
        # Check if query matches any brand prefix
        for prefix, completions in self.brand_prefixes.items():
            if query.lower().startswith(prefix.lower()):
                for completion in completions:
                    # Find actual queries that match this completion
                    matching_queries = [
                        q for q in self.all_queries 
                        if completion.lower() in q.lower()
                    ]
                    
                    for match in matching_queries[:3]:  # Top 3 matches
                        score = self.context_weights['brand_prefix_match']
                        suggestions.append((match, score))
        
        return suggestions
    
    def _get_enhanced_semantic_suggestions(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Enhanced semantic suggestions with better filtering."""
        if not self.faiss_index or not self.sbert_model:
            return []
        
        try:
            query_embedding = self.sbert_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            scores, indices = self.faiss_index.search(query_embedding, top_k)
            
            suggestions = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.all_queries):
                    suggestion = self.all_queries[idx]
                    metadata = self.query_metadata[idx]
                    
                    # Skip if too similar to query
                    if suggestion.lower() == query.lower():
                        continue
                    
                    # Boost score based on metadata
                    boosted_score = float(score) * metadata.get('popularity_score', 1.0)
                    suggestions.append((suggestion, boosted_score))
            
            return suggestions
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    def _combine_and_rank_suggestions(self, prefix_suggestions: List[Tuple[str, float]], 
                                    semantic_suggestions: List[Tuple[str, float]],
                                    original_query: str, corrected_query: str, 
                                    context: Dict) -> List[Tuple[str, float]]:
        """Combine and intelligently rank all suggestions."""
        
        # Combine suggestions
        combined = {}
        
        # Add prefix suggestions with high priority
        for suggestion, score in prefix_suggestions:
            combined[suggestion] = max(combined.get(suggestion, 0), score)
        
        # Add semantic suggestions
        for suggestion, score in semantic_suggestions:
            if suggestion not in combined:
                combined[suggestion] = score
        
        # Convert back to list and apply additional scoring
        suggestions = []
        for suggestion, score in combined.items():
            # Boost exact brand matches
            if self._is_exact_brand_match(suggestion, original_query):
                score *= self.context_weights['exact_brand_match']
            
            # Boost category matches
            if self._is_category_match(suggestion, original_query):
                score *= self.context_weights['category_match']
            
            suggestions.append((suggestion, score))
        
        return suggestions
    
    def _is_exact_brand_match(self, suggestion: str, query: str) -> bool:
        """Check if suggestion matches the brand in query."""
        query_brand = self._extract_brand_advanced(query)
        suggestion_brand = self._extract_brand_advanced(suggestion)
        
        return query_brand and suggestion_brand and query_brand.lower() == suggestion_brand.lower()
    
    def _is_category_match(self, suggestion: str, query: str) -> bool:
        """Check if suggestion matches the category implied by query."""
        query_lower = query.lower()
        suggestion_lower = suggestion.lower()
        
        for category, keywords in self.category_patterns.items():
            query_has_category = any(keyword in query_lower for keyword in keywords)
            suggestion_has_category = any(keyword in suggestion_lower for keyword in keywords)
            
            if query_has_category and suggestion_has_category:
                return True
        
        return False
    
    def _apply_enhanced_contextual_boosting(self, suggestions: List[Tuple[str, float]], context: Dict) -> List[Tuple[str, float]]:
        """Apply enhanced contextual boosting."""
        boosted_suggestions = []
        
        persona = context.get('persona', '')
        location = context.get('location', '')
        event = context.get('event', '')
        
        for suggestion, score in suggestions:
            boosted_score = score
            suggestion_lower = suggestion.lower()
            
            # Persona-based boosting
            if persona == 'tech_enthusiast':
                if any(tech in suggestion_lower for tech in ['samsung', 'xiaomi', 'apple', 'laptop', 'phone']):
                    boosted_score *= self.context_weights['persona_match']
            elif persona == 'sports_enthusiast':
                if any(sport in suggestion_lower for sport in ['nike', 'adidas', 'cricket', 'football', 'jersey']):
                    boosted_score *= self.context_weights['persona_match']
            
            # Event-based boosting
            if event and 'ipl' in event.lower():
                if any(ipl_term in suggestion_lower for ipl_term in ['cricket', 'jersey', 'ipl']):
                    boosted_score *= self.context_weights['event_match']
            
            boosted_suggestions.append((suggestion, boosted_score))
        
        return boosted_suggestions
    
    def get_suggestions(self, query: str, context: Dict = None, max_suggestions: int = 5) -> List[Tuple[str, float]]:
        """Main method to get suggestions (V5 enhanced)."""
        return self.get_intelligent_suggestions(query, context, max_suggestions)

# Test the enhanced system
if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    
    print("=== Testing Enhanced Autosuggest System V5 ===")
    
    # Load data
    preprocessor = DataPreprocessor()
    preprocessor.run_all_preprocessing()
    data = preprocessor.get_processed_data()
    
    # Build system
    autosuggest = EnhancedAutosuggestV5()
    autosuggest.build_system(data)
    
    # Critical test cases
    critical_tests = [
        ("sam", {"persona": "tech_enthusiast", "location": "Delhi", "event": "none"}),
        ("samsung", {"persona": "tech_enthusiast", "location": "Mumbai", "event": "none"}),
        ("xiomi", {"persona": "tech_enthusiast", "location": "Delhi", "event": "none"}),
        ("nike", {"persona": "sports_enthusiast", "location": "Chennai", "event": "none"}),
    ]
    
    print("\n=== CRITICAL BRAND RECOGNITION TESTS ===")
    for query, context in critical_tests:
        suggestions = autosuggest.get_suggestions(query, context)
        corrected = autosuggest.correct_spelling_advanced(query)
        
        print(f"\n🔍 Query: '{query}' → Corrected: '{corrected}'")
        print(f"💡 Suggestions: {[s for s, score in suggestions]}")
        print(f"📊 Scores: {[f'{score:.3f}' for s, score in suggestions]}")
        
        # Check if results make sense
        if query == "sam":
            samsung_found = any('samsung' in s.lower() for s, _ in suggestions)
            print(f"✅ Samsung found: {samsung_found}" if samsung_found else "❌ Samsung NOT found")
        elif query == "xiomi":
            xiaomi_found = any('xiaomi' in s.lower() for s, _ in suggestions)
            print(f"✅ Xiaomi found: {xiaomi_found}" if xiaomi_found else "❌ Xiaomi NOT found")