import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Import our components
from trie_autosuggest import TrieAutosuggest
from semantic_correction import SemanticCorrection
from bert_completion import BERTCompletion

class IntegratedAutosuggest:
    """Integrated autosuggest system combining all components."""
    
    def __init__(self):
        self.trie_autosuggest = TrieAutosuggest()
        self.semantic_correction = SemanticCorrection()
        self.bert_completion = BERTCompletion()
        self.reranker = None
        self.vectorizer = None
        self.user_queries = None
        self.session_log = None
        
    def build_system(self, data: Dict):
        """Build the complete autosuggest system."""
        print("Building integrated autosuggest system...")
        
        self.user_queries = data['user_queries']
        self.session_log = data['session_log']
        
        # Build Trie component
        print("Building Trie component...")
        self.trie_autosuggest.build_trie(data['user_queries'])
        
        # Build Semantic Correction component
        print("Building Semantic Correction component...")
        self.semantic_correction.build_semantic_index(data['user_queries'])
        
        # Build BERT Completion component
        print("Building BERT Completion component...")
        self.bert_completion.build_completion_patterns(data['user_queries'])
        
        # Build Reranker
        print("Building XGBoost Reranker...")
        self._build_reranker(data)
        
        print("Integrated autosuggest system built successfully!")
    
    def _build_reranker(self, data: Dict):
        """Build XGBoost reranker for final ranking."""
        # Prepare training data for reranker
        training_data = self._prepare_reranker_data(data)
        
        # Initialize XGBoost model
        self.reranker = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        # Train the model
        X = training_data['features']
        y = training_data['labels']
        
        self.reranker.fit(X, y)
        
        # Save the model
        os.makedirs('../models', exist_ok=True)
        with open('../models/reranker.pkl', 'wb') as f:
            pickle.dump(self.reranker, f)
        
        print("Reranker trained and saved")
    
    def _prepare_reranker_data(self, data: Dict) -> Dict:
        """Prepare training data for the reranker."""
        features_list = []
        labels_list = []
        
        # Create TF-IDF vectorizer for text similarity
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Get all corrected queries for vectorization
        all_queries = data['user_queries']['corrected_query'].tolist()
        self.vectorizer.fit(all_queries)
        
        # Generate training examples
        for _, row in data['user_queries'].iterrows():
            query = row['corrected_query']
            frequency = row['frequency']
            
            # Create positive example
            features = self._extract_features(query, query, frequency, data)
            features_list.append(features)
            labels_list.append(2.0)  # High relevance for exact match
            
            # Create negative examples (random queries with low frequency)
            negative_candidates = data['user_queries'][
                data['user_queries']['frequency'] < frequency
            ]
            if len(negative_candidates) > 0:
                sample_size = min(3, len(negative_candidates))
                negative_queries = negative_candidates.sample(sample_size)
            else:
                # If no negative candidates, use random queries
                negative_queries = data['user_queries'].sample(min(3, len(data['user_queries'])))
            
            for _, neg_row in negative_queries.iterrows():
                neg_query = neg_row['corrected_query']
                neg_frequency = neg_row['frequency']
                
                features = self._extract_features(query, neg_query, neg_frequency, data)
                features_list.append(features)
                labels_list.append(0.0)  # Low relevance
        
        return {
            'features': np.array(features_list),
            'labels': np.array(labels_list)
        }
    
    def _extract_features(self, original_query: str, candidate_query: str, 
                         frequency: int, data: Dict) -> List[float]:
        """Extract features for reranking."""
        features = []
        
        # 1. Frequency feature
        features.append(frequency)
        
        # 2. Text similarity (TF-IDF)
        try:
            query_vec = self.vectorizer.transform([original_query])
            candidate_vec = self.vectorizer.transform([candidate_query])
            similarity = cosine_similarity(query_vec, candidate_vec)[0][0]
            features.append(similarity)
        except:
            features.append(0.0)
        
        # 3. Length difference
        features.append(abs(len(original_query) - len(candidate_query)))
        
        # 4. Word overlap
        original_words = set(original_query.lower().split())
        candidate_words = set(candidate_query.lower().split())
        overlap = len(original_words.intersection(candidate_words))
        features.append(overlap)
        
        # 5. Category match (if available)
        category_match = 0.0
        if 'category' in data['user_queries'].columns:
            orig_category = data['user_queries'][
                data['user_queries']['corrected_query'] == original_query
            ]['category'].iloc[0] if len(data['user_queries'][
                data['user_queries']['corrected_query'] == original_query
            ]) > 0 else ''
            
            cand_category = data['user_queries'][
                data['user_queries']['corrected_query'] == candidate_query
            ]['category'].iloc[0] if len(data['user_queries'][
                data['user_queries']['corrected_query'] == candidate_query
            ]) > 0 else ''
            
            category_match = 1.0 if orig_category == cand_category else 0.0
        features.append(category_match)
        
        # 6. Event relevance (if available)
        event_relevance = 0.0
        if 'event' in data['user_queries'].columns:
            orig_event = data['user_queries'][
                data['user_queries']['corrected_query'] == original_query
            ]['event'].iloc[0] if len(data['user_queries'][
                data['user_queries']['corrected_query'] == original_query
            ]) > 0 else ''
            
            cand_event = data['user_queries'][
                data['user_queries']['corrected_query'] == candidate_query
            ]['event'].iloc[0] if len(data['user_queries'][
                data['user_queries']['corrected_query'] == candidate_query
            ]) > 0 else ''
            
            event_relevance = 1.0 if orig_event == cand_event else 0.0
        features.append(event_relevance)
        
        return features
    
    def get_suggestions(self, query: str, max_suggestions: int = 10, 
                       context: Dict = None) -> List[Tuple[str, float]]:
        """Get integrated autosuggestions."""
        if not query.strip():
            return []
        
        query = query.lower().strip()
        
        # Step 1: Get Trie suggestions
        trie_suggestions = self.trie_autosuggest.get_suggestions_with_scores(query)
        
        # Step 2: Get Semantic Correction suggestions
        semantic_suggestions = self.semantic_correction.get_semantic_suggestions(query)
        
        # Step 3: Get BERT Completion suggestions
        bert_suggestions = self.bert_completion.complete_query(query)
        
        # Step 4: Combine all suggestions
        all_suggestions = []
        
        # Add Trie suggestions
        for suggestion, frequency in trie_suggestions:
            all_suggestions.append((suggestion, 'trie', frequency))
        
        # Add Semantic suggestions
        for suggestion, similarity in semantic_suggestions:
            all_suggestions.append((suggestion, 'semantic', similarity))
        
        # Add BERT suggestions
        for suggestion in bert_suggestions:
            all_suggestions.append((suggestion, 'bert', 0.5))  # Default score
        
        # Step 5: Remove duplicates and rerank
        unique_suggestions = {}
        for suggestion, source, score in all_suggestions:
            if suggestion not in unique_suggestions:
                unique_suggestions[suggestion] = {'source': source, 'score': score}
            else:
                # Keep the higher score
                unique_suggestions[suggestion]['score'] = max(
                    unique_suggestions[suggestion]['score'], score
                )
        
        # Step 6: Rerank using XGBoost
        reranked_suggestions = []
        for suggestion, info in unique_suggestions.items():
            # Extract features for reranking
            features = self._extract_features(query, suggestion, info['score'], {
                'user_queries': self.user_queries
            })
            
            # Get reranker score
            reranker_score = self.reranker.predict([features])[0]
            
            reranked_suggestions.append((suggestion, reranker_score))
        
        # Sort by reranker score
        reranked_suggestions.sort(key=lambda x: x[1], reverse=True)
        
        return reranked_suggestions[:max_suggestions]
    
    def get_contextual_suggestions(self, query: str, session_context: Dict = None,
                                 location: str = None, event: str = None) -> List[Tuple[str, float]]:
        """Get contextual autosuggestions with session, location, and event awareness."""
        base_suggestions = self.get_suggestions(query)
        
        if not session_context and not location and not event:
            return base_suggestions
        
        # Apply contextual boosts
        contextual_suggestions = []
        
        for suggestion, score in base_suggestions:
            boosted_score = score
            
            # Location-based boost
            if location:
                location_boost = self._get_location_boost(suggestion, location)
                boosted_score += location_boost
            
            # Event-based boost
            if event:
                event_boost = self._get_event_boost(suggestion, event)
                boosted_score += event_boost
            
            # Session-based boost
            if session_context:
                session_boost = self._get_session_boost(suggestion, session_context)
                boosted_score += session_boost
            
            contextual_suggestions.append((suggestion, boosted_score))
        
        # Re-sort by boosted scores
        contextual_suggestions.sort(key=lambda x: x[1], reverse=True)
        
        return contextual_suggestions
    
    def _get_location_boost(self, suggestion: str, location: str) -> float:
        """Get location-based boost for suggestions."""
        # Simple location-based boosting
        location_keywords = {
            'mumbai': ['fast delivery', 'same day'],
            'delhi': ['express delivery', 'quick'],
            'bangalore': ['tech', 'gaming', 'laptop'],
            'chennai': ['traditional', 'formal'],
            'kolkata': ['budget', 'affordable'],
            'hyderabad': ['tech', 'mobile'],
            'pune': ['student', 'budget'],
            'ahmedabad': ['business', 'formal'],
            'jaipur': ['traditional', 'ethnic'],
            'lucknow': ['traditional', 'cultural']
        }
        
        location = location.lower()
        if location in location_keywords:
            for keyword in location_keywords[location]:
                if keyword in suggestion.lower():
                    return 0.1
        
        return 0.0
    
    def _get_event_boost(self, suggestion: str, event: str) -> float:
        """Get event-based boost for suggestions."""
        # Event-based boosting
        event_keywords = {
            'diwali': ['lights', 'decor', 'gifts', 'sweets', 'traditional'],
            'holi': ['colors', 'water', 'party', 'celebration'],
            'christmas': ['gifts', 'decor', 'winter', 'sweater'],
            'eid': ['traditional', 'clothes', 'gifts'],
            'rakhi': ['gifts', 'sweets', 'traditional'],
            'navratri': ['traditional', 'dress', 'garba'],
            'ganesh_chaturthi': ['traditional', 'decor', 'sweets'],
            'ipl': ['jersey', 'sports', 'cricket', 'team'],
            'wedding': ['formal', 'traditional', 'gifts', 'jewelry'],
            'birthday': ['gifts', 'party', 'celebration']
        }
        
        event = event.lower()
        if event in event_keywords:
            for keyword in event_keywords[event]:
                if keyword in suggestion.lower():
                    return 0.2
        
        return 0.0
    
    def _get_session_boost(self, suggestion: str, session_context: Dict) -> float:
        """Get session-based boost for suggestions."""
        boost = 0.0
        
        # Check if suggestion matches previous queries in session
        if 'previous_queries' in session_context:
            for prev_query in session_context['previous_queries']:
                if any(word in suggestion.lower() for word in prev_query.lower().split()):
                    boost += 0.05
        
        # Check if suggestion matches clicked categories
        if 'clicked_categories' in session_context:
            for category in session_context['clicked_categories']:
                if category.lower() in suggestion.lower():
                    boost += 0.1
        
        # Check if suggestion matches clicked brands
        if 'clicked_brands' in session_context:
            for brand in session_context['clicked_brands']:
                if brand.lower() in suggestion.lower():
                    boost += 0.15
        
        return boost

# Test the integrated autosuggest system
if __name__ == "__main__":
    # Load and preprocess data
    from data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    preprocessor.run_all_preprocessing()
    data = preprocessor.get_processed_data()
    
    # Initialize integrated autosuggest
    autosuggest = IntegratedAutosuggest()
    
    # Build the system
    autosuggest.build_system(data)
    
    # Test cases
    test_queries = [
        "sam",           # Should suggest "samsung"
        "app",           # Should suggest "apple"
        "nik",           # Should suggest "nike"
        "smart",         # Should suggest "smartphone", "smartwatch"
        "lap",           # Should suggest "laptop"
        "head",          # Should suggest "headphones"
        "sho",           # Should suggest "shoes"
        "tv",            # Should suggest "tv"
        "phone",         # Should suggest "mobile phone"
        "ear",           # Should suggest "earbuds"
        "key",           # Should suggest "keyboard"
        "char",          # Should suggest "charger"
        "watch",         # Should suggest "watch", "smartwatch"
        "tab",           # Should suggest "tablet"
        "cam",           # Should suggest "camera"
        "speak",         # Should suggest "speaker"
        "mous",          # Should suggest "mouse"
        "case",          # Should suggest "case"
        "bag",           # Should suggest "bag"
        "wallet",        # Should suggest "wallet"
        "hood",          # Should suggest "hoodie"
        "jean",          # Should suggest "jeans"
        "shirt",         # Should suggest "shirt", "t shirt"
        "sneak",         # Should suggest "sneakers"
        "notebook",      # Should suggest "notebook"
        "televis",       # Should suggest "television"
        "mobil",         # Should suggest "mobile phone"
        "smartphon",     # Should suggest "smartphone"
        "headphon",      # Should suggest "headphones"
        "earbud",        # Should suggest "earbuds"
        "televisn",      # Should suggest "television"
        "sneakr",        # Should suggest "sneakers"
        "smartwach",     # Should suggest "smartwatch"
        "tablit",        # Should suggest "tablet"
        "camra",         # Should suggest "camera"
        "speakr",        # Should suggest "speaker"
        "keybord",       # Should suggest "keyboard"
        "chargr",        # Should suggest "charger"
        "hoodi",         # Should suggest "hoodie"
        "jens",          # Should suggest "jeans"
        "notbook",       # Should suggest "notebook"
        "shoos",         # Should suggest "shoes"
        "wach",          # Should suggest "watch"
        "shrt",          # Should suggest "shirt"
        "walet",         # Should suggest "wallet"
        "mous",          # Should suggest "mouse"
        "cas",           # Should suggest "case"
        "bg",            # Should suggest "bag"
    ]
    
    print("\n=== Integrated Autosuggest Test Results ===")
    
    for query in test_queries:
        start_time = time.time()
        suggestions = autosuggest.get_suggestions(query)
        end_time = time.time()
        
        print(f"\nQuery: '{query}'")
        print(f"Suggestions: {suggestions[:5]}")
        print(f"Response time: {(end_time - start_time)*1000:.2f}ms")
    
    # Test contextual suggestions
    print(f"\n=== Contextual Suggestions Test ===")
    
    # Test with location context
    location_context = "Mumbai"
    test_query = "lights"
    suggestions = autosuggest.get_contextual_suggestions(
        test_query, 
        location=location_context,
        event="diwali"
    )
    
    print(f"\nQuery: '{test_query}' with location: {location_context}, event: diwali")
    print(f"Contextual suggestions: {suggestions[:5]}")
    
    # Test with session context
    session_context = {
        'previous_queries': ['samsung', 'mobile'],
        'clicked_categories': ['Electronics'],
        'clicked_brands': ['Samsung']
    }
    
    test_query = "phone"
    suggestions = autosuggest.get_contextual_suggestions(
        test_query,
        session_context=session_context
    )
    
    print(f"\nQuery: '{test_query}' with session context")
    print(f"Session-aware suggestions: {suggestions[:5]}")
    
    # Test performance
    print(f"\n=== Performance Test ===")
    test_query = "smart"
    start_time = time.time()
    for _ in range(100):
        autosuggest.get_suggestions(test_query)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100 * 1000
    print(f"Average response time for '{test_query}': {avg_time:.2f}ms")
    print(f"QPS: {100 / (end_time - start_time):.0f}") 