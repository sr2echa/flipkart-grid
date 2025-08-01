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
from enhanced_trie import EnhancedTrieAutosuggest
from enhanced_semantic_correction import EnhancedSemanticCorrection
from bert_completion import BERTCompletion
from reranker_xgboost_v2 import RerankerXGBoostV2

class IntegratedAutosuggest:
    """
    V2: A fully integrated and enhanced autosuggest system with a powerful reranker.
    """
    
    def __init__(self):
        self.trie_autosuggest = EnhancedTrieAutosuggest()
        self.semantic_correction = EnhancedSemanticCorrection()
        self.bert_completion = BERTCompletion()
        self.reranker = RerankerXGBoostV2()
        self.data = {}

    def build_system(self, data: Dict):
        """Build the complete autosuggest system."""
        print("🚀 Building Integrated Autosuggest System V2...")
        self.data = data
        
        print("-> Building Trie component...")
        self.trie_autosuggest.build_trie(data['user_queries'])
        
        print("-> Building Semantic Correction component...")
        self.semantic_correction.build_semantic_index(data['user_queries'])
        
        print("-> Building BERT Completion component...")
        self.bert_completion.build_completion_patterns(data['user_queries'])
        
        print("-> Building Reranker V2 component...")
        self._build_reranker()
        
        print("✅ Integrated autosuggest system built successfully!")
    
    def _build_reranker(self):
        """Prepares data and trains the V2 reranker."""
        # First, check if a trained model already exists
        if self.reranker.load_model():
            return
            
        print("-> No pre-trained V2 reranker found. Training a new one.")
        features_df, labels, groups = self._prepare_reranker_data()
        self.reranker.train_model(features_df, labels, groups)

    def _prepare_reranker_data(self) -> Tuple[pd.DataFrame, pd.Series, list]:
        """Prepares rich features and labels from session logs for the reranker."""
        print("-> Preparing data for reranker training...")
        session_log = self.data['session_log']
        user_queries = self.data['user_queries']
        
        # We need to generate suggestions for each query in the log to create training data
        all_features = []
        
        # Use a smaller subset of sessions for faster training preparation during startup
        # For a production system, this training should be an offline process.
        sample_size = min(1000, len(session_log.groupby('session_id')))
        sampled_sessions = session_log.groupby('session_id').head(1).sample(n=sample_size, random_state=42)
        
        for _, session_info in sampled_sessions.iterrows():
            query = session_info['query']
            if not isinstance(query, str) or not query.strip():
                continue

            # Generate candidate suggestions for this historical query
            trie_suggs = self.trie_autosuggest.get_suggestions(query, 5)
            semantic_suggs = [s for s, _ in self.semantic_correction.get_semantic_suggestions(query, 2)]
            
            candidates = list(set(trie_suggs + semantic_suggs))
            if not candidates:
                continue

            # Create features for each candidate
            for sugg in candidates:
                features = {'suggestion_text': sugg, 'query_group_id': session_info['session_id']}
                
                # Context features
                context = {
                    'location': session_info.get('location'),
                    'event': session_info.get('event'),
                    'persona': session_info.get('persona_tag'),
                }
                features.update(self.reranker._get_contextual_features(sugg, context))
                
                # Base scores
                features['initial_score'] = self.trie_autosuggest._calculate_enhanced_score(sugg, 1, {}, query)
                features['query_length'] = len(query)
                features['suggestion_length'] = len(sugg)
                
                all_features.append(features)

        if not all_features:
            raise ValueError("No features were generated for reranker training.")

        features_df = pd.DataFrame(all_features).fillna(0)
        
        # Create labels based on interactions
        # This is a more realistic way to generate labels
        clicked_df = session_log[session_log['clicked_product_id'].notna()]
        purchased_df = session_log[session_log['purchased'] == True]

        def get_label(row):
            if row['suggestion_text'] in purchased_df['query'].values:
                return 2 # High relevance
            if row['suggestion_text'] in clicked_df['query'].values:
                return 1 # Medium relevance
            return 0 # Low relevance

        features_df['label'] = features_df.apply(get_label, axis=1)
        
        # Filter out rows with only 0 labels for a query group
        group_labels = features_df.groupby('query_group_id')['label'].sum()
        valid_groups = group_labels[group_labels > 0].index
        features_df = features_df[features_df['query_group_id'].isin(valid_groups)]

        if features_df.empty:
            raise ValueError("No valid training data left after filtering for non-zero labels.")

        labels = features_df['label']
        groups = features_df.groupby('query_group_id').size().tolist()
        
        # Drop helper columns
        features_df = features_df.drop(columns=['label', 'query_group_id'])
        
        print(f"-> Prepared {len(features_df)} samples for reranker training across {len(groups)} query groups.")
        return features_df, labels, groups

    def get_contextual_suggestions(self, query: str, context: Dict) -> List[Tuple[str, float]]:
        """The main entry point for getting high-quality, context-aware suggestions."""
        if not query.strip():
            return []
        
        query = query.lower().strip()
        
        # Step 1: Candidate Generation
        trie_suggs = self.trie_autosuggest.get_suggestions_with_scores(query, max_suggestions=7)
        semantic_suggs = self.semantic_correction.get_semantic_suggestions(query, 3)
        bert_suggs = self.bert_completion.complete_query(query, 2)
        
        # Combine and deduplicate
        all_suggestions = {}
        for s, score in trie_suggs:
            all_suggestions[s] = max(all_suggestions.get(s, 0), score)
        for s, score in semantic_suggs:
            all_suggestions[s] = max(all_suggestions.get(s, 0), score)
        for s in bert_suggs:
            all_suggestions[s] = max(all_suggestions.get(s, 0), 0.5) # Assign a default score for BERT suggestions
            
        candidate_suggestions = list(all_suggestions.items())
        
        if not candidate_suggestions:
            return []
            
        # Step 2: Reranking
        reranked_suggestions = self.reranker.rerank(query, candidate_suggestions, context)
        
        return reranked_suggestions[:5]

# Kept for backward compatibility if any old script calls it, but the main flow is contextual
    def get_suggestions(self, query: str, max_suggestions: int = 5) -> List[Tuple[str, float]]:
        return self.get_contextual_suggestions(query, context={})

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.run_all_preprocessing()
    data = preprocessor.get_processed_data()
    
    autosuggest = IntegratedAutosuggest()
    autosuggest.build_system(data)
    
    print("\n\n=== V2 Integration Test ===")
    
    test_query = "jersy"
    contexts = [
        {
            'persona': 'sports_enthusiast', 'location': 'Chennai', 'event': 'IPL', 
            'description': 'Sports fan in Chennai during IPL'
        },
        {
            'persona': 'fashion_lover', 'location': 'Mumbai', 'event': 'None',
            'description': 'Fashion lover in Mumbai'
        }
    ]

    for context in contexts:
        print(f"\n--- Testing with context: {context['description']} ---")
        suggestions = autosuggest.get_contextual_suggestions(test_query, context)
        print(f"Query: '{test_query}' -> Suggestions: {[s for s, score in suggestions]}")

    print("\n--- Testing typo correction ---")
    suggestions = autosuggest.get_contextual_suggestions("leptop", context={'persona': 'tech_enthusiast'})
    print(f"Query: 'leptop' -> Suggestions: {[s for s, score in suggestions]}")