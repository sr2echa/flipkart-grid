import numpy as np
import pandas as pd
import xgboost as xgb
from sentence_transformers import SentenceTransformer
import torch
import pickle
import os
from typing import List, Dict, Tuple, Optional, Any

class QueryReranker:
    """XGBoost-based reranker for autosuggest queries."""
    
    def __init__(
        self,
        trie_autosuggest: Any,
        semantic_correction: Any,
        model_path: str = '../models/reranker.xgb',
        festival_keywords: Optional[List[str]] = None
    ):
        self.model_path = model_path
        self.model = None
        self.trie = trie_autosuggest
        self.semantic_correction = semantic_correction
        self.semantic_model = semantic_correction.model  # Shared embedding model
        self.query_frequency = self._get_frequency_dict()
        self.persona_mapping = {}
        self.festival_keywords = festival_keywords or [
            'diwali', 'christmas', 'eid', 'holi', 'dussehra', 'navratri'
        ]
    
    def _get_frequency_dict(self) -> Dict[str, int]:
        """Create frequency dictionary from Trie data."""
        if hasattr(self.trie, 'user_queries') and self.trie.user_queries is not None:
            return self.trie.user_queries.groupby('corrected_query')['frequency'].sum().to_dict()
        return {}
        
    def build_persona_mapping(self, session_log_df: pd.DataFrame):
        """Build persona mapping from session logs."""
        print("Building persona mapping...")
        for _, row in session_log_df.iterrows():
            session_id = row['session_id']
            query = row['query'].lower()
            
            if 'brand' in query or 'apple' in query or 'samsung' in query:
                self.persona_mapping[session_id] = 'brand_loyalist'
            elif 'offer' in query or 'discount' in query or 'deal' in query:
                self.persona_mapping[session_id] = 'offer_seeker'
            elif 'cheap' in query or 'budget' in query or 'affordable' in query:
                self.persona_mapping[session_id] = 'budget_friendly'
        print(f"Built persona mapping for {len(self.persona_mapping)} sessions")

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train XGBoost reranker model."""
        print("Training XGBoost reranker model...")
        
        self.model = xgb.XGBRanker(
            objective='rank:pairwise',
            learning_rate=0.1,
            gamma=0.1,
            min_child_weight=0.1,
            max_depth=6,
            n_estimators=100
        )
        
        # Create groups (simulate sessions)
        groups = np.array([100] * (len(X_train) // 100))
        
        self.model.fit(X_train, y_train, group=groups)
        print("Model training complete")
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save_model(self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load pretrained XGBoost model."""
        if os.path.exists(self.model_path):
            self.model = xgb.XGBRanker()
            self.model.load_model(self.model_path)
            print("Reranker model loaded from disk")
            return True
        return False

    def extract_features(
        self,
        candidate_query: str,
        prefix: str,
        session_context: dict,
        semantic_sim: float
    ) -> List[float]:
        """Extract features for a candidate query."""
        features = []
        
        # 1. Query frequency feature
        freq = self.query_frequency.get(candidate_query, 1)
        features.append(np.log1p(freq))
        
        # 2. Semantic similarity score
        features.append(semantic_sim)
        
        # 3. Session history relevance
        if session_context.get('session_queries'):
            session_emb = self.semantic_model.encode(session_context['session_queries'])
            candidate_emb = self.semantic_model.encode([candidate_query])
            session_sim = np.dot(candidate_emb[0], np.mean(session_emb, axis=0))
            features.append(session_sim)
        else:
            features.append(0.0)
        
        # 4. Festival relevance
        festival_boost = 1.0 if any(
            fest in candidate_query 
            for fest in self.festival_keywords
        ) else 0.0
        features.append(festival_boost)
        
        # 5. Persona matching
        persona_match = 0.0
        session_id = session_context.get('session_id')
        if session_id and session_id in self.persona_mapping:
            persona = self.persona_mapping[session_id]
            if persona == 'brand_loyalist' and \
               any(brand in candidate_query for brand in ['apple', 'samsung']):
                persona_match = 1.0
            elif persona == 'offer_seeker' and \
                 ('offer' in candidate_query or 'discount' in candidate_query):
                persona_match = 1.0
            elif persona == 'budget_friendly' and \
                 ('cheap' in candidate_query or 'affordable' in candidate_query):
                persona_match = 1.0
        features.append(persona_match)
        
        # 6. Query length
        features.append(len(candidate_query.split()))
        
        # 7. Prefix match ratio
        prefix_match = len(prefix) / len(candidate_query) if prefix and candidate_query else 0.0
        features.append(prefix_match)
        
        return features

    def rerank_queries(
        self,
        prefix: str,
        candidates: List[Tuple[str, float]],
        session_context: dict,
        top_k: int = 5
    ) -> List[str]:
        """Rerank candidate queries using XGBoost."""
        if not self.model:
            print("Warning: No reranker model loaded. Returning original ranking")
            return [candidate for candidate, _ in candidates[:top_k]]
        
        # Prepare features for all candidates
        X = []
        candidate_list = []
        
        for candidate, semantic_score in candidates:
            features = self.extract_features(
                candidate, 
                prefix, 
                session_context, 
                semantic_score
            )
            X.append(features)
            candidate_list.append(candidate)
        
        # Predict scores
        scores = self.model.predict(np.array(X))
        
        # Sort candidates by predicted scores
        ranked_indices = np.argsort(scores)[::-1]  # Descending order
        reranked = [candidate_list[i] for i in ranked_indices[:top_k]]
        
        return reranked

# Test the reranker
if __name__ == "__main__":
    # Mock data for testing
    user_queries = pd.DataFrame({
        'corrected_query': ['iphone 13', 'samsung galaxy', 'cheap headphones', 
                           'diwali offers', 'gaming laptop'],
        'frequency': [1500, 1200, 800, 2000, 700]
    })
    
    session_log = pd.DataFrame({
        'session_id': ['sess1', 'sess1', 'sess2', 'sess2', 'sess3'],
        'query': ['apple iphone', 'samsung deal', 'budget headphones', 
                 'diwali discounts', 'gaming laptop offer']
    })
    
    # Initialize components
    class MockTrie:
        def __init__(self, user_queries):
            self.user_queries = user_queries
            
    class MockSemanticCorrection:
        def __init__(self):
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    trie = MockTrie(user_queries)
    semantic_corrector = MockSemanticCorrection()
    
    # Initialize reranker
    reranker = QueryReranker(
        trie_autosuggest=trie,
        semantic_correction=semantic_corrector,
        festival_keywords=['diwali', 'christmas', 'eid']
    )
    reranker.build_persona_mapping(session_log)
    
    # Mock training data
    X_train = np.random.rand(500, 7)  # 7 features
    y_train = np.random.randint(0, 2, 500)  # Binary relevance
    reranker.train_model(X_train, y_train)
    
    # Test reranking
    prefix = "gaming"
    candidates = [
        ('gaming laptop', 0.92),
        ('gaming mouse', 0.85),
        ('gaming chair', 0.78),
        ('gaming pc', 0.88),
        ('gaming accessories', 0.75)
    ]
    
    session_context = {
        'session_id': 'sess3',
        'session_queries': ['gaming accessories', 'cheap gaming gear']
    }
    
    print("\nBefore reranking:")
    print([c[0] for c in candidates])
    
    reranked = reranker.rerank_queries(prefix, candidates, session_context)
    
    print("\nAfter reranking:")
    print(reranked)
    
    # Test with festival context
    festival_candidates = [
        ('diwali lights', 0.76),
        ('diwali gifts', 0.92),
        ('christmas decorations', 0.65),
        ('eid special', 0.58),
        ('diwali offers', 0.95)
    ]
    
    festival_context = {
        'session_id': 'sess2',
        'session_queries': ['diwali sale', 'festival discounts']
    }
    
    print("\nFestival candidates before reranking:")
    print([c[0] for c in festival_candidates])
    
    reranked_festival = reranker.rerank_queries("diw", festival_candidates, festival_context)
    
    print("\nFestival candidates after reranking:")
    print(reranked_festival)