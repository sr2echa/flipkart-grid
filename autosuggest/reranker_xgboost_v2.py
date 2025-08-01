import xgboost as xgb
import pandas as pd
from typing import List, Dict, Tuple
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

class RerankerXGBoostV2:
    """
    An enhanced XGBoost reranker with sophisticated contextual and semantic features.
    """
    def __init__(self, iterations=100, learning_rate=0.1, max_depth=5):
        self.model = xgb.XGBRanker(
            objective='rank:ndcg',
            n_estimators=iterations,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
            tree_method='hist' # Faster for larger datasets
        )
        self.vectorizer = TfidfVectorizer()
        self.is_trained = False

    def train_model(self, features: pd.DataFrame, labels: pd.Series, group: list):
        """Train the XGBoost reranking model."""
        print("🚀 Training V2 XGBoost Reranker...")
        if features.empty or labels.empty:
            print("⚠️ Cannot train reranker: features or labels are empty.")
            return

        # Fit the text vectorizer on the suggestion text
        self.vectorizer.fit(features['suggestion_text'])
        
        self.model.fit(features.drop(columns=['suggestion_text']), labels, group=group)
        self.is_trained = True
        print("✅ Reranker V2 training complete.")
        self.save_model()

    def rerank(self, query: str, suggestions: List[Tuple[str, float]], context: Dict) -> List[Tuple[str, float]]:
        """Rerank suggestions based on query and context."""
        if not self.is_trained or not suggestions:
            return suggestions

        # Create features for the given suggestions
        features_df = self._create_inference_features(query, suggestions, context)
        
        # Predict scores
        scores = self.model.predict(features_df.drop(columns=['suggestion_text']))
        
        # Combine suggestions with new scores
        reranked_suggestions = list(zip([s[0] for s in suggestions], scores))
        
        # Sort by the new XGBoost score
        reranked_suggestions.sort(key=lambda x: x[1], reverse=True)
        
        return reranked_suggestions

    def _create_inference_features(self, query: str, suggestions: List[Tuple[str, float]], context: Dict) -> pd.DataFrame:
        """Create a feature DataFrame for a list of suggestions at inference time."""
        features = []
        for suggestion_text, initial_score in suggestions:
            feature_dict = {
                'initial_score': initial_score,
                'query_length': len(query),
                'suggestion_length': len(suggestion_text),
                'suggestion_text': suggestion_text,
            }
            
            # Contextual Features
            feature_dict.update(self._get_contextual_features(suggestion_text, context))
            features.append(feature_dict)
            
        features_df = pd.DataFrame(features)
        
        # Semantic Features (using the fitted vectorizer)
        query_vec = self.vectorizer.transform([query])
        suggestion_vecs = self.vectorizer.transform(features_df['suggestion_text'])
        
        # Calculate cosine similarity
        similarities = (suggestion_vecs * query_vec.T).toarray()
        features_df['semantic_similarity'] = similarities
        
        return features_df

    def _get_contextual_features(self, suggestion_text: str, context: Dict) -> Dict:
        """Extracts advanced contextual features."""
        features = {
            'is_diwali': 0, 'is_ipl': 0, 'is_sale': 0,
            'location_boost': 0.0
        }
        
        event = str(context.get('event', '')).lower()
        if 'diwali' in event and any(k in suggestion_text for k in ['light', 'diya', 'gift']):
            features['is_diwali'] = 1
        if 'ipl' in event and any(k in suggestion_text for k in ['jersey', 'cricket', 'csk', 'rcb']):
            features['is_ipl'] = 1
        if 'sale' in event and any(k in suggestion_text for k in ['offer', 'discount', 'sale']):
            features['is_sale'] = 1
            
        location = str(context.get('location', '')).lower()
        if location in ['bangalore', 'hyderabad'] and any(k in suggestion_text for k in ['laptop', 'phone', 'gadget']):
            features['location_boost'] = 0.5
        if location in ['mumbai', 'delhi'] and any(k in suggestion_text for k in ['fashion', 'luxury', 'zara']):
            features['location_boost'] = 0.5
            
        return features

    def save_model(self, path: str = "../models/reranker_xgboost_v2.pkl"):
        """Save the trained model and vectorizer."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'vectorizer': self.vectorizer}, f)
        print(f"✅ Reranker V2 model saved to {path}")

    def load_model(self, path: str = "../models/reranker_xgboost_v2.pkl") -> bool:
        """Load the trained model and vectorizer."""
        if not os.path.exists(path):
            return False
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.vectorizer = data['vectorizer']
        
        self.is_trained = True
        print(f"✅ Reranker V2 model loaded from {path}")
        return True