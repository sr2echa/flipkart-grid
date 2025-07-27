import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

class SemanticCorrection:
    """Semantic correction using SBERT for handling typos and semantic similarity."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.user_queries = None
        self.query_embeddings = None
        self.faiss_index = None
        self.corrected_queries = None
        
    def build_semantic_index(self, user_queries_df: pd.DataFrame):
        """Build semantic index from user queries."""
        print("Building semantic index from user queries...")
        
        self.user_queries = user_queries_df
        
        # Get unique corrected queries
        self.corrected_queries = user_queries_df['corrected_query'].unique().tolist()
        
        # Generate embeddings for all corrected queries
        print("Generating embeddings...")
        self.query_embeddings = self.model.encode(self.corrected_queries, show_progress_bar=True)
        
        # Build FAISS index
        print("Building FAISS index...")
        dimension = self.query_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.faiss_index.add(self.query_embeddings.astype('float32'))
        
        print(f"Semantic index built with {len(self.corrected_queries)} queries")
        
        # Save the index for later use
        self._save_index()
    
    def _save_index(self):
        """Save the semantic index to disk."""
        os.makedirs('../models', exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, '../models/semantic_index.faiss')
        
        # Save corrected queries
        with open('../models/corrected_queries.pkl', 'wb') as f:
            pickle.dump(self.corrected_queries, f)
        
        print("Semantic index saved to disk")
    
    def _load_index(self):
        """Load the semantic index from disk."""
        if os.path.exists('../models/semantic_index.faiss'):
            self.faiss_index = faiss.read_index('../models/semantic_index.faiss')
            
            with open('../models/corrected_queries.pkl', 'rb') as f:
                self.corrected_queries = pickle.load(f)
            
            print("Semantic index loaded from disk")
            return True
        return False
    
    def get_semantic_suggestions(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Get semantically similar suggestions for a query."""
        if not query.strip():
            return []
        
        # Load index if not already loaded
        if self.faiss_index is None:
            if not self._load_index():
                print("No semantic index found. Please build the index first.")
                return []
        
        # Encode the input query
        query_embedding = self.model.encode([query])
        
        # Search for similar queries
        similarities, indices = self.faiss_index.search(
            query_embedding.astype('float32'), top_k
        )
        
        # Return suggestions with similarity scores
        suggestions = []
        for idx, similarity in zip(indices[0], similarities[0]):
            if idx < len(self.corrected_queries):
                suggestions.append((self.corrected_queries[idx], float(similarity)))
        
        return suggestions
    
    def correct_typo(self, raw_query: str, top_k: int = 3) -> List[str]:
        """Correct typos using semantic similarity."""
        suggestions = self.get_semantic_suggestions(raw_query, top_k)
        return [query for query, _ in suggestions]
    
    def get_semantic_similarity_score(self, query1: str, query2: str) -> float:
        """Get semantic similarity score between two queries."""
        embeddings = self.model.encode([query1, query2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)

# Test the semantic correction
if __name__ == "__main__":
    # Load and preprocess data
    from data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    preprocessor.run_all_preprocessing()
    data = preprocessor.get_processed_data()
    
    # Initialize semantic correction
    semantic_correction = SemanticCorrection()
    
    # Build semantic index
    semantic_correction.build_semantic_index(data['user_queries'])
    
    # Test cases for typos and semantic similarity
    test_queries = [
        # Common typos
        "aple fon",           # Should suggest "apple phone"
        "samsng",             # Should suggest "samsung"
        "nkie",               # Should suggest "nike"
        "addidas",            # Should suggest "adidas"
        "soney",              # Should suggest "sony"
        "onepls",             # Should suggest "oneplus"
        "xiomi",              # Should suggest "xiaomi"
        "vvo",                # Should suggest "vivo"
        "opo",                # Should suggest "oppo"
        "realmi",             # Should suggest "realme"
        "del",                # Should suggest "dell"
        "lenvo",              # Should suggest "lenovo"
        "asuss",              # Should suggest "asus"
        "bot",                # Should suggest "boat"
        "jb",                 # Should suggest "jbl"
        "pma",                # Should suggest "puma"
        "rebbok",             # Should suggest "reebok"
        "zarra",              # Should suggest "zara"
        "ikia",               # Should suggest "ikea"
        "prestig",            # Should suggest "prestige"
        "h p",                # Should suggest "hp"
        "l g",                # Should suggest "lg"
        "bta",                # Should suggest "bata"
        "h m",                # Should suggest "hm"
        
        # Product typos
        "smartphon",          # Should suggest "smartphone"
        "mobil phone",        # Should suggest "mobile phone"
        "laptap",             # Should suggest "laptop"
        "headphons",          # Should suggest "headphones"
        "televisn",           # Should suggest "television"
        "sneakrs",            # Should suggest "sneakers"
        "smartwach",          # Should suggest "smartwatch"
        "tablit",             # Should suggest "tablet"
        "camra",              # Should suggest "camera"
        "speakr",             # Should suggest "speaker"
        "keybord",            # Should suggest "keyboard"
        "chargr",             # Should suggest "charger"
        "hoodi",              # Should suggest "hoodie"
        "jens",               # Should suggest "jeans"
        "notbook",            # Should suggest "notebook"
        "shoos",              # Should suggest "shoes"
        "wach",               # Should suggest "watch"
        "shrt",               # Should suggest "shirt"
        "walet",              # Should suggest "wallet"
        "mous",               # Should suggest "mouse"
        "cas",                # Should suggest "case"
        "bg",                 # Should suggest "bag"
        
        # Semantic variations
        "mobile",             # Should suggest "mobile phone"
        "phone",              # Should suggest "mobile phone"
        "smartphone",         # Should suggest "smartphone"
        "laptop",             # Should suggest "laptop"
        "computer",           # Should suggest "laptop"
        "headphones",         # Should suggest "headphones"
        "earbuds",            # Should suggest "earbuds"
        "tv",                 # Should suggest "tv"
        "television",         # Should suggest "television"
        "shoes",              # Should suggest "shoes"
        "sneakers",           # Should suggest "sneakers"
        "jeans",              # Should suggest "jeans"
        "shirt",              # Should suggest "shirt"
        "hoodie",             # Should suggest "hoodie"
        "watch",              # Should suggest "watch"
        "smartwatch",         # Should suggest "smartwatch"
        "tablet",             # Should suggest "tablet"
        "camera",             # Should suggest "camera"
        "speaker",            # Should suggest "speaker"
        "keyboard",           # Should suggest "keyboard"
        "mouse",              # Should suggest "mouse"
        "charger",            # Should suggest "charger"
        "case",               # Should suggest "case"
        "bag",                # Should suggest "bag"
        "wallet",             # Should suggest "wallet"
    ]
    
    print("\n=== Semantic Correction Test Results ===")
    
    for query in test_queries:
        start_time = time.time()
        suggestions = semantic_correction.correct_typo(query)
        end_time = time.time()
        
        print(f"\nQuery: '{query}'")
        print(f"Suggestions: {suggestions}")
        print(f"Response time: {(end_time - start_time)*1000:.2f}ms")
        
        # Test semantic similarity scores
        if suggestions:
            similarity_score = semantic_correction.get_semantic_similarity_score(query, suggestions[0])
            print(f"Top suggestion similarity: {similarity_score:.3f}")
    
    # Test performance
    print(f"\n=== Performance Test ===")
    test_query = "aple fon"
    start_time = time.time()
    for _ in range(100):
        semantic_correction.correct_typo(test_query)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100 * 1000
    print(f"Average response time for '{test_query}': {avg_time:.2f}ms")
    print(f"QPS: {100 / (end_time - start_time):.0f}")
    
    # Test semantic similarity between related terms
    print(f"\n=== Semantic Similarity Test ===")
    similarity_tests = [
        ("mobile", "mobile phone"),
        ("phone", "mobile phone"),
        ("smartphone", "mobile phone"),
        ("laptop", "computer"),
        ("headphones", "earbuds"),
        ("tv", "television"),
        ("shoes", "sneakers"),
        ("watch", "smartwatch"),
        ("camera", "phone"),
        ("speaker", "headphones"),
    ]
    
    for query1, query2 in similarity_tests:
        similarity = semantic_correction.get_semantic_similarity_score(query1, query2)
        print(f"'{query1}' vs '{query2}': {similarity:.3f}") 