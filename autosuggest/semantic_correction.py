import pandas as pd
import numpy as np
from typing import List, Tuple
import pickle
import os
import faiss
from sentence_transformers import SentenceTransformer
import difflib

class SemanticCorrection:
    """Semantic correction using SBERT embeddings and FAISS."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.faiss_index = None
        self.corrected_queries = []
        self.query_frequencies = {}
        
    def build_semantic_index(self, user_queries_df: pd.DataFrame):
        """Build FAISS index for semantic search."""
        print("Building semantic index...")
        
        # Extract corrected queries and frequencies
        self.corrected_queries = user_queries_df['corrected_query'].dropna().unique().tolist()
        
        # Create frequency mapping
        for _, row in user_queries_df.iterrows():
            query = row['corrected_query']
            freq = row.get('frequency', 1)
            if query in self.query_frequencies:
                self.query_frequencies[query] = max(self.query_frequencies[query], freq)
            else:
                self.query_frequencies[query] = freq
        
        # Encode all queries
        embeddings = self.model.encode(self.corrected_queries)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.faiss_index.add(embeddings.astype('float32'))
        
        # Save index for future use
        self._save_index()
        print(f"✅ Semantic index built with {len(self.corrected_queries)} queries")
    
    def _save_index(self):
        """Save FAISS index and metadata."""
        if not os.path.exists('models'):
            os.makedirs('models')
        
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, 'models/semantic_index.faiss')
        
        with open('models/semantic_metadata.pkl', 'wb') as f:
            pickle.dump({
                'corrected_queries': self.corrected_queries,
                'query_frequencies': self.query_frequencies
            }, f)
    
    def _load_index(self):
        """Load FAISS index and metadata."""
        try:
            if os.path.exists('models/semantic_index.faiss'):
                self.faiss_index = faiss.read_index('models/semantic_index.faiss')
                
            if os.path.exists('models/semantic_metadata.pkl'):
                with open('models/semantic_metadata.pkl', 'rb') as f:
                    metadata = pickle.load(f)
                    self.corrected_queries = metadata['corrected_queries']
                    self.query_frequencies = metadata['query_frequencies']
                return True
        except Exception as e:
            print(f"Error loading semantic index: {e}")
        return False

    def get_semantic_suggestions(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Get semantic suggestions for a query using SBERT embeddings."""
        if not query.strip():
            return []
        
        query_lower = query.lower().strip()
        
        # First, try simple edit distance for common typos
        simple_corrections = self._get_simple_typo_corrections(query_lower)
        if simple_corrections:
            return simple_corrections[:top_k]
        
        # Then try semantic similarity
        try:
            # Encode the query
            query_embedding = self.model.encode([query_lower])
            
            # Search in FAISS index
            similarities, indices = self.faiss_index.search(query_embedding, min(top_k * 3, len(self.corrected_queries)))
            
            # Calculate proper cosine similarity and filter
            suggestions = {}  # Use a dictionary to store unique suggestions and their highest score
            for idx, faiss_similarity in zip(indices[0], similarities[0]):
                if idx < len(self.corrected_queries):
                    candidate_query = self.corrected_queries[idx]
                    
                    # Calculate proper cosine similarity
                    query_embedding_single = self.model.encode(query_lower)
                    candidate_embedding = self.model.encode(candidate_query)
                    cosine_similarity = np.dot(query_embedding_single, candidate_embedding) / (
                        np.linalg.norm(query_embedding_single) * np.linalg.norm(candidate_embedding)
                    )
                    
                    # Normalize to 0-1 range (adjusting from -1 to 1 to 0 to 1)
                    normalized_similarity = (cosine_similarity + 1) / 2
                    
                    # Only include if similarity is reasonable (above 0.3)
                    if normalized_similarity > 0.3:
                        # Store the highest score for each unique suggestion
                        if candidate_query not in suggestions or normalized_similarity > suggestions[candidate_query]:
                            suggestions[candidate_query] = normalized_similarity
            
            # Convert to list and sort by score
            result = [(suggestion, score) for suggestion, score in suggestions.items()]
            result.sort(key=lambda x: x[1], reverse=True)
            
            # Add dynamic suggestions based on query patterns
            dynamic_suggestions = self._generate_dynamic_suggestions(query_lower)
            for suggestion, score in dynamic_suggestions:
                if suggestion not in [s[0] for s in result]:
                    result.append((suggestion, score))
            
            # Sort again and return top_k
            result.sort(key=lambda x: x[1], reverse=True)
            return result[:top_k]
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            # Fallback to brand suggestions
            return self._get_brand_suggestions(query_lower)

    def _generate_dynamic_suggestions(self, query: str) -> List[Tuple[str, float]]:
        """Generate dynamic suggestions based on query patterns."""
        suggestions = []
        query_words = query.split()
        
        # Product category patterns
        category_patterns = {
            'laptop': ['gaming laptop', 'business laptop', 'student laptop', 'budget laptop', 'premium laptop'],
            'phone': ['smartphone', 'mobile phone', 'android phone', 'iphone', 'budget phone', 'premium phone'],
            'mobile': ['smartphone', 'mobile phone', 'android phone', 'iphone', 'budget mobile', 'premium mobile'],
            'shoes': ['running shoes', 'casual shoes', 'formal shoes', 'sports shoes', 'sneakers', 'boots'],
            'shirt': ['formal shirt', 'casual shirt', 'polo shirt', 't-shirt', 'dress shirt'],
            'headphones': ['wireless headphones', 'bluetooth headphones', 'noise cancelling headphones', 'gaming headphones'],
            'camera': ['dslr camera', 'mirrorless camera', 'action camera', 'point and shoot', 'canon camera', 'nikon camera'],
            'watch': ['smartwatch', 'digital watch', 'analog watch', 'fitness watch', 'luxury watch'],
            'bag': ['laptop bag', 'handbag', 'backpack', 'travel bag', 'messenger bag'],
            'jersey': ['cricket jersey', 'football jersey', 'ipl jersey', 'team jersey', 'sports jersey'],
            'formal': ['formal shirt', 'formal shoes', 'formal dress', 'formal pants', 'formal wear'],
            'gaming': ['gaming laptop', 'gaming mouse', 'gaming keyboard', 'gaming headset', 'gaming chair'],
            'wireless': ['wireless headphones', 'wireless earbuds', 'wireless keyboard', 'wireless mouse', 'wireless speaker'],
            'budget': ['budget phone', 'budget laptop', 'budget headphones', 'budget camera', 'budget watch'],
            'premium': ['premium phone', 'premium laptop', 'premium headphones', 'premium camera', 'premium watch']
        }
        
        # Brand patterns
        brand_patterns = {
            'samsung': ['samsung galaxy', 'samsung mobile', 'samsung phone', 'samsung tablet', 'samsung tv'],
            'apple': ['iphone', 'ipad', 'macbook', 'apple watch', 'airpods'],
            'nike': ['nike shoes', 'nike sneakers', 'nike running', 'nike sports', 'nike air max'],
            'adidas': ['adidas shoes', 'adidas sneakers', 'adidas running', 'adidas sports'],
            'xiaomi': ['xiaomi phone', 'xiaomi mobile', 'xiaomi redmi', 'xiaomi mi', 'xiaomi pocophone'],
            'oneplus': ['oneplus phone', 'oneplus mobile', 'oneplus nord', 'oneplus 11'],
            'dell': ['dell laptop', 'dell inspiron', 'dell xps', 'dell latitude'],
            'hp': ['hp laptop', 'hp pavilion', 'hp envy', 'hp spectre'],
            'canon': ['canon camera', 'canon dslr', 'canon mirrorless', 'canon lens'],
            'nikon': ['nikon camera', 'nikon dslr', 'nikon mirrorless', 'nikon lens']
        }
        
        # Feature patterns
        feature_patterns = {
            'wireless': ['wireless headphones', 'wireless earbuds', 'wireless keyboard', 'wireless mouse'],
            'bluetooth': ['bluetooth headphones', 'bluetooth speaker', 'bluetooth keyboard'],
            'gaming': ['gaming laptop', 'gaming mouse', 'gaming keyboard', 'gaming headset'],
            'waterproof': ['waterproof phone', 'waterproof watch', 'waterproof camera'],
            'fast': ['fast charger', 'fast laptop', 'fast phone', 'fast delivery'],
            'lightweight': ['lightweight laptop', 'lightweight headphones', 'lightweight bag'],
            'portable': ['portable speaker', 'portable charger', 'portable camera'],
            'smart': ['smartphone', 'smartwatch', 'smart tv', 'smart speaker']
        }
        
        # Use case patterns
        use_case_patterns = {
            'student': ['student laptop', 'student bag', 'student headphones', 'student watch'],
            'business': ['business laptop', 'business shirt', 'business bag', 'business watch'],
            'gaming': ['gaming laptop', 'gaming mouse', 'gaming keyboard', 'gaming headset'],
            'fitness': ['fitness watch', 'fitness tracker', 'sports shoes', 'fitness headphones'],
            'travel': ['travel bag', 'travel camera', 'travel adapter', 'travel pillow'],
            'office': ['office chair', 'office desk', 'office laptop', 'office headphones'],
            'home': ['home speaker', 'home camera', 'home laptop', 'home tv'],
            'outdoor': ['outdoor camera', 'outdoor watch', 'outdoor shoes', 'outdoor bag']
        }
        
        # Match query to patterns
        for pattern, suggestions_list in category_patterns.items():
            if pattern in query:
                for suggestion in suggestions_list:
                    suggestions.append((suggestion, 0.85))
        
        for brand, brand_suggestions in brand_patterns.items():
            if brand in query:
                for suggestion in brand_suggestions:
                    suggestions.append((suggestion, 0.90))
        
        for feature, feature_suggestions in feature_patterns.items():
            if feature in query:
                for suggestion in feature_suggestions:
                    suggestions.append((suggestion, 0.80))
        
        for use_case, use_case_suggestions in use_case_patterns.items():
            if use_case in query:
                for suggestion in use_case_suggestions:
                    suggestions.append((suggestion, 0.75))
        
        # Generate combinations based on query words
        if len(query_words) >= 2:
            for i, word1 in enumerate(query_words):
                for j, word2 in enumerate(query_words[i+1:], i+1):
                    if len(word1) > 2 and len(word2) > 2:
                        combination = f"{word1} {word2}"
                        suggestions.append((combination, 0.70))
        
        return suggestions

    def _get_simple_typo_corrections(self, query: str) -> List[Tuple[str, float]]:
        """Get simple typo corrections using direct mapping and edit distance."""
        # Direct typo mappings
        typo_mappings = {
            'laptap': 'laptop',
            'samsng': 'samsung',
            'nkie': 'nike',
            'addidas': 'adidas',
            'xiomi': 'xiaomi',
            'headphons': 'headphones',
            'camra': 'camera',
            'mobil': 'mobile',
            'sho': 'shoes',
            'phon': 'phone',
            'soney': 'sony',
            'onepls': 'oneplus',
            'vvo': 'vivo',
            'opo': 'oppo',
            'realmi': 'realme',
            'del': 'dell',
            'lenvo': 'lenovo',
            'asuss': 'asus',
            'bot': 'boat',
            'jb': 'jbl',
            'pma': 'puma',
            'rebbok': 'reebok',
            'zarra': 'zara',
            'ikia': 'ikea',
            'prestig': 'prestige',
            'h p': 'hp',
            'l g': 'lg',
            'bta': 'bata',
            'h m': 'hm'
        }
        
        # Check for direct mappings first
        if query in typo_mappings:
            return [(typo_mappings[query], 0.95)]
        
        # Check for partial matches
        for typo, correction in typo_mappings.items():
            if typo in query or query in typo:
                return [(correction, 0.90)]
        
        # Use edit distance for fuzzy matching
        best_match = None
        best_score = 0
        
        for correction in typo_mappings.values():
            similarity = difflib.SequenceMatcher(None, query, correction).ratio()
            if similarity > 0.7 and similarity > best_score:
                best_match = correction
                best_score = similarity
        
        if best_match:
            return [(best_match, best_score)]
        
        return []
    
    def _get_brand_suggestions(self, query: str) -> List[Tuple[str, float]]:
        """Get brand-specific suggestions."""
        brand_mappings = {
            'samsung': ['samsung mobile', 'samsung galaxy s23', 'samsung earbuds', 'samsung tv'],
            'apple': ['iphone 15 pro', 'apple watch', 'macbook air'],
            'nike': ['nike shoes', 'nike running shoes', 'nike sneakers'],
            'adidas': ['adidas running shoes', 'adidas sneakers', 'adidas sports shoes'],
            'puma': ['puma sneakers', 'puma sports shoes', 'puma casual shoes'],
            'sony': ['sony headphones', 'sony tv', 'sony camera'],
            'xiaomi': ['xiaomi phone', 'xiaomi redmi', 'xiaomi mi'],
            'oneplus': ['oneplus phone', 'oneplus nord', 'oneplus 11'],
            'vivo': ['vivo mobile', 'vivo v series', 'vivo x series'],
            'oppo': ['oppo reno', 'oppo find', 'oppo a series'],
            'realme': ['realme smartphone', 'realme narzo', 'realme gt'],
            'dell': ['dell laptop', 'dell inspiron', 'dell xps'],
            'hp': ['hp laptop', 'hp pavilion', 'hp elitebook'],
            'lenovo': ['lenovo laptop', 'lenovo thinkpad', 'lenovo ideapad'],
            'asus': ['asus laptop', 'asus rog', 'asus vivobook'],
            'canon': ['canon camera', 'canon dslr', 'canon mirrorless'],
            'nikon': ['nikon dslr', 'nikon mirrorless', 'nikon camera'],
            'jbl': ['jbl speaker', 'jbl headphones', 'jbl earbuds'],
            'bose': ['bose earbuds', 'bose headphones', 'bose speaker']
        }
        
        query_lower = query.lower()
        suggestions = []
        
        # Check exact brand matches
        if query_lower in brand_mappings:
            for suggestion in brand_mappings[query_lower]:
                suggestions.append((suggestion, 0.9))
        
        # Check fuzzy brand matches
        for brand, brand_suggestions in brand_mappings.items():
            similarity = difflib.SequenceMatcher(None, query_lower, brand).ratio()
            if similarity > 0.8:
                for suggestion in brand_suggestions:
                    suggestions.append((suggestion, similarity * 0.8))
        
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