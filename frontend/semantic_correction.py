#!/usr/bin/env python3
"""
Semantic Correction System using SBERT
=====================================

Embeds corrected queries using all-MiniLM-L6-v2 and stores in FAISS index
for intelligent typo correction and query suggestion improvement.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import faiss
from sentence_transformers import SentenceTransformer
import re
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticCorrection:
    """
    SBERT-based semantic correction system for handling typos and query variations.
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.sbert_model = None
        self.faiss_index = None
        self.corrected_queries = []
        self.raw_queries = []
        self.query_embeddings = None
        self.correction_mapping = {}
        self.index_built = False
        
        logger.info(f"🔧 Initializing Semantic Correction with {model_name}")
        
    def _load_sbert_model(self):
        """Load SBERT model for embeddings."""
        if self.sbert_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.sbert_model = SentenceTransformer(self.model_name)
                logger.info(f"✅ SBERT model loaded: {self.model_name}")
            except Exception as e:
                logger.error(f"❌ Failed to load SBERT model: {e}")
                raise
    
    def build_correction_index(self, query_csv_path: str = "../dataset/user_queries_enhanced_v2.csv"):
        """
        Build FAISS index from corrected queries for semantic similarity search.
        
        Args:
            query_csv_path: Path to CSV with raw_query and corrected_query columns
        """
        logger.info("🚀 Building semantic correction index...")
        
        try:
            # Load SBERT model
            self._load_sbert_model()
            
            # Load query data
            df = pd.read_csv(query_csv_path)
            logger.info(f"📊 Loaded {len(df)} query records")
            
            # Filter valid correction pairs
            valid_pairs = df.dropna(subset=['raw_query', 'corrected_query'])
            valid_pairs = valid_pairs[valid_pairs['raw_query'] != valid_pairs['corrected_query']]
            
            logger.info(f"🔍 Found {len(valid_pairs)} valid correction pairs")
            
            if len(valid_pairs) == 0:
                logger.warning("⚠️ No valid correction pairs found")
                return False
            
            # Extract unique corrected queries
            self.corrected_queries = valid_pairs['corrected_query'].unique().tolist()
            
            # Build mapping from raw to corrected
            for _, row in valid_pairs.iterrows():
                raw = row['raw_query'].lower().strip()
                corrected = row['corrected_query'].lower().strip()
                if raw not in self.correction_mapping:
                    self.correction_mapping[raw] = []
                if corrected not in self.correction_mapping[raw]:
                    self.correction_mapping[raw].append(corrected)
            
            logger.info(f"📝 Built correction mapping for {len(self.correction_mapping)} raw queries")
            
            # Generate embeddings for corrected queries
            logger.info("🧠 Generating embeddings for corrected queries...")
            self.query_embeddings = self.sbert_model.encode(
                self.corrected_queries, 
                normalize_embeddings=True,
                show_progress_bar=True
            )
            
            # Build FAISS index
            dimension = self.query_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for normalized embeddings
            self.faiss_index.add(self.query_embeddings.astype('float32'))
            
            self.index_built = True
            
            logger.info(f"✅ Semantic correction index built successfully!")
            logger.info(f"📊 Index contains {len(self.corrected_queries)} corrected queries")
            logger.info(f"🎯 Embedding dimension: {dimension}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to build correction index: {e}")
            return False
    
    def get_corrections(self, query: str, top_k: int = 3, min_similarity: float = 0.7) -> List[Tuple[str, float]]:
        """
        Get top-k semantic corrections for a potentially misspelled query.
        
        Args:
            query: Input query (potentially with typos)
            top_k: Number of corrections to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (corrected_query, similarity_score) tuples
        """
        if not self.index_built:
            logger.warning("⚠️ Correction index not built")
            return []
        
        if not query or len(query.strip()) == 0:
            return []
        
        query = query.lower().strip()
        
        try:
            # Check if exact correction exists
            if query in self.correction_mapping:
                exact_corrections = [(corr, 1.0) for corr in self.correction_mapping[query]]
                return exact_corrections[:top_k]
            
            # Generate embedding for input query
            query_embedding = self.sbert_model.encode([query], normalize_embeddings=True)
            
            # Search FAISS index
            similarities, indices = self.faiss_index.search(
                query_embedding.astype('float32'), 
                min(top_k * 2, len(self.corrected_queries))  # Get more candidates
            )
            
            # Filter by similarity threshold and return results
            corrections = []
            for sim, idx in zip(similarities[0], indices[0]):
                if sim >= min_similarity:
                    corrected_query = self.corrected_queries[idx]
                    corrections.append((corrected_query, float(sim)))
            
            # Remove duplicates and sort by similarity
            seen = set()
            unique_corrections = []
            for corr, sim in corrections:
                if corr not in seen:
                    seen.add(corr)
                    unique_corrections.append((corr, sim))
            
            unique_corrections.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"🔍 Found {len(unique_corrections)} semantic corrections for '{query}'")
            return unique_corrections[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Error getting corrections: {e}")
            return []
    
    def is_likely_typo(self, query: str, threshold: float = 0.8) -> bool:
        """
        Determine if a query likely contains typos based on similarity to known queries.
        
        Args:
            query: Input query
            threshold: Similarity threshold for typo detection
            
        Returns:
            True if query likely contains typos
        """
        if not self.index_built or not query:
            return False
        
        query = query.lower().strip()
        
        # Check exact match first
        if query in self.corrected_queries:
            return False
        
        # Check if corrections are available with high confidence
        corrections = self.get_corrections(query, top_k=1, min_similarity=threshold)
        return len(corrections) > 0
    
    def save_index(self, output_dir: str):
        """Save the correction index to disk."""
        if not self.index_built:
            logger.error("❌ No index to save")
            return False
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.faiss_index, os.path.join(output_dir, "correction_index.faiss"))
            
            # Save query data
            with open(os.path.join(output_dir, "corrected_queries.json"), 'w') as f:
                json.dump(self.corrected_queries, f)
            
            with open(os.path.join(output_dir, "correction_mapping.json"), 'w') as f:
                json.dump(self.correction_mapping, f)
            
            logger.info(f"✅ Correction index saved to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save index: {e}")
            return False
    
    def load_index(self, input_dir: str):
        """Load the correction index from disk."""
        try:
            # Load FAISS index
            index_path = os.path.join(input_dir, "correction_index.faiss")
            if not os.path.exists(index_path):
                logger.error(f"❌ Index file not found: {index_path}")
                return False
            
            self.faiss_index = faiss.read_index(index_path)
            
            # Load query data
            with open(os.path.join(input_dir, "corrected_queries.json"), 'r') as f:
                self.corrected_queries = json.load(f)
            
            with open(os.path.join(input_dir, "correction_mapping.json"), 'r') as f:
                self.correction_mapping = json.load(f)
            
            # Load SBERT model
            self._load_sbert_model()
            
            self.index_built = True
            logger.info(f"✅ Correction index loaded from {input_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load index: {e}")
            return False

def main():
    """Test the semantic correction system."""
    corrector = SemanticCorrection()
    
    # Build index
    if corrector.build_correction_index():
        # Test corrections
        test_queries = [
            "beist 4k monitor",
            "prmium perfume for men", 
            "adidas footbal shoes",
            "iphone 14 pro max",
            "samung galaxy s23"
        ]
        
        for query in test_queries:
            corrections = corrector.get_corrections(query)
            print(f"\nQuery: '{query}'")
            for corr, sim in corrections:
                print(f"  → '{corr}' (similarity: {sim:.3f})")

if __name__ == "__main__":
    main()