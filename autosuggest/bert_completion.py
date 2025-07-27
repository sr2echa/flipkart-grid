import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import re

class BERTCompletion:
    """BERT-based context-aware suggestion completion."""
    
    def __init__(self, model_name: str = 'distilbert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.user_queries = None
        self.query_patterns = []
        
    def build_completion_patterns(self, user_queries_df: pd.DataFrame):
        """Build completion patterns from user queries."""
        print("Building BERT completion patterns...")
        
        self.user_queries = user_queries_df
        
        # Extract common patterns for completion
        self.query_patterns = self._extract_patterns()
        
        print(f"Built {len(self.query_patterns)} completion patterns")
    
    def _extract_patterns(self) -> List[str]:
        """Extract common query patterns for completion."""
        patterns = []
        
        # Common e-commerce patterns
        base_patterns = [
            "{} under {}",
            "{} above {}",
            "{} between {} and {}",
            "{} below {}",
            "{} for {}",
            "{} with {}",
            "{} {}",
            "{} {} {}",
        ]
        
        # Common product categories
        categories = ["phone", "laptop", "headphones", "shoes", "shirt", "watch", "tv", "tablet"]
        
        # Common price ranges
        price_ranges = ["1000", "2000", "5000", "10000", "15000", "20000", "30000", "50000"]
        
        # Common attributes
        attributes = ["gaming", "bluetooth", "wireless", "smart", "digital", "analog", "casual", "formal"]
        
        # Generate patterns
        for category in categories:
            for pattern in base_patterns:
                if "{}" in pattern:
                    if pattern.count("{}") == 1:
                        patterns.append(pattern.format(category))
                    elif pattern.count("{}") == 2:
                        for price in price_ranges:
                            patterns.append(pattern.format(category, price))
                    elif pattern.count("{}") == 3:
                        for attr in attributes:
                            patterns.append(pattern.format(category, attr, "men"))
                            patterns.append(pattern.format(category, attr, "women"))
        
        return patterns
    
    def complete_query(self, prefix: str, max_completions: int = 5) -> List[str]:
        """Complete a query prefix using BERT masked language modeling."""
        if not prefix.strip():
            return []
        
        prefix = prefix.lower().strip()
        
        # Try different completion strategies
        completions = []
        
        # Strategy 1: Direct pattern matching
        pattern_completions = self._pattern_completion(prefix)
        completions.extend(pattern_completions)
        
        # Strategy 2: BERT masked completion
        bert_completions = self._bert_masked_completion(prefix)
        completions.extend(bert_completions)
        
        # Strategy 3: Context-aware completion
        context_completions = self._context_completion(prefix)
        completions.extend(context_completions)
        
        # Remove duplicates and limit results
        unique_completions = list(dict.fromkeys(completions))
        return unique_completions[:max_completions]
    
    def _pattern_completion(self, prefix: str) -> List[str]:
        """Complete using predefined patterns."""
        completions = []
        
        # Common e-commerce completions
        if prefix.startswith("phone"):
            completions.extend([
                "phone under 10000",
                "phone above 20000",
                "phone between 10000 and 20000"
            ])
        elif prefix.startswith("laptop"):
            completions.extend([
                "laptop under 50000",
                "laptop above 100000",
                "laptop for gaming"
            ])
        elif prefix.startswith("headphone"):
            completions.extend([
                "headphones under 5000",
                "headphones above 10000",
                "bluetooth headphones"
            ])
        elif prefix.startswith("shoe"):
            completions.extend([
                "shoes under 2000",
                "shoes above 5000",
                "casual shoes"
            ])
        elif prefix.startswith("watch"):
            completions.extend([
                "watch under 5000",
                "watch above 10000",
                "smartwatch"
            ])
        elif prefix.startswith("tv"):
            completions.extend([
                "tv under 30000",
                "tv above 50000",
                "smart tv"
            ])
        elif prefix.startswith("tablet"):
            completions.extend([
                "tablet under 20000",
                "tablet above 30000"
            ])
        elif prefix.startswith("camera"):
            completions.extend([
                "camera under 10000",
                "camera above 20000"
            ])
        elif prefix.startswith("speaker"):
            completions.extend([
                "speaker under 5000",
                "bluetooth speaker"
            ])
        elif prefix.startswith("keyboard"):
            completions.extend([
                "keyboard under 2000",
                "mechanical keyboard"
            ])
        elif prefix.startswith("mouse"):
            completions.extend([
                "mouse under 1000",
                "gaming mouse"
            ])
        elif prefix.startswith("charger"):
            completions.extend([
                "charger under 1000",
                "fast charger"
            ])
        elif prefix.startswith("case"):
            completions.extend([
                "case under 500",
                "phone case"
            ])
        elif prefix.startswith("bag"):
            completions.extend([
                "bag under 2000",
                "laptop bag"
            ])
        elif prefix.startswith("wallet"):
            completions.extend([
                "wallet under 1000",
                "leather wallet"
            ])
        elif prefix.startswith("hoodie"):
            completions.extend([
                "hoodie under 2000",
                "casual hoodie"
            ])
        elif prefix.startswith("jean"):
            completions.extend([
                "jeans under 2000",
                "blue jeans"
            ])
        elif prefix.startswith("shirt"):
            completions.extend([
                "shirt under 1500",
                "formal shirt"
            ])
        elif prefix.startswith("sneaker"):
            completions.extend([
                "sneakers under 3000",
                "running sneakers"
            ])
        
        return completions
    
    def _bert_masked_completion(self, prefix: str) -> List[str]:
        """Complete using BERT masked language modeling."""
        completions = []
        
        # Create masked input
        masked_input = prefix + " [MASK]"
        
        try:
            # Tokenize input
            inputs = self.tokenizer(masked_input, return_tensors="pt")
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits
            
            # Get top predictions for the masked token
            masked_index = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0]
            if len(masked_index) > 0:
                masked_index = masked_index[0]
                probs = torch.softmax(predictions[0, masked_index], dim=-1)
                top_tokens = torch.topk(probs, 10).indices
                
                # Decode top tokens
                for token_id in top_tokens:
                    token = self.tokenizer.decode([token_id])
                    completion = prefix + " " + token.strip()
                    if len(completion.split()) <= 4:  # Limit completion length
                        completions.append(completion)
        
        except Exception as e:
            print(f"BERT completion error: {e}")
        
        return completions[:3]  # Limit BERT completions
    
    def _context_completion(self, prefix: str) -> List[str]:
        """Complete using context-aware patterns."""
        completions = []
        
        # Context-aware completions based on common e-commerce patterns
        if "phone" in prefix or "mobile" in prefix:
            completions.extend([
                prefix + " under 15000",
                prefix + " above 25000",
                prefix + " with camera",
                prefix + " for gaming"
            ])
        elif "laptop" in prefix:
            completions.extend([
                prefix + " under 50000",
                prefix + " for work",
                prefix + " with ssd",
                prefix + " gaming"
            ])
        elif "headphone" in prefix or "earbud" in prefix:
            completions.extend([
                prefix + " under 5000",
                prefix + " bluetooth",
                prefix + " wireless",
                prefix + " noise cancelling"
            ])
        elif "shoe" in prefix or "sneaker" in prefix:
            completions.extend([
                prefix + " under 3000",
                prefix + " for running",
                prefix + " casual",
                prefix + " sports"
            ])
        elif "watch" in prefix:
            completions.extend([
                prefix + " under 5000",
                prefix + " digital",
                prefix + " analog",
                prefix + " smart"
            ])
        elif "tv" in prefix:
            completions.extend([
                prefix + " under 30000",
                prefix + " smart",
                prefix + " 4k",
                prefix + " led"
            ])
        elif "tablet" in prefix:
            completions.extend([
                prefix + " under 20000",
                prefix + " android",
                prefix + " for kids"
            ])
        
        return completions

# Test the BERT completion
if __name__ == "__main__":
    # Load and preprocess data
    from data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    preprocessor.run_all_preprocessing()
    data = preprocessor.get_processed_data()
    
    # Initialize BERT completion
    bert_completion = BERTCompletion()
    bert_completion.build_completion_patterns(data['user_queries'])
    
    # Test cases
    test_prefixes = [
        "phone",
        "laptop",
        "headphone",
        "shoe",
        "watch",
        "tv",
        "tablet",
        "camera",
        "speaker",
        "keyboard",
        "mouse",
        "charger",
        "case",
        "bag",
        "wallet",
        "hoodie",
        "jean",
        "shirt",
        "sneaker",
        "smart",
        "bluetooth",
        "gaming",
        "casual",
        "formal",
        "digital",
        "wireless",
        "fast",
        "leather",
        "blue",
        "black",
        "white",
        "red",
        "green",
        "under",
        "above",
        "between",
        "for",
        "with",
    ]
    
    print("\n=== BERT Completion Test Results ===")
    
    for prefix in test_prefixes:
        start_time = time.time()
        completions = bert_completion.complete_query(prefix)
        end_time = time.time()
        
        print(f"\nPrefix: '{prefix}'")
        print(f"Completions: {completions}")
        print(f"Response time: {(end_time - start_time)*1000:.2f}ms")
    
    # Test performance
    print(f"\n=== Performance Test ===")
    test_prefix = "phone"
    start_time = time.time()
    for _ in range(50):
        bert_completion.complete_query(test_prefix)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 50 * 1000
    print(f"Average response time for '{test_prefix}': {avg_time:.2f}ms")
    print(f"QPS: {50 / (end_time - start_time):.0f}") 