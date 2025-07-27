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
    
    def complete_query(self, query: str, max_suggestions: int = 5) -> List[str]:
        """Complete a query using BERT and pattern matching."""
        if not query.strip():
            return []
        
        query_lower = query.lower().strip()
        suggestions = []
        
        # 1. Pattern-based completions
        pattern_suggestions = self._get_pattern_completions(query_lower)
        suggestions.extend(pattern_suggestions)
        
        # 2. Contextual completions
        contextual_suggestions = self._get_contextual_completions(query_lower)
        suggestions.extend(contextual_suggestions)
        
        # 3. Brand completions
        brand_suggestions = self._get_brand_completions(query_lower)
        suggestions.extend(brand_suggestions)
        
        # 4. Category completions
        category_suggestions = self._get_category_completions(query_lower)
        suggestions.extend(category_suggestions)
        
        # 5. Feature completions
        feature_suggestions = self._get_feature_completions(query_lower)
        suggestions.extend(feature_suggestions)
        
        # 6. Use case completions
        use_case_suggestions = self._get_use_case_completions(query_lower)
        suggestions.extend(use_case_suggestions)
        
        # 7. Dynamic completions based on query analysis
        dynamic_suggestions = self._get_dynamic_completions(query_lower)
        suggestions.extend(dynamic_suggestions)
        
        # Remove duplicates and return top suggestions
        unique_suggestions = list(dict.fromkeys(suggestions))  # Preserve order
        return unique_suggestions[:max_suggestions]

    def _get_pattern_completions(self, query: str) -> List[str]:
        """Get pattern-based completions."""
        suggestions = []
        
        # Product patterns
        product_patterns = {
            'laptop': ['gaming laptop', 'business laptop', 'student laptop', 'budget laptop', 'premium laptop', 'dell laptop', 'hp laptop'],
            'phone': ['smartphone', 'mobile phone', 'android phone', 'iphone', 'budget phone', 'premium phone', '5g phone'],
            'mobile': ['smartphone', 'mobile phone', 'android phone', 'iphone', 'budget mobile', 'premium mobile', '5g mobile'],
            'shoes': ['running shoes', 'casual shoes', 'formal shoes', 'sports shoes', 'sneakers', 'boots', 'nike shoes'],
            'shirt': ['formal shirt', 'casual shirt', 'polo shirt', 't-shirt', 'dress shirt', 'business shirt'],
            'headphones': ['wireless headphones', 'bluetooth headphones', 'noise cancelling headphones', 'gaming headphones', 'sports headphones'],
            'camera': ['dslr camera', 'mirrorless camera', 'action camera', 'point and shoot', 'canon camera', 'nikon camera'],
            'watch': ['smartwatch', 'digital watch', 'analog watch', 'fitness watch', 'luxury watch', 'apple watch'],
            'bag': ['laptop bag', 'handbag', 'backpack', 'travel bag', 'messenger bag', 'school bag'],
            'jersey': ['cricket jersey', 'football jersey', 'ipl jersey', 'team jersey', 'sports jersey', 'mumbai indians jersey'],
            'formal': ['formal shirt', 'formal shoes', 'formal dress', 'formal pants', 'formal wear', 'business formal'],
            'gaming': ['gaming laptop', 'gaming mouse', 'gaming keyboard', 'gaming headset', 'gaming chair', 'gaming monitor'],
            'wireless': ['wireless headphones', 'wireless earbuds', 'wireless keyboard', 'wireless mouse', 'wireless speaker'],
            'budget': ['budget phone', 'budget laptop', 'budget headphones', 'budget camera', 'budget watch', 'budget mobile'],
            'premium': ['premium phone', 'premium laptop', 'premium headphones', 'premium camera', 'premium watch', 'premium mobile']
        }
        
        # Match query to patterns
        for pattern, completions in product_patterns.items():
            if pattern in query:
                suggestions.extend(completions)
        
        return suggestions

    def _get_contextual_completions(self, query: str) -> List[str]:
        """Get contextual completions based on query context."""
        suggestions = []
        
        # Contextual patterns based on query analysis
        contextual_patterns = {
            'student': ['student laptop', 'student bag', 'student headphones', 'student watch', 'student essentials'],
            'business': ['business laptop', 'business shirt', 'business bag', 'business watch', 'business formal'],
            'gaming': ['gaming laptop', 'gaming mouse', 'gaming keyboard', 'gaming headset', 'gaming setup'],
            'fitness': ['fitness watch', 'fitness tracker', 'sports shoes', 'fitness headphones', 'workout gear'],
            'travel': ['travel bag', 'travel camera', 'travel adapter', 'travel pillow', 'travel essentials'],
            'office': ['office chair', 'office desk', 'office laptop', 'office headphones', 'office supplies'],
            'home': ['home speaker', 'home camera', 'home laptop', 'home tv', 'home decor'],
            'outdoor': ['outdoor camera', 'outdoor watch', 'outdoor shoes', 'outdoor bag', 'outdoor gear'],
            'professional': ['professional laptop', 'professional camera', 'professional headphones', 'professional attire'],
            'casual': ['casual shoes', 'casual shirt', 'casual bag', 'casual wear', 'casual style']
        }
        
        # Match query to contextual patterns
        for context, completions in contextual_patterns.items():
            if context in query:
                suggestions.extend(completions)
        
        return suggestions

    def _get_brand_completions(self, query: str) -> List[str]:
        """Get brand-specific completions."""
        suggestions = []
        
        # Brand patterns
        brand_patterns = {
            'samsung': ['samsung galaxy', 'samsung mobile', 'samsung phone', 'samsung tablet', 'samsung tv', 'samsung galaxy s23'],
            'apple': ['iphone', 'ipad', 'macbook', 'apple watch', 'airpods', 'iphone 15', 'macbook air'],
            'nike': ['nike shoes', 'nike sneakers', 'nike running', 'nike sports', 'nike air max', 'nike air force'],
            'adidas': ['adidas shoes', 'adidas sneakers', 'adidas running', 'adidas sports', 'adidas ultraboost'],
            'xiaomi': ['xiaomi phone', 'xiaomi mobile', 'xiaomi redmi', 'xiaomi mi', 'xiaomi pocophone', 'xiaomi redmi note'],
            'oneplus': ['oneplus phone', 'oneplus mobile', 'oneplus nord', 'oneplus 11', 'oneplus nord ce'],
            'dell': ['dell laptop', 'dell inspiron', 'dell xps', 'dell latitude', 'dell precision'],
            'hp': ['hp laptop', 'hp pavilion', 'hp envy', 'hp spectre', 'hp omen'],
            'canon': ['canon camera', 'canon dslr', 'canon mirrorless', 'canon lens', 'canon eos'],
            'nikon': ['nikon camera', 'nikon dslr', 'nikon mirrorless', 'nikon lens', 'nikon z'],
            'sony': ['sony tv', 'sony headphones', 'sony camera', 'sony playstation', 'sony wh'],
            'lg': ['lg tv', 'lg refrigerator', 'lg washing machine', 'lg oled', 'lg gram'],
            'lenovo': ['lenovo laptop', 'lenovo thinkpad', 'lenovo tablet', 'lenovo ideapad', 'lenovo yoga'],
            'asus': ['asus laptop', 'asus gaming', 'asus rog', 'asus tuf', 'asus zenbook']
        }
        
        # Match query to brand patterns
        for brand, completions in brand_patterns.items():
            if brand in query:
                suggestions.extend(completions)
        
        return suggestions

    def _get_category_completions(self, query: str) -> List[str]:
        """Get category-specific completions."""
        suggestions = []
        
        # Category patterns
        category_patterns = {
            'electronics': ['smartphone', 'laptop', 'tablet', 'headphones', 'camera', 'tv', 'speaker'],
            'fashion': ['shoes', 'shirt', 'jeans', 'dress', 'bag', 'watch', 'jewelry'],
            'home': ['furniture', 'kitchen', 'decor', 'appliances', 'lighting', 'storage'],
            'sports': ['sports shoes', 'sports jersey', 'fitness tracker', 'sports bag', 'sports equipment'],
            'beauty': ['cosmetics', 'skincare', 'makeup', 'perfume', 'hair care', 'beauty tools'],
            'books': ['fiction', 'non-fiction', 'academic', 'children books', 'comics', 'magazines'],
            'toys': ['educational toys', 'board games', 'outdoor toys', 'building blocks', 'dolls'],
            'automotive': ['car accessories', 'bike accessories', 'car care', 'bike care', 'safety gear']
        }
        
        # Match query to category patterns
        for category, completions in category_patterns.items():
            if category in query:
                suggestions.extend(completions)
        
        return suggestions

    def _get_feature_completions(self, query: str) -> List[str]:
        """Get feature-specific completions."""
        suggestions = []
        
        # Feature patterns
        feature_patterns = {
            'wireless': ['wireless headphones', 'wireless earbuds', 'wireless keyboard', 'wireless mouse', 'wireless speaker'],
            'bluetooth': ['bluetooth headphones', 'bluetooth speaker', 'bluetooth keyboard', 'bluetooth mouse'],
            'gaming': ['gaming laptop', 'gaming mouse', 'gaming keyboard', 'gaming headset', 'gaming chair'],
            'waterproof': ['waterproof phone', 'waterproof watch', 'waterproof camera', 'waterproof bag'],
            'fast': ['fast charger', 'fast laptop', 'fast phone', 'fast delivery', 'fast processor'],
            'lightweight': ['lightweight laptop', 'lightweight headphones', 'lightweight bag', 'lightweight camera'],
            'portable': ['portable speaker', 'portable charger', 'portable camera', 'portable laptop'],
            'smart': ['smartphone', 'smartwatch', 'smart tv', 'smart speaker', 'smart home'],
            'noise': ['noise cancelling headphones', 'noise reduction', 'noise isolation'],
            'touch': ['touch screen', 'touch laptop', 'touch phone', 'touch tablet']
        }
        
        # Match query to feature patterns
        for feature, completions in feature_patterns.items():
            if feature in query:
                suggestions.extend(completions)
        
        return suggestions

    def _get_use_case_completions(self, query: str) -> List[str]:
        """Get use-case specific completions."""
        suggestions = []
        
        # Use case patterns
        use_case_patterns = {
            'work': ['work laptop', 'work bag', 'work shoes', 'work headphones', 'office supplies'],
            'study': ['study laptop', 'study table', 'study chair', 'study lamp', 'study materials'],
            'travel': ['travel bag', 'travel camera', 'travel adapter', 'travel pillow', 'travel essentials'],
            'gym': ['gym shoes', 'gym bag', 'fitness tracker', 'sports headphones', 'workout clothes'],
            'party': ['party dress', 'party shoes', 'party bag', 'party accessories', 'party makeup'],
            'wedding': ['wedding dress', 'wedding shoes', 'wedding jewelry', 'wedding accessories'],
            'birthday': ['birthday gift', 'birthday cake', 'birthday decorations', 'birthday party'],
            'christmas': ['christmas gift', 'christmas tree', 'christmas decorations', 'christmas lights']
        }
        
        # Match query to use case patterns
        for use_case, completions in use_case_patterns.items():
            if use_case in query:
                suggestions.extend(completions)
        
        return suggestions

    def _get_dynamic_completions(self, query: str) -> List[str]:
        """Get dynamic completions based on query analysis."""
        suggestions = []
        query_words = query.split()
        
        # Generate word combinations
        if len(query_words) >= 2:
            for i, word1 in enumerate(query_words):
                for j, word2 in enumerate(query_words[i+1:], i+1):
                    if len(word1) > 2 and len(word2) > 2:
                        combination = f"{word1} {word2}"
                        suggestions.append(combination)
        
        # Add common modifiers
        modifiers = ['best', 'top', 'latest', 'new', 'popular', 'trending', 'cheap', 'expensive', 'premium', 'budget']
        for modifier in modifiers:
            if modifier not in query:
                suggestions.append(f"{modifier} {query}")
        
        # Add common suffixes
        suffixes = ['online', 'near me', 'delivery', 'pickup', 'discount', 'offer', 'sale']
        for suffix in suffixes:
            suggestions.append(f"{query} {suffix}")
        
        return suggestions
    
    def _bert_completion(self, query: str) -> List[str]:
        """Generate BERT-based completions for short queries."""
        suggestions = []
        
        # Only use BERT for very short queries to avoid generic completions
        if len(query) <= 4:
            try:
                # Create masked input for short queries
                masked_input = f"{query} [MASK]"
                inputs = self.tokenizer(masked_input, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = outputs.logits[0, -1, :]
                    top_k = torch.topk(predictions, 10)
                
                # Get top predictions and filter for relevance
                for token_id in top_k.indices:
                    token = self.tokenizer.decode([token_id])
                    if token.strip() and len(token.strip()) > 1:
                        completion = f"{query} {token.strip()}"
                        # Only add if it makes sense (not punctuation or generic words)
                        if not any(char in token for char in '.,;:!?') and token.strip() not in ['the', 'a', 'an', 'and', 'or']:
                            suggestions.append(completion)
                
            except Exception as e:
                print(f"BERT completion error: {e}")
        
        return suggestions[:5]  # Limit BERT suggestions

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