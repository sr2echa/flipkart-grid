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
        self.product_catalog = None
        self.realtime_product_info = None
        
    def build_system(self, data: Dict):
        """Build the complete autosuggest system."""
        print("Building integrated autosuggest system...")
        
        self.user_queries = data['user_queries']
        self.session_log = data['session_log']
        self.product_catalog = data['product_catalog']
        self.realtime_product_info = data['realtime_product_info']
        
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
    
    def _extract_product_specific_suggestions(self, query: str) -> List[Tuple[str, float]]:
        """Extract product-specific suggestions based on catalog and realtime info with semantic understanding."""
        suggestions = []
        query_lower = query.lower()
        
        # Get all products that match the query semantically
        matching_products = self._get_semantic_matches(query)
        
        if len(matching_products) == 0:
            return [] 
        
        # Generate contextual suggestions based on query type
        contextual_suggestions = self._generate_contextual_suggestions(query_lower, matching_products)
        suggestions.extend(contextual_suggestions)
        
        # Generate specific product suggestions
        specific_suggestions = self._generate_specific_suggestions(query_lower, matching_products)
        suggestions.extend(specific_suggestions)
        
        # Generate brand-specific suggestions
        brand_suggestions = self._generate_brand_suggestions(query_lower, matching_products)
        suggestions.extend(brand_suggestions)
        
        # Generate price-based suggestions
        price_suggestions = self._generate_price_suggestions(query_lower, matching_products)
        suggestions.extend(price_suggestions)
        
        # Generate feature-based suggestions
        feature_suggestions = self._generate_feature_suggestions(query_lower, matching_products)
        suggestions.extend(feature_suggestions)
        
        # Generate use-case suggestions
        use_case_suggestions = self._generate_use_case_suggestions(query_lower, matching_products)
        suggestions.extend(use_case_suggestions)
        
        # Generate combination suggestions
        combination_suggestions = self._generate_combination_suggestions(query_lower, matching_products)
        suggestions.extend(combination_suggestions)
        
        return suggestions

    def _generate_contextual_suggestions(self, query: str, products: pd.DataFrame) -> List[Tuple[str, float]]:
        """Generate contextual suggestions based on query and product context."""
        suggestions = []
        query_words = query.split()
        
        # Category-based suggestions
        categories = self._extract_categories_from_products(products)
        for category in categories:
            if any(word in category.lower() for word in query_words):
                suggestions.extend(self._get_category_suggestions(category, products))
        
        # Brand-based suggestions
        brands = self._extract_brands(products)
        for brand in brands:
            if brand.lower() in query or any(word in brand.lower() for word in query.split()):
                suggestions.extend(self._get_brand_suggestions(brand, products))
        
        # Price-based suggestions
        price_ranges = self._extract_price_ranges(products)
        suggestions.extend(self._get_price_suggestions(query, price_ranges))
        
        # Feature-based suggestions
        features = self._extract_features_from_products(products)
        suggestions.extend(self._get_feature_suggestions(query, features))
        
        return suggestions

    def _generate_specific_suggestions(self, query: str, products: pd.DataFrame) -> List[Tuple[str, float]]:
        """Generate specific product suggestions."""
        suggestions = []
        
        # Extract common product patterns
        product_patterns = {
            'laptop': ['gaming laptop', 'business laptop', 'student laptop', 'ultrabook', 'macbook', 'dell laptop', 'hp laptop'],
            'phone': ['smartphone', 'mobile phone', 'android phone', 'iphone', 'budget phone', 'premium phone'],
            'shoes': ['running shoes', 'casual shoes', 'formal shoes', 'sports shoes', 'sneakers', 'boots'],
            'shirt': ['formal shirt', 'casual shirt', 'polo shirt', 't-shirt', 'dress shirt'],
            'headphones': ['wireless headphones', 'bluetooth headphones', 'noise cancelling headphones', 'gaming headphones'],
            'camera': ['dslr camera', 'mirrorless camera', 'action camera', 'point and shoot', 'canon camera', 'nikon camera'],
            'watch': ['smartwatch', 'digital watch', 'analog watch', 'fitness watch', 'luxury watch'],
            'bag': ['laptop bag', 'handbag', 'backpack', 'travel bag', 'messenger bag'],
            'jersey': ['cricket jersey', 'football jersey', 'ipl jersey', 'team jersey', 'sports jersey'],
            'formal': ['formal shirt', 'formal shoes', 'formal dress', 'formal pants', 'formal wear'],
            'gaming': ['gaming laptop', 'gaming mouse', 'gaming keyboard', 'gaming headset'],
            'wireless': ['wireless headphones', 'wireless earbuds', 'wireless keyboard', 'wireless mouse'],
            'budget': ['budget phone', 'budget laptop', 'budget headphones', 'budget camera', 'budget watch'],
            'premium': ['premium phone', 'premium laptop', 'premium headphones', 'premium camera', 'premium watch']
        }
        
        # Match query to patterns
        for pattern, suggestions_list in product_patterns.items():
            if pattern in query:
                for suggestion in suggestions_list:
                    suggestions.append((suggestion, 0.85))
        
        # Brand-specific patterns
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
        
        for brand, brand_suggestions in brand_patterns.items():
            if brand in query:
                for suggestion in brand_suggestions:
                    suggestions.append((suggestion, 0.90))
        
        return suggestions

    def _generate_brand_suggestions(self, query: str, products: pd.DataFrame) -> List[Tuple[str, float]]:
        """Generate brand-specific suggestions."""
        suggestions = []
        
        # Get unique brands from products
        brands = products['brand'].dropna().unique()
        
        for brand in brands:
            brand_lower = brand.lower()
            if brand_lower in query or any(word in brand_lower for word in query.split()):
                # Generate brand-specific suggestions
                brand_products = products[products['brand'] == brand]
                
                # Get categories for this brand
                categories = brand_products['category'].dropna().unique()
                for category in categories:
                    suggestions.append((f"{brand} {category.lower()}", 0.85))
                
                # Get specific product types
                titles = brand_products['title'].dropna()
                for title in titles[:5]:  # Limit to 5 suggestions
                    # Extract key words from title
                    words = title.lower().split()
                    if len(words) >= 2:
                        key_words = [w for w in words if len(w) > 3 and w not in ['the', 'and', 'for', 'with']]
                        if key_words:
                            suggestion = f"{brand} {' '.join(key_words[:2])}"
                            suggestions.append((suggestion, 0.80))
        
        return suggestions

    def _generate_price_suggestions(self, query: str, products: pd.DataFrame) -> List[Tuple[str, float]]:
        """Generate price-based suggestions."""
        suggestions = []
        
        # Extract price ranges
        prices = products['price'].dropna()
        if len(prices) == 0:
            return suggestions
        
        # Price-based patterns
        price_patterns = {
            'under 10000': ['budget phone', 'budget laptop', 'budget headphones'],
            'under 20000': ['mid-range phone', 'student laptop', 'wireless headphones'],
            'under 50000': ['premium phone', 'business laptop', 'gaming laptop'],
            'above 50000': ['luxury phone', 'premium laptop', 'professional camera']
        }
        
        # Check if query mentions price
        price_keywords = ['cheap', 'budget', 'expensive', 'premium', 'luxury', 'affordable']
        for keyword in price_keywords:
            if keyword in query:
                if keyword in ['cheap', 'budget', 'affordable']:
                    suggestions.extend([(s, 0.75) for s in price_patterns['under 10000']])
                elif keyword in ['premium', 'luxury']:
                    suggestions.extend([(s, 0.85) for s in price_patterns['above 50000']])
        
        return suggestions

    def _generate_feature_suggestions(self, query: str, products: pd.DataFrame) -> List[Tuple[str, float]]:
        """Generate feature-based suggestions."""
        suggestions = []
        
        # Feature keywords
        feature_keywords = {
            'wireless': ['wireless headphones', 'wireless earbuds', 'wireless keyboard', 'wireless mouse'],
            'bluetooth': ['bluetooth headphones', 'bluetooth speaker', 'bluetooth keyboard'],
            'gaming': ['gaming laptop', 'gaming mouse', 'gaming keyboard', 'gaming headset'],
            'waterproof': ['waterproof phone', 'waterproof watch', 'waterproof camera'],
            'fast': ['fast charger', 'fast laptop', 'fast phone', 'fast delivery'],
            'lightweight': ['lightweight laptop', 'lightweight headphones', 'lightweight bag'],
            'portable': ['portable speaker', 'portable charger', 'portable camera'],
            'smart': ['smartphone', 'smartwatch', 'smart tv', 'smart speaker']
        }
        
        for feature, feature_suggestions in feature_keywords.items():
            if feature in query:
                suggestions.extend([(s, 0.80) for s in feature_suggestions])
        
        return suggestions

    def _generate_use_case_suggestions(self, query: str, products: pd.DataFrame) -> List[Tuple[str, float]]:
        """Generate use-case based suggestions."""
        suggestions = []
        
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
        
        for use_case, use_case_suggestions in use_case_patterns.items():
            if use_case in query:
                suggestions.extend([(s, 0.75) for s in use_case_suggestions])
        
        return suggestions

    def _generate_combination_suggestions(self, query: str, products: pd.DataFrame) -> List[Tuple[str, float]]:
        """Generate combination suggestions."""
        suggestions = []
        
        # Extract brands and categories
        brands = products['brand'].dropna().unique()
        categories = products['category'].dropna().unique()
        
        # Generate brand + category combinations
        for brand in brands[:5]:  # Limit to top 5 brands
            for category in categories[:3]:  # Limit to top 3 categories
                suggestion = f"{brand} {category.lower()}"
                suggestions.append((suggestion, 0.70))
        
        # Generate feature + product combinations
        features = ['wireless', 'gaming', 'premium', 'budget', 'smart']
        products_list = ['laptop', 'phone', 'headphones', 'camera', 'watch']
        
        for feature in features:
            for product in products_list:
                if feature in query or product in query:
                    suggestion = f"{feature} {product}"
                    suggestions.append((suggestion, 0.75))
        
        return suggestions

    def _extract_categories(self, product: pd.Series) -> List[str]:
        """Extract categories from a product row, including major and subcategories."""
        categories = []
        if 'major_category' in product and pd.notna(product['major_category']):
            categories.append(product['major_category'])
        if 'category' in product and pd.notna(product['category']):
            # Split full category tree
            full_category = str(product['category'])
            # Remove ["", " and "] and split by >>
            # Example: ["", "Electronics", "Mobiles & Accessories", "Smartphones"]
            # To: Electronics >> Mobiles & Accessories >> Smartphones
            try:
                cleaned_category = full_category.replace('["", ', 1).replace('"]", ', 1).replace('", "', ' >> ')
            except (TypeError, AttributeError):
                # If replace fails, try a simpler approach
                cleaned_category = str(full_category).replace('["', '').replace('"]', '').replace('", "', ' >> ')
            parts = [p.strip() for p in cleaned_category.split(' >> ') if p.strip()]
            categories.extend(parts)
        
        # Add subcategory if available (check for column existence first)
        if 'subcategory' in product.index and pd.notna(product['subcategory']):
            try:
                categories.append(str(product['subcategory']))
            except:
                pass  # Skip if there's an error with subcategory
        
        return list(set(categories)) # Return unique categories

    def _extract_brands(self, products: pd.DataFrame) -> List[str]:
        """Extract unique brands from products."""
        if len(products) == 0:
            return []
        
        brands = products['brand'].unique().tolist()
        return [brand.lower() for brand in brands if pd.notna(brand)]
    
    def _extract_price_ranges(self, products: pd.DataFrame) -> Dict:
        """Extract price range information."""
        if len(products) == 0:
            return {}
        
        # Get realtime pricing for these products
        product_ids = products['product_id'].tolist()
        pricing = self.realtime_product_info[
            self.realtime_product_info['product_id'].isin(product_ids)
        ]
        
        if len(pricing) == 0:
            return {}
        
        prices = pricing['current_price'].dropna()
        
        return {
            'min_price': prices.min(),
            'max_price': prices.max(),
            'avg_price': prices.mean(),
            'median_price': prices.median(),
            'budget_count': len(prices[prices < 10000]),
            'mid_range_count': len(prices[(prices >= 10000) & (prices < 50000)]),
            'premium_count': len(prices[(prices >= 50000) & (prices < 100000)]),
            'luxury_count': len(prices[prices >= 100000])
        }
    
    def _extract_product_features(self, query: str, candidate_query: str,
                                 initial_score: float, data: Dict, product: pd.Series = None) -> List[float]:
        """Extract features for reranking, leveraging product catalog data.
        If product Series is provided, it extracts product-specific features.
        Otherwise, it falls back to query-candidate features.
        """
        features = []
        
        # 1. Initial Score (from Trie, Semantic, BERT, or Product-specific generation)
        features.append(initial_score)

        if product is not None and not product.empty:
            # Product-specific features
            # 2. Product Rating
            features.append(product['rating'] if pd.notna(product['rating']) else 0.0)

            # 3. Number of Reviews
            features.append(product['num_reviews'] if pd.notna(product['num_reviews']) else 0)

            # 4. Price (normalized)
            max_price = 150000.0 # Example max price, should be dynamic or config
            normalized_price = (product['price'] / max_price) if pd.notna(product['price']) else 0.0
            features.append(normalized_price)

            # 5. Length of Product Title vs. Query Length
            features.append(abs(len(query) - len(str(product['title']))))

            # 6. Query-Product Text Overlap (using combined_text)
            query_words = set(query.lower().split())
            product_text_words = set(str(product['combined_text']).lower().split())
            overlap = len(query_words.intersection(product_text_words)) / (len(query_words) + 1e-6)
            features.append(overlap)

            # 7. Brand Match
            brand_match = 1.0 if query.lower() == str(product['brand']).lower() else 0.0
            features.append(brand_match)

            # 8. Category Match
            category_match = 0.0
            query_categories = set(query.lower().split())
            product_categories = set([c.lower() for c in self._extract_categories(product)])
            if query_categories.intersection(product_categories):
                category_match = 1.0
            features.append(category_match)
            
            # 9. Has Specifications
            features.append(1.0 if pd.notna(product['specifications']) and str(product['specifications']) != '' else 0.0)

            # 10. Presence of 'new' or 'latest' in query for electronics
            is_electronics = any(cat in product_categories for cat in ['electronics', 'mobile', 'laptop', 'television', 'camera'])
            new_or_latest = 1.0 if is_electronics and ('new' in query_words or 'latest' in query_words) else 0.0
            features.append(new_or_latest)

            # 11. Contextual features (location, event) - from realtime_product_info
            location_relevance = 0.0
            event_relevance = 0.0

            if 'realtime_product_info' in data and not data['realtime_product_info'].empty:
                realtime_info = data['realtime_product_info']
                product_realtime = realtime_info[realtime_info['product_id'] == product['product_id']]
                
                if not product_realtime.empty:
                    # Location relevance
                    user_location = data.get('location', '').lower()
                    if user_location and 'location' in product_realtime.columns:
                        if user_location == str(product_realtime['location'].iloc[0]).lower():
                            location_relevance = 1.0
                    
                    # Event relevance (placeholder - this would be more complex with actual event data)
                    user_event = data.get('event', '').lower()
                    if user_event and 'event_keywords' in product_realtime.columns:
                        # Assuming event_keywords is a comma-separated string
                        event_keywords = [k.strip().lower() for k in str(product_realtime['event_keywords'].iloc[0]).split(',')]
                        if user_event in event_keywords:
                            event_relevance = 1.0

            features.append(location_relevance)
            features.append(event_relevance)
            
            # 12. Session History Relevance (placeholder for now)
            session_history_relevance = 0.0
            # This would involve comparing query/product with recent user session history
            features.append(session_history_relevance)
            
            # 13. Query type (e.g., brand query, category query, specific model query)
            # This can be derived from NER or simple heuristics
            query_type_brand = 1.0 if query.lower() == str(product['brand']).lower() else 0.0
            features.append(query_type_brand)
            
            # 14. Subcategory presence in query or product category (new feature)
            subcategory_match_feature = 0.0
            if 'subcategory' in product.index:
                try:
                    query_subcategories = set(query.lower().split())
                    product_subcategory = str(product['subcategory']).lower()
                    if product_subcategory and any(sub_q in product_subcategory for sub_q in query_subcategories):
                        subcategory_match_feature = 1.0
                except:
                    pass  # Skip if there's an error with subcategory
            features.append(subcategory_match_feature)

            # 15. Tags match (new feature)
            tags_match_feature = 0.0
            if 'tags' in product.index:
                query_words_lower = set(query.lower().split())
                product_tags = str(product['tags']).lower().split(',') if pd.notna(product['tags']) else []
                product_tags_set = set([tag.strip() for tag in product_tags])
                if query_words_lower.intersection(product_tags_set):
                    tags_match_feature = 1.0
            features.append(tags_match_feature)
            
        else:
            # Fallback for general query-candidate features if no product is provided
            # These are the original features that were always extracted
            # 2. Text similarity (TF-IDF)
            try:
                query_vec = self.vectorizer.transform([query])
                candidate_vec = self.vectorizer.transform([candidate_query])
                similarity = cosine_similarity(query_vec, candidate_vec)[0][0]
                features.append(similarity)
            except:
                features.append(0.0)
            
            # 3. Length difference
            features.append(abs(len(query) - len(candidate_query)))
            
            # 4. Word overlap
            original_words = set(query.lower().split())
            candidate_words = set(candidate_query.lower().split())
            overlap = len(original_words.intersection(candidate_words)) / (len(original_words) + 1e-6)
            features.append(overlap)
            
            # Pad with zeros if less than 15 features to match expected input size for the model
            while len(features) < 15:
                features.append(0.0)

        return features
    
    def _get_category_suggestions(self, category: str, products: pd.DataFrame) -> List[Tuple[str, float]]:
        """Get category-specific suggestions."""
        suggestions = []
        category_lower = category.lower()
        
        # Category-specific patterns
        category_patterns = {
            'laptop': ['gaming laptop', 'business laptop', 'student laptop', 'budget laptop', 'premium laptop'],
            'mobile': ['smartphone', 'budget mobile', 'premium phone', '5g mobile', 'android phone'],
            'phone': ['smartphone', 'budget phone', 'premium phone', '5g phone', 'android phone'],
            'shoes': ['running shoes', 'casual shoes', 'formal shoes', 'sports shoes', 'sneakers'],
            'headphones': ['wireless headphones', 'bluetooth headphones', 'noise cancelling', 'gaming headset'],
            'camera': ['dslr camera', 'mirrorless camera', 'action camera', 'point and shoot'],
            'watch': ['smartwatch', 'digital watch', 'analog watch', 'fitness watch'],
            'shirt': ['formal shirt', 'casual shirt', 'polo shirt', 't-shirt', 'dress shirt']
        }
        
        for pattern, pattern_suggestions in category_patterns.items():
            if pattern in category_lower:
                suggestions.extend([(s, 0.80) for s in pattern_suggestions])
        
        return suggestions

    def _get_brand_suggestions(self, brand: str, products: pd.DataFrame) -> List[Tuple[str, float]]:
        """Get brand-specific suggestions."""
        suggestions = []
        
        # Get products from this brand
        brand_products = products[products['brand'].str.lower() == brand.lower()]
        
        if len(brand_products) == 0:
            return suggestions
        
        # Get categories for this brand
        categories = brand_products['category'].dropna().unique()
        
        # Generate brand + category combinations
        for category in categories:
            if pd.notna(category):
                suggestions.append((f"{brand} {category.lower()}", 0.85))
        
        return suggestions

    def _extract_price_ranges(self, products: pd.DataFrame) -> Dict[str, float]:
        """Extract price ranges from products."""
        price_ranges = {}
        if 'price' in products.columns:
            prices = products['price'].dropna()
            if len(prices) > 0:
                price_ranges['min'] = prices.min()
                price_ranges['max'] = prices.max()
                price_ranges['avg'] = prices.mean()
        return price_ranges

    def _get_price_suggestions(self, query: str, price_ranges: Dict[str, float]) -> List[Tuple[str, float]]:
        """Get price-based suggestions."""
        suggestions = []
        
        # Price-based patterns
        price_patterns = {
            'cheap': ['budget phone', 'budget laptop', 'affordable mobile'],
            'budget': ['budget phone', 'budget laptop', 'budget headphones'],
            'premium': ['premium phone', 'premium laptop', 'luxury watch'],
            'expensive': ['premium phone', 'premium laptop', 'luxury camera']
        }
        
        for keyword, keyword_suggestions in price_patterns.items():
            if keyword in query:
                suggestions.extend([(s, 0.75) for s in keyword_suggestions])
        
        return suggestions

    def _get_feature_suggestions(self, query: str, features: List[str]) -> List[Tuple[str, float]]:
        """Get feature-based suggestions."""
        suggestions = []
        
        for feature in features:
            if feature in query:
                if feature == 'wireless':
                    suggestions.extend([
                        ('wireless headphones', 0.80),
                        ('wireless earbuds', 0.80),
                        ('wireless keyboard', 0.75)
                    ])
                elif feature == 'gaming':
                    suggestions.extend([
                        ('gaming laptop', 0.85),
                        ('gaming mouse', 0.80),
                        ('gaming keyboard', 0.80)
                    ])
                elif feature == 'bluetooth':
                    suggestions.extend([
                        ('bluetooth headphones', 0.80),
                        ('bluetooth speaker', 0.75),
                        ('bluetooth keyboard', 0.75)
                    ])
        
        return suggestions

    def _get_semantic_matches(self, query: str) -> pd.DataFrame:
        """Get products that semantically match the query."""
        query_lower = query.lower()
        
        # Multiple matching strategies
        matches = []
        
        # 1. Direct title match
        title_matches = self.product_catalog[
            self.product_catalog['title'].str.contains(query_lower, case=False, na=False)
        ]
        matches.append(title_matches)
        
        # 2. Category match
        category_matches = self.product_catalog[
            self.product_catalog['category'].str.contains(query_lower, case=False, na=False)
        ]
        matches.append(category_matches)
        
        # 3. Brand match
        try:
            brand_matches = self.product_catalog[
                self.product_catalog['brand'].astype(str).str.contains(query_lower, case=False, na=False)
            ]
            matches.append(brand_matches)
        except:
            pass  # Skip if there's an error with brand matching
        
        # 4. Combined text match (title + description + specs)
        combined_matches = self.product_catalog[
            self.product_catalog['combined_text'].str.contains(query_lower, case=False, na=False)
        ]
        matches.append(combined_matches)
        
        # Combine all matches and remove duplicates
        if matches:
            all_matches = pd.concat(matches, ignore_index=True)
            return all_matches.drop_duplicates(subset=['product_id'])
        
        return pd.DataFrame()

    def _extract_brands(self, products: pd.DataFrame) -> List[str]:
        """Extract brands from products."""
        brands = []
        if 'brand' in products.columns:
            brands = products['brand'].dropna().unique().tolist()
        return brands
    
    def _build_reranker(self, data: Dict):
        """Build XGBoost reranker for final ranking."""
        # Prepare training data for reranker
        training_data = self._prepare_reranker_data(data)
        
        # Initialize XGBoost model with proper objective for ranking
        self.reranker = xgb.XGBRegressor(
            objective='reg:logistic',  # Changed to logistic for 0-1 range
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
            labels_list.append(1.0)  # High relevance for exact match
            
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
                         initial_score: float, data: Dict) -> List[float]:
        """Extract features for reranking, leveraging product catalog data."""
        features = []
        
        # 1. Initial Score (from Trie, Semantic, BERT, or Product-specific generation)
        features.append(initial_score)

        # 2. Text similarity (TF-IDF)
        try:
            query_vec = self.vectorizer.transform([original_query])
            candidate_vec = self.vectorizer.transform([candidate_query])
            similarity = cosine_similarity(query_vec, candidate_vec)[0][0]
            features.append(similarity)
        except:
            features.append(0.0)

        # 3. Length difference
        features.append(abs(len(original_query) - len(candidate_query)) / (len(original_query) + 1e-6))

        # 4. Word overlap (normalized)
        original_words = set(original_query.lower().split())
        candidate_words = set(candidate_query.lower().split())
        overlap = len(original_words.intersection(candidate_words)) / (len(original_words) + 1e-6)
        features.append(overlap)

        # 5. Prefix match (binary)
        features.append(1.0 if candidate_query.lower().startswith(original_query.lower()) else 0.0)

        # 6. Exact match (binary)
        features.append(1.0 if candidate_query.lower() == original_query.lower() else 0.0)

        # 7. Edit distance (normalized)
        try:
            import difflib
            edit_sim = difflib.SequenceMatcher(None, original_query.lower(), candidate_query.lower()).ratio()
            features.append(edit_sim)
        except:
            features.append(0.0)

        # Features based on Product Catalog match for candidate_query
        product_match = self.product_catalog[
            self.product_catalog['title'].str.lower() == candidate_query.lower()
        ]
        if not product_match.empty:
            product_info = product_match.iloc[0]
            features.append(1.0)  # is_product_title_match
            features.append(float(product_info['rating']) if pd.notna(product_info['rating']) else 0.0)
            features.append(float(product_info['num_reviews']) if pd.notna(product_info['num_reviews']) else 0.0)
            orig_query_brand = ''
            if 'brand' in data['user_queries'].columns:
                brand_matches = data['user_queries'][
                    data['user_queries']['corrected_query'] == original_query
                ]
                if not brand_matches.empty:
                    orig_query_brand = brand_matches['brand'].iloc[0]
            orig_query_category = ''
            if 'category' in data['user_queries'].columns:
                category_matches = data['user_queries'][
                    data['user_queries']['corrected_query'] == original_query
                ]
                if not category_matches.empty:
                    orig_query_category = category_matches['category'].iloc[0]
            features.append(1.0 if (pd.notna(product_info['brand']) and \
                                    str(product_info['brand']).lower() == str(orig_query_brand).lower()) else 0.0) # is_brand_match_with_original
            features.append(1.0 if (pd.notna(product_info['category']) and \
                                    str(product_info['category']).lower() == str(orig_query_category).lower()) else 0.0) # is_category_match_with_original
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # 8. Category match with current suggestion (if the suggestion itself is a category)
        is_candidate_a_category = 1.0 if candidate_query.lower() in [c.lower() for c in data['major_categories']] else 0.0
        features.append(is_candidate_a_category)

        # 9. Brand match with current suggestion (if the suggestion itself is a brand)
        is_candidate_a_brand = 1.0 if candidate_query.lower() in [b.lower() for b in self.product_catalog['brand'].dropna().unique()] else 0.0
        features.append(is_candidate_a_brand)

        # 10. Contextual boosts (event/location)
        event_relevance = 0.0
        location_relevance = 0.0
        if 'event' in data:
            if data['event'] and data['event'].lower() in candidate_query.lower():
                event_relevance = 1.0
        if 'location' in data:
            if data['location'] and data['location'].lower() in candidate_query.lower():
                location_relevance = 1.0
        features.append(event_relevance)
        features.append(location_relevance)

        # 11. Frequency (normalized)
        freq = 0.0
        if 'user_queries' in data:
            freq_row = data['user_queries'][data['user_queries']['corrected_query'] == candidate_query]
            if not freq_row.empty:
                freq = float(freq_row['frequency'].iloc[0])
        max_freq = float(data['user_queries']['frequency'].max()) if 'user_queries' in data and not data['user_queries'].empty else 1.0
        features.append(freq / (max_freq + 1e-6))

        return features
    
    def get_suggestions(self, query: str, max_suggestions: int = 10, 
                       context: Dict = None) -> List[Tuple[str, float]]:
        """Main method to get autosuggestions based on query and context."""
        if context is None:
            context = {}

        all_suggestions = {}

        # Create a proper data dictionary for feature extraction
        data_dict = {
            'product_catalog': self.product_catalog,
            'user_queries': self.user_queries,
            'realtime_product_info': self.realtime_product_info,
            'session_log': self.session_log,
            'major_categories': self.product_catalog['major_category'].dropna().unique().tolist() if 'major_category' in self.product_catalog.columns else [],
            'locations': ['mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata', 'hyderabad', 'pune', 'ahmedabad', 'jaipur', 'lucknow'],
            'location': context.get('location', ''),
            'event': context.get('event', '')
        }
        
        # 1. Trie-based prefix matching (high precision for exact starts)
        trie_suggestions = self.trie_autosuggest.get_suggestions_with_scores(query, max_suggestions)
        for s, freq in trie_suggestions:
            if s and len(s.strip()) > 0:
                # Normalize frequency score
                max_freq = self.user_queries['frequency'].max() if self.user_queries['frequency'].max() > 0 else 1.0
                initial_score = min(freq / max_freq, 1.0)
                
                # Apply reranker if available
                if self.reranker is not None:
                    features = self._extract_features(query, s, initial_score, data_dict)
                    if features:
                        reranker_score = self.reranker.predict([features])[0]
                        normalized_score = 1 / (1 + np.exp(-reranker_score))
                        all_suggestions[s] = max(all_suggestions.get(s, 0), normalized_score)
                else:
                    all_suggestions[s] = max(all_suggestions.get(s, 0), initial_score)

        # 2. Semantic Correction (for typos and semantic variants)
        if len(query) >= 2:  # Try semantic correction for shorter queries too
            semantic_suggestions = self.semantic_correction.get_semantic_suggestions(query, top_k=max_suggestions * 2)
            for s, similarity in semantic_suggestions:
                if s and len(s.strip()) > 0:
                    # Apply reranker if available
                    if self.reranker is not None:
                        features = self._extract_features(query, s, similarity, data_dict)
                        if features:
                            reranker_score = self.reranker.predict([features])[0]
                            normalized_score = 1 / (1 + np.exp(-reranker_score))
                            all_suggestions[s] = max(all_suggestions.get(s, 0), normalized_score)
                    else:
                        all_suggestions[s] = max(all_suggestions.get(s, 0), similarity)

        # 3. Product-Specific Suggestions (rich, dynamic suggestions)
        product_specific_suggestions = self._extract_product_specific_suggestions(query)
        for s, score in product_specific_suggestions:
            if s and len(s.strip()) > 0:
                all_suggestions[s] = max(all_suggestions.get(s, 0), score)

        # 4. Context-Aware Completion (using BERT) - only for short queries
        if len(query) <= 6:
            bert_completions = self.bert_completion.complete_query(query, max_suggestions)
            for s in bert_completions:
                if s and len(s.strip()) > 0:
                    # For BERT completions, assign a moderate initial score
                    if self.reranker is not None:
                        features = self._extract_features(query, s, 0.5, data_dict)
                        if features:
                            reranker_score = self.reranker.predict([features])[0]
                            normalized_score = 1 / (1 + np.exp(-reranker_score))
                            all_suggestions[s] = max(all_suggestions.get(s, 0), normalized_score)
                    else:
                        all_suggestions[s] = max(all_suggestions.get(s, 0), 0.5)

        # Apply contextual boosts and final processing
        final_suggestions = []
        for suggestion, score in all_suggestions.items():
            # Filter out poor suggestions
            if self._is_poor_suggestion(suggestion):
                continue
                
            boosted_score = score
            
            # Apply contextual boosts
            if 'location' in context and context['location']:
                boosted_score += self._get_location_boost(suggestion, context['location'])
            if 'event' in context and context['event']:
                boosted_score += self._get_event_boost(suggestion, context['event'])
            if 'session_context' in context and context['session_context']:
                boosted_score += self._get_session_boost(suggestion, context['session_context'])
            
            # Ensure score is within 0-1 range
            boosted_score = max(0.0, min(1.0, boosted_score))
            
            final_suggestions.append((suggestion, boosted_score))

        # Sort by score and return top suggestions
        final_suggestions.sort(key=lambda x: x[1], reverse=True)
        return final_suggestions[:max_suggestions]
    
    def _is_poor_suggestion(self, suggestion: str) -> bool:
        """Check if a suggestion is poor quality and should be filtered out."""
        if not suggestion or len(suggestion.strip()) == 0:
            return True
        
        suggestion_lower = suggestion.lower().strip()
        
        # Filter out suggestions with punctuation or special characters
        if any(char in suggestion_lower for char in [';', '.', ',', '!', '?', ':', '"', "'", '##']):
            return True
        
        # Filter out very short suggestions (less than 2 characters)
        if len(suggestion_lower) < 2:
            return True
        
        # Filter out suggestions that are just single letters or numbers
        if len(suggestion_lower) == 1 and suggestion_lower.isalnum():
            return True
        
        # Filter out suggestions that are just common words without context
        common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        if suggestion_lower in common_words:
            return True
        
        return False
    
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
        suggestion_lower = suggestion.lower()
        location_lower = location.lower()
        
        # Location-specific boost patterns (mainly for sports teams)
        location_patterns = {
            'mumbai': {
                'mumbai indians': 0.4, 'mumbai': 0.2, 'indians': 0.3
            },
            'delhi': {
                'delhi capitals': 0.4, 'delhi': 0.2, 'capitals': 0.3
            },
            'bangalore': {
                'rcb': 0.4, 'royal challengers': 0.4, 'bangalore': 0.2
            },
            'chennai': {
                'csk': 0.4, 'chennai super kings': 0.4, 'chennai': 0.2, 'super kings': 0.3
            },
            'kolkata': {
                'kkr': 0.4, 'kolkata knight riders': 0.4, 'kolkata': 0.2, 'knight riders': 0.3
            },
            'hyderabad': {
                'srh': 0.4, 'sunrisers hyderabad': 0.4, 'hyderabad': 0.2, 'sunrisers': 0.3
            },
            'pune': {
                'pune': 0.2, 'cricket': 0.2
            },
            'ahmedabad': {
                'gujarat titans': 0.4, 'ahmedabad': 0.2, 'titans': 0.3
            },
            'jaipur': {
                'rajasthan royals': 0.4, 'jaipur': 0.2, 'royals': 0.3
            },
            'lucknow': {
                'lucknow super giants': 0.4, 'lucknow': 0.2, 'super giants': 0.3
            }
        }
        
        if location_lower in location_patterns:
            patterns = location_patterns[location_lower]
            for pattern, boost in patterns.items():
                if pattern in suggestion_lower:
                    return boost
        
        return 0.0
    
    def _get_event_boost(self, suggestion: str, event: str) -> float:
        """Get event-based boost for suggestions."""
        suggestion_lower = suggestion.lower()
        event_lower = event.lower()
        
        # Event-specific boost patterns
        event_patterns = {
            'diwali': {
                'lights': 0.3, 'decoration': 0.3, 'sweets': 0.2, 'gifts': 0.2, 'clothes': 0.2,
                'diya': 0.3, 'rangoli': 0.3, 'candles': 0.3, 'lantern': 0.3
            },
            'holi': {
                'colors': 0.3, 'clothes': 0.2, 'sweets': 0.2, 'gifts': 0.2,
                'gulal': 0.3, 'pichkari': 0.3, 'water gun': 0.3
            },
            'christmas': {
                'tree': 0.3, 'decoration': 0.3, 'gifts': 0.2, 'lights': 0.3,
                'santa': 0.3, 'ornaments': 0.3, 'star': 0.3
            },
            'eid': {
                'clothes': 0.3, 'sweets': 0.2, 'gifts': 0.2, 'decoration': 0.2,
                'kurta': 0.3, 'sherwani': 0.3, 'sewai': 0.2
            },
            'ipl': {
                'jersey': 0.4, 'cricket': 0.3, 'team': 0.3, 'sports': 0.2,
                'mumbai indians': 0.4, 'csk': 0.4, 'rcb': 0.4, 'kkr': 0.4,
                'delhi capitals': 0.4, 'sunrisers': 0.4, 'gujarat titans': 0.4
            },
            'wedding': {
                'dress': 0.3, 'shoes': 0.2, 'jewelry': 0.3, 'gifts': 0.2, 'formal': 0.3,
                'saree': 0.3, 'lehenga': 0.3, 'suit': 0.3, 'sherwani': 0.3
            },
            'birthday': {
                'cake': 0.3, 'gifts': 0.2, 'decoration': 0.2, 'party': 0.2,
                'balloons': 0.3, 'candles': 0.3, 'wrapping': 0.2
            }
        }
        
        if event_lower in event_patterns:
            patterns = event_patterns[event_lower]
            for pattern, boost in patterns.items():
                if pattern in suggestion_lower:
                    return boost
        
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

    def _extract_categories_from_products(self, products: pd.DataFrame) -> List[str]:
        """Extract categories from products."""
        categories = []
        if 'category' in products.columns:
            categories = products['category'].dropna().unique().tolist()
        return categories

    def _extract_features_from_products(self, products: pd.DataFrame) -> List[str]:
        """Extract features from products."""
        features = []
        if 'specifications' in products.columns:
            specs = products['specifications'].dropna()
            for spec in specs:
                # Extract key features from specifications
                spec_lower = str(spec).lower()
                if 'wireless' in spec_lower:
                    features.append('wireless')
                if 'bluetooth' in spec_lower:
                    features.append('bluetooth')
                if 'gaming' in spec_lower:
                    features.append('gaming')
                if 'waterproof' in spec_lower:
                    features.append('waterproof')
        return list(set(features))

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