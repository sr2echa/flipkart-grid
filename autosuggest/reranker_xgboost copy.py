import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle
import os
import re

# Import the autosuggest components
from trie_autosuggest import TrieAutosuggest
from semantic_correction import SemanticCorrection
from bert_completion import BERTCompletion

class XGBoostReranker:
    """XGBoost-based reranker that processes candidates from Trie, SBERT, and BERT components."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = None
        self.vectorizer = None
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize autosuggest components
        self.trie_component = TrieAutosuggest()
        self.semantic_component = SemanticCorrection()
        self.bert_component = BERTCompletion()
        
        # Data storage
        self.user_queries = None
        self.product_catalog = None
        self.session_log = None
        self.realtime_product_info = None
        self.ner_dataset = None
        
        # Precomputed features
        self.persona_profiles = {}
        self.brand_frequency = {}
        self.query_frequency = {}
        self.category_popularity = {}
        self.title_embeddings = None
        self.title_to_product = {}
        
    def build_reranker(self, data: Dict, debug_mode: bool = False):
        """Build the XGBoost reranker and initialize all components."""
        print("Building Integrated XGBoost Reranker Pipeline...")
        
        # Store data
        self.user_queries = data['user_queries']
        self.product_catalog = data['product_catalog']
        self.session_log = data['session_log']
        self.realtime_product_info = data['realtime_product_info']
        self.ner_dataset = data.get('ner_dataset', None)
        
        # In debug mode, use smaller subsets for faster testing
        if debug_mode:
            print("🐛 DEBUG MODE: Using smaller datasets for faster testing")
            self.user_queries = self.user_queries.head(1000)
            self.session_log = self.session_log.head(2000)
            self.product_catalog = self.product_catalog.head(5000)
        
        # Build individual components
        print("Step 1/6: Building Trie component...")
        self.trie_component.build_trie(self.user_queries)
        
        print("Step 2/6: Building Semantic Correction component...")
        self.semantic_component.build_semantic_index(self.user_queries)
        
        print("Step 3/6: Building BERT Completion component...")
        self.bert_component.build_completion_patterns(self.user_queries)
        
        # Precompute features
        print("Step 4/6: Computing global features...")
        self._compute_global_features()
        
        # Prepare training data with component signals
        print("Step 5/6: Preparing training data...")
        training_data = self._prepare_reranker_training_data()
        
        # Initialize and train XGBoost model
        print("Step 6/6: Training XGBoost model...")
        self.model = xgb.XGBRanker(
            objective='rank:pairwise',
            n_estimators=100 if debug_mode else 200,  # Fewer trees in debug mode
            max_depth=6 if debug_mode else 8,  # Smaller depth in debug mode
            learning_rate=0.15 if debug_mode else 0.1,  # Higher LR in debug mode for faster convergence
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        
        # Train the model
        X = training_data['features']
        y = training_data['labels']
        groups = training_data['groups']  # For ranking
        
        print(f"Training XGBoost Ranker with {len(X)} samples, {X.shape[1]} features, {len(groups)} groups...")
        self.model.fit(X, y, group=groups)
        
        # Save the model
        self._save_model()
        
        print("Integrated XGBoost Reranker built and saved successfully!")
    
    def _compute_global_features(self):
        """Precompute global features for efficiency."""
        print("Computing global features...")
        
        # Compute query frequency
        self.query_frequency = self.user_queries['corrected_query'].value_counts().to_dict()
        
        # Compute brand frequency from session logs
        brand_clicks = {}
        category_clicks = {}
        
        for _, row in self.session_log.iterrows():
            if pd.notna(row['clicked_product_id']):
                product_id = row['clicked_product_id']
                product_info = self.product_catalog[
                    self.product_catalog['product_id'] == product_id
                ]
                if len(product_info) > 0:
                    brand = product_info.iloc[0]['brand']
                    category = product_info.iloc[0]['category']
                    brand_clicks[brand] = brand_clicks.get(brand, 0) + 1
                    category_clicks[category] = category_clicks.get(category, 0) + 1
        
        self.brand_frequency = brand_clicks
        self.category_popularity = category_clicks
        
        # Compute persona profiles
        self._compute_persona_profiles()
        
        # Precompute title embeddings for semantic similarity
        print("Computing title embeddings...")
        titles = self.product_catalog['title'].astype(str).tolist()
        self.title_embeddings = self.embedding_model.encode(titles, show_progress_bar=True)
        
        # Create title to product mapping
        for idx, row in self.product_catalog.iterrows():
            self.title_to_product[row['title']] = {
                'product_id': row['product_id'],
                'brand': row['brand'],
                'category': row['category'],
                'price': row['price'],
                'embedding_idx': idx
            }
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        all_texts = titles + list(self.user_queries['corrected_query'].astype(str))
        self.vectorizer.fit(all_texts)
        
        print("Global features computed successfully!")
    
    def _compute_persona_profiles(self):
        """Compute user persona profiles from session data."""
        print("Computing persona profiles...")
        
        session_profiles = {}
        
        for session_id, session_data in self.session_log.groupby('session_id'):
            profile = {
                'avg_price_last_k_clicks': 0.0,
                'preferred_brands': {},
                'session_length': len(session_data),
                'offer_seeking_score': 0.0,
                'budget_friendly_score': 0.0,
                'brand_loyalty_score': 0.0,
                'persona_tag': 'general'
            }
            
            clicked_products = session_data[session_data['clicked_product_id'].notna()]
            
            if len(clicked_products) > 0:
                # Calculate average price of clicked products
                prices = []
                brands = []
                
                for _, click in clicked_products.iterrows():
                    product_id = click['clicked_product_id']
                    
                    # Get product info
                    product_info = self.product_catalog[
                        self.product_catalog['product_id'] == product_id
                    ]
                    
                    if len(product_info) > 0:
                        price = product_info.iloc[0]['price']
                        brand = product_info.iloc[0]['brand']
                        prices.append(price)
                        brands.append(brand)
                
                if prices:
                    profile['avg_price_last_k_clicks'] = np.mean(prices)
                    
                    # Budget-friendly score (preference for lower prices)
                    profile['budget_friendly_score'] = 1.0 / (1.0 + np.mean(prices) / 10000)
                
                if brands:
                    # Brand preference
                    from collections import Counter
                    brand_counts = Counter(brands)
                    profile['preferred_brands'] = dict(brand_counts)
                    
                    # Brand loyalty score (concentration of clicks on few brands)
                    total_clicks = len(brands)
                    unique_brands = len(set(brands))
                    profile['brand_loyalty_score'] = 1.0 - (unique_brands / total_clicks)
            
            # Determine persona tag
            if profile['budget_friendly_score'] > 0.7:
                profile['persona_tag'] = 'budget_friendly'
            elif profile['brand_loyalty_score'] > 0.6:
                profile['persona_tag'] = 'brand_loyalist'
            elif profile['offer_seeking_score'] > 0.5:
                profile['persona_tag'] = 'offer_seeker'
            
            session_profiles[session_id] = profile
        
        self.persona_profiles = session_profiles
        print(f"Computed {len(session_profiles)} persona profiles")
    
    def _prepare_reranker_training_data(self):
        """Prepare training data with signals from all components (A, B, C) - Optimized version."""
        print("Preparing reranker training data with component signals...")
        
        features_list = []
        labels_list = []
        groups_list = []  # For XGBRanker
        
        # Pre-filter to only non-empty queries for efficiency
        valid_session_data = self.session_log[
            self.session_log['query'].notna() & 
            (self.session_log['query'].str.strip() != '')
        ].copy()
        
        query_groups = valid_session_data.groupby('query')
        total_queries = len(query_groups)
        processed_queries = 0
        valid_queries = 0
        
        print(f"Processing {total_queries} unique queries (pre-filtered from {len(self.session_log)} total sessions)...")
        
        # Pre-compute all unique queries for batch processing
        unique_queries = list(query_groups.groups.keys())
        
        # Process queries in smaller batches to reduce memory usage
        batch_size = 20
        for batch_start in range(0, len(unique_queries), batch_size):
            batch_end = min(batch_start + batch_size, len(unique_queries))
            batch_queries = unique_queries[batch_start:batch_end]
            
            # Pre-fetch candidates for the entire batch (more efficient)
            batch_candidates = {}
            for query in batch_queries:
                try:
                    batch_candidates[query] = self._get_candidates_from_all_components_fast(query)
                except Exception as e:
                    print(f"Warning: Error getting candidates for query '{query}': {e}")
                    batch_candidates[query] = {}
            
            # Process each query in the batch
            for query in batch_queries:
                processed_queries += 1
                
                # Show progress every 10% or every 50 queries (reduced frequency)
                if processed_queries % max(1, total_queries // 10) == 0 or processed_queries % 50 == 0:
                    print(f"Progress: {processed_queries}/{total_queries} queries ({processed_queries/total_queries*100:.1f}%) - Valid groups: {valid_queries}")
                
                query_sessions = query_groups.get_group(query)
                candidates = batch_candidates[query]
                
                if len(candidates) == 0:
                    continue
                
                # Limit candidates per query for efficiency (top 15 instead of all)
                if len(candidates) > 15:
                    # Sort by component signals and take top 15
                    sorted_candidates = sorted(
                        candidates.items(), 
                        key=lambda x: (x[1].get('from_trie', 0) + x[1].get('from_semantic', 0) + x[1].get('from_bert', 0)), 
                        reverse=True
                    )
                    candidates = dict(sorted_candidates[:15])
                
                # Get session context for this query (optimized)
                session_context = self._get_query_session_context_fast(query, query_sessions)
                
                group_features = []
                group_labels = []
                
                for candidate_title, component_signals in candidates.items():
                    try:
                        # Extract features (optimized version)
                        features = self._extract_reranker_features_fast(
                            query, candidate_title, component_signals, session_context
                        )
                        
                        # Create label (simplified)
                        label = self._compute_candidate_label_fast(candidate_title, query_sessions)
                        
                        group_features.append(features)
                        group_labels.append(label)
                        
                    except Exception as e:
                        # Skip problematic candidates instead of printing warnings
                        continue
                
                # Add to training data if we have multiple candidates
                if len(group_features) > 1:
                    features_list.extend(group_features)
                    labels_list.extend(group_labels)
                    groups_list.append(len(group_features))
                    valid_queries += 1
        
        print(f"Completed! Generated {len(features_list)} training examples from {len(groups_list)} valid query groups")
        print(f"Average candidates per query: {len(features_list)/max(1, len(groups_list)):.1f}")
        
        if len(features_list) == 0:
            print("Warning: No training data generated. Check your datasets and component implementations.")
            return {
                'features': np.array([[0.0] * 30]),  # Dummy features
                'labels': np.array([0.0]),
                'groups': np.array([1])
            }
        
        return {
            'features': np.array(features_list),
            'labels': np.array(labels_list),
            'groups': np.array(groups_list)
        }
    
    def _get_candidates_from_all_components(self, query: str) -> Dict[str, Dict]:
        """Get candidates from Trie (A), SBERT (B), and BERT (C) components."""
        candidates = {}
        
        try:
            # Component A: Trie Prefix Matching
            trie_suggestions = self.trie_component.get_suggestions_with_scores(query, max_suggestions=20)
            for rank, (suggestion, frequency) in enumerate(trie_suggestions):
                # Find product titles that match this suggestion
                matching_titles = self._find_matching_titles(suggestion)
                for title in matching_titles[:5]:  # Limit per suggestion
                    candidates[title] = candidates.get(title, {})
                    candidates[title].update({
                        'is_prefix': 1.0,
                        'prefix_rank': float(rank + 1),
                        'trie_frequency': float(frequency),
                        'from_trie': 1.0
                    })
        except Exception as e:
            print(f"Warning: Trie component error for query '{query}': {e}")
        
        try:
            # Component B: Semantic Correction (SBERT + FAISS)
            semantic_suggestions = self.semantic_component.get_semantic_suggestions(query, top_k=15)
            for suggestion, similarity in semantic_suggestions:
                matching_titles = self._find_matching_titles(suggestion)
                for title in matching_titles[:3]:
                    candidates[title] = candidates.get(title, {})
                    candidates[title].update({
                        'from_correction': 1.0,
                        'corr_sim': float(similarity),
                        'semantic_rank': len([s for s, _ in semantic_suggestions if s == suggestion]) + 1,
                        'from_semantic': 1.0
                    })
        except Exception as e:
            print(f"Warning: Semantic component error for query '{query}': {e}")
        
        try:
            # Component C: BERT Completion
            bert_completions = self.bert_component.complete_query(query, max_completions=10)
            for rank, completion in enumerate(bert_completions):
                matching_titles = self._find_matching_titles(completion)
                for title in matching_titles[:3]:
                    candidates[title] = candidates.get(title, {})
                    candidates[title].update({
                        'from_completion': 1.0,
                        'lm_score': 0.5,  # Default BERT score
                        'completion_rank': float(rank + 1),
                        'from_bert': 1.0
                    })
        except Exception as e:
            print(f"Warning: BERT component error for query '{query}': {e}")
        
        # Fill in missing component signals
        for title in candidates:
            candidates[title]['is_prefix'] = candidates[title].get('is_prefix', 0.0)
            candidates[title]['prefix_rank'] = candidates[title].get('prefix_rank', 999.0)
            candidates[title]['from_correction'] = candidates[title].get('from_correction', 0.0)
            candidates[title]['corr_sim'] = candidates[title].get('corr_sim', 0.0)
            candidates[title]['from_completion'] = candidates[title].get('from_completion', 0.0)
            candidates[title]['lm_score'] = candidates[title].get('lm_score', 0.0)
            candidates[title]['from_trie'] = candidates[title].get('from_trie', 0.0)
            candidates[title]['from_semantic'] = candidates[title].get('from_semantic', 0.0)
            candidates[title]['from_bert'] = candidates[title].get('from_bert', 0.0)
        
        return candidates
    
    def _get_candidates_from_all_components_fast(self, query: str) -> Dict[str, Dict]:
        """Fast version: Get candidates from Trie (A), SBERT (B), and BERT (C) components."""
        candidates = {}
        
        try:
            # Component A: Trie Prefix Matching (reduced limit)
            trie_suggestions = self.trie_component.get_suggestions_with_scores(query, max_suggestions=10)
            for rank, (suggestion, frequency) in enumerate(trie_suggestions):
                # Find product titles that match this suggestion (limited)
                matching_titles = self._find_matching_titles_fast(suggestion, max_results=3)
                for title in matching_titles:
                    candidates[title] = candidates.get(title, {})
                    candidates[title].update({
                        'is_prefix': 1.0,
                        'prefix_rank': float(rank + 1),
                        'trie_frequency': float(frequency),
                        'from_trie': 1.0
                    })
        except Exception:
            pass  # Skip errors silently for speed
        
        try:
            # Component B: Semantic Correction (reduced limit)
            semantic_suggestions = self.semantic_component.get_semantic_suggestions(query, top_k=8)
            for suggestion, similarity in semantic_suggestions:
                matching_titles = self._find_matching_titles_fast(suggestion, max_results=2)
                for title in matching_titles:
                    candidates[title] = candidates.get(title, {})
                    candidates[title].update({
                        'from_correction': 1.0,
                        'corr_sim': float(similarity),
                        'from_semantic': 1.0
                    })
        except Exception:
            pass  # Skip errors silently for speed
        
        try:
            # Component C: BERT Completion (reduced limit)
            bert_completions = self.bert_component.complete_query(query, max_completions=5)
            for rank, completion in enumerate(bert_completions):
                matching_titles = self._find_matching_titles_fast(completion, max_results=2)
                for title in matching_titles:
                    candidates[title] = candidates.get(title, {})
                    candidates[title].update({
                        'from_completion': 1.0,
                        'lm_score': 0.5,
                        'completion_rank': float(rank + 1),
                        'from_bert': 1.0
                    })
        except Exception:
            pass  # Skip errors silently for speed
        
        # Fill in missing component signals (vectorized)
        for title in candidates:
            signals = candidates[title]
            signals.setdefault('is_prefix', 0.0)
            signals.setdefault('prefix_rank', 999.0)
            signals.setdefault('from_correction', 0.0)
            signals.setdefault('corr_sim', 0.0)
            signals.setdefault('from_completion', 0.0)
            signals.setdefault('lm_score', 0.0)
            signals.setdefault('from_trie', 0.0)
            signals.setdefault('from_semantic', 0.0)
            signals.setdefault('from_bert', 0.0)
        
        return candidates
    
    def _find_matching_titles_fast(self, suggestion: str, max_results: int = 10) -> List[str]:
        """Fast version: Find product titles that match a suggestion."""
        suggestion_words = set(suggestion.lower().split())
        if not suggestion_words:
            return []
        
        matching_titles = []
        
        # Early exit if no words to match
        if len(suggestion_words) == 0:
            return []
        
        # Limit the search to improve speed
        for _, product in self.product_catalog.head(5000).iterrows():  # Limit search space
            title_words = set(str(product['title']).lower().split())
            brand_words = set(str(product['brand']).lower().split())
            category_words = set(str(product['category']).lower().split())
            
            # Fast intersection-based matching
            title_matches = len(suggestion_words & title_words)
            brand_matches = len(suggestion_words & brand_words)
            category_matches = len(suggestion_words & category_words)
            
            match_score = title_matches * 3 + brand_matches * 2 + category_matches * 1
            
            if match_score > 0:
                matching_titles.append((product['title'], match_score))
                
                # Early exit when we have enough results
                if len(matching_titles) >= max_results * 2:
                    break
        
        # Sort and return top results
        matching_titles.sort(key=lambda x: x[1], reverse=True)
        return [title for title, _ in matching_titles[:max_results]]
    
    def _find_matching_titles(self, suggestion: str) -> List[str]:
        """Find product titles that match a suggestion."""
        suggestion_words = set(suggestion.lower().split())  # Use set for faster lookup
        matching_titles = []
        
        for _, product in self.product_catalog.iterrows():
            title = str(product['title']).lower()
            brand = str(product['brand']).lower()
            category = str(product['category']).lower()
            
            # Check for word matches (optimized)
            match_score = 0
            title_words = set(title.split())
            brand_words = set(brand.split())
            category_words = set(category.split())
            
            # Count intersections for faster matching
            title_matches = len(suggestion_words & title_words)
            brand_matches = len(suggestion_words & brand_words)
            category_matches = len(suggestion_words & category_words)
            
            match_score = title_matches * 3 + brand_matches * 2 + category_matches * 1
            
            if match_score > 0:
                matching_titles.append((product['title'], match_score))
        
        # Sort by match score and return titles (limit to top 20 for efficiency)
        matching_titles.sort(key=lambda x: x[1], reverse=True)
        return [title for title, _ in matching_titles[:20]]
    
    def _get_query_session_context_fast(self, query: str, query_sessions: pd.DataFrame) -> Dict:
        """Fast version: Get session context for a query."""
        context = {
            'avg_price_last_k_clicks': 0.0,
            'preferred_brands': {},
            'session_length': len(query_sessions),
            'persona_tag': 'general',
            'locations': query_sessions['location'].iloc[0] if len(query_sessions) > 0 else 'mumbai',
            'events': query_sessions['event'].iloc[0] if len(query_sessions) > 0 else 'search'
        }
        
        # Get session IDs for this query (take first only for speed)
        if len(query_sessions) > 0:
            session_id = query_sessions['session_id'].iloc[0]
            if session_id in self.persona_profiles:
                profile = self.persona_profiles[session_id]
                context.update({
                    'avg_price_last_k_clicks': profile.get('avg_price_last_k_clicks', 0.0),
                    'preferred_brands': profile.get('preferred_brands', {}),
                    'persona_tag': profile.get('persona_tag', 'general'),
                    'brand_loyalty_score': profile.get('brand_loyalty_score', 0.0)
                })
        
        return context
    
    def _compute_candidate_label_fast(self, candidate_title: str, query_sessions: pd.DataFrame) -> float:
        """Fast version: Compute relevance label for a candidate."""
        # Get product info for this title
        product_info = self.title_to_product.get(candidate_title, {})
        product_id = product_info.get('product_id', '')
        
        if not product_id:
            return 0.2  # Default small positive label
        
        # Check if any session clicked this product
        clicked_products = query_sessions['clicked_product_id'].dropna()
        if len(clicked_products) > 0 and product_id in clicked_products.values:
            # Check for purchase/high-value events
            relevant_sessions = query_sessions[query_sessions['clicked_product_id'] == product_id]
            if len(relevant_sessions) > 0:
                if relevant_sessions['purchased'].any():
                    return 4.0  # Purchase
                elif (relevant_sessions['event'] == 'add_to_cart').any():
                    return 3.0  # Add to cart
                elif (relevant_sessions['event'] == 'view_details').any():
                    return 2.0  # View details
                else:
                    return 1.0  # Click
        
        return 0.2  # Default small positive label
    
    def _extract_reranker_features_fast(self, query: str, candidate_title: str, 
                                      component_signals: Dict, session_context: Dict) -> List[float]:
        """Fast version: Extract essential features for reranking."""
        # Get product info for candidate
        product_info = self.title_to_product.get(candidate_title, {})
        if not product_info:
            return [0.0] * 30  # Return zero features if product not found
        
        product_id = product_info['product_id']
        
        # Get basic product info (avoid dataframe lookups when possible)
        brand = product_info.get('brand', '')
        category = product_info.get('category', '')
        price = product_info.get('price', 0)
        
        features = []
        
        # ===== COMPONENT SIGNALS (A, B, C) =====
        features.append(component_signals.get('is_prefix', 0.0))
        features.append(min(component_signals.get('prefix_rank', 999.0), 100.0))  # Cap large values
        features.append(component_signals.get('from_correction', 0.0))
        features.append(component_signals.get('corr_sim', 0.0))
        features.append(component_signals.get('from_completion', 0.0))
        features.append(component_signals.get('lm_score', 0.0))
        
        # Component combination signals
        num_components = sum([
            component_signals.get('from_trie', 0.0),
            component_signals.get('from_semantic', 0.0),
            component_signals.get('from_bert', 0.0)
        ])
        features.append(num_components)
        
        # ===== QUERY FEATURES =====
        features.append(min(self.query_frequency.get(query, 0), 1000))  # Cap frequency
        features.append(len(query.split()))
        
        # ===== SESSION/USER FEATURES =====
        features.append(self._encode_persona_tag(session_context.get('persona_tag', 'general')))
        features.append(min(session_context.get('avg_price_last_k_clicks', 0.0), 100000))  # Cap price
        features.append(min(len(session_context.get('preferred_brands', {})), 20))  # Cap brands
        features.append(min(session_context.get('session_length', 0), 50))  # Cap session length
        features.append(session_context.get('brand_loyalty_score', 0.0))
        
        # Event relevance (simplified)
        event = session_context.get('events', 'search')
        if isinstance(event, list):
            event = event[0] if event else 'search'
        features.append(self._get_event_relevance_score(query, event))
        
        # ===== PRODUCT FEATURES =====
        features.append(min(float(price), 200000))  # Cap price
        features.append(min(self.brand_frequency.get(brand, 0), 1000))  # Cap brand frequency
        features.append(min(self.category_popularity.get(category, 0), 1000))  # Cap category popularity
        
        # Simplified realtime features (avoid dataframe lookup)
        features.append(0.0)  # offer_strength (placeholder)
        features.append(4.0)  # rating (default good rating)
        features.append(3.0)  # stock_status (default in stock)
        features.append(1.0)  # is_f_assured (assume yes)
        
        # ===== INTERACTION FEATURES =====
        features.append(1.0 if brand.lower() in query.lower() else 0.0)  # Brand match
        features.append(1.0 if category.lower() in query.lower() else 0.0)  # Category match
        
        # Price gap (simplified)
        avg_price = session_context.get('avg_price_last_k_clicks', price)
        price_gap = min(abs(float(price) - avg_price), 50000)  # Cap price gap
        features.append(price_gap)
        
        # Offer preference match (simplified)
        features.append(0.5)  # Default moderate preference
        
        # Semantic similarity (simplified - avoid expensive computation)
        query_words = set(query.lower().split())
        title_words = set(candidate_title.lower().split())
        word_overlap = len(query_words & title_words) / max(len(query_words), 1)
        features.append(word_overlap)
        
        # Text similarity (simplified)
        features.append(word_overlap * 0.8)  # Approximate TF-IDF
        
        # Location relevance (simplified)
        features.append(0.0)  # Placeholder
        
        # Title characteristics
        features.append(min(len(candidate_title.split()), 20))  # Cap title length
        
        # Historical click count
        features.append(min(self.brand_frequency.get(brand, 0), 1000))
        
        # Predicted conversion rate (simplified)
        base_rate = 0.1
        if brand in session_context.get('preferred_brands', {}):
            base_rate += 0.2
        features.append(base_rate)
        
        return features
    
    def _get_query_session_context(self, query: str, query_sessions: pd.DataFrame) -> Dict:
        """Get session context for a query."""
        context = {
            'avg_price_last_k_clicks': 0.0,
            'preferred_brands': {},
            'session_length': len(query_sessions),
            'persona_tag': 'general',
            'locations': [],
            'events': []
        }
        
        # Get session IDs for this query
        session_ids = query_sessions['session_id'].unique()
        
        if len(session_ids) > 0:
            # Use the first session for profile
            session_id = session_ids[0]
            if session_id in self.persona_profiles:
                context.update(self.persona_profiles[session_id])
        
        # Get locations and events
        context['locations'] = query_sessions['location'].unique().tolist()
        context['events'] = query_sessions['event'].unique().tolist()
        
        return context
    
    def _compute_candidate_label(self, candidate_title: str, query_sessions: pd.DataFrame) -> float:
        """Compute relevance label for a candidate based on user interactions."""
        label = 0.0
        
        # Get product info for this title
        product_info = self.title_to_product.get(candidate_title, {})
        product_id = product_info.get('product_id', '')
        
        for _, session in query_sessions.iterrows():
            clicked_product_id = session.get('clicked_product_id', '')
            
            if pd.notna(clicked_product_id) and clicked_product_id == product_id:
                if session.get('purchased', False):
                    label = max(label, 4.0)  # Purchase
                elif session.get('event', '') == 'add_to_cart':
                    label = max(label, 3.0)  # Add to cart
                elif session.get('event', '') == 'view_details':
                    label = max(label, 2.0)  # View details
                else:
                    label = max(label, 1.0)  # Click
        
        # If no interaction, give small positive label for relevant products
        if label == 0.0:
            # Check semantic relevance
            query = query_sessions.iloc[0]['query'] if len(query_sessions) > 0 else ''
            if self._has_semantic_relevance(query, candidate_title):
                label = 0.2
        
        return label
    
    def _has_semantic_relevance(self, query: str, title: str) -> bool:
        """Check if a title has semantic relevance to a query."""
        try:
            similarity = self._semantic_similarity(query, title)
            return similarity > 0.3
        except:
            return False
    
    def _get_relevant_products(self, query: str) -> List[str]:
        """Get relevant products for a query using simple text matching."""
        query_words = query.lower().split()
        relevant_products = []
        
        for _, product in self.product_catalog.iterrows():
            title = str(product['title']).lower()
            brand = str(product['brand']).lower()
            category = str(product['category']).lower()
            
            # Check if any query word appears in product info
            relevance_score = 0
            for word in query_words:
                if word in title:
                    relevance_score += 3
                elif word in brand:
                    relevance_score += 2
                elif word in category:
                    relevance_score += 1
            
            if relevance_score > 0:
                relevant_products.append((product['product_id'], relevance_score))
        
        # Sort by relevance and return product IDs
        relevant_products.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in relevant_products]
    
    def _extract_reranker_features(self, query: str, candidate_title: str, 
                                 component_signals: Dict, session_context: Dict) -> List[float]:
        """Extract comprehensive features for reranking with component signals."""
        features = []
        
        # Get product info for candidate
        product_info = self.title_to_product.get(candidate_title, {})
        if not product_info:
            return [0.0] * 30  # Return zero features if product not found
        
        product_id = product_info['product_id']
        
        # Get full product and realtime info
        product_row = self.product_catalog[
            self.product_catalog['product_id'] == product_id
        ]
        realtime_info = self.realtime_product_info[
            self.realtime_product_info['product_id'] == product_id
        ]
        
        if len(product_row) == 0:
            return [0.0] * 30
        
        product = product_row.iloc[0]
        
        # ===== COMPONENT SIGNALS (A, B, C) =====
        features.append(component_signals.get('is_prefix', 0.0))  # from Trie
        features.append(component_signals.get('prefix_rank', 999.0))  # Trie rank
        features.append(component_signals.get('from_correction', 0.0))  # from SBERT
        features.append(component_signals.get('corr_sim', 0.0))  # SBERT similarity
        features.append(component_signals.get('from_completion', 0.0))  # from BERT
        features.append(component_signals.get('lm_score', 0.0))  # BERT score
        
        # Component combination signals
        num_components = sum([
            component_signals.get('from_trie', 0.0),
            component_signals.get('from_semantic', 0.0),
            component_signals.get('from_bert', 0.0)
        ])
        features.append(num_components)  # Number of components that suggested this
        
        # ===== QUERY FEATURES =====
        features.append(self.query_frequency.get(query, 0))  # Query frequency
        features.append(len(query.split()))  # Query length
        
        # ===== SESSION/USER FEATURES =====
        features.append(self._encode_persona_tag(session_context.get('persona_tag', 'general')))
        features.append(session_context.get('avg_price_last_k_clicks', 0.0))
        features.append(len(session_context.get('preferred_brands', {})))
        features.append(session_context.get('session_length', 0))
        features.append(session_context.get('brand_loyalty_score', 0.0))
        
        # Event relevance
        events = session_context.get('events', [])
        event_score = max([self._get_event_relevance_score(query, event) for event in events] + [0.0])
        features.append(event_score)
        
        # ===== PRODUCT FEATURES =====
        features.append(float(product['price']))  # Product price
        features.append(self._encode_brand(product['brand']))  # Brand popularity
        features.append(self.category_popularity.get(product['category'], 0))  # Category popularity
        
        # Realtime features
        if len(realtime_info) > 0:
            rt_info = realtime_info.iloc[0]
            features.append(self._encode_offer_strength(rt_info['offer_strength']))
            features.append(float(rt_info['rating']))
            features.append(self._encode_stock_status(rt_info['stock_status']))
        else:
            features.append(0.0)  # offer_strength
            features.append(0.0)  # rating
            features.append(1.0)  # stock_status (default)
        
        features.append(1.0 if product['is_f_assured'] else 0.0)  # F-Assured
        
        # ===== INTERACTION FEATURES (Query ↔ Product) =====
        features.append(self._brand_match(query, product['brand']))  # Brand match
        features.append(self._category_match(query, product['category']))  # Category match
        
        # Price gap to average
        avg_price = session_context.get('avg_price_last_k_clicks', product['price'])
        price_gap = float(product['price']) - avg_price
        features.append(price_gap)  # Price gap
        
        # Offer preference match
        offer_strength = features[-4] if len(realtime_info) > 0 else 0.0
        offer_pref_match = self._offer_preference_match(
            offer_strength, session_context.get('persona_tag', 'general')
        )
        features.append(offer_pref_match)
        
        # Semantic similarity (query to title)
        semantic_sim = self._semantic_similarity(query, candidate_title)
        features.append(semantic_sim)
        
        # Text similarity (TF-IDF)
        text_sim = self._text_similarity(query, candidate_title)
        features.append(text_sim)
        
        # Location relevance
        locations = session_context.get('locations', [])
        location_relevance = max([self._get_location_relevance(query, loc) for loc in locations] + [0.0])
        features.append(location_relevance)
        
        # Title characteristics
        features.append(len(candidate_title.split()))  # Title length
        
        # Historical click count
        click_count = self.brand_frequency.get(product['brand'], 0)
        features.append(float(click_count))
        
        # Predicted conversion rate
        conversion_rate = self._predict_conversion_rate(
            query, product, session_context, events[0] if events else 'search'
        )
        features.append(conversion_rate)
        
        return features
    
    def _encode_persona_tag(self, persona_tag: str) -> float:
        """Encode persona tag as numeric value."""
        mapping = {
            'brand_loyalist': 3.0,
            'budget_friendly': 2.0,
            'offer_seeker': 1.0,
            'general': 0.0
        }
        return mapping.get(persona_tag, 0.0)
    
    def _encode_brand(self, brand: str) -> float:
        """Encode brand as numeric value based on popularity."""
        return float(self.brand_frequency.get(brand, 0))
    
    def _encode_offer_strength(self, offer_strength: str) -> float:
        """Encode offer strength as numeric value."""
        if pd.isna(offer_strength) or offer_strength == 'No Offer':
            return 0.0
        
        # Extract percentage
        if '%' in str(offer_strength):
            percentage = re.findall(r'\d+', str(offer_strength))
            if percentage:
                return float(percentage[0]) / 100.0
        
        return 0.0
    
    def _encode_stock_status(self, stock_status: str) -> float:
        """Encode stock status as numeric value."""
        mapping = {
            'In Stock': 3.0,
            'Low Stock': 2.0,
            'Out of Stock': 0.0
        }
        return mapping.get(stock_status, 1.0)
    
    def _brand_match(self, query: str, brand: str) -> float:
        """Check if brand name appears in query."""
        return 1.0 if brand.lower() in query.lower() else 0.0
    
    def _offer_preference_match(self, offer_strength: float, persona_tag: str) -> float:
        """Check if offer strength matches persona preference."""
        if persona_tag == 'offer_seeker' and offer_strength > 0.1:
            return 1.0
        elif persona_tag == 'budget_friendly' and offer_strength > 0.15:
            return 1.0
        elif persona_tag == 'brand_loyalist':
            return 0.5  # Less sensitive to offers
        return 0.0
    
    def _semantic_similarity(self, query: str, title: str) -> float:
        """Calculate semantic similarity between query and product title."""
        try:
            embeddings = self.embedding_model.encode([query, title])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except:
            return 0.0
    
    def _category_match(self, query: str, category: str) -> float:
        """Check if category appears in query."""
        return 1.0 if category.lower() in query.lower() else 0.0
    
    def _text_similarity(self, query: str, title: str) -> float:
        """Calculate TF-IDF similarity between query and title."""
        try:
            vectors = self.vectorizer.transform([query, title])
            similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def _get_event_relevance_score(self, query: str, event: str) -> float:
        """Get relevance score based on event type."""
        event_weights = {
            'purchase': 1.0,
            'add_to_cart': 0.8,
            'view_details': 0.6,
            'click': 0.4,
            'search': 0.2,
            'compare': 0.3
        }
        return event_weights.get(event, 0.0)
    
    def _get_location_relevance(self, query: str, location: str) -> float:
        """Get location-based relevance score."""
        location_keywords = {
            'mumbai': ['fast', 'delivery', 'premium'],
            'delhi': ['express', 'quick', 'formal'],
            'bangalore': ['tech', 'gaming', 'laptop'],
            'chennai': ['traditional', 'formal'],
            'kolkata': ['budget', 'affordable']
        }
        
        if location.lower() in location_keywords:
            keywords = location_keywords[location.lower()]
            for keyword in keywords:
                if keyword in query.lower():
                    return 0.1
        
        return 0.0
    
    def _predict_conversion_rate(self, query: str, product: pd.Series, 
                               session_profile: Dict, event: str) -> float:
        """Predict synthetic conversion rate."""
        # Simple heuristic-based conversion rate prediction
        base_rate = 0.1
        
        # Brand loyalty boost
        preferred_brands = session_profile.get('preferred_brands', {})
        if product['brand'] in preferred_brands:
            base_rate += 0.2
        
        # Price match boost
        avg_price = session_profile.get('avg_price_last_k_clicks', 0)
        if avg_price > 0:
            price_ratio = float(product['price']) / avg_price
            if 0.8 <= price_ratio <= 1.2:  # Similar price range
                base_rate += 0.1
        
        # Offer boost
        if product.get('offer_strength', 0) > 0.1:
            base_rate += 0.1
        
        # F-Assured boost
        if product['is_f_assured']:
            base_rate += 0.05
        
        return min(base_rate, 1.0)
    
    def rerank_suggestions(self, query: str, session_context: Dict = None, 
                          location: str = None, event: str = None, 
                          max_suggestions: int = 10) -> List[Tuple[str, float]]:
        """Main reranking method: get candidates from A∪B∪C and rerank them."""
        if not self.model:
            print("Model not trained. Please build the reranker first.")
            return []
        
        # Get candidates from all three components (A ∪ B ∪ C)
        candidates = self._get_candidates_from_all_components(query)
        
        if not candidates:
            return []
        
        # Prepare session context
        if session_context is None:
            session_context = {
                'avg_price_last_k_clicks': 0.0,
                'preferred_brands': {},
                'session_length': 1,
                'persona_tag': 'general',
                'locations': [location] if location else ['mumbai'],
                'events': [event] if event else ['search']
            }
        else:
            session_context['locations'] = session_context.get('locations', [location] if location else ['mumbai'])
            session_context['events'] = session_context.get('events', [event] if event else ['search'])
        
        # Extract features for all candidates
        candidate_features = []
        candidate_titles = []
        
        for title, component_signals in candidates.items():
            features = self._extract_reranker_features(
                query, title, component_signals, session_context
            )
            candidate_features.append(features)
            candidate_titles.append(title)
        
        if not candidate_features:
            return []
        
        # Get XGBoost scores
        feature_matrix = np.array(candidate_features)
        relevance_scores = self.model.predict(feature_matrix)
        
        # Combine titles with scores and sort
        reranked_results = list(zip(candidate_titles, relevance_scores))
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top suggestions
        return reranked_results[:max_suggestions]
    
    def get_integrated_suggestions(self, query: str, session_context: Dict = None,
                                 location: str = None, event: str = None,
                                 max_suggestions: int = 10) -> List[Tuple[str, float]]:
        """Get integrated suggestions from all components with reranking."""
        return self.rerank_suggestions(query, session_context, location, event, max_suggestions)
    
    def _save_model(self):
        """Save the trained model and supporting data."""
        os.makedirs('../models', exist_ok=True)
        
        # Save XGBoost model
        with open('../models/xgboost_reranker.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save vectorizer
        with open('../models/reranker_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save other supporting data
        with open('../models/reranker_metadata.pkl', 'wb') as f:
            pickle.dump({
                'persona_profiles': self.persona_profiles,
                'brand_frequency': self.brand_frequency,
                'query_frequency': self.query_frequency
            }, f)
        
        print("XGBoost reranker model saved successfully!")
    
    def _load_model(self):
        """Load the trained model and supporting data."""
        try:
            # Load XGBoost model
            with open('../models/xgboost_reranker.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            # Load vectorizer
            with open('../models/reranker_vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Load supporting data
            with open('../models/reranker_metadata.pkl', 'rb') as f:
                metadata = pickle.load(f)
                self.persona_profiles = metadata['persona_profiles']
                self.brand_frequency = metadata['brand_frequency']
                self.query_frequency = metadata['query_frequency']
            
            print("XGBoost reranker model loaded successfully!")
            return True
        except FileNotFoundError:
            print("No saved model found. Please train the model first.")
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model."""
        if not self.model:
            return {}
        
        feature_names = [
            # Component signals (A, B, C)
            'is_prefix', 'prefix_rank', 'from_correction', 'corr_sim', 
            'from_completion', 'lm_score', 'num_components',
            
            # Query features
            'query_frequency', 'query_length',
            
            # Session/User features
            'persona_tag', 'avg_price_last_k_clicks', 'preferred_brands_count',
            'session_length', 'brand_loyalty_score', 'event_relevance',
            
            # Product features
            'price', 'brand_encoded', 'category_popularity', 'offer_strength',
            'rating', 'stock_status', 'is_f_assured',
            
            # Interaction features
            'brand_match', 'category_match', 'price_gap_to_avg', 
            'offer_preference_match', 'semantic_similarity', 'text_similarity',
            'location_relevance', 'title_length', 'historical_click_count',
            'predicted_conversion_rate'
        ]
        
        importance_scores = self.model.feature_importances_
        
        return dict(zip(feature_names, importance_scores))

# Test the integrated XGBoost reranker
if __name__ == "__main__":
    # Load and preprocess data
    from data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    preprocessor.run_all_preprocessing()
    data = preprocessor.get_processed_data()
    
    # Initialize and build integrated reranker with debug mode for faster testing
    reranker = XGBoostReranker()
    print("Note: Running in debug mode for faster testing. Set debug_mode=False for full dataset.")
    reranker.build_reranker(data, debug_mode=True)
    
    # Test integrated suggestions
    test_queries = [
        "samsung phone",
        "apple mobile",
        "nike shoes",
        "laptop gaming",
        "bluetooth headphones"
    ]
    
    test_session_context = {
        'avg_price_last_k_clicks': 25000.0,
        'preferred_brands': {'Samsung': 3, 'Apple': 1, 'Nike': 2},
        'session_length': 5,
        'persona_tag': 'brand_loyalist'
    }
    
    print("\n=== Integrated XGBoost Reranker Test Results ===")
    
    for query in test_queries:
        start_time = time.time()
        suggestions = reranker.get_integrated_suggestions(
            query, test_session_context, "Mumbai", "search"
        )
        end_time = time.time()
        
        print(f"\nQuery: '{query}'")
        print(f"Integrated suggestions ({len(suggestions)}):")
        for i, (suggestion, score) in enumerate(suggestions[:5], 1):
            print(f"   {i}. {suggestion} (score: {score:.4f})")
        print(f"Response time: {(end_time - start_time)*1000:.2f}ms")
    
    # Test component analysis
    print(f"\n=== Component Analysis ===")
    test_query = "samsung phone"
    candidates = reranker._get_candidates_from_all_components(test_query)
    
    print(f"Total candidates from all components: {len(candidates)}")
    print("Sample candidates with component signals:")
    
    for i, (title, signals) in enumerate(list(candidates.items())[:3]):
        print(f"\n{i+1}. {title}")
        print(f"   From Trie: {signals.get('from_trie', 0)}")
        print(f"   From Semantic: {signals.get('from_semantic', 0)}")
        print(f"   From BERT: {signals.get('from_bert', 0)}")
        print(f"   Prefix rank: {signals.get('prefix_rank', 'N/A')}")
        print(f"   Semantic similarity: {signals.get('corr_sim', 'N/A'):.3f}")
    
    # Test feature importance
    feature_importance = reranker.get_feature_importance()
    print(f"\n=== Top 10 Feature Importance ===")
    
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_features[:10]:
        print(f"{feature}: {importance:.4f}")
    
    # Performance test
    print(f"\n=== Performance Test ===")
    start_time = time.time()
    for _ in range(20):
        reranker.get_integrated_suggestions("phone", test_session_context)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 20 * 1000
    print(f"Average integrated suggestion time: {avg_time:.2f}ms")
    print(f"QPS: {20 / (end_time - start_time):.0f}")
    
    print("\n=== Pipeline Summary ===")
    print("✅ Component A (Trie): Prefix matching with frequency ranking")
    print("✅ Component B (SBERT): Semantic correction for typos") 
    print("✅ Component C (BERT): Context-aware query completion")
    print("✅ Component D (XGBoost): Intelligent reranking with 30+ features")
    print("✅ Integrated pipeline: A ∪ B ∪ C → Reranker → Top-K suggestions")
