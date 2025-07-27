import pandas as pd
import numpy as np
import json
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from collections import defaultdict, Counter
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDataPreprocessor:
    """
    Enhanced data preprocessing with advanced features for autosuggest and SRP systems.
    Implements the improvements outlined in the requirements.
    """
    
    def __init__(self, 
                 sbert_model_name: str = 'all-MiniLM-L6-v2',
                 cache_dir: str = 'cache',
                 enable_incremental: bool = True):
        """
        Initialize the enhanced data preprocessor.
        
        Args:
            sbert_model_name: SBERT model for semantic correction
            cache_dir: Directory for caching processed data
            enable_incremental: Enable incremental updates
        """
        self.sbert_model = SentenceTransformer(sbert_model_name)
        self.cache_dir = cache_dir
        self.enable_incremental = enable_incremental
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load cached data if available
        self.load_cached_data()
        
    def load_cached_data(self):
        """Load cached processed data to enable incremental updates."""
        self.cached_embeddings = {}
        self.cached_combined_texts = {}
        self.cached_major_categories = set()
        self.cached_locations = set()
        
        cache_file = os.path.join(self.cache_dir, 'processed_data_cache.pkl')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.cached_embeddings = cached_data.get('embeddings', {})
                    self.cached_combined_texts = cached_data.get('combined_texts', {})
                    self.cached_major_categories = cached_data.get('major_categories', set())
                    self.cached_locations = cached_data.get('locations', set())
                logger.info(f"Loaded cached data: {len(self.cached_embeddings)} embeddings, "
                          f"{len(self.cached_combined_texts)} combined texts")
            except Exception as e:
                logger.warning(f"Failed to load cached data: {e}")
    
    def save_cached_data(self):
        """Save processed data to cache for incremental updates."""
        cache_file = os.path.join(self.cache_dir, 'processed_data_cache.pkl')
        cached_data = {
            'embeddings': self.cached_embeddings,
            'combined_texts': self.cached_combined_texts,
            'major_categories': self.cached_major_categories,
            'locations': self.cached_locations
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cached_data, f)
        logger.info("Saved processed data to cache")
    
    def extract_key_features(self, specifications: str, description: str) -> str:
        """
        Extract key features from product specifications and description.
        Enhanced version that goes beyond first 100 characters.
        
        Args:
            specifications: JSON string of product specifications
            description: Product description
            
        Returns:
            Extracted key features as string
        """
        features = []
        
        # Parse specifications
        try:
            if isinstance(specifications, str):
                specs = json.loads(specifications)
            else:
                specs = specifications
                
            # Extract key specs
            key_specs = []
            for key, value in specs.items():
                if key.lower() in ['display', 'processor', 'ram', 'storage', 'camera', 
                                 'battery', 'type', 'material', 'features', 'capacity']:
                    key_specs.append(f"{key}: {value}")
            
            features.extend(key_specs)
            
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Extract key phrases from description using TF-IDF
        if description and len(description) > 100:
            # Use TF-IDF to find important terms
            try:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform([description])
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                tfidf_scores = tfidf_matrix.toarray()[0]
                
                # Get top 5 most important terms
                top_indices = np.argsort(tfidf_scores)[-5:]
                important_terms = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
                features.extend(important_terms)
            except Exception as e:
                logger.warning(f"TF-IDF extraction failed: {e}")
        
        return " | ".join(features) if features else ""
    
    def create_enriched_combined_text(self, product_data: pd.Series) -> str:
        """
        Create enriched combined_text with weighting and key features.
        
        Args:
            product_data: Product data row
            
        Returns:
            Enriched combined text
        """
        # Weighted concatenation with delimiters
        title = product_data.get('title', '')
        description = product_data.get('description', '')
        category = product_data.get('category', '')
        brand = product_data.get('brand', '')
        specifications = product_data.get('specifications', '')
        
        # Extract key features
        key_features = self.extract_key_features(specifications, description)
        
        # Weighted combination (title repeated for emphasis)
        combined_parts = [
            title,  # High weight (repeated)
            f"BRAND: {brand}",
            f"CATEGORY: {category}",
            title,  # Repeat title for emphasis
            description[:200] if description else "",  # First 200 chars
            key_features
        ]
        
        # Join with clear delimiters
        combined_text = " || ".join([part for part in combined_parts if part])
        
        return combined_text
    
    def update_major_categories_and_locations(self, product_catalog: pd.DataFrame, 
                                            realtime_info: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Automatically update major categories and locations lists.
        
        Args:
            product_catalog: Product catalog data
            realtime_info: Real-time product info
            
        Returns:
            Updated major categories and locations
        """
        # Extract major categories from product catalog
        categories = []
        for cat in product_catalog['category'].dropna():
            if '>' in cat:
                major_cat = cat.split('>')[0].strip()
                categories.append(major_cat)
            else:
                categories.append(cat.strip())
        
        # Get unique categories and locations
        major_categories = list(set(categories))
        locations = list(realtime_info['location'].unique())
        
        # Update cached sets
        self.cached_major_categories.update(major_categories)
        self.cached_locations.update(locations)
        
        return list(self.cached_major_categories), list(self.cached_locations)
    
    def parse_delivery_speed(self, delivery_estimate: str) -> float:
        """
        Parse delivery estimate into granular delivery speed score.
        
        Args:
            delivery_estimate: Delivery estimate string
            
        Returns:
            Delivery speed score (lower is faster)
        """
        if pd.isna(delivery_estimate) or not delivery_estimate:
            return 7.0  # Default to 7 days
        
        delivery_estimate = str(delivery_estimate).lower().strip()
        
        # Parse different formats
        if 'minute' in delivery_estimate:
            # Extract minutes
            minutes = re.findall(r'(\d+)', delivery_estimate)
            if minutes:
                return float(minutes[0]) / 1440  # Convert to days
        elif 'hour' in delivery_estimate:
            # Extract hours
            hours = re.findall(r'(\d+)', delivery_estimate)
            if hours:
                return float(hours[0]) / 24  # Convert to days
        elif 'day' in delivery_estimate:
            # Extract days
            days = re.findall(r'(\d+)', delivery_estimate)
            if days:
                if len(days) == 2:  # Range like "2-3 days"
                    return (float(days[0]) + float(days[1])) / 2
                else:
                    return float(days[0])
        elif 'next day' in delivery_estimate or 'tomorrow' in delivery_estimate:
            return 1.0
        elif 'same day' in delivery_estimate or 'today' in delivery_estimate:
            return 0.1
        
        # Default parsing for numeric values
        numbers = re.findall(r'(\d+)', delivery_estimate)
        if numbers:
            return float(numbers[0])
        
        return 7.0  # Default
    
    def calculate_offer_strength(self, offer_strength: str) -> float:
        """
        Calculate numerical offer strength from offer string.
        
        Args:
            offer_strength: Offer strength string
            
        Returns:
            Numerical offer strength (0-1)
        """
        if pd.isna(offer_strength) or not offer_strength:
            return 0.0
        
        offer_strength = str(offer_strength).lower().strip()
        
        if 'no offer' in offer_strength:
            return 0.0
        
        # Extract percentage
        percentage_match = re.search(r'(\d+)%', offer_strength)
        if percentage_match:
            percentage = float(percentage_match.group(1))
            return min(percentage / 100.0, 1.0)  # Normalize to 0-1
        
        # Extract other offer indicators
        if 'flash' in offer_strength:
            return 0.8
        elif 'clearance' in offer_strength:
            return 0.6
        elif 'sale' in offer_strength:
            return 0.4
        
        return 0.0
    
    def advanced_semantic_correction(self, raw_query: str, query_log: pd.DataFrame) -> str:
        """
        Advanced semantic correction using SBERT and edit distance.
        
        Args:
            raw_query: Raw user query
            query_log: Historical query log
            
        Returns:
            Corrected query
        """
        if not raw_query or pd.isna(raw_query):
            return raw_query
        
        raw_query = raw_query.lower().strip()
        
        # Get unique corrected queries from log
        corrected_queries = query_log['corrected_query'].unique()
        
        # First, try simple edit distance for obvious typos
        simple_corrections = {
            'samsng': 'samsung', 'aple': 'apple', 'nkie': 'nike',
            'addidas': 'adidas', 'soney': 'sony', 'onepls': 'oneplus',
            'xiomi': 'xiaomi', 'vvo': 'vivo', 'opo': 'oppo',
            'realmi': 'realme', 'del': 'dell', 'lenvo': 'lenovo',
            'asuss': 'asus', 'bot': 'boat', 'jb': 'jbl',
            'pma': 'puma', 'rebbok': 'reebok', 'zarra': 'zara',
            'ikia': 'ikea', 'prestig': 'prestige', 'bta': 'bata'
        }
        
        if raw_query in simple_corrections:
            return simple_corrections[raw_query]
        
        # For more complex cases, use SBERT semantic similarity
        if len(raw_query) > 2:  # Only for queries with sufficient length
            try:
                # Get embeddings for raw query and corrected queries
                query_embedding = self.sbert_model.encode([raw_query])
                corrected_embeddings = self.sbert_model.encode(corrected_queries)
                
                # Calculate similarities
                similarities = cosine_similarity(query_embedding, corrected_embeddings)[0]
                
                # Find best match
                best_idx = np.argmax(similarities)
                best_similarity = similarities[best_idx]
                
                # Only correct if similarity is high enough
                if best_similarity > 0.7:
                    return corrected_queries[best_idx]
                    
            except Exception as e:
                logger.warning(f"SBERT correction failed for '{raw_query}': {e}")
        
        return raw_query
    
    def synthesize_predicted_purchase(self, clicked_product_ids: str, 
                                    product_catalog: pd.DataFrame,
                                    query: str) -> str:
        """
        Synthesize predicted_purchase based on product attributes and query alignment.
        
        Args:
            clicked_product_ids: Comma-separated product IDs
            product_catalog: Product catalog
            query: User query
            
        Returns:
            Predicted purchase product ID
        """
        if not clicked_product_ids or pd.isna(clicked_product_ids):
            return ""
        
        # Parse clicked product IDs
        product_ids = [pid.strip() for pid in clicked_product_ids.split(',') if pid.strip()]
        
        if not product_ids:
            return ""
        
        # Get product details for clicked products
        clicked_products = product_catalog[product_catalog['product_id'].isin(product_ids)]
        
        if clicked_products.empty:
            return product_ids[0]  # Return first if not found
        
        # Calculate alignment scores
        alignment_scores = []
        query_lower = query.lower()
        
        for _, product in clicked_products.iterrows():
            score = 0
            
            # Brand alignment
            if product['brand'].lower() in query_lower:
                score += 3
            
            # Category alignment
            if product['category'].lower() in query_lower:
                score += 2
            
            # Title alignment
            if any(word in product['title'].lower() for word in query_lower.split()):
                score += 1
            
            # Rating bonus
            score += product.get('rating', 0) / 5.0
            
            alignment_scores.append((product['product_id'], score))
        
        # Return product with highest alignment score
        if alignment_scores:
            best_product = max(alignment_scores, key=lambda x: x[1])
            return str(best_product[0])
        
        return str(product_ids[0])
    
    def deduplicate_and_aggregate_queries(self, query_log: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate and aggregate user queries.
        
        Args:
            query_log: Raw query log
            
        Returns:
            Deduplicated and aggregated query log
        """
        # Group by corrected_query and aggregate
        aggregated = query_log.groupby('corrected_query').agg({
            'raw_query': lambda x: list(x),
            'frequency': 'sum',
            'event': lambda x: list(x),
            'category': lambda x: list(x),
            'clicked_product_ids': lambda x: list(x)
        }).reset_index()
        
        # Flatten lists and take most common values
        def get_most_common(lst):
            if not lst:
                return ""
            # Flatten list of lists
            flat_list = []
            for item in lst:
                if isinstance(item, str) and ',' in item:
                    flat_list.extend([pid.strip() for pid in item.split(',')])
                else:
                    flat_list.append(item)
            
            if not flat_list:
                return ""
            
            # Return most common item
            counter = Counter(flat_list)
            return counter.most_common(1)[0][0]
        
        aggregated['raw_query'] = aggregated['raw_query'].apply(lambda x: x[0] if x else "")
        aggregated['event'] = aggregated['event'].apply(lambda x: Counter(x).most_common(1)[0][0] if x else "")
        aggregated['category'] = aggregated['category'].apply(lambda x: Counter(x).most_common(1)[0][0] if x else "")
        aggregated['clicked_product_ids'] = aggregated['clicked_product_ids'].apply(get_most_common)
        
        return aggregated
    
    def process_product_catalog(self, product_catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Process product catalog with enhanced features.
        
        Args:
            product_catalog: Raw product catalog
            
        Returns:
            Processed product catalog
        """
        logger.info("Processing product catalog with enhanced features...")
        
        processed_catalog = product_catalog.copy()
        
        # Create enriched combined_text
        processed_catalog['combined_text'] = processed_catalog.apply(
            self.create_enriched_combined_text, axis=1
        )
        
        # Add product hash for incremental updates
        processed_catalog['product_hash'] = processed_catalog.apply(
            lambda row: hashlib.md5(
                f"{row['title']}{row['description']}{row['specifications']}".encode()
            ).hexdigest(), axis=1
        )
        
        # Extract specifications as separate columns
        processed_catalog['specs_dict'] = processed_catalog['specifications'].apply(
            lambda x: json.loads(x) if isinstance(x, str) else {}
        )
        
        # Add key specification columns
        processed_catalog['display_size'] = processed_catalog['specs_dict'].apply(
            lambda x: x.get('display', '')
        )
        processed_catalog['processor'] = processed_catalog['specs_dict'].apply(
            lambda x: x.get('processor', '')
        )
        processed_catalog['ram'] = processed_catalog['specs_dict'].apply(
            lambda x: x.get('ram', '')
        )
        processed_catalog['storage'] = processed_catalog['specs_dict'].apply(
            lambda x: x.get('storage', '')
        )
        
        logger.info(f"Processed {len(processed_catalog)} products")
        return processed_catalog
    
    def process_user_queries(self, user_queries: pd.DataFrame, 
                           product_catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Process user queries with advanced semantic correction.
        
        Args:
            user_queries: Raw user queries
            product_catalog: Product catalog for purchase prediction
            
        Returns:
            Processed user queries
        """
        logger.info("Processing user queries with advanced semantic correction...")
        
        processed_queries = user_queries.copy()
        
        # Apply advanced semantic correction
        processed_queries['corrected_query'] = processed_queries['raw_query'].apply(
            lambda x: self.advanced_semantic_correction(x, user_queries)
        )
        
        # Synthesize predicted_purchase
        processed_queries['predicted_purchase'] = processed_queries.apply(
            lambda row: self.synthesize_predicted_purchase(
                row['clicked_product_ids'], product_catalog, row['corrected_query']
            ), axis=1
        )
        
        # Deduplicate and aggregate
        processed_queries = self.deduplicate_and_aggregate_queries(processed_queries)
        
        logger.info(f"Processed {len(processed_queries)} unique queries")
        return processed_queries
    
    def process_realtime_info(self, realtime_info: pd.DataFrame) -> pd.DataFrame:
        """
        Process real-time product info with granular features.
        
        Args:
            realtime_info: Raw real-time info
            
        Returns:
            Processed real-time info
        """
        logger.info("Processing real-time product info...")
        
        processed_info = realtime_info.copy()
        
        # Parse delivery speed
        processed_info['delivery_speed_score'] = processed_info['delivery_estimate'].apply(
            self.parse_delivery_speed
        )
        
        # Calculate offer strength
        processed_info['offer_strength_numeric'] = processed_info['offer_strength'].apply(
            self.calculate_offer_strength
        )
        
        # Add stock status numeric
        stock_status_map = {
            'In Stock': 1.0,
            'Low Stock': 0.3,
            'Out of Stock': 0.0
        }
        processed_info['stock_status_numeric'] = processed_info['stock_status'].map(
            stock_status_map
        ).fillna(0.0)
        
        # Add regional pricing indicator
        processed_info['has_regional_pricing'] = processed_info['location'].apply(
            lambda x: x in ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata']
        )
        
        logger.info(f"Processed {len(processed_info)} real-time records")
        return processed_info
    
    def process_session_log(self, session_log: pd.DataFrame) -> pd.DataFrame:
        """
        Process session log with deeper context.
        
        Args:
            session_log: Raw session log
            
        Returns:
            Processed session log
        """
        logger.info("Processing session log with deeper context...")
        
        processed_sessions = session_log.copy()
        
        # Convert timestamp to datetime
        processed_sessions['timestamp'] = pd.to_datetime(processed_sessions['timestamp'])
        
        # Add session features
        processed_sessions['session_duration'] = processed_sessions.groupby('session_id')['timestamp'].transform(
            lambda x: (x.max() - x.min()).total_seconds() / 60  # Duration in minutes
        )
        
        # Add time-based features
        processed_sessions['hour_of_day'] = processed_sessions['timestamp'].dt.hour
        processed_sessions['day_of_week'] = processed_sessions['timestamp'].dt.dayofweek
        processed_sessions['is_weekend'] = processed_sessions['day_of_week'].isin([5, 6])
        
        # Add session sequence number
        processed_sessions['session_sequence'] = processed_sessions.groupby('session_id').cumcount() + 1
        
        # Add purchase indicator
        processed_sessions['is_purchase'] = processed_sessions['purchased'].astype(bool)
        
        logger.info(f"Processed {len(processed_sessions)} session records")
        return processed_sessions
    
    def create_session_context(self, session_log: pd.DataFrame, 
                             window_hours: int = 24) -> Dict[str, Dict]:
        """
        Create session context for recent sessions.
        
        Args:
            session_log: Processed session log
            window_hours: Time window for context
            
        Returns:
            Session context dictionary
        """
        logger.info("Creating session context...")
        
        # Filter recent sessions
        cutoff_time = session_log['timestamp'].max() - timedelta(hours=window_hours)
        recent_sessions = session_log[session_log['timestamp'] >= cutoff_time]
        
        session_context = {}
        
        for session_id in recent_sessions['session_id'].unique():
            session_data = recent_sessions[recent_sessions['session_id'] == session_id]
            
            context = {
                'recent_queries': session_data['query'].tolist()[-5:],  # Last 5 queries
                'recent_categories': session_data['event'].tolist()[-3:],  # Last 3 events
                'recent_products': session_data['clicked_product_id'].dropna().tolist()[-3:],
                'session_duration': session_data['session_duration'].iloc[0],
                'purchase_made': session_data['is_purchase'].any(),
                'location': session_data['location'].iloc[0] if not session_data['location'].empty else None
            }
            
            session_context[session_id] = context
        
        logger.info(f"Created context for {len(session_context)} sessions")
        return session_context
    
    def process_all_data(self, 
                        product_catalog_path: str,
                        user_queries_path: str,
                        realtime_info_path: str,
                        session_log_path: str,
                        output_dir: str = 'processed_data') -> Dict[str, pd.DataFrame]:
        """
        Process all datasets with enhanced features.
        
        Args:
            product_catalog_path: Path to product catalog CSV
            user_queries_path: Path to user queries CSV
            realtime_info_path: Path to real-time info CSV
            session_log_path: Path to session log CSV
            output_dir: Output directory for processed data
            
        Returns:
            Dictionary of processed DataFrames
        """
        logger.info("Starting enhanced data processing...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        product_catalog = pd.read_csv(product_catalog_path)
        user_queries = pd.read_csv(user_queries_path)
        realtime_info = pd.read_csv(realtime_info_path)
        session_log = pd.read_csv(session_log_path)
        
        # Process each dataset
        processed_catalog = self.process_product_catalog(product_catalog)
        processed_queries = self.process_user_queries(user_queries, product_catalog)
        processed_realtime = self.process_realtime_info(realtime_info)
        processed_sessions = self.process_session_log(session_log)
        
        # Update major categories and locations
        major_categories, locations = self.update_major_categories_and_locations(
            processed_catalog, processed_realtime
        )
        
        # Create session context
        session_context = self.create_session_context(processed_sessions)
        
        # Save processed data
        processed_catalog.to_csv(os.path.join(output_dir, 'enhanced_product_catalog.csv'), index=False)
        processed_queries.to_csv(os.path.join(output_dir, 'enhanced_user_queries.csv'), index=False)
        processed_realtime.to_csv(os.path.join(output_dir, 'enhanced_realtime_info.csv'), index=False)
        processed_sessions.to_csv(os.path.join(output_dir, 'enhanced_session_log.csv'), index=False)
        
        # Save metadata
        metadata = {
            'major_categories': major_categories,
            'locations': locations,
            'session_context': session_context,
            'processing_timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, 'processing_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save cache
        self.save_cached_data()
        
        logger.info("Enhanced data processing completed!")
        
        return {
            'product_catalog': processed_catalog,
            'user_queries': processed_queries,
            'realtime_info': processed_realtime,
            'session_log': processed_sessions,
            'metadata': metadata
        }

if __name__ == "__main__":
    # Example usage
    preprocessor = EnhancedDataPreprocessor()
    
    processed_data = preprocessor.process_all_data(
        product_catalog_path='../dataset/synthetic_product_catalog.csv',
        user_queries_path='../dataset/user_queries.csv',
        realtime_info_path='../dataset/realtime_product_info.csv',
        session_log_path='../dataset/session_log.csv'
    )
    
    print("Processing completed!")
    print(f"Processed {len(processed_data['product_catalog'])} products")
    print(f"Processed {len(processed_data['user_queries'])} unique queries")
    print(f"Processed {len(processed_data['realtime_info'])} real-time records")
    print(f"Processed {len(processed_data['session_log'])} session records") 