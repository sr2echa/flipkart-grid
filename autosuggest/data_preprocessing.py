import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import List, Dict, Tuple, Optional
import json

class DataPreprocessor:
    """Handles all data preprocessing for the autosuggest system."""
    
    def __init__(self):
        self.product_catalog = None
        self.user_queries = None
        self.realtime_product_info = None
        self.session_log = None
        self.ner_dataset = None
        self.major_categories = []
        self.locations = []
        
    def load_all_datasets(self):
        """Load all CSV datasets into DataFrames."""
        print("Loading datasets...")
        
        # Load product catalog
        self.product_catalog = pd.read_csv('../dataset/product_catalog.csv')
        print(f"Product catalog loaded: {len(self.product_catalog)} products")
        
        # Load user queries
        self.user_queries = pd.read_csv('../dataset/user_queries.csv')
        print(f"User queries loaded: {len(self.user_queries)} queries")
        
        # Load realtime product info
        self.realtime_product_info = pd.read_csv('../dataset/realtime_product_info.csv')
        print(f"Realtime product info loaded: {len(self.realtime_product_info)} records")
        
        # Load session log
        self.session_log = pd.read_csv('../dataset/session_log.csv')
        print(f"Session log loaded: {len(self.session_log)} sessions")
        
        # Load NER dataset
        self.ner_dataset = pd.read_csv('../dataset/ner_dataset.csv')
        print(f"NER dataset loaded: {len(self.ner_dataset)} tokens")
        
        return True
    
    def process_product_catalog(self) -> pd.DataFrame:
        """Process product catalog data."""
        print("Processing product catalog...")
        
        # Handle missing values
        self.product_catalog['description'] = self.product_catalog['description'].fillna('')
        self.product_catalog['specifications'] = self.product_catalog['specifications'].fillna('')
        
        # Create combined_text field
        self.product_catalog['combined_text'] = (
            self.product_catalog['title'] + '. ' +
            'Category: ' + self.product_catalog['category'] + '. ' +
            'Description: ' + self.product_catalog['description'].str[:100] + '. ' +
            self.product_catalog['specifications']
        )
        
        # Extract major categories
        self.major_categories = self.product_catalog['category'].unique().tolist()
        
        # Extract locations from realtime_product_info
        self.locations = self.realtime_product_info['location'].unique().tolist()
        
        print(f"Major categories: {self.major_categories}")
        print(f"Locations: {self.locations}")
        
        return self.product_catalog
    
    def process_user_queries(self) -> pd.DataFrame:
        """Process user queries data."""
        print("Processing user queries...")
        
        # Clean corrected_query
        self.user_queries['corrected_query'] = (
            self.user_queries['corrected_query']
            .str.lower()
            .str.strip()
        )
        
        # Ensure frequency is numeric
        self.user_queries['frequency'] = pd.to_numeric(self.user_queries['frequency'], errors='coerce')
        
        # Clean raw_query
        self.user_queries['raw_query'] = (
            self.user_queries['raw_query']
            .str.lower()
            .str.strip()
        )
        
        # Add predicted_purchase column (placeholder for now)
        self.user_queries['predicted_purchase'] = 0.0
        
        print(f"Processed {len(self.user_queries)} user queries")
        return self.user_queries
    
    def process_realtime_product_info(self) -> pd.DataFrame:
        """Process realtime product info data."""
        print("Processing realtime product info...")
        
        # Ensure numeric fields are properly typed
        self.realtime_product_info['current_price'] = pd.to_numeric(
            self.realtime_product_info['current_price'], errors='coerce'
        )
        self.realtime_product_info['rating'] = pd.to_numeric(
            self.realtime_product_info['rating'], errors='coerce'
        )
        self.realtime_product_info['review_count'] = pd.to_numeric(
            self.realtime_product_info['review_count'], errors='coerce'
        )
        
        # Create delivery_speed_score
        def extract_delivery_score(delivery_estimate):
            if pd.isna(delivery_estimate):
                return 1
            # Extract number from delivery estimate
            numbers = re.findall(r'\d+', str(delivery_estimate))
            if numbers:
                days = int(numbers[0])
                if 'minute' in str(delivery_estimate).lower():
                    return 10  # Fast delivery
                elif days <= 2:
                    return 8   # Very fast
                elif days <= 5:
                    return 5   # Fast
                else:
                    return 2   # Slow
            return 1
        
        self.realtime_product_info['delivery_speed_score'] = (
            self.realtime_product_info['delivery_estimate'].apply(extract_delivery_score)
        )
        
        print(f"Processed {len(self.realtime_product_info)} realtime product records")
        return self.realtime_product_info
    
    def process_session_log(self) -> pd.DataFrame:
        """Process session log data."""
        print("Processing session log...")
        
        # Parse timestamp
        self.session_log['timestamp'] = pd.to_datetime(self.session_log['timestamp'])
        
        # Sort by session_id and timestamp
        self.session_log = self.session_log.sort_values(['session_id', 'timestamp'])
        
        print(f"Processed {len(self.session_log)} session records")
        return self.session_log
    
    def process_ner_dataset(self) -> pd.DataFrame:
        """Process NER dataset."""
        print("Processing NER dataset...")
        
        # Group by query_id to create training examples
        ner_examples = []
        current_query_id = None
        current_tokens = []
        current_tags = []
        
        for _, row in self.ner_dataset.iterrows():
            if row['query_id'] != current_query_id:
                if current_tokens:
                    ner_examples.append({
                        'query_id': current_query_id,
                        'tokens': current_tokens,
                        'tags': current_tags
                    })
                current_query_id = row['query_id']
                current_tokens = [row['token']]
                current_tags = [row['tag']]
            else:
                current_tokens.append(row['token'])
                current_tags.append(row['tag'])
        
        # Add the last example
        if current_tokens:
            ner_examples.append({
                'query_id': current_query_id,
                'tokens': current_tokens,
                'tags': current_tags
            })
        
        self.ner_dataset_processed = pd.DataFrame(ner_examples)
        print(f"Processed {len(self.ner_dataset_processed)} NER examples")
        return self.ner_dataset_processed
    
    def run_all_preprocessing(self):
        """Run all preprocessing steps."""
        print("Starting data preprocessing...")
        
        # Load datasets
        self.load_all_datasets()
        
        # Process each dataset
        self.process_product_catalog()
        self.process_user_queries()
        self.process_realtime_product_info()
        self.process_session_log()
        self.process_ner_dataset()
        
        print("Data preprocessing completed successfully!")
        return True
    
    def get_processed_data(self) -> Dict:
        """Return all processed data."""
        return {
            'product_catalog': self.product_catalog,
            'user_queries': self.user_queries,
            'realtime_product_info': self.realtime_product_info,
            'session_log': self.session_log,
            'ner_dataset': self.ner_dataset_processed,
            'major_categories': self.major_categories,
            'locations': self.locations
        }

# Test the preprocessing
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.run_all_preprocessing()
    
    # Test the processed data
    data = preprocessor.get_processed_data()
    
    print("\n=== Preprocessing Test Results ===")
    print(f"Product catalog shape: {data['product_catalog'].shape}")
    print(f"User queries shape: {data['user_queries'].shape}")
    print(f"Realtime product info shape: {data['realtime_product_info'].shape}")
    print(f"Session log shape: {data['session_log'].shape}")
    print(f"NER dataset shape: {data['ner_dataset'].shape}")
    
    # Test specific features
    print(f"\nCombined text example: {data['product_catalog']['combined_text'].iloc[0][:100]}...")
    print(f"Delivery speed score example: {data['realtime_product_info']['delivery_speed_score'].iloc[0]}")
    print(f"Major categories: {data['major_categories']}")
    print(f"Locations: {data['locations']}") 