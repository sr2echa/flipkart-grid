"""
Minimal Data Preprocessing for Autosuggest System
"""
import pandas as pd
import os
from typing import Dict

class DataPreprocessor:
    """Simple data preprocessor with minimal dependencies."""
    
    def __init__(self):
        # Use robust relative path
        current_file_dir = os.path.dirname(os.path.abspath(__file__))  # frontend/
        project_root = os.path.dirname(current_file_dir)  # project root
        self.data_dir = os.path.join(project_root, "dataset")
        
    def run_all_preprocessing(self):
        """Run basic preprocessing."""
        print("📊 Running minimal data preprocessing...")
        
        # Load basic datasets
        self.user_queries = self._load_user_queries()
        self.product_catalog = self._load_product_catalog()
        
        print("✅ Data preprocessing completed!")
    
    def _load_user_queries(self) -> pd.DataFrame:
        """Load user queries from available files."""
        possible_files = [
            'user_queries_enhanced_v2.csv',
            'user_queries_enhanced.csv', 
            'user_queries.csv',
            'session_log_enhanced_v2.csv',
            'session_log.csv'
        ]
        
        for filename in possible_files:
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    print(f"   -> Loaded user queries from {filename}: {len(df)} rows")
                    return df
                except Exception as e:
                    print(f"   -> Failed to load {filename}: {e}")
                    continue
        
        print("   -> No user queries file found, creating empty DataFrame")
        return pd.DataFrame()
    
    def _load_product_catalog(self) -> pd.DataFrame:
        """Load product catalog from available files."""
        possible_files = [
            'product_catalog.csv',
            'flipkart_com-ecommerce_sample.csv',
            'realtime_product_info.csv'
        ]
        
        for filename in possible_files:
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    print(f"   -> Loaded product catalog from {filename}: {len(df)} rows")
                    return df
                except Exception as e:
                    print(f"   -> Failed to load {filename}: {e}")
                    continue
        
        print("   -> No product catalog file found, creating empty DataFrame")
        return pd.DataFrame()
    
    def get_processed_data(self) -> Dict:
        """Get processed data for autosuggest system."""
        return {
            'user_queries': self.user_queries,
            'product_catalog': self.product_catalog,
            'locations': ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata'],
            'major_categories': ['Electronics', 'Fashion', 'Home', 'Sports', 'Books']
        }