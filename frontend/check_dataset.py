#!/usr/bin/env python3
"""Check dataset structure for persona information."""

import pandas as pd
import sys

def check_dataset():
    try:
        print("📊 Loading dataset...")
        df = pd.read_csv('../dataset/query_product_training_features_only.csv', nrows=5)
        
        print(f"✅ Dataset loaded successfully")
        print(f"📋 Total columns: {len(df.columns)}")
        
        print("\nFirst 10 columns:")
        for i, col in enumerate(df.columns[:10]):
            print(f"  {i}: {col}")
        
        print("\nColumns containing 'persona', 'user', 'tag', or 'category':")
        relevant_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['persona', 'user', 'tag', 'category'])]
        for col in relevant_cols:
            print(f"  - {col}")
        
        print(f"\nSample data (first 5 columns):")
        print(df.iloc[:, :5])
        
        if relevant_cols:
            print(f"\nSample data from relevant columns:")
            print(df[relevant_cols[:3]].head() if len(relevant_cols) >= 3 else df[relevant_cols].head())
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

if __name__ == "__main__":
    check_dataset()