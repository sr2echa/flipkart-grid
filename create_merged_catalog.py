#!/usr/bin/env python3
"""
Create Merged Product Catalog for Search System
=============================================

Creates the required product_catalog_merged.csv file from existing product_catalog.csv
"""

import pandas as pd
import os

def create_merged_catalog():
    """Create a merged product catalog CSV file."""
    
    # Configuration - use relative paths that work from project root
    CATALOG_PATH = "dataset/product_catalog.csv"
    OUTPUT_PATH = "dataset/product_catalog_merged.csv"
    
    print("🔄 Creating merged product catalog for search system...")
    
    if not os.path.exists(CATALOG_PATH):
        print(f"❌ Catalog file not found: {CATALOG_PATH}")
        return False
    
    print(f"📊 Loading catalog: {CATALOG_PATH}")
    
    try:
        # Load the main catalog
        df = pd.read_csv(CATALOG_PATH)
        print(f"✅ Loaded {len(df):,} products from {CATALOG_PATH}")
        
        # Add source_catalog column if it doesn't exist
        if 'source_catalog' not in df.columns:
            df['source_catalog'] = 'product_catalog'
        
        # Ensure required columns exist with defaults
        required_columns = {
            'product_id': lambda x: x if 'product_id' in df.columns else range(len(df)),
            'title': lambda x: x if 'title' in df.columns else df.get('product_name', 'Unknown Product'),
            'brand': lambda x: x if 'brand' in df.columns else 'Unknown Brand',
            'category': lambda x: x if 'category' in df.columns else df.get('product_category_tree', 'General'),
            'price': lambda x: x if 'price' in df.columns else df.get('retail_price', 0),
            'rating': lambda x: x if 'rating' in df.columns else 0,
            'description': lambda x: x if 'description' in df.columns else df.get('description', '')
        }
        
        # Apply column mappings
        for col, mapper in required_columns.items():
            if col not in df.columns:
                if col == 'product_id' and 'product_id' not in df.columns:
                    df['product_id'] = range(len(df))
                elif col == 'title' and 'product_name' in df.columns:
                    df['title'] = df['product_name']
                elif col == 'category' and 'product_category_tree' in df.columns:
                    df['category'] = df['product_category_tree']
                elif col == 'price' and 'retail_price' in df.columns:
                    df['price'] = pd.to_numeric(df['retail_price'], errors='coerce').fillna(0)
                else:
                    df[col] = 'Unknown' if col in ['brand', 'title', 'category', 'description'] else 0
        
        # Clean up price column
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
        
        # Remove duplicates based on product_id if it exists
        initial_count = len(df)
        if 'product_id' in df.columns:
            df = df.drop_duplicates(subset=['product_id'], keep='first')
            final_count = len(df)
            
            if initial_count != final_count:
                print(f"🧹 Removed {initial_count - final_count} duplicate products")
        else:
            final_count = len(df)
        
        # Save merged catalog
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"✅ Successfully created merged catalog: {final_count:,} unique products")
        print(f"📂 Saved to: {OUTPUT_PATH}")
        
        # Show column info
        print(f"\n📋 Catalog Columns:")
        for col in df.columns:
            non_null = df[col].notna().sum()
            print(f"   {col}: {non_null:,} non-null values")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create merged catalog: {e}")
        return False

if __name__ == "__main__":
    success = create_merged_catalog()
    if success:
        print("\n🎉 Merged catalog created successfully!")
    else:
        print("\n❌ Failed to create merged catalog")