#!/usr/bin/env python3
"""
Create Merged Product Catalog
=============================

This script creates a merged product catalog CSV file from the combined data.
"""

import pandas as pd
import os

def create_merged_catalog():
    """Create a merged product catalog CSV file."""
    
    # Configuration
    CATALOG_PATHS = [
        "R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\dataset\\product_catalog.csv",
        "R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\dataset\\product_catalog2.csv"
    ]
    OUTPUT_PATH = "R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\dataset\\product_catalog_merged.csv"
    
    print("🔄 Creating merged product catalog...")
    
    merged_data = []
    
    for i, catalog_path in enumerate(CATALOG_PATHS, 1):
        if not os.path.exists(catalog_path):
            print(f"⚠️ Catalog file not found: {catalog_path}")
            continue
            
        print(f"📊 Loading catalog {i}/{len(CATALOG_PATHS)}: {catalog_path}")
        
        try:
            df = pd.read_csv(catalog_path)
            print(f"✅ Loaded {len(df):,} products from {catalog_path}")
            merged_data.append(df)
            
        except Exception as e:
            print(f"❌ Failed to load {catalog_path}: {e}")
            continue
    
    if not merged_data:
        print("❌ No valid catalogs found!")
        return
    
    # Concatenate all dataframes
    merged_df = pd.concat(merged_data, ignore_index=True)
    
    # Remove duplicates based on product_id
    initial_count = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['product_id'], keep='first')
    final_count = len(merged_df)
    
    if initial_count != final_count:
        print(f"🧹 Removed {initial_count - final_count} duplicate products")
    
    # Save merged catalog
    merged_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Successfully created merged catalog: {final_count:,} unique products")
    print(f"📂 Saved to: {OUTPUT_PATH}")
    
    # Show breakdown
    print("\n📋 Catalog Breakdown:")
    if 'source_catalog' in merged_df.columns:
        for catalog, count in merged_df['source_catalog'].value_counts().items():
            print(f"   {catalog}: {count:,} products")
    else:
        print(f"   Total: {final_count:,} products")

if __name__ == "__main__":
    create_merged_catalog() 