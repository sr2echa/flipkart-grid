#!/usr/bin/env python3
"""
Build FAISS Index for Search System
===================================

Creates the required FAISS index files for the search system.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any

def check_dependencies():
    """Check if required packages are available."""
    required_packages = {
        'sentence_transformers': 'sentence-transformers',
        'faiss': 'faiss-cpu',
        'numpy': 'numpy',
        'pandas': 'pandas'
    }
    
    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            if package == 'faiss':
                import faiss
                print(f"✅ {package} is available")
            elif package == 'sentence_transformers':
                from sentence_transformers import SentenceTransformer
                print(f"✅ {package} is available")
            else:
                __import__(package)
                print(f"✅ {package} is available")
        except ImportError:
            print(f"❌ {package} is missing. Please install: pip install {pip_name}")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\n🔧 Run this command to install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    return True

def prepare_text_for_embedding(row: pd.Series) -> str:
    """Prepare text for embedding generation."""
    text_parts = []
    
    # Add title
    if pd.notna(row.get('title')):
        text_parts.append(str(row['title']))
    
    # Add brand
    if pd.notna(row.get('brand')) and str(row['brand']) != 'Unknown Brand':
        text_parts.append(f"Brand: {row['brand']}")
    
    # Add category
    if pd.notna(row.get('category')) and str(row['category']) != 'General':
        text_parts.append(f"Category: {row['category']}")
    
    # Add description (limited)
    if pd.notna(row.get('description')):
        desc = str(row['description'])[:200]  # Limit description length
        text_parts.append(desc)
    
    # Add tags if available
    if pd.notna(row.get('tags')):
        text_parts.append(f"Tags: {row['tags']}")
    
    return " | ".join(text_parts)

def build_faiss_index(catalog_path: str, output_dir: str = "./searchresultpage/faiss_index", max_products: int = 25000):
    """Build FAISS index from product catalog."""
    
    print("🔧 Building FAISS index for search system...")
    print("=" * 60)
    
    if not check_dependencies():
        return False
    
    # Import after dependency check
    import faiss
    from sentence_transformers import SentenceTransformer
    
    # Load product catalog
    print(f"📊 Loading product catalog: {catalog_path}")
    if not os.path.exists(catalog_path):
        print(f"❌ Catalog file not found: {catalog_path}")
        return False
    
    df = pd.read_csv(catalog_path)
    print(f"✅ Loaded {len(df):,} products")
    
    # Limit products to prevent memory issues
    if len(df) > max_products:
        print(f"📊 Limiting to {max_products:,} products (from {len(df):,})")
        df = df.head(max_products)
    
    # Load SBERT model
    model_name = 'all-MiniLM-L6-v2'
    print(f"🧠 Loading SBERT model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Prepare texts for embedding
    print("📝 Preparing texts for embedding...")
    texts = []
    product_ids = []
    metadata = {}
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"   Processing product {idx+1:,}/{len(df):,}")
        
        text = prepare_text_for_embedding(row)
        product_id = str(row['product_id'])
        
        texts.append(text)
        product_ids.append(product_id)
        
        # Store metadata
        metadata[product_id] = {
            'title': str(row.get('title', 'N/A')),
            'brand': str(row.get('brand', 'N/A')),
            'category': str(row.get('category', 'N/A')),
            'subcategory': str(row.get('subcategory', 'N/A')),
            'price': float(row.get('price', 0)),
            'color': str(row.get('color', 'N/A')),
            'rating': float(row.get('rating', 0)),
            'is_f_assured': bool(row.get('is_f_assured', False)),
            'description': str(row.get('description', 'N/A'))[:500],  # Limit length
            'image_url': str(row.get('image_url', '')),
            'tags': str(row.get('tags', ''))
        }
    
    # Generate embeddings
    print(f"🧠 Generating embeddings for {len(texts):,} products...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    print(f"✅ Generated embeddings with shape: {embeddings.shape}")
    
    # Build FAISS index
    print("🔧 Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
    index.add(embeddings.astype('float32'))
    
    print(f"✅ FAISS index built with {index.ntotal:,} products")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save index files
    print("💾 Saving FAISS index files...")
    
    # Save FAISS index
    index_path = os.path.join(output_dir, "product_index.faiss")
    faiss.write_index(index, index_path)
    print(f"✅ Saved FAISS index: {index_path}")
    
    # Save product ID mapping
    mapping_path = os.path.join(output_dir, "product_id_mapping.json")
    with open(mapping_path, 'w') as f:
        json.dump(product_ids, f, indent=2)
    print(f"✅ Saved product ID mapping: {mapping_path}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "product_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved product metadata: {metadata_path}")
    
    # Save embedding stats
    stats = {
        'total_products': len(product_ids),
        'embedding_dimension': dimension,
        'model_name': model_name,
        'index_type': 'IndexFlatIP',
        'normalized_embeddings': True
    }
    stats_path = os.path.join(output_dir, "embedding_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Saved embedding stats: {stats_path}")
    
    print("=" * 60)
    print("🎉 FAISS index building completed successfully!")
    print(f"📊 Total products indexed: {len(product_ids):,}")
    print(f"📁 Output directory: {output_dir}")
    print("📦 Files created:")
    print("   - product_index.faiss")
    print("   - product_id_mapping.json") 
    print("   - product_metadata.json")
    print("   - embedding_stats.json")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    catalog_path = "dataset/product_catalog_merged.csv"
    success = build_faiss_index(catalog_path)
    
    if success:
        print("\n✅ FAISS index is ready for search system!")
    else:
        print("\n❌ Failed to build FAISS index")