#!/usr/bin/env python3
"""
Merge Product Catalogs and Rebuild FAISS Index
==============================================

This script merges product_catalog.csv and product_catalog2.csv,
then rebuilds the FAISS index with the combined dataset.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductCatalogMerger:
    """Merge multiple product catalogs and rebuild FAISS index."""
    
    def __init__(self, 
                 catalog_paths: List[str],
                 output_dir: str = "./faiss_index",
                 model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the merger.
        
        Args:
            catalog_paths: List of paths to product catalog CSV files
            output_dir: Directory to save FAISS index files
            model_name: SBERT model name for embeddings
        """
        self.catalog_paths = catalog_paths
        self.output_dir = output_dir
        self.model_name = model_name
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("🔧 Product Catalog Merger Initializing")
        logger.info("=" * 60)
        logger.info(f"📂 Input catalogs: {len(catalog_paths)} files")
        logger.info(f"📂 Output directory: {output_dir}")
        logger.info(f"🧠 SBERT model: {model_name}")
        logger.info("=" * 60)
    
    def merge_catalogs(self) -> pd.DataFrame:
        """Merge all product catalogs into a single DataFrame."""
        logger.info("🔄 Merging product catalogs...")
        
        merged_data = []
        total_products = 0
        
        for i, catalog_path in enumerate(self.catalog_paths, 1):
            if not os.path.exists(catalog_path):
                logger.warning(f"⚠️ Catalog file not found: {catalog_path}")
                continue
                
            logger.info(f"📊 Loading catalog {i}/{len(self.catalog_paths)}: {catalog_path}")
            
            try:
                # Load catalog
                df = pd.read_csv(catalog_path)
                logger.info(f"✅ Loaded {len(df):,} products from {catalog_path}")
                
                # Add source information
                df['source_catalog'] = f"catalog_{i}"
                
                merged_data.append(df)
                total_products += len(df)
                
            except Exception as e:
                logger.error(f"❌ Failed to load {catalog_path}: {e}")
                continue
        
        if not merged_data:
            raise RuntimeError("No valid catalogs found!")
        
        # Concatenate all dataframes
        merged_df = pd.concat(merged_data, ignore_index=True)
        
        # Remove duplicates based on product_id
        initial_count = len(merged_df)
        merged_df = merged_df.drop_duplicates(subset=['product_id'], keep='first')
        final_count = len(merged_df)
        
        if initial_count != final_count:
            logger.info(f"🧹 Removed {initial_count - final_count} duplicate products")
        
        logger.info(f"✅ Successfully merged catalogs: {final_count:,} unique products")
        return merged_df
    
    def prepare_text_for_embedding(self, row: pd.Series) -> str:
        """Prepare text for embedding by combining relevant fields."""
        text_parts = []
        
        # Add title
        if pd.notna(row.get('title')):
            text_parts.append(str(row['title']))
        
        # Add brand
        if pd.notna(row.get('brand')):
            text_parts.append(str(row['brand']))
        
        # Add category
        if pd.notna(row.get('category')):
            text_parts.append(str(row['category']))
        
        # Add subcategory
        if pd.notna(row.get('subcategory')):
            text_parts.append(str(row['subcategory']))
        
        # Add description (truncate if too long)
        if pd.notna(row.get('description')):
            desc = str(row['description'])
            if len(desc) > 500:  # Limit description length
                desc = desc[:500] + "..."
            text_parts.append(desc)
        
        # Add specifications (truncate if too long)
        if pd.notna(row.get('specifications')):
            specs = str(row['specifications'])
            if len(specs) > 300:  # Limit specs length
                specs = specs[:300] + "..."
            text_parts.append(specs)
        
        # Add color
        if pd.notna(row.get('color')):
            text_parts.append(str(row['color']))
        
        # Add tags
        if pd.notna(row.get('tags')):
            text_parts.append(str(row['tags']))
        
        return " | ".join(text_parts)
    
    def build_faiss_index(self, merged_df: pd.DataFrame, max_products: int = 50000):
        """Build FAISS index from merged product catalog."""
        logger.info("🔧 Building FAISS index...")
        
        # Limit products if needed
        if len(merged_df) > max_products:
            logger.info(f"📊 Limiting to {max_products:,} products (from {len(merged_df):,})")
            merged_df = merged_df.head(max_products)
        
        # Load SBERT model
        logger.info(f"🧠 Loading SBERT model: {self.model_name}")
        model = SentenceTransformer(self.model_name)
        
        # Prepare texts for embedding
        logger.info("📝 Preparing texts for embedding...")
        texts = []
        product_ids = []
        metadata = {}
        
        for idx, row in merged_df.iterrows():
            text = self.prepare_text_for_embedding(row)
            product_id = str(row['product_id'])
            
            texts.append(text)
            product_ids.append(product_id)
            
            # Store metadata
            metadata[product_id] = {
                'title': row.get('title', 'N/A'),
                'brand': row.get('brand', 'N/A'),
                'category': row.get('category', 'N/A'),
                'subcategory': row.get('subcategory', 'N/A'),
                'price': float(row.get('price', 0)),
                'color': row.get('color', 'N/A'),
                'source_catalog': row.get('source_catalog', 'unknown')
            }
        
        # Generate embeddings
        logger.info(f"🧠 Generating embeddings for {len(texts):,} products...")
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        
        # Build FAISS index
        logger.info("🔧 Building FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
        index.add(embeddings.astype('float32'))
        
        # Save index components
        logger.info("💾 Saving FAISS index components...")
        
        # Save FAISS index
        index_path = os.path.join(self.output_dir, "product_index.faiss")
        faiss.write_index(index, index_path)
        logger.info(f"✅ FAISS index saved: {index_path}")
        
        # Save product ID mapping
        mapping_path = os.path.join(self.output_dir, "product_id_mapping.json")
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(product_ids, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Product ID mapping saved: {mapping_path}")
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, "product_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"✅ Product metadata saved: {metadata_path}")
        
        # Save build statistics
        stats = {
            'total_products': len(product_ids),
            'embedding_dimension': dimension,
            'model_name': self.model_name,
            'index_type': 'FlatIP',
            'build_timestamp': pd.Timestamp.now().isoformat(),
            'source_catalogs': list(merged_df['source_catalog'].unique()),
            'catalog_sizes': merged_df['source_catalog'].value_counts().to_dict()
        }
        
        stats_path = os.path.join(self.output_dir, "embedding_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Build statistics saved: {stats_path}")
        
        logger.info("=" * 60)
        logger.info("✅ FAISS Index Build Complete!")
        logger.info(f"📊 Total products: {len(product_ids):,}")
        logger.info(f"🧠 Embedding dimension: {dimension}")
        logger.info(f"📂 Index size: {os.path.getsize(index_path) / (1024*1024):.1f} MB")
        logger.info("=" * 60)
        
        return index, product_ids, metadata, stats

def main():
    """Main function to merge catalogs and rebuild index."""
    print("🛒 Grid 7.0 - Product Catalog Merger")
    print("=" * 60)
    
    # Configuration
    CATALOG_PATHS = [
        "R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\dataset\\product_catalog.csv",
        "R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\dataset\\product_catalog2.csv"
    ]
    OUTPUT_DIR = "R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\searchresultpage\\faiss_index"
    MODEL_NAME = 'all-MiniLM-L6-v2'
    MAX_PRODUCTS = 50000  # Limit to prevent memory issues
    
    try:
        # Initialize merger
        merger = ProductCatalogMerger(
            catalog_paths=CATALOG_PATHS,
            output_dir=OUTPUT_DIR,
            model_name=MODEL_NAME
        )
        
        # Merge catalogs
        merged_df = merger.merge_catalogs()
        
        # Build FAISS index
        index, product_ids, metadata, stats = merger.build_faiss_index(merged_df, MAX_PRODUCTS)
        
        print("\n✅ Successfully merged catalogs and rebuilt FAISS index!")
        print(f"📊 Total products in index: {len(product_ids):,}")
        print(f"📂 Index saved to: {OUTPUT_DIR}")
        
        # Show catalog breakdown
        print("\n📋 Catalog Breakdown:")
        for catalog, count in stats['catalog_sizes'].items():
            print(f"   {catalog}: {count:,} products")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during merge: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 