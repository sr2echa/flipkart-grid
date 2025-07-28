"""
Grid 7.0 - FAISS Index Builder for Kaggle (Robust Version)
==========================================================

This script builds the FAISS index from product_catalog.csv with enhanced error handling
and compatibility checks for different environments.
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
import time
import logging
from typing import List, Dict, Any
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check and report on dependency versions"""
    logger.info("🔍 Checking dependencies...")
    
    required_packages = {
        'sentence_transformers': 'sentence-transformers',
        'faiss': 'faiss-cpu',
        'pandas': 'pandas',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            logger.info(f"✅ {package} is available")
        except ImportError:
            logger.error(f"❌ {package} is missing")
            missing_packages.append(pip_name)
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.error("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def load_sentence_transformer_safely(model_name="all-MiniLM-L6-v2"):
    """Safely load SentenceTransformer with fallback options"""
    
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"📥 Loading SBERT model: {model_name}")
        model = SentenceTransformer(model_name)
        logger.info("✅ SBERT model loaded successfully")
        return model
        
    except ImportError as e:
        logger.error(f"❌ Failed to import sentence_transformers: {e}")
        logger.error("Fix: pip install sentence-transformers")
        raise
        
    except Exception as e:
        logger.error(f"❌ Failed to load model {model_name}: {e}")
        
        # Try alternative models
        fallback_models = [
            "all-MiniLM-L12-v2",
            "paraphrase-MiniLM-L6-v2",
            "distilbert-base-nli-stsb-mean-tokens"
        ]
        
        for fallback in fallback_models:
            try:
                logger.info(f"🔄 Trying fallback model: {fallback}")
                model = SentenceTransformer(fallback)
                logger.info(f"✅ Fallback model {fallback} loaded successfully")
                return model
            except Exception as fallback_error:
                logger.warning(f"❌ Fallback {fallback} also failed: {fallback_error}")
                continue
        
        raise Exception("All models failed to load. Check your internet connection and package versions.")

def load_faiss_safely():
    """Safely load FAISS with fallback options"""
    
    try:
        import faiss
        logger.info("✅ FAISS loaded successfully")
        return faiss
    except ImportError:
        try:
            import faiss_cpu as faiss
            logger.info("✅ FAISS-CPU loaded successfully")
            return faiss
        except ImportError:
            logger.error("❌ Neither faiss nor faiss-cpu found")
            logger.error("Fix: pip install faiss-cpu")
            raise

class RobustKaggleFAISSBuilder:
    """
    Robust FAISS Index Builder with enhanced error handling
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the FAISS builder with error handling"""
        
        logger.info("🎯 Grid 7.0 - Robust Kaggle FAISS Index Builder")
        logger.info("=" * 60)
        
        # Check dependencies first
        if not check_dependencies():
            raise Exception("Required dependencies are missing")
        
        self.model_name = model_name
        self.faiss = load_faiss_safely()
        self.sbert_model = load_sentence_transformer_safely(model_name)
        self.build_stats = {}
    
    def create_rich_embedding_text(self, row: pd.Series) -> str:
        """Create rich descriptive text from all dataset attributes"""
        
        try:
            # Extract and clean all fields safely
            title = str(row.get('title', '')).strip()
            brand = str(row.get('brand', '')).strip()
            category = str(row.get('category', '')).strip()
            subcategory = str(row.get('subcategory', '')).strip()
            description = str(row.get('description', ''))[:300]
            specifications = str(row.get('specifications', ''))[:200]
            color = str(row.get('color', '')).strip()
            tags = str(row.get('tags', '')).strip()
            
            # Handle numeric fields safely
            try:
                price = float(row.get('price', 0))
            except (ValueError, TypeError):
                price = 0
                
            try:
                seller_rating = float(row.get('seller_rating', 0))
            except (ValueError, TypeError):
                seller_rating = 0
            
            # Handle boolean fields safely
            is_f_assured = bool(row.get('is_f_assured', False))
            cod_available = bool(row.get('cod_available', False))
            
            return_policy = str(row.get('return_policy', ''))[:50]
            seller_name = str(row.get('seller_name', '')).strip()
            
            # Build rich embedding text
            embedding_parts = []
            
            # Core product info
            if title and title.lower() not in ['nan', '']:
                embedding_parts.append(title)
                
            if brand and brand.lower() not in ['nan', '']:
                embedding_parts.append(f"by {brand}")
            
            # Category hierarchy
            category_info = []
            if category and category.lower() not in ['nan', '']:
                category_info.append(f"Category: {category}")
            if subcategory and subcategory.lower() not in ['nan', ''] and subcategory != category:
                category_info.append(subcategory)
            if category_info:
                embedding_parts.append(" - ".join(category_info))
            
            # Product attributes
            if color and color.lower() not in ['nan', '']:
                embedding_parts.append(f"Color: {color}")
                
            if price and price > 0:
                embedding_parts.append(f"Price: ₹{price}")
            
            # Rich descriptions
            if description and description.lower() not in ['nan', '']:
                embedding_parts.append(f"Description: {description}")
                
            if specifications and specifications.lower() not in ['nan', '']:
                embedding_parts.append(f"Specifications: {specifications}")
            
            # Service features
            service_features = []
            if is_f_assured:
                service_features.append("Flipkart Assured")
            if cod_available:
                service_features.append("Cash on Delivery")
            if return_policy and return_policy.lower() not in ['nan', '']:
                service_features.append(f"Return: {return_policy}")
            if service_features:
                embedding_parts.append(f"Features: {', '.join(service_features)}")
            
            # Seller info
            if seller_name and seller_name.lower() not in ['nan', '']:
                seller_info = f"Seller: {seller_name}"
                if seller_rating and seller_rating > 0:
                    seller_info += f" (Rating: {seller_rating})"
                embedding_parts.append(seller_info)
            
            # Tags for additional context
            if tags and tags.lower() not in ['nan', '']:
                embedding_parts.append(f"Tags: {tags}")
            
            result = ". ".join(embedding_parts)
            return result if result else title  # Fallback to title if nothing else
            
        except Exception as e:
            logger.warning(f"Error creating embedding text for product: {e}")
            return str(row.get('title', 'Unknown Product'))
    
    def load_and_clean_dataset(self, csv_path: str) -> pd.DataFrame:
        """Load and clean the product catalog dataset with robust error handling"""
        
        logger.info(f"📂 Loading dataset from: {csv_path}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found at: {csv_path}")
        
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding)
                    logger.info(f"✅ Dataset loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise Exception("Could not read CSV with any encoding")
            
            original_count = len(df)
            logger.info(f"📊 Original dataset size: {original_count:,} products")
            
            # Check required columns
            required_cols = ['product_id', 'title']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Data cleaning
            logger.info("🧹 Cleaning dataset...")
            
            # Remove rows without essential fields
            df = df.dropna(subset=['product_id', 'title'])
            df = df[df['title'].astype(str).str.len() > 0]
            
            # Clean text fields safely
            text_columns = ['title', 'brand', 'category', 'subcategory', 'description', 
                           'specifications', 'color', 'return_policy', 'seller_name', 
                           'image_url', 'tags']
            
            for col in text_columns:
                if col in df.columns:
                    df[col] = df[col].fillna('').astype(str)
            
            # Clean numeric fields safely
            if 'price' in df.columns:
                df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
            if 'seller_rating' in df.columns:
                df['seller_rating'] = pd.to_numeric(df['seller_rating'], errors='coerce').fillna(0)
            
            # Clean boolean fields safely
            if 'is_f_assured' in df.columns:
                df['is_f_assured'] = df['is_f_assured'].fillna(False).astype(bool)
            if 'cod_available' in df.columns:
                df['cod_available'] = df['cod_available'].fillna(False).astype(bool)
            
            final_count = len(df)
            logger.info(f"✅ Dataset cleaned. Final size: {final_count:,} products")
            
            if final_count == 0:
                raise ValueError("No valid products remain after cleaning")
            
            # Store statistics
            self.build_stats.update({
                'original_products': original_count,
                'final_products': final_count,
                'unique_categories': df['category'].nunique() if 'category' in df.columns else 0,
                'unique_brands': df['brand'].nunique() if 'brand' in df.columns else 0,
            })
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading dataset: {e}")
            raise
    
    def build_embeddings_batch(self, df: pd.DataFrame, batch_size: int = 32) -> tuple:
        """Generate embeddings with batch processing and error handling"""
        
        logger.info("🔧 Generating product embeddings...")
        
        embedding_texts = []
        product_ids = []
        metadata = {}
        failed_products = 0
        
        # Create embedding texts
        for idx, row in df.iterrows():
            try:
                product_id = row['product_id']
                embedding_text = self.create_rich_embedding_text(row)
                
                if not embedding_text or len(embedding_text.strip()) == 0:
                    failed_products += 1
                    continue
                
                embedding_texts.append(embedding_text)
                product_ids.append(product_id)
                
                # Store metadata safely
                metadata[product_id] = {
                    'title': str(row.get('title', '')),
                    'brand': str(row.get('brand', '')),
                    'category': str(row.get('category', '')),
                    'subcategory': str(row.get('subcategory', '')),
                    'description': str(row.get('description', '')),
                    'specifications': str(row.get('specifications', '')),
                    'price': float(row.get('price', 0)) if pd.notna(row.get('price')) else 0,
                    'color': str(row.get('color', '')),
                    'is_f_assured': bool(row.get('is_f_assured', False)),
                    'cod_available': bool(row.get('cod_available', False)),
                    'return_policy': str(row.get('return_policy', '')),
                    'seller_name': str(row.get('seller_name', '')),
                    'seller_rating': float(row.get('seller_rating', 0)) if pd.notna(row.get('seller_rating')) else 0,
                    'image_url': str(row.get('image_url', '')),
                    'tags': str(row.get('tags', '')),
                    'embedding_text': embedding_text
                }
                
            except Exception as e:
                logger.warning(f"Failed to process product {row.get('product_id', 'unknown')}: {e}")
                failed_products += 1
                continue
        
        if failed_products > 0:
            logger.warning(f"⚠️ Failed to process {failed_products} products")
        
        if not embedding_texts:
            raise ValueError("No valid embedding texts generated")
        
        logger.info(f"📝 Created {len(embedding_texts):,} embedding texts")
        
        # Generate embeddings with error handling
        logger.info("🧠 Generating SBERT embeddings...")
        start_time = time.time()
        
        try:
            embeddings = self.sbert_model.encode(
                embedding_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            embedding_time = time.time() - start_time
            logger.info(f"⚡ Embeddings generated in {embedding_time:.2f} seconds")
            logger.info(f"📐 Embedding shape: {embeddings.shape}")
            
            self.build_stats.update({
                'embedding_dimension': embeddings.shape[1],
                'embedding_generation_time': embedding_time,
                'failed_products': failed_products
            })
            
            return embeddings, product_ids, metadata
            
        except Exception as e:
            logger.error(f"❌ Error generating embeddings: {e}")
            raise
    
    def build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS index with error handling"""
        
        logger.info("🏗️ Building FAISS index...")
        start_time = time.time()
        
        try:
            dimension = embeddings.shape[1]
            index = self.faiss.IndexFlatL2(dimension)
            index.add(embeddings.astype('float32'))
            
            build_time = time.time() - start_time
            logger.info(f"✅ FAISS index built in {build_time:.2f} seconds")
            logger.info(f"📊 Index contains {index.ntotal:,} products")
            
            self.build_stats['faiss_build_time'] = build_time
            return index
            
        except Exception as e:
            logger.error(f"❌ Error building FAISS index: {e}")
            raise
    
    def save_index_files(self, index, product_ids: List[str], 
                        metadata: Dict[str, Any], output_dir: str = "./faiss_output"):
        """Save all index files with error handling"""
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"💾 Saving index files to: {output_dir}")
            
            # Save FAISS index
            index_path = os.path.join(output_dir, "product_index.faiss")
            self.faiss.write_index(index, index_path)
            logger.info(f"✅ FAISS index saved: {index_path}")
            
            # Save product ID mapping
            mapping_path = os.path.join(output_dir, "product_id_mapping.json")
            with open(mapping_path, 'w', encoding='utf-8') as f:
                json.dump(product_ids, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ Product mapping saved: {mapping_path}")
            
            # Save metadata
            metadata_path = os.path.join(output_dir, "product_metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            logger.info(f"✅ Product metadata saved: {metadata_path}")
            
            # Save build statistics
            self.build_stats.update({
                'model_name': self.model_name,
                'build_timestamp': time.time(),
            })
            
            stats_path = os.path.join(output_dir, "embedding_stats.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(self.build_stats, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Build stats saved: {stats_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving files: {e}")
            raise
    
    def build_complete_index(self, csv_path: str, output_dir: str = "./faiss_output"):
        """Complete index building pipeline with comprehensive error handling"""
        
        logger.info("🚀 Starting robust FAISS index building pipeline")
        logger.info("=" * 70)
        
        total_start_time = time.time()
        
        try:
            # Step 1: Load and clean dataset
            df = self.load_and_clean_dataset(csv_path)
            
            # Step 2: Generate embeddings
            embeddings, product_ids, metadata = self.build_embeddings_batch(df)
            
            # Step 3: Build FAISS index
            index = self.build_faiss_index(embeddings)
            
            # Step 4: Save all files
            self.save_index_files(index, product_ids, metadata, output_dir)
            
            total_time = time.time() - total_start_time
            logger.info("=" * 70)
            logger.info(f"🎉 FAISS Index Building Completed Successfully!")
            logger.info(f"⏱️ Total time: {total_time:.2f} seconds")
            logger.info(f"📊 Products indexed: {len(product_ids):,}")
            logger.info(f"📁 Output directory: {output_dir}")
            logger.info("=" * 70)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Critical error in pipeline: {str(e)}")
            logger.error("Please check the error message above and fix the underlying issue")
            return False


def main():
    """Main function with comprehensive error handling"""
    
    print("🎯 Grid 7.0 - Robust Kaggle FAISS Index Builder")
    print("=" * 50)
    
    # Configuration
    CSV_PATH = "R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\dataset\\product_catalog.csv"
    OUTPUT_DIR = "./faiss_index"
    
    try:
        # Check if dataset exists
        if not os.path.exists(CSV_PATH):
            logger.error(f"❌ Dataset not found at: {CSV_PATH}")
            logger.error("📋 Please ensure the path is correct and the file exists")
            return
        
        # Build index
        builder = RobustKaggleFAISSBuilder()
        success = builder.build_complete_index(CSV_PATH, OUTPUT_DIR)
        
        if success:
            print("\n📦 Files ready:")
            print("   - product_index.faiss")
            print("   - product_id_mapping.json")
            print("   - product_metadata.pkl")
            print("   - embedding_stats.json")
            print("\n✅ Index building completed successfully!")
        else:
            print("\n❌ Index building failed. Check the logs above.")
            
    except KeyboardInterrupt:
        logger.info("🛑 Process interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        logger.error("Please check your environment and dependencies")


if __name__ == "__main__":
    main()