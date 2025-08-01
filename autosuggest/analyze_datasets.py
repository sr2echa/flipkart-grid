"""
Comprehensive Dataset Analysis for Autosuggest Enhancement
"""
import pandas as pd
import numpy as np
from collections import Counter
import re

def analyze_all_datasets():
    """Analyze all available datasets to understand data quality and structure."""
    
    print("🔍 COMPREHENSIVE DATASET ANALYSIS")
    print("=" * 50)
    
    # 1. Analyze Product Catalog
    print("\n📦 PRODUCT CATALOG ANALYSIS")
    try:
        # Read in chunks to handle large file
        chunk_list = []
        chunk_size = 1000
        for chunk in pd.read_csv('../dataset/product_catalog.csv', chunksize=chunk_size):
            chunk_list.append(chunk)
            if len(chunk_list) >= 5:  # Read first 5000 rows
                break
        
        products_df = pd.concat(chunk_list, ignore_index=True)
        print(f"✅ Sample size: {len(products_df)} products")
        print(f"📊 Columns: {list(products_df.columns)}")
        
        # Analyze brands
        if 'brand' in products_df.columns:
            brands = products_df['brand'].dropna().unique()
            print(f"🏷️ Unique brands: {len(brands)}")
            print(f"🔝 Top brands: {list(brands[:10])}")
            
            # Check for real brands
            real_brands = []
            brand_patterns = ['samsung', 'xiaomi', 'mi', 'nike', 'adidas', 'apple', 'lenovo', 'hp', 'dell']
            for brand in brands:
                if isinstance(brand, str):
                    for pattern in brand_patterns:
                        if pattern.lower() in brand.lower():
                            real_brands.append(brand)
                            break
            print(f"🎯 Real brands found: {real_brands}")
        
        # Analyze titles for potential queries
        if 'title' in products_df.columns:
            titles_sample = products_df['title'].dropna().head(20).tolist()
            print(f"\n📝 Sample product titles:")
            for i, title in enumerate(titles_sample[:10], 1):
                print(f"   {i}. {title}")
        
        # Analyze categories
        if 'category' in products_df.columns:
            categories = products_df['category'].dropna().value_counts().head(10)
            print(f"\n📂 Top categories:\n{categories}")
            
    except Exception as e:
        print(f"❌ Error analyzing product catalog: {e}")
    
    # 2. Analyze User Queries
    print("\n🔍 USER QUERIES ANALYSIS")
    try:
        queries_df = pd.read_csv('../dataset/user_queries_comprehensive.csv')
        print(f"✅ Total queries: {len(queries_df)}")
        print(f"📊 Columns: {list(queries_df.columns)}")
        
        # Analyze query patterns
        if 'corrected_query' in queries_df.columns:
            top_queries = queries_df['corrected_query'].value_counts().head(10)
            print(f"🔝 Top queries:\n{top_queries}")
            
            # Analyze query lengths
            query_lengths = queries_df['corrected_query'].str.len()
            print(f"📏 Query length stats:")
            print(f"   Average: {query_lengths.mean():.1f} chars")
            print(f"   Min: {query_lengths.min()}, Max: {query_lengths.max()}")
            
            # Look for brands in queries
            brand_queries = []
            brand_patterns = ['samsung', 'xiaomi', 'xiomi', 'nike', 'adidas', 'apple', 'iphone']
            for pattern in brand_patterns:
                matching = queries_df[queries_df['corrected_query'].str.contains(pattern, case=False, na=False)]
                if len(matching) > 0:
                    brand_queries.extend(matching['corrected_query'].tolist())
            
            print(f"🏷️ Brand-related queries found: {len(brand_queries)}")
            print(f"   Examples: {brand_queries[:5]}")
            
    except Exception as e:
        print(f"❌ Error analyzing user queries: {e}")
    
    # 3. Analyze Session Log
    print("\n📊 SESSION LOG ANALYSIS")
    try:
        session_df = pd.read_csv('../dataset/session_log_enhanced_v2.csv')
        print(f"✅ Total sessions: {len(session_df)}")
        print(f"📊 Columns: {list(session_df.columns)}")
        
        if 'query' in session_df.columns:
            unique_session_queries = session_df['query'].nunique()
            print(f"🔍 Unique session queries: {unique_session_queries}")
            
            # Sample queries from sessions
            sample_queries = session_df['query'].dropna().head(10).tolist()
            print(f"📝 Sample session queries: {sample_queries}")
            
    except Exception as e:
        print(f"❌ Error analyzing session log: {e}")
    
    # 4. Check for typos and corrections needed
    print("\n🔧 TYPO ANALYSIS & CORRECTION OPPORTUNITIES")
    try:
        # Common typos that should be corrected
        typo_patterns = {
            'xiomi': 'xiaomi',
            'samsng': 'samsung',
            'samung': 'samsung', 
            'jersy': 'jersey',
            'jesery': 'jersey',
            'lapto': 'laptop',
            'mobil': 'mobile',
            'headphone': 'headphones'
        }
        
        found_typos = []
        for typo, correct in typo_patterns.items():
            # Check in user queries
            if 'queries_df' in locals():
                typo_matches = queries_df[queries_df['corrected_query'].str.contains(typo, case=False, na=False)]
                if len(typo_matches) > 0:
                    found_typos.append((typo, correct, len(typo_matches)))
        
        if found_typos:
            print("🚨 Typos found that need correction:")
            for typo, correct, count in found_typos:
                print(f"   '{typo}' → '{correct}' ({count} instances)")
        else:
            print("✅ No common typos found in current dataset")
            
    except Exception as e:
        print(f"❌ Error in typo analysis: {e}")
    
    return {
        'products_sample': products_df if 'products_df' in locals() else None,
        'queries': queries_df if 'queries_df' in locals() else None,
        'recommendations': generate_recommendations()
    }

def generate_recommendations():
    """Generate recommendations for improving the autosuggest system."""
    
    recommendations = [
        "🎯 RECOMMENDATIONS FOR IMPROVEMENT:",
        "",
        "1. DATA ENHANCEMENT:",
        "   - Extract real product titles from catalog for suggestions",
        "   - Create comprehensive brand-product mappings",
        "   - Add spelling correction dataset with common typos",
        "",
        "2. SEMANTIC IMPROVEMENTS:",
        "   - Use SBERT to embed product titles for semantic matching",
        "   - Implement FAISS index for fast similarity search", 
        "   - Add context-aware ranking based on categories",
        "",
        "3. NLTK INTEGRATION:",
        "   - Use NLTK for stemming and tokenization",
        "   - Implement spell correction using NLTK",
        "   - Add synonyms and related terms expansion",
        "",
        "4. QUERY UNDERSTANDING:",
        "   - Map partial queries to product categories",
        "   - Use NER for brand/product entity recognition",
        "   - Implement intent classification",
        "",
        "5. PERSONALIZATION:",
        "   - Use session history for personalized suggestions",
        "   - Implement location-based product filtering",
        "   - Add persona-based query expansion"
    ]
    
    return "\n".join(recommendations)

if __name__ == "__main__":
    results = analyze_all_datasets()
    print("\n" + "="*50)
    print(results['recommendations'])