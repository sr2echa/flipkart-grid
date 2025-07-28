# end_to_end_search.py
import pandas as pd
import numpy as np
import requests
import time
from typing import List, Dict

def _to_python_type(val):
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    return val

class ProductRanker:
    """
    Enriches candidate products with real-time data and ranks them
    based on a rule-based scoring formula.
    """
    def __init__(self, realtime_data_path: str):
        try:
            self.realtime_df = pd.read_csv(realtime_data_path)
            print("✅ Ranker Initialized: Successfully loaded real-time product data.")
        except FileNotFoundError:
            print(f"❌ ERROR: Real-time data file not found at '{realtime_data_path}'.")
            raise

    def rank_products(self, candidates: List[Dict], context: Dict) -> List[Dict]:
        enriched_candidates = self._enrich_features(candidates, context)
        for product in enriched_candidates:
            product['ranking_score'] = self._calculate_score(product)
        ranked_results = sorted(enriched_candidates, key=lambda p: p['ranking_score'], reverse=True)
        # Convert all numpy types to Python types for FastAPI serialization
        for product in ranked_results:
            for k, v in product.items():
                product[k] = _to_python_type(v)
        return ranked_results

    def _enrich_features(self, candidates: List[Dict], context: Dict) -> List[Dict]:
        user_location = context.get('location')
        if not user_location:
            raise ValueError("User location not found in context.")
        location_data = self.realtime_df[self.realtime_df['location'] == user_location].set_index('product_id')
        if location_data.empty:
            print(f"⚠️ No real-time data found for location: {user_location}. All products will be marked unavailable.")
        enriched_list = []
        for product in candidates:
            pid = product['product_id']
            product_details = {
                'is_available': False, 'rating': 0, 'review_count': 0,
                'delivery_speed_score': 0, 'offer_score': 0.0, **product
            }
            if pid in location_data.index:
                info = location_data.loc[pid]
                is_available = info.get('stock_status', 'Out of Stock').lower() in ['in stock', 'low stock']
                delivery_score = 0
                if is_available:
                    est = info.get('delivery_estimate', '')
                    if "1 day" in est: delivery_score = 10
                    elif "2 days" in est: delivery_score = 5
                    else: delivery_score = 1
                offer_map = {'High': 0.9, 'Medium': 0.5, 'Low': 0.1}
                offer_score = offer_map.get(info.get('offer_strength', 'None'), 0.0)
                product_details.update({
                    'is_available': is_available,
                    'rating': info.get('rating', 0),
                    'review_count': info.get('review_count', 0),
                    'delivery_speed_score': delivery_score,
                    'offer_score': offer_score
                })
            enriched_list.append(product_details)
        return enriched_list

    def _calculate_score(self, product: Dict) -> float:
        if not product['is_available']:
            return 0.0
        
        W_SEMANTIC = 0.40
        W_RATING = 0.25
        W_REVIEWS = 0.15
        W_DELIVERY = 0.10
        W_OFFER = 0.10
        
        score = (
            (product.get('similarity_score', 0) * 10) * W_SEMANTIC +
            ((product.get('rating', 0) / 5) * 10) * W_RATING +
            (np.log1p(product.get('review_count', 0))) * W_REVIEWS +
            (product.get('delivery_speed_score', 0)) * W_DELIVERY +
            (product.get('offer_score', 0) * 10) * W_OFFER
        )
        return round(score, 4)

def run_full_search_process(query: str, context: Dict, ranker: ProductRanker, api_url: str):
    """
    Executes the full end-to-end search and ranking workflow.
    """
    print(f"\n{'='*25}\nExecuting search for: '{query}'\n{'='*25}")

    # --- Component 2: Semantic Retrieval ---
    print("Step 1: Calling Search API to get initial 100 candidates...")
    try:
        response = requests.post(api_url, json={"query": query, "top_k": 100}, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        candidates = response.json()
        print(f"✅ Received {len(candidates)} candidates from the API.")
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Could not connect to the search API. Is it running? Details: {e}")
        return

    # --- Component 3 & Ranking ---
    print("\nStep 2: Enriching features and re-ranking candidates...")
    start_time = time.time()
    final_ranked_list = ranker.rank_products(candidates, context)
    duration = time.time() - start_time
    print(f"✅ Enrichment and ranking finished in {duration:.4f} seconds.")

    # --- Final Output ---
    print("\n--- Final Top 5 Ranked Products ---")
    for rank, product in enumerate(final_ranked_list[:5], 1):
        # Only show available products in the final result
        if product['is_available']:
            print(f"  #{rank}: {product.get('title', product['product_id'])} (Final Score: {product['ranking_score']:.2f})")
            print(f"     (Initial Similarity: {product['similarity_score']:.2f}, Rating: {product['rating']}, Delivery: {product['delivery_speed_score']})")


# --- Main execution block ---
if __name__ == "__main__":
    API_URL = "http://127.0.0.1:8000/search/"
    REALTIME_DATA_PATH = "R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\dataset\\realtime_product_info.csv"
    
    # 1. Initialize the Ranker
    try:
        product_ranker = ProductRanker(realtime_data_path=REALTIME_DATA_PATH)
    except Exception:
        # Error is already printed in the constructor
        exit()

    # 2. Simulate the user context from the frontend
    user_context = {'location': 'Surat'}

    # 3. Define test queries
    test_queries = [
        "running shoes for men",
        "summer pants",
        "budget phone with good camera"
    ]

    # 4. Run the full process for each query
    for test_query in test_queries:
        run_full_search_process(test_query, user_context, product_ranker, API_URL)