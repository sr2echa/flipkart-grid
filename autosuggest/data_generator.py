#!/usr/bin/env python3
"""
Enhanced Data Generator for Flipkart Autosuggest System
Generates synthetic data for testing and training purposes.
"""

import pandas as pd
import numpy as np
import json
import random
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging
from faker import Faker
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedDataGenerator:
    """
    Enhanced data generator for creating synthetic datasets for autosuggest system.
    Generates realistic e-commerce data for testing and training.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the data generator."""
        self.fake = Faker()
        Faker.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        # E-commerce specific data
        self.brands = {
            'Electronics': ['Samsung', 'Apple', 'Xiaomi', 'OnePlus', 'Realme', 'Oppo', 'Vivo', 'Nokia', 'Motorola', 'LG'],
            'Fashion': ['Nike', 'Adidas', 'Puma', 'Zara', 'H&M', 'Levi\'s', 'Tommy Hilfiger', 'Calvin Klein', 'Ray-Ban', 'Fastrack'],
            'Home': ['IKEA', 'Godrej Interio', 'Prestige', 'Philips', 'Bajaj', 'Crompton', 'Blue Star', 'Daikin', 'Whirlpool', 'Bosch']
        }
        
        self.categories = {
            'Electronics': ['Mobiles', 'Laptops', 'Televisions', 'Audio', 'Wearables', 'Cameras', 'Gaming'],
            'Fashion': ['Footwear', 'Apparel', 'Accessories', 'Watches', 'Bags', 'Jewelry'],
            'Home': ['Kitchen Appliances', 'Furniture', 'Cooling', 'Laundry', 'Lighting', 'Storage']
        }
        
        self.price_ranges = {
            'budget': (500, 5000),
            'mid_range': (5000, 25000),
            'premium': (25000, 100000),
            'luxury': (100000, 500000)
        }
        
    def generate_product_catalog(self, num_products: int = 1000) -> pd.DataFrame:
        """Generate synthetic product catalog."""
        logger.info(f"Generating {num_products} products...")
        
        products = []
        for i in range(num_products):
            # Select major category
            major_category = random.choice(list(self.categories.keys()))
            subcategory = random.choice(self.categories[major_category])
            brand = random.choice(self.brands[major_category])
            
            # Generate product details
            product = self._generate_product_details(i + 1, major_category, subcategory, brand)
            products.append(product)
        
        df = pd.DataFrame(products)
        logger.info(f"Generated product catalog with {len(df)} products")
        return df
    
    def _generate_product_details(self, product_id: int, major_category: str, 
                                subcategory: str, brand: str) -> Dict:
        """Generate detailed product information."""
        # Generate title
        title = self._generate_product_title(brand, subcategory, major_category)
        
        # Generate description
        description = self._generate_product_description(title, subcategory)
        
        # Generate specifications
        specifications = self._generate_specifications(subcategory, major_category)
        
        # Generate price and ratings
        price_range = random.choice(list(self.price_ranges.keys()))
        price = random.randint(*self.price_ranges[price_range])
        rating = round(random.uniform(3.5, 5.0), 1)
        num_reviews = random.randint(50, 2000)
        
        return {
            'product_id': product_id,
            'title': title,
            'description': description,
            'category': f"{major_category}>{subcategory}",
            'brand': brand,
            'price': price,
            'rating': rating,
            'num_reviews': num_reviews,
            'specifications': json.dumps(specifications),
            'major_category': major_category
        }
    
    def _generate_product_title(self, brand: str, subcategory: str, major_category: str) -> str:
        """Generate realistic product title."""
        if major_category == 'Electronics':
            if subcategory == 'Mobiles':
                models = ['Galaxy', 'iPhone', 'Redmi', 'Nord', 'Realme', 'Reno', 'V', 'G', 'Edge', 'Pro']
                model = random.choice(models)
                return f"{brand} {model} {random.randint(8, 15)}"
            elif subcategory == 'Laptops':
                series = ['Inspiron', 'XPS', 'Spectre', 'Victus', 'ThinkPad', 'MacBook', 'ZenBook', 'Swift']
                series_name = random.choice(series)
                return f"{brand} {series_name} {random.choice(['Pro', 'Air', 'Gaming', ''])}"
            elif subcategory == 'Televisions':
                sizes = ['32"', '43"', '50"', '55"', '65"', '75"']
                tech = ['LED', 'OLED', 'QLED', '4K', 'Smart TV']
                return f"{brand} {random.choice(sizes)} {random.choice(tech)}"
        elif major_category == 'Fashion':
            if subcategory == 'Footwear':
                styles = ['Air Max', 'Ultraboost', 'RS-X', 'Revolution', 'Grand Court', 'Smash']
                style = random.choice(styles)
                return f"{brand} {style}"
            elif subcategory == 'Apparel':
                types = ['T-Shirt', 'Shirt', 'Jeans', 'Dress', 'Jacket', 'Hoodie']
                type_name = random.choice(types)
                return f"{brand} {type_name}"
        
        # Generic fallback
        return f"{brand} {subcategory} {random.randint(1000, 9999)}"
    
    def _generate_product_description(self, title: str, subcategory: str) -> str:
        """Generate product description."""
        descriptions = [
            f"High-quality {subcategory.lower()} with premium features and excellent performance.",
            f"Stylish and durable {subcategory.lower()} perfect for everyday use.",
            f"Advanced {subcategory.lower()} with cutting-edge technology and modern design.",
            f"Reliable {subcategory.lower()} offering great value for money.",
            f"Premium {subcategory.lower()} designed for comfort and style."
        ]
        return random.choice(descriptions)
    
    def _generate_specifications(self, subcategory: str, major_category: str) -> Dict:
        """Generate product specifications."""
        specs = {}
        
        if major_category == 'Electronics':
            if subcategory == 'Mobiles':
                specs.update({
                    'display': f"{random.choice(['6.1', '6.5', '6.7'])}-inch {random.choice(['AMOLED', 'LCD', 'OLED'])}",
                    'processor': random.choice(['Snapdragon 8 Gen 2', 'A17 Bionic', 'Helio G88', 'Dimensity 7050']),
                    'ram': f"{random.choice([4, 6, 8, 12])}GB",
                    'storage': f"{random.choice([64, 128, 256, 512])}GB",
                    'camera': f"{random.choice([48, 50, 64])}MP {random.choice(['Triple', 'Quad', 'Dual'])}",
                    'battery': f"{random.randint(4000, 6000)}mAh"
                })
            elif subcategory == 'Laptops':
                specs.update({
                    'display': f"{random.choice(['13.3', '14', '15.6', '16'])}-inch {random.choice(['FHD', 'QHD', '4K'])}",
                    'processor': random.choice(['Intel i5', 'Intel i7', 'Intel i9', 'AMD Ryzen 5', 'AMD Ryzen 7']),
                    'ram': f"{random.choice([8, 16, 32])}GB",
                    'storage': f"{random.choice([256, 512, 1])}TB SSD",
                    'graphics': random.choice(['Integrated', 'NVIDIA RTX 3050', 'NVIDIA RTX 4070', 'AMD Radeon'])
                })
        elif major_category == 'Fashion':
            if subcategory == 'Footwear':
                specs.update({
                    'type': random.choice(['Sneakers', 'Running Shoes', 'Casual Shoes', 'Formal Shoes']),
                    'material': random.choice(['Mesh', 'Leather', 'Synthetic', 'Canvas']),
                    'color': random.choice(['Black', 'White', 'Blue', 'Red', 'Various']),
                    'style': random.choice(['Casual', 'Sporty', 'Retro', 'Modern'])
                })
            elif subcategory == 'Apparel':
                specs.update({
                    'type': random.choice(['T-Shirt', 'Shirt', 'Jeans', 'Dress', 'Jacket']),
                    'material': random.choice(['Cotton', 'Polyester', 'Denim', 'Viscose', 'Wool']),
                    'fit': random.choice(['Regular', 'Slim', 'Loose', 'Oversized']),
                    'color': random.choice(['Black', 'White', 'Blue', 'Red', 'Green'])
                })
        
        return specs
    
    def generate_user_queries(self, num_queries: int = 500, 
                            product_catalog: pd.DataFrame = None) -> pd.DataFrame:
        """Generate synthetic user queries."""
        logger.info(f"Generating {num_queries} user queries...")
        
        queries = []
        for i in range(num_queries):
            query = self._generate_realistic_query(product_catalog)
            queries.append({
                'query_id': i + 1,
                'original_query': query,
                'corrected_query': query,  # Assume no typos for synthetic data
                'frequency': random.randint(1, 100),
                'predicted_purchase': random.choice(['yes', 'no', 'maybe']),
                'timestamp': self.fake.date_time_between(
                    start_date='-30d', end_date='now'
                ).isoformat()
            })
        
        df = pd.DataFrame(queries)
        logger.info(f"Generated {len(df)} user queries")
        return df
    
    def _generate_realistic_query(self, product_catalog: pd.DataFrame = None) -> str:
        """Generate realistic search queries."""
        query_patterns = [
            # Brand-based queries
            lambda: f"{random.choice(list(self.brands.keys()))} {random.choice(['phone', 'laptop', 'shoes', 'shirt'])}",
            lambda: f"{random.choice(['samsung', 'apple', 'nike', 'adidas'])}",
            
            # Category-based queries
            lambda: f"{random.choice(['gaming', 'budget', 'premium', 'wireless'])} {random.choice(['laptop', 'phone', 'headphones'])}",
            lambda: f"{random.choice(['running', 'casual', 'formal', 'sports'])} {random.choice(['shoes', 'shirt', 'dress'])}",
            
            # Feature-based queries
            lambda: f"{random.choice(['bluetooth', 'wireless', 'smart', 'portable'])} {random.choice(['speaker', 'earbuds', 'watch'])}",
            lambda: f"{random.choice(['4k', 'oled', 'smart', 'curved'])} {random.choice(['tv', 'monitor'])}",
            
            # Generic queries
            lambda: f"{random.choice(['best', 'top', 'cheap', 'expensive'])} {random.choice(['phone', 'laptop', 'shoes', 'camera'])}",
            lambda: f"{random.choice(['mobile', 'laptop', 'shoes', 'camera', 'headphones'])} under {random.choice(['5000', '10000', '20000', '50000'])}",
            
            # Specific product queries
            lambda: f"{random.choice(['iphone', 'samsung galaxy', 'nike air', 'adidas ultraboost'])}",
        ]
        
        return random.choice(query_patterns)()
    
    def generate_realtime_info(self, product_catalog: pd.DataFrame) -> pd.DataFrame:
        """Generate real-time product information."""
        logger.info("Generating real-time product information...")
        
        realtime_data = []
        for _, product in product_catalog.iterrows():
            # Generate delivery estimates
            delivery_days = random.randint(1, 7)
            delivery_estimate = f"{delivery_days} day{'s' if delivery_days > 1 else ''}"
            
            # Generate offers
            offer_types = ['No Cost EMI', 'Bank Offer', 'Exchange Offer', 'Cashback', 'Discount', None]
            offer_strength = random.choice(offer_types)
            
            # Generate availability
            availability = random.choice(['In Stock', 'Out of Stock', 'Limited Stock'])
            
            realtime_data.append({
                'product_id': product['product_id'],
                'delivery_estimate': delivery_estimate,
                'delivery_speed_score': 1.0 / delivery_days,
                'offer_strength': offer_strength,
                'availability': availability,
                'last_updated': datetime.now().isoformat()
            })
        
        df = pd.DataFrame(realtime_data)
        logger.info(f"Generated real-time info for {len(df)} products")
        return df
    
    def generate_session_log(self, num_sessions: int = 1000, 
                           user_queries: pd.DataFrame = None) -> pd.DataFrame:
        """Generate session log data."""
        logger.info(f"Generating {num_sessions} sessions...")
        
        sessions = []
        for session_id in range(num_sessions):
            # Generate session details
            session_start = self.fake.date_time_between(
                start_date='-7d', end_date='now'
            )
            session_duration = random.randint(60, 3600)  # 1 minute to 1 hour
            session_end = session_start + timedelta(seconds=session_duration)
            
            # Generate session events
            num_events = random.randint(1, 10)
            for event_id in range(num_events):
                event_time = session_start + timedelta(
                    seconds=random.randint(0, session_duration)
                )
                
                event_type = random.choice(['search', 'view', 'click', 'add_to_cart', 'purchase'])
                
                if event_type == 'search':
                    query = self._generate_realistic_query() if user_queries is None else \
                           random.choice(user_queries['original_query'].tolist())
                    sessions.append({
                        'session_id': session_id,
                        'user_id': random.randint(1, 100),
                        'event_type': event_type,
                        'event_data': query,
                        'timestamp': event_time.isoformat(),
                        'session_start': session_start.isoformat(),
                        'session_end': session_end.isoformat()
                    })
                else:
                    product_id = random.randint(1, 1000)
                    sessions.append({
                        'session_id': session_id,
                        'user_id': random.randint(1, 100),
                        'event_type': event_type,
                        'event_data': str(product_id),
                        'timestamp': event_time.isoformat(),
                        'session_start': session_start.isoformat(),
                        'session_end': session_end.isoformat()
                    })
        
        df = pd.DataFrame(sessions)
        logger.info(f"Generated {len(df)} session events across {num_sessions} sessions")
        return df
    
    def generate_all_datasets(self, output_dir: str = 'dataset') -> Dict[str, pd.DataFrame]:
        """Generate all synthetic datasets."""
        logger.info("Generating all synthetic datasets...")
        
        # Create output directory
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate datasets
        product_catalog = self.generate_product_catalog(1000)
        user_queries = self.generate_user_queries(500, product_catalog)
        realtime_info = self.generate_realtime_info(product_catalog)
        session_log = self.generate_session_log(1000, user_queries)
        
        # Save datasets
        product_catalog.to_csv(f'{output_dir}/synthetic_product_catalog.csv', index=False)
        user_queries.to_csv(f'{output_dir}/synthetic_user_queries.csv', index=False)
        realtime_info.to_csv(f'{output_dir}/synthetic_realtime_info.csv', index=False)
        session_log.to_csv(f'{output_dir}/synthetic_session_log.csv', index=False)
        
        logger.info("All datasets generated and saved successfully!")
        
        return {
            'product_catalog': product_catalog,
            'user_queries': user_queries,
            'realtime_info': realtime_info,
            'session_log': session_log
        }

def main():
    """Main function to generate all datasets."""
    generator = EnhancedDataGenerator(seed=42)
    datasets = generator.generate_all_datasets()
    
    print("✅ All synthetic datasets generated successfully!")
    print(f"📊 Product Catalog: {len(datasets['product_catalog'])} products")
    print(f"🔍 User Queries: {len(datasets['user_queries'])} queries")
    print(f"⚡ Realtime Info: {len(datasets['realtime_info'])} records")
    print(f"📱 Session Log: {len(datasets['session_log'])} events")

if __name__ == "__main__":
    main()
