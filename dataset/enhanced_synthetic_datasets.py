import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import os

class EnhancedSyntheticDataGenerator:
    """
    Generate enhanced synthetic datasets for the Flikart autosuggest and SRP systems.
    Creates more realistic and granular data to support advanced features.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the data generator with a seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        
        # Product data
        self.brands = {
            'Electronics': ['Samsung', 'Apple', 'Sony', 'OnePlus', 'Xiaomi', 'Vivo', 'Oppo', 'Realme', 
                           'Dell', 'HP', 'Lenovo', 'Asus', 'LG', 'Boat', 'JBL', 'Bose', 'Canon', 'Nikon'],
            'Fashion': ['Nike', 'Adidas', 'Puma', 'Reebok', 'Bata', 'Zara', 'H&M', 'Levis', 'IKEA'],
            'Home': ['Philips', 'Prestige', 'Bajaj', 'Crompton', 'Symphony', 'Blue Star', 'Daikin', 
                     'Whirlpool', 'Bosch', 'Godrej Interio']
        }
        
        self.categories = {
            'Electronics': ['Mobiles', 'Laptops', 'Televisions', 'Audio', 'Wearables', 'Cameras'],
            'Fashion': ['Footwear', 'Apparel', 'Accessories'],
            'Home': ['Kitchen Appliances', 'Cooling', 'Furniture', 'Laundry']
        }
        
        self.locations = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 
                         'Ahmedabad', 'Pune', 'Jaipur', 'Lucknow']
        
        self.events = ['search', 'click', 'view_details', 'add_to_cart', 'purchase', 'compare']
        
    def generate_enhanced_product_catalog(self, num_products: int = 1000) -> pd.DataFrame:
        """
        Generate enhanced product catalog with more detailed specifications.
        
        Args:
            num_products: Number of products to generate
            
        Returns:
            Enhanced product catalog DataFrame
        """
        print(f"Generating enhanced product catalog with {num_products} products...")
        
        products = []
        
        for i in range(num_products):
            # Determine major category and subcategory
            major_category = random.choice(list(self.categories.keys()))
            subcategory = random.choice(self.categories[major_category])
            brand = random.choice(self.brands[major_category])
            
            # Generate product details based on category
            product_data = self._generate_product_details(i + 1, major_category, subcategory, brand)
            products.append(product_data)
        
        df = pd.DataFrame(products)
        print(f"Generated {len(df)} products")
        return df
    
    def _generate_product_details(self, product_id: int, major_category: str, 
                                subcategory: str, brand: str) -> Dict:
        """Generate detailed product specifications based on category."""
        
        if major_category == 'Electronics':
            return self._generate_electronics_product(product_id, subcategory, brand)
        elif major_category == 'Fashion':
            return self._generate_fashion_product(product_id, subcategory, brand)
        else:  # Home
            return self._generate_home_product(product_id, subcategory, brand)
    
    def _generate_electronics_product(self, product_id: int, subcategory: str, brand: str) -> Dict:
        """Generate electronics product with detailed specs."""
        
        if subcategory == 'Mobiles':
            models = ['Galaxy', 'iPhone', 'Xperia', 'Nord', 'Redmi', 'V', 'Reno', 'GT']
            model = random.choice(models)
            title = f"{brand} {model} {random.randint(10, 25)}"
            
            specs = {
                'display': f"{random.choice([6.1, 6.3, 6.7, 6.8])}-inch {random.choice(['AMOLED', 'IPS LCD', 'Super Retina'])}",
                'processor': random.choice(['Snapdragon 8 Gen 2', 'A17 Bionic', 'Dimensity 9200', 'Helio G99']),
                'ram': f"{random.choice([6, 8, 12])}GB",
                'storage': f"{random.choice([128, 256, 512])}GB",
                'camera': f"{random.choice([48, 50, 64, 108])}MP {random.choice(['Triple', 'Quad', 'Dual'])}",
                'battery': f"{random.choice([4000, 4500, 5000, 5500])}mAh"
            }
            
            price = random.randint(15000, 150000)
            rating = round(random.uniform(3.5, 5.0), 1)
            
        elif subcategory == 'Laptops':
            models = ['Inspiron', 'Pavilion', 'ThinkPad', 'ZenBook', 'Gram', 'Swift']
            model = random.choice(models)
            title = f"{brand} {model} {random.randint(3000, 5000)}"
            
            specs = {
                'display': f"{random.choice([13.3, 14, 15.6, 16])}-inch {random.choice(['FHD', 'QHD', '4K'])}",
                'processor': random.choice(['Intel i5', 'Intel i7', 'Intel i9', 'AMD Ryzen 5', 'AMD Ryzen 7']),
                'ram': f"{random.choice([8, 16, 32])}GB",
                'storage': f"{random.choice([256, 512, 1])}TB SSD",
                'graphics': random.choice(['Integrated', 'RTX 3050', 'RTX 4060', 'RTX 4070'])
            }
            
            price = random.randint(40000, 200000)
            rating = round(random.uniform(3.8, 5.0), 1)
            
        else:  # Audio, Wearables, etc.
            models = ['Pro', 'Ultra', 'Max', 'Plus', 'Elite']
            model = random.choice(models)
            title = f"{brand} {subcategory} {model}"
            
            specs = {
                'type': subcategory,
                'features': random.choice(['Noise Cancellation', 'Waterproof', 'Bluetooth 5.3', 'Fast Charging']),
                'battery': f"{random.randint(6, 30)} hours"
            }
            
            price = random.randint(1000, 50000)
            rating = round(random.uniform(3.5, 5.0), 1)
        
        return {
            'product_id': f"P{product_id:05d}",
            'title': title,
            'description': f"High-quality {subcategory.lower()} from {brand} with advanced features and excellent performance.",
            'category': f"{major_category}>{subcategory}",
            'brand': brand,
            'price': price,
            'rating': rating,
            'num_reviews': random.randint(100, 5000),
            'specifications': json.dumps(specs),
            'major_category': subcategory
        }
    
    def _generate_fashion_product(self, product_id: int, subcategory: str, brand: str) -> Dict:
        """Generate fashion product with detailed specs."""
        
        if subcategory == 'Footwear':
            styles = ['Air Max', 'Ultraboost', 'RS-X', 'Revolution', 'Grand Court', 'Smash']
            style = random.choice(styles)
            title = f"{brand} {style}"
            
            specs = {
                'type': random.choice(['Sneakers', 'Running Shoes', 'Casual Shoes']),
                'style': style,
                'material': random.choice(['Mesh', 'Synthetic Leather', 'Primeknit', 'Canvas']),
                'color': random.choice(['Black', 'White', 'Blue', 'Red', 'Green', 'Various'])
            }
            
            price = random.randint(2000, 15000)
            
        else:  # Apparel
            types = ['T-Shirt', 'Shirt', 'Jeans', 'Dress', 'Hoodie', 'Jacket']
            product_type = random.choice(types)
            title = f"{brand} {product_type}"
            
            specs = {
                'type': product_type,
                'style': random.choice(['Casual', 'Formal', 'Sporty', 'Elegant']),
                'material': random.choice(['Cotton', 'Polyester', 'Denim', 'Viscose']),
                'fit': random.choice(['Regular', 'Slim', 'Loose', 'Oversized'])
            }
            
            price = random.randint(500, 5000)
        
        rating = round(random.uniform(3.5, 5.0), 1)
        
        return {
            'product_id': f"P{product_id:05d}",
            'title': title,
            'description': f"Stylish {subcategory.lower()} from {brand} with premium quality and comfortable fit.",
            'category': f"{major_category}>{subcategory}",
            'brand': brand,
            'price': price,
            'rating': rating,
            'num_reviews': random.randint(50, 2000),
            'specifications': json.dumps(specs),
            'major_category': subcategory
        }
    
    def _generate_home_product(self, product_id: int, subcategory: str, brand: str) -> Dict:
        """Generate home product with detailed specs."""
        
        if subcategory == 'Kitchen Appliances':
            types = ['Mixer Grinder', 'Air Fryer', 'Induction Cooktop', 'Refrigerator', 'Dishwasher']
            product_type = random.choice(types)
            title = f"{brand} {product_type}"
            
            specs = {
                'type': product_type,
                'capacity': f"{random.randint(1, 10)} Liters" if 'Grinder' in product_type or 'Fryer' in product_type else f"{random.randint(200, 800)} Liters",
                'power': f"{random.randint(500, 2000)}W" if 'Grinder' in product_type or 'Cooktop' in product_type else "",
                'features': random.choice(['Digital Display', 'Auto Shut-off', 'Energy Efficient', 'Smart Control'])
            }
            
            price = random.randint(2000, 100000)
            
        else:  # Cooling, Furniture, etc.
            types = ['AC', 'Cooler', 'Bed Frame', 'Wardrobe', 'Sofa', 'Table']
            product_type = random.choice(types)
            title = f"{brand} {product_type}"
            
            specs = {
                'type': product_type,
                'capacity': f"{random.randint(1, 2)} Ton" if product_type == 'AC' else f"{random.randint(20, 100)} Liters",
                'material': random.choice(['Steel', 'Wood', 'Plastic', 'Fabric']),
                'features': random.choice(['Inverter', 'Smart Control', 'Energy Star', 'Easy Installation'])
            }
            
            price = random.randint(5000, 150000)
        
        rating = round(random.uniform(3.5, 5.0), 1)
        
        return {
            'product_id': f"P{product_id:05d}",
            'title': title,
            'description': f"Reliable {subcategory.lower()} from {brand} for modern homes.",
            'category': f"{major_category}>{subcategory}",
            'brand': brand,
            'price': price,
            'rating': rating,
            'num_reviews': random.randint(20, 1000),
            'specifications': json.dumps(specs),
            'major_category': subcategory
        }
    
    def generate_enhanced_user_queries(self, num_queries: int = 2000) -> pd.DataFrame:
        """
        Generate enhanced user queries with more realistic patterns.
        
        Args:
            num_queries: Number of queries to generate
            
        Returns:
            Enhanced user queries DataFrame
        """
        print(f"Generating enhanced user queries with {num_queries} queries...")
        
        queries = []
        
        # Generate brand-based queries
        for brand in [brand for brands in self.brands.values() for brand in brands]:
            frequency = random.randint(100, 2000)
            queries.append({
                'query_id': f"Q{len(queries):06d}",
                'raw_query': brand.lower(),
                'corrected_query': brand.lower(),
                'frequency': frequency,
                'event': random.choice(self.events),
                'category': self._get_brand_category(brand),
                'clicked_product_ids': self._generate_clicked_products(brand)
            })
        
        # Generate product type queries
        product_types = ['smartphone', 'laptop', 'headphones', 'shoes', 'watch', 'tv', 'camera']
        for product_type in product_types:
            frequency = random.randint(200, 1500)
            queries.append({
                'query_id': f"Q{len(queries):06d}",
                'raw_query': product_type,
                'corrected_query': product_type,
                'frequency': frequency,
                'event': random.choice(self.events),
                'category': self._get_product_category(product_type),
                'clicked_product_ids': self._generate_clicked_products(product_type)
            })
        
        # Generate price range queries
        price_ranges = [
            ('under 10000', 300), ('under 20000', 400), ('under 50000', 250),
            ('above 50000', 200), ('between 10000 and 30000', 150)
        ]
        for price_range, freq in price_ranges:
            queries.append({
                'query_id': f"Q{len(queries):06d}",
                'raw_query': f"smartphone {price_range}",
                'corrected_query': f"smartphone {price_range}",
                'frequency': freq,
                'event': random.choice(self.events),
                'category': 'Electronics',
                'clicked_product_ids': self._generate_clicked_products('smartphone')
            })
        
        # Generate typo queries (20% of total)
        num_typos = int(num_queries * 0.2)
        for i in range(num_typos):
            typo_query = self._generate_typo_query()
            queries.append({
                'query_id': f"Q{len(queries):06d}",
                'raw_query': typo_query['raw'],
                'corrected_query': typo_query['corrected'],
                'frequency': random.randint(50, 300),
                'event': random.choice(self.events),
                'category': self._get_product_category(typo_query['corrected']),
                'clicked_product_ids': self._generate_clicked_products(typo_query['corrected'])
            })
        
        # Fill remaining with random queries
        while len(queries) < num_queries:
            query_text = self._generate_random_query()
            queries.append({
                'query_id': f"Q{len(queries):06d}",
                'raw_query': query_text,
                'corrected_query': query_text,
                'frequency': random.randint(10, 200),
                'event': random.choice(self.events),
                'category': random.choice(['Electronics', 'Fashion', 'Home']),
                'clicked_product_ids': self._generate_clicked_products(query_text)
            })
        
        df = pd.DataFrame(queries)
        print(f"Generated {len(df)} queries")
        return df
    
    def _get_brand_category(self, brand: str) -> str:
        """Get category for a brand."""
        for category, brands in self.brands.items():
            if brand in brands:
                return category
        return 'Electronics'
    
    def _get_product_category(self, product_type: str) -> str:
        """Get category for a product type."""
        electronics = ['smartphone', 'laptop', 'headphones', 'tv', 'camera', 'watch']
        fashion = ['shoes', 'clothes', 'dress', 'shirt']
        
        if product_type in electronics:
            return 'Electronics'
        elif product_type in fashion:
            return 'Fashion'
        else:
            return 'Home'
    
    def _generate_clicked_products(self, query: str) -> str:
        """Generate clicked product IDs for a query."""
        num_products = random.randint(1, 8)
        product_ids = [f"P{random.randint(1, 1000):05d}" for _ in range(num_products)]
        return ", ".join(product_ids)
    
    def _generate_typo_query(self) -> Dict[str, str]:
        """Generate a query with a typo."""
        typos = {
            'samsng': 'samsung', 'aple': 'apple', 'nkie': 'nike',
            'addidas': 'adidas', 'soney': 'sony', 'onepls': 'oneplus',
            'xiomi': 'xiaomi', 'vvo': 'vivo', 'opo': 'oppo',
            'realmi': 'realme', 'del': 'dell', 'lenvo': 'lenovo',
            'smartphon': 'smartphone', 'laptap': 'laptop',
            'headphons': 'headphones', 'televisn': 'television'
        }
        
        typo = random.choice(list(typos.keys()))
        return {'raw': typo, 'corrected': typos[typo]}
    
    def _generate_random_query(self) -> str:
        """Generate a random query."""
        words = ['best', 'cheap', 'premium', 'gaming', 'wireless', 'bluetooth', 
                'waterproof', 'fast', 'lightweight', 'durable']
        products = ['phone', 'laptop', 'headphones', 'shoes', 'watch', 'camera']
        
        return f"{random.choice(words)} {random.choice(products)}"
    
    def generate_enhanced_realtime_info(self, product_catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Generate enhanced real-time product info with granular delivery and pricing.
        
        Args:
            product_catalog: Product catalog to base real-time info on
            
        Returns:
            Enhanced real-time info DataFrame
        """
        print("Generating enhanced real-time product info...")
        
        realtime_data = []
        
        for _, product in product_catalog.iterrows():
            # Generate multiple location entries for each product
            num_locations = random.randint(1, 5)
            selected_locations = random.sample(self.locations, num_locations)
            
            for location in selected_locations:
                # Generate delivery estimate
                delivery_options = [
                    '10 minutes', '30 minutes', '1 hour', '2 hours', '4 hours',
                    '1 day', '2 days', '3 days', '4 days', '5 days', '6 days', '7 days'
                ]
                delivery_estimate = random.choice(delivery_options)
                
                # Generate current price (with some variation from catalog price)
                price_variation = random.uniform(0.8, 1.2)
                current_price = round(product['price'] * price_variation, 2)
                
                # Generate stock status
                stock_status = random.choices(
                    ['In Stock', 'Low Stock', 'Out of Stock'],
                    weights=[0.7, 0.2, 0.1]
                )[0]
                
                # Generate offer strength
                offer_options = ['No Offer', '5% OFF', '10% OFF', '15% OFF', '20% OFF', '25% OFF', '30% OFF']
                offer_strength = random.choice(offer_options)
                
                # Generate rating and review count
                rating = round(random.uniform(3.0, 5.0), 1)
                review_count = random.randint(10, 50000)
                
                realtime_data.append({
                    'product_id': product['product_id'],
                    'location': location,
                    'delivery_estimate': delivery_estimate,
                    'current_price': current_price,
                    'stock_status': stock_status,
                    'offer_strength': offer_strength,
                    'rating': rating,
                    'review_count': review_count
                })
        
        df = pd.DataFrame(realtime_data)
        print(f"Generated {len(df)} real-time records")
        return df
    
    def generate_enhanced_session_log(self, num_sessions: int = 5000) -> pd.DataFrame:
        """
        Generate enhanced session log with deeper context and realistic patterns.
        
        Args:
            num_sessions: Number of sessions to generate
            
        Returns:
            Enhanced session log DataFrame
        """
        print(f"Generating enhanced session log with {num_sessions} sessions...")
        
        session_data = []
        session_id = 100000
        
        for _ in range(num_sessions):
            # Generate session duration (1-30 minutes)
            session_duration = random.randint(1, 30)
            
            # Generate session start time (within last 30 days)
            start_time = datetime.now() - timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            # Generate number of interactions per session (1-8)
            num_interactions = random.randint(1, 8)
            
            # Generate interactions for this session
            for i in range(num_interactions):
                # Calculate timestamp for this interaction
                interaction_time = start_time + timedelta(
                    minutes=random.randint(0, session_duration)
                )
                
                # Generate query
                query = self._generate_session_query()
                
                # Generate clicked product (sometimes empty)
                clicked_product = None
                if random.random() < 0.7:  # 70% chance of clicking
                    clicked_product = f"P{random.randint(1, 1000):05d}"
                
                # Generate purchase (rare)
                purchased = random.random() < 0.1  # 10% chance of purchase
                
                # Generate location
                location = random.choice(self.locations)
                
                # Generate event
                event = random.choices(
                    self.events,
                    weights=[0.4, 0.3, 0.15, 0.1, 0.03, 0.02]  # search most common, purchase rare
                )[0]
                
                session_data.append({
                    'session_id': f"S{session_id}",
                    'timestamp': interaction_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'query': query,
                    'clicked_product_id': clicked_product,
                    'purchased': purchased,
                    'location': location,
                    'event': event
                })
            
            session_id += 1
        
        df = pd.DataFrame(session_data)
        print(f"Generated {len(df)} session records")
        return df
    
    def _generate_session_query(self) -> str:
        """Generate a query for session log."""
        query_types = [
            # Brand queries
            lambda: random.choice([brand for brands in self.brands.values() for brand in brands]).lower(),
            # Product type queries
            lambda: random.choice(['smartphone', 'laptop', 'headphones', 'shoes', 'watch', 'tv']),
            # Price range queries
            lambda: f"smartphone under {random.choice(['10000', '20000', '50000'])}",
            # Typo queries
            lambda: random.choice(['samsng', 'aple', 'nkie', 'addidas', 'smartphon']),
            # Complex queries
            lambda: f"{random.choice(['best', 'cheap', 'gaming'])} {random.choice(['laptop', 'phone', 'headphones'])}"
        ]
        
        return random.choice(query_types)()
    
    def generate_enhanced_ner_dataset(self, user_queries: pd.DataFrame) -> pd.DataFrame:
        """
        Generate enhanced NER dataset with more comprehensive entity tagging.
        
        Args:
            user_queries: User queries to base NER data on
            
        Returns:
            Enhanced NER dataset DataFrame
        """
        print("Generating enhanced NER dataset...")
        
        ner_data = []
        
        # Entity types and their patterns
        entity_patterns = {
            'BRAND': [brand.lower() for brands in self.brands.values() for brand in brands],
            'PRODUCT': ['smartphone', 'laptop', 'headphones', 'earbuds', 'tv', 'television', 
                       'camera', 'watch', 'smartwatch', 'shoes', 'sneakers', 'jeans', 'shirt'],
            'PRICE_RANGE': ['under 10000', 'under 20000', 'under 50000', 'above 50000', 
                           'between 10000 and 30000'],
            'FEATURE': ['gaming', 'wireless', 'bluetooth', 'waterproof', 'fast', 'lightweight'],
            'COLOR': ['black', 'white', 'blue', 'red', 'green', 'pink', 'yellow']
        }
        
        for _, query_row in user_queries.iterrows():
            query = query_row['corrected_query'].lower()
            tokens = query.split()
            
            for token in tokens:
                tag = 'O'  # Default tag
                
                # Check for entity matches
                for entity_type, patterns in entity_patterns.items():
                    if token in patterns:
                        tag = f'B-{entity_type}'
                        break
                
                ner_data.append({
                    'query_id': query_row['query_id'],
                    'token': token,
                    'tag': tag
                })
        
        df = pd.DataFrame(ner_data)
        print(f"Generated {len(df)} NER records")
        return df
    
    def generate_all_enhanced_datasets(self, output_dir: str = 'dataset') -> Dict[str, pd.DataFrame]:
        """
        Generate all enhanced datasets.
        
        Args:
            output_dir: Output directory for datasets
            
        Returns:
            Dictionary of generated DataFrames
        """
        print("Generating all enhanced synthetic datasets...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate datasets
        product_catalog = self.generate_enhanced_product_catalog(1000)
        user_queries = self.generate_enhanced_user_queries(2000)
        realtime_info = self.generate_enhanced_realtime_info(product_catalog)
        session_log = self.generate_enhanced_session_log(5000)
        ner_dataset = self.generate_enhanced_ner_dataset(user_queries)
        
        # Save datasets
        product_catalog.to_csv(os.path.join(output_dir, 'enhanced_product_catalog.csv'), index=False)
        user_queries.to_csv(os.path.join(output_dir, 'enhanced_user_queries.csv'), index=False)
        realtime_info.to_csv(os.path.join(output_dir, 'enhanced_realtime_info.csv'), index=False)
        session_log.to_csv(os.path.join(output_dir, 'enhanced_session_log.csv'), index=False)
        ner_dataset.to_csv(os.path.join(output_dir, 'enhanced_ner_dataset.csv'), index=False)
        
        print("All enhanced datasets generated and saved!")
        
        return {
            'product_catalog': product_catalog,
            'user_queries': user_queries,
            'realtime_info': realtime_info,
            'session_log': session_log,
            'ner_dataset': ner_dataset
        }

if __name__ == "__main__":
    # Generate all enhanced datasets
    generator = EnhancedSyntheticDataGenerator()
    datasets = generator.generate_all_enhanced_datasets()
    
    print("\nDataset Summary:")
    for name, df in datasets.items():
        print(f"{name}: {len(df)} records")
    
    print("\nSample data from each dataset:")
    for name, df in datasets.items():
        print(f"\n{name.upper()} (first 3 rows):")
        print(df.head(3))
        print("-" * 50) 