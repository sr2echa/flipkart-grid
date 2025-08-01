import pandas as pd
import numpy as np
import random
from typing import List, Dict
import itertools

class DatasetEnhancer:
    """Enhanced dataset generator for better autosuggest performance."""
    
    def __init__(self):
        self.brands = [
            'samsung', 'apple', 'nike', 'adidas', 'sony', 'oneplus', 'xiaomi', 
            'vivo', 'oppo', 'realme', 'dell', 'hp', 'lenovo', 'asus', 'lg',
            'boat', 'jbl', 'puma', 'reebok', 'zara', 'h&m', 'bata', 'titan',
            'fossil', 'casio', 'canon', 'nikon', 'gopro', 'fitbit', 'garmin'
        ]
        
        self.categories = [
            'phone', 'smartphone', 'mobile', 'laptop', 'computer', 'tablet',
            'headphones', 'earbuds', 'speaker', 'camera', 'watch', 'smartwatch',
            'shoes', 'sneakers', 'sandals', 'boots', 'shirt', 'tshirt', 't-shirt',
            'jeans', 'trousers', 'jacket', 'hoodie', 'dress', 'skirt',
            'tv', 'television', 'monitor', 'keyboard', 'mouse', 'charger',
            'cable', 'case', 'cover', 'bag', 'backpack', 'wallet', 'belt',
            'cap', 'hat', 'sunglasses', 'perfume', 'deodorant', 'shampoo'
        ]
        
        self.attributes = [
            'wireless', 'bluetooth', 'waterproof', 'gaming', 'professional',
            'casual', 'formal', 'sports', 'running', 'fitness', 'smart',
            'digital', 'analog', 'leather', 'cotton', 'denim', 'silk',
            'black', 'white', 'blue', 'red', 'green', 'brown', 'gray',
            'small', 'medium', 'large', 'xl', 'xxl', 'slim', 'regular',
            'comfortable', 'lightweight', 'durable', 'premium', 'budget'
        ]
        
        self.price_ranges = [
            'under 1000', 'under 2000', 'under 5000', 'under 10000',
            'under 15000', 'under 20000', 'under 30000', 'under 50000',
            'above 1000', 'above 2000', 'above 5000', 'above 10000',
            'above 20000', 'above 50000', 'between 1000 and 5000',
            'between 5000 and 10000', 'between 10000 and 20000',
            'between 20000 and 50000'
        ]
        
        self.locations = [
            'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata',
            'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow'
        ]
        
        self.events = [
            'diwali', 'holi', 'christmas', 'new year', 'valentine',
            'ipl', 'cricket', 'football', 'wedding', 'birthday',
            'graduation', 'festival', 'sale', 'offer', 'discount'
        ]

    def generate_enhanced_user_queries(self, num_queries: int = 5000) -> pd.DataFrame:
        """Generate enhanced user queries with better variety."""
        queries = []
        
        # Generate brand + category combinations
        for brand in self.brands:
            for category in self.categories:
                if random.random() < 0.8:  # 80% chance to include this combination
                    queries.append(f"{brand} {category}")
                    
                    # Add variations with attributes
                    for attr in random.sample(self.attributes, min(3, len(self.attributes))):
                        if random.random() < 0.3:  # 30% chance
                            queries.append(f"{brand} {attr} {category}")
                            queries.append(f"{attr} {brand} {category}")
                    
                    # Add price range variations
                    for price in random.sample(self.price_ranges, min(2, len(self.price_ranges))):
                        if random.random() < 0.2:  # 20% chance
                            queries.append(f"{brand} {category} {price}")

        # Generate category + attribute combinations
        for category in self.categories:
            for attr in self.attributes:
                if random.random() < 0.5:  # 50% chance
                    queries.append(f"{attr} {category}")
                    queries.append(f"{category} {attr}")

        # Generate price-focused queries
        for category in self.categories:
            for price in self.price_ranges:
                if random.random() < 0.4:  # 40% chance
                    queries.append(f"{category} {price}")

        # Generate event-based queries
        for event in self.events:
            for category in self.categories:
                if random.random() < 0.3:  # 30% chance
                    queries.append(f"{event} {category}")
                    if category in ['lights', 'decor', 'gifts', 'dress', 'shirt']:
                        queries.append(f"{category} for {event}")

        # Generate common e-commerce patterns
        common_patterns = [
            "best {category}",
            "latest {category}",
            "top {category}",
            "buy {category}",
            "{category} online",
            "{category} offer",
            "{category} sale",
            "{category} deals",
            "cheap {category}",
            "premium {category}",
            "{category} for men",
            "{category} for women",
            "{category} for kids"
        ]
        
        for pattern in common_patterns:
            for category in self.categories:
                if random.random() < 0.3:  # 30% chance
                    queries.append(pattern.format(category=category))

        # Remove duplicates and create DataFrame
        unique_queries = list(set(queries))
        
        # If we have fewer queries than requested, pad with variations
        while len(unique_queries) < num_queries:
            # Create variations by combining random elements
            brand = random.choice(self.brands)
            category = random.choice(self.categories)
            attr = random.choice(self.attributes)
            
            variations = [
                f"{brand} {category}",
                f"{attr} {category}",
                f"{brand} {attr} {category}",
                f"buy {brand} {category}",
                f"best {category}",
                f"{category} under 10000"
            ]
            
            for var in variations:
                if var not in unique_queries:
                    unique_queries.append(var)
                    if len(unique_queries) >= num_queries:
                        break
            if len(unique_queries) >= num_queries:
                break

        # Take only the requested number
        unique_queries = unique_queries[:num_queries]
        
        # Create DataFrame with additional columns
        df_data = []
        for i, query in enumerate(unique_queries):
            df_data.append({
                'query_id': f'enhanced_{i:06d}',
                'user_id': f'user_{random.randint(10000, 99999)}',
                'raw_query': query,
                'query_timestamp': pd.Timestamp.now(),
                'query_source': random.choice(['website', 'mobile_app']),
                'location_data': f"{random.choice(self.locations)}, India",
                'device_type': random.choice(['desktop', 'mobile', 'tablet']),
                'ip_address': f"192.168.{random.randint(1,255)}.{random.randint(1,255)}"
            })
        
        return pd.DataFrame(df_data)

    def generate_enhanced_session_log(self, num_sessions: int = 10000) -> pd.DataFrame:
        """Generate enhanced session log with realistic user behavior patterns."""
        sessions = []
        
        # Define user personas
        personas = {
            'tech_enthusiast': {
                'queries': ['laptop', 'smartphone', 'gaming', 'apple', 'samsung', 'oneplus'],
                'events': ['sale', 'new launch', 'tech'],
                'locations': ['Bangalore', 'Mumbai', 'Delhi']
            },
            'fashion_lover': {
                'queries': ['shoes', 'dress', 'shirt', 'jeans', 'nike', 'adidas', 'zara'],
                'events': ['fashion week', 'sale', 'wedding'],
                'locations': ['Mumbai', 'Delhi', 'Chennai']
            },
            'budget_shopper': {
                'queries': ['under 1000', 'under 2000', 'cheap', 'offer', 'discount'],
                'events': ['sale', 'discount', 'offer'],
                'locations': ['Pune', 'Ahmedabad', 'Lucknow']
            },
            'premium_shopper': {
                'queries': ['premium', 'luxury', 'apple', 'samsung', 'sony', 'above 50000'],
                'events': ['launch', 'premium'],
                'locations': ['Mumbai', 'Delhi', 'Bangalore']
            },
            'sports_enthusiast': {
                'queries': ['sports', 'running', 'fitness', 'nike', 'adidas', 'shoes'],
                'events': ['sports', 'fitness', 'ipl'],
                'locations': ['Chennai', 'Kolkata', 'Hyderabad']
            }
        }
        
        for i in range(num_sessions):
            # Choose a persona
            persona_name = random.choice(list(personas.keys()))
            persona = personas[persona_name]
            
            session_id = f'session_{i:06d}'
            base_time = pd.Timestamp.now() - pd.Timedelta(days=random.randint(1, 30))
            
            # Generate 1-5 queries per session
            num_queries_in_session = random.randint(1, 5)
            
            for j in range(num_queries_in_session):
                # Choose query based on persona
                if random.random() < 0.7:  # 70% chance to use persona-specific query
                    query_base = random.choice(persona['queries'])
                    if random.random() < 0.5:
                        # Add another element
                        other_element = random.choice(self.categories + self.attributes)
                        query = f"{query_base} {other_element}"
                    else:
                        query = query_base
                else:
                    # Use random query
                    query = random.choice(self.categories)
                
                sessions.append({
                    'session_id': session_id,
                    'timestamp': base_time + pd.Timedelta(minutes=j*2),
                    'query': query,
                    'clicked_product_id': f'P{random.randint(10000, 99999)}' if random.random() < 0.6 else None,
                    'purchased': 1 if random.random() < 0.1 else 0,  # 10% purchase rate
                    'location': random.choice(persona['locations']),
                    'event': random.choice(persona['events']) if random.random() < 0.3 else None
                })
        
        return pd.DataFrame(sessions)

    def enhance_existing_datasets(self):
        """Enhance existing datasets with new high-quality data."""
        print("Enhancing datasets with more realistic e-commerce data...")
        
        # Load existing datasets
        try:
            existing_queries = pd.read_csv('../dataset/user_queries.csv')
            existing_sessions = pd.read_csv('../dataset/session_log.csv')
            
            print(f"Existing queries: {len(existing_queries)}")
            print(f"Existing sessions: {len(existing_sessions)}")
            
        except Exception as e:
            print(f"Error loading existing datasets: {e}")
            existing_queries = pd.DataFrame()
            existing_sessions = pd.DataFrame()
        
        # Generate enhanced data
        enhanced_queries = self.generate_enhanced_user_queries(3000)
        enhanced_sessions = self.generate_enhanced_session_log(15000)
        
        print(f"Generated enhanced queries: {len(enhanced_queries)}")
        print(f"Generated enhanced sessions: {len(enhanced_sessions)}")
        
        # Combine with existing data
        if not existing_queries.empty:
            combined_queries = pd.concat([existing_queries, enhanced_queries], ignore_index=True)
        else:
            combined_queries = enhanced_queries
            
        if not existing_sessions.empty:
            combined_sessions = pd.concat([existing_sessions, enhanced_sessions], ignore_index=True)
        else:
            combined_sessions = enhanced_sessions
        
        # Save enhanced datasets
        combined_queries.to_csv('../dataset/user_queries_enhanced.csv', index=False)
        combined_sessions.to_csv('../dataset/session_log_enhanced.csv', index=False)
        
        print(f"Final enhanced queries: {len(combined_queries)}")
        print(f"Final enhanced sessions: {len(combined_sessions)}")
        print("Enhanced datasets saved!")
        
        return combined_queries, combined_sessions

if __name__ == "__main__":
    enhancer = DatasetEnhancer()
    enhancer.enhance_existing_datasets()