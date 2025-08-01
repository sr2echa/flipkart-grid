"""
Create a comprehensive query database for high-quality autosuggest
"""
import pandas as pd
import random
from collections import defaultdict

def create_comprehensive_queries():
    """Create a comprehensive set of realistic product search queries."""
    
    # Product categories with related queries
    query_patterns = {
        # Electronics & Tech
        'electronics': [
            'laptop', 'gaming laptop', 'laptop under 50000', 'laptop above 100000', 'macbook',
            'dell laptop', 'hp laptop', 'lenovo laptop', 'asus laptop', 'cheap laptop',
            'mobile', 'smartphone', 'iphone', 'samsung phone', 'android phone', 'phone under 20000',
            'samsung galaxy', 'oneplus', 'redmi', 'vivo', 'oppo', 'realme',
            'headphones', 'wireless headphones', 'bluetooth headphones', 'earphones', 'airpods',
            'tablet', 'ipad', 'android tablet', 'kindle', 'e-reader',
            'camera', 'dslr camera', 'canon camera', 'nikon camera', 'gopro', 'webcam',
            'smartwatch', 'fitness tracker', 'apple watch', 'garmin watch',
            'tv', 'smart tv', 'led tv', '4k tv', '32 inch tv', '55 inch tv', 'lg tv', 'samsung tv'
        ],
        
        # Fashion & Apparel
        'fashion': [
            'shirt', 'mens shirt', 'womens shirt', 'formal shirt', 'casual shirt', 'cotton shirt',
            'jeans', 'mens jeans', 'womens jeans', 'blue jeans', 'black jeans', 'skinny jeans',
            'dress', 'party dress', 'summer dress', 'maxi dress', 'casual dress', 'formal dress',
            'tshirt', 'mens tshirt', 'womens tshirt', 'cotton tshirt', 'polo tshirt',
            'shoes', 'running shoes', 'casual shoes', 'formal shoes', 'sports shoes',
            'nike shoes', 'adidas shoes', 'puma shoes', 'reebok shoes',
            'saree', 'silk saree', 'cotton saree', 'designer saree', 'wedding saree',
            'kurta', 'mens kurta', 'womens kurta', 'cotton kurta', 'silk kurta'
        ],
        
        # Sports & Fitness
        'sports': [
            'cricket bat', 'cricket ball', 'cricket kit', 'cricket helmet', 'cricket gloves',
            'football', 'soccer ball', 'football boots', 'football jersey', 'fifa football',
            'badminton racket', 'shuttlecock', 'badminton shoes', 'yonex racket',
            'tennis racket', 'tennis ball', 'tennis shoes', 'wilson racket',
            'gym equipment', 'dumbbells', 'treadmill', 'exercise bike', 'yoga mat',
            'jersey', 'cricket jersey', 'football jersey', 'team jersey', 'sports jersey',
            'nike jersey', 'adidas jersey', 'puma jersey',
            'csk jersey', 'rcb jersey', 'mi jersey', 'ipl jersey', 'chennai super kings jersey'
        ],
        
        # Home & Living
        'home': [
            'sofa', 'sofa set', 'leather sofa', '3 seater sofa', 'corner sofa', 'recliner sofa',
            'bed', 'king size bed', 'queen size bed', 'single bed', 'wooden bed', 'metal bed',
            'mattress', 'memory foam mattress', 'spring mattress', 'single mattress',
            'dining table', 'wooden dining table', '4 seater dining table', '6 seater dining table',
            'chair', 'office chair', 'study chair', 'dining chair', 'plastic chair',
            'curtains', 'blackout curtains', 'silk curtains', 'cotton curtains', 'door curtains',
            'carpet', 'area rug', 'bedroom carpet', 'living room carpet', 'persian carpet',
            'lights', 'led lights', 'ceiling lights', 'wall lights', 'decorative lights',
            'diwali lights', 'string lights', 'fairy lights', 'festival lights'
        ],
        
        # Kitchen & Appliances
        'kitchen': [
            'mixer grinder', 'juicer', 'blender', 'food processor', 'pressure cooker',
            'rice cooker', 'induction cooktop', 'gas stove', 'microwave oven', 'oven',
            'refrigerator', 'fridge', 'double door fridge', 'single door fridge', 'lg fridge',
            'washing machine', 'front load washing machine', 'top load washing machine',
            'air conditioner', 'split ac', 'window ac', '1 ton ac', '1.5 ton ac', 'lg ac',
            'water purifier', 'ro water purifier', 'uv water purifier', 'aquaguard',
            'cookware', 'non stick pan', 'pressure cooker', 'kadai', 'tawa'
        ],
        
        # Beauty & Personal Care
        'beauty': [
            'shampoo', 'hair oil', 'conditioner', 'hair mask', 'anti dandruff shampoo',
            'face wash', 'face cream', 'moisturizer', 'sunscreen', 'face mask',
            'lipstick', 'foundation', 'mascara', 'eyeliner', 'kajal', 'nail polish',
            'perfume', 'deodorant', 'body spray', 'cologne', 'attar',
            'soap', 'body wash', 'hand wash', 'face soap', 'herbal soap',
            'toothbrush', 'toothpaste', 'mouthwash', 'dental floss'
        ],
        
        # Books & Education
        'books': [
            'books', 'novel', 'fiction books', 'non fiction books', 'biography',
            'textbook', 'ncert books', 'cbse books', 'competitive exam books',
            'notebook', 'diary', 'pen', 'pencil', 'marker', 'highlighter',
            'backpack', 'school bag', 'laptop bag', 'college bag'
        ]
    }
    
    # Common typos and variations
    typo_variations = {
        'laptop': ['lapto', 'laptap', 'leptop', 'laptop'],
        'mobile': ['mobil', 'moble', 'moblie', 'mobile'],
        'samsung': ['samsng', 'samung', 'samsung', 'samsumg'],
        'jersey': ['jersy', 'jesery', 'jersay', 'jersey'],
        'iPhone': ['iphone', 'ifone', 'iPhon', 'iphone'],
        'cricket': ['criket', 'crickt', 'cricket', 'criket'],
        'football': ['footbal', 'futbol', 'football', 'fotball'],
        'headphones': ['headphone', 'hedphones', 'headfones', 'headphones']
    }
    
    # Generate comprehensive query list
    all_queries = []
    query_frequencies = defaultdict(int)
    
    # Add all base queries
    for category, queries in query_patterns.items():
        for query in queries:
            all_queries.append(query)
            # Simulate realistic frequency distribution
            if 'popular' in ['laptop', 'mobile', 'jeans', 'tshirt', 'shoes'] and any(pop in query for pop in ['laptop', 'mobile', 'jeans', 'tshirt', 'shoes']):
                query_frequencies[query] = random.randint(50, 200)
            else:
                query_frequencies[query] = random.randint(5, 50)
    
    # Add typo variations
    for correct, typos in typo_variations.items():
        for typo in typos:
            if typo != correct:  # Don't add the correct spelling again
                all_queries.append(typo)
                query_frequencies[typo] = random.randint(2, 15)
    
    # Add common combinations
    combinations = [
        'nike running shoes', 'adidas football boots', 'samsung galaxy phone',
        'hp gaming laptop', 'dell business laptop', 'apple macbook pro',
        'cricket bat and ball', 'football and jersey', 'badminton racket and shuttlecock',
        'sofa set with table', 'bed with mattress', 'dining table and chairs',
        'mixer grinder and juicer', 'pressure cooker set', 'non stick cookware set',
        'bluetooth headphones with mic', 'wireless mouse and keyboard',
        'formal shirt and pants', 'jeans and tshirt', 'dress for party',
        'winter jacket for men', 'summer dress for women', 'cotton shirt for office',
        'ipl team jersey', 'csk yellow jersey', 'rcb red jersey', 'mi blue jersey'
    ]
    
    for combo in combinations:
        all_queries.append(combo)
        query_frequencies[combo] = random.randint(10, 40)
    
    # Add seasonal and contextual queries
    seasonal_queries = {
        'diwali': ['diwali lights', 'diwali decoration', 'diwali gifts', 'diyas', 'rangoli'],
        'ipl': ['ipl jersey', 'cricket bat for ipl', 'team jersey ipl', 'csk jersey', 'rcb jersey'],
        'summer': ['summer dress', 'summer shirt', 'cooling fan', 'air conditioner', 'summer shoes'],
        'winter': ['winter jacket', 'sweater', 'woolen clothes', 'heater', 'winter boots'],
        'wedding': ['wedding dress', 'wedding shoes', 'wedding jewelry', 'silk saree', 'formal wear']
    }
    
    for season, queries in seasonal_queries.items():
        for query in queries:
            all_queries.append(query)
            query_frequencies[query] = random.randint(8, 30)
    
    # Create DataFrame
    query_data = []
    for i, query in enumerate(all_queries):
        query_data.append({
            'query_id': f'q_{i+1}',
            'user_id': f'user_{random.randint(1, 1000)}',
            'raw_query': query,
            'corrected_query': query,
            'frequency': query_frequencies[query],
            'query_timestamp': f'2024-0{random.randint(1, 7)}-{random.randint(1, 28):02d}',
            'query_source': random.choice(['search_bar', 'voice_search', 'category_browse']),
            'location_data': random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Pune', 'Kolkata', 'Hyderabad']),
            'device_type': random.choice(['mobile', 'desktop', 'tablet']),
            'ip_address': f'192.168.{random.randint(1, 255)}.{random.randint(1, 255)}'
        })
    
    df = pd.DataFrame(query_data)
    
    # Remove duplicates and keep highest frequency
    df_grouped = df.groupby('corrected_query').agg({
        'query_id': 'first',
        'user_id': 'first', 
        'raw_query': 'first',
        'frequency': 'max',
        'query_timestamp': 'first',
        'query_source': 'first',
        'location_data': 'first',
        'device_type': 'first',
        'ip_address': 'first'
    }).reset_index()
    
    print(f"Created {len(df_grouped)} unique comprehensive queries")
    return df_grouped

if __name__ == "__main__":
    # Create comprehensive queries
    queries_df = create_comprehensive_queries()
    
    # Save to file
    output_file = '../dataset/user_queries_comprehensive.csv'
    queries_df.to_csv(output_file, index=False)
    print(f"Saved comprehensive queries to {output_file}")
    
    # Display some examples
    print("\nSample queries:")
    print(queries_df[['corrected_query', 'frequency']].head(20))