"""
Create Realistic Product Data with Real Brands for Autosuggest
Uses actual brand names, product categories, and realistic combinations
"""
import pandas as pd
import random
from collections import defaultdict

def create_realistic_product_suggestions():
    """Create realistic product suggestions with real brand names."""
    
    # Real brand mappings by category
    brand_categories = {
        'Electronics': {
            'smartphones': ['Samsung', 'Xiaomi', 'Apple', 'OnePlus', 'Vivo', 'Oppo', 'Realme', 'iPhone'],
            'laptops': ['HP', 'Dell', 'Lenovo', 'Asus', 'Acer', 'MacBook', 'MSI', 'Apple'],
            'headphones': ['Sony', 'Boat', 'JBL', 'Bose', 'Sennheiser', 'Audio-Technica', 'AirPods'],
            'tvs': ['Samsung', 'LG', 'Sony', 'Mi', 'TCL', 'Panasonic', 'Toshiba'],
            'tablets': ['iPad', 'Samsung', 'Lenovo', 'Amazon', 'Xiaomi'],
            'smartwatch': ['Apple Watch', 'Samsung', 'Amazfit', 'Fastrack', 'Noise', 'Fire-Boltt']
        },
        'Fashion': {
            'shoes': ['Nike', 'Adidas', 'Puma', 'Reebok', 'Converse', 'Vans', 'New Balance', 'Skechers'],
            'clothing': ['Zara', 'H&M', 'Nike', 'Adidas', 'Levis', 'Tommy Hilfiger', 'Calvin Klein', 'Puma'],
            'watches': ['Titan', 'Fastrack', 'Casio', 'Fossil', 'Daniel Wellington', 'Apple Watch'],
            'bags': ['American Tourister', 'VIP', 'Skybags', 'Wildcraft', 'Nike', 'Adidas']
        },
        'Sports': {
            'cricket': ['MRF', 'SG', 'Kookaburra', 'Gray-Nicolls', 'SS', 'Adidas', 'Nike'],
            'football': ['Nike', 'Adidas', 'Puma', 'Nivia', 'Cosco', 'Wilson'],
            'fitness': ['Reebok', 'Nike', 'Adidas', 'Decathlon', 'Cosco', 'Strauss']
        },
        'Home': {
            'appliances': ['LG', 'Samsung', 'Whirlpool', 'Godrej', 'Haier', 'Panasonic', 'Bosch'],
            'furniture': ['IKEA', 'Urban Ladder', 'Pepperfry', 'Nilkamal', 'Godrej'],
            'kitchen': ['Prestige', 'Hawkins', 'Pigeon', 'Butterfly', 'Bajaj', 'Philips']
        }
    }
    
    # Product type templates with realistic variations
    product_templates = {
        # Electronics
        'smartphone': [
            '{brand} Galaxy S{num}', '{brand} {num} Pro', '{brand} {num} Ultra',
            '{brand} Note {num}', '{brand} {num} Plus', '{brand} {num}',
            'iPhone {num}', 'iPhone {num} Pro', 'iPhone {num} Pro Max'
        ],
        'laptop': [
            '{brand} {series} {num}', '{brand} Gaming Laptop', '{brand} Business Laptop',
            '{brand} Ultrabook', '{brand} {num} inch Laptop', 'MacBook Pro {num}',
            '{brand} ThinkPad', '{brand} Inspiron {num}', '{brand} Pavilion {num}'
        ],
        'headphones': [
            '{brand} Wireless Headphones', '{brand} Bluetooth Headphones', 
            '{brand} Noise Cancelling Headphones', '{brand} Gaming Headset',
            '{brand} Earbuds', '{brand} True Wireless Earbuds', 'AirPods Pro'
        ],
        'tv': [
            '{brand} {num} inch Smart TV', '{brand} {num} inch LED TV',
            '{brand} {num} inch 4K TV', '{brand} QLED {num} inch',
            '{brand} OLED {num} inch', '{brand} Android TV {num} inch'
        ],
        
        # Fashion
        'shoes': [
            '{brand} Running Shoes', '{brand} Casual Shoes', '{brand} Sports Shoes',
            '{brand} Sneakers', '{brand} Football Boots', '{brand} Basketball Shoes',
            '{brand} Training Shoes', '{brand} Lifestyle Shoes'
        ],
        'clothing': [
            '{brand} T-Shirt', '{brand} Polo T-Shirt', '{brand} Hoodie',
            '{brand} Tracksuit', '{brand} Jeans', '{brand} Jacket',
            '{brand} Sports Wear', '{brand} Casual Wear'
        ],
        'jersey': [
            '{brand} Cricket Jersey', '{brand} Football Jersey', '{brand} Team Jersey',
            'IPL {team} Jersey', 'India Cricket Jersey', '{brand} Sports Jersey',
            'Champions League Jersey', 'World Cup Jersey'
        ],
        
        # Sports specific
        'cricket': [
            '{brand} Cricket Bat', '{brand} Cricket Ball', '{brand} Cricket Kit',
            '{brand} Cricket Helmet', '{brand} Cricket Gloves', '{brand} Cricket Pads'
        ],
        'football': [
            '{brand} Football', '{brand} Football Boots', '{brand} Football Kit',
            '{brand} Goal Keeper Gloves', '{brand} Shin Guards'
        ],
        
        # Home & Kitchen
        'home_appliance': [
            '{brand} Washing Machine', '{brand} Refrigerator', '{brand} Air Conditioner',
            '{brand} Microwave Oven', '{brand} Dishwasher', '{brand} Water Purifier'
        ],
        'kitchen': [
            '{brand} Mixer Grinder', '{brand} Pressure Cooker', '{brand} Non-Stick Pan',
            '{brand} Induction Cooktop', '{brand} Rice Cooker', '{brand} Food Processor'
        ]
    }
    
    # Generate realistic product suggestions
    all_suggestions = []
    
    # Electronics suggestions
    for product_type in ['smartphone', 'laptop', 'headphones', 'tv']:
        templates = product_templates[product_type]
        brands = brand_categories['Electronics'].get(f'{product_type}s', ['Samsung', 'Apple', 'Sony'])
        
        for template in templates:
            for brand in brands:
                if '{series}' in template and '{num}' in template:
                    for series in ['Inspiron', 'Pavilion', 'ThinkPad', 'VivoBook']:
                        for num in [13, 14, 15, 16, 17]:
                            suggestion = template.format(brand=brand, series=series, num=num)
                            all_suggestions.append({
                                'suggestion': suggestion,
                                'category': 'Electronics',
                                'subcategory': product_type,
                                'brand': brand,
                                'frequency': random.randint(10, 100)
                            })
                elif '{num}' in template:
                    for num in [12, 13, 14, 15, 20, 32, 43, 55, 65]:
                        suggestion = template.format(brand=brand, num=num)
                        all_suggestions.append({
                            'suggestion': suggestion,
                            'category': 'Electronics',
                            'subcategory': product_type,
                            'brand': brand,
                            'frequency': random.randint(10, 100)
                        })
                else:
                    suggestion = template.format(brand=brand)
                    all_suggestions.append({
                        'suggestion': suggestion,
                        'category': 'Electronics',
                        'subcategory': product_type,
                        'brand': brand,
                        'frequency': random.randint(10, 100)
                    })
    
    # Fashion suggestions
    for product_type in ['shoes', 'clothing', 'jersey']:
        templates = product_templates[product_type]
        if product_type == 'jersey':
            brands = ['Nike', 'Adidas', 'Puma', 'Under Armour']
            teams = ['CSK', 'RCB', 'MI', 'KKR', 'DC', 'SRH', 'RR', 'PBKS']
        else:
            brands = brand_categories['Fashion'][product_type]
            teams = []
        
        for template in templates:
            for brand in brands:
                if '{team}' in template:
                    for team in teams:
                        suggestion = template.format(brand=brand, team=team)
                        all_suggestions.append({
                            'suggestion': suggestion,
                            'category': 'Fashion',
                            'subcategory': product_type,
                            'brand': brand,
                            'frequency': random.randint(15, 80)
                        })
                else:
                    suggestion = template.format(brand=brand)
                    all_suggestions.append({
                        'suggestion': suggestion,
                        'category': 'Fashion',
                        'subcategory': product_type,
                        'brand': brand,
                        'frequency': random.randint(15, 80)
                    })
    
    # Sports suggestions
    for product_type in ['cricket', 'football']:
        templates = product_templates[product_type]
        brands = brand_categories['Sports'][product_type]
        
        for template in templates:
            for brand in brands:
                suggestion = template.format(brand=brand)
                all_suggestions.append({
                    'suggestion': suggestion,
                    'category': 'Sports',
                    'subcategory': product_type,
                    'brand': brand,
                    'frequency': random.randint(5, 60)
                })
    
    # Add common typos and their corrections
    typo_corrections = {
        'xiomi': 'xiaomi',
        'samsng': 'samsung',
        'samung': 'samsung',
        'jersy': 'jersey',
        'jesery': 'jersey',
        'lapto': 'laptop',
        'leptop': 'laptop',
        'mobil': 'mobile',
        'moble': 'mobile',
        'headphone': 'headphones',
        'hedphones': 'headphones',
        'aple': 'apple',
        'ifone': 'iphone',
        'criket': 'cricket',
        'footbal': 'football'
    }
    
    # Add typo entries
    for typo, correction in typo_corrections.items():
        # Find suggestions with the correct spelling
        matching_suggestions = [s for s in all_suggestions if correction.lower() in s['suggestion'].lower()]
        if matching_suggestions:
            # Create typo version
            sample = random.choice(matching_suggestions)
            typo_suggestion = sample['suggestion'].lower().replace(correction.lower(), typo)
            all_suggestions.append({
                'suggestion': typo_suggestion,
                'category': sample['category'],
                'subcategory': sample['subcategory'],
                'brand': sample['brand'],
                'frequency': random.randint(1, 10),
                'is_typo': True,
                'corrected_to': sample['suggestion']
            })
    
    return all_suggestions

def create_enhanced_queries_dataset():
    """Create enhanced queries dataset with realistic product suggestions."""
    
    print("🚀 Creating Enhanced Queries Dataset with Real Brands...")
    
    suggestions = create_realistic_product_suggestions()
    
    # Convert to DataFrame
    df = pd.DataFrame(suggestions)
    
    # Add additional columns for compatibility
    df['query_id'] = [f'enhanced_q_{i+1}' for i in range(len(df))]
    df['user_id'] = [f'user_{random.randint(1, 10000)}' for _ in range(len(df))]
    df['raw_query'] = df['suggestion']
    df['corrected_query'] = df['suggestion']
    df['query_timestamp'] = [f'2024-0{random.randint(1, 7)}-{random.randint(1, 28):02d}' for _ in range(len(df))]
    df['query_source'] = [random.choice(['search_bar', 'voice_search', 'category_browse']) for _ in range(len(df))]
    df['location_data'] = [random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Pune', 'Kolkata', 'Hyderabad']) for _ in range(len(df))]
    df['device_type'] = [random.choice(['mobile', 'desktop', 'tablet']) for _ in range(len(df))]
    df['ip_address'] = [f'192.168.{random.randint(1, 255)}.{random.randint(1, 255)}' for _ in range(len(df))]
    
    # Reorder columns to match expected format
    final_columns = ['corrected_query', 'query_id', 'user_id', 'raw_query', 'frequency', 
                     'query_timestamp', 'query_source', 'location_data', 'device_type', 'ip_address']
    df_final = df[final_columns]
    
    # Save to file
    output_file = '../dataset/user_queries_realistic_v4.csv'
    df_final.to_csv(output_file, index=False)
    
    print(f"✅ Enhanced dataset saved to {output_file}")
    print(f"📊 Total suggestions: {len(df_final)}")
    print(f"🏷️ Unique brands: {df['brand'].nunique()}")
    print(f"📂 Categories: {df['category'].unique()}")
    
    # Show sample data
    print(f"\n📝 Sample realistic suggestions:")
    samples = df_final['corrected_query'].sample(15).tolist()
    for i, sample in enumerate(samples, 1):
        print(f"   {i}. {sample}")
    
    # Show typo corrections
    typos = df[df.get('is_typo', False) == True]
    if len(typos) > 0:
        print(f"\n🔧 Sample typo corrections:")
        for _, typo in typos.head(5).iterrows():
            print(f"   '{typo['suggestion']}' → '{typo['corrected_to']}'")
    
    return df_final

if __name__ == "__main__":
    enhanced_df = create_enhanced_queries_dataset()