#!/usr/bin/env python3
"""
Generate NER Training Dataset
=============================

This script generates a comprehensive NER training dataset by combining
user_queries.csv and user_queries1.csv, and creating entity annotations
for brands, products, categories, colors, sizes, and other relevant entities.
"""

import pandas as pd
import re
import json
from typing import List, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NERDatasetGenerator:
    """Generate NER training dataset from user queries."""
    
    def __init__(self):
        """Initialize the generator with entity patterns."""
        self.brands = {
            'samsung', 'apple', 'nike', 'adidas', 'sony', 'oneplus', 'xiaomi', 'vivo', 'oppo', 
            'realme', 'dell', 'hp', 'lenovo', 'asus', 'lg', 'boat', 'jbl', 'puma', 'reebok', 
            'bata', 'zara', 'hm', 'levis', 'ikea', 'prestige', 'philips', 'oral-b', 'dji',
            'garmin', 'nest', 'brita', 'ring', 'alexa', 'oculus', 'dji', 'bose', 'beats',
            'canon', 'nikon', 'fujifilm', 'gopro', 'roku', 'firestick', 'chromecast'
        }
        
        self.products = {
            'smartphone', 'mobile', 'phone', 'laptop', 'notebook', 'headphones', 'earbuds',
            'tv', 'television', 'shoes', 'sneakers', 'jeans', 'shirt', 'hoodie', 'watch',
            'smartwatch', 'tablet', 'camera', 'speaker', 'keyboard', 'mouse', 'charger',
            'case', 'bag', 'wallet', 'bluetooth', 'wireless', 'noise', 'cancelling',
            'running', 'leather', 'cotton', 'organic', 'compatible', 'android', 'coffee',
            'maker', 'yoga', 'mat', 'hard', 'drive', 'stroller', 'food', 'kettle',
            'backpack', 'vacuum', 'cleaner', 'skincare', 'tent', 'hose', 'power', 'bank',
            'fryer', 'bicycle', 'plant', 'pots', 'gaming', 'toothbrush', 'robot', 'seat',
            'blender', 'knives', 'camera', 'security', 'cooker', 'luggage', 'thermostat',
            'drone', 'scooter', 'espresso', 'machine', 'accessories', 'filter', 'bulb',
            'guitar', 'purifier', 'chair', 'storage', 'player', 'headset', 'projector',
            'screen', 'transmitter', 'doorbell', 'kit', 'monitor', 'bike', 'cleaning',
            'assistant', 'scale', 'panel', 'beauty', 'bottle', 'garden', 'converter',
            'washing', 'machine', 'gear', 'feeder', 'curtain', 'trash', 'sanitizer',
            'shower', 'tools', 'charger', 'luggage', 'tracking', 'purifier'
        }
        
        self.colors = {
            'red', 'blue', 'green', 'yellow', 'black', 'white', 'gray', 'grey', 'brown',
            'pink', 'purple', 'orange', 'silver', 'gold', 'navy', 'maroon', 'beige',
            'cream', 'tan', 'olive', 'teal', 'coral', 'lavender', 'indigo', 'violet'
        }
        
        self.sizes = {
            'xs', 's', 'm', 'l', 'xl', 'xxl', 'xxxl', 'small', 'medium', 'large',
            'extra', 'tiny', 'mini', 'big', 'huge', 'compact', 'portable', 'lightweight'
        }
        
        self.materials = {
            'leather', 'cotton', 'polyester', 'nylon', 'wool', 'silk', 'denim', 'canvas',
            'plastic', 'metal', 'wood', 'glass', 'ceramic', 'stainless', 'steel', 'aluminum',
            'copper', 'brass', 'bronze', 'titanium', 'carbon', 'fiber', 'silicone', 'rubber'
        }
        
        self.features = {
            'bluetooth', 'wireless', 'noise', 'cancelling', 'waterproof', 'shockproof',
            'dustproof', 'antimicrobial', 'anti-aging', 'non-slip', 'ergonomic', 'adjustable',
            'foldable', 'portable', 'rechargeable', 'battery', 'solar', 'automatic', 'manual',
            'digital', 'analog', 'smart', 'touch', 'voice', 'remote', 'app', 'compatible',
            'android', 'ios', 'windows', 'mac', 'linux', '4k', 'hd', 'full', 'hdmi',
            'usb', 'type-c', 'lightning', 'micro', 'sd', 'card', 'memory', 'ram', 'gb',
            'tb', 'processor', 'intel', 'amd', 'ryzen', 'core', 'i3', 'i5', 'i7', 'i9',
            'rtx', 'gtx', 'graphics', 'gpu', 'cpu', 'ssd', 'hdd', 'wifi', 'ethernet',
            'gps', 'nfc', 'fingerprint', 'face', 'recognition', 'camera', 'front', 'back',
            'dual', 'triple', 'quad', 'mega', 'pixel', 'zoom', 'optical', 'digital'
        }
        
        self.categories = {
            'electronics', 'fashion', 'home', 'kitchen', 'beauty', 'sports', 'outdoor',
            'automotive', 'baby', 'pet', 'garden', 'office', 'gaming', 'health', 'fitness',
            'travel', 'luggage', 'accessories', 'jewelry', 'watches', 'bags', 'wallets',
            'shoes', 'clothing', 'appliances', 'tools', 'hardware', 'software', 'books',
            'music', 'movies', 'toys', 'games', 'art', 'crafts', 'food', 'beverages',
            'pharmaceuticals', 'medical', 'dental', 'optical', 'hearing', 'mobility'
        }
        
        self.price_indicators = {
            'budget', 'cheap', 'affordable', 'expensive', 'premium', 'luxury', 'under',
            'over', 'between', 'range', 'price', 'cost', 'value', 'deal', 'offer',
            'discount', 'sale', 'clearance', 'bargain', 'economy', 'economic'
        }
        
        logger.info("🔧 NER Dataset Generator Initialized")
        logger.info(f"📊 Entity Categories: {len(self.brands)} brands, {len(self.products)} products")
        logger.info(f"📊 Entity Categories: {len(self.colors)} colors, {len(self.sizes)} sizes")
        logger.info(f"📊 Entity Categories: {len(self.materials)} materials, {len(self.features)} features")
    
    def load_user_queries(self, file_path: str) -> pd.DataFrame:
        """Load user queries from CSV file."""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"✅ Loaded {len(df)} queries from {file_path}")
            return df
        except Exception as e:
            logger.error(f"❌ Failed to load {file_path}: {e}")
            return pd.DataFrame()
    
    def extract_entities(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract entities from text using pattern matching.
        
        Returns:
            List of tuples: (start_pos, end_pos, entity_type)
        """
        entities = []
        text_lower = text.lower()
        words = text.split()
        
        for i, word in enumerate(words):
            word_clean = re.sub(r'[^\w\s]', '', word.lower())
            
            # Check brands
            if word_clean in self.brands:
                start = text.find(word)
                end = start + len(word)
                entities.append((start, end, 'BRAND'))
            
            # Check products
            elif word_clean in self.products:
                start = text.find(word)
                end = start + len(word)
                entities.append((start, end, 'PRODUCT'))
            
            # Check colors
            elif word_clean in self.colors:
                start = text.find(word)
                end = start + len(word)
                entities.append((start, end, 'COLOR'))
            
            # Check sizes
            elif word_clean in self.sizes:
                start = text.find(word)
                end = start + len(word)
                entities.append((start, end, 'SIZE'))
            
            # Check materials
            elif word_clean in self.materials:
                start = text.find(word)
                end = start + len(word)
                entities.append((start, end, 'MATERIAL'))
            
            # Check features
            elif word_clean in self.features:
                start = text.find(word)
                end = start + len(word)
                entities.append((start, end, 'FEATURE'))
            
            # Check categories
            elif word_clean in self.categories:
                start = text.find(word)
                end = start + len(word)
                entities.append((start, end, 'CATEGORY'))
            
            # Check price indicators
            elif word_clean in self.price_indicators:
                start = text.find(word)
                end = start + len(word)
                entities.append((start, end, 'PRICE_INDICATOR'))
        
        # Handle multi-word entities
        self._extract_multi_word_entities(text, entities)
        
        return entities
    
    def _extract_multi_word_entities(self, text: str, entities: List[Tuple[str, str, str]]):
        """Extract multi-word entities like 'mobile phone', 'running shoes', etc."""
        text_lower = text.lower()
        
        # Multi-word patterns
        multi_word_patterns = [
            ('mobile phone', 'PRODUCT'),
            ('running shoes', 'PRODUCT'),
            ('bluetooth headphones', 'PRODUCT'),
            ('noise cancelling', 'FEATURE'),
            ('leather wallet', 'PRODUCT'),
            ('cotton t-shirt', 'PRODUCT'),
            ('smart watch', 'PRODUCT'),
            ('coffee maker', 'PRODUCT'),
            ('yoga mat', 'PRODUCT'),
            ('hard drive', 'PRODUCT'),
            ('baby stroller', 'PRODUCT'),
            ('dog food', 'PRODUCT'),
            ('electric kettle', 'PRODUCT'),
            ('hiking backpack', 'PRODUCT'),
            ('vacuum cleaner', 'PRODUCT'),
            ('skincare set', 'PRODUCT'),
            ('camping tent', 'PRODUCT'),
            ('garden hose', 'PRODUCT'),
            ('power bank', 'PRODUCT'),
            ('air fryer', 'PRODUCT'),
            ('kids bicycle', 'PRODUCT'),
            ('plant pots', 'PRODUCT'),
            ('gaming mouse', 'PRODUCT'),
            ('electric toothbrush', 'PRODUCT'),
            ('robot vacuum', 'PRODUCT'),
            ('car seat', 'PRODUCT'),
            ('blender powerful', 'PRODUCT'),
            ('security camera', 'PRODUCT'),
            ('kitchen knives', 'PRODUCT'),
            ('pressure cooker', 'PRODUCT'),
            ('gaming laptop', 'PRODUCT'),
            ('luggage set', 'PRODUCT'),
            ('coffee maker', 'PRODUCT'),
            ('smart thermostat', 'PRODUCT'),
            ('drone camera', 'PRODUCT'),
            ('electric scooter', 'PRODUCT'),
            ('portable speaker', 'PRODUCT'),
            ('espresso machine', 'PRODUCT'),
            ('yoga accessories', 'PRODUCT'),
            ('water filter', 'PRODUCT'),
            ('smart bulb', 'PRODUCT'),
            ('electric guitar', 'PRODUCT'),
            ('air purifier', 'PRODUCT'),
            ('ergonomic chair', 'PRODUCT'),
            ('portable storage', 'PRODUCT'),
            ('blu-ray player', 'PRODUCT'),
            ('virtual reality', 'PRODUCT'),
            ('home theatre', 'PRODUCT'),
            ('fitness tracker', 'PRODUCT'),
            ('solar power', 'PRODUCT'),
            ('baby gear', 'PRODUCT'),
            ('electric shaver', 'PRODUCT'),
            ('portable projector', 'PRODUCT'),
            ('smart lock', 'PRODUCT'),
            ('digital camera', 'PRODUCT'),
            ('pet supplies', 'PRODUCT'),
            ('noise cancelling', 'FEATURE'),
            ('gaming headset', 'PRODUCT'),
            ('portable ssd', 'PRODUCT'),
            ('smart home', 'PRODUCT'),
            ('graphics card', 'PRODUCT'),
            ('kitchen appliances', 'PRODUCT'),
            ('portable air', 'PRODUCT'),
            ('standing desk', 'PRODUCT'),
            ('projector screen', 'PRODUCT'),
            ('bluetooth fm', 'PRODUCT'),
            ('gaming chair', 'PRODUCT'),
            ('outdoor gear', 'PRODUCT'),
            ('smart doorbell', 'PRODUCT'),
            ('robot building', 'PRODUCT'),
            ('portable monitor', 'PRODUCT'),
            ('drone dji', 'PRODUCT'),
            ('electric bike', 'PRODUCT'),
            ('home cleaning', 'PRODUCT'),
            ('virtual assistant', 'PRODUCT'),
            ('smart scale', 'PRODUCT'),
            ('robot vacuum', 'PRODUCT'),
            ('portable solar', 'PRODUCT'),
            ('electric bike', 'PRODUCT'),
            ('beauty products', 'PRODUCT'),
            ('smart water', 'PRODUCT'),
            ('portable espresso', 'PRODUCT'),
            ('smart garden', 'PRODUCT'),
            ('electric standing', 'PRODUCT'),
            ('portable washing', 'PRODUCT'),
            ('camping gear', 'PRODUCT'),
            ('smart pet', 'PRODUCT'),
            ('automatic curtain', 'PRODUCT'),
            ('smart trash', 'PRODUCT'),
            ('uv light', 'PRODUCT'),
            ('portable camping', 'PRODUCT'),
            ('gardening tools', 'PRODUCT'),
            ('electric car', 'PRODUCT'),
            ('smart luggage', 'PRODUCT'),
            ('portable air', 'PRODUCT')
        ]
        
        for pattern, entity_type in multi_word_patterns:
            if pattern in text_lower:
                start = text_lower.find(pattern)
                end = start + len(pattern)
                # Check if not already added
                if not any(start == e[0] and end == e[1] for e in entities):
                    entities.append((start, end, entity_type))
    
    def convert_to_bio_format(self, text: str, entities: List[Tuple[str, str, str]]) -> List[Tuple[str, str]]:
        """
        Convert entities to BIO format for NER training.
        
        Returns:
            List of tuples: (token, tag)
        """
        # Sort entities by start position
        entities = sorted(entities, key=lambda x: x[0])
        
        # Initialize all tokens as O
        tokens = text.split()
        tags = ['O'] * len(tokens)
        
        # Map character positions to token positions
        char_to_token = {}
        current_pos = 0
        for i, token in enumerate(tokens):
            start_char = text.find(token, current_pos)
            end_char = start_char + len(token)
            char_to_token[start_char] = i
            current_pos = end_char
        
        # Apply entity tags
        for start_char, end_char, entity_type in entities:
            # Find tokens that overlap with this entity
            entity_tokens = []
            for i, token in enumerate(tokens):
                token_start = text.find(token, 0)
                token_end = token_start + len(token)
                
                # Check if token overlaps with entity
                if (token_start < end_char and token_end > start_char):
                    entity_tokens.append(i)
            
            if entity_tokens:
                # Apply BIO tags
                for i, token_idx in enumerate(entity_tokens):
                    if i == 0:
                        tags[token_idx] = f'B-{entity_type}'
                    else:
                        tags[token_idx] = f'I-{entity_type}'
        
        return list(zip(tokens, tags))
    
    def generate_dataset(self, output_path: str = "ner_dataset_new.csv"):
        """Generate the complete NER dataset."""
        logger.info("🔄 Generating new NER dataset...")
        
        # Load user queries
        queries1 = self.load_user_queries("R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\dataset\\user_queries.csv")
        queries2 = self.load_user_queries("R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\dataset\\user_queries1.csv")
        
        if queries1.empty and queries2.empty:
            logger.error("❌ No user queries loaded!")
            return
        
        # Combine queries
        all_queries = []
        
        if not queries1.empty:
            # Use raw_query column from user_queries.csv
            for idx, row in queries1.iterrows():
                query = row.get('raw_query', '')
                if query and isinstance(query, str):
                    all_queries.append(query)
        
        if not queries2.empty:
            # Use raw_query column from user_queries1.csv
            for idx, row in queries2.iterrows():
                query = row.get('raw_query', '')
                if query and isinstance(query, str):
                    all_queries.append(query)
        
        logger.info(f"📊 Processing {len(all_queries)} unique queries...")
        
        # Generate BIO format data
        bio_data = []
        query_id = 1
        
        for query in all_queries:
            if not query or not isinstance(query, str):
                continue
                
            # Extract entities
            entities = self.extract_entities(query)
            
            # Convert to BIO format
            bio_tokens = self.convert_to_bio_format(query, entities)
            
            # Add to dataset
            for token, tag in bio_tokens:
                bio_data.append({
                    'query_id': f'Q{query_id:06d}',
                    'token': token,
                    'tag': tag
                })
            
            query_id += 1
        
        # Create DataFrame and save
        df = pd.DataFrame(bio_data)
        df.to_csv(output_path, index=False)
        
        logger.info(f"✅ Generated NER dataset with {len(df)} tokens")
        logger.info(f"📂 Saved to: {output_path}")
        
        # Show statistics
        tag_counts = df['tag'].value_counts()
        logger.info("📊 Entity Distribution:")
        for tag, count in tag_counts.items():
            logger.info(f"   {tag}: {count}")
        
        return df

def main():
    """Main function to generate the NER dataset."""
    generator = NERDatasetGenerator()
    
    # Generate new dataset
    new_dataset = generator.generate_dataset("R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\dataset\\ner_dataset_new.csv")
    
    if new_dataset is not None:
        print("\n🎉 NER Dataset Generation Complete!")
        print("📊 The new dataset has been created and is ready for training.")
        print("📂 File: ner_dataset_new.csv")
    else:
        print("\n❌ Failed to generate NER dataset!")

if __name__ == "__main__":
    main() 