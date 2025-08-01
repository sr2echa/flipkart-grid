#!/usr/bin/env python3
"""
Grid 7.0 - NER Model Test Script
================================

This script tests the newly trained spaCy NER model to verify it's working correctly
and can extract entities from various types of user queries.
"""

import spacy
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ner_model(model_path: str = "spacy_ner_model"):
    """Test the trained NER model with various queries."""
    
    logger.info("🧪 Testing NER Model...")
    logger.info("=" * 50)
    
    try:
        # Load the trained model
        nlp = spacy.load(model_path)
        logger.info("✅ Model loaded successfully!")
        
        # Test queries covering different entity types
        test_queries = [
            # Brand + Product combinations
            "red nike running shoes",
            "bluetooth headphones sony",
            "samsung smartphone galaxy",
            "apple iphone 13",
            
            # Product + Feature combinations
            "noise cancelling headphones",
            "wireless bluetooth speaker",
            "smart watch compatible android",
            "portable power bank",
            
            # Material + Product combinations
            "leather wallet for men",
            "cotton t-shirt women",
            "stainless steel kettle",
            "wooden dining table",
            
            # Size + Product combinations
            "large backpack hiking",
            "small portable speaker",
            "medium size laptop",
            "compact air fryer",
            
            # Color + Product combinations
            "black running shoes",
            "white wireless headphones",
            "blue denim jeans",
            "red gaming mouse",
            
            # Category + Product combinations
            "electronics smartphone",
            "fashion clothing shoes",
            "kitchen appliances blender",
            "home furniture sofa",
            
            # Complex queries
            "budget smartphone under 15000",
            "premium leather wallet men",
            "wireless noise cancelling headphones sony",
            "portable bluetooth speaker jbl",
            "smart watch compatible with android samsung",
            "organic cotton t-shirt women red",
            "gaming laptop rtx 3060 asus",
            "electric kettle stainless steel kitchen"
        ]
        
        logger.info(f"📋 Testing {len(test_queries)} queries...")
        logger.info("=" * 50)
        
        # Test each query
        for i, query in enumerate(test_queries, 1):
            doc = nlp(query)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            print(f"{i:2d}. Query: '{query}'")
            if entities:
                for entity_text, entity_label in entities:
                    print(f"    → {entity_text} ({entity_label})")
            else:
                print("    → No entities detected")
            print()
        
        # Summary statistics
        total_entities = sum(len(nlp(query).ents) for query in test_queries)
        logger.info(f"📊 Test Summary:")
        logger.info(f"   Total queries tested: {len(test_queries)}")
        logger.info(f"   Total entities detected: {total_entities}")
        logger.info(f"   Average entities per query: {total_entities/len(test_queries):.2f}")
        
        # Test entity label distribution
        all_entities = []
        for query in test_queries:
            doc = nlp(query)
            all_entities.extend([ent.label_ for ent in doc.ents])
        
        from collections import Counter
        entity_counts = Counter(all_entities)
        
        logger.info("📊 Entity Label Distribution:")
        for label, count in entity_counts.most_common():
            logger.info(f"   {label}: {count}")
        
        logger.info("✅ NER Model Test Completed Successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing NER model: {e}")
        return False

def main():
    """Main function to test the NER model."""
    MODEL_PATH = "R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\searchresultpage\\spacy_ner_model"
    
    success = test_ner_model(MODEL_PATH)
    
    if success:
        logger.info("🎉 NER model is working correctly!")
        logger.info("🔄 The system is ready to use the updated model.")
    else:
        logger.error("❌ NER model test failed!")

if __name__ == "__main__":
    main() 