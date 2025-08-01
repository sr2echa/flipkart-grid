#!/usr/bin/env python3
"""
Grid 7.0 - NER Model Training Script
====================================

This script trains a spaCy NER model using the new comprehensive dataset
generated from user_queries.csv and user_queries1.csv.

The new model will replace the old one and provide better entity recognition
for brands, products, colors, sizes, materials, and other relevant entities.
"""

import pandas as pd
import spacy
from spacy.training import Example
import logging
import os
import shutil
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_bio_to_spacy_format(csv_path: str) -> List[tuple]:
    """
    Convert BIO format CSV to spaCy training format.
    
    Args:
        csv_path: Path to the BIO format CSV file
        
    Returns:
        List of (text, {"entities": [(start, end, label)]}) tuples
    """
    logger.info(f"📊 Loading dataset from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"✅ Loaded {len(df)} tokens from dataset")
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        return []
    
    data = []
    grouped = df.groupby("query_id")
    
    logger.info(f"🔄 Converting {len(grouped)} queries to spaCy format...")
    
    for query_id, group in grouped:
        tokens = group["token"].tolist()
        tags = group["tag"].tolist()
        
        # Reconstruct the original text
        text = " ".join(tokens)
        entities = []
        
        current_pos = 0
        for token, tag in zip(tokens, tags):
            if tag.startswith("B-"):
                # Find the start position of this token in the text
                start = text.find(token, current_pos)
                if start != -1:
                    end = start + len(token)
                    entity_label = tag[2:]  # Remove "B-" prefix
                    entities.append((start, end, entity_label))
                    current_pos = end
            elif tag.startswith("I-"):
                # Continue the previous entity
                if entities:
                    # Extend the last entity
                    last_start, last_end, last_label = entities[-1]
                    start = text.find(token, last_end)
                    if start != -1:
                        end = start + len(token)
                        entities[-1] = (last_start, end, last_label)
                        current_pos = end
        
        if entities:
            data.append((text, {"entities": entities}))
    
    logger.info(f"✅ Converted {len(data)} training examples")
    return data

def train_spacy_ner(data_path: str, output_dir: str = "spacy_ner_model", 
                    iterations: int = 30, dropout: float = 0.3):
    """
    Train a spaCy NER model with the provided dataset.
    
    Args:
        data_path: Path to the BIO format CSV dataset
        output_dir: Directory to save the trained model
        iterations: Number of training iterations
        dropout: Dropout rate for training
    """
    logger.info("🚀 Starting NER model training...")
    logger.info("=" * 60)
    logger.info(f"📂 Dataset: {data_path}")
    logger.info(f"📂 Output: {output_dir}")
    logger.info(f"🔄 Iterations: {iterations}")
    logger.info(f"🎯 Dropout: {dropout}")
    logger.info("=" * 60)
    
    # Convert dataset to spaCy format
    training_data = convert_bio_to_spacy_format(data_path)
    
    if not training_data:
        logger.error("❌ No training data available!")
        return False
    
    # Create a blank spaCy model
    logger.info("🧠 Creating blank spaCy model...")
    nlp = spacy.blank("en")
    
    # Add NER component
    ner = nlp.add_pipe("ner")
    
    # Add all entity labels from the training data
    entity_labels = set()
    for _, annotations in training_data:
        for ent in annotations["entities"]:
            entity_labels.add(ent[2])
    
    for label in entity_labels:
        ner.add_label(label)
    
    logger.info(f"🏷️ Added {len(entity_labels)} entity labels: {list(entity_labels)}")
    
    # Begin training
    logger.info("🎯 Starting training...")
    optimizer = nlp.begin_training()
    
    # Training loop
    for iteration in range(iterations):
        losses = {}
        
        # Shuffle the training data
        import random
        random.shuffle(training_data)
        
        for text, annotations in training_data:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=dropout, sgd=optimizer, losses=losses)
        
        # Log progress
        if (iteration + 1) % 5 == 0:
            logger.info(f"📈 Iteration {iteration + 1}/{iterations} - Loss: {losses.get('ner', 0):.4f}")
    
    # Save the trained model
    logger.info("💾 Saving trained model...")
    
    # Remove old model if it exists
    if os.path.exists(output_dir):
        logger.info(f"🗑️ Removing old model from: {output_dir}")
        shutil.rmtree(output_dir)
    
    # Save new model
    nlp.to_disk(output_dir)
    logger.info(f"✅ NER model saved to: {output_dir}")
    
    # Test the model
    logger.info("🧪 Testing the trained model...")
    test_model = spacy.load(output_dir)
    
    # Test with a few examples
    test_queries = [
        "red nike running shoes",
        "bluetooth headphones sony",
        "leather wallet for men",
        "smartphone samsung galaxy"
    ]
    
    logger.info("📋 Model Test Results:")
    for query in test_queries:
        doc = test_model(query)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        logger.info(f"   '{query}' -> {entities}")
    
    logger.info("🎉 NER model training completed successfully!")
    return True

def main():
    """Main function to train the NER model."""
    # Configuration
    DATASET_PATH = "R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\dataset\\ner_dataset_new.csv"
    MODEL_OUTPUT_DIR = "R:\\sem VII\\Flipkart Grid 7.0\\flipkart-grid\\searchresultpage\\spacy_ner_model"
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        logger.error(f"❌ Dataset not found: {DATASET_PATH}")
        logger.error("Please run generate_ner_dataset.py first!")
        return
    
    # Train the model
    success = train_spacy_ner(
        data_path=DATASET_PATH,
        output_dir=MODEL_OUTPUT_DIR,
        iterations=30,
        dropout=0.3
    )
    
    if success:
        logger.info("🎉 NER model training completed!")
        logger.info(f"📂 New model saved to: {MODEL_OUTPUT_DIR}")
        logger.info("🔄 The old model has been replaced with the new one.")
    else:
        logger.error("❌ NER model training failed!")

if __name__ == "__main__":
    main()