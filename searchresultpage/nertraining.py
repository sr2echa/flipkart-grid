# train_ner.py - Fixed version

import pandas as pd
import numpy as np
import json
from typing import List, Dict
import torch
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

class EcommerceNERDataPreparator:
    """
    Prepares NER training data from a CSV file.
    """
    def load_dataset(self, csv_path: str) -> List[Dict]:
        print("Loading NER dataset...")
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} rows from {csv_path}")
            training_data = []
            for query_id in df['query_id'].unique():
                query_df = df[df['query_id'] == query_id]
                tokens = query_df['token'].tolist()
                labels = query_df['tag'].tolist()
                if len(tokens) > 1:
                    training_data.append({'query_id': query_id, 'tokens': tokens, 'labels': labels})
            print(f"Created {len(training_data)} training examples.")
            return training_data
        except FileNotFoundError:
            print(f"Error: Dataset not found at '{csv_path}'. Using sample data.")
            return self._create_sample_data()

    def _create_sample_data(self) -> List[Dict]:
        return [
            {'query_id': 'SAMPLE001', 'tokens': ['samsung', 'phone', 'under', '20000'], 'labels': ['B-BRAND', 'B-PRODUCT', 'B-PRICE', 'I-PRICE']},
            {'query_id': 'SAMPLE002', 'tokens': ['blue', 'nike', 'shoes', 'for', 'men'], 'labels': ['B-COLOR', 'B-BRAND', 'B-PRODUCT', 'O', 'B-GENDER']},
            {'query_id': 'SAMPLE003', 'tokens': ['formal', 'leather', 'shoes'], 'labels': ['B-STYLE', 'B-MATERIAL', 'B-PRODUCT']},
            {'query_id': 'SAMPLE004', 'tokens': ['gaming', 'laptop', '16gb', 'ram'], 'labels': ['B-CATEGORY', 'B-PRODUCT', 'B-SPEC', 'I-SPEC']}
        ]

    def get_unique_labels(self, training_data: List[Dict]) -> List[str]:
        labels = set(['O'])
        for example in training_data:
            labels.update(example['labels'])
        return sorted(list(labels))

class NERDataset(Dataset):
    """PyTorch Dataset for NER training."""
    def __init__(self, training_data: List[Dict], tokenizer, label2id: Dict, max_length: int = 128):
        self.training_data = training_data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        example = self.training_data[idx]
        tokens = example['tokens']
        labels = example['labels']
        
        # Tokenize with proper return_tensors handling
        tokenized = self.tokenizer(
            tokens, 
            is_split_into_words=True, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length,
            return_tensors=None  # Return lists instead of tensors
        )
        
        word_ids = tokenized.word_ids()
        aligned_labels = []
        previous_word_idx = None
        
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != previous_word_idx:
                # First token of a word
                if word_id < len(labels):
                    aligned_labels.append(self.label2id.get(labels[word_id], self.label2id['O']))
                else:
                    aligned_labels.append(self.label2id['O'])
            else:
                # Continuation of a word - convert B- to I-
                if word_id < len(labels):
                    label = labels[word_id]
                    if label.startswith('B-'):
                        label = 'I-' + label[2:]
                    aligned_labels.append(self.label2id.get(label, self.label2id['O']))
                else:
                    aligned_labels.append(self.label2id['O'])
            previous_word_idx = word_id
        
        return {
            'input_ids': torch.tensor(tokenized['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(tokenized['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }

class EcommerceNERTrainer:
    """Handles the model fine-tuning process."""
    def __init__(self, model_name: str, labels: List[str]):
        self.model_name = model_name
        self.unique_labels = labels
        self.label2id = {label: i for i, label in enumerate(self.unique_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        
        print(f"\nSetting up model: {self.model_name}")
        print(f"Number of labels: {len(self.unique_labels)}")
        print(f"Labels: {self.unique_labels}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, 
            num_labels=len(self.unique_labels), 
            id2label=self.id2label, 
            label2id=self.label2id
        )

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100] 
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100] 
            for prediction, label in zip(predictions, labels)
        ]
        
        # Flatten lists
        flat_true = [label for sublist in true_labels for label in sublist]
        flat_pred = [label for sublist in true_predictions for label in sublist]
        
        if len(flat_true) == 0 or len(flat_pred) == 0:
            return {"f1_weighted": 0.0}
        
        try:
            f1_weighted = f1_score(flat_true, flat_pred, average='weighted', zero_division=0)
            accuracy = sum(t == p for t, p in zip(flat_true, flat_pred)) / len(flat_true)
            return {
                "f1_weighted": f1_weighted,
                "accuracy": accuracy
            }
        except Exception as e:
            print(f"Error computing metrics: {e}")
            return {"f1_weighted": 0.0, "accuracy": 0.0}

    def train(self, train_dataset: Dataset, val_dataset: Dataset, output_dir: str = "./ecommerce-ner-model"):
        """Configures and runs the training loop."""
        print("\nConfiguring training...")
        
        # Updated training arguments for compatibility
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=8,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,
            eval_strategy="steps",  # Changed from evaluation_strategy
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="f1_weighted",
            greater_is_better=True,
            save_total_limit=2,
            report_to="none",
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            remove_unused_columns=False,  # Keep all columns
        )

        data_collator = DataCollatorForTokenClassification(
            self.tokenizer,
            padding=True,
            return_tensors="pt"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        print("Starting training...🚀")
        try:
            trainer.train()
            
            print(f"\nTraining complete! Saving best model to '{output_dir}'")
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            # Save label mappings
            with open(f"{output_dir}/label_mappings.json", 'w') as f:
                json.dump({
                    'label2id': self.label2id, 
                    'id2label': self.id2label,
                    'unique_labels': self.unique_labels
                }, f, indent=2)
            
            # Save training info
            eval_results = trainer.evaluate()
            with open(f"{output_dir}/training_info.json", 'w') as f:
                json.dump({
                    'model_name': self.model_name,
                    'num_labels': len(self.unique_labels),
                    'training_examples': len(train_dataset),
                    'validation_examples': len(val_dataset),
                    'final_eval_results': eval_results
                }, f, indent=2)
                
            print(f"Final evaluation results: {eval_results}")
            
        except Exception as e:
            print(f"Training failed with error: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function to run the entire training pipeline."""
    print("="*60)
    print("E-COMMERCE NER MODEL TRAINING")
    print("="*60)
    
    preparator = EcommerceNERDataPreparator()
    
    # Try to load the dataset - adjust path as needed
    dataset_paths = [
        "R:\\sem VII\\Flipkart Grid 7.0\\ner_dataset.csv",
        "ner_dataset.csv",
        "./ner_dataset.csv"
    ]
    
    training_data = None
    for path in dataset_paths:
        try:
            training_data = preparator.load_dataset(path)
            if training_data:
                break
        except:
            continue
    
    if not training_data:
        print("Could not find dataset, using sample data")
        training_data = preparator._create_sample_data()
    
    unique_labels = preparator.get_unique_labels(training_data)
    print(f"\nFinal label set found: {unique_labels}")
    print(f"Total training examples: {len(training_data)}")

    # Split data
    train_data, val_data = train_test_split(training_data, test_size=0.2, random_state=42)
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    
    # Initialize trainer
    ner_trainer = EcommerceNERTrainer(model_name="distilbert-base-uncased", labels=unique_labels)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = NERDataset(train_data, ner_trainer.tokenizer, ner_trainer.label2id)
    val_dataset = NERDataset(val_data, ner_trainer.tokenizer, ner_trainer.label2id)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # Start training
    ner_trainer.train(train_dataset, val_dataset)
    print("\n✅ NER model fine-tuning finished successfully!")

if __name__ == "__main__":
    main()