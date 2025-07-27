import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
import logging
import pickle
import os
import time
from collections import defaultdict
from datetime import datetime
import json
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge distillation loss for training smaller models.
    """
    
    def __init__(self, temperature: float = 2.0, alpha: float = 0.7):
        """
        Initialize knowledge distillation loss.
        
        Args:
            temperature: Temperature for softmax
            alpha: Weight for distillation vs task loss
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate knowledge distillation loss.
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            labels: Ground truth labels
            
        Returns:
            Combined loss
        """
        # Task loss (cross-entropy)
        task_loss = F.cross_entropy(student_logits, labels)
        
        # Distillation loss (KL divergence)
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        distillation_loss = self.kl_loss(student_probs, teacher_probs) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * task_loss
        
        return total_loss

class LightweightTransformer(nn.Module):
    """
    Lightweight transformer for knowledge distillation.
    """
    
    def __init__(self, vocab_size: int, hidden_size: int = 256, num_layers: int = 4, 
                 num_heads: int = 8, max_length: int = 128):
        """
        Initialize lightweight transformer.
        
        Args:
            vocab_size: Vocabulary size
            hidden_size: Hidden dimension size
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            max_length: Maximum sequence length
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Logits
        """
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        embeddings = token_embeds + position_embeds
        embeddings = self.layer_norm(embeddings)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)
        
        # Convert to transformer format
        src_key_padding_mask = (attention_mask == 0)
        
        # Pass through transformer
        transformer_output = self.transformer(embeddings, src_key_padding_mask=src_key_padding_mask)
        
        # Get logits
        logits = self.output_layer(transformer_output)
        
        return logits

class AttributeAwareMasking:
    """
    Attribute-aware masking for better completion suggestions.
    """
    
    def __init__(self):
        """Initialize attribute-aware masking."""
        # Define attribute patterns
        self.attribute_patterns = {
            'brand': ['samsung', 'apple', 'nike', 'adidas', 'sony', 'dell', 'hp'],
            'color': ['black', 'white', 'blue', 'red', 'green', 'pink', 'yellow'],
            'size': ['small', 'medium', 'large', 'xl', 'xxl'],
            'price_range': ['under', 'above', 'between', 'cheap', 'expensive'],
            'product_type': ['smartphone', 'laptop', 'shoes', 'headphones', 'watch']
        }
        
        # Define masking strategies
        self.masking_strategies = {
            'random': 0.3,      # 30% random masking
            'attribute': 0.4,   # 40% attribute-aware masking
            'position': 0.3     # 30% position-based masking
        }
    
    def create_masked_sequence(self, query: str, target_attributes: Optional[List[str]] = None) -> Tuple[str, int]:
        """
        Create masked sequence with attribute awareness.
        
        Args:
            query: Input query
            target_attributes: Target attributes to mask
            
        Returns:
            Tuple of (masked_query, masked_position)
        """
        words = query.lower().split()
        if len(words) < 2:
            return query, -1
        
        # Determine masking strategy
        strategy = random.choices(
            list(self.masking_strategies.keys()),
            weights=list(self.masking_strategies.values())
        )[0]
        
        if strategy == 'random':
            # Random masking
            mask_pos = random.randint(0, len(words) - 1)
            words[mask_pos] = '[MASK]'
            return ' '.join(words), mask_pos
        
        elif strategy == 'attribute':
            # Attribute-aware masking
            if target_attributes:
                # Mask specific attributes
                for attr in target_attributes:
                    for i, word in enumerate(words):
                        if word in self.attribute_patterns.get(attr, []):
                            words[i] = '[MASK]'
                            return ' '.join(words), i
            
            # Fallback: mask any attribute word
            for i, word in enumerate(words):
                for attr_patterns in self.attribute_patterns.values():
                    if word in attr_patterns:
                        words[i] = '[MASK]'
                        return ' '.join(words), i
            
            # If no attribute found, mask random word
            mask_pos = random.randint(0, len(words) - 1)
            words[mask_pos] = '[MASK]'
            return ' '.join(words), mask_pos
        
        else:  # position-based
            # Position-based masking (prefer end positions)
            weights = [1.0] * len(words)
            weights[-1] = 2.0  # Higher weight for last position
            if len(words) > 1:
                weights[-2] = 1.5  # Higher weight for second-to-last
            
            mask_pos = random.choices(range(len(words)), weights=weights)[0]
            words[mask_pos] = '[MASK]'
            return ' '.join(words), mask_pos

class EnhancedBERTCompletion:
    """
    Enhanced BERT completion with knowledge distillation and attribute-aware masking.
    """
    
    def __init__(self, 
                 model_name: str = 'bert-base-uncased',
                 enable_distillation: bool = True,
                 enable_attribute_masking: bool = True,
                 cache_dir: str = 'cache',
                 max_length: int = 128,
                 temperature: float = 1.0,
                 top_k: int = 10,
                 top_p: float = 0.9):
        """
        Initialize enhanced BERT completion.
        
        Args:
            model_name: BERT model name
            enable_distillation: Enable knowledge distillation
            enable_attribute_masking: Enable attribute-aware masking
            cache_dir: Directory for caching
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
        """
        self.enable_distillation = enable_distillation
        self.enable_attribute_masking = enable_attribute_masking
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        # Initialize components
        logger.info(f"Loading BERT model: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name)
        
        if enable_distillation:
            self._setup_distillation()
        
        if enable_attribute_masking:
            self.attribute_masking = AttributeAwareMasking()
        
        # Cache for completions
        self.completion_cache = {}
        
        # Statistics
        self.total_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load cached data
        self.load_cached_data()
    
    def _setup_distillation(self):
        """Setup knowledge distillation components."""
        logger.info("Setting up knowledge distillation...")
        
        # Create student model
        vocab_size = self.tokenizer.vocab_size
        self.student_model = LightweightTransformer(
            vocab_size=vocab_size,
            hidden_size=256,
            num_layers=4,
            num_heads=8,
            max_length=self.max_length
        )
        
        # Create distillation loss
        self.distillation_loss = KnowledgeDistillationLoss(temperature=2.0, alpha=0.7)
        
        # Freeze teacher model
        for param in self.model.parameters():
            param.requires_grad = False
        
        logger.info("Knowledge distillation setup completed")
    
    def load_cached_data(self):
        """Load cached completions."""
        cache_file = os.path.join(self.cache_dir, 'bert_completion_cache.pkl')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.completion_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.completion_cache)} cached completions")
            except Exception as e:
                logger.warning(f"Failed to load cached data: {e}")
    
    def save_cached_data(self):
        """Save completions to cache."""
        cache_file = os.path.join(self.cache_dir, 'bert_completion_cache.pkl')
        with open(cache_file, 'wb') as f:
            pickle.dump(self.completion_cache, f)
        logger.info("Saved cached data")
    
    def generate_training_data(self, queries: List[str], 
                             target_attributes: Optional[List[str]] = None) -> List[Dict]:
        """
        Generate training data with attribute-aware masking.
        
        Args:
            queries: List of training queries
            target_attributes: Target attributes for masking
            
        Returns:
            List of training examples
        """
        logger.info(f"Generating training data from {len(queries)} queries...")
        
        training_data = []
        
        for query in queries:
            if not query or len(query.split()) < 2:
                continue
            
            # Create masked sequence
            if self.enable_attribute_masking:
                masked_query, mask_pos = self.attribute_masking.create_masked_sequence(
                    query, target_attributes
                )
            else:
                # Simple random masking
                words = query.split()
                mask_pos = random.randint(0, len(words) - 1)
                words[mask_pos] = '[MASK]'
                masked_query = ' '.join(words)
            
            # Get original word
            original_words = query.split()
            if 0 <= mask_pos < len(original_words):
                original_word = original_words[mask_pos]
                
                training_data.append({
                    'masked_query': masked_query,
                    'original_word': original_word,
                    'mask_position': mask_pos,
                    'original_query': query
                })
        
        logger.info(f"Generated {len(training_data)} training examples")
        return training_data
    
    def train_student_model(self, training_data: List[Dict], 
                          epochs: int = 3, batch_size: int = 32,
                          learning_rate: float = 1e-4):
        """
        Train student model using knowledge distillation.
        
        Args:
            training_data: Training data
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        if not self.enable_distillation:
            logger.warning("Knowledge distillation not enabled")
            return
        
        logger.info(f"Training student model for {epochs} epochs...")
        
        # Setup training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.student_model.to(device)
        self.model.to(device)
        
        optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # Shuffle training data
            random.shuffle(training_data)
            
            for i in range(0, len(training_data), batch_size):
                batch_data = training_data[i:i + batch_size]
                
                # Prepare batch
                masked_queries = [item['masked_query'] for item in batch_data]
                original_words = [item['original_word'] for item in batch_data]
                
                # Tokenize
                inputs = self.tokenizer(
                    masked_queries,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Get labels
                labels = self.tokenizer(
                    original_words,
                    padding=True,
                    truncation=True,
                    max_length=1,
                    return_tensors='pt'
                )['input_ids'].squeeze(-1)
                
                # Move to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)
                
                # Forward pass
                with torch.no_grad():
                    teacher_outputs = self.model(**inputs)
                    teacher_logits = teacher_outputs.logits
                
                student_logits = self.student_model(inputs['input_ids'], inputs['attention_mask'])
                
                # Calculate loss
                loss = self.distillation_loss(student_logits, teacher_logits, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            logger.info(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        logger.info("Student model training completed")
    
    def get_completions(self, prefix: str, max_completions: int = 5,
                       target_attributes: Optional[List[str]] = None) -> List[Dict]:
        """
        Get completions for a prefix.
        
        Args:
            prefix: Query prefix
            max_completions: Maximum number of completions
            target_attributes: Target attributes for completion
            
        Returns:
            List of completion dictionaries
        """
        self.total_queries += 1
        prefix = prefix.strip()
        
        if not prefix:
            return []
        
        # Check cache first
        cache_key = f"{prefix}_{max_completions}_{str(target_attributes)}"
        if cache_key in self.completion_cache:
            self.cache_hits += 1
            return self.completion_cache[cache_key]
        
        self.cache_misses += 1
        
        # Create masked sequence
        if self.enable_attribute_masking:
            masked_query, mask_pos = self.attribute_masking.create_masked_sequence(
                prefix, target_attributes
            )
        else:
            # Simple masking at the end
            masked_query = prefix + " [MASK]"
            mask_pos = len(prefix.split())
        
        # Get completions
        completions = self._generate_completions(masked_query, mask_pos, max_completions)
        
        # Cache results
        self.completion_cache[cache_key] = completions
        
        return completions
    
    def _generate_completions(self, masked_query: str, mask_pos: int, 
                            max_completions: int) -> List[Dict]:
        """
        Generate completions for masked query.
        
        Args:
            masked_query: Query with [MASK] token
            mask_pos: Position of mask
            max_completions: Maximum number of completions
            
        Returns:
            List of completion dictionaries
        """
        try:
            # Tokenize
            inputs = self.tokenizer(
                masked_query,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            
            # Get model predictions
            if self.enable_distillation and hasattr(self, 'student_model'):
                # Use student model
                device = next(self.student_model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.student_model(inputs['input_ids'], inputs['attention_mask'])
                    logits = outputs
            else:
                # Use teacher model
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
            
            # Get predictions for mask position
            mask_token_id = self.tokenizer.mask_token_id
            mask_positions = (inputs['input_ids'][0] == mask_token_id).nonzero(as_tuple=True)[0]
            
            if len(mask_positions) == 0:
                return []
            
            mask_pos_idx = mask_positions[0]
            mask_logits = logits[0, mask_pos_idx, :]
            
            # Apply temperature and sampling
            mask_logits = mask_logits / self.temperature
            
            # Top-k filtering
            if self.top_k > 0:
                top_k_logits, top_k_indices = torch.topk(mask_logits, min(self.top_k, mask_logits.size(-1)))
                mask_logits = torch.full_like(mask_logits, float('-inf'))
                mask_logits.scatter_(0, top_k_indices, top_k_logits)
            
            # Top-p (nucleus) sampling
            if self.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(mask_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > self.top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                mask_logits[indices_to_remove] = float('-inf')
            
            # Get probabilities
            probs = F.softmax(mask_logits, dim=-1)
            
            # Sample completions
            completions = []
            for _ in range(max_completions):
                # Sample token
                token_id = torch.multinomial(probs, 1).item()
                token = self.tokenizer.decode([token_id])
                
                # Create completion
                completion_text = masked_query.replace('[MASK]', token)
                
                # Calculate confidence
                confidence = probs[token_id].item()
                
                completions.append({
                    'completion': completion_text,
                    'token': token,
                    'confidence': confidence,
                    'method': 'distillation' if self.enable_distillation else 'bert'
                })
            
            # Sort by confidence
            completions.sort(key=lambda x: x['confidence'], reverse=True)
            
            return completions
            
        except Exception as e:
            logger.error(f"Failed to generate completions: {e}")
            return []
    
    def get_attribute_suggestions(self, prefix: str, attribute_type: str, 
                                max_suggestions: int = 5) -> List[Dict]:
        """
        Get attribute-specific suggestions.
        
        Args:
            prefix: Query prefix
            attribute_type: Type of attribute (brand, color, size, etc.)
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of attribute suggestions
        """
        if not self.enable_attribute_masking:
            return []
        
        # Get attribute patterns
        attribute_patterns = self.attribute_masking.attribute_patterns.get(attribute_type, [])
        
        if not attribute_patterns:
            return []
        
        # Create masked query with attribute focus
        masked_query = prefix + " [MASK]"
        
        # Get completions
        completions = self._generate_completions(masked_query, len(prefix.split()), max_suggestions * 2)
        
        # Filter for attribute matches
        attribute_suggestions = []
        for completion in completions:
            token = completion['token'].lower().strip()
            if token in attribute_patterns:
                attribute_suggestions.append({
                    'suggestion': completion['completion'],
                    'attribute': token,
                    'confidence': completion['confidence'],
                    'attribute_type': attribute_type
                })
        
        # Sort and limit
        attribute_suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        return attribute_suggestions[:max_suggestions]
    
    def get_statistics(self) -> Dict:
        """Get completion statistics."""
        return {
            'total_queries': self.total_queries,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'cache_size': len(self.completion_cache),
            'distillation_enabled': self.enable_distillation,
            'attribute_masking_enabled': self.enable_attribute_masking
        }
    
    def clear_cache(self):
        """Clear completion cache."""
        self.completion_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Cleared completion cache")
    
    def save_model(self, model_path: str):
        """Save trained student model."""
        if self.enable_distillation and hasattr(self, 'student_model'):
            os.makedirs(model_path, exist_ok=True)
            torch.save(self.student_model.state_dict(), os.path.join(model_path, 'student_model.pth'))
            logger.info(f"Saved student model to {model_path}")
    
    def load_model(self, model_path: str):
        """Load trained student model."""
        if self.enable_distillation and hasattr(self, 'student_model'):
            model_file = os.path.join(model_path, 'student_model.pth')
            if os.path.exists(model_file):
                self.student_model.load_state_dict(torch.load(model_file))
                logger.info(f"Loaded student model from {model_path}")

# Example usage and testing
if __name__ == "__main__":
    # Create enhanced BERT completion
    bert_completion = EnhancedBERTCompletion(
        enable_distillation=True,
        enable_attribute_masking=True,
        temperature=1.0,
        top_k=10,
        top_p=0.9
    )
    
    # Sample training data
    sample_queries = [
        "samsung galaxy smartphone",
        "apple iphone black",
        "nike shoes red",
        "dell laptop gaming",
        "sony headphones wireless",
        "adidas sneakers white",
        "hp laptop under 50000",
        "canon camera professional"
    ]
    
    # Generate training data
    training_data = bert_completion.generate_training_data(
        sample_queries, 
        target_attributes=['brand', 'color', 'product_type']
    )
    
    # Train student model (uncomment to train)
    # bert_completion.train_student_model(training_data, epochs=2)
    
    # Test completions
    test_prefixes = [
        "samsung",
        "apple iphone",
        "nike shoes",
        "dell laptop",
        "sony"
    ]
    
    print("Testing Enhanced BERT Completion:")
    print("=" * 60)
    
    for prefix in test_prefixes:
        completions = bert_completion.get_completions(prefix, max_completions=3)
        
        print(f"\nPrefix: '{prefix}'")
        print(f"Completions ({len(completions)}):")
        for i, completion in enumerate(completions, 1):
            print(f"  {i}. '{completion['completion']}' "
                  f"(confidence: {completion['confidence']:.3f}, "
                  f"method: {completion['method']})")
    
    # Test attribute-specific suggestions
    print("\n" + "=" * 60)
    print("Testing Attribute-Specific Suggestions:")
    
    attribute_tests = [
        ("samsung", "brand"),
        ("nike", "color"),
        ("laptop", "brand")
    ]
    
    for prefix, attr_type in attribute_tests:
        suggestions = bert_completion.get_attribute_suggestions(prefix, attr_type, max_suggestions=3)
        
        print(f"\nAttribute suggestions for '{prefix}' ({attr_type}):")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. '{suggestion['suggestion']}' "
                  f"(attribute: {suggestion['attribute']}, "
                  f"confidence: {suggestion['confidence']:.3f})")
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Statistics:")
    stats = bert_completion.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save data
    bert_completion.save_cached_data() 