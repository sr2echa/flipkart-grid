#!/usr/bin/env python3
"""
Context-Aware Suggestion Completion using BERT/DistilBERT
==========================================================

Implements masked language modeling for intelligent query completion
with persona and context awareness.
"""

import logging
import torch
from typing import List, Tuple, Dict, Optional
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompletionSystem:
    """
    BERT-based completion system for intelligent query suggestions.
    Uses masked language modeling to complete short prefixes with context awareness.
    """
    
    def __init__(self, model_name='distilbert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        
        # E-commerce specific completion templates
        self.completion_templates = {
            'product_search': [
                "{prefix} for men",
                "{prefix} for women", 
                "{prefix} with free delivery",
                "{prefix} under 1000",
                "{prefix} best quality",
                "{prefix} latest",
                "{prefix} online shopping"
            ],
            'brand_search': [
                "{prefix} shoes",
                "{prefix} clothing",
                "{prefix} electronics",
                "{prefix} mobile phone",
                "{prefix} laptop",
                "{prefix} watch"
            ],
            'category_search': [
                "best {prefix}",
                "cheap {prefix}",
                "branded {prefix}",
                "{prefix} with discount",
                "{prefix} sale",
                "{prefix} offer"
            ]
        }
        
        # Persona-specific context words
        self.persona_contexts = {
            'tech_enthusiast': ['latest', 'advanced', 'premium', 'features', 'specifications', 'performance'],
            'fashion_lover': ['trendy', 'stylish', 'designer', 'fashionable', 'elegant', 'chic'],
            'budget_shopper': ['cheap', 'affordable', 'budget', 'discount', 'sale', 'offer'],
            'sports_enthusiast': ['sports', 'fitness', 'athletic', 'performance', 'durable', 'professional']
        }
        
        logger.info(f"🔧 Initializing Completion System with {model_name}")
    
    def _load_model(self):
        """Load BERT/DistilBERT model and tokenizer."""
        if self.model_loaded:
            return True
        
        try:
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            
            logger.info(f"📥 Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            self.model_loaded = True
            logger.info(f"✅ Model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def complete_prefix(self, prefix: str, context: Optional[Dict] = None, max_completions: int = 5) -> List[Tuple[str, float]]:
        """
        Complete a short prefix using masked language modeling with context awareness.
        
        Args:
            prefix: Short query prefix (e.g., "blu", "nike sho")
            context: User context including persona, location, etc.
            max_completions: Maximum number of completions to return
            
        Returns:
            List of (completed_query, confidence_score) tuples
        """
        if not prefix or len(prefix.strip()) == 0:
            return []
        
        prefix = prefix.lower().strip()
        
        # If prefix is too long, use template-based completion
        if len(prefix) > 10:
            return self._template_based_completion(prefix, context, max_completions)
        
        # Load model if not loaded
        if not self._load_model():
            return self._fallback_completion(prefix, context, max_completions)
        
        try:
            completions = []
            
            # Method 1: Masked LM completion
            masked_completions = self._masked_lm_completion(prefix, context, max_completions // 2)
            completions.extend(masked_completions)
            
            # Method 2: Template-based completion with persona context
            template_completions = self._template_based_completion(prefix, context, max_completions // 2)
            completions.extend(template_completions)
            
            # Remove duplicates and sort by confidence
            seen = set()
            unique_completions = []
            for completion, score in completions:
                completion_clean = completion.lower().strip()
                if completion_clean not in seen and completion_clean != prefix:
                    seen.add(completion_clean)
                    unique_completions.append((completion, score))
            
            unique_completions.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"🔍 Generated {len(unique_completions)} completions for '{prefix}'")
            return unique_completions[:max_completions]
            
        except Exception as e:
            logger.error(f"❌ Error generating completions: {e}")
            return self._fallback_completion(prefix, context, max_completions)
    
    def _masked_lm_completion(self, prefix: str, context: Optional[Dict], max_completions: int) -> List[Tuple[str, float]]:
        """Use BERT masked language modeling for completion."""
        completions = []
        
        try:
            # Get persona context words
            persona_id = context.get('persona', 'tech_enthusiast') if context else 'tech_enthusiast'
            persona_words = self.persona_contexts.get(persona_id, [])
            
            # Create masked input patterns
            patterns = [
                f"{prefix} [MASK]",
                f"{prefix} [MASK] [MASK]", 
                f"[MASK] {prefix}",
                f"best {prefix} [MASK]"
            ]
            
            for pattern in patterns:
                try:
                    # Tokenize
                    inputs = self.tokenizer(pattern, return_tensors="pt", padding=True, truncation=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get predictions
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        predictions = outputs.logits
                    
                    # Find mask positions
                    mask_positions = (inputs['input_ids'] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
                    
                    if len(mask_positions) == 0:
                        continue
                    
                    # Get top predictions for first mask
                    mask_pos = mask_positions[0]
                    mask_predictions = predictions[0, mask_pos]
                    top_predictions = torch.topk(mask_predictions, k=10)
                    
                    # Convert to completions
                    for score, token_id in zip(top_predictions.values, top_predictions.indices):
                        token = self.tokenizer.decode(token_id.item()).strip()
                        if len(token) > 1 and token.isalpha():  # Valid word
                            completed = pattern.replace('[MASK]', token, 1)
                            completed = re.sub(r'\[MASK\]', '', completed).strip()
                            completed = re.sub(r'\s+', ' ', completed)
                            
                            confidence = float(torch.softmax(mask_predictions, dim=-1)[token_id])
                            
                            # Boost score if contains persona-relevant words
                            if any(word in completed.lower() for word in persona_words):
                                confidence *= 1.2
                            
                            completions.append((completed, confidence))
                
                except Exception as e:
                    logger.debug(f"Pattern '{pattern}' failed: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"❌ Masked LM completion failed: {e}")
        
        return completions[:max_completions]
    
    def _template_based_completion(self, prefix: str, context: Optional[Dict], max_completions: int) -> List[Tuple[str, float]]:
        """Generate completions using predefined templates and context."""
        completions = []
        
        try:
            # Get persona for context
            persona_id = context.get('persona', 'tech_enthusiast') if context else 'tech_enthusiast'
            persona_words = self.persona_contexts.get(persona_id, [])
            
            # Determine template category
            template_category = self._classify_prefix(prefix)
            templates = self.completion_templates.get(template_category, self.completion_templates['product_search'])
            
            # Generate completions from templates
            base_score = 0.6
            for template in templates:
                completed = template.format(prefix=prefix)
                
                # Adjust score based on persona relevance
                score = base_score
                if any(word in completed.lower() for word in persona_words):
                    score += 0.2
                
                # Add location context if available
                if context and 'location' in context:
                    location = context['location']
                    if location.lower() in ['mumbai', 'delhi', 'bangalore']:
                        completed += f" in {location}"
                        score += 0.1
                
                completions.append((completed, score))
            
            # Add persona-specific completions
            for word in persona_words[:3]:  # Top 3 persona words
                persona_completion = f"{prefix} {word}"
                completions.append((persona_completion, base_score + 0.15))
            
        except Exception as e:
            logger.error(f"❌ Template completion failed: {e}")
        
        return completions[:max_completions]
    
    def _classify_prefix(self, prefix: str) -> str:
        """Classify prefix to determine appropriate template category."""
        prefix_lower = prefix.lower()
        
        # Brand names
        brands = ['nike', 'adidas', 'apple', 'samsung', 'sony', 'lg', 'hp', 'dell']
        if any(brand in prefix_lower for brand in brands):
            return 'brand_search'
        
        # Category indicators
        categories = ['shoes', 'shirt', 'phone', 'laptop', 'watch', 'bag']
        if any(cat in prefix_lower for cat in categories):
            return 'category_search'
        
        return 'product_search'
    
    def _fallback_completion(self, prefix: str, context: Optional[Dict], max_completions: int) -> List[Tuple[str, float]]:
        """Fallback completion when BERT model is not available."""
        logger.info("🔄 Using fallback completion method")
        
        # Simple rule-based completions
        fallback_completions = [
            f"{prefix} for men",
            f"{prefix} for women",
            f"{prefix} online",
            f"best {prefix}",
            f"{prefix} with free delivery"
        ]
        
        return [(comp, 0.3) for comp in fallback_completions[:max_completions]]

def main():
    """Test the completion system."""
    completion_system = CompletionSystem()
    
    test_cases = [
        ("blu", {"persona": "fashion_lover", "location": "Mumbai"}),
        ("nike sho", {"persona": "sports_enthusiast"}),
        ("iph", {"persona": "tech_enthusiast"}),
        ("cheap", {"persona": "budget_shopper"}),
        ("premium", {"persona": "tech_enthusiast"})
    ]
    
    for prefix, context in test_cases:
        print(f"\nPrefix: '{prefix}' | Context: {context}")
        completions = completion_system.complete_prefix(prefix, context)
        for completion, score in completions:
            print(f"  → '{completion}' (confidence: {score:.3f})")

if __name__ == "__main__":
    main()