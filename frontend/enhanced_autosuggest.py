#!/usr/bin/env python3
"""
Enhanced Autosuggest System with SBERT + BERT Integration
==========================================================

Combines traditional autosuggest with semantic correction and context-aware completion
for superior query suggestion quality.
"""

import logging
from typing import List, Dict, Tuple, Optional
from simple_autosuggest import SimpleAutosuggestSystem
from semantic_correction import SemanticCorrection
from completion_system import CompletionSystem
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedAutosuggestSystem:
    """
    Enhanced autosuggest system that combines multiple suggestion strategies:
    1. SBERT semantic typo correction
    2. BERT context-aware completion
    3. Traditional trie-based matching
    4. Persona-aware suggestions
    """
    
    def __init__(self):
        self.semantic_correction = None
        self.completion_system = None
        self.simple_autosuggest = None
        self.correction_loaded = False
        self.completion_loaded = False
        self.autosuggest_loaded = False
        
        logger.info("🚀 Initializing Enhanced Autosuggest System...")
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize all autosuggest components."""
        
        # Initialize simple autosuggest
        try:
            self.simple_autosuggest = SimpleAutosuggestSystem()
            logger.info("✅ Simple autosuggest initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize simple autosuggest: {e}")
        
        # Initialize semantic correction (lazy loading)
        try:
            self.semantic_correction = SemanticCorrection()
            logger.info("✅ Semantic correction initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize semantic correction: {e}")
        
        # Initialize completion system (lazy loading)
        try:
            self.completion_system = CompletionSystem()
            logger.info("✅ Completion system initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize completion system: {e}")
    
    def build_system(self, data: Dict):
        """Build the enhanced autosuggest system with all components."""
        logger.info("🔧 Building Enhanced Autosuggest System...")
        
        # Build traditional autosuggest
        if self.simple_autosuggest:
            try:
                self.simple_autosuggest.build_system(data)
                self.autosuggest_loaded = True
                logger.info("✅ Traditional autosuggest built successfully")
            except Exception as e:
                logger.error(f"❌ Failed to build traditional autosuggest: {e}")
        
        # Build semantic correction index
        if self.semantic_correction:
            try:
                success = self.semantic_correction.build_correction_index()
                self.correction_loaded = success
                if success:
                    logger.info("✅ Semantic correction index built successfully")
                else:
                    logger.warning("⚠️ Semantic correction index build failed")
            except Exception as e:
                logger.error(f"❌ Failed to build semantic correction: {e}")
        
        # Completion system doesn't need building (model loads on demand)
        self.completion_loaded = True
        
        logger.info("🎉 Enhanced Autosuggest System ready!")
        logger.info(f"   Traditional: {'✅' if self.autosuggest_loaded else '❌'}")
        logger.info(f"   Correction:  {'✅' if self.correction_loaded else '❌'}")
        logger.info(f"   Completion:  {'✅' if self.completion_loaded else '❌'}")
    
    def get_suggestions(self, query: str, context: Optional[Dict] = None, max_suggestions: int = 5) -> List[Tuple[str, float]]:
        """
        Get enhanced suggestions combining all methods.
        
        Args:
            query: User input query
            context: User context (persona, location, etc.)
            max_suggestions: Maximum suggestions to return
            
        Returns:
            List of (suggestion, score) tuples
        """
        if not query or len(query.strip()) == 0:
            return []
        
        start_time = time.time()
        query = query.lower().strip()
        all_suggestions = []
        
        logger.debug(f"🔍 Getting enhanced suggestions for: '{query}'")
        
        # Strategy 1: Handle very short queries with completion
        if len(query) <= 3 and self.completion_loaded and self.completion_system:
            try:
                completions = self.completion_system.complete_prefix(query, context, max_suggestions)
                for completion, score in completions:
                    all_suggestions.append((completion, score * 0.9, 'completion'))  # Slightly lower weight
                logger.debug(f"   Completion: {len(completions)} suggestions")
            except Exception as e:
                logger.debug(f"   Completion failed: {e}")
        
        # Strategy 2: Semantic typo correction
        if self.correction_loaded and self.semantic_correction:
            try:
                if self.semantic_correction.is_likely_typo(query):
                    corrections = self.semantic_correction.get_corrections(query, top_k=3)
                    for correction, score in corrections:
                        all_suggestions.append((correction, score * 1.1, 'correction'))  # Higher weight for corrections
                    logger.debug(f"   Corrections: {len(corrections)} suggestions")
            except Exception as e:
                logger.debug(f"   Correction failed: {e}")
        
        # Strategy 3: Traditional autosuggest
        if self.autosuggest_loaded and self.simple_autosuggest:
            try:
                traditional = self.simple_autosuggest.get_suggestions(query, max_suggestions * 2, context)
                for suggestion, score in traditional:
                    all_suggestions.append((suggestion, score, 'traditional'))
                logger.debug(f"   Traditional: {len(traditional)} suggestions")
            except Exception as e:
                logger.debug(f"   Traditional failed: {e}")
        
        # Strategy 4: Context-aware completion for longer queries
        if len(query) > 3 and len(query) <= 10 and self.completion_loaded and self.completion_system:
            try:
                context_completions = self.completion_system.complete_prefix(query, context, max_suggestions // 2)
                for completion, score in context_completions:
                    all_suggestions.append((completion, score * 0.8, 'context_completion'))
                logger.debug(f"   Context completion: {len(context_completions)} suggestions")
            except Exception as e:
                logger.debug(f"   Context completion failed: {e}")
        
        # Combine and rank suggestions
        final_suggestions = self._rank_and_deduplicate(all_suggestions, query, context)
        
        # Apply persona boosting
        if context and 'persona_details' in context:
            final_suggestions = self._apply_persona_boost(final_suggestions, context['persona_details'])
        
        response_time = (time.time() - start_time) * 1000
        logger.debug(f"✅ Generated {len(final_suggestions)} suggestions in {response_time:.2f}ms")
        
        return final_suggestions[:max_suggestions]
    
    def _rank_and_deduplicate(self, suggestions: List[Tuple[str, float, str]], original_query: str, context: Optional[Dict]) -> List[Tuple[str, float]]:
        """Rank and deduplicate suggestions from multiple sources."""
        
        # Deduplicate by suggestion text
        seen = {}
        for suggestion, score, source in suggestions:
            suggestion_clean = suggestion.lower().strip()
            if suggestion_clean != original_query.lower():  # Don't suggest the same as input
                if suggestion_clean not in seen or seen[suggestion_clean][1] < score:
                    seen[suggestion_clean] = (suggestion, score, source)
        
        # Convert back to list and apply additional ranking
        ranked_suggestions = []
        for suggestion, score, source in seen.values():
            
            # Boost score based on suggestion quality
            final_score = score
            
            # Boost longer, more specific suggestions
            if len(suggestion.split()) > 2:
                final_score *= 1.1
            
            # Boost if contains numbers (often product models)
            if any(char.isdigit() for char in suggestion):
                final_score *= 1.05
            
            # Boost if starts with original query
            if suggestion.lower().startswith(original_query.lower()):
                final_score *= 1.2
            
            ranked_suggestions.append((suggestion, final_score))
        
        # Sort by final score
        ranked_suggestions.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_suggestions
    
    def _apply_persona_boost(self, suggestions: List[Tuple[str, float]], persona_details: Dict) -> List[Tuple[str, float]]:
        """Apply persona-based boosting to suggestions."""
        
        persona_keywords = []
        if 'previous_queries' in persona_details:
            persona_keywords.extend(persona_details['previous_queries'])
        if 'clicked_brands' in persona_details:
            persona_keywords.extend([brand.lower() for brand in persona_details['clicked_brands']])
        if 'clicked_categories' in persona_details:
            persona_keywords.extend([cat.lower() for cat in persona_details['clicked_categories']])
        
        boosted_suggestions = []
        for suggestion, score in suggestions:
            suggestion_lower = suggestion.lower()
            
            # Check for persona keyword matches
            boost_factor = 1.0
            for keyword in persona_keywords:
                if keyword.lower() in suggestion_lower:
                    boost_factor += 0.1  # 10% boost per matching keyword
            
            # Cap the boost
            boost_factor = min(boost_factor, 1.5)
            
            boosted_suggestions.append((suggestion, score * boost_factor))
        
        # Re-sort after boosting
        boosted_suggestions.sort(key=lambda x: x[1], reverse=True)
        
        return boosted_suggestions
    
    def get_system_status(self) -> Dict:
        """Get status of all system components."""
        return {
            'traditional_autosuggest': self.autosuggest_loaded,
            'semantic_correction': self.correction_loaded,
            'context_completion': self.completion_loaded,
            'overall_status': all([self.autosuggest_loaded, self.correction_loaded, self.completion_loaded])
        }

def main():
    """Test the enhanced autosuggest system."""
    from data_preprocessing import DataPreprocessor
    
    # Load data
    preprocessor = DataPreprocessor()
    preprocessor.run_all_preprocessing()
    data = preprocessor.get_processed_data()
    
    # Initialize enhanced system
    enhanced_autosuggest = EnhancedAutosuggestSystem()
    enhanced_autosuggest.build_system(data)
    
    # Test queries
    test_queries = [
        ("blu", {"persona": "fashion_lover", "persona_details": {"previous_queries": ["shoes", "dress"], "clicked_brands": ["Nike", "Adidas"]}}),
        ("beist 4k", {"persona": "tech_enthusiast", "persona_details": {"previous_queries": ["monitor", "laptop"], "clicked_brands": ["Dell", "HP"]}}),
        ("samung", {"persona": "tech_enthusiast"}),
        ("nike sho", {"persona": "sports_enthusiast"}),
        ("cheap lap", {"persona": "budget_shopper"})
    ]
    
    for query, context in test_queries:
        print(f"\n🔍 Query: '{query}' | Persona: {context.get('persona', 'none')}")
        suggestions = enhanced_autosuggest.get_suggestions(query, context)
        for i, (suggestion, score) in enumerate(suggestions, 1):
            print(f"   {i}. '{suggestion}' (score: {score:.3f})")

if __name__ == "__main__":
    main()