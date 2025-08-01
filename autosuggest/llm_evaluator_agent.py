"""
LLM-based Agent Evaluator for Autosuggest Quality Assessment
Uses OpenRouter API with Gemini to evaluate suggestion quality and provide improvement feedback.
"""
import requests
import json
import time
from typing import List, Dict, Tuple, Optional
from advanced_autosuggest_v4 import AdvancedAutosuggestV4
from data_preprocessing import DataPreprocessor

class LLMEvaluatorAgent:
    """
    LLM-based evaluator that assesses autosuggest quality and provides improvement feedback.
    """
    
    def __init__(self):
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.api_key = "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"
        self.model = "google/gemini-2.5-flash"
        
        # Evaluation criteria
        self.evaluation_criteria = {
            "relevance": "How relevant are the suggestions to the original query?",
            "spelling_correction": "Are typos correctly identified and corrected?",
            "brand_accuracy": "Are brand names correctly suggested and spelled?",
            "contextual_awareness": "Do suggestions adapt based on persona, location, and event context?",
            "diversity": "Are suggestions diverse and not repetitive?",
            "semantic_quality": "Do suggestions make semantic sense for the query?"
        }
        
        # Test scenarios for comprehensive evaluation
        self.test_scenarios = [
            {
                "query": "xiomi",
                "context": {"persona": "tech_enthusiast", "location": "Delhi", "event": "none"},
                "expected_patterns": ["xiaomi", "phone", "mobile", "smartphone"],
                "description": "Typo correction for brand name"
            },
            {
                "query": "jersy",
                "context": {"persona": "sports_enthusiast", "location": "Chennai", "event": "ipl"},
                "expected_patterns": ["jersey", "cricket", "ipl", "team", "csk", "rcb"],
                "description": "Sports context with IPL event"
            },
            {
                "query": "samsng",
                "context": {"persona": "tech_enthusiast", "location": "Bangalore", "event": "none"},
                "expected_patterns": ["samsung", "galaxy", "phone", "tv", "electronics"],
                "description": "Brand typo correction with tech context"
            },
            {
                "query": "nike sho",
                "context": {"persona": "sports_enthusiast", "location": "Mumbai", "event": "none"},
                "expected_patterns": ["nike", "shoes", "running", "sports", "sneakers"],
                "description": "Partial brand and product query"
            },
            {
                "query": "lights",
                "context": {"persona": "home_maker", "location": "Mumbai", "event": "diwali"},
                "expected_patterns": ["diwali", "decoration", "led", "festive", "lighting"],
                "description": "Contextual event-based suggestions"
            },
            {
                "query": "lapto",
                "context": {"persona": "tech_enthusiast", "location": "Bangalore", "event": "none"},
                "expected_patterns": ["laptop", "gaming", "business", "hp", "dell", "lenovo"],
                "description": "Typo correction with tech persona"
            }
        ]
    
    def evaluate_autosuggest_system(self, autosuggest_system: AdvancedAutosuggestV4) -> Dict:
        """
        Comprehensive evaluation of the autosuggest system using LLM.
        """
        print("🤖 Starting LLM-based Autosuggest Evaluation...")
        
        evaluation_results = {
            "overall_score": 0,
            "scenario_results": [],
            "improvement_suggestions": [],
            "timestamp": time.time()
        }
        
        total_score = 0
        scenario_count = 0
        
        for scenario in self.test_scenarios:
            print(f"\n🧪 Testing Scenario: {scenario['description']}")
            print(f"   Query: '{scenario['query']}'")
            print(f"   Context: {scenario['context']}")
            
            # Get suggestions from the system
            suggestions = autosuggest_system.get_suggestions(
                scenario['query'], 
                scenario['context'], 
                max_suggestions=5
            )
            
            # Evaluate using LLM
            scenario_result = self._evaluate_scenario_with_llm(scenario, suggestions)
            evaluation_results["scenario_results"].append(scenario_result)
            
            total_score += scenario_result.get("score", 0)
            scenario_count += 1
            
            print(f"   📊 Scenario Score: {scenario_result.get('score', 0)}/10")
        
        # Calculate overall score
        evaluation_results["overall_score"] = total_score / scenario_count if scenario_count > 0 else 0
        
        # Get overall improvement suggestions
        evaluation_results["improvement_suggestions"] = self._get_overall_improvement_suggestions(
            evaluation_results["scenario_results"]
        )
        
        print(f"\n🎯 Overall Evaluation Score: {evaluation_results['overall_score']:.1f}/10")
        
        return evaluation_results
    
    def _evaluate_scenario_with_llm(self, scenario: Dict, suggestions: List[Tuple[str, float]]) -> Dict:
        """
        Evaluate a single scenario using LLM.
        """
        # Prepare suggestions for LLM
        suggestion_texts = [s[0] for s in suggestions]
        
        # Create evaluation prompt
        prompt = self._create_evaluation_prompt(scenario, suggestion_texts)
        
        try:
            # Call LLM API
            response = self._call_llm_api(prompt)
            
            if response:
                # Parse LLM response
                return self._parse_llm_evaluation(response, scenario, suggestions)
            else:
                return self._fallback_evaluation(scenario, suggestions)
                
        except Exception as e:
            print(f"   ⚠️ LLM evaluation failed: {e}")
            return self._fallback_evaluation(scenario, suggestions)
    
    def _create_evaluation_prompt(self, scenario: Dict, suggestions: List[str]) -> str:
        """
        Create a detailed evaluation prompt for the LLM.
        """
        prompt = f"""
You are an expert evaluator for an e-commerce autosuggest system. Please evaluate the quality of search suggestions.

**Input Query:** "{scenario['query']}"
**User Context:**
- Persona: {scenario['context']['persona']}
- Location: {scenario['context']['location']} 
- Event/Season: {scenario['context']['event']}

**Generated Suggestions:**
{chr(10).join([f"{i+1}. {suggestion}" for i, suggestion in enumerate(suggestions)])}

**Expected Patterns:** {', '.join(scenario['expected_patterns'])}

**Evaluation Criteria:**
1. **Spelling Correction (0-2 points):** Are typos correctly identified and fixed?
2. **Relevance (0-2 points):** How relevant are suggestions to the original query?
3. **Brand Accuracy (0-2 points):** Are brand names correctly suggested and spelled?
4. **Contextual Awareness (0-2 points):** Do suggestions adapt to persona/location/event?
5. **Semantic Quality (0-2 points):** Do suggestions make logical sense?

**Please provide a JSON response with the following structure:**
{{
    "spelling_correction_score": 0-2,
    "relevance_score": 0-2,
    "brand_accuracy_score": 0-2,
    "contextual_awareness_score": 0-2,
    "semantic_quality_score": 0-2,
    "total_score": 0-10,
    "strengths": ["list of what the system did well"],
    "weaknesses": ["list of areas needing improvement"],
    "specific_feedback": "detailed feedback on this scenario",
    "improvement_suggestions": ["specific suggestions for improvement"]
}}

Be critical but fair in your evaluation. Focus on practical e-commerce search quality.
"""
        return prompt
    
    def _call_llm_api(self, prompt: str) -> Optional[str]:
        """
        Call the LLM API with the evaluation prompt.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"   ⚠️ API Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"   ⚠️ API Call Failed: {e}")
            return None
    
    def _parse_llm_evaluation(self, llm_response: str, scenario: Dict, suggestions: List[Tuple[str, float]]) -> Dict:
        """
        Parse the LLM evaluation response.
        """
        try:
            # Try to extract JSON from LLM response
            start_idx = llm_response.find('{')
            end_idx = llm_response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = llm_response[start_idx:end_idx]
                evaluation = json.loads(json_str)
                
                # Add additional metadata
                evaluation["scenario"] = scenario
                evaluation["suggestions"] = suggestions
                evaluation["llm_response"] = llm_response
                
                return evaluation
            else:
                return self._fallback_evaluation(scenario, suggestions)
                
        except json.JSONDecodeError:
            print(f"   ⚠️ Failed to parse LLM JSON response")
            return self._fallback_evaluation(scenario, suggestions)
    
    def _fallback_evaluation(self, scenario: Dict, suggestions: List[Tuple[str, float]]) -> Dict:
        """
        Fallback evaluation when LLM is not available.
        """
        suggestion_texts = [s[0].lower() for s in suggestions]
        score = 0
        
        # Simple pattern matching evaluation
        expected_patterns = [p.lower() for p in scenario['expected_patterns']]
        
        for pattern in expected_patterns:
            if any(pattern in suggestion for suggestion in suggestion_texts):
                score += 1
        
        # Normalize score to 0-10 scale
        normalized_score = min(10, (score / len(expected_patterns)) * 10)
        
        return {
            "total_score": normalized_score,
            "scenario": scenario,
            "suggestions": suggestions,
            "evaluation_method": "fallback",
            "strengths": ["Generated suggestions"],
            "weaknesses": ["Unable to perform detailed LLM evaluation"],
            "specific_feedback": f"Simple pattern matching gave score: {normalized_score}/10"
        }
    
    def _get_overall_improvement_suggestions(self, scenario_results: List[Dict]) -> List[str]:
        """
        Generate overall improvement suggestions based on all scenario results.
        """
        suggestions = set()
        
        for result in scenario_results:
            if "improvement_suggestions" in result:
                suggestions.update(result["improvement_suggestions"])
            
            # Add suggestions based on weak areas
            if result.get("total_score", 0) < 6:
                suggestions.add("Improve suggestion relevance and quality")
                
            if "weaknesses" in result:
                for weakness in result["weaknesses"]:
                    if "spelling" in weakness.lower():
                        suggestions.add("Enhance spelling correction algorithms")
                    if "context" in weakness.lower():
                        suggestions.add("Improve contextual awareness")
                    if "brand" in weakness.lower():
                        suggestions.add("Better brand name recognition and correction")
        
        return list(suggestions)
    
    def continuous_evaluation_loop(self, autosuggest_system: AdvancedAutosuggestV4, iterations: int = 3):
        """
        Run continuous evaluation and improvement loop.
        """
        print("🔄 Starting Continuous Evaluation and Improvement Loop...")
        
        evaluation_history = []
        
        for iteration in range(iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{iterations}")
            
            # Evaluate current system
            evaluation = self.evaluate_autosuggest_system(autosuggest_system)
            evaluation_history.append(evaluation)
            
            # Display results
            self._display_evaluation_results(evaluation)
            
            # If not the last iteration, suggest improvements
            if iteration < iterations - 1:
                print("\n🛠️ Applying improvements for next iteration...")
                self._apply_improvements(autosuggest_system, evaluation)
                time.sleep(2)  # Brief pause between iterations
        
        # Summary of all iterations
        self._display_improvement_summary(evaluation_history)
        
        return evaluation_history
    
    def _display_evaluation_results(self, evaluation: Dict):
        """
        Display evaluation results in a formatted way.
        """
        print(f"\n📊 EVALUATION RESULTS")
        print(f"Overall Score: {evaluation['overall_score']:.1f}/10")
        
        print(f"\n📈 Scenario Breakdown:")
        for i, result in enumerate(evaluation['scenario_results'], 1):
            scenario = result.get('scenario', {})
            score = result.get('total_score', 0)
            description = scenario.get('description', f'Scenario {i}')
            print(f"   {i}. {description}: {score:.1f}/10")
        
        if evaluation['improvement_suggestions']:
            print(f"\n🎯 Improvement Suggestions:")
            for suggestion in evaluation['improvement_suggestions']:
                print(f"   • {suggestion}")
    
    def _apply_improvements(self, autosuggest_system: AdvancedAutosuggestV4, evaluation: Dict):
        """
        Apply improvements based on evaluation results.
        """
        # For now, this is a placeholder for actual improvements
        # In a real system, this would modify the autosuggest system based on feedback
        
        suggestions = evaluation.get('improvement_suggestions', [])
        
        for suggestion in suggestions:
            if "spelling" in suggestion.lower():
                # Could enhance spelling correction here
                print(f"   🔧 Noted: {suggestion}")
            elif "context" in suggestion.lower():
                # Could adjust contextual boosting here
                print(f"   🔧 Noted: {suggestion}")
            else:
                print(f"   🔧 Noted: {suggestion}")
    
    def _display_improvement_summary(self, evaluation_history: List[Dict]):
        """
        Display summary of improvements across iterations.
        """
        print(f"\n📈 IMPROVEMENT SUMMARY")
        print(f"Number of iterations: {len(evaluation_history)}")
        
        if len(evaluation_history) > 1:
            initial_score = evaluation_history[0]['overall_score']
            final_score = evaluation_history[-1]['overall_score']
            improvement = final_score - initial_score
            
            print(f"Initial Score: {initial_score:.1f}/10")
            print(f"Final Score: {final_score:.1f}/10")
            print(f"Improvement: {improvement:+.1f} points")
        
        # Show score progression
        scores = [eval_result['overall_score'] for eval_result in evaluation_history]
        print(f"Score Progression: {' → '.join([f'{score:.1f}' for score in scores])}")

def main():
    """
    Main function to run the LLM evaluation.
    """
    print("🚀 LLM-Based Autosuggest Evaluation System")
    print("=" * 50)
    
    # Initialize data and autosuggest system
    print("Loading autosuggest system...")
    preprocessor = DataPreprocessor()
    preprocessor.run_all_preprocessing()
    data = preprocessor.get_processed_data()
    
    autosuggest = AdvancedAutosuggestV4()
    autosuggest.build_system(data)
    
    # Initialize evaluator
    evaluator = LLMEvaluatorAgent()
    
    # Run evaluation
    evaluation_results = evaluator.continuous_evaluation_loop(autosuggest, iterations=2)
    
    print("\n🎉 Evaluation completed!")
    return evaluation_results

if __name__ == "__main__":
    results = main()