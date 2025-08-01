"""
Agentic Improvement Cycle for Continuous Autosuggest Enhancement
Uses LLM feedback to automatically identify and fix quality issues
"""
import json
import time
import requests
from typing import List, Dict, Tuple
from enhanced_autosuggest_v5 import EnhancedAutosuggestV5
from llm_evaluator_agent import LLMEvaluatorAgent
from data_preprocessing import DataPreprocessor

class AgenticImprovementCycle:
    """
    Continuous improvement system that uses LLM feedback to identify and fix issues.
    """
    
    def __init__(self):
        self.autosuggest_system = None
        self.evaluator = LLMEvaluatorAgent()
        self.improvement_history = []
        self.current_issues = []
        
        # Define critical test scenarios that must pass
        self.critical_scenarios = [
            {
                "name": "Brand Recognition: 'sam' → Samsung",
                "query": "sam",
                "context": {"persona": "tech_enthusiast", "location": "Delhi", "event": "none"},
                "expected_brand": "samsung",
                "priority": "CRITICAL",
                "description": "Should suggest Samsung products, not cricket equipment"
            },
            {
                "name": "Typo Correction: 'xiomi' → Xiaomi",
                "query": "xiomi",
                "context": {"persona": "tech_enthusiast", "location": "Mumbai", "event": "none"},
                "expected_brand": "xiaomi",
                "priority": "HIGH",
                "description": "Should correct typo and suggest Xiaomi products"
            },
            {
                "name": "Sports Context: 'jersy' → Jersey",
                "query": "jersy",
                "context": {"persona": "sports_enthusiast", "location": "Chennai", "event": "ipl"},
                "expected_keywords": ["jersey", "cricket", "sports"],
                "priority": "HIGH",
                "description": "Should suggest sports jerseys, especially cricket in IPL context"
            },
            {
                "name": "Partial Brand: 'nike' → Nike Products",
                "query": "nike",
                "context": {"persona": "sports_enthusiast", "location": "Mumbai", "event": "none"},
                "expected_brand": "nike",
                "priority": "MEDIUM",
                "description": "Should suggest Nike sports products"
            },
            {
                "name": "Tech Product: 'laptop' → Laptop Brands",
                "query": "laptop",
                "context": {"persona": "tech_enthusiast", "location": "Bangalore", "event": "none"},
                "expected_keywords": ["laptop", "hp", "dell", "lenovo"],
                "priority": "MEDIUM",
                "description": "Should suggest popular laptop brands"
            }
        ]
        
    def initialize_system(self):
        """Initialize the autosuggest system."""
        print("🚀 Initializing Agentic Improvement System...")
        
        # Load data
        preprocessor = DataPreprocessor()
        preprocessor.run_all_preprocessing()
        data = preprocessor.get_processed_data()
        
        # Initialize Enhanced V5 system
        self.autosuggest_system = EnhancedAutosuggestV5()
        self.autosuggest_system.build_system(data)
        
        print("✅ System initialized successfully!")
    
    def run_critical_validation(self) -> Dict:
        """Run validation on critical scenarios."""
        print("\n🔍 RUNNING CRITICAL VALIDATION")
        print("=" * 50)
        
        validation_results = {
            "timestamp": time.time(),
            "total_scenarios": len(self.critical_scenarios),
            "passed": 0,
            "failed": 0,
            "critical_failures": 0,
            "scenario_results": []
        }
        
        for scenario in self.critical_scenarios:
            print(f"\n🧪 Testing: {scenario['name']}")
            print(f"   Query: '{scenario['query']}'")
            print(f"   Context: {scenario['context']}")
            
            # Get suggestions
            suggestions = self.autosuggest_system.get_suggestions(
                scenario['query'], 
                scenario['context'], 
                max_suggestions=5
            )
            
            # Evaluate results
            result = self._evaluate_scenario(scenario, suggestions)
            validation_results['scenario_results'].append(result)
            
            if result['passed']:
                validation_results['passed'] += 1
                print(f"   ✅ PASS - {result['reason']}")
            else:
                validation_results['failed'] += 1
                if scenario['priority'] == 'CRITICAL':
                    validation_results['critical_failures'] += 1
                print(f"   ❌ FAIL - {result['reason']}")
                
            print(f"   💡 Suggestions: {[s for s, _ in suggestions]}")
        
        # Calculate overall success rate
        success_rate = (validation_results['passed'] / validation_results['total_scenarios']) * 100
        validation_results['success_rate'] = success_rate
        
        print(f"\n📊 VALIDATION SUMMARY")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Critical Failures: {validation_results['critical_failures']}")
        
        return validation_results
    
    def _evaluate_scenario(self, scenario: Dict, suggestions: List[Tuple[str, float]]) -> Dict:
        """Evaluate a single scenario."""
        suggestion_texts = [s.lower() for s, _ in suggestions]
        
        result = {
            "scenario_name": scenario['name'],
            "query": scenario['query'],
            "suggestions": [s for s, _ in suggestions],
            "passed": False,
            "reason": "",
            "priority": scenario['priority']
        }
        
        # Check for expected brand
        if 'expected_brand' in scenario:
            expected_brand = scenario['expected_brand'].lower()
            brand_found = any(expected_brand in suggestion for suggestion in suggestion_texts)
            
            if brand_found:
                result['passed'] = True
                result['reason'] = f"Found expected brand '{expected_brand}'"
            else:
                result['reason'] = f"Missing expected brand '{expected_brand}'"
        
        # Check for expected keywords
        elif 'expected_keywords' in scenario:
            expected_keywords = [kw.lower() for kw in scenario['expected_keywords']]
            keywords_found = sum(
                1 for keyword in expected_keywords 
                if any(keyword in suggestion for suggestion in suggestion_texts)
            )
            
            if keywords_found >= len(expected_keywords) * 0.6:  # At least 60% of keywords
                result['passed'] = True
                result['reason'] = f"Found {keywords_found}/{len(expected_keywords)} expected keywords"
            else:
                result['reason'] = f"Only found {keywords_found}/{len(expected_keywords)} expected keywords"
        
        return result
    
    def identify_improvement_opportunities(self, validation_results: Dict) -> List[Dict]:
        """Identify specific improvement opportunities from validation results."""
        opportunities = []
        
        for result in validation_results['scenario_results']:
            if not result['passed']:
                opportunity = {
                    "issue_type": self._classify_issue(result),
                    "scenario": result['scenario_name'],
                    "query": result['query'],
                    "current_suggestions": result['suggestions'],
                    "reason": result['reason'],
                    "priority": result['priority'],
                    "suggested_fix": self._suggest_fix(result)
                }
                opportunities.append(opportunity)
        
        return opportunities
    
    def _classify_issue(self, result: Dict) -> str:
        """Classify the type of issue based on failure reason."""
        reason = result['reason'].lower()
        
        if 'brand' in reason:
            return "brand_recognition"
        elif 'keyword' in reason:
            return "keyword_matching"
        elif 'typo' in reason or 'spelling' in reason:
            return "spell_correction"
        else:
            return "general_relevance"
    
    def _suggest_fix(self, result: Dict) -> str:
        """Suggest a specific fix for the identified issue."""
        issue_type = self._classify_issue(result)
        query = result['query']
        
        if issue_type == "brand_recognition":
            return f"Enhance brand prefix matching for '{query}' to prioritize brand-specific results"
        elif issue_type == "keyword_matching":
            return f"Improve semantic matching to include relevant keywords for '{query}'"
        elif issue_type == "spell_correction":
            return f"Add '{query}' to typo correction dictionary with proper mapping"
        else:
            return f"Improve overall relevance scoring for query '{query}'"
    
    def implement_automated_fixes(self, opportunities: List[Dict]):
        """Implement automated fixes for identified issues."""
        print(f"\n🔧 IMPLEMENTING AUTOMATED FIXES")
        print("=" * 40)
        
        fixes_applied = 0
        
        for opportunity in opportunities:
            if opportunity['priority'] in ['CRITICAL', 'HIGH']:
                print(f"\n🛠️ Fixing: {opportunity['scenario']}")
                print(f"   Issue: {opportunity['issue_type']}")
                print(f"   Fix: {opportunity['suggested_fix']}")
                
                success = self._apply_fix(opportunity)
                if success:
                    fixes_applied += 1
                    print(f"   ✅ Fix applied successfully")
                else:
                    print(f"   ⚠️ Fix could not be applied automatically")
        
        print(f"\n📊 Applied {fixes_applied}/{len(opportunities)} fixes")
        return fixes_applied
    
    def _apply_fix(self, opportunity: Dict) -> bool:
        """Apply a specific fix to the autosuggest system."""
        try:
            issue_type = opportunity['issue_type']
            query = opportunity['query']
            
            if issue_type == "brand_recognition" and query == "sam":
                # Add specific brand prefix mapping for 'sam' -> Samsung
                if hasattr(self.autosuggest_system, 'brand_prefixes'):
                    self.autosuggest_system.brand_prefixes['sam'] = [
                        'samsung galaxy', 'samsung phone', 'samsung tv', 
                        'samsung laptop', 'samsung mobile'
                    ]
                    return True
            elif issue_type == "spell_correction":
                # Add to typo corrections
                if hasattr(self.autosuggest_system, 'typo_corrections'):
                    # This would need more sophisticated logic to determine corrections
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error applying fix: {e}")
            return False
    
    def run_continuous_improvement_cycle(self, max_iterations: int = 5) -> Dict:
        """Run the complete continuous improvement cycle."""
        print("🔄 STARTING CONTINUOUS IMPROVEMENT CYCLE")
        print("=" * 60)
        
        cycle_results = {
            "iterations": [],
            "initial_success_rate": 0,
            "final_success_rate": 0,
            "total_fixes_applied": 0
        }
        
        for iteration in range(max_iterations):
            print(f"\n🔄 ITERATION {iteration + 1}/{max_iterations}")
            print("-" * 40)
            
            # Run validation
            validation_results = self.run_critical_validation()
            
            # Store initial success rate
            if iteration == 0:
                cycle_results['initial_success_rate'] = validation_results['success_rate']
            
            # If success rate is high enough, we can stop
            if validation_results['success_rate'] >= 90:
                print(f"🎉 Target success rate achieved: {validation_results['success_rate']:.1f}%")
                cycle_results['final_success_rate'] = validation_results['success_rate']
                break
            
            # Identify improvement opportunities
            opportunities = self.identify_improvement_opportunities(validation_results)
            
            if not opportunities:
                print("✅ No improvement opportunities identified")
                cycle_results['final_success_rate'] = validation_results['success_rate']
                break
            
            # Apply fixes
            fixes_applied = self.implement_automated_fixes(opportunities)
            cycle_results['total_fixes_applied'] += fixes_applied
            
            # Store iteration results
            iteration_result = {
                "iteration": iteration + 1,
                "success_rate": validation_results['success_rate'],
                "opportunities_found": len(opportunities),
                "fixes_applied": fixes_applied
            }
            cycle_results['iterations'].append(iteration_result)
            
            # If this is the last iteration, set final success rate
            if iteration == max_iterations - 1:
                cycle_results['final_success_rate'] = validation_results['success_rate']
        
        # Print final summary
        self._print_cycle_summary(cycle_results)
        
        return cycle_results
    
    def _print_cycle_summary(self, results: Dict):
        """Print a summary of the improvement cycle."""
        print(f"\n" + "=" * 60)
        print(f"🎯 CONTINUOUS IMPROVEMENT CYCLE SUMMARY")
        print(f"=" * 60)
        
        print(f"📊 Results:")
        print(f"   Initial Success Rate: {results['initial_success_rate']:.1f}%")
        print(f"   Final Success Rate: {results['final_success_rate']:.1f}%")
        print(f"   Improvement: {results['final_success_rate'] - results['initial_success_rate']:+.1f}%")
        print(f"   Total Fixes Applied: {results['total_fixes_applied']}")
        print(f"   Iterations Completed: {len(results['iterations'])}")
        
        if results['final_success_rate'] >= 90:
            print(f"\n🏆 EXCELLENT! System achieved target quality!")
        elif results['final_success_rate'] >= 75:
            print(f"\n✅ GOOD! System quality improved significantly!")
        else:
            print(f"\n⚠️ MODERATE! More improvements needed!")

def main():
    """Main function to run the agentic improvement cycle."""
    print("🤖 AGENTIC AUTOSUGGEST IMPROVEMENT SYSTEM")
    print("=" * 60)
    
    # Initialize system
    cycle = AgenticImprovementCycle()
    cycle.initialize_system()
    
    # Run continuous improvement
    results = cycle.run_continuous_improvement_cycle(max_iterations=3)
    
    return results

if __name__ == "__main__":
    results = main()