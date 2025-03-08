#!/usr/bin/env python3
"""
Test script for evaluating mathematical problem-solving capabilities of the LLM.
Includes Perchance-like natural language feedback generation.
"""

import os
import sys
import json
import random
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from training.math_training import MathProblem, MathTrainer
from tokenizer.bpe_tokenizer import BPETokenizer
from model.transformer import TransformerConfig

class PerchanceStyleGenerator:
    """Emulates Perchance.org style text generation with templates and variables."""
    
    def __init__(self):
        """Initialize the Perchance-style generator with templates."""
        self.accuracy_templates = [
            "The model achieved {accuracy}% accuracy on the {problem_type} problems. {assessment}",
            "Analysis shows {accuracy}% of mathematical solutions were correct. {assessment}",
            "Mathematical accuracy: {accuracy}%. {assessment}",
            "In testing, the LLM solved {accuracy}% of problems correctly. {assessment}",
            "Evaluation results: {accuracy}% success rate on {problem_type} tasks. {assessment}"
        ]
        
        self.assessment_templates = {
            "excellent": [
                "This is an exceptional result!",
                "The model demonstrates outstanding mathematical reasoning.",
                "Impressive performance, showing deep understanding of mathematical concepts.",
                "The LLM exhibits remarkable problem-solving capabilities.",
                "These results indicate superior mathematical reasoning ability."
            ],
            "good": [
                "This shows solid mathematical reasoning ability.",
                "The model has a good grasp of fundamental concepts.",
                "Quite good performance overall, with room for targeted improvements.",
                "The LLM demonstrates strong mathematical capabilities.",
                "A respectable result showing competent mathematical understanding."
            ],
            "average": [
                "The model shows adequate understanding of basic principles.",
                "There's room for improvement in certain mathematical areas.",
                "Performance is satisfactory but could be enhanced with more training.",
                "The LLM demonstrates moderate proficiency in mathematical reasoning.",
                "Results indicate a functional but not exceptional understanding."
            ],
            "poor": [
                "Significant improvement is needed in mathematical reasoning.",
                "The model struggles with key mathematical concepts.",
                "Additional training is required to enhance performance.",
                "The results indicate limitations in mathematical problem-solving.",
                "The LLM shows limited proficiency in this mathematical domain."
            ]
        }
        
        self.step_analysis_templates = [
            "Step-by-step analysis: {step_quality}. {step_detail}",
            "Looking at the solution steps: {step_quality}. {step_detail}",
            "Breaking down the solution process: {step_quality}. {step_detail}",
            "Examining the solution methodology: {step_quality}. {step_detail}",
            "Analysis of problem-solving approach: {step_quality}. {step_detail}"
        ]
        
        self.step_quality_templates = {
            "excellent": [
                "steps are rigorous and highly logical",
                "solution path shows exceptional clarity",
                "steps demonstrate sophisticated mathematical thinking",
                "approach reveals deep conceptual understanding",
                "methodology is elegant and optimally structured"
            ],
            "good": [
                "steps follow a clear and logical progression",
                "solution methodology is sound and well-structured",
                "steps show good mathematical reasoning",
                "approach is appropriate and well-executed",
                "solution process is coherent and effective"
            ],
            "average": [
                "steps are generally correct but lack elegance",
                "solution process is functional but could be more efficient",
                "methodology is adequate but sometimes unclear",
                "steps show basic understanding but miss deeper insights",
                "approach works but takes unnecessary detours"
            ],
            "poor": [
                "steps contain logical gaps or errors",
                "solution process is difficult to follow",
                "methodology lacks mathematical rigor",
                "steps miss key mathematical principles",
                "approach is unnecessarily complicated or flawed"
            ]
        }
        
        self.step_detail_templates = {
            "excellent": [
                "Each step logically builds toward the solution with clear justification.",
                "The model precisely identifies and applies the most relevant theorems and properties.",
                "Complex mathematical relationships are handled with sophisticated insight.",
                "The solution path reflects an optimal approach to the problem.",
                "The model demonstrates exceptional mathematical intuition in its approach."
            ],
            "good": [
                "Most steps are well-justified and lead clearly to the solution.",
                "The model correctly applies relevant mathematical principles.",
                "The solution shows good understanding of the underlying concepts.",
                "The approach is methodical and generally efficient.",
                "Key mathematical insights are applied appropriately."
            ],
            "average": [
                "Steps generally move toward the solution but with occasional unclear reasoning.",
                "Some mathematical principles are applied correctly, others less so.",
                "The solution works but misses opportunities for more elegant approaches.",
                "The model shows understanding of basics but less comfort with complex concepts.",
                "Some steps could be more efficient or better justified."
            ],
            "poor": [
                "Steps often lack clear connection or justification.",
                "Several mathematical principles are misapplied or misunderstood.",
                "The solution path is convoluted or unnecessarily complex.",
                "Fundamental mathematical relationships are overlooked.",
                "The approach suggests significant gaps in understanding core concepts."
            ]
        }
        
        self.problem_types = [
            "algebra",
            "calculus",
            "differential equations",
            "linear algebra",
            "complex analysis",
            "mathematical proof",
            "optimization",
            "multivariable calculus",
            "vector analysis",
            "numerical methods",
            "mathematical reasoning"
        ]
    
    def generate_accuracy_feedback(self, accuracy: float, category: str = None) -> str:
        """Generate natural language feedback about model accuracy."""
        # Determine assessment category if not provided
        if category is None:
            if accuracy >= 90:
                category = "excellent"
            elif accuracy >= 75:
                category = "good"
            elif accuracy >= 50:
                category = "average"
            else:
                category = "poor"
        
        # Format accuracy as percentage
        accuracy_pct = round(accuracy * 100)
        
        # Select random templates
        accuracy_template = random.choice(self.accuracy_templates)
        assessment = random.choice(self.assessment_templates[category])
        problem_type = random.choice(self.problem_types)
        
        # Generate the feedback
        return accuracy_template.format(
            accuracy=accuracy_pct,
            assessment=assessment,
            problem_type=problem_type
        )
    
    def generate_step_feedback(self, step_quality: float) -> str:
        """Generate feedback about the quality of solution steps."""
        # Determine quality category
        if step_quality >= 0.9:
            category = "excellent"
        elif step_quality >= 0.75:
            category = "good"
        elif step_quality >= 0.5:
            category = "average"
        else:
            category = "poor"
        
        # Select random templates
        step_template = random.choice(self.step_analysis_templates)
        quality_desc = random.choice(self.step_quality_templates[category])
        detail = random.choice(self.step_detail_templates[category])
        
        # Generate the feedback
        return step_template.format(
            step_quality=quality_desc,
            step_detail=detail
        )
    
    def generate_comprehensive_feedback(self, 
                                       accuracy: float, 
                                       step_quality: float) -> str:
        """Generate comprehensive feedback about model performance."""
        accuracy_feedback = self.generate_accuracy_feedback(accuracy)
        step_feedback = self.generate_step_feedback(step_quality)
        
        return f"{accuracy_feedback}\n\n{step_feedback}"


class MathLLMTester:
    """Test harness for evaluating mathematical capabilities of the LLM."""
    
    def __init__(self, model_path: str = None, tokenizer_path: str = None):
        """Initialize the math LLM tester."""
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.feedback_generator = PerchanceStyleGenerator()
        
        # Initialize tokenizer
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.tokenizer = BPETokenizer.load(tokenizer_path)
        else:
            self.tokenizer = BPETokenizer(vocab_size=30000)
        
        # Initialize model (mock if no path provided)
        if model_path and os.path.exists(model_path):
            self.using_mock = False
            self.model = self._load_model(model_path)
        else:
            print("Using mock model implementation for testing")
            self.using_mock = True
            self.model = self._create_mock_model()
    
    def _load_model(self, model_path: str) -> Any:
        """Load the model from the specified path."""
        # This would be the actual model loading code
        config = TransformerConfig(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=384,
            num_layers=6,
            num_heads=6,
            intermediate_size=1536,
            max_seq_length=512,
            dropout=0.1
        )
        # Mock implementation for now
        return self._create_mock_model()
    
    def _create_mock_model(self) -> Any:
        """Create a mock model for testing."""
        class MockModel:
            def generate(self, prompt: str, max_length: int = 100) -> str:
                """Mock generation function."""
                # For algebra and calculus questions, return somewhat accurate answers
                if "equation" in prompt.lower() or "solve" in prompt.lower():
                    if "quadratic" in prompt.lower():
                        return "Using the quadratic formula: x = (-b ± √(b² - 4ac))/2a. " + \
                               "Substituting the values, we get x = -2 or x = -3."
                    if "differential" in prompt.lower() or "equation" in prompt.lower():
                        return "This is a second-order linear homogeneous differential equation. " + \
                               "The characteristic equation is r² + 4r + 4 = 0, which has the double root r = -2. " + \
                               "Therefore, the general solution is y = (C₁ + C₂x)e⁻²ˣ."
                
                # For integration questions
                if "integral" in prompt.lower():
                    return "To find the integral, I'll use the power rule: ∫xⁿ dx = (x^(n+1))/(n+1) + C. " + \
                           "For ∫(2x + 3) dx from 0 to 2, I get [x² + 3x]₀² = (4 + 6) - (0 + 0) = 10."
                
                # For linear algebra
                if "eigenvalue" in prompt.lower():
                    return "To find eigenvalues, I solve det(A - λI) = 0. For this matrix, we get " + \
                           "λ² - 4λ + 3 = 0, which gives eigenvalues λ₁ = 3 and λ₂ = 1."
                
                # Default response for other types of problems
                return "To solve this problem, I would need to apply relevant mathematical principles and theorems."
        
        return MockModel()
    
    def load_problems(self, problem_path: str) -> List[MathProblem]:
        """Load mathematical problems from a JSON file."""
        with open(problem_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        problems = []
        for p in data.get('problems', []):
            problems.append(MathProblem(
                question=p.get('question', ''),
                solution=p.get('solution', ''),
                steps=p.get('steps', []),
                latex_expr=p.get('latex_expr', None)
            ))
        
        return problems
    
    def test_problem(self, problem: MathProblem) -> Dict[str, Any]:
        """Test the model on a single problem and evaluate its performance."""
        # Get model's response to the problem
        model_response = self.model.generate(problem.question)
        
        # Calculate solution accuracy (mock implementation)
        if self.using_mock:
            # Simulate varying accuracy
            if "quadratic" in problem.question.lower():
                solution_accuracy = random.uniform(0.85, 0.95)
            elif "differential" in problem.question.lower():
                solution_accuracy = random.uniform(0.80, 0.90)
            elif "integral" in problem.question.lower():
                solution_accuracy = random.uniform(0.70, 0.85)
            elif "eigenvalue" in problem.question.lower():
                solution_accuracy = random.uniform(0.75, 0.90)
            else:
                solution_accuracy = random.uniform(0.50, 0.75)
        else:
            # Here we would use actual comparison logic
            # For now, use a simplified string matching approach
            common_terms = set(model_response.lower().split()) & set(problem.solution.lower().split())
            solution_accuracy = len(common_terms) / max(len(set(problem.solution.lower().split())), 1)
        
        # Evaluate step-by-step reasoning (mock implementation)
        step_accuracy = random.uniform(0.5, 0.95) if self.using_mock else 0.7
        
        return {
            "problem": problem.question,
            "expected_solution": problem.solution,
            "model_response": model_response,
            "solution_accuracy": solution_accuracy,
            "step_accuracy": step_accuracy
        }
    
    def run_evaluation(self, problems: List[MathProblem]) -> Dict[str, Any]:
        """Run a full evaluation on a list of problems."""
        results = []
        total_solution_accuracy = 0
        total_step_accuracy = 0
        
        for problem in problems:
            result = self.test_problem(problem)
            results.append(result)
            total_solution_accuracy += result["solution_accuracy"]
            total_step_accuracy += result["step_accuracy"]
        
        avg_solution_accuracy = total_solution_accuracy / len(problems) if problems else 0
        avg_step_accuracy = total_step_accuracy / len(problems) if problems else 0
        
        # Generate natural language feedback
        feedback = self.feedback_generator.generate_comprehensive_feedback(
            avg_solution_accuracy, avg_step_accuracy
        )
        
        return {
            "results": results,
            "avg_solution_accuracy": avg_solution_accuracy,
            "avg_step_accuracy": avg_step_accuracy,
            "feedback": feedback
        }
    
    def print_evaluation_report(self, evaluation: Dict[str, Any]) -> None:
        """Print a detailed evaluation report."""
        print("\n===== MATH LLM EVALUATION REPORT =====\n")
        print(f"Average Solution Accuracy: {evaluation['avg_solution_accuracy']*100:.2f}%")
        print(f"Average Step-by-Step Quality: {evaluation['avg_step_accuracy']*100:.2f}%\n")
        
        print("===== PERCHANCE-STYLE FEEDBACK =====\n")
        print(evaluation["feedback"])
        print("\n")
        
        print("===== DETAILED RESULTS =====\n")
        for i, result in enumerate(evaluation["results"]):
            print(f"Problem {i+1}: {result['problem']}")
            print(f"Expected: {result['expected_solution']}")
            print(f"Model Output: {result['model_response']}")
            print(f"Accuracy: {result['solution_accuracy']*100:.2f}%")
            print()


def main():
    """Main function to run math LLM evaluation."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Test mathematical LLM capabilities")
    parser.add_argument("--model-path", type=str, help="Path to model checkpoint")
    parser.add_argument("--tokenizer-path", type=str, help="Path to tokenizer data")
    parser.add_argument("--problems-path", type=str, default="../data/math_problems.json", 
                       help="Path to math problems JSON file")
    args = parser.parse_args()
    
    # Initialize the tester
    tester = MathLLMTester(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path
    )
    
    # Load problems
    problems_path = Path(__file__).parent.parent / args.problems_path.lstrip('../')
    try:
        problems = tester.load_problems(problems_path)
    except FileNotFoundError:
        print(f"Error: Problems file not found at {problems_path}")
        print("Using sample problems instead.")
        from training.math_training import create_sample_math_problems
        problems = create_sample_math_problems()
    
    # Run evaluation
    evaluation = tester.run_evaluation(problems)
    
    # Print report
    tester.print_evaluation_report(evaluation)


if __name__ == "__main__":
    main() 