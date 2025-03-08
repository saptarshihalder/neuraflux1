#!/usr/bin/env python3
"""
Generate mathematical solutions using the fine-tuned model.

This script loads a fine-tuned mathematical model and uses it to
generate solutions to mathematical problems.
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from training.math_training import MathProblem, MathTrainer
from tokenizer.bpe_tokenizer import BPETokenizer

def load_math_problems(problem_path: str) -> List[MathProblem]:
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

def format_solution_with_steps(solution: str, steps: List[str]) -> str:
    """Format a solution with steps for display."""
    result = f"Solution: {solution}\n\nStep-by-step explanation:\n"
    for step in steps:
        result += f"{step}\n"
    return result

def generate_math_solution(model_path: str, tokenizer_path: str, problem: str, 
                          show_steps: bool = True) -> Dict[str, Any]:
    """
    Generate a solution to a mathematical problem.
    
    Args:
        model_path: Path to the fine-tuned model
        tokenizer_path: Path to the tokenizer
        problem: The mathematical problem to solve
        show_steps: Whether to show step-by-step solution
        
    Returns:
        Dict containing the problem, solution, and steps
    """
    print(f"Problem: {problem}")
    print("\nGenerating solution...\n")
    
    # Load tokenizer
    if os.path.exists(tokenizer_path):
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        print(f"Tokenizer not found at {tokenizer_path}. Using default tokenizer.")
        tokenizer = BPETokenizer(vocab_size=30000)
    
    # Initialize model (use mock if no path provided)
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        # In a real implementation, we would load the model here
        # For now, use a mock implementation
        trainer = MathTrainer(None, tokenizer, "output/temp")
    else:
        print("Using mock model for demonstration")
        trainer = MathTrainer(None, tokenizer, "output/temp")
    
    # Generate solution
    solution = trainer.generate(problem)
    
    # Generate steps if requested
    steps = []
    if show_steps:
        steps = trainer.generate_steps(problem)
    
    # Show results
    if show_steps:
        print(format_solution_with_steps(solution, steps))
    else:
        print(f"Solution: {solution}")
    
    # Return results
    return {
        "problem": problem,
        "solution": solution,
        "steps": steps
    }

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate mathematical solutions")
    parser.add_argument("--model-path", type=str, 
                       default="output/math_finetuned/final_model",
                       help="Path to fine-tuned model")
    parser.add_argument("--tokenizer-path", type=str, 
                       default="output/math_finetuned/final_model/tokenizer",
                       help="Path to tokenizer")
    parser.add_argument("--problem", type=str,
                       help="Mathematical problem to solve")
    parser.add_argument("--problems-path", type=str,
                       default="data/math_problems.json",
                       help="Path to JSON file with math problems")
    parser.add_argument("--random", action="store_true",
                       help="Use a random problem from the problems file")
    parser.add_argument("--no-steps", action="store_true",
                       help="Don't show step-by-step solution")
    args = parser.parse_args()
    
    # Figure out which problem to solve
    if args.random:
        # Use a random problem from the file
        problems = load_math_problems(args.problems_path)
        problem = random.choice(problems).question
    elif args.problem:
        # Use the provided problem
        problem = args.problem
    else:
        # Default to a sample problem
        problem = "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 3"
    
    # Generate solution
    generate_math_solution(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        problem=problem,
        show_steps=not args.no_steps
    )

if __name__ == "__main__":
    main() 