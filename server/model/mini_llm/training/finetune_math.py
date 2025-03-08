#!/usr/bin/env python3
"""
Fine-tuning script for specializing our LLM on mathematical problem-solving.
This script loads our pre-trained model and further trains it on mathematical problems.
"""

import os
import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from training.math_training import MathProblem, MathTrainer
from tokenizer.bpe_tokenizer import BPETokenizer
from model.transformer import TransformerConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent.parent / "output/math_finetune.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("math_finetune")


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
    
    logger.info(f"Loaded {len(problems)} math problems from {problem_path}")
    return problems


def generate_training_data(problems: List[MathProblem]) -> List[str]:
    """
    Generate training examples from math problems.
    Each example is formatted as a question-answer pair.
    """
    training_examples = []
    
    for problem in problems:
        # Format as question with step-by-step answer
        example = f"Question: {problem.question}\n\nAnswer:"
        
        # Add solution with steps
        if problem.steps:
            example += f" {problem.solution}\n\nStep-by-step solution:\n"
            for step in problem.steps:
                example += f"{step}\n"
        else:
            example += f" {problem.solution}"
        
        training_examples.append(example)
        
        # Also add a variant that asks for steps
        step_example = f"Question: {problem.question}\nProvide a step-by-step solution.\n\nAnswer:"
        
        if problem.steps:
            step_example += "\n"
            for step in problem.steps:
                step_example += f"{step}\n"
            step_example += f"Therefore, {problem.solution}"
        else:
            step_example += f" {problem.solution}"
        
        training_examples.append(step_example)
    
    logger.info(f"Generated {len(training_examples)} training examples")
    return training_examples


def generate_augmented_data(problems: List[MathProblem], num_augmented: int = 100) -> List[str]:
    """
    Generate augmented training data by creating variations of existing problems.
    This helps the model generalize better.
    """
    augmented_examples = []
    
    # Define templates for augmentation
    templates = [
        "Solve the following problem: {question}\n\nSolution: {solution}",
        "Math problem: {question}\n\nAnswer: {solution}",
        "Find the solution: {question}\n\nResult: {solution}",
        "Given the problem: {question}\n\nShow that: {solution}",
        "Calculate: {question}\n\nThe answer is: {solution}"
    ]
    
    # Create augmented examples
    for _ in range(num_augmented):
        problem = random.choice(problems)
        template = random.choice(templates)
        
        example = template.format(
            question=problem.question,
            solution=problem.solution
        )
        
        augmented_examples.append(example)
    
    logger.info(f"Generated {len(augmented_examples)} augmented examples")
    return augmented_examples


def finetune_on_math(
    model_path: str,
    tokenizer_path: str,
    problems_path: str,
    output_dir: str,
    batch_size: int = 4,
    num_epochs: int = 5,
    learning_rate: float = 3e-5,
    step_weight: float = 0.3,
    expression_weight: float = 0.3,
    augment_data: bool = True
) -> None:
    """
    Fine-tune the model on mathematical problems.
    
    Args:
        model_path: Path to the pre-trained model
        tokenizer_path: Path to the tokenizer
        problems_path: Path to the JSON file containing math problems
        output_dir: Directory to save fine-tuned model
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        step_weight: Weight for step-by-step explanation loss
        expression_weight: Weight for mathematical expression loss
        augment_data: Whether to augment training data
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define special tokens for math
    special_tokens = {
        "<PAD>": 0,
        "<BOS>": 1,
        "<EOS>": 2,
        "<UNK>": 3,
        "<MASK>": 4,
        "<QUERY>": 5,
        "<TEACHER_RESPONSE>": 6,
        "<MATH>": 7,
        "<FRACTION>": 8,
        "<SQRT>": 9,
        "<INTEGRAL>": 10,
        "<DERIVATIVE>": 11,
        "<LIMIT>": 12,
        "<SUM>": 13,
        "<PRODUCT>": 14,
    }
    
    # Load tokenizer (or create a new one if not found)
    if os.path.exists(tokenizer_path):
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = BPETokenizer.load(tokenizer_path)
        # Ensure special tokens exist
        for token, idx in special_tokens.items():
            if token not in tokenizer.special_tokens:
                logger.info(f"Adding missing special token: {token}")
                tokenizer.special_tokens[token] = idx
                tokenizer.id_to_token[idx] = token
    else:
        logger.info("Creating new tokenizer with math special tokens")
        tokenizer = BPETokenizer(vocab_size=30000, special_tokens=special_tokens)
    
    # Load math problems
    problems = load_math_problems(problems_path)
    
    # Generate training data
    training_examples = generate_training_data(problems)
    
    # Augment training data if requested
    if augment_data:
        augmented_examples = generate_augmented_data(problems)
        training_examples.extend(augmented_examples)
    
    # Shuffle training examples
    random.shuffle(training_examples)
    
    # Initialize trainer
    trainer = MathTrainer(
        model=None,  # Will be loaded or created in trainer
        tokenizer=tokenizer,
        output_dir=output_dir
    )
    
    # Add math symbols to tokenizer and train if it's new
    math_symbols = trainer.math_symbols
    if not os.path.exists(tokenizer_path):
        logger.info("Training tokenizer on math examples")
        tokenizer.train(training_examples)
        
        # Add math symbols after initial training
        logger.info(f"Adding {len(math_symbols)} math symbols to tokenizer")
        for symbol in math_symbols:
            tokenizer.add_token(symbol)
        
        # Save tokenizer
        tokenizer.save(os.path.join(output_dir, "tokenizer"))
    
    # Load or create model
    try:
        if os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            # Real implementation would load the model here
            # For now, we'll use the mock implementation
            logger.warning("Using mock implementation since model loading is not implemented")
        else:
            logger.info("Creating new model")
            # This would create a new model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.warning("Using mock implementation")
    
    # Start fine-tuning
    logger.info("Starting fine-tuning on mathematical problems")
    start_time = time.time()
    
    # Convert problem list to the format expected by train_on_math
    math_problems = problems  # Already in the right format
    
    # Fine-tune
    try:
        trainer.train_on_math(
            problems=math_problems,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            step_weight=step_weight,
            expression_weight=expression_weight
        )
        
        # Log training completion
        training_time = time.time() - start_time
        logger.info(f"Fine-tuning completed in {training_time:.2f} seconds")
        
        # Save fine-tuned model
        final_model_path = os.path.join(output_dir, "final_model")
        logger.info(f"Saving fine-tuned model to {final_model_path}")
        trainer.save_model(final_model_path)
        
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        raise


def main():
    """Main function to run fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune LLM on mathematical problems")
    parser.add_argument("--model-path", type=str, help="Path to pre-trained model")
    parser.add_argument("--tokenizer-path", type=str, default="output/tokenizer", 
                       help="Path to tokenizer data")
    parser.add_argument("--problems-path", type=str, default="data/math_problems.json", 
                       help="Path to math problems JSON file")
    parser.add_argument("--output-dir", type=str, default="output/math_finetuned", 
                       help="Output directory for fine-tuned model")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--step-weight", type=float, default=0.3, 
                       help="Weight for step-by-step explanation loss")
    parser.add_argument("--expression-weight", type=float, default=0.3, 
                       help="Weight for mathematical expression loss")
    parser.add_argument("--no-augment", action="store_true", 
                       help="Disable training data augmentation")
    args = parser.parse_args()
    
    # Convert arguments to proper paths
    base_dir = Path(__file__).parent.parent
    model_path = args.model_path
    tokenizer_path = base_dir / args.tokenizer_path
    problems_path = base_dir / args.problems_path
    output_dir = base_dir / args.output_dir
    
    # Start fine-tuning
    finetune_on_math(
        model_path=model_path,
        tokenizer_path=str(tokenizer_path),
        problems_path=str(problems_path),
        output_dir=str(output_dir),
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        step_weight=args.step_weight,
        expression_weight=args.expression_weight,
        augment_data=not args.no_augment
    )
    
    logger.info("Fine-tuning process completed successfully")


if __name__ == "__main__":
    main() 