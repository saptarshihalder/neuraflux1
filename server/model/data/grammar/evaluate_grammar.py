#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to evaluate the NeuraFlux model's grammar capabilities after training.
This script loads a trained model and evaluates it on grammar correction tasks.
"""

import os
import sys
import argparse
import logging
import json
from typing import List, Dict, Tuple

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.dirname(os.path.dirname(current_dir))
server_dir = os.path.dirname(model_dir)
sys.path.append(server_dir)

# Import from mini_llm
from model.mini_llm.tokenizer.bpe_tokenizer import BPETokenizer
from model.mini_llm.model.transformer import MiniTransformer, TransformerConfig


def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO) -> None:
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "grammar_evaluation.log")
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_model_and_tokenizer(model_dir: str) -> Tuple[MiniTransformer, BPETokenizer]:
    """
    Load the trained model and tokenizer.
    
    Args:
        model_dir: Directory containing the model and tokenizer
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logging.info(f"Loading model and tokenizer from {model_dir}")
    
    # Load tokenizer
    tokenizer_path = os.path.join(model_dir, "tokenizer")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    
    tokenizer = BPETokenizer.load(tokenizer_path)
    logging.info(f"Loaded tokenizer with vocab size {len(tokenizer.vocab)}")
    
    # Load model configuration
    config_path = os.path.join(model_dir, "final_model", "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = TransformerConfig(**config_dict)
    
    # Load model weights
    model_path = os.path.join(model_dir, "final_model", "model.bin")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    model = MiniTransformer(config)
    model.load(model_path)
    
    logging.info("Model and tokenizer loaded successfully")
    return model, tokenizer


def load_evaluation_set(file_path: str) -> List[Dict[str, str]]:
    """
    Load the evaluation dataset with incorrect/correct sentence pairs.
    
    Args:
        file_path: Path to the evaluation file
        
    Returns:
        List of dictionaries with 'incorrect' and 'correct' keys
    """
    eval_pairs = []
    incorrect = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("INCORRECT:"):
                incorrect = line.replace("INCORRECT:", "").strip()
            elif line.startswith("CORRECT:") and incorrect:
                correct = line.replace("CORRECT:", "").strip()
                eval_pairs.append({
                    'incorrect': incorrect,
                    'correct': correct
                })
                incorrect = None
    
    logging.info(f"Loaded {len(eval_pairs)} evaluation pairs")
    return eval_pairs


def evaluate_grammar_correction(
    model: MiniTransformer,
    tokenizer: BPETokenizer,
    eval_pairs: List[Dict[str, str]],
    max_seq_length: int = 128,
    output_file: str = None
) -> Dict:
    """
    Evaluate the model's grammar correction capabilities.
    
    Args:
        model: The trained model
        tokenizer: The tokenizer
        eval_pairs: List of incorrect/correct sentence pairs
        max_seq_length: Maximum sequence length for generation
        output_file: Path to save detailed results
        
    Returns:
        Dictionary with evaluation metrics
    """
    results = []
    correct_count = 0
    
    for i, pair in enumerate(eval_pairs):
        incorrect = pair['incorrect']
        correct = pair['correct']
        
        # Prepare prompt
        prompt = f"Correct this sentence: {incorrect}"
        
        # Generate correction
        input_ids = tokenizer.encode(prompt)
        generated_ids = model.generate(
            input_ids, 
            max_length=max_seq_length,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        generated_text = tokenizer.decode(generated_ids)
        
        # Extract the correction from generated text
        # This assumes the model responds with the correction
        # We may need to adjust this based on actual model output format
        correction = generated_text.replace(prompt, "").strip()
        
        # Compare with correct sentence
        is_correct = correction.lower() == correct.lower()
        if is_correct:
            correct_count += 1
        
        # Save result
        results.append({
            'id': i,
            'incorrect': incorrect,
            'correct': correct,
            'model_output': correction,
            'is_correct': is_correct
        })
        
        # Log progress
        if (i + 1) % 10 == 0:
            logging.info(f"Evaluated {i + 1}/{len(eval_pairs)} examples")
    
    # Calculate accuracy
    accuracy = correct_count / len(eval_pairs) if eval_pairs else 0
    
    # Log results
    logging.info(f"Evaluation complete. Accuracy: {accuracy:.4f}")
    
    # Save detailed results if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'results': results,
                'metrics': {
                    'accuracy': accuracy,
                    'correct_count': correct_count,
                    'total': len(eval_pairs)
                }
            }, f, indent=2)
        logging.info(f"Detailed results saved to {output_file}")
    
    return {
        'accuracy': accuracy,
        'correct_count': correct_count,
        'total': len(eval_pairs)
    }


def main():
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate NeuraFlux on grammar correction")
    
    # Input and output arguments
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Directory containing the trained model")
    parser.add_argument("--eval_file", type=str, default=None,
                       help="Path to the evaluation file (if None, uses grammar_corrections.txt)")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Path to save detailed evaluation results")
    
    # Evaluation parameters
    parser.add_argument("--max_seq_length", type=int, default=128,
                       help="Maximum sequence length for generation")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging()
    
    # Set default eval file if not provided
    if args.eval_file is None:
        grammar_dir = os.path.dirname(os.path.abspath(__file__))
        args.eval_file = os.path.join(grammar_dir, "grammar_corrections.txt")
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model_dir)
        
        # Load evaluation set
        eval_pairs = load_evaluation_set(args.eval_file)
        
        # Run evaluation
        metrics = evaluate_grammar_correction(
            model=model,
            tokenizer=tokenizer,
            eval_pairs=eval_pairs,
            max_seq_length=args.max_seq_length,
            output_file=args.output_file
        )
        
        # Print summary
        print("\nEvaluation Summary:")
        print(f"Total examples: {metrics['total']}")
        print(f"Correct predictions: {metrics['correct_count']}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main() 