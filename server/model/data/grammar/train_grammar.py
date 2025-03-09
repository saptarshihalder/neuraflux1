#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to train the NeuraFlux model on English grammar datasets.
This script combines the grammar datasets and then trains the model.
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.dirname(os.path.dirname(current_dir))
server_dir = os.path.dirname(model_dir)
sys.path.append(server_dir)

# Import the combine script
from model.data.grammar.combine_grammar_datasets import combine_datasets


def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO) -> None:
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "grammar_training.log")
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def train_on_grammar(
    output_dir: str = "grammar_model",
    combined_dataset_path: str = None,
    tokenizer_path: str = None,
    vocab_size: int = 30000,
    hidden_size: int = 384,
    num_layers: int = 6,
    num_heads: int = 6,
    intermediate_size: int = 1536,
    max_seq_length: int = 512,
    batch_size: int = 8,
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
    use_mock: bool = False
) -> None:
    """
    Combine grammar datasets and train the model.
    
    Args:
        output_dir: Directory to save model checkpoints
        combined_dataset_path: Path to save the combined dataset
        tokenizer_path: Path to pre-trained tokenizer
        vocab_size: Size of the vocabulary
        hidden_size: Hidden dimension of the model
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        intermediate_size: Size of feedforward network
        max_seq_length: Maximum sequence length
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        use_mock: Whether to use mock implementation
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    setup_logging(os.path.join(output_dir, "logs"))
    
    # Define paths
    grammar_dir = os.path.dirname(os.path.abspath(__file__))
    if combined_dataset_path is None:
        combined_dataset_path = os.path.join(grammar_dir, "combined_grammar_dataset.txt")
    
    # Combine the datasets
    logging.info("Combining grammar datasets...")
    combine_datasets(combined_dataset_path)
    
    # Get the path to the training script
    train_script_path = os.path.join(model_dir, "mini_llm", "train.py")
    
    if not os.path.exists(train_script_path):
        logging.error(f"Training script not found at {train_script_path}")
        return
    
    # Prepare command for training
    cmd = [
        sys.executable,
        train_script_path,
        "--data_path", combined_dataset_path,
        "--output_dir", output_dir,
        "--vocab_size", str(vocab_size),
        "--hidden_size", str(hidden_size),
        "--num_layers", str(num_layers),
        "--num_heads", str(num_heads),
        "--intermediate_size", str(intermediate_size),
        "--max_seq_length", str(max_seq_length),
        "--batch_size", str(batch_size),
        "--num_epochs", str(num_epochs),
        "--learning_rate", str(learning_rate)
    ]
    
    if tokenizer_path:
        cmd.extend(["--tokenizer_path", tokenizer_path])
    
    if use_mock:
        cmd.append("--use_mock")
    
    # Execute the training script
    logging.info(f"Starting training with command: {' '.join(cmd)}")
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Stream the output
        for line in process.stdout:
            logging.info(line.strip())
        
        process.wait()
        
        if process.returncode == 0:
            logging.info("Training completed successfully!")
        else:
            logging.error(f"Training failed with return code {process.returncode}")
    
    except Exception as e:
        logging.error(f"Error during training: {e}")


def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(description="Train NeuraFlux on grammar datasets")
    
    # Output and data arguments
    parser.add_argument("--output_dir", type=str, default="grammar_model",
                       help="Directory to save model checkpoints and logs")
    parser.add_argument("--combined_dataset_path", type=str, default=None,
                       help="Path to save the combined dataset")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                       help="Path to pre-trained tokenizer")
    
    # Model configuration
    parser.add_argument("--vocab_size", type=int, default=30000,
                       help="Vocabulary size for tokenizer")
    parser.add_argument("--hidden_size", type=int, default=384,
                       help="Hidden dimension of the model")
    parser.add_argument("--num_layers", type=int, default=6,
                       help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=6,
                       help="Number of attention heads")
    parser.add_argument("--intermediate_size", type=int, default=1536,
                       help="Size of feedforward network")
    parser.add_argument("--max_seq_length", type=int, default=512,
                       help="Maximum sequence length")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    
    # Debugging options
    parser.add_argument("--use_mock", action="store_true",
                       help="Use mock implementation (for testing)")
    
    args = parser.parse_args()
    
    # Start training
    train_on_grammar(
        output_dir=args.output_dir,
        combined_dataset_path=args.combined_dataset_path,
        tokenizer_path=args.tokenizer_path,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        use_mock=args.use_mock
    )


if __name__ == "__main__":
    main() 