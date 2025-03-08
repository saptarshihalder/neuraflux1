#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for the Mini LLM using knowledge transfer.
This script integrates the tokenizer and knowledge transfer components
to train a small language model from scratch using teacher models.
"""

import os
import argparse
import logging
import random
from typing import List, Dict, Optional, Tuple
import json

# Import our components
from tokenizer.bpe_tokenizer import BPETokenizer
from training.knowledge_transfer import KnowledgeTransferSystem
from model.transformer import MiniTransformer, TransformerConfig

# For demonstration purposes when C++ components aren't compiled
from training.mock_training_demo import MockKnowledgeTransfer, MockStudentModel, MockTeacherModel


def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO) -> None:
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_training_data(data_path: str) -> List[str]:
    """
    Load training data from a file.
    Supports .txt (one example per line) or .json (list of strings).
    """
    logging.info(f"Loading training data from {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data file not found: {data_path}")
    
    if data_path.endswith('.txt'):
        with open(data_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    elif data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                raise ValueError("JSON file should contain a list of strings")
    
    else:
        raise ValueError("Unsupported file format. Use .txt or .json")


def create_tokenizer(
    train_texts: List[str],
    vocab_size: int = 30000,
    min_frequency: int = 2,
    save_dir: Optional[str] = None
) -> BPETokenizer:
    """Create and train a BPE tokenizer."""
    logging.info(f"Creating tokenizer with vocab size {vocab_size}")
    
    # Initialize tokenizer
    tokenizer = BPETokenizer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens={
            "<pad>": 0,
            "<sos>": 1,
            "<eos>": 2,
            "<unk>": 3
        }
    )
    
    # Train the tokenizer
    logging.info(f"Training tokenizer on {len(train_texts)} texts")
    tokenizer.train(train_texts)
    
    # Save the tokenizer if a directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        tokenizer_path = os.path.join(save_dir, "tokenizer")
        logging.info(f"Saving tokenizer to {tokenizer_path}")
        tokenizer.save(tokenizer_path)
    
    return tokenizer


def train_model(
    training_data: List[str],
    output_dir: str = "output",
    tokenizer_path: Optional[str] = None,
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
    Train the Mini LLM using knowledge transfer.
    
    Args:
        training_data: List of training examples
        output_dir: Directory to save model checkpoints
        tokenizer_path: Path to pre-trained tokenizer (if None, will train from scratch)
        vocab_size: Size of the vocabulary
        hidden_size: Hidden dimension of the model
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        intermediate_size: Size of feedforward network
        max_seq_length: Maximum sequence length
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        use_mock: Whether to use mock implementation (for testing)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup tokenizer
    if tokenizer_path and os.path.exists(tokenizer_path):
        logging.info(f"Loading pre-trained tokenizer from {tokenizer_path}")
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        logging.info("Training tokenizer from scratch")
        tokenizer = create_tokenizer(
            training_data[:min(1000, len(training_data))],  # Use subset for tokenizer training
            vocab_size=vocab_size,
            save_dir=output_dir
        )
    
    # Log tokenizer info
    logging.info(f"Tokenizer vocab size: {len(tokenizer.vocab)}")
    
    # Setup training
    if use_mock:
        logging.info("Using mock implementation (for demonstration)")
        
        # Create mock knowledge transfer system
        kt_system = MockKnowledgeTransfer()
        
        # Add mock teacher models
        llama_teacher = MockTeacherModel("LLaMA", vocab_size=vocab_size, hidden_size=768)
        flux_teacher = MockTeacherModel("Flux", vocab_size=vocab_size, hidden_size=512)
        kt_system.add_teacher(llama_teacher)
        kt_system.add_teacher(flux_teacher)
        
        # Train with mock system
        logging.info(f"Starting mock training for {num_epochs} epochs")
        total_steps = (len(training_data) // batch_size) * num_epochs
        kt_system.train(
            training_data, 
            num_steps=total_steps,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
    else:
        logging.info("Initializing real knowledge transfer system")
        
        # Create model configuration
        config = TransformerConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            max_seq_length=max_seq_length
        )
        
        # Initialize model
        model = MiniTransformer(config)
        
        # Initialize knowledge transfer system
        kt_system = KnowledgeTransferSystem(
            model=model,
            tokenizer=tokenizer,
            output_dir=output_dir
        )
        
        # Add teacher models
        kt_system.load_teacher_model(
            model_type="llama",
            model_path="models/llama"
        )
        
        kt_system.load_teacher_model(
            model_type="flux",
            model_path="models/flux"
        )
        
        # Start training
        logging.info(f"Starting knowledge transfer training for {num_epochs} epochs")
        kt_system.train(
            training_data=training_data,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            max_seq_length=max_seq_length
        )
        
        # Save the final model
        model_path = os.path.join(output_dir, "final_model")
        logging.info(f"Saving final model to {model_path}")
        kt_system.save_model(model_path)

    logging.info("Training complete!")


def main():
    """Main function to parse arguments and start training."""
    parser = argparse.ArgumentParser(description="Train a Mini LLM using knowledge transfer")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training data file (.txt or .json)")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save model checkpoints and logs")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to pre-trained tokenizer (if not provided, will train from scratch)")
    
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
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Debugging options
    parser.add_argument("--use_mock", action="store_true",
                        help="Use mock implementation (for testing)")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logging(os.path.join(args.output_dir, "logs"), log_level)
    
    # Log arguments
    logging.info(f"Arguments: {args}")
    
    # Load training data
    training_data = load_training_data(args.data_path)
    logging.info(f"Loaded {len(training_data)} training examples")
    
    # Start training
    train_model(
        training_data=training_data,
        output_dir=args.output_dir,
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