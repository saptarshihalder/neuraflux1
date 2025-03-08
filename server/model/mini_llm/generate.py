#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text generation script for MiniLLM.
This script demonstrates how to use a trained model for generating text.
"""

import os
import argparse
import logging
from typing import List, Optional
import time

from tokenizer.bpe_tokenizer import BPETokenizer
from model.transformer import MiniTransformer, TransformerConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def load_model_and_tokenizer(model_path: str, tokenizer_path: Optional[str] = None) -> tuple:
    """
    Load a trained model and tokenizer.
    
    Args:
        model_path: Path to the model directory
        tokenizer_path: Path to the tokenizer directory (optional, defaults to model_path/tokenizer)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Determine tokenizer path if not provided
    if tokenizer_path is None:
        tokenizer_path = os.path.join(model_path, "tokenizer")
    
    # Check if paths exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_path}")
    
    # Load tokenizer
    logging.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = BPETokenizer.load(tokenizer_path)
    
    # Check if this is a mock model
    mock_path = os.path.join(model_path, "model.bin.mock")
    config_path = os.path.join(model_path, "config.json")
    
    if os.path.exists(mock_path):
        logging.info("Detected mock model, using mock implementation")
        
        # If config exists, load it
        if os.path.exists(config_path):
            config = TransformerConfig.load(config_path)
        else:
            # Create default config
            logging.info("No config found, using default configuration")
            config = TransformerConfig(vocab_size=len(tokenizer.vocab))
        
        # Create model directly
        model = MiniTransformer(config)
    else:
        # Load real model
        logging.info(f"Loading model from {model_path}")
        model = MiniTransformer.load(model_path)
    
    return model, tokenizer


def generate_text(
    model: MiniTransformer,
    tokenizer: BPETokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    num_samples: int = 1,
    show_tokens: bool = False
) -> List[str]:
    """
    Generate text from a prompt.
    
    Args:
        model: Transformer model
        tokenizer: BPE tokenizer
        prompt: Input prompt
        max_length: Maximum length of generated text
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling probability
        top_k: Top-k sampling parameter
        num_samples: Number of different samples to generate
        show_tokens: Whether to show token IDs during generation
        
    Returns:
        List of generated texts
    """
    # Set model to evaluation mode
    model.eval()
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt)
    
    if show_tokens:
        print(f"Prompt token IDs: {input_ids}")
    
    results = []
    
    for i in range(num_samples):
        logging.info(f"Generating sample {i+1}/{num_samples}")
        
        # Generate text
        start_time = time.time()
        output_ids = model.generate(
            prompt=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        generation_time = time.time() - start_time
        
        # Decode the generated text
        generated_text = tokenizer.decode(output_ids)
        
        if show_tokens:
            print(f"Generated token IDs: {output_ids}")
        
        # Add to results
        results.append(generated_text)
        
        # Print generation info
        tokens_per_second = (len(output_ids) - len(input_ids)) / generation_time
        logging.info(f"Generated {len(output_ids) - len(input_ids)} tokens in {generation_time:.2f}s "
                     f"({tokens_per_second:.2f} tokens/s)")
    
    return results


def main():
    """Parse arguments and generate text."""
    parser = argparse.ArgumentParser(description="Generate text using MiniLLM")
    
    # Model and tokenizer paths
    parser.add_argument("--model_path", type=str, default="output",
                        help="Path to the model directory")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to the tokenizer directory (defaults to model_path/tokenizer)")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, required=True,
                        help="Input prompt for text generation")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling probability")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of different samples to generate")
    
    # Display options
    parser.add_argument("--show_tokens", action="store_true",
                        help="Show token IDs during generation")
    
    args = parser.parse_args()
    
    try:
        print("Starting text generation...")
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path
        )
        
        print(f"Model loaded successfully: {model}")
        print(f"Tokenizer loaded successfully: {tokenizer}")
        
        # Generate text
        print(f"Generating text from prompt: '{args.prompt}'")
        generated_texts = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_samples=args.num_samples,
            show_tokens=args.show_tokens
        )
        
        # Print results
        print("\n" + "="*50)
        print("GENERATED SAMPLES:")
        print("="*50)
        
        for i, text in enumerate(generated_texts):
            print(f"\nSample {i+1}:")
            print(f"{text}")
        
    except Exception as e:
        logging.error(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main() 