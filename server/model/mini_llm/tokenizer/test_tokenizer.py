#!/usr/bin/env python3
"""
Test script for the BPE tokenizer implementation.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import the BPE tokenizer
from tokenizer.bpe_tokenizer import BPETokenizer

def test_tokenizer_basic():
    """Test basic tokenizer functionality."""
    # Sample training data
    train_texts = [
        "This is a test sentence for BPE tokenization.",
        "BPE stands for Byte-Pair Encoding.",
        "Language models use tokenizers to process text.",
        "Tokenization is a crucial step in natural language processing."
    ]
    
    # Create tokenizer
    tokenizer = BPETokenizer(vocab_size=100)
    
    # Train the tokenizer
    print("Training tokenizer...")
    tokenizer.train(train_texts, num_merges=20)
    
    # Show initial vocabulary size
    print(f"Vocabulary size after training: {len(tokenizer.vocab)}")
    
    # Test encoding and decoding
    test_text = "This is a new test sentence."
    print(f"\nTest text: '{test_text}'")
    
    # Encode
    token_ids = tokenizer.encode(test_text)
    print(f"Encoded token IDs: {token_ids}")
    
    # Decode
    decoded_text = tokenizer.decode(token_ids)
    print(f"Decoded text: '{decoded_text}'")
    
    return tokenizer

def test_tokenizer_save_load():
    """Test saving and loading the tokenizer."""
    # First create and train a tokenizer
    tokenizer = test_tokenizer_basic()
    
    # Save directory
    save_dir = "./test_tokenizer"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save tokenizer
    print(f"\nSaving tokenizer to {save_dir}...")
    tokenizer.save(save_dir)
    
    # Load tokenizer
    print(f"Loading tokenizer from {save_dir}...")
    loaded_tokenizer = BPETokenizer.load(save_dir)
    
    # Test loaded tokenizer
    test_text = "This is another test sentence."
    print(f"\nTest text: '{test_text}'")
    
    # Compare encoding with original and loaded tokenizer
    original_ids = tokenizer.encode(test_text)
    loaded_ids = loaded_tokenizer.encode(test_text)
    
    print(f"Original tokenizer IDs: {original_ids}")
    print(f"Loaded tokenizer IDs: {loaded_ids}")
    print(f"Encodings match: {original_ids == loaded_ids}")
    
    # Check vocabulary size
    print(f"Original vocab size: {len(tokenizer.vocab)}")
    print(f"Loaded vocab size: {len(loaded_tokenizer.vocab)}")

def test_tokenizer_special_tokens():
    """Test the handling of special tokens."""
    # Custom special tokens
    special_tokens = {
        "<PAD>": 0,
        "<BOS>": 1,
        "<EOS>": 2,
        "<UNK>": 3,
        "<MASK>": 4,
        "<SEP>": 5,
        "<CLS>": 6,
    }
    
    # Create tokenizer with custom special tokens
    tokenizer = BPETokenizer(vocab_size=100, special_tokens=special_tokens)
    
    # Train on minimal data
    tokenizer.train(["This is a test."], num_merges=5)
    
    # Test special token handling
    test_text = "This is a test."
    print(f"\nTest text: '{test_text}'")
    
    # Encode with special tokens
    token_ids = tokenizer.encode(test_text, add_special_tokens=True)
    print(f"Encoded with special tokens: {token_ids}")
    
    # Decode with special tokens
    decoded_with_special = tokenizer.decode(token_ids, skip_special_tokens=False)
    print(f"Decoded with special tokens: '{decoded_with_special}'")
    
    # Decode without special tokens
    decoded_without_special = tokenizer.decode(token_ids, skip_special_tokens=True)
    print(f"Decoded without special tokens: '{decoded_without_special}'")
    
    # Check if special tokens are at the expected positions
    print(f"First token (BOS): {token_ids[0] == special_tokens['<BOS>']}")
    print(f"Last token (EOS): {token_ids[-1] == special_tokens['<EOS>']}")

def main():
    """Run all tests."""
    print("=== TESTING BASIC FUNCTIONALITY ===")
    test_tokenizer_basic()
    
    print("\n=== TESTING SAVE/LOAD FUNCTIONALITY ===")
    test_tokenizer_save_load()
    
    print("\n=== TESTING SPECIAL TOKENS HANDLING ===")
    test_tokenizer_special_tokens()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main() 