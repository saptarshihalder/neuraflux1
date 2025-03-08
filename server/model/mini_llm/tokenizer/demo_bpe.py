#!/usr/bin/env python3
"""
Demo script for visualizing the BPE merging process.
"""

import os
import sys
import json
import re
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import the BPE tokenizer
from tokenizer.bpe_tokenizer import BPETokenizer

def visualize_bpe_training(train_texts, num_merges=50, step=5):
    """
    Visualize the BPE training process by showing merges step by step.
    
    Args:
        train_texts: List of training texts
        num_merges: Total number of merge operations to perform
        step: Show progress every N steps
    """
    print("=== BPE TRAINING VISUALIZATION ===\n")
    print(f"Training data ({len(train_texts)} texts):")
    for i, text in enumerate(train_texts):
        print(f"  Text {i+1}: '{text}'")
    print()
    
    # Create initial tokenizer
    tokenizer = BPETokenizer(vocab_size=1000)
    
    # Get initial vocabulary (base characters)
    tokenizer._initialize_base_vocab()
    base_vocab_size = len(tokenizer.vocab)
    print(f"Initial vocabulary size (base characters): {base_vocab_size}")
    
    # Get word frequencies
    word_freqs = tokenizer._get_word_frequencies(train_texts)
    print(f"Number of distinct words: {len(word_freqs)}")
    print()
    
    # Get initial pair frequencies
    pairs = tokenizer._get_pair_frequencies(word_freqs)
    
    # Show most common pairs initially
    print("Top 5 most frequent character pairs before any merges:")
    top_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:5]
    for pair, freq in top_pairs:
        print(f"  '{pair[0]}' + '{pair[1]}' = {freq} occurrences")
    print()
    
    # Perform BPE algorithm in steps
    test_text = "This is an example sentence to tokenize."
    
    # Use the basic word splitting from the tokenizer's encode method
    test_words = re.findall(r'\w+|[^\w\s]', test_text.lower())
    test_tokens_orig = []
    for word in test_words:
        test_tokens_orig.extend(list(word))
    
    print(f"Initial tokenization of '{test_text}':")
    print(f"  {test_tokens_orig}")
    print(f"  Token count: {len(test_tokens_orig)}")
    print()
    
    print("Performing BPE merges step by step:")
    
    for i in range(1, num_merges + 1):
        if not pairs:
            print("No more valid pairs to merge.")
            break
            
        # Find the most frequent pair
        best_pair = max(pairs, key=pairs.get)
        freq = pairs[best_pair]
        
        # Create a new token from the pair
        new_token = ''.join(best_pair)
        next_id = len(tokenizer.vocab)
        
        # Add to vocabulary
        if new_token not in tokenizer.vocab:
            tokenizer.vocab[new_token] = next_id
            tokenizer.id_to_token[next_id] = new_token
            
        # Add to merges dictionary
        tokenizer.merges[best_pair] = new_token
        
        # Update word frequencies with the new merged token
        word_freqs = tokenizer._apply_merge(word_freqs, best_pair, new_token)
        
        # Recompute pair frequencies
        pairs = tokenizer._get_pair_frequencies(word_freqs)
        
        # Print progress at each step
        if i % step == 0 or i == 1:
            print(f"\nMerge #{i}:")
            print(f"  Merged '{best_pair[0]}' + '{best_pair[1]}' → '{new_token}' (frequency: {freq})")
            print(f"  Vocabulary size: {len(tokenizer.vocab)}")
            
            # Show tokenization of test text after this merge
            test_tokens = []
            for word in test_words:
                # Reconstruct words applying merges iteratively
                tokens = list(word)
                j = 0
                while j < len(tokens) - 1:
                    pair = (tokens[j], tokens[j + 1])
                    if pair in tokenizer.merges:
                        merged_token = tokenizer.merges[pair]
                        tokens[j] = merged_token
                        tokens.pop(j + 1)
                    else:
                        j += 1
                test_tokens.extend(tokens)
            
            print(f"  Tokenization of test text: {test_tokens}")
            print(f"  Token count: {len(test_tokens)} (reduced from {len(test_tokens_orig)})")
    
    print("\n=== FINAL RESULTS ===")
    print(f"Total merges performed: {i}")
    print(f"Final vocabulary size: {len(tokenizer.vocab)}")
    
    # Encode and decode the test text
    token_ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(token_ids)
    
    print(f"\nFinal encoding of '{test_text}':")
    print(f"  Token IDs: {token_ids}")
    print(f"  Decoded: '{decoded}'")
    
    return tokenizer

def main():
    """Run the BPE visualization demo."""
    # Simple example texts
    train_texts = [
        "This is a simple example for BPE tokenization.",
        "BPE works by merging the most frequent pairs of characters.",
        "The algorithm starts with individual characters and builds up tokens.",
        "This approach is effective for handling rare and unseen words.",
        "Tokenization is a crucial preprocessing step for language models."
    ]
    
    # Visualize BPE training with 30 merges, showing progress every 5 steps
    tokenizer = visualize_bpe_training(train_texts, num_merges=30, step=5)
    
    # Save the trained tokenizer
    save_dir = "./demo_tokenizer"
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(save_dir)
    print(f"\nTokenizer saved to {save_dir}")

if __name__ == "__main__":
    main() 