#!/usr/bin/env python3
"""
BPE Tokenizer Implementation From Scratch

This module implements a Byte-Pair Encoding (BPE) tokenizer without
relying on external libraries. It includes the full BPE algorithm,
vocabulary building, and special token handling.
"""

import os
import re
import json
import collections
from typing import Dict, List, Tuple, Set, Optional, Union, Any


class BPETokenizer:
    """
    A from-scratch implementation of Byte-Pair Encoding tokenization.
    
    BPE is a data compression technique that iteratively replaces the most
    common pair of consecutive bytes (or characters) with a single unused byte.
    For tokenization, we use this to build a vocabulary of subwords.
    """
    
    def __init__(
        self,
        vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens: Dict[str, int] = None
    ):
        """
        Initialize the BPE tokenizer.
        
        Args:
            vocab_size: Maximum size of the vocabulary
            min_frequency: Minimum frequency for a token pair to be merged
            special_tokens: Dictionary of special tokens and their ids
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Initialize with default special tokens if none provided
        if special_tokens is None:
            self.special_tokens = {
                "<PAD>": 0,
                "<BOS>": 1,
                "<EOS>": 2,
                "<UNK>": 3,
                "<MASK>": 4,
                "<QUERY>": 5,
                "<TEACHER_RESPONSE>": 6,
            }
        else:
            self.special_tokens = special_tokens
            
        # Reverse mapping for token ids to tokens
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        
        # Initialize empty vocabulary and merges
        self.vocab = {}  # token -> id
        self.merges = {}  # (token1, token2) -> merged_token
        
        # Pre-populate with special tokens
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
            
        # Start vocabulary with basic bytes (characters)
        self._initialize_base_vocab()
    
    def add_token(self, token: str) -> int:
        """
        Add a new token to the vocabulary.
        
        Args:
            token: The token to add
            
        Returns:
            The ID of the token (new or existing)
        """
        if token in self.vocab:
            # Token already exists
            return self.vocab[token]
        
        # Add new token with the next available ID
        new_id = len(self.vocab)
        self.vocab[token] = new_id
        self.id_to_token[new_id] = token
        
        return new_id
    
    def add_tokens(self, tokens: List[str]) -> List[int]:
        """
        Add multiple tokens to the vocabulary.
        
        Args:
            tokens: List of tokens to add
            
        Returns:
            List of token IDs (new or existing)
        """
        return [self.add_token(token) for token in tokens]
    
    def _initialize_base_vocab(self):
        """Initialize the vocabulary with basic byte values."""
        # Add single byte tokens (0-255)
        next_id = len(self.vocab)
        for i in range(256):
            byte_token = bytes([i]).decode('latin-1')
            if byte_token not in self.vocab:
                self.vocab[byte_token] = next_id
                self.id_to_token[next_id] = byte_token
                next_id += 1
    
    def train(self, texts: List[str], num_merges: Optional[int] = None) -> None:
        """
        Train the BPE tokenizer on a corpus of texts.
        
        Args:
            texts: List of training texts
            num_merges: Number of merge operations to perform. If None, will merge
                        until vocab_size is reached.
        """
        if num_merges is None:
            num_merges = self.vocab_size - len(self.vocab)
            
        # Preprocess text to get initial tokens
        word_freqs = self._get_word_frequencies(texts)
        
        # Compute initial stats (character pairs)
        pairs = self._get_pair_frequencies(word_freqs)
        
        # Perform BPE algorithm
        for i in range(num_merges):
            if not pairs:
                break
                
            # Find the most frequent pair
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < self.min_frequency:
                break
                
            # Create a new token from the pair
            new_token = ''.join(best_pair)
            next_id = len(self.vocab)
            
            # Add to vocabulary if not already present
            if new_token not in self.vocab:
                self.vocab[new_token] = next_id
                self.id_to_token[next_id] = new_token
                
            # Add to merges dictionary
            self.merges[best_pair] = new_token
            
            # Update word frequencies with the new merged token
            word_freqs = self._apply_merge(word_freqs, best_pair, new_token)
            
            # Recompute pair frequencies
            pairs = self._get_pair_frequencies(word_freqs)
            
            # Check if we've reached the vocabulary size limit
            if len(self.vocab) >= self.vocab_size:
                break
                
        print(f"BPE training complete. Vocabulary size: {len(self.vocab)}")
    
    def _get_word_frequencies(self, texts: List[str]) -> Dict[Tuple[str, ...], int]:
        """
        Count word frequencies in the training texts.
        
        Args:
            texts: List of training texts
            
        Returns:
            Dictionary mapping words (as tuples of characters) to their frequencies
        """
        word_freqs = collections.defaultdict(int)
        
        for text in texts:
            # Apply basic preprocessing
            text = text.lower()
            # Split into words (can be customized)
            words = re.findall(r'\w+|[^\w\s]', text)
            
            for word in words:
                # Convert each word to a tuple of characters
                char_tuple = tuple(c for c in word)
                word_freqs[char_tuple] += 1
                
        return word_freqs
    
    def _get_pair_frequencies(self, word_freqs: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        """
        Count frequencies of adjacent character pairs across all words.
        
        Args:
            word_freqs: Dictionary mapping words to their frequencies
            
        Returns:
            Dictionary mapping character pairs to their frequencies
        """
        pair_freqs = collections.defaultdict(int)
        
        for word, freq in word_freqs.items():
            # Count pairs within each word
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freqs[pair] += freq
                
        return pair_freqs
    
    def _apply_merge(self, word_freqs: Dict[Tuple[str, ...], int], 
                    pair: Tuple[str, str], new_token: str) -> Dict[Tuple[str, ...], int]:
        """
        Apply a merge operation to all words in the frequency dictionary.
        
        Args:
            word_freqs: Dictionary mapping words to their frequencies
            pair: The pair of tokens to merge
            new_token: The new token created from the merge
            
        Returns:
            Updated word frequency dictionary
        """
        updated_word_freqs = {}
        
        for word, freq in word_freqs.items():
            # Create a new word by applying the merge
            new_word = []
            i = 0
            while i < len(word):
                # If we find the pair to merge
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
                    
            # Update the frequency dictionary
            updated_word_freqs[tuple(new_word)] = freq
            
        return updated_word_freqs
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add special tokens like BOS/EOS
            
        Returns:
            List of token IDs
        """
        # Preprocess text
        text = text.lower()
        
        # Tokenize into words first
        words = re.findall(r'\w+|[^\w\s]', text)
        
        token_ids = []
        
        # Add beginning of sequence token
        if add_special_tokens:
            token_ids.append(self.special_tokens["<BOS>"])
        
        # Process each word
        for word in words:
            # Start with character-level tokens
            tokens = list(word)
            
            # Apply merges iteratively
            i = 0
            while i < len(tokens) - 1:
                pair = (tokens[i], tokens[i + 1])
                
                if pair in self.merges:
                    merged_token = self.merges[pair]
                    tokens[i] = merged_token
                    tokens.pop(i + 1)
                else:
                    i += 1
            
            # Convert tokens to IDs
            for token in tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                else:
                    # Handle unknown tokens
                    token_ids.append(self.special_tokens["<UNK>"])
                    
            # Add space between words (optional)
            if " " in self.vocab:
                token_ids.append(self.vocab[" "])
        
        # Add end of sequence token
        if add_special_tokens:
            token_ids.append(self.special_tokens["<EOS>"])
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in the output
            
        Returns:
            Decoded text
        """
        tokens = []
        
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                
                # Skip special tokens if requested
                if skip_special_tokens and token in self.special_tokens:
                    continue
                    
                tokens.append(token)
            else:
                # Handle unknown token IDs
                tokens.append(self.id_to_token.get(self.special_tokens["<UNK>"]))
        
        # Simple string concatenation (can be improved)
        return ''.join(tokens)
    
    def save(self, directory: str) -> None:
        """
        Save the tokenizer to disk.
        
        Args:
            directory: Directory where to save tokenizer files
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save vocabulary
        vocab_path = os.path.join(directory, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
            
        # Save merges
        merges_path = os.path.join(directory, "merges.json")
        merges_dict = {" ".join(pair): merged for pair, merged in self.merges.items()}
        with open(merges_path, "w", encoding="utf-8") as f:
            json.dump(merges_dict, f, ensure_ascii=False, indent=2)
            
        # Save special tokens
        special_tokens_path = os.path.join(directory, "special_tokens.json")
        with open(special_tokens_path, "w", encoding="utf-8") as f:
            json.dump(self.special_tokens, f, ensure_ascii=False, indent=2)
            
        # Save configuration
        config_path = os.path.join(directory, "config.json")
        config = {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, directory: str) -> 'BPETokenizer':
        """
        Load a tokenizer from disk.
        
        Args:
            directory: Directory containing the tokenizer files
            
        Returns:
            BPETokenizer: Loaded tokenizer
        """
        # Load configuration
        config_path = os.path.join(directory, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {"vocab_size": 30000, "min_frequency": 2}
            
        # Load special tokens
        special_tokens_path = os.path.join(directory, "special_tokens.json")
        if os.path.exists(special_tokens_path):
            with open(special_tokens_path, "r", encoding="utf-8") as f:
                special_tokens = json.load(f)
        else:
            special_tokens = None
            
        # Create tokenizer instance
        tokenizer = cls(
            vocab_size=config.get("vocab_size", 30000),
            min_frequency=config.get("min_frequency", 2),
            special_tokens=special_tokens
        )
        
        # Load vocabulary
        vocab_path = os.path.join(directory, "vocab.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            tokenizer.vocab = json.load(f)
            
        # Set up reverse mapping
        tokenizer.id_to_token = {v: k for k, v in tokenizer.vocab.items()}
        
        # Load merges
        merges_path = os.path.join(directory, "merges.json")
        with open(merges_path, "r", encoding="utf-8") as f:
            merges_dict = json.load(f)
            tokenizer.merges = {tuple(k.split()): v for k, v in merges_dict.items()}
            
        return tokenizer

# Example usage
if __name__ == "__main__":
    # Sample training data
    sample_texts = [
        "This is a sample text for BPE tokenizer training.",
        "BPE works by iteratively merging the most frequent pairs of tokens.",
        "We start with characters and build up to subwords and words.",
        "This custom implementation follows the core BPE algorithm without external dependencies.",
        "Special tokens like <BOS> and <EOS> mark the beginning and end of sequences.",
        "The tokenizer can be trained on a corpus of texts to build a vocabulary.",
        "It encodes text into token IDs and decodes token IDs back to text.",
        "The vocabulary size can be customized based on the application needs.",
    ]
    
    # Create and train tokenizer
    tokenizer = BPETokenizer(vocab_size=1000)
    tokenizer.train(sample_texts, num_merges=200)
    
    # Test encoding and decoding
    test_text = "This is a test. How well does the tokenizer work?"
    token_ids = tokenizer.encode(test_text)
    reconstructed = tokenizer.decode(token_ids)
    
    print(f"Original: {test_text}")
    print(f"Token IDs: {token_ids}")
    print(f"Reconstructed: {reconstructed}")
    
    # Save and load test
    tokenizer.save("./bpe_tokenizer")
    loaded_tokenizer = BPETokenizer.load("./bpe_tokenizer")
    
    loaded_token_ids = loaded_tokenizer.encode(test_text)
    print(f"Loaded tokenizer token IDs: {loaded_token_ids}")
    print(f"Tokens match: {token_ids == loaded_token_ids}") 