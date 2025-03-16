import os
import json
import regex as re
from typing import Dict, List, Tuple, Optional, Set, Union
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm
import itertools

class NanoTokenizer:
    """
    Byte-level BPE tokenizer optimized for small language models.
    Implements a vocabulary-efficient tokenization strategy with dynamic merging.
    """
    
    def __init__(
        self, 
        vocab_size: int = 16000, 
        min_frequency: int = 2,
        special_tokens: Dict[str, str] = None
    ):
        # Default special tokens
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        
        # Override with custom special tokens if provided
        if special_tokens:
            for key, value in special_tokens.items():
                setattr(self, f"{key}_token", value)
        
        # Initialize vocabulary with special tokens
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        
        # Add special tokens to vocabulary
        self._add_special_tokens()
        
        # Set vocabulary size and build initial byte vocabulary
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Initialize regex pattern for tokenization
        self.pattern = None
        
        # Initialize byte-level vocabulary
        self._init_byte_vocab()
        
        # Dictionary to cache tokenized results
        self.cache = {}
        self.cache_max_size = 10000
    
    def _add_special_tokens(self):
        """Add special tokens to vocabulary."""
        special_tokens = [self.bos_token, self.eos_token, self.pad_token, self.unk_token]
        for i, token in enumerate(special_tokens):
            self.vocab[token] = i
            self.inv_vocab[i] = token
    
    def _init_byte_vocab(self):
        """Initialize base vocabulary with byte tokens."""
        # Start after special tokens
        token_id = len(self.vocab)
        
        # Add single byte tokens (0-255)
        for i in range(256):
            byte_token = bytes([i]).decode('latin-1')
            if byte_token not in self.vocab:
                self.vocab[byte_token] = token_id
                self.inv_vocab[token_id] = byte_token
                token_id += 1

    def train_from_texts(self, texts: List[str], num_merges: Optional[int] = None):
        """
        Train tokenizer from a list of texts.
        
        Args:
            texts: List of text samples
            num_merges: Number of BPE merges to perform (if None, uses vocab_size)
        """
        print(f"Training tokenizer from {len(texts)} texts")
        
        # Build initial vocabulary from bytes
        vocab = set(self.vocab.keys())
        
        # Tokenize texts into bytes
        byte_encoded_texts = []
        for text in tqdm(texts, desc="Byte encoding texts"):
            # Latin-1 encoding preserves byte values
            bytes_text = [c.encode('latin-1').decode('latin-1') for c in text]
            byte_encoded_texts.append(bytes_text)
        
        # Count token pairs
        if num_merges is None:
            # Calculate space for merges (vocab_size minus special tokens and bytes)
            num_merges = self.vocab_size - len(self.vocab)
        
        # Apply BPE merges
        merges = self._learn_bpe(byte_encoded_texts, num_merges)
        
        # Apply merges to update vocabulary
        self._update_vocab_with_merges(merges)
        
        # Update the tokenization pattern
        self._compile_pattern()
        
        print(f"Trained tokenizer with {len(self.vocab)} tokens")
    
    def _learn_bpe(self, texts: List[List[str]], num_merges: int) -> List[Tuple[str, str]]:
        """
        Learn BPE merges from tokenized texts.
        
        Args:
            texts: List of tokenized texts
            num_merges: Number of merges to perform
            
        Returns:
            List of merge operations (pairs)
        """
        merges = []
        text_copies = [text.copy() for text in texts]
        
        for i in tqdm(range(num_merges), desc="Learning BPE merges"):
            # Count pairs
            pair_counts = Counter()
            for text in text_copies:
                if len(text) < 2:
                    continue
                
                # Count consecutive pairs
                for j in range(len(text) - 1):
                    pair = (text[j], text[j + 1])
                    pair_counts[pair] += 1
            
            if not pair_counts:
                break
            
            # Find most frequent pair
            best_pair = max(pair_counts.items(), key=lambda x: x[1])[0]
            
            # Skip pairs that occur less than min_frequency
            if pair_counts[best_pair] < self.min_frequency:
                break
                
            # Create new token by concatenating pair
            new_token = best_pair[0] + best_pair[1]
            
            # Record the merge
            merges.append(best_pair)
            
            # Apply the merge to all texts
            for k in range(len(text_copies)):
                text = text_copies[k]
                new_text = []
                
                # Apply merge
                i = 0
                while i < len(text):
                    if i < len(text) - 1 and text[i] == best_pair[0] and text[i + 1] == best_pair[1]:
                        new_text.append(new_token)
                        i += 2
                    else:
                        new_text.append(text[i])
                        i += 1
                
                text_copies[k] = new_text
        
        return merges
    
    def _update_vocab_with_merges(self, merges: List[Tuple[str, str]]):
        """Update vocabulary with learned BPE merges."""
        token_id = len(self.vocab)
        
        for pair in merges:
            new_token = pair[0] + pair[1]
            if new_token not in self.vocab and token_id < self.vocab_size:
                self.vocab[new_token] = token_id
                self.inv_vocab[token_id] = new_token
                token_id += 1
    
    def _compile_pattern(self):
        """Compile regex pattern for tokenization."""
        # Escape special characters for regex pattern
        tokens = sorted(self.vocab.keys(), key=len, reverse=True)
        escaped_tokens = [re.escape(token) for token in tokens if token not in [self.bos_token, self.eos_token, self.pad_token, self.unk_token]]
        
        # Join tokens with | for regex alternation
        pattern_str = '|'.join(escaped_tokens)
        self.pattern = re.compile(pattern_str)
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        # Check cache first
        cache_key = (text, add_special_tokens)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Tokenize text
        tokens = self.tokenize(text)
        
        # Convert tokens to IDs
        ids = [self.vocab.get(token, self.vocab.get(self.unk_token)) for token in tokens]
        
        # Add special tokens if requested
        if add_special_tokens:
            ids = [self.vocab[self.bos_token]] + ids + [self.vocab[self.eos_token]]
        
        # Update cache
        if len(self.cache) >= self.cache_max_size:
            # If cache is full, clear it
            self.cache.clear()
        
        self.cache[cache_key] = ids
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs into text.
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text
        """
        # Convert IDs to tokens
        tokens = [self.inv_vocab.get(id, self.unk_token) for id in ids]
        
        # Filter out special tokens if requested
        if skip_special_tokens:
            special_tokens = [self.bos_token, self.eos_token, self.pad_token, self.unk_token]
            tokens = [token for token in tokens if token not in special_tokens]
        
        # Join tokens to reconstruct text
        text = ''.join(tokens)
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into tokens.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        if not self.pattern:
            # If pattern not compiled, use byte fallback
            return [c.encode('latin-1').decode('latin-1') for c in text]
        
        # Use regex pattern to split text into tokens
        matches = self.pattern.findall(text)
        
        # Handle any unmatched parts
        current_idx = 0
        result = []
        
        for match in matches:
            start_idx = text.find(match, current_idx)
            
            # If there's unmatched text before this match
            if start_idx > current_idx:
                unmatched = text[current_idx:start_idx]
                # Tokenize unmatched text byte by byte
                for char in unmatched:
                    byte_char = char.encode('latin-1').decode('latin-1')
                    result.append(byte_char)
            
            # Add the matched token
            result.append(match)
            current_idx = start_idx + len(match)
        
        # Handle any unmatched text at the end
        if current_idx < len(text):
            unmatched = text[current_idx:]
            for char in unmatched:
                byte_char = char.encode('latin-1').decode('latin-1')
                result.append(byte_char)
        
        return result
    
    def save_vocab(self, vocab_path: str):
        """Save vocabulary to a file."""
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, 'w', encoding='utf-8') as f:
            # Encode bytes as hex for safe serialization
            serializable_vocab = {}
            for token, idx in self.vocab.items():
                # Special tokens saved as is
                if token in [self.bos_token, self.eos_token, self.pad_token, self.unk_token]:
                    serializable_vocab[token] = idx
                else:
                    # Save byte tokens as hex for safety
                    hex_token = ''.join([f'\\x{ord(c):02x}' for c in token])
                    serializable_vocab[hex_token] = idx
            
            json.dump(serializable_vocab, f, ensure_ascii=False, indent=2)
    
    def load_vocab(self, vocab_path: str):
        """Load vocabulary from a file."""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            serialized_vocab = json.load(f)
            
            # Clear existing vocab
            self.vocab = {}
            self.inv_vocab = {}
            
            # Load vocab
            for token_hex, idx in serialized_vocab.items():
                # Check if this is a special token (not hex encoded)
                if token_hex in [self.bos_token, self.eos_token, self.pad_token, self.unk_token]:
                    token = token_hex
                else:
                    # Parse hex back to bytes
                    token = ''
                    i = 0
                    while i < len(token_hex):
                        if token_hex[i:i+2] == '\\x':
                            # Parse hex byte
                            byte_val = int(token_hex[i+2:i+4], 16)
                            token += bytes([byte_val]).decode('latin-1')
                            i += 4
                        else:
                            # Regular character
                            token += token_hex[i]
                            i += 1
                
                self.vocab[token] = idx
                self.inv_vocab[idx] = token
            
            # Recompile pattern
            self._compile_pattern()
    
    def __len__(self):
        return len(self.vocab) 