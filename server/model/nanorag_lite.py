"""
NeuraFlux Small Language Model (Lite Version)
This is a simplified, conceptual implementation of a transformer-based language model.
This version focuses on demonstrating the core concepts without requiring external dependencies.
"""

import re
import sys
import json
import os
import math
import random
from typing import List, Dict, Any, Optional, Tuple, Union

# Simple matrix operations without numpy/torch
class Tensor:
    def __init__(self, data=None, shape=None):
        if data is not None:
            self.data = data
        elif shape is not None:
            # Initialize with zeros
            self.data = self._create_zeros(shape)
        else:
            self.data = []
        
    def _create_zeros(self, shape):
        if len(shape) == 1:
            return [0.0] * shape[0]
        else:
            return [self._create_zeros(shape[1:]) for _ in range(shape[0])]
    
    def shape(self):
        if not self.data:
            return (0,)
        if isinstance(self.data[0], list):
            return (len(self.data), len(self.data[0]))
        else:
            return (len(self.data),)
            
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __setitem__(self, idx, value):
        self.data[idx] = value

# Configuration for the small language model
class NanoConfig:
    def __init__(
        self,
        vocab_size: int = 10000,
        hidden_size: int = 384,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 6,
        intermediate_size: int = 1536,
        max_position_embeddings: int = 512,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        
        # Special token IDs
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

class NanoTokenizer:
    def __init__(
        self, 
        vocab_file: Optional[str] = None, 
        vocab_size: int = 10000, 
        unk_token: str = "[UNK]", 
        pad_token: str = "[PAD]", 
        bos_token: str = "[BOS]", 
        eos_token: str = "[EOS]"
    ):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        
        # Special tokens
        self.special_tokens = {
            self.pad_token: 0,
            self.bos_token: 1,
            self.eos_token: 2,
            self.unk_token: 3,
        }
        
        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)
        else:
            # Initialize with just special tokens
            self.vocab = {k: v for k, v in self.special_tokens.items()}
            self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
    
    def load_vocab(self, vocab_file: str):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        
        # Make sure special tokens have the correct IDs
        self.vocab = {**{k: v for k, v in self.special_tokens.items()}, **vocab}
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
    
    def save_vocab(self, vocab_file: str):
        with open(vocab_file, 'w', encoding='utf-8') as f:
            # Save vocabulary without special tokens
            vocab_to_save = {k: v for k, v in self.vocab.items() if k not in self.special_tokens}
            json.dump(vocab_to_save, f, ensure_ascii=False, indent=2)
    
    def tokenize(self, text: str) -> List[str]:
        # Very basic tokenization - split by spaces and punctuation
        tokens = []
        for word in re.findall(r'\w+|[^\w\s]', text.lower()):
            tokens.append(word)
        return tokens
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.ids_to_tokens.get(id, self.unk_token) for id in ids]
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self.tokenize(text)
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        return self.convert_tokens_to_ids(tokens)
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens = self.convert_ids_to_tokens(token_ids)
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens]
        return ' '.join(tokens)
    
    def train_from_texts(self, texts: List[str], min_frequency: int = 2):
        """Train a vocabulary from a list of texts"""
        word_counts = {}
        for text in texts:
            for token in self.tokenize(text):
                if token in word_counts:
                    word_counts[token] += 1
                else:
                    word_counts[token] = 1
        
        # Filter by frequency and sort by count
        word_counts = {word: count for word, count in word_counts.items() if count >= min_frequency}
        words_sorted = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
        
        # Add to vocabulary, preserving special tokens
        vocab_size = min(self.vocab_size, len(words_sorted) + len(self.special_tokens))
        new_words = [word for word, _ in words_sorted[:vocab_size - len(self.special_tokens)]]
        
        # Reset vocab with special tokens
        self.vocab = {k: v for k, v in self.special_tokens.items()}
        
        # Add new words
        for i, word in enumerate(new_words):
            self.vocab[word] = i + len(self.special_tokens)
        
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

class NanoNLPEngine:
    """A simplified language model that demonstrates the concepts without requiring deep learning libraries"""
    
    def __init__(self, config: Optional[NanoConfig] = None, tokenizer: Optional[NanoTokenizer] = None):
        self.config = config if config is not None else NanoConfig()
        self.tokenizer = tokenizer if tokenizer is not None else NanoTokenizer(vocab_size=self.config.vocab_size)
        
        # For conceptual demonstration, we'll have a simplified knowledge store
        self.knowledge = {
            "model_info": "NeuraFlux is a small language model with 1.45M parameters. It was created by Saptarshi Halder using transformer architecture.",
            "architecture": "NeuraFlux uses a transformer architecture with self-attention mechanisms across 6 layers with 6 attention heads.",
            "capabilities": "The model can generate text, answer questions, and retrieve information from its memory.",
            "language_models": "Language models use neural networks to predict the next word in a sequence based on previous words.",
            "transformers": "Transformers are a type of neural network architecture that uses self-attention mechanisms to process sequential data.",
            "attention": "Attention mechanisms allow models to focus on relevant parts of the input when generating outputs.",
            "training": "Language models are trained on large text corpora to learn patterns and relationships in language."
        }
        
        # For factual knowledge
        self.facts = {
            "capital france": "Paris is the capital of France.",
            "largest planet": "Jupiter is the largest planet in our solar system.",
            "tallest mountain": "Mount Everest is the tallest mountain on Earth.",
            "hamlet author": "William Shakespeare wrote Hamlet.",
            "nile river": "The Nile is the longest river in the world.",
            "human bones": "The adult human body has 206 bones.",
            "dna stands for": "DNA stands for deoxyribonucleic acid.",
            "speed of light": "The speed of light is approximately 299,792 kilometers per second.",
            "water formula": "The chemical formula for water is H2O.",
            "mona lisa": "Leonardo da Vinci painted the Mona Lisa."
        }
        
        # Self-reference information
        self.self_info = {
            "name": "NeuraFlux",
            "creator": "Saptarshi Halder",
            "architecture": "Transformer",
            "layers": "6",
            "attention_heads": "6",
            "hidden_size": "384",
            "parameters": "1.45 million",
            "purpose": "Demonstrate language model concepts",
            "capabilities": "Text generation, question answering, information retrieval"
        }
    
    def _compute_similarity(self, query: str, document: str) -> float:
        """Compute simple word overlap similarity between query and document"""
        query_words = set(re.findall(r'\w+', query.lower()))
        doc_words = set(re.findall(r'\w+', document.lower()))
        
        if not query_words or not doc_words:
            return 0.0
        
        intersection = query_words.intersection(doc_words)
        return len(intersection) / ((len(query_words) + len(doc_words)) / 2)
    
    def retrieve(self, query: str, top_k: int = 2) -> List[str]:
        """Retrieve relevant documents from knowledge base"""
        similarities = []
        
        # Check knowledge base
        for key, document in self.knowledge.items():
            score = self._compute_similarity(query, document)
            similarities.append((document, score))
        
        # Sort by score and take top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in similarities[:top_k] if score > 0.1]
    
    def is_self_query(self, query: str) -> bool:
        """Check if query is about the model itself"""
        query_lower = query.lower()
        self_terms = ["you", "your", "yourself", "neuraflux"]
        
        # Check for self-reference terms
        if any(term in query_lower for term in self_terms):
            return True
        
        # Check for specific self-attributes
        attributes = ["name", "creator", "made", "built", "parameters", "architecture"]
        if any(attr in query_lower for attr in attributes):
            return True
            
        return False
    
    def answer_self_query(self, query: str) -> str:
        """Answer questions about the model itself"""
        query_lower = query.lower()
        
        # Check for specific attribute queries
        if "name" in query_lower:
            return f"My name is {self.self_info['name']}."
        elif "creator" in query_lower or "made" in query_lower or "built" in query_lower:
            return f"I was created by {self.self_info['creator']}."
        elif "architecture" in query_lower:
            return f"I use a {self.self_info['architecture']} architecture with {self.self_info['layers']} layers and {self.self_info['attention_heads']} attention heads."
        elif "parameters" in query_lower:
            return f"I have {self.self_info['parameters']} parameters."
        elif "do" in query_lower or "capable" in query_lower or "capabilities" in query_lower:
            return f"I can perform {self.self_info['capabilities']}."
        elif "purpose" in query_lower:
            return f"My purpose is to {self.self_info['purpose']}."
        
        # General self introduction
        return (f"I am {self.self_info['name']}, a {self.self_info['architecture']} language model "
                f"with {self.self_info['parameters']} parameters created by {self.self_info['creator']}.")
    
    def search_facts(self, query: str) -> Optional[str]:
        """Search factual knowledge base for answers"""
        query_lower = query.lower()
        
        # Try direct fact lookup
        for key, fact in self.facts.items():
            if all(term in query_lower for term in key.split()):
                return fact
        
        # Try partial matches
        best_match = None
        best_score = 0
        for key, fact in self.facts.items():
            key_terms = key.split()
            match_count = sum(1 for term in key_terms if term in query_lower)
            if match_count > 0:
                score = match_count / len(key_terms)
                if score > best_score:
                    best_score = score
                    best_match = fact
        
        if best_score > 0.5:
            return best_match
            
        return None
    
    def generate_text(self, prompt: str, max_length: int = 50) -> str:
        """Generate text based on the prompt (simplified demonstration)"""
        # In a real model, this would use the transformer architecture to predict tokens
        # Here we'll use a simplified approach for demonstration
        
        responses = [
            f"Based on {prompt}, we can explore language models as computational systems.",
            f"When we consider {prompt}, the transformer architecture shows its strengths.",
            f"Looking at {prompt}, we see how attention mechanisms help with context.",
            f"The concept of {prompt} relates to how neural networks process sequences.",
            f"In language modeling, {prompt} demonstrates pattern recognition capabilities."
        ]
        
        return random.choice(responses)
    
    def answer(self, query: str) -> str:
        """Main interface to generate answers"""
        # First check if it's a self-query
        if self.is_self_query(query):
            return self.answer_self_query(query)
        
        # Then check factual knowledge
        fact_answer = self.search_facts(query)
        if fact_answer:
            return fact_answer
        
        # Retrieve context and generate response
        context = self.retrieve(query)
        
        if context:
            # Combine context and generate response
            context_text = " ".join(context)
            return f"Based on my knowledge: {context_text}"
        else:
            # Fall back to simpler generation
            return self.generate_text(query)


# Simple console interface for testing when run directly
if __name__ == "__main__":
    print("NeuraFlux Lite Model (Type 'quit' to exit)")
    print("Using simplified implementation without external dependencies.")
    
    # Create the model and tokenizer
    config = NanoConfig()
    tokenizer = NanoTokenizer()
    model = NanoNLPEngine(config, tokenizer)
    
    while True:
        # Read input from stdin
        if not sys.stdin.isatty():
            # If reading from a pipe, just read one line
            try:
                query = sys.stdin.readline().strip()
                if not query:
                    break
            except:
                break
        else:
            query = input("\nYou: ").strip()
        
        if query.lower() in ["quit", "exit", "bye"]:
            break
        
        # Generate response
        response = model.answer(query)
        print(f"\nNeuraFlux: {response}")
        
        # If reading from a pipe, just process one message and exit
        if not sys.stdin.isatty():
            break 