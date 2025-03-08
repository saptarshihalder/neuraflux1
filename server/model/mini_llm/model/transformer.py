#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MiniTransformer model for the Mini LLM project.
This provides a Python interface for the C++ transformer model implementation.
"""

import os
import sys
import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

# For demonstration purposes when C++ components aren't compiled
class TransformerConfig:
    """Configuration for the MiniTransformer model."""
    
    def __init__(
        self,
        vocab_size: int = 30000,
        hidden_size: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        intermediate_size: int = 1536,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_glu: bool = True,
        use_rope: bool = True,
        window_size: int = 256,
        use_sliding_window: bool = True
    ):
        """
        Initialize transformer configuration.
        
        Args:
            vocab_size: Size of the vocabulary
            hidden_size: Size of the hidden representations
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            intermediate_size: Size of the feedforward network
            max_seq_length: Maximum sequence length
            dropout: Dropout probability
            activation: Activation function to use (gelu or relu)
            use_glu: Whether to use Gated Linear Unit in feed-forward
            use_rope: Whether to use Rotary Position Embeddings
            window_size: Size of the attention window for sliding window attention
            use_sliding_window: Whether to use sliding window attention
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        self.activation = activation
        self.use_glu = use_glu
        self.use_rope = use_rope
        self.window_size = window_size
        self.use_sliding_window = use_sliding_window
        
        # Derived parameters
        self.head_size = hidden_size // num_heads
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "max_seq_length": self.max_seq_length,
            "dropout": self.dropout,
            "activation": self.activation,
            "use_glu": self.use_glu,
            "use_rope": self.use_rope,
            "window_size": self.window_size,
            "use_sliding_window": self.use_sliding_window,
            "head_size": self.head_size
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TransformerConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def save(self, config_path: str) -> None:
        """Save configuration to file."""
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, config_path: str) -> 'TransformerConfig':
        """Load configuration from file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class MiniTransformer:
    """
    MiniTransformer model for language generation.
    This class provides a Python interface to the C++ transformer implementation.
    """
    
    def __init__(self, config: TransformerConfig):
        """
        Initialize the transformer model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        
        # Try to load C++ implementation
        try:
            from model.transformer_lib import TransformerLib
            self.impl = TransformerLib()
            self.impl.create_model(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                intermediate_size=config.intermediate_size,
                max_seq_len=config.max_seq_length,
                dropout=config.dropout,
                use_glu=config.use_glu,
                use_rope=config.use_rope,
                window_size=config.window_size,
                use_sliding_window=config.use_sliding_window
            )
            self._use_mock = False
            logging.info("Using C++ transformer implementation")
        except ImportError:
            # Use mock implementation
            self._use_mock = True
            logging.warning("C++ implementation not available, using mock implementation")
    
    def forward(self, input_ids: List[int]) -> np.ndarray:
        """
        Forward pass through the model.
        
        Args:
            input_ids: List of token IDs
            
        Returns:
            Output logits of shape [sequence_length, vocab_size]
        """
        if not self._use_mock:
            return self.impl.forward(input_ids)
        else:
            # Mock implementation
            seq_len = len(input_ids)
            return np.random.randn(seq_len, self.config.vocab_size)
    
    def get_hidden_states(self, layer_idx: int) -> np.ndarray:
        """
        Get hidden states from a specific layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Hidden states from the specified layer
        """
        if not self._use_mock:
            return self.impl.get_hidden_states(layer_idx)
        else:
            # Mock implementation
            return np.random.randn(100, self.config.hidden_size)
    
    def generate(
        self,
        prompt: List[int],
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> List[int]:
        """
        Generate text tokens from a prompt.
        
        Args:
            prompt: List of token IDs for the prompt
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling parameter
            
        Returns:
            List of token IDs for the generated text
        """
        if not self._use_mock:
            return self.impl.generate(prompt, max_length, temperature, top_p, top_k)
        else:
            # Mock implementation
            result = prompt.copy()
            for _ in range(max_length - len(prompt)):
                # Append a random token in vocabulary range
                next_token = np.random.randint(0, self.config.vocab_size)
                result.append(next_token)
            return result
    
    def save(self, model_path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            model_path: Path to save the model
        """
        os.makedirs(model_path, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(model_path, "config.json")
        self.config.save(config_path)
        
        if not self._use_mock:
            # Save model weights
            weights_path = os.path.join(model_path, "model.bin")
            self.impl.save(weights_path)
        else:
            # Mock implementation - just save a placeholder
            with open(os.path.join(model_path, "model.bin.mock"), 'w') as f:
                f.write("Mock model weights")
        
        logging.info(f"Saved model to {model_path}")
    
    @classmethod
    def load(cls, model_path: str) -> 'MiniTransformer':
        """
        Load a model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model
        """
        config_path = os.path.join(model_path, "config.json")
        config = TransformerConfig.load(config_path)
        
        model = cls(config)
        
        if not model._use_mock:
            # Load model weights
            weights_path = os.path.join(model_path, "model.bin")
            model.impl.load(weights_path)
        
        logging.info(f"Loaded model from {model_path}")
        return model
    
    def train(self, learning_rate: float = 5e-5) -> None:
        """
        Set the model to training mode.
        
        Args:
            learning_rate: Learning rate for optimization
        """
        if not self._use_mock:
            self.impl.set_learning_rate(learning_rate)
            self.impl.train_mode()
    
    def eval(self) -> None:
        """Set the model to evaluation mode."""
        if not self._use_mock:
            self.impl.eval_mode()
    
    def zero_grad(self) -> None:
        """Zero out gradients for optimizer."""
        if not self._use_mock:
            self.impl.zero_grad()
    
    def backward(self, grad_output: np.ndarray) -> None:
        """
        Backward pass through the model.
        
        Args:
            grad_output: Gradient tensor with respect to output
        """
        if not self._use_mock:
            self.impl.backward(grad_output)
    
    def step(self) -> None:
        """Perform one optimization step."""
        if not self._use_mock:
            self.impl.step() 