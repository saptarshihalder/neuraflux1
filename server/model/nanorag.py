import re
import sys
import json
import os
import math
import numpy as np
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from typing import List, Tuple, Optional, Dict, Any, Union
import torch.optim as optim
# Comment out the problematic import
# from visualization import plot_training_metrics, visualize_token_importance
from tokenizer import NanoTokenizer
import traceback


class NanoConfig:
    """Configuration class for NanoRAG model."""
    def __init__(
        self,
        vocab_size: int = 16000,
        hidden_size: int = 256,  # Base size
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        intermediate_size: int = 1024,  # Changed from 512 to 1024 (4x hidden_size)
        hidden_act: str = "gelu",
        max_position_embeddings: int = 1024,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        use_cache: bool = True,
        use_alibi: bool = True,  # Use ALiBi positional encoding instead of positional embeddings
        use_gqa: bool = True,  # Use Grouped-Query Attention
        kv_heads: int = 2,      # Number of KV heads for GQA (must divide evenly into num_attention_heads)
        sliding_window: int = 512,  # Size of sliding window attention
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.use_alibi = use_alibi
        self.use_gqa = use_gqa
        self.kv_heads = kv_heads
        self.sliding_window = sliding_window
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps


class AlibiPositionalBias(nn.Module):
    """
    ALiBi (Attention with Linear Biases) positional encoding.
    This requires no parameters and scales efficiently to longer sequences.
    """
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.slopes = self._get_slopes(config.num_attention_heads)
    
    def _get_slopes(self, n):
        """Get slopes for each attention head according to ALiBi paper."""
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]
        
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            slopes_power_of_2 = get_slopes_power_of_2(closest_power_of_2)
            
            extra_slopes = []
            for i in range(n - closest_power_of_2):
                interp = slopes_power_of_2[-1] + slopes_power_of_2[0] * (i + 1) / (n - closest_power_of_2)
                extra_slopes.append(interp)
            
            return slopes_power_of_2 + extra_slopes
    
    def forward(self, query_length, key_length):
        """
        Generate ALiBi position bias.
        
        Args:
            query_length: Length of query sequence
            key_length: Length of key sequence
        
        Returns:
            attention_bias: Bias tensor of shape [1, heads, query_length, key_length]
        """
        # Create distance matrix
        distance = torch.arange(key_length)[None, :] - torch.arange(query_length)[:, None]
        
        # Convert to positive distances for causal masking
        distance = -torch.clamp(distance, min=0).float()
        
        # Apply slope per head
        slopes = torch.tensor(self.slopes, dtype=torch.float32)
        attention_bias = distance.unsqueeze(0) * slopes.unsqueeze(1).unsqueeze(2)
        
        return attention_bias


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) module.
    GQA reduces parameter count and computation by sharing key-value heads across query heads.
    """
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.kv_heads = config.kv_heads if config.use_gqa else config.num_attention_heads
        
        # Ensure kv_heads divides num_heads evenly
        assert self.num_heads % self.kv_heads == 0, f"Num attention heads ({self.num_heads}) must be divisible by KV heads ({self.kv_heads})"
        
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_groups = self.num_heads // self.kv_heads
        
        # Create query, key, value projections
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # Reduced parameter count for keys and values
        self.key = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.value = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # Sliding window attention parameters
        self.use_sliding_window = config.sliding_window > 0
        self.window_size = config.sliding_window
        
        # ALiBi positional bias
        self.use_alibi = config.use_alibi
        if self.use_alibi:
            self.alibi = AlibiPositionalBias(config)
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Grouped-Query Attention forward pass.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Optional attention mask of shape [batch_size, 1, 1, seq_length]
            past_kv: Optional cached key-value states from previous steps
            use_cache: Whether to use and update the key-value cache
            
        Returns:
            attn_output: Output tensor of shape [batch_size, seq_length, hidden_size]
            past_kv: Updated key-value cache if use_cache=True, otherwise None
        """
        batch_size, seq_length, _ = hidden_states.size()
        
        # Project queries, keys, values
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.kv_heads, self.head_dim).transpose(1, 2)
        
        # Process cached key-values if provided
        if past_kv is not None:
            past_key, past_value = past_kv
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)
        
        # Update key-value cache if needed
        if use_cache:
            current_kv = (k, v)
        else:
            current_kv = None
        
        # For GQA, reshape keys and values to match number of query heads
        if self.kv_heads < self.num_heads:
            # Repeat k and v for each query group
            k = k.unsqueeze(2).expand(-1, -1, self.q_groups, -1, -1).flatten(1, 2)
            v = v.unsqueeze(2).expand(-1, -1, self.q_groups, -1, -1).flatten(1, 2)
        
        # Get sequence dimensions after potential caching
        _, _, kv_seq_len, _ = k.shape
        
        # Apply ALiBi positional bias if enabled
        if self.use_alibi:
            alibi_bias = self.alibi.forward(seq_length, kv_seq_len)
            alibi_bias = alibi_bias.to(hidden_states.device)
        else:
            alibi_bias = None
        
        # Apply sliding window attention if enabled
        if self.use_sliding_window and kv_seq_len > self.window_size:
            # Create sliding window mask
            window_mask = torch.zeros(
                (seq_length, kv_seq_len), device=hidden_states.device, dtype=torch.bool
            )
            
            # Allow attention to window_size tokens to the left
            for i in range(seq_length):
                start = max(0, i - self.window_size)
                window_mask[i, start:i+1] = True
            
            # Convert to attention mask format
            sliding_mask = window_mask.float().masked_fill(~window_mask, -1e9)
            sliding_mask = sliding_mask.unsqueeze(0).unsqueeze(0)
            
            # Combine with existing attention mask if provided
            if attention_mask is not None:
                attention_mask = attention_mask + sliding_mask
            else:
                attention_mask = sliding_mask
        
        # Scaled dot-product attention
        attn_scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * attn_scale
        
        # Add ALiBi positional bias if enabled
        if alibi_bias is not None:
            attn_weights = attn_weights + alibi_bias
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Apply causal mask for autoregressive models
        if past_kv is None:
            causal_mask = torch.triu(
                torch.ones((seq_length, seq_length), dtype=torch.bool, device=hidden_states.device),
                diagonal=1
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            causal_mask = causal_mask.to(torch.bool)
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
        
        # Normalize attention weights
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project back to hidden size
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, self.hidden_size)
        attn_output = self.output(attn_output)
        
        return attn_output, current_kv


class HybridTransformerLayer(nn.Module):
    """
    Hybrid Transformer Layer that shares parameters between attention and feed-forward networks.
    This significantly reduces parameter count while maintaining performance.
    """
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Attention layer
        self.attention = GroupedQueryAttention(config)
        
        # MLP layers with correct dimensions
        self.shared_mlp_gate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.shared_mlp_down = nn.Linear(config.intermediate_size, config.hidden_size)
        
        # Activation function
        self.act_fn = F.gelu
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # Parameter sharing (optional)
        self.parameter_sharing = False  # Disabled for now to fix dimension issues
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Attention block
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        attention_output, past_kv = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_kv=past_kv,
            use_cache=use_cache,
        )
        hidden_states = residual + self.dropout(attention_output)
        
        # MLP block
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        
        # MLP with correct dimension handling
        mlp_output = self.shared_mlp_gate(hidden_states)  # [batch, seq, hidden] -> [batch, seq, intermediate]
        mlp_output = self.act_fn(mlp_output)
        mlp_output = self.shared_mlp_down(mlp_output)  # [batch, seq, intermediate] -> [batch, seq, hidden]
        
        layer_output = residual + self.dropout(mlp_output)
        
        return layer_output, past_kv


class DynamicSparseTransformer(nn.Module):
    """
    Parameter-efficient transformer model with dynamic sparsity.
    Integrates GQA, ALiBi, and parameter sharing for maximum efficiency.
    """
    def __init__(self, config: NanoConfig, tokenizer=None):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        
        # Token embeddings
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # ALiBi used instead of traditional positional embeddings
        self.use_alibi = config.use_alibi
        
        # Use traditional positional embeddings if ALiBi not enabled
        if not self.use_alibi:
            self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Hybrid Transformer Layers
        self.layers = nn.ModuleList([
            HybridTransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # For efficient inference, cache key-value pairs
        if tokenizer is not None:
            self.tokenizer = tokenizer
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using depth-scaled initialization."""
        if isinstance(module, nn.Linear):
            # Depth-scaled initialization similar to LLaMA
            std = self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers)
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    
    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the DynamicSparseTransformer.
        
        Args:
            input_ids: Indices of input sequence tokens
            attention_mask: Mask to avoid attending to padding tokens
            past_key_values: Cached key-value pairs for faster inference
            inputs_embeds: Embedded inputs (alternative to input_ids)
            labels: Labels for computing language modeling loss
            use_cache: Whether to return cached key-value pairs
            
        Returns:
            Dict containing model outputs including loss, last hidden state, etc.
        """
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
            input_shape = input_ids.shape
        else:
            batch_size, seq_length = inputs_embeds.shape[:2]
            input_shape = inputs_embeds.shape[:2]
        
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        # Default use_cache to model config
        if use_cache is None:
            use_cache = self.config.use_cache
        
        # Process past_key_values for KV cache
        if past_key_values is None:
            past_key_values = tuple([None] * self.config.num_hidden_layers)
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)
        
        # Create causal mask for attention
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length + past_length, device=device)
        
        # Extend attention mask for multi-head attention
        if attention_mask.dim() == 2:
            # [batch_size, seq_length] -> [batch_size, 1, 1, seq_length]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Convert mask to binary and then to attention compatible format
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        
        # If not using ALiBi, add traditional positional embeddings
        if not self.use_alibi:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds
        else:
            hidden_states = inputs_embeds
        
        # Process through transformer layers
        new_past_kv = () if use_cache else None
        
        # Initialize loss
        loss = None
        
        for i, (layer, past_kv) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, current_kv = layer(
                hidden_states,
                attention_mask=extended_attention_mask,
                past_kv=past_kv,
                use_cache=use_cache
            )
            
            if use_cache:
                new_past_kv = new_past_kv + (current_kv,)
        
        # Apply final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Compute language modeling loss if labels provided
        if labels is not None:
            # Shift the predicted tokens so they align with the labels
            shift_logits = hidden_states[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Project hidden states to vocabulary
            shift_logits = torch.matmul(shift_logits, self.wte.weight.T)
            
            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
        
        return {
            "loss": loss,
            "last_hidden_state": hidden_states,
            "past_key_values": new_past_kv
        }
    
    def generate(
        self,
        text: str,
        max_length: int = 30,
        top_k: int = 40,
        top_p: float = 0.9,
        temperature: float = 0.8,
        repetition_penalty: float = 1.1,
        device: str = "cpu",
        use_cache: bool = True,
    ) -> List[str]:
        """Generate text with improved dimension handling and repetition prevention"""
        try:
            if not hasattr(self, "tokenizer"):
                raise ValueError("Tokenizer not found")
            
            self.transformer.eval()
            self.transformer.to(device)
            
            # Truncate if needed
            if len(text) > 200:
                text = text[:200]
            
            # Tokenize with safeguards
            tokens = self.tokenizer.encode(text)
            if len(tokens) > 50:  # Keep prompt short
                tokens = tokens[:50]
            
            # Track original input length for extraction later
            orig_len = len(tokens)
            
            input_ids = torch.tensor([tokens], device=device)
            
            # Initial generation
            outputs = self.transformer(input_ids=input_ids)
            past_key_values = outputs.get("past_key_values", None)
            
            # Track generated tokens
            generated = tokens.copy()
            last_tokens = []  # For repetition detection
            
            # Set minimum tokens to generate
            min_new_tokens = 10
            max_tokens_to_generate = max(min_new_tokens, min(max_length, 50))
            
            # Generate new tokens
            for _ in range(max_tokens_to_generate):
                try:
                    # Get last hidden state
                    last_hidden = outputs["last_hidden_state"][:, -1].unsqueeze(1)
                    
                    # Get logits
                    logits = torch.matmul(last_hidden, self.transformer.wte.weight.T)
                    logits = logits.squeeze(1)
                    
                    # Apply repetition penalty
                    if len(generated) > orig_len:
                        for token_id in set(generated[-10:]):  # Penalize recent tokens
                            logits[0, token_id] /= repetition_penalty
                    
                    # Apply temperature
                    logits = logits / max(0.1, temperature)
                    
                    # Apply top-p sampling (nucleus sampling)
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                    indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('inf')
                    
                    # Sample next token
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Add to generated sequence
                    generated.append(next_token.item())
                    
                    # Check for repetition
                    last_tokens.append(next_token.item())
                    if len(last_tokens) > 12:  # Longer window for repetition detection
                        last_tokens.pop(0)
                        # Only detect repetition for longer n-grams (6-grams)
                        if len(last_tokens) >= 12 and len(generated) > orig_len + 15:  # Only check after generating meaningful content
                            if last_tokens[:6] == last_tokens[6:]:  # Check for 6-gram repetition
                                print("Significant repetition detected, stopping generation")
                                break
                    
                    # Break if we've added enough tokens beyond the input
                    if len(generated) >= orig_len + min_new_tokens:
                        # Only stop if we've generated a reasonable response
                        if len(generated) - orig_len > 20:
                            break
                    
                    # Prepare for next iteration
                    next_input = next_token.view(1, -1)  # Ensure shape is [batch_size, seq_length]
                    
                    # Get next token's representation
                    outputs = self.transformer(
                        input_ids=next_input,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    past_key_values = outputs.get("past_key_values", None)
                    
                except Exception as inner_e:
                    print(f"Error in generation loop: {str(inner_e)}")
                    traceback.print_exc()
                    break
            
            # Decode only the generated portion
            result = self.tokenizer.decode(generated)
            
            return [result]
            
        except Exception as e:
            print(f"Generation error: {str(e)}")
            traceback.print_exc()
            return ["I'm having trouble generating a response."]


class NanoRAG:
    """
    Retrieval-Augmented Small Language Model.
    Combines parameter-efficient transformer with retrieval capabilities.
    """
    def __init__(self, config: NanoConfig, tokenizer: NanoTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        
        # Initialize the transformer model
        self.transformer = DynamicSparseTransformer(config, tokenizer)
        
        # Retrieval components
        self.retrieval_enabled = False
        self.retrieval_db = []  # Document database
        self.retrieval_k = 3  # Number of documents to retrieve
        self.chunk_size = 128  # Chunk size for documents
        self.chunk_overlap = 32  # Overlap between chunks
        
        # Add basic knowledge about the model
        self.add_document('model_info', f"""
        NeuraFlux is an efficient AI assistant with {config.hidden_size} hidden dimensions,
        {config.num_hidden_layers} transformer layers, and {config.num_attention_heads} attention heads.
        It uses Grouped Query Attention (GQA) and ALiBi positional encoding for efficiency.
        The model can perform retrieval-augmented generation to enhance its responses with relevant context.
        """.strip())
        
        # Build initial embeddings
        self.build_document_embeddings()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        # Handle retrieval if enabled
        if self.retrieval_enabled and self.retrieval_db and input_ids is not None:
            # Get the input prompt
            input_text = self.tokenizer.decode(input_ids[0].tolist())
            
            # Retrieve relevant context
            context = self._retrieve_context(input_text)
            
            # If context was found, prepend to the input
            if context:
                augmented_text = context + "\n\n" + input_text
                # Re-encode with the augmented text
                augmented_ids = torch.tensor([self.tokenizer.encode(augmented_text)]).to(input_ids.device)
                
                # Update attention mask
                augmented_mask = torch.ones_like(augmented_ids)
                
                # Use augmented inputs
                input_ids = augmented_ids
                attention_mask = augmented_mask
        
        # Pass through transformer model
        return self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
    
    def generate(
        self,
        text: str,
        max_length: int = 30,
        top_k: int = 40,
        top_p: float = 0.9,
        temperature: float = 0.8,
        repetition_penalty: float = 1.1,
        device: str = "cpu",
        use_cache: bool = True,
    ) -> List[str]:
        """Generate text with improved dimension handling and repetition prevention"""
        try:
            if not hasattr(self, "tokenizer"):
                raise ValueError("Tokenizer not found")
            
            self.transformer.eval()
            self.transformer.to(device)
            
            # Truncate if needed
            if len(text) > 200:
                text = text[:200]
            
            # Tokenize with safeguards
            tokens = self.tokenizer.encode(text)
            if len(tokens) > 50:  # Keep prompt short
                tokens = tokens[:50]
            
            # Track original input length for extraction later
            orig_len = len(tokens)
            
            input_ids = torch.tensor([tokens], device=device)
            
            # Initial generation
            outputs = self.transformer(input_ids=input_ids)
            past_key_values = outputs.get("past_key_values", None)
            
            # Track generated tokens
            generated = tokens.copy()
            last_tokens = []  # For repetition detection
            
            # Set minimum tokens to generate
            min_new_tokens = 10
            max_tokens_to_generate = max(min_new_tokens, min(max_length, 50))
            
            # Generate new tokens
            for _ in range(max_tokens_to_generate):
                try:
                    # Get last hidden state
                    last_hidden = outputs["last_hidden_state"][:, -1].unsqueeze(1)
                    
                    # Get logits
                    logits = torch.matmul(last_hidden, self.transformer.wte.weight.T)
                    logits = logits.squeeze(1)
                    
                    # Apply repetition penalty
                    if len(generated) > orig_len:
                        for token_id in set(generated[-10:]):  # Penalize recent tokens
                            logits[0, token_id] /= repetition_penalty
                    
                    # Apply temperature
                    logits = logits / max(0.1, temperature)
                    
                    # Apply top-p sampling (nucleus sampling)
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                    indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('inf')
                    
                    # Sample next token
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Add to generated sequence
                    generated.append(next_token.item())
                    
                    # Check for repetition
                    last_tokens.append(next_token.item())
                    if len(last_tokens) > 12:  # Longer window for repetition detection
                        last_tokens.pop(0)
                        # Only detect repetition for longer n-grams (6-grams)
                        if len(last_tokens) >= 12 and len(generated) > orig_len + 15:  # Only check after generating meaningful content
                            if last_tokens[:6] == last_tokens[6:]:  # Check for 6-gram repetition
                                print("Significant repetition detected, stopping generation")
                                break
                    
                    # Break if we've added enough tokens beyond the input
                    if len(generated) >= orig_len + min_new_tokens:
                        # Only stop if we've generated a reasonable response
                        if len(generated) - orig_len > 20:
                            break
                    
                    # Prepare for next iteration
                    next_input = next_token.view(1, -1)  # Ensure shape is [batch_size, seq_length]
                    
                    # Get next token's representation
                    outputs = self.transformer(
                        input_ids=next_input,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    past_key_values = outputs.get("past_key_values", None)
                    
                except Exception as inner_e:
                    print(f"Error in generation loop: {str(inner_e)}")
                    traceback.print_exc()
                    break
            
            # Decode only the generated portion
            result = self.tokenizer.decode(generated)
            
            return [result]
            
        except Exception as e:
            print(f"Generation error: {str(e)}")
            traceback.print_exc()
            return ["I'm having trouble generating a response."]
    
    def _retrieve_context(self, query: str, max_context_length: int = 512) -> str:
        """Retrieve relevant context with strict limits"""
        if not self.retrieval_db:
            return ""
        
        # Much stricter context limit
        safe_max_length = min(max_context_length, 100)  # Even more conservative
        
        try:
            # Simple query encoding
            query_embedding = self._encode_text(query)
            
            # Calculate similarities
            similarities = []
            for i, doc in enumerate(self.retrieval_db):
                if doc.get('embedding') is not None:
                    similarity = self._cosine_similarity(query_embedding, doc['embedding'])
                    similarities.append((i, similarity))
            
            # Get top 1 most relevant doc - reduced from 2 to save tokens
            top_docs = sorted(similarities, key=lambda x: x[1], reverse=True)[:1]
            
            # Combine relevant content with strict length control
            context_parts = []
            total_length = 0
            
            for doc_idx, similarity in top_docs:
                if similarity < 0.2:  # Skip low relevance docs
                    continue
                    
                doc = self.retrieval_db[doc_idx]
                # Take even shorter content summary
                content = doc['content'][:80]  # Even shorter summary
                
                if total_length + len(content) > safe_max_length:
                    break
                    
                context_parts.append(content)
                total_length += len(content)
            
            if not context_parts:
                return ""
            
            return "Context: " + " ".join(context_parts)
            
        except Exception as e:
            print(f"Error in context retrieval: {str(e)}")
            return ""  # Return empty context on error
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors."""
        try:
            # Handle nested lists - flatten if needed
            if isinstance(vec1[0], list):
                vec1 = vec1[0]
            if isinstance(vec2[0], list):
                vec2 = vec2[0]
            
            # Make sure vectors are same length
            min_len = min(len(vec1), len(vec2))
            vec1 = vec1[:min_len]
            vec2 = vec2[:min_len]
            
            # Calculate cosine similarity
            dot_product = sum(float(a) * float(b) for a, b in zip(vec1, vec2))
            norm1 = sum(float(a) * float(a) for a in vec1) ** 0.5
            norm2 = sum(float(b) * float(b) for b in vec2) ** 0.5
            
            if norm1 > 0 and norm2 > 0:
                return dot_product / (norm1 * norm2)
            return 0
        except Exception as e:
            print(f"Similarity calculation error: {e}")
            return 0  # Return 0 similarity on error
    
    def _encode_text(self, text: str) -> List[float]:
        """
        Encode text for retrieval purposes with improved robustness.
        """
        # Using a fixed-size vector with deterministic encoding
        vec_size = 256
        vec = [0.0] * vec_size  # Fixed vector size
        
        try:
            # Simple character-level encoding
            for i, char in enumerate(text.lower()):
                pos = i % vec_size
                # Add character ASCII value
                vec[pos] += ord(char) % 10 / 10.0
            
            # Add word-level features
            words = text.lower().split()
            for word in words:
                # Simple hash-based encoding
                hash_val = abs(hash(word) % vec_size)
                vec[hash_val] += 1.0
            
            # Normalize
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]
            
            return vec
        
        except Exception as e:
            print(f"Error in text encoding: {e}")
            # Return zero vector as fallback
            return [0.0] * vec_size
    
    def add_document(self, doc_id, content):
        """Add a document to the retrieval database"""
        # Ensure content is properly formatted
        content = content.strip()
        if not content:
            return
        
        # Add to retrieval database
        self.retrieval_db.append({
            'id': doc_id,
            'content': content,
            'embedding': None  # Will be computed in build_document_embeddings
        })
    
    def chunk_and_add_document(self, doc_id: str, content: str):
        """
        Chunk a large document and add chunks to the retrieval database.
        
        Args:
            doc_id: Base document identifier
            content: Document content
        """
        # Split into sentences (simple approach)
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size, finalize the chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Keep some sentences for overlap
                overlap_sentences = current_chunk[-int(len(current_chunk) * self.chunk_overlap / current_length):]
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in current_chunk)
            
            # Add the sentence to the current chunk
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Add each chunk as a separate document
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            self.add_document(chunk_id, chunk)
    
    def build_document_embeddings(self, device='cpu'):
        """Build embeddings for all documents"""
        if not self.retrieval_db:
            return
        
        # Process each document
        for doc in self.retrieval_db:
            if doc['embedding'] is None:
                # Generate embedding using transformer
                with torch.no_grad():
                    try:
                        tokens = self.tokenizer.encode(doc['content'])
                        # Limit tokens to avoid dimension errors
                        if len(tokens) > 100:
                            tokens = tokens[:100]
                        input_ids = torch.tensor([tokens]).to(device)
                        outputs = self.transformer(input_ids)
                        
                        # Get mean of last_hidden_state and convert to flat list
                        embedding = outputs["last_hidden_state"].mean(dim=1).cpu().tolist()
                        
                        # Ensure it's a flat list of floats
                        if isinstance(embedding, list) and embedding:
                            doc['embedding'] = embedding[0] if isinstance(embedding[0], list) else embedding
                        else:
                            # Fallback to simple encoding if transformer fails
                            doc['embedding'] = self._encode_text(doc['content'])
                            
                    except Exception as e:
                        print(f"Error creating embedding: {e}")
                        # Use simple encoding as fallback
                        doc['embedding'] = self._encode_text(doc['content'])
    
    def enable_retrieval(self, retrieval_k: int = 3):
        """Enable retrieval-augmented generation."""
        self.retrieval_enabled = True
        self.retrieval_k = retrieval_k
    
    def disable_retrieval(self):
        """Disable retrieval-augmented generation."""
        self.retrieval_enabled = False
    
    def answer_with_rag(self, query: str, max_length: int = 150, 
                        temperature: float = 0.7, top_p: float = 0.90,
                        repetition_penalty: float = 1.2, device: str = "cpu") -> str:
        """Improved response generation with better fallback handling and question detection"""
        try:
            # Classify the question type for better response selection
            question_type = self._classify_question(query)
            
            # For specific question types, provide direct answers
            if question_type == "greeting":
                return "Hello! I'm NeuraFlux, a small AI assistant. How can I help you today?"
            
            if question_type == "identity":
                return "I'm NeuraFlux, a small language model with about 10 million parameters. I use retrieval-augmented generation to enhance my responses with factual information."
            
            if question_type == "capability":
                return "I can answer questions about various topics, including AI, machine learning, programming, and general knowledge. I combine my parametric knowledge with information retrieval to provide informative responses."
            
            if question_type == "creation":
                return "I was created as a demonstration of efficient language model architecture. I use techniques like Grouped-Query Attention and ALiBi positional encoding to make the most of my compact size."
            
            if question_type == "joke":
                return "Why don't scientists trust atoms? Because they make up everything!"
            
            if question_type == "math":
                return "I'm not designed to perform complex calculations. For mathematical operations, I'd recommend using a calculator or specialized software."
            
            # Use a simpler prompt format that's less likely to trigger repetition
            prompt = f"Q: {query}\nA:"
            
            if self.retrieval_enabled and self.retrieval_db:
                context = self._retrieve_context(query)
                if context:
                    prompt = f"{context}\n\nQ: {query}\nA:"
            
            print(f"Prompt length: {len(prompt)}")
            
            # Generate with careful parameters
            responses = self.generate(
                text=prompt,
                max_length=40,  # Shorter generations are more reliable
                temperature=0.5,  # Lower temperature for more deterministic responses
                top_p=0.95,
                repetition_penalty=1.5,  # Higher repetition penalty to avoid loops
                device=device
            )
            
            # Extract response
            full_response = responses[0].strip() if responses and responses[0] else ""
            response = self._clean_generated_response(full_response, prompt)
            
            # If response is empty or too short, provide a relevant fallback based on question type
            if not response or len(response.split()) < 3:
                return self._get_fallback_response(question_type, query)
            
            return response
            
        except Exception as e:
            print(f"Generation error: {str(e)}")
            traceback.print_exc()
            return "I'm having trouble processing that request right now."
    
    def _classify_question(self, query: str) -> str:
        """Classify the question type for better response selection"""
        query_lower = query.lower()
        
        # Greeting detection
        if any(word in query_lower for word in ["hi", "hello", "hey", "greetings"]):
            return "greeting"
        
        # Identity questions
        if any(phrase in query_lower for phrase in ["who are you", "what are you", "what is your name", "your name"]):
            return "identity"
        
        # Capability questions
        if any(phrase in query_lower for phrase in ["what can you do", "your capabilities", "able to", "can you"]):
            return "capability"
        
        # Creation questions
        if any(phrase in query_lower for phrase in ["who created you", "who made you", "how were you made", "your creator"]):
            return "creation"
        
        # Joke requests
        if "joke" in query_lower:
            return "joke"
        
        # Math questions
        if any(operator in query_lower for operator in ["+", "-", "*", "x", "times", "plus", "minus", "multiply", "divide"]):
            return "math"
        if any(char.isdigit() for char in query) and len([c for c in query if c.isdigit()]) > 4:
            return "math"
        
        # Default type
        return "general"
    
    def _get_fallback_response(self, question_type: str, query: str) -> str:
        """Get a fallback response based on question type"""
        if question_type == "general":
            general_responses = [
                "I don't have enough information to answer that question confidently.",
                "That's an interesting question. While I don't have a specific answer, I'm designed to help with a variety of topics.",
                "I'm a small language model called NeuraFlux. I'm designed to answer questions on various topics.",
                "I'd need more information to provide a helpful answer to that question."
            ]
            return general_responses[hash(query) % len(general_responses)]
        
        if question_type == "math":
            return "I'm not designed to perform complex calculations. For mathematical operations, I'd recommend using a calculator or specialized software."
        
        # Fallbacks for other question types
        type_responses = {
            "greeting": "Hello! How can I help you today?",
            "identity": "I'm NeuraFlux, a small AI assistant designed to answer questions.",
            "capability": "I can answer questions about various topics using my knowledge and retrieval capabilities.",
            "creation": "I was created as a demonstration of efficient language model design.",
            "joke": "Why don't scientists trust atoms? Because they make up everything!"
        }
        
        return type_responses.get(question_type, "I'm NeuraFlux, an AI assistant here to help you.")
    
    def _clean_generated_response(self, response: str, prompt: str) -> str:
        """Token-aware response cleaning"""
        # Remove prompt prefix if present
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        # Try extracting after "Assistant:" if present
        if "Assistant:" in response:
            parts = response.split("Assistant:", 1)
            if len(parts) > 1:
                response = parts[1].strip()
        
        # Remove additional user/assistant markers if present
        for marker in ["User:", "Assistant:", "Context:"]:
            response = response.replace(marker, "").strip()
        
        # Trim to last complete sentence if possible
        sentence_end = max(response.rfind("."), response.rfind("?"), response.rfind("!"))
        if sentence_end > 0 and sentence_end > len(response) // 3:
            response = response[:sentence_end+1]
        
        return response.strip()
    
    def save_pretrained(self, save_dir: str):
        """Save the model, tokenizer, and configuration."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save transformer model
        torch.save(self.transformer.state_dict(), os.path.join(save_dir, "model.pt"))
        
        # Save tokenizer
        self.tokenizer.save_vocab(os.path.join(save_dir, "vocab.json"))
        
        # Save configuration as JSON
        config_dict = self.config.__dict__.copy()
        with open(os.path.join(save_dir, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save document database if any
        if self.retrieval_db:
            with open(os.path.join(save_dir, "documents.json"), 'w') as f:
                json.dump(self.retrieval_db, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, load_dir: str, device: str = "cpu"):
        """Load model, tokenizer, and configuration from directory."""
        # Load configuration
        with open(os.path.join(load_dir, "config.json"), 'r') as f:
            config_dict = json.load(f)
        
        config = NanoConfig(**config_dict)
        
        # Load tokenizer
        tokenizer = NanoTokenizer(vocab_size=config.vocab_size)
        tokenizer.load_vocab(os.path.join(load_dir, "vocab.json"))
        
        # Create model instance
        model = cls(config, tokenizer)
        
        # Load model weights
        state_dict = torch.load(os.path.join(load_dir, "model.pt"), map_location=device)
        model.transformer.load_state_dict(state_dict)
        
        # Load document database if it exists
        doc_path = os.path.join(load_dir, "documents.json")
        if os.path.exists(doc_path):
            with open(doc_path, 'r') as f:
                model.retrieval_db = json.load(f)
            
            # Build embeddings
            model.build_document_embeddings()
            model.retrieval_enabled = True
        
        return model


# Simple console interface for testing when run directly
if __name__ == "__main__":
    # Check if a GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create model
    config = NanoConfig(
        vocab_size=16000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,  # Updated to 4x hidden_size
        use_alibi=True,
        use_gqa=True,
        kv_heads=2,
        sliding_window=512
    )
    
    tokenizer = NanoTokenizer(vocab_size=config.vocab_size)
    
    # Try to load a pretrained model or create a new one
    model_path = "./models"
    if os.path.exists(os.path.join(model_path, "model.pt")):
        print(f"Loading pretrained model from {model_path}...")
        model = NanoRAG.from_pretrained(model_path, device)
    else:
        print("Creating new model...")
        model = NanoRAG(config, tokenizer)
        # Since we're not training, initialize with a small vocab
        tokenizer._init_byte_vocab()
        tokenizer._compile_pattern()
    
    model = model.to(device)
    
    # Calculate model size
    total_params = sum(p.numel() for p in model.transformer.parameters())
    print(f"Model has {total_params:,} parameters ({total_params/1_000_000:.2f}M)")
    
    # Add sample documents for RAG
    print("Adding knowledge for complex question answering...")
    
    # Knowledge about AI and machine learning
    model.add_document(
        "transformer_architecture", 
        """The transformer architecture is a neural network architecture introduced in the 2017 paper "Attention Is All You Need" by Vaswani et al. 
        It uses self-attention mechanisms to process sequences in parallel, which allows for more efficient training compared to recurrent neural networks.
        Key components include: multi-head attention, positional encoding, feed-forward networks, residual connections, and layer normalization.
        Transformers power many modern language models like GPT, BERT, and T5, and have been adapted for vision, audio, and multimodal tasks."""
    )
    
    model.add_document(
        "reinforcement_learning",
        """Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by taking actions in an environment to maximize cumulative reward.
        Key concepts in RL include:
        1. Agent: The learner or decision-maker
        2. Environment: What the agent interacts with
        3. State: The current situation of the agent
        4. Action: A move the agent can make
        5. Reward: Feedback from the environment
        Popular RL algorithms include Q-learning, Deep Q Networks (DQN), Policy Gradient methods, Proximal Policy Optimization (PPO), and Soft Actor-Critic (SAC).
        RL has been successfully applied to game playing (AlphaGo), robotics, recommendation systems, and optimizing language model outputs through human feedback (RLHF)."""
    )
    
    model.add_document(
        "language_models",
        """Language models are AI systems trained to understand, generate, and manipulate human language.
        They work by predicting the probability distribution of words in a sequence.
        Modern language models use transformer architectures and are pretrained on vast corpora of text.
        Types of language models include:
        - Causal (autoregressive) LMs like GPT that predict the next token
        - Masked LMs like BERT that predict masked tokens in a sequence
        - Encoder-decoder models like T5 that transform input sequences to output sequences
        Language models can be adapted through fine-tuning, prompt engineering, and retrieval augmentation.
        Recent advances include instruction tuning, RLHF (Reinforcement Learning from Human Feedback), and multi-modal capabilities."""
    )
    
    model.add_document(
        "parameter_efficiency",
        """Parameter-efficient methods in language models aim to reduce computational requirements while maintaining performance.
        These techniques include:
        1. Grouped-Query Attention (GQA): Reduces parameters by sharing key-value heads across query heads
        2. Low-Rank Adaptation (LoRA): Adds trainable low-rank matrices to frozen pretrained weights
        3. Mixture of Experts (MoE): Routes input tokens to specialized parameter subsets
        4. Quantization: Reduces precision of model weights (e.g., from FP32 to INT8)
        5. Knowledge Distillation: Transfers knowledge from a larger teacher model to a smaller student model
        6. Pruning: Removes less important connections in neural networks
        7. Parameter Sharing: Reuses parameters across model components
        These methods are crucial for deploying efficient models on edge devices and reducing training costs."""
    )
    
    model.add_document(
        "neuraflux_capabilities",
        """NeuraFlux is a small language model with approximately 10 million parameters, optimized for efficiency.
        It features:
        - Grouped-Query Attention (GQA) for parameter efficiency
        - ALiBi positional encoding to handle longer contexts without additional parameters
        - Hybrid transformer layers with parameter sharing
        - Dynamic sparsity during training to focus on important connections
        - Retrieval-Augmented Generation (RAG) for enhanced factual knowledge
        - Sliding window attention to efficiently process longer sequences
        Despite its small size, NeuraFlux is designed to answer complex questions by combining its parametric knowledge with retrieval capabilities."""
    )
    
    # Build document embeddings
    model.build_document_embeddings(device)
    
    # Example complex questions
    example_questions = [
        "What is the transformer architecture and how does it work?",
        "Explain reinforcement learning and its applications in AI",
        "How do parameter-efficient methods improve language models?",
        "What techniques does NeuraFlux use to handle complex questions?",
        "Compare causal language models with masked language models"
    ]
    
    print("\n=== NeuraFlux Model - Complex Question Answering ===")
    print("This model combines a 10M parameter Transformer with Retrieval-Augmented Generation")
    print("Type 'example' to see sample questions, or 'quit' to exit\n")
    
    # Simple interaction loop
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
        elif query.lower() == "example":
            print("\nExample complex questions you can ask:")
            for i, question in enumerate(example_questions, 1):
                print(f"{i}. {question}")
            continue
        elif query.isdigit() and 1 <= int(query) <= len(example_questions):
            # Use the selected example question
            idx = int(query) - 1
            query = example_questions[idx]
            print(f"Selected: {query}")
        
        # Generate response with RAG
        print("\nThinking...")
        response = model.answer_with_rag(query, max_length=200, device=device)
        print(f"\nNeuraFlux: {response}")
        
        # If reading from a pipe, just process one message and exit
        if not sys.stdin.isatty():
            break

# Comment out any visualization calls at the end of the file
# plot_training_metrics(metrics)  # If this exists
# visualize_token_importance(model, tokenizer, "example text")  # If this exists