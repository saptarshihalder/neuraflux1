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


class NanoConfig:
    def __init__(
        self,
        vocab_size: int = 10000,
        hidden_size: int = 384,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 6,
        intermediate_size: int = 1536,
        hidden_dropout_prob: int = 0.1,
        attention_probs_dropout_prob: int = 0.1,
        max_position_embeddings: int = 512,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id


# Basic components of the transformer architecture
class NanoEmbeddings(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = word_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class NanoSelfAttention(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) is not a multiple of num_attention_heads ({config.num_attention_heads})"
            )
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = x.size(0), x.size(1)
        new_shape = (batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_size]
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length = hidden_states.size(0), hidden_states.size(1)
        
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))
        
        # Calculate attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Normalize the attention scores
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context_layer = torch.matmul(attention_probs, value)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = (batch_size, seq_length, self.all_head_size)
        context_layer = context_layer.view(*new_shape)
        
        output = self.output(context_layer)
        
        return output, attention_probs


class NanoIntermediate(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.gelu = nn.GELU()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class NanoOutput(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class NanoLayer(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.attention = NanoSelfAttention(config)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.intermediate = NanoIntermediate(config)
        self.output = NanoOutput(config)
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)
        
        # Residual connection and layer norm
        attention_output = self.attention_norm(attention_output + hidden_states)
        
        # Feed-forward network
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        
        return layer_output, attention_probs


class NanoModel(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.config = config
        self.embeddings = NanoEmbeddings(config)
        self.layers = nn.ModuleList([NanoLayer(config) for _ in range(config.num_hidden_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Convert 2D mask to 4D for attention calculations
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Get embeddings
        embedding_output = self.embeddings(input_ids, position_ids)
        
        hidden_states = embedding_output
        all_attentions = []
        
        # Forward through layers
        for layer in self.layers:
            hidden_states, attention_probs = layer(hidden_states, extended_attention_mask)
            all_attentions.append(attention_probs)
        
        # Apply final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        return hidden_states, all_attentions


class NanoLMHead(nn.Module):
    def __init__(self, config: NanoConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        
        logits = self.decoder(hidden_states) + self.bias
        
        return logits


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
        # In a real implementation, use something more sophisticated like BPE or WordPiece
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


class NanoRAG(nn.Module):
    def __init__(
        self, 
        config: Optional[NanoConfig] = None,
        tokenizer: Optional[NanoTokenizer] = None
    ):
        super().__init__()
        self.config = config if config is not None else NanoConfig()
        self.tokenizer = tokenizer if tokenizer is not None else NanoTokenizer(vocab_size=self.config.vocab_size)
        
        # Core model
        self.transformer = NanoModel(self.config)
        self.lm_head = NanoLMHead(self.config)
        
        # RAG component - documents for retrieval
        self.documents = []
        self.document_embeddings = None
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_input_embeddings(self):
        return self.transformer.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        self.transformer.embeddings.word_embeddings = value
    
    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]:
        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        hidden_states = transformer_outputs[0]
        
        # Generate language model logits
        lm_logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        if return_dict:
            return {
                "loss": loss,
                "logits": lm_logits,
                "hidden_states": hidden_states,
            }
        else:
            outputs = (lm_logits,) + transformer_outputs[1:]
            if loss is not None:
                outputs = (loss,) + outputs
            return outputs
    
    def prepare_inputs_for_generation(
        self, input_ids: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        # Create attention mask
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    
    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        device: str = "cpu"
    ) -> List[str]:
        self.eval()
        
        # Tokenize the prompt
        input_ids = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
        batch_size = input_ids.shape[0]
        
        # Set generated so far
        generated = input_ids
        
        # Set the model in evaluation mode
        with torch.no_grad():
            for _ in range(max_length):
                # Get logits for next token
                outputs = self(input_ids=generated)
                next_token_logits = outputs["logits"][:, -1, :]  # Get logits for the last token
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for previous_token in generated[i]:
                            if previous_token in next_token_logits[i]:
                                next_token_logits[i, previous_token] /= repetition_penalty
                
                # Apply top-k and top-p filtering
                if do_sample:
                    # Top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = -float("Inf")
                    
                    # Top-p filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Shift the indices to the right to keep also the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        for i in range(batch_size):
                            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                            next_token_logits[i, indices_to_remove] = -float("Inf")
                    
                    # Sample from the filtered distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding - take the token with highest probability
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Add the new token to the sequence
                generated = torch.cat((generated, next_token), dim=-1)
                
                # Stop if we generate the EOS token
                if generated[0, -1].item() == self.config.eos_token_id:
                    break

        # Convert back to text
        generated_sequences = []
        for seq in generated:
            generated_text = self.tokenizer.decode(seq.tolist())
            generated_sequences.append(generated_text)
        
        return generated_sequences
    
    def add_document(self, doc_id: str, content: str):
        """Add a document to the RAG memory"""
        self.documents.append({"id": doc_id, "content": content})
    
    def build_document_embeddings(self, device: str = "cpu"):
        """Create embeddings for RAG documents"""
        self.eval()
        with torch.no_grad():
            embeddings = []
            for doc in self.documents:
                # Get document input IDs
                input_ids = torch.tensor(
                    self.tokenizer.encode(doc["content"], add_special_tokens=True),
                    dtype=torch.long
                ).unsqueeze(0).to(device)
                
                # Get document embedding from the last layer's [CLS] token
                outputs = self.transformer(input_ids=input_ids)
                embedding = outputs[0][:, 0, :].squeeze(0)  # Take the first token embedding
                embeddings.append(embedding)
            
            if embeddings:
                self.document_embeddings = torch.stack(embeddings)
    
    def retrieve(self, query: str, top_k: int = 3, device: str = "cpu") -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query"""
        self.eval()
        
        # Build document embeddings if not already built
        if self.document_embeddings is None and self.documents:
            self.build_document_embeddings(device)
        
        # If no documents, return empty list
        if not self.documents:
            return []
        
        with torch.no_grad():
            # Get query embedding
            input_ids = torch.tensor(
                self.tokenizer.encode(query, add_special_tokens=True),
                dtype=torch.long
            ).unsqueeze(0).to(device)
            
            outputs = self.transformer(input_ids=input_ids)
            query_embedding = outputs[0][:, 0, :].squeeze(0)
            
            # Calculate similarity
            similarities = F.cosine_similarity(query_embedding.unsqueeze(0), self.document_embeddings)
            
            # Get top-k results
            top_k_values, top_k_indices = torch.topk(similarities, min(top_k, len(self.documents)))
            
            # Format results
            results = []
            for i, idx in enumerate(top_k_indices.tolist()):
                results.append({
                    "id": self.documents[idx]["id"],
                    "content": self.documents[idx]["content"],
                    "score": float(top_k_values[i])
                })
            
            return results
    
    def answer_with_rag(
        self, 
        query: str, 
        max_length: int = 50,
        device: str = "cpu"
    ) -> str:
        """Generate an answer using RAG"""
        # Retrieve documents
        docs = self.retrieve(query, top_k=3, device=device)
        
        if not docs:
            # No documents found, just use the model to generate
            return self.generate(query, max_length=max_length, device=device)[0]
        
        # Create a prompt with retrieved documents
        context = "\n".join([doc["content"] for doc in docs])
        rag_prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        
        # Generate with the context
        return self.generate(rag_prompt, max_length=max_length, device=device)[0]


# Simple console interface for testing when run directly
if __name__ == "__main__":
    # Check if a GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create model
    config = NanoConfig()
    tokenizer = NanoTokenizer()
    model = NanoRAG(config, tokenizer)
    model = model.to(device)
    
    # Add sample documents for RAG
    model.add_document(
        "model_info", 
        "NeuraFlux is a small language model with 1.45M parameters. It was created by Saptarshi Halder."
    )
    model.add_document(
        "model_architecture",
        "NeuraFlux uses a hybrid transformer architecture with 6 layers and 6 attention heads."
    )
    model.add_document(
        "model_capabilities",
        "NeuraFlux can answer questions, generate text, and retrieve information from its memory."
    )
    
    # Build document embeddings
    model.build_document_embeddings(device)
    
    # Simple interaction loop
    print("NeuraFlux Model (Type 'quit' to exit)")
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
        
        # Generate response with RAG
        response = model.answer_with_rag(query, device=device)
        print(f"\nNeuraFlux: {response}")
        
        # If reading from a pipe, just process one message and exit
        if not sys.stdin.isatty():
            break