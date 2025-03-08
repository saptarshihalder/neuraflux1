# NeuraFlux Small Language Model Implementation

This directory contains the implementation of a small language model built from scratch. The project has two implementations:

1. **nanorag.py**: Full implementation using PyTorch with complete transformer architecture
2. **nanorag_lite.py**: Simplified implementation without external dependencies

## Architecture

The full implementation follows the transformer architecture as described in "Attention is All You Need" with:

- Multi-head self-attention mechanism
- Position-wise feed-forward neural network
- Learned positional embeddings
- Layer normalization and residual connections

## Core Components

### 1. NanoConfig

Configuration class for model hyperparameters:
- Vocabulary size (default: 10,000)
- Hidden size (default: 384)
- Number of layers (default: 6)
- Number of attention heads (default: 6)
- Intermediate size (default: 1536)
- Maximum sequence length (default: 512)

### 2. NanoTokenizer

Custom tokenizer implementation:
- Basic tokenization (splitting by spaces and punctuation)
- Special token handling (PAD, BOS, EOS, UNK)
- Method to train vocabulary from texts
- Encoding/decoding methods

### 3. NanoModel

Core transformer encoder implementation:
- Self-attention layers
- Feed-forward neural networks
- Residual connections
- Layer normalization

### 4. NanoLMHead

Language modeling head for next-token prediction:
- Projects hidden states to vocabulary logits
- Applies final transformation before token prediction

### 5. NanoRAG

Full model with retrieval capabilities:
- Combines transformer model with document retrieval
- Encodes documents for semantic search
- Enhances generation with retrieved context

## Training

The `training.py` file contains a complete training pipeline:
- Dataset creation and processing
- Training loop with optimizer
- Model evaluation
- Checkpointing

## Simplified Implementation (nanorag_lite.py)

The lite version is designed to demonstrate the concepts without requiring external dependencies:
- Pure Python implementation
- Rules-based approach
- Demonstrates core concepts like tokenization, self-attention (conceptually)
- Includes knowledge retrieval capabilities

## Usage

### Direct Interaction

```bash
# Full implementation (requires PyTorch)
python nanorag.py

# Lite implementation (no dependencies)
python nanorag_lite.py
```

### Training

```bash
python training.py
```

## Current Limitations

- Simple tokenization without BPE/WordPiece
- Limited vocabulary size
- Training data is minimal
- No quantization or optimization techniques

## Future Enhancements

- Implement more sophisticated tokenization (BPE)
- Add benchmarking and evaluation metrics
- Implement advanced training techniques
- Add model quantization
- Expand document retrieval capabilities 