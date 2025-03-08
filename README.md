# NeuraFlux Small Language Model

A fully-implemented small language model built from scratch. This project demonstrates the fundamental concepts of modern transformer-based language models by implementing a complete, albeit small, transformer architecture with RAG (Retrieval Augmented Generation) capabilities.

## Features

- **Complete Transformer Architecture**: Implemented from scratch with PyTorch
- **Small But Functional**: ~1.45M parameters (6 layers, 6 attention heads, 384 hidden size)
- **Retrieval-Augmented Generation**: Combines parametric knowledge with retrieval from a document store
- **Custom Tokenizer**: Simple but functional tokenization with special token handling
- **Training Pipeline**: Complete training loop with dataset handling
- **Interactive Interface**: WebSocket API for real-time interaction
- **Educational Purpose**: Code designed for clarity and learning

## Components

- **NanoConfig**: Configuration class for model hyperparameters
- **NanoEmbeddings**: Word and position embedding layers
- **NanoSelfAttention**: Multi-head self-attention mechanism
- **NanoLayer**: Complete transformer layer with attention and feed-forward network
- **NanoModel**: Core transformer model with multiple layers
- **NanoLMHead**: Language modeling head for next-token prediction
- **NanoTokenizer**: Custom tokenizer implementation
- **NanoRAG**: Full model with transformer and retrieval components

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neuraflux.git
cd neuraflux

# Install dependencies
pip install -e .
```

## Usage

### Start the Server

```bash
npm run dev
```

### Training the Model

```bash
cd server/model
python training.py
```

### Interact with the Model Directly

```bash
cd server/model
python nanorag.py
```

## Architecture

The model follows the transformer architecture described in "Attention is All You Need" with:

- Input embeddings + positional embeddings
- Multi-head self-attention mechanism
- Feed-forward neural network with GELU activation
- Layer normalization and residual connections
- RAG component for document retrieval and context-enhanced generation

## Project Structure

```
neuraflux/
├── client/           # Frontend code
├── server/
│   ├── model/        # Core language model implementation
│   │   ├── nanorag.py    # Main model implementation
│   │   ├── training.py   # Training pipeline
│   │   ├── data/         # Training data
│   │   └── models/       # Saved model checkpoints
│   ├── routes.ts     # API routes
│   └── index.ts      # Server initialization
├── shared/           # Shared types/schemas
└── api/              # API endpoints
```

## Limitations

- Vocabulary size is limited to 10,000 tokens
- Maximum sequence length is 512 tokens
- Simple tokenization without BPE or WordPiece
- Training data is minimal

## Future Improvements

- Implement more sophisticated tokenization (BPE)
- Add more advanced training techniques (mixed precision, gradient accumulation)
- Improve the document retrieval algorithm
- Add finetuning capabilities for specific tasks
- Implement model quantization for better efficiency

## License

MIT

## Author

Saptarshi Halder 