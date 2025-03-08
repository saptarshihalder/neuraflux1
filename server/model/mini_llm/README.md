# MiniLLM: Small Language Model with Knowledge Transfer

MiniLLM is a lightweight language model designed to run efficiently on consumer hardware while maintaining strong capabilities through knowledge transfer from larger teacher models.

## Architecture

MiniLLM combines several state-of-the-art techniques to create an efficient yet powerful language model:

- **Transformer Architecture**: Core attention-based architecture similar to larger LLMs but with fewer layers and parameters
- **Rotary Position Embeddings (RoPE)**: Better relative position encoding compared to absolute embeddings
- **Sliding Window Attention**: Reduces computational complexity for processing longer sequences
- **Gated Linear Units (GLU)**: More parameter-efficient feed-forward networks
- **Knowledge Transfer**: Learns from multiple teacher models efficiently

### Key Components

1. **BPE Tokenizer**: Custom implementation of Byte Pair Encoding for efficient token representation
2. **MiniTransformer**: Transformer architecture with configurable parameters
3. **Knowledge Transfer System**: Framework for distilling knowledge from larger models
4. **Teacher Adapters**: Interfaces for various teacher models (LLaMA, Flux, etc.)

## Directory Structure

```
mini_llm/
├── model/             # Core model implementation
│   ├── transformer.py # Python interface for the transformer
│   └── transformer_lib.cpp  # C++ implementation (when available)
├── tokenizer/         # Tokenization components
│   ├── bpe_tokenizer.py     # BPE tokenizer implementation
│   ├── test_tokenizer.py    # Tests for the tokenizer
│   └── demo_bpe.py          # Demonstration of BPE algorithm
├── training/          # Training infrastructure
│   ├── knowledge_transfer.py        # Knowledge transfer system
│   └── mock_training_demo.py        # Mock implementation for demonstration
├── data/              # Sample training data
│   └── sample_training.txt          # Example training texts
├── train.py           # Main training script
└── README.md          # Documentation
```

## Setup

### Prerequisites

- Python 3.8+
- C++ compiler (if building C++ components)
- CUDA-compatible GPU (optional, for faster training)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mini_llm.git
   cd mini_llm
   ```

2. Install Python dependencies:
   ```bash
   pip install numpy tqdm
   ```

3. (Optional) Build C++ components:
   ```bash
   cd model
   mkdir build && cd build
   cmake ..
   make
   ```

## Usage

### Training a Model

The main training script `train.py` provides a simple interface for training models:

```bash
python train.py --data_path data/sample_training.txt --output_dir output --use_mock
```

Key parameters:

- `--data_path`: Path to training data file (.txt or .json)
- `--output_dir`: Directory to save model checkpoints and logs
- `--tokenizer_path`: Path to pre-trained tokenizer (optional)
- `--use_mock`: Use mock implementation for testing/development

### Model Configuration

Configure the model size and architecture:

```bash
python train.py --data_path data/sample_training.txt \
    --vocab_size 30000 \
    --hidden_size 384 \
    --num_layers 6 \
    --num_heads 6 \
    --intermediate_size 1536 \
    --max_seq_length 512 \
    --batch_size 8
```

### Training Configuration

Control the training process:

```bash
python train.py --data_path data/sample_training.txt \
    --num_epochs 3 \
    --learning_rate 5e-5 \
    --batch_size 8 \
    --seed 42
```

## Knowledge Transfer

MiniLLM uses a multi-teacher knowledge transfer approach with several loss components:

1. **KL Divergence Loss**: Aligns output distributions between teacher and student
2. **Hidden State Matching**: Aligns internal representations at intermediate layers
3. **Contrastive Loss**: Prevents direct copying and encourages independent learning

This approach allows the student model to benefit from multiple teacher models while maintaining its own unique characteristics.

## Tokenizer

The BPE tokenizer implementation provides efficient subword tokenization:

```python
from tokenizer.bpe_tokenizer import BPETokenizer

# Create and train tokenizer
tokenizer = BPETokenizer(vocab_size=30000)
tokenizer.train(training_texts)

# Save tokenizer
tokenizer.save("path/to/tokenizer")

# Load tokenizer
tokenizer = BPETokenizer.load("path/to/tokenizer")

# Tokenize text
token_ids = tokenizer.encode("Example text to tokenize")

# Decode token IDs back to text
decoded_text = tokenizer.decode(token_ids)
```

## Examples

### Training with Mock Implementation

For testing without C++ components:

```bash
python train.py --data_path data/sample_training.txt --use_mock --num_epochs 1
```

### BPE Tokenization Demo

Visualize how BPE tokenization works:

```bash
python tokenizer/demo_bpe.py
```

### Evaluating the Model

Evaluate a trained model:

```bash
# Coming soon...
```

## License

[MIT License](LICENSE)

## Citation

If you use MiniLLM in your research, please cite:

```
@software{minillm2023,
  author = {Your Name},
  title = {MiniLLM: Small Language Model with Knowledge Transfer},
  url = {https://github.com/yourusername/mini_llm},
  year = {2023},
}
``` 