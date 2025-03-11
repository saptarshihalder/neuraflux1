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
- **NanoRewardModel**: Reward model for predicting human preferences
- **NanoPPO**: Proximal Policy Optimization (PPO) implementation for policy learning
- **PreferenceDataset**: Dataset class for loading human preference data

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
- **Implement Reinforcement Learning from Human Feedback (RLHF)**
- **Add Direct Preference Optimization (DPO) for alignment**
- **Incorporate Constitutional AI principles**
- **Implement Proximal Policy Optimization (PPO) for policy learning**
- **Add multi-task reinforcement learning capabilities**

## Advanced Learning Methodologies

### Reinforcement Learning

The model includes multiple reinforcement learning approaches:

- **RLHF Pipeline**: Complete pipeline for human feedback collection and reward modeling
- **NanoRewardModel**: A model to predict human preferences
- **NanoPPO**: Implementation of PPO algorithm for policy optimization
- **NanoDPO**: Direct Preference Optimization for more efficient alignment without separate reward models
- **NanoREINFORCE**: Classic policy gradient method for reinforcement learning
- **Preference Dataset**: Framework for collecting and utilizing human preference data

### Constitutional AI

The model includes Constitutional AI components for safety and alignment:

- **NanoConstitutionalAI**: Rule-based constraints to ensure model outputs adhere to ethical guidelines
- **Constitutional Filtering**: Evaluates generated content against safety rules
- **Safety Alignment Training**: Fine-tuning process to avoid harmful outputs

### Multi-Task Reinforcement Learning

The model supports optimizing for multiple objectives simultaneously:

- **NanoMultiTaskRL**: Balance different reward signals like helpfulness, harmlessness, and honesty
- **Task Weighting**: Configure importance of different objectives
- **Composite Rewards**: Combine multiple reward functions into a unified learning signal

## Training Pipeline

The advanced training pipeline consists of multiple stages:

1. **Supervised Fine-tuning**: Initial training on next-token prediction
2. **Reward Model Training**: Learning to predict human preferences
3. **PPO Training**: Core RLHF using Proximal Policy Optimization
4. **Direct Preference Optimization**: More efficient alignment from preference pairs
5. **Constitutional AI Alignment**: Safety-focused training with rule-based filtering
6. **Multi-task RL**: Optimizing for multiple objectives simultaneously

Each stage builds on the previous, creating a curriculum that gradually improves model capabilities and alignment with human values.

## License

MIT

## Author

Saptarshi Halder 

- Implements core transformer architecture from "Attention is All You Need"
- Has embedding layers (NanoEmbeddings)
- Uses self-attention mechanisms (NanoSelfAttention)
- Includes positional encoding
- Implements language modeling head for generation 

- Retrieval-Augmented Generation system (combines parametric knowledge with document retrieval)
- This is a cutting-edge LLM technique used in models like ChatGPT 

- Explicit training pipeline
- Model checkpoint handling
- Dataset management 

## Training

The model is trained using a combination of supervised fine-tuning and reinforcement learning with human feedback (RLHF).

### Supervised Fine-tuning

The model is first trained on a standard language modeling task using a cross-entropy loss. This allows the model to learn general language understanding and generation capabilities.

### Reinforcement Learning

After supervised fine-tuning, the model undergoes additional training using reinforcement learning techniques. This aligns the model's outputs with human preferences.

The reinforcement learning loop consists of two main components:

1. **Reward Modeling**: The NanoRewardModel is trained to predict human preferences given pairs of generated outputs. This allows the model to learn what humans consider good or bad responses.

2. **Policy Optimization**: The main NanoRAG model is optimized using Proximal Policy Optimization (PPO) to maximize the predicted rewards from the reward model. This updates the model's policy to generate outputs that are more likely to be preferred by humans.

The reinforcement learning loop alternates between reward modeling and policy optimization, gradually aligning the model's behavior with human preferences. 

## Training Visualization

The model includes comprehensive training visualization capabilities:

```python
from training import plot_training_metrics, plot_attention_patterns

# To plot training metrics
plot_training_metrics(metrics_dict)  # Metrics collected during training

# To visualize attention patterns
attention = model.get_attention_maps(input_text)  # Implement this in your model
plot_attention_patterns(attention, layer=0)
```

Supported visualizations:
- Learning rate schedule
- Training/validation loss curves
- Reward progression
- Gradient flow analysis
- Attention head patterns
- Layer-wise activation statistics

Example training metrics plot:
![Training Metrics](training_metrics.png) 