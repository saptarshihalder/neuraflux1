# Teacher-Student Knowledge Transfer Training System

This directory contains the implementation of a knowledge transfer training system that enables a smaller "student" model to learn from one or more larger "teacher" models.

## Overview

The knowledge transfer approach allows our small language model to achieve better performance by learning from larger, more powerful models like LLaMA or Flux. Instead of simply copying the teacher's outputs, our method uses multiple techniques to ensure the student develops its own understanding while benefiting from the teacher's knowledge.

## Key Features

- **Multiple Loss Functions**:
  - **KL Divergence Loss**: Aligns the student's probability distribution with the teacher's
  - **Hidden State Matching**: Encourages intermediate representations to match between models
  - **Contrastive Loss**: Prevents direct copying by penalizing exact reproduction

- **Layer Mapping**:
  - Automatically maps teacher layers to student layers, handling models of different sizes
  - Supports custom layer mappings for fine-grained control

- **Multiple Teachers**:
  - Learn from multiple different teacher models simultaneously
  - Combine knowledge from different architectures and capabilities

- **Training Efficiency**:
  - Gradient checkpointing to reduce memory usage
  - Optimized C++ interface for performance
  - Masking techniques to focus learning on important parts

## Usage

### Installation

Before using the knowledge transfer system, make sure you have:

1. Compiled the `libtransformer.so` shared library from the C++ implementation
2. Compiled any teacher model adapters (e.g., `libllama_teacher.so`)
3. Installed Python dependencies: numpy, tqdm

### Basic Training

```bash
python knowledge_transfer.py \
  --student-lib path/to/libtransformer.so \
  --tokenizer-path path/to/tokenizer \
  --llama-model path/to/llama/model \
  --train-data path/to/training/data.txt \
  --output-dir path/to/output
```

### Advanced Options

```bash
python knowledge_transfer.py \
  --student-lib path/to/libtransformer.so \
  --tokenizer-path path/to/tokenizer \
  --llama-model path/to/llama/model \
  --train-data path/to/training/data.txt \
  --output-dir path/to/output \
  --batch-size 8 \
  --epochs 5 \
  --learning-rate 3e-5 \
  --kl-weight 1.0 \
  --hidden-weight 0.5 \
  --contrastive-weight 0.2 \
  --save-every 500 \
  --config model_config.json
```

### Model Configuration

You can specify a JSON configuration file for the student model:

```json
{
  "vocab_size": 30000,
  "hidden_size": 384,
  "num_layers": 6,
  "num_heads": 6,
  "intermediate_size": 1536,
  "max_seq_len": 512,
  "dropout": 0.1
}
```

### Custom Teacher Layer Mapping

To create a custom layer mapping, create a JSON file like:

```json
{
  "llama": {
    "0": 0,
    "8": 1,
    "16": 2,
    "24": 3,
    "32": 4
  },
  "flux": {
    "0": 0,
    "6": 2,
    "12": 4
  }
}
```

This maps specific teacher layers to student layers (e.g., LLaMA layer 8 maps to student layer 1).

## Technical Details

### Training Process

1. **Initialization**:
   - Load student model and tokenizer
   - Initialize teacher models
   - Set up layer mappings

2. **For each batch**:
   - Generate teacher responses for the input
   - Forward pass through student model
   - Calculate combined loss (KL + hidden state + contrastive)
   - Backward pass and optimization

3. **Masking Technique**:
   - Randomly mask 30% of teacher responses
   - Force student to fill in the gaps

4. **Rejection Sampling**:
   - Generate multiple teacher responses
   - Student learns to mimic the best one

## Extended Example

```python
from knowledge_transfer import KnowledgeTransfer, TeacherLLaMA

# Initialize knowledge transfer system
kt = KnowledgeTransfer("./libtransformer.so", "./tokenizer")

# Add teacher models
llama_teacher = TeacherLLaMA("./models/llama-7b")
kt.add_teacher(llama_teacher)

# Initialize student model
config = {
    "vocab_size": 30000,
    "hidden_size": 384,
    "num_layers": 6,
    "num_heads": 6,
    "intermediate_size": 1536
}
kt.initialize_student_model(config)

# Load training data
with open("training_data.txt", "r") as f:
    train_data = [line.strip() for line in f if line.strip()]

# Train tokenizer if needed
kt.train_tokenizer(train_data)

# Train the model
kt.train(
    train_data=train_data,
    output_dir="./output",
    batch_size=8,
    epochs=3,
    learning_rate=5e-5
)
```

## Adding Custom Teacher Models

To add support for a new teacher model:

1. Create a new class that inherits from `TeacherModel`
2. Implement all required methods: generate, get_logits, get_hidden_states, etc.
3. Create a C++ adapter if using a compiled model

Example:

```python
class CustomTeacher(TeacherModel):
    def __init__(self, model_path):
        super().__init__("custom")
        # Initialize your model
        
    def generate(self, prompt, max_length=100, temperature=0.7):
        # Generate text with your model
        
    def get_logits(self, text):
        # Get logits for the text
        
    # Implement other required methods
```