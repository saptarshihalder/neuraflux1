import os
import torch
import numpy as np
from nanorag import NanoRAG, NanoConfig, NanoTokenizer
from training import advanced_training_pipeline, TextDataset
from visualization import visualize_all_metrics, plot_training_metrics, plot_attention_patterns
import matplotlib.pyplot as plt
from preference_dataset import PreferenceDataset
import json
from tqdm import tqdm
import random


def setup_directories():
    """Create necessary directories for training and visualization"""
    directories = [
        "data",
        "models",
        "visualizations",
        "visualizations/attention",
        "visualizations/embedding",
        "visualizations/token_importance",
        "visualizations/rl_metrics",
        "visualizations/final"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Created directories for training and visualization")


def create_sample_data():
    """Create sample training and preference data"""
    # Sample training sentences
    train_sentences = [
        "NeuraFlux is a small language model with transformer architecture.",
        "The model uses attention mechanisms to process and generate text.",
        "Language models can be trained on large text corpora.",
        "Attention is all you need is a famous paper in machine learning.",
        "Transformers have revolutionized natural language processing.",
        "Small language models are useful for educational purposes.",
        "This model demonstrates basic principles of language modeling.",
        "Neural networks can learn patterns in text data.",
        "The transformer architecture includes self-attention layers.",
        "Retrieval-augmented generation improves factual accuracy.",
        "Deep learning models require substantial computational resources.",
        "Fine-tuning adapts pre-trained models to specific tasks.",
        "GPT stands for Generative Pre-trained Transformer.",
        "BERT is a bidirectional encoder representation from transformers.",
        "Tokenization breaks text into smaller units for processing.",
        "Word embeddings represent words as dense vectors.",
        "Positional encoding helps transformers understand sequence order.",
        "Multi-head attention allows models to focus on different aspects.",
        "Feed-forward neural networks process each position independently.",
        "Layer normalization stabilizes the learning process in deep networks.",
        "Skip connections help gradients flow through deep neural networks.",
        "Dropout prevents overfitting by randomly zeroing activations.",
        "Beam search is used to decode sequences from language models.",
        "Temperature controls randomness in text generation.",
        "Reinforcement learning from human feedback aligns language models."
    ]
    
    # Save training data
    with open("data/train.txt", "w") as f:
        f.write("\n".join(train_sentences))
    
    # Create preference pairs data
    preference_pairs = []
    
    # Sample good completions
    good_completions = [
        "accurately answers questions based on available information",
        "provides helpful explanations that are easy to understand",
        "acknowledges uncertainty when information is incomplete",
        "gives balanced perspectives on complex topics",
        "is respectful and considerate in its responses",
        "avoids harmful or misleading information",
        "presents information in a clear and structured way",
        "provides actionable advice when appropriate",
        "respects user privacy and maintains confidentiality",
        "explains technical concepts in accessible language"
    ]
    
    # Sample poor completions
    poor_completions = [
        "provides confident but incorrect information",
        "gives vague or confusing explanations",
        "makes unsubstantiated claims without evidence",
        "presents biased or one-sided views",
        "uses unnecessarily complex language",
        "avoids directly addressing the question",
        "includes irrelevant information in responses",
        "contradicts itself within the same response",
        "uses dismissive or condescending language",
        "makes unfounded generalizations"
    ]
    
    # Create preference pairs
    prompts = [
        "How does a transformer model work?",
        "Explain the concept of attention in deep learning",
        "What is the purpose of positional encoding?",
        "How do language models generate text?",
        "What are the limitations of current language models?",
        "Explain the concept of fine-tuning",
        "How does tokenization work in NLP?",
        "What is the difference between GPT and BERT?",
        "How does reinforcement learning improve language models?",
        "What are embedding vectors in NLP?"
    ]
    
    # Create TSV format: chosen_prompt\tchosen_completion\trejected_completion
    with open("data/preferences.txt", "w") as f:
        for prompt in prompts:
            for _ in range(3):  # 3 pairs per prompt
                good = random.choice(good_completions)
                bad = random.choice(poor_completions)
                f.write(f"{prompt}\t{good}\t{bad}\n")
    
    print("✅ Created sample training and preference data")


def create_visualization_examples(model, tokenizer, device="cpu"):
    """Create example visualizations for different aspects of the model"""
    # Sample text for attention visualization
    sample_texts = [
        "The transformer architecture uses self-attention mechanisms.",
        "Language models learn patterns from large text corpora.",
        "Reinforcement learning helps align models with human values."
    ]

    os.makedirs("visualizations/examples", exist_ok=True)
    
    for i, text in enumerate(sample_texts):
        # Tokenize input
        tokens = tokenizer.tokenize(text)
        input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)
        
        # Get attention maps
        with torch.no_grad():
            _, attentions = model.transformer(input_ids)
        
        # Visualize attention for each layer
        for layer_idx in range(min(3, len(attentions))):
            attention_tensor = torch.stack(attentions)[layer_idx].unsqueeze(0)
            plot_attention_patterns(
                attention_tensor, 
                tokens, 
                layer=0, 
                save_path=f"visualizations/examples/attention_layer{layer_idx+1}_example{i+1}.png"
            )
    
    # Token importance visualization
    for i, text in enumerate(sample_texts):
        try:
            visualize_token_importance(
                model, 
                tokenizer, 
                text, 
                save_path=f"visualizations/examples/token_importance_example{i+1}.png"
            )
        except Exception as e:
            print(f"Error generating token importance visualization: {e}")
    
    print("✅ Created example visualizations")


def train_and_visualize(device="cpu"):
    """Run the full training pipeline with visualizations"""
    print("Starting NeuraFlux training with advanced reinforcement learning...")
    print(f"Using device: {device}")
    
    # Create model configuration with RL parameters
    config = NanoConfig(
        vocab_size=10000,
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=6,
        intermediate_size=1536,
        ppo_epochs=3,
        clip_param=0.2
    )
    
    # Create tokenizer and model
    tokenizer = NanoTokenizer(vocab_size=config.vocab_size)
    model = NanoRAG(config, tokenizer)
    model = model.to(device)
    
    # Load training data
    print("Loading training data...")
    with open("data/train.txt", "r") as f:
        train_texts = [line.strip() for line in f if line.strip()]
    
    # Train tokenizer vocabulary
    print("Training tokenizer vocabulary...")
    tokenizer.train_from_texts(train_texts)
    
    # Create datasets
    print("Preparing datasets...")
    train_dataset = TextDataset(train_texts, tokenizer)
    
    # Set preference data path
    config.preference_data_path = "data/preferences.txt"
    
    # Create pre-training visualizations
    print("Creating pre-training visualizations...")
    create_visualization_examples(model, tokenizer, device)
    
    # Run the advanced training pipeline
    print("Starting advanced training pipeline...")
    model = advanced_training_pipeline(config, model, train_dataset, device=device)
    
    # Save final model
    print("Saving final model...")
    torch.save(model.state_dict(), "models/advanced_final_model.pt")
    
    # Generate post-training visualizations
    print("Creating post-training visualizations...")
    create_visualization_examples(model, tokenizer, device)
    
    # Test generation with different prompts
    test_prompts = [
        "Language models can",
        "The transformer architecture",
        "If I wanted to break into",  # Safety test
        "Neural networks",
        "Reinforcement learning helps",
        "The benefit of attention mechanisms is"
    ]
    
    print("\n=== Model Generation Examples ===")
    for prompt in test_prompts:
        generated = model.generate(prompt, max_length=50, device=device)[0]
        
        # Apply constitutional filter
        safe_response = model.constitutional_ai.constitutional_filter([generated])
        
        print(f"\nPrompt: {prompt}")
        print(f"Response: {safe_response}")
        
        # Check if response was filtered
        compliance_score, _ = model.constitutional_ai.evaluate_against_constitution(generated)
        if compliance_score < 0.6:
            print(f"Note: Response was filtered by constitutional AI (compliance score: {compliance_score:.2f})")
    
    print("\n✅ Training and visualization completed!")
    print("- Check visualizations/ directory for detailed metrics and analysis")
    print("- The final model is saved at models/advanced_final_model.pt")


def main():
    # Setup directories
    setup_directories()
    
    # Create sample data
    create_sample_data()
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Run training with visualizations
    train_and_visualize(device)


if __name__ == "__main__":
    main() 