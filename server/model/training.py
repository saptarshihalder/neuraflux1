import os
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
import numpy as np
from tqdm import tqdm
from nanorag import NanoRAG, NanoConfig, NanoTokenizer
import json
from typing import List, Dict, Any, Optional, Tuple

class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: NanoTokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []
        
        # Encode all texts
        for text in tqdm(texts, desc="Encoding texts"):
            # Tokenize text
            encoding = self.tokenizer.encode(text)
            # Truncate if too long
            if len(encoding) > self.max_length:
                encoding = encoding[:self.max_length]
            # Pad if too short
            elif len(encoding) < self.max_length:
                encoding = encoding + [self.tokenizer.vocab[self.tokenizer.pad_token]] * (self.max_length - len(encoding))
            
            self.encodings.append(encoding)
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        item = self.encodings[idx]
        # Create input_ids and labels for language modeling
        input_ids = torch.tensor(item[:-1], dtype=torch.long)
        labels = torch.tensor(item[1:], dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.vocab[self.tokenizer.pad_token]).long()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def load_sample_data(file_path: str) -> List[str]:
    """Load sample text data from a file or return default samples if file doesn't exist"""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        # Default sample texts
        return [
            "NeuraFlux is a small language model with transformer architecture.",
            "The model uses attention mechanisms to process and generate text.",
            "Language models can be trained on large text corpora.",
            "Attention is all you need is a famous paper in machine learning.",
            "Transformers have revolutionized natural language processing.",
            "Small language models are useful for educational purposes.",
            "This model demonstrates basic principles of language modeling.",
            "Neural networks can learn patterns in text data.",
            "The transformer architecture includes self-attention layers.",
            "Retrieval-augmented generation improves factual accuracy."
        ]

def train(
    model: NanoRAG,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    device: str = "cpu",
    output_dir: str = "./models"
):
    """Train the model on the given dataset"""
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size) if eval_dataset else None
    
    # Set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs["loss"]
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
        
        # Evaluate if eval dataset is provided
        if eval_loader:
            eval_loss = evaluate(model, eval_loader, device)
            print(f"Evaluation loss: {eval_loss:.4f}")
        
        # Save model checkpoint
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch+1}.pt"))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, "final_model.pt"))
    # Save tokenizer vocabulary
    model.tokenizer.save_vocab(os.path.join(output_dir, "vocab.json"))
    
    print(f"Training completed. Model saved to {output_dir}")

def evaluate(model: NanoRAG, eval_loader: DataLoader, device: str) -> float:
    """Evaluate the model on the evaluation dataset"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs["loss"].item()
    
    avg_loss = total_loss / len(eval_loader)
    model.train()
    return avg_loss

def main():
    # Check if a GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create model configuration
    config = NanoConfig(
        vocab_size=10000,
        hidden_size=384,
        num_hidden_layers=6,
        num_attention_heads=6,
        intermediate_size=1536
    )
    
    # Create tokenizer and model
    tokenizer = NanoTokenizer(vocab_size=config.vocab_size)
    model = NanoRAG(config, tokenizer)
    model = model.to(device)
    
    # Load sample training data
    train_texts = load_sample_data("data/train.txt")
    
    # Train tokenizer vocabulary
    print("Training tokenizer vocabulary...")
    tokenizer.train_from_texts(train_texts)
    
    # Create dataset
    print("Creating dataset...")
    train_dataset = TextDataset(train_texts, tokenizer)
    
    # Train model
    print("Starting model training...")
    train(
        model=model,
        train_dataset=train_dataset,
        num_epochs=3,
        batch_size=4,
        learning_rate=5e-5,
        device=device
    )
    
    # Test generation
    print("\nTesting model generation:")
    test_prompts = [
        "Language models can",
        "The transformer architecture",
        "Neural networks"
    ]
    
    for prompt in test_prompts:
        generated = model.generate(prompt, max_length=30, device=device)[0]
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}\n")

if __name__ == "__main__":
    main()
