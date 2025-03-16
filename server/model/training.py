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
from datasets import load_dataset
from transformers import AdamW
from preference_dataset import PreferenceDataset
import matplotlib.pyplot as plt
from visualization import plot_training_metrics, plot_rl_specific_metrics, visualize_all_metrics, plot_attention_patterns
import gc
import random
import math

# Import optional bitsandbytes for 8-bit optimization if available
try:
    import bitsandbytes as bnb
    has_bnb = True
except ImportError:
    has_bnb = False
    print("bitsandbytes not available, falling back to standard optimizers")

# Try to import flashattention if available
try:
    import flash_attn
    has_flash_attn = True
except ImportError:
    has_flash_attn = False
    print("flash_attn not available, using standard attention")

class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: NanoTokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            self._process_batch(batch)
    
    def _process_batch(self, texts):
        # Dynamic batching - group similar length texts
        text_lengths = [(i, len(text)) for i, text in enumerate(texts)]
        text_lengths.sort(key=lambda x: x[1])
        
        # Group into length-based buckets
        num_buckets = min(10, len(texts) // 10 + 1)
        bucket_size = len(texts) // num_buckets + 1
        
        # Process each bucket
        for bucket_idx in range(num_buckets):
            start_idx = bucket_idx * bucket_size
            end_idx = min((bucket_idx + 1) * bucket_size, len(text_lengths))
            
            if start_idx >= len(text_lengths):
                break
            
            # Get texts for this bucket
            bucket_indices = [text_lengths[i][0] for i in range(start_idx, end_idx)]
            bucket_texts = [texts[i] for i in bucket_indices]
            
            # Calculate optimal length for this bucket
            max_len = min(self.max_length, max(len(texts[i]) for i in bucket_indices) + 10
            
            # Tokenize this bucket
            for text in tqdm(bucket_texts, desc=f"Encoding bucket {bucket_idx+1}/{num_buckets}"):
                # Encode and handle length
                encoding = self.tokenizer.encode(text)
                
                # Truncate if too long
                if len(encoding) > self.max_length:
                    encoding = encoding[:self.max_length]
                # Pad if too short
                elif len(encoding) < self.max_length:
                    pad_token_id = self.tokenizer.vocab.get(self.tokenizer.pad_token, 0)
                    encoding = encoding + [pad_token_id] * (self.max_length - len(encoding))
                
                self.encodings.append(encoding)
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        item = self.encodings[idx]
        # Create input_ids and labels for language modeling
        input_ids = torch.tensor(item[:-1], dtype=torch.long)
        labels = torch.tensor(item[1:], dtype=torch.long)
        
        # Create attention mask
        pad_token_id = self.tokenizer.vocab.get(self.tokenizer.pad_token, 0)
        attention_mask = (input_ids != pad_token_id).long()
        
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
            "Retrieval-augmented generation improves factual accuracy.",
            "Efficient language models can run on consumer hardware.",
            "Parameter sharing helps reduce model size significantly.",
            "Knowledge distillation transfers teacher knowledge to student models.",
            "Sparsity in neural networks can improve computational efficiency.",
            "Quantization reduces precision but preserves model capabilities.",
            "Grouped-query attention is more efficient than standard attention.",
            "Sliding window attention patterns limit computational complexity.",
            "Positional encodings help transformers understand token order.",
            "Byte-pair encoding creates efficient tokenization for languages.",
            "Gradient checkpointing trades computation for memory savings."
        ]

def train(
    model: NanoRAG,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    num_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-4,
    device: str = "cpu",
    output_dir: str = "./models",
    use_8bit: bool = True,
    teacher_model: Optional[NanoRAG] = None,
    gradient_accumulation_steps: int = 1,
    warmup_steps: int = 100,
    weight_decay: float = 0.01,
):
    """Train the model with optimized techniques for parameter efficiency."""
    # Initialize metrics tracking
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'grad_norm': [],
        'rewards': []
    }
    
    # Create data loaders with dynamic batching
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=os.cpu_count() or 2
    )
    
    if eval_dataset:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size * 2,  # Larger eval batches
            num_workers=os.cpu_count() or 2
        )
    else:
        eval_loader = None
    
    # Setup optimizer with 8-bit quantization if available
    if has_bnb and use_8bit:
        print("Using 8-bit AdamW optimizer")
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        print("Using standard AdamW optimizer")
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    # Setup learning rate scheduler with warmup
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Setup gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if device != "cpu" else None
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()
    
    # Setup distillation if teacher model is provided
    distillation_enabled = teacher_model is not None
    if distillation_enabled:
        teacher_model.eval()
        print("Knowledge distillation enabled")
        # Weight for teacher loss vs standard loss
        distill_alpha = 0.5  
    
    # Training loop
    model.train()
    global_step = 0
    accumulated_loss = 0.0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        
        # Dynamic sparsity parameters - gradually increase sparsity
        sparsity = min(0.3, 0.1 + 0.1 * epoch)  # Start at 10%, increase to 30%
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            for k, v in batch.items():
                batch[k] = v.to(device)
            
            # Mixed precision training
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    # Forward pass
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    
                    loss = outputs["loss"]
                    
                    # Apply knowledge distillation if enabled
                    if distillation_enabled:
                        with torch.no_grad():
                            teacher_outputs = teacher_model(
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"]
                            )
                        
                        # Extract logits
                        student_logits = torch.matmul(
                            outputs["last_hidden_state"], 
                            model.transformer.wte.weight.transpose(0, 1)
                        )
                        teacher_logits = torch.matmul(
                            teacher_outputs["last_hidden_state"], 
                            teacher_model.transformer.wte.weight.transpose(0, 1)
                        )
                        
                        # KL divergence loss
                        temperature = 2.0
                        soft_targets = nn.functional.softmax(teacher_logits / temperature, dim=-1)
                        log_probs = nn.functional.log_softmax(student_logits / temperature, dim=-1)
                        distill_loss = -torch.sum(soft_targets * log_probs, dim=-1).mean() * (temperature ** 2)
                        
                        # Combine losses
                        loss = distill_alpha * distill_loss + (1 - distill_alpha) * loss
                    
                    # Normalize loss for gradient accumulation
                    loss = loss / gradient_accumulation_steps
                
                # Backward pass with gradient scaler
                scaler.scale(loss).backward()
            else:
                # Forward pass without mixed precision
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                loss = outputs["loss"]
                
                # Apply knowledge distillation if enabled
                if distillation_enabled:
                    with torch.no_grad():
                        teacher_outputs = teacher_model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"]
                        )
                    
                    # Extract logits (might need adjustments based on model implementation)
                    student_logits = torch.matmul(
                        outputs["last_hidden_state"], 
                        model.transformer.wte.weight.transpose(0, 1)
                    )
                    teacher_logits = torch.matmul(
                        teacher_outputs["last_hidden_state"], 
                        teacher_model.transformer.wte.weight.transpose(0, 1)
                    )
                    
                    # KL divergence loss
                    temperature = 2.0
                    soft_targets = nn.functional.softmax(teacher_logits / temperature, dim=-1)
                    log_probs = nn.functional.log_softmax(student_logits / temperature, dim=-1)
                    distill_loss = -torch.sum(soft_targets * log_probs, dim=-1).mean() * (temperature ** 2)
                    
                    # Combine losses
                    loss = distill_alpha * distill_loss + (1 - distill_alpha) * loss
                
                # Normalize loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                loss.backward()
            
            # Update loss metrics
            accumulated_loss += loss.item() * gradient_accumulation_steps
            
            # Apply dynamic sparsity during training
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Track learning rate
                current_lr = optimizer.param_groups[0]['lr']
                metrics['lr'].append(current_lr)
                
                # Apply weight pruning dynamically
                if sparsity > 0:
                    apply_dynamic_sparsity(model, sparsity)
                
                # Gradient clipping
                if scaler is not None:
                    scaler.unscale_(optimizer)
                
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Track gradient norms for monitoring
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                metrics['grad_norm'].append(total_norm)
                
                # Update weights
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                # Update learning rate
                scheduler.step()
                
                # Reset gradients
                optimizer.zero_grad()
                
                # Update step metrics
                global_step += 1
                epoch_loss += accumulated_loss
                metrics['train_loss'].append(accumulated_loss)
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": accumulated_loss,
                    "lr": current_lr,
                    "grad_norm": total_norm,
                    "sparsity": sparsity
                })
                
                # Reset accumulated loss
                accumulated_loss = 0.0
        
        # Average epoch loss
        epoch_loss /= len(train_loader)
        print(f"Epoch {epoch+1} average loss: {epoch_loss:.4f}")
        
        # Garbage collection to free up memory
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()
        
        # Evaluation
        if eval_loader:
            eval_loss = evaluate(model, eval_loader, device, teacher_model if distillation_enabled else None)
            metrics['val_loss'].append(eval_loss)
            print(f"Evaluation loss: {eval_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pt")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save in 8-bit quantized format if possible
        if has_bnb and use_8bit:
            # Save compressed checkpoint
            save_compressed_checkpoint(model, checkpoint_path)
        else:
            # Standard saving
            torch.save(model.state_dict(), checkpoint_path)
    
    # Save final model
    final_model_path = os.path.join(output_dir, "final_model.pt")
    
    # Save in 8-bit quantized format if possible
    if has_bnb and use_8bit:
        save_compressed_checkpoint(model, final_model_path)
    else:
        torch.save(model.state_dict(), final_model_path)
    
    # Save tokenizer
    model.tokenizer.save_vocab(os.path.join(output_dir, "vocab.json"))
    
    # Save training metrics
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        # Convert tensors to floats for JSON serialization
        serializable_metrics = {}
        for key, values in metrics.items():
            serializable_metrics[key] = [float(v) for v in values]
        json.dump(serializable_metrics, f)
    
    print(f"Training completed. Model saved to {output_dir}")
    return metrics

def evaluate(
    model: NanoRAG, 
    eval_loader: DataLoader, 
    device: str,
    teacher_model: Optional[NanoRAG] = None,
) -> float:
    """Evaluate the model on the evaluation dataset."""
    model.eval()
    total_loss = 0.0
    
    # Use tqdm for progress tracking
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            # Move batch to device
            for k, v in batch.items():
                batch[k] = v.to(device)
            
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            loss = outputs["loss"]
            
            # Apply knowledge distillation if teacher model is provided
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"]
                    )
                
                # Extract logits
                student_logits = torch.matmul(
                    outputs["last_hidden_state"], 
                    model.transformer.wte.weight.transpose(0, 1)
                )
                teacher_logits = torch.matmul(
                    teacher_outputs["last_hidden_state"], 
                    teacher_model.transformer.wte.weight.transpose(0, 1)
                )
                
                # KL divergence loss
                temperature = 2.0
                soft_targets = nn.functional.softmax(teacher_logits / temperature, dim=-1)
                log_probs = nn.functional.log_softmax(student_logits / temperature, dim=-1)
                distill_loss = -torch.sum(soft_targets * log_probs, dim=-1).mean() * (temperature ** 2)
                
                # Combine losses (50% CE, 50% KL)
                loss = 0.5 * loss + 0.5 * distill_loss
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(eval_loader)
    return avg_loss

def apply_dynamic_sparsity(model, sparsity_ratio=0.5):
    """Apply dynamic sparsity to model weights."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Only apply to weight matrices (not biases, embeddings or layernorms)
            if len(param.shape) == 2 and all(x not in name for x in ['layernorm', 'ln', 'bias', 'embedding']):
                # Get current values
                mask = torch.ones_like(param, dtype=torch.bool)
                values = param.abs().view(-1)
                
                # Identify lowest magnitude weights
                if len(values) > 100:  # Only worth it for larger matrices
                    k = int(len(values) * sparsity_ratio)
                    if k > 0:
                        threshold = torch.kthvalue(values, k).values
                        mask = param.abs() > threshold
                
                # Apply mask
                param.mul_(mask.float())

def save_compressed_checkpoint(model, path):
    """Save model in 8-bit quantized format if supported."""
    try:
        # Placeholder - normally would use int8 compression techniques
        # For 8-bit saving, you'd likely want custom serialization approaches
        # Here we're just using standard PyTorch to avoid dependencies
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")
    except Exception as e:
        print(f"Error saving compressed checkpoint: {e}. Falling back to standard save.")
        torch.save(model.state_dict(), path)

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float = 0.1
):
    """
    Create a schedule with a learning rate that decreases following the
    values of the cosine function between the initial lr and min_lr,
    after a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def main():
    """Example usage of training process."""
    print("Starting NeuraFlux training with optimized settings.")
    
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create model configuration with ultra-efficient settings
    config = NanoConfig(
        vocab_size=16000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        use_alibi=True,
        use_gqa=True,
        kv_heads=2,
        sliding_window=512,
        learning_rate=2e-4,
        warmup_steps=100
    )
    
    # Create tokenizer and model
    tokenizer = NanoTokenizer(vocab_size=config.vocab_size)
    model = NanoRAG(config, tokenizer)
    model = model.to(device)
    
    # Calculate model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters ({total_params/1_000_000:.2f}M)")
    
    # Load sample texts (or actual training data)
    sample_texts = load_sample_data("data/train.txt")
    
    # Train tokenizer vocabulary
    print("Training tokenizer vocabulary...")
    tokenizer.train_from_texts(sample_texts)
    
    # Create datasets
    train_dataset = TextDataset(sample_texts, tokenizer, max_length=128)
    
    # Split train/val
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create output directory
    os.makedirs("models", exist_ok=True)
    
    # Train
    train(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        num_epochs=3,
        batch_size=32,
        learning_rate=config.learning_rate,
        device=device,
        output_dir="models",
        use_8bit=has_bnb,  # Use 8-bit if available
        gradient_accumulation_steps=4,  # For memory efficiency
        warmup_steps=config.warmup_steps
    )
    
    # Generate some text as a test
    test_prompts = [
        "NeuraFlux is",
        "The transformer architecture",
        "Small language models"
    ]
    
    print("\nTesting model outputs:")
    for prompt in test_prompts:
        generated = model.generate(prompt, max_length=30, device=device)[0]
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}\n")

if __name__ == "__main__":
    main()
