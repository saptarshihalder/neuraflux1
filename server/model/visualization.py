import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import math

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Create output directory for visualizations
os.makedirs('visualizations', exist_ok=True)

def plot_training_metrics(metrics: Dict[str, List[float]], save_path: str = 'visualizations/training_metrics.png', show: bool = False):
    """
    Plot comprehensive training metrics with smoothing.
    
    Args:
        metrics: Dictionary containing training metrics
        save_path: Path to save the visualization
        show: Whether to display the plot (useful in notebooks)
    """
    plt.figure(figsize=(20, 15))
    
    # Smoothing function
    def smooth(scalars, weight=0.6):
        if len(scalars) == 0:
            return []
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed
    
    # Plot layout - 3x2 grid
    # Training Loss
    plt.subplot(3, 2, 1)
    if 'train_loss' in metrics and len(metrics['train_loss']) > 0:
        plt.plot(smooth(metrics['train_loss'], 0.7), label='Train Loss (Smoothed)', color=COLORS[0], alpha=0.8)
        plt.plot(metrics['train_loss'], label='Train Loss (Raw)', color=COLORS[0], alpha=0.3, linewidth=1)
    if 'val_loss' in metrics and len(metrics['val_loss']) > 0:
        plt.plot(metrics['val_loss'], label='Validation Loss', color=COLORS[1], marker='o')
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Learning Rate
    plt.subplot(3, 2, 2)
    if 'lr' in metrics and len(metrics['lr']) > 0:
        plt.plot(metrics['lr'], label='Learning Rate', color=COLORS[2])
        plt.title('Learning Rate Schedule', fontsize=14)
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
    
    # Rewards Plot - For RL training
    plt.subplot(3, 2, 3)
    if 'rewards' in metrics and len(metrics['rewards']) > 0:
        plt.plot(smooth(metrics['rewards'], 0.7), label='Average Reward (Smoothed)', color=COLORS[3], alpha=0.8)
        plt.plot(metrics['rewards'], label='Average Reward (Raw)', color=COLORS[3], alpha=0.3, linewidth=1)
        plt.title('Training Rewards', fontsize=14)
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)
    
    # Gradient Norm Plot
    plt.subplot(3, 2, 4)
    if 'grad_norm' in metrics and len(metrics['grad_norm']) > 0:
        plt.plot(smooth(metrics['grad_norm'], 0.7), label='Gradient Norm (Smoothed)', color=COLORS[4], alpha=0.8)
        plt.plot(metrics['grad_norm'], label='Gradient Norm (Raw)', color=COLORS[4], alpha=0.3, linewidth=1)
        plt.title('Gradient Flow', fontsize=14)
        plt.xlabel('Step')
        plt.ylabel('Norm')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
    
    # Policy KL Divergence - For RLHF/PPO
    plt.subplot(3, 2, 5)
    if 'policy_kl' in metrics and len(metrics['policy_kl']) > 0:
        plt.plot(smooth(metrics['policy_kl'], 0.7), label='KL Divergence (Smoothed)', color=COLORS[5], alpha=0.8)
        plt.plot(metrics['policy_kl'], label='KL Divergence (Raw)', color=COLORS[5], alpha=0.3, linewidth=1)
        plt.title('Policy KL Divergence', fontsize=14)
        plt.xlabel('Step')
        plt.ylabel('KL Divergence')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
    
    # Multi-task Reward Components
    plt.subplot(3, 2, 6)
    task_rewards = [key for key in metrics.keys() if key.startswith('reward_')]
    if task_rewards:
        for i, reward_key in enumerate(task_rewards):
            task_name = reward_key.replace('reward_', '')
            plt.plot(smooth(metrics[reward_key], 0.7), label=f'{task_name.capitalize()} (Smoothed)', color=COLORS[i % len(COLORS)], alpha=0.8)
        plt.title('Multi-task Reward Components', fontsize=14)
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_attention_maps(attention_maps: torch.Tensor, tokens: List[str] = None, 
                         layer: int = 0, save_path: str = 'visualizations/attention_maps.png', 
                         show: bool = False):
    """
    Visualize attention patterns for a specific layer.
    
    Args:
        attention_maps: Attention tensor (layers, batch, heads, seq_len, seq_len)
        tokens: List of token strings for axis labels
        layer: Which layer to visualize
        save_path: Path to save the visualization
        show: Whether to display the plot
    """
    # If tensor has a batch dimension but we only want one item
    if attention_maps.dim() == 5:
        attention = attention_maps[layer, 0]  # (heads, seq_len, seq_len)
    elif attention_maps.dim() == 4:
        attention = attention_maps[layer]  # (heads, seq_len, seq_len)
    else:
        raise ValueError(f"Unexpected attention maps shape: {attention_maps.shape}")
    
    n_heads = attention.size(0)
    seq_len = attention.size(1)
    
    # Create token labels if not provided
    if tokens is None:
        tokens = [f"Token {i+1}" for i in range(seq_len)]
    elif len(tokens) > seq_len:
        tokens = tokens[:seq_len]
    elif len(tokens) < seq_len:
        tokens = tokens + [f"Token {i+1}" for i in range(len(tokens), seq_len)]
    
    # Set up the figure
    fig = plt.figure(figsize=(20, 5 * math.ceil(n_heads / 4)))
    
    # Determine grid layout
    cols = 4
    rows = math.ceil(n_heads / cols)
    
    # Add a colorbar axis
    grid = plt.GridSpec(rows, cols + 1, wspace=0.4, hspace=0.3)
    cbar_ax = fig.add_subplot(grid[:, -1])
    
    # Plot each attention head
    for h in range(n_heads):
        ax = fig.add_subplot(grid[h // cols, h % cols])
        
        # Convert to numpy for plotting
        attn = attention[h].cpu().detach().numpy()
        
        # Create heatmap
        im = ax.imshow(attn, cmap='viridis', aspect='auto')
        
        # Add labels
        if seq_len <= 20:  # Only show labels for shorter sequences
            ax.set_xticks(np.arange(len(tokens)))
            ax.set_yticks(np.arange(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(tokens, fontsize=8)
        else:
            # Only label every 5th token
            step = max(1, seq_len // 10)
            ax.set_xticks(np.arange(0, len(tokens), step))
            ax.set_yticks(np.arange(0, len(tokens), step))
            ax.set_xticklabels([tokens[i] for i in range(0, len(tokens), step)], rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels([tokens[i] for i in range(0, len(tokens), step)], fontsize=8)
        
        ax.set_title(f"Head {h+1}")
    
    # Add colorbar
    plt.colorbar(im, cax=cbar_ax)
    
    plt.suptitle(f"Attention Patterns - Layer {layer+1}", fontsize=16)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_layer_activations(activations: Dict[str, torch.Tensor], 
                          save_path: str = 'visualizations/layer_activations.png',
                          show: bool = False):
    """
    Visualize the distribution of activations across model layers.
    
    Args:
        activations: Dictionary mapping layer names to activation tensors
        save_path: Path to save the visualization
        show: Whether to display the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Number of layers to plot
    n_layers = len(activations)
    
    # Set up for violin plots
    data = []
    labels = []
    
    for i, (layer_name, activation) in enumerate(activations.items()):
        # Flatten and convert to numpy
        act_numpy = activation.abs().flatten().cpu().detach().numpy()
        
        # Sample if too large (for efficiency)
        if len(act_numpy) > 10000:
            indices = np.random.choice(len(act_numpy), 10000, replace=False)
            act_numpy = act_numpy[indices]
        
        data.append(act_numpy)
        labels.append(layer_name)
    
    # Create violin plot
    plt.subplot(2, 1, 1)
    plt.violinplot(data, showmeans=True, showmedians=True)
    plt.xticks(np.arange(1, n_layers + 1), labels, rotation=45, ha='right')
    plt.title('Activation Distributions Across Layers', fontsize=14)
    plt.ylabel('Absolute Activation Value')
    plt.grid(True, axis='y')
    
    # Create boxplot for more detailed statistics
    plt.subplot(2, 1, 2)
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xticks(rotation=45, ha='right')
    plt.title('Activation Statistics (Quartiles)', fontsize=14)
    plt.ylabel('Absolute Activation Value')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_embedding_projections(embeddings: torch.Tensor, tokens: List[str], 
                              method: str = 'pca', # or 'tsne'
                              save_path: str = 'visualizations/embedding_space.png',
                              show: bool = False):
    """
    Visualize token embeddings in a 2D projection.
    
    Args:
        embeddings: Token embedding tensor (vocab_size, hidden_dim)
        tokens: List of token strings corresponding to the embeddings
        method: Projection method ('pca' or 'tsne')
        save_path: Path to save the visualization
        show: Whether to display the plot
    """
    # Convert to numpy for sklearn
    embed_np = embeddings.cpu().detach().numpy()
    
    # Apply dimensionality reduction
    if method.lower() == 'pca':
        projection = PCA(n_components=2).fit_transform(embed_np)
        title = 'PCA Projection of Token Embeddings'
    elif method.lower() == 'tsne':
        projection = TSNE(n_components=2, perplexity=min(30, len(tokens)-1)).fit_transform(embed_np)
        title = 't-SNE Projection of Token Embeddings'
    else:
        raise ValueError(f"Unknown projection method: {method}")
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'x': projection[:, 0],
        'y': projection[:, 1],
        'token': tokens
    })
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Scatter plot
    plt.scatter(df['x'], df['y'], alpha=0.7)
    
    # Only label points if not too many
    if len(tokens) <= 50:
        for i, row in df.iterrows():
            plt.annotate(row['token'], (row['x'], row['y']), fontsize=9)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_rl_specific_metrics(metrics: Dict[str, List[float]], 
                             save_path: str = 'visualizations/rl_metrics.png',
                             show: bool = False):
    """
    Plot metrics specific to reinforcement learning training.
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save the visualization
        show: Whether to display the plot
    """
    plt.figure(figsize=(15, 15))
    
    # PPO specific metrics
    plt.subplot(3, 2, 1)
    if 'ppo_clip_fraction' in metrics and len(metrics['ppo_clip_fraction']) > 0:
        plt.plot(metrics['ppo_clip_fraction'], label='Clip Fraction', color=COLORS[0])
        plt.title('PPO Clip Fraction', fontsize=14)
        plt.xlabel('Update Step')
        plt.ylabel('Fraction')
        plt.grid(True)
    
    plt.subplot(3, 2, 2)
    if 'ppo_value_loss' in metrics and len(metrics['ppo_value_loss']) > 0:
        plt.plot(metrics['ppo_value_loss'], label='Value Loss', color=COLORS[1])
        plt.title('PPO Value Function Loss', fontsize=14)
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.grid(True)
    
    # DPO specific metrics
    plt.subplot(3, 2, 3)
    if 'dpo_loss' in metrics and len(metrics['dpo_loss']) > 0:
        plt.plot(metrics['dpo_loss'], label='DPO Loss', color=COLORS[2])
        plt.title('Direct Preference Optimization Loss', fontsize=14)
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.grid(True)
    
    plt.subplot(3, 2, 4)
    if 'dpo_accuracy' in metrics and len(metrics['dpo_accuracy']) > 0:
        plt.plot(metrics['dpo_accuracy'], label='DPO Accuracy', color=COLORS[3])
        plt.title('DPO Preference Prediction Accuracy', fontsize=14)
        plt.xlabel('Update Step')
        plt.ylabel('Accuracy')
        plt.grid(True)
    
    # Constitutional AI metrics
    plt.subplot(3, 2, 5)
    if 'constitutional_compliance' in metrics and len(metrics['constitutional_compliance']) > 0:
        plt.plot(metrics['constitutional_compliance'], label='Compliance Rate', color=COLORS[4])
        plt.title('Constitutional AI Compliance Rate', fontsize=14)
        plt.xlabel('Update Step')
        plt.ylabel('Rate')
        plt.grid(True)
    
    # Multi-task weights
    plt.subplot(3, 2, 6)
    task_weights = [key for key in metrics.keys() if key.startswith('weight_')]
    if task_weights:
        for i, weight_key in enumerate(task_weights):
            task_name = weight_key.replace('weight_', '')
            plt.plot(metrics[weight_key], label=f'{task_name.capitalize()}', color=COLORS[i % len(COLORS)])
        plt.title('Multi-task RL Weights', fontsize=14)
        plt.xlabel('Update Step')
        plt.ylabel('Weight')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_token_importance(model, tokenizer, text: str, 
                              save_path: str = 'visualizations/token_importance.png',
                              show: bool = False):
    """
    Visualize which tokens contribute most to the model's predictions.
    
    Args:
        model: NanoRAG model
        tokenizer: The tokenizer
        text: Input text to analyze
        save_path: Path to save the visualization
        show: Whether to display the plot
    """
    # Tokenize input
    tokens = tokenizer.tokenize(text)
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    
    # Put model in eval mode
    model.eval()
    
    # Get baseline output
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        baseline_logits = outputs["logits"]
    
    # Get importance scores by zeroing out each token one by one
    importance_scores = []
    
    for i in range(len(tokens)):
        # Create modified input with one token masked
        modified_ids = input_ids.clone()
        modified_ids[0, i+1] = tokenizer.vocab[tokenizer.pad_token]  # +1 for the BOS token
        
        # Get prediction
        with torch.no_grad():
            outputs = model(input_ids=modified_ids)
            modified_logits = outputs["logits"]
        
        # Compute KL divergence between original and modified predictions
        kl_div = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(modified_logits[:, -1], dim=-1),
            torch.nn.functional.softmax(baseline_logits[:, -1], dim=-1),
            reduction='sum'
        ).item()
        
        importance_scores.append(kl_div)
    
    # Normalize scores
    max_score = max(importance_scores)
    if max_score > 0:
        importance_scores = [score / max_score for score in importance_scores]
    
    # Plot
    plt.figure(figsize=(14, 5))
    
    bars = plt.bar(range(len(tokens)), importance_scores, color='skyblue')
    
    # Color code by importance
    for i, score in enumerate(importance_scores):
        bars[i].set_color(plt.cm.viridis(score))
    
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
    plt.title('Token Importance for Prediction', fontsize=14)
    plt.xlabel('Token')
    plt.ylabel('Normalized Importance Score')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_stage_comparison(metrics_by_stage: Dict[str, Dict[str, List[float]]],
                                  metric_name: str = 'train_loss',
                                  save_path: str = 'visualizations/stage_comparison.png',
                                  show: bool = False):
    """
    Compare a specific metric across different training stages.
    
    Args:
        metrics_by_stage: Dictionary mapping stage names to metric dictionaries
        metric_name: Name of the metric to compare
        save_path: Path to save the visualization
        show: Whether to display the plot
    """
    plt.figure(figsize=(12, 8))
    
    for i, (stage, metrics) in enumerate(metrics_by_stage.items()):
        if metric_name in metrics and len(metrics[metric_name]) > 0:
            # Smooth the metrics
            def smooth(scalars, weight=0.6):
                if len(scalars) == 0:
                    return []
                last = scalars[0]
                smoothed = []
                for point in scalars:
                    smoothed_val = last * weight + (1 - weight) * point
                    smoothed.append(smoothed_val)
                    last = smoothed_val
                return smoothed
            
            plt.plot(smooth(metrics[metric_name], 0.7), 
                    label=f'{stage} (Smoothed)', 
                    color=COLORS[i % len(COLORS)],
                    alpha=0.8)
    
    plt.title(f'Comparison of {metric_name.replace("_", " ").title()} Across Training Stages', fontsize=14)
    plt.xlabel('Step')
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def generate_training_animation(metrics_over_time: List[Dict[str, List[float]]], 
                               save_path: str = 'visualizations/training_animation.gif',
                               fps: int = 2):
    """
    Create an animation showing how metrics evolve during training.
    
    Args:
        metrics_over_time: List of metric dictionaries at different training steps
        save_path: Path to save the animation
        fps: Frames per second
    """
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    def update(frame):
        # Clear all axes
        for ax in axs.flat:
            ax.clear()
        
        metrics = metrics_over_time[frame]
        
        # Plot training loss
        if 'train_loss' in metrics and len(metrics['train_loss']) > 0:
            axs[0, 0].plot(metrics['train_loss'], color=COLORS[0])
            axs[0, 0].set_title('Training Loss')
            axs[0, 0].set_xlabel('Step')
            axs[0, 0].set_ylabel('Loss')
            axs[0, 0].grid(True)
        
        # Plot rewards
        if 'rewards' in metrics and len(metrics['rewards']) > 0:
            axs[0, 1].plot(metrics['rewards'], color=COLORS[1])
            axs[0, 1].set_title('Rewards')
            axs[0, 1].set_xlabel('Step')
            axs[0, 1].set_ylabel('Reward')
            axs[0, 1].grid(True)
        
        # Plot gradient norm
        if 'grad_norm' in metrics and len(metrics['grad_norm']) > 0:
            axs[1, 0].plot(metrics['grad_norm'], color=COLORS[2])
            axs[1, 0].set_title('Gradient Norm')
            axs[1, 0].set_xlabel('Step')
            axs[1, 0].set_ylabel('Norm')
            axs[1, 0].set_yscale('log')
            axs[1, 0].grid(True)
        
        # Plot learning rate
        if 'lr' in metrics and len(metrics['lr']) > 0:
            axs[1, 1].plot(metrics['lr'], color=COLORS[3])
            axs[1, 1].set_title('Learning Rate')
            axs[1, 1].set_xlabel('Step')
            axs[1, 1].set_ylabel('LR')
            axs[1, 1].grid(True)
        
        fig.suptitle(f'Training Progress - Step {frame+1}/{len(metrics_over_time)}', fontsize=16)
        fig.tight_layout()
        
        return axs.flat
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(metrics_over_time), interval=1000//fps)
    
    # Save animation
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close()


def visualize_all_metrics(metrics: Dict[str, List[float]], 
                         model=None, tokenizer=None, 
                         example_text: str = None,
                         output_dir: str = 'visualizations'):
    """
    Generate all available visualizations based on provided metrics and model.
    
    Args:
        metrics: Dictionary of training metrics
        model: Optional model for additional visualizations
        tokenizer: Optional tokenizer for additional visualizations
        example_text: Optional example text for token-based visualizations
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Basic training metrics
    plot_training_metrics(metrics, save_path=f'{output_dir}/training_metrics.png')
    
    # RL specific metrics
    plot_rl_specific_metrics(metrics, save_path=f'{output_dir}/rl_metrics.png')
    
    # Model-based visualizations
    if model is not None and tokenizer is not None:
        # Get token embeddings
        token_embeddings = model.get_input_embeddings().weight
        
        # Get list of tokens in vocab
        vocab_tokens = [tokenizer.ids_to_tokens.get(i, "[UNK]") for i in range(min(100, len(tokenizer.vocab)))]
        vocab_embeddings = token_embeddings[:min(100, len(tokenizer.vocab))]
        
        # Plot embedding projections
        plot_embedding_projections(vocab_embeddings, vocab_tokens, 
                                method='pca', 
                                save_path=f'{output_dir}/embedding_pca.png')
        
        plot_embedding_projections(vocab_embeddings, vocab_tokens, 
                                method='tsne', 
                                save_path=f'{output_dir}/embedding_tsne.png')
        
        # If example text provided, do token importance visualization
        if example_text:
            visualize_token_importance(model, tokenizer, example_text, 
                                     save_path=f'{output_dir}/token_importance.png')
    
    print(f"All visualizations saved to {output_dir}/")


# Example usage
if __name__ == "__main__":
    # Generate sample metrics for demo
    steps = 100
    sample_metrics = {
        'train_loss': [3.0 * np.exp(-0.03 * i) + 0.2 * np.random.randn() for i in range(steps)],
        'val_loss': [3.0 * np.exp(-0.02 * i) + 0.5 + 0.3 * np.random.randn() for i in range(0, steps, 10)],
        'lr': [0.001 * (0.99 ** i) for i in range(steps)],
        'grad_norm': [10.0 * np.exp(-0.02 * i) + 1.0 + np.random.rand() for i in range(steps)],
        'rewards': [1.0 - 2.0 * np.exp(-0.05 * i) + 0.2 * np.random.randn() for i in range(steps)],
        'policy_kl': [0.5 * np.exp(-0.02 * i) + 0.1 * np.random.rand() for i in range(steps)],
        'ppo_clip_fraction': [0.3 * np.exp(-0.03 * i) + 0.05 * np.random.rand() for i in range(steps)],
        'dpo_loss': [1.0 * np.exp(-0.03 * i) + 0.2 * np.random.randn() for i in range(steps)],
        'constitutional_compliance': [0.5 + 0.5 * (1 - np.exp(-0.05 * i)) + 0.1 * np.random.rand() for i in range(steps)]
    }
    
    # Generate and save visualizations
    plot_training_metrics(sample_metrics, save_path='visualizations/demo_training_metrics.png', show=True)
    plot_rl_specific_metrics(sample_metrics, save_path='visualizations/demo_rl_metrics.png', show=True) 