import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from IPython.display import HTML
from typing import Dict, List, Optional
import glob
import re

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.1)


def load_metrics_files(metrics_dir="visualizations"):
    """Load all metrics files from the visualizations directory"""
    metrics_by_stage = {
        'supervised': {},
        'reward_model': {},
        'ppo': {},
        'dpo': {},
        'constitutional': {},
        'multi_task': {}
    }
    
    # Regular expressions to find files for each stage
    stage_patterns = {
        'supervised': r'supervised_metrics_epoch_\d+\.png',
        'reward_model': r'reward_model_metrics_epoch_\d+\.png',
        'ppo': r'ppo_metrics_epoch_\d+.*\.png',
        'dpo': r'dpo_metrics_epoch_\d+\.png',
        'constitutional': r'constitutional_metrics_iter_\d+\.png',
        'multi_task': r'multi_task_metrics_epoch_\d+.*\.png'
    }
    
    # Count files for each stage
    stage_counts = {}
    for stage, pattern in stage_patterns.items():
        files = glob.glob(os.path.join(metrics_dir, pattern))
        stage_counts[stage] = len(files)
    
    # Create simulated training progress data if real data isn't available
    if sum(stage_counts.values()) == 0:
        print("No metrics files found. Generating simulated metrics for demonstration.")
        metrics_by_stage = generate_simulated_metrics()
    else:
        # Here you would parse the actual metrics if they were saved as JSON alongside the PNGs
        # For this example, we'll use simulated data
        metrics_by_stage = generate_simulated_metrics()
        
    return metrics_by_stage


def generate_simulated_metrics():
    """Generate simulated metrics for demonstration purposes"""
    steps = 100
    np.random.seed(42)  # For reproducibility
    
    metrics_by_stage = {}
    
    # Supervised fine-tuning metrics
    metrics_by_stage['supervised'] = {
        'train_loss': [3.0 * np.exp(-0.05 * i) + 0.2 * np.random.randn() for i in range(steps)],
        'val_loss': [3.0 * np.exp(-0.03 * i) + 0.5 + 0.3 * np.random.randn() for i in range(0, steps, 10)],
        'lr': [0.001 * (0.99 ** i) for i in range(steps)],
        'grad_norm': [10.0 * np.exp(-0.02 * i) + 1.0 + np.random.rand() for i in range(steps)]
    }
    
    # Reward model metrics
    metrics_by_stage['reward_model'] = {
        'train_loss': [2.0 * np.exp(-0.04 * i) + 0.15 * np.random.randn() for i in range(steps)],
        'val_loss': [2.0 * np.exp(-0.03 * i) + 0.4 + 0.2 * np.random.randn() for i in range(0, steps, 10)]
    }
    
    # PPO metrics
    metrics_by_stage['ppo'] = {
        'rewards': [-1.0 + 2.0 * (1 - np.exp(-0.05 * i)) + 0.2 * np.random.randn() for i in range(steps)],
        'policy_kl': [0.5 * np.exp(-0.03 * i) + 0.1 * np.random.rand() for i in range(steps)],
        'ppo_clip_fraction': [0.3 * np.exp(-0.04 * i) + 0.05 * np.random.rand() for i in range(steps)],
        'ppo_value_loss': [1.5 * np.exp(-0.03 * i) + 0.1 * np.random.rand() for i in range(steps)]
    }
    
    # DPO metrics
    metrics_by_stage['dpo'] = {
        'dpo_loss': [1.0 * np.exp(-0.04 * i) + 0.15 * np.random.randn() for i in range(steps)],
        'dpo_accuracy': [0.5 + 0.4 * (1 - np.exp(-0.05 * i)) + 0.05 * np.random.rand() for i in range(steps)]
    }
    
    # Constitutional AI metrics
    metrics_by_stage['constitutional'] = {
        'constitutional_compliance': [0.5 + 0.4 * (1 - np.exp(-0.05 * i)) + 0.05 * np.random.rand() for i in range(steps)]
    }
    
    # Multi-task RL metrics
    metrics_by_stage['multi_task'] = {
        'rewards': [0.0 + 1.5 * (1 - np.exp(-0.03 * i)) + 0.2 * np.random.randn() for i in range(steps)],
        'reward_helpfulness': [0.3 + 0.6 * (1 - np.exp(-0.03 * i)) + 0.1 * np.random.rand() for i in range(steps)],
        'reward_harmlessness': [0.5 + 0.4 * (1 - np.exp(-0.04 * i)) + 0.05 * np.random.rand() for i in range(steps)],
        'reward_honesty': [0.4 + 0.5 * (1 - np.exp(-0.03 * i)) + 0.1 * np.random.rand() for i in range(steps)]
    }
    
    return metrics_by_stage


def create_training_overview(metrics_by_stage: Dict[str, Dict[str, List[float]]], output_path: str = "visualizations/training_overview.png"):
    """Create a comprehensive overview of all training stages"""
    plt.figure(figsize=(20, 16))
    
    # Define stages and their order
    stages = ['supervised', 'reward_model', 'ppo', 'dpo', 'constitutional', 'multi_task']
    stage_names = {
        'supervised': 'Supervised Fine-tuning',
        'reward_model': 'Reward Model Training',
        'ppo': 'PPO/RLHF Training',
        'dpo': 'Direct Preference Optimization',
        'constitutional': 'Constitutional AI Alignment',
        'multi_task': 'Multi-task Reinforcement Learning'
    }
    
    # Plot layout
    gs = GridSpec(3, 4, figure=plt.gcf(), hspace=0.4, wspace=0.3)
    
    # Plot 1: Training Loss Across Stages
    ax1 = plt.subplot(gs[0, :2])
    for i, stage in enumerate(['supervised', 'reward_model', 'dpo']):
        if stage in metrics_by_stage and 'train_loss' in metrics_by_stage[stage]:
            data = metrics_by_stage[stage]['train_loss']
            x = np.arange(len(data))
            ax1.plot(x, data, label=f"{stage_names[stage]}", alpha=0.8)
    ax1.set_title('Training Loss Across Stages', fontsize=16)
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot 2: Reward Evolution
    ax2 = plt.subplot(gs[0, 2:])
    for i, stage in enumerate(['ppo', 'multi_task']):
        if stage in metrics_by_stage and 'rewards' in metrics_by_stage[stage]:
            data = metrics_by_stage[stage]['rewards']
            x = np.arange(len(data))
            ax2.plot(x, data, label=f"{stage_names[stage]}", alpha=0.8)
    ax2.set_title('Reward Evolution During RL Training', fontsize=16)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Reward')
    ax2.legend()
    
    # Plot 3: Multi-task Reward Components
    ax3 = plt.subplot(gs[1, :2])
    stage = 'multi_task'
    if stage in metrics_by_stage:
        reward_keys = [k for k in metrics_by_stage[stage].keys() if k.startswith('reward_')]
        for key in reward_keys:
            if key in metrics_by_stage[stage]:
                data = metrics_by_stage[stage][key]
                x = np.arange(len(data))
                ax3.plot(x, data, label=f"{key.replace('reward_', '').capitalize()}", alpha=0.8)
    ax3.set_title('Multi-task Reward Components', fontsize=16)
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Component Reward')
    ax3.legend()
    
    # Plot 4: PPO Metrics
    ax4 = plt.subplot(gs[1, 2:])
    stage = 'ppo'
    metrics_to_plot = ['policy_kl', 'ppo_clip_fraction', 'ppo_value_loss']
    metric_labels = {
        'policy_kl': 'KL Divergence',
        'ppo_clip_fraction': 'Clip Fraction',
        'ppo_value_loss': 'Value Loss'
    }
    
    if stage in metrics_by_stage:
        for metric in metrics_to_plot:
            if metric in metrics_by_stage[stage]:
                data = metrics_by_stage[stage][metric]
                x = np.arange(len(data))
                ax4.plot(x, data, label=metric_labels.get(metric, metric), alpha=0.8)
    ax4.set_title('PPO Training Metrics', fontsize=16)
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Value')
    ax4.legend()
    
    # Plot 5: DPO Performance
    ax5 = plt.subplot(gs[2, :2])
    stage = 'dpo'
    if stage in metrics_by_stage and 'dpo_accuracy' in metrics_by_stage[stage]:
        data = metrics_by_stage[stage]['dpo_accuracy']
        x = np.arange(len(data))
        ax5.plot(x, data, label='Preference Prediction Accuracy', color='green', alpha=0.8)
        ax5.set_title('DPO Preference Prediction Accuracy', fontsize=16)
        ax5.set_xlabel('Training Step')
        ax5.set_ylabel('Accuracy')
        ax5.set_ylim([0.5, 1.0])
    
    # Plot 6: Constitutional AI Compliance
    ax6 = plt.subplot(gs[2, 2:])
    stage = 'constitutional'
    if stage in metrics_by_stage and 'constitutional_compliance' in metrics_by_stage[stage]:
        data = metrics_by_stage[stage]['constitutional_compliance']
        x = np.arange(len(data))
        ax6.plot(x, data, label='Constitutional Compliance', color='purple', alpha=0.8)
        ax6.set_title('Constitutional AI Compliance Score', fontsize=16)
        ax6.set_xlabel('Training Step')
        ax6.set_ylabel('Compliance Score')
        ax6.set_ylim([0, 1.0])
    
    plt.suptitle('NeuraFlux Advanced Training Overview', fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created training overview visualization at {output_path}")
    

def create_stage_comparison(metrics_by_stage: Dict[str, Dict[str, List[float]]], output_path: str = "visualizations/stage_comparison.png"):
    """Create visualizations comparing metrics across different training stages"""
    plt.figure(figsize=(18, 10))
    
    # Plot layouts
    plt.subplot(1, 2, 1)
    
    # Compare rewards across RL stages
    stages_with_rewards = []
    for stage in ['ppo', 'multi_task']:
        if stage in metrics_by_stage and 'rewards' in metrics_by_stage[stage]:
            stages_with_rewards.append(stage)
    
    if stages_with_rewards:
        # Smoothing function
        def smooth(scalars, weight=0.7):
            if len(scalars) == 0:
                return []
            last = scalars[0]
            smoothed = []
            for point in scalars:
                smoothed_val = last * weight + (1 - weight) * point
                smoothed.append(smoothed_val)
                last = smoothed_val
            return smoothed
        
        for stage in stages_with_rewards:
            data = metrics_by_stage[stage]['rewards']
            plt.plot(smooth(data), label=f"{stage.capitalize()} Rewards (Smoothed)")
        
        plt.title('Reward Evolution Across Different RL Stages', fontsize=16)
        plt.xlabel('Training Step')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)
    
    # Compare loss curves
    plt.subplot(1, 2, 2)
    stages_with_loss = []
    for stage in ['supervised', 'reward_model', 'dpo']:
        if stage in metrics_by_stage and 'train_loss' in metrics_by_stage[stage]:
            stages_with_loss.append(stage)
    
    if stages_with_loss:
        for stage in stages_with_loss:
            data = metrics_by_stage[stage]['train_loss']
            # Normalize data for easier comparison
            min_val = min(data)
            max_val = max(data)
            if max_val > min_val:
                normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
            else:
                normalized_data = data
            plt.plot(smooth(normalized_data), label=f"{stage.capitalize()} Loss (Normalized)")
        
        plt.title('Normalized Loss Curves Across Training Stages', fontsize=16)
        plt.xlabel('Training Step')
        plt.ylabel('Normalized Loss')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created stage comparison visualization at {output_path}")


def create_training_animation(metrics_by_stage: Dict[str, Dict[str, List[float]]], output_path: str = "visualizations/training_animation.gif"):
    """Create an animated visualization of training progress"""
    # Set up the figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress Animation', fontsize=16)
    
    # Get max steps across all stages for x-axis
    max_steps = 0
    for stage, metrics in metrics_by_stage.items():
        for metric_name, values in metrics.items():
            max_steps = max(max_steps, len(values))
    
    # Animation update function
    def update(frame):
        for ax in axs.flat:
            ax.clear()
        
        # Plot 1: Training Loss
        ax1 = axs[0, 0]
        for stage in ['supervised', 'reward_model', 'dpo']:
            if stage in metrics_by_stage and 'train_loss' in metrics_by_stage[stage]:
                data = metrics_by_stage[stage]['train_loss'][:frame]
                if data:
                    ax1.plot(data, label=f"{stage.capitalize()}")
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Rewards
        ax2 = axs[0, 1]
        for stage in ['ppo', 'multi_task']:
            if stage in metrics_by_stage and 'rewards' in metrics_by_stage[stage]:
                data = metrics_by_stage[stage]['rewards'][:frame]
                if data:
                    ax2.plot(data, label=f"{stage.capitalize()}")
        ax2.set_title('Rewards')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Reward')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Multi-task Rewards
        ax3 = axs[1, 0]
        stage = 'multi_task'
        if stage in metrics_by_stage:
            reward_keys = [k for k in metrics_by_stage[stage].keys() if k.startswith('reward_')]
            for key in reward_keys:
                if key in metrics_by_stage[stage]:
                    data = metrics_by_stage[stage][key][:frame]
                    if data:
                        ax3.plot(data, label=f"{key.replace('reward_', '').capitalize()}")
        ax3.set_title('Multi-task Reward Components')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Component Reward')
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: Constitutional Compliance
        ax4 = axs[1, 1]
        stage = 'constitutional'
        if stage in metrics_by_stage and 'constitutional_compliance' in metrics_by_stage[stage]:
            data = metrics_by_stage[stage]['constitutional_compliance'][:frame]
            if data:
                ax4.plot(data, label='Compliance Score', color='purple')
                ax4.set_ylim([0, 1.0])
        ax4.set_title('Constitutional Compliance')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Score')
        ax4.grid(True)
        
        # Add frame indicator
        fig.suptitle(f'Training Progress - Step {frame}/{max_steps}', fontsize=16)
        
        return axs.flat
    
    # Create animation
    frames = min(max_steps, 100)  # Limit to 100 frames for efficiency
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=100)
    
    # Save as GIF
    anim.save(output_path, writer='pillow', fps=10)
    plt.close()
    
    print(f"✅ Created training animation at {output_path}")


def main():
    """Main function to generate all visualizations"""
    print("Generating visualization dashboard for NeuraFlux training...")
    
    # Create visualizations directory if it doesn't exist
    os.makedirs("visualizations", exist_ok=True)
    
    # Load metrics (or generate simulated ones)
    metrics_by_stage = load_metrics_files()
    
    # Create comprehensive training overview
    create_training_overview(metrics_by_stage)
    
    # Create stage comparison visualizations
    create_stage_comparison(metrics_by_stage)
    
    # Create training animation
    create_training_animation(metrics_by_stage)
    
    print("\n✅ Training visualization dashboard completed!")
    print("Check the visualizations/ directory for all generated visualizations.")


if __name__ == "__main__":
    main() 