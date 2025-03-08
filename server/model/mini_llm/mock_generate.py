#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mock text generation script for MiniLLM.
This script demonstrates how text generation would work with a trained model.
"""

import argparse
import random
from tokenizer.bpe_tokenizer import BPETokenizer

def mock_generate(prompt, max_length=50, temperature=0.7):
    """
    Mock text generation function.
    
    Args:
        prompt: Input prompt
        max_length: Maximum length of generated text
        temperature: Controls randomness (higher = more random)
        
    Returns:
        Generated text
    """
    # Sample continuations based on the prompt
    continuations = {
        "Small language models are": [
            "more efficient alternatives to large language models. They require less computational resources while still providing good performance for many tasks.",
            "becoming increasingly popular due to their efficiency and ability to run on consumer hardware. Unlike larger models, they can be deployed in resource-constrained environments.",
            "designed to be lightweight and efficient. They use techniques like knowledge distillation to learn from larger teacher models while maintaining a smaller parameter count."
        ],
        "Transformers have": [
            "revolutionized natural language processing with their self-attention mechanism. This allows them to capture long-range dependencies in text.",
            "become the dominant architecture for language models. Their parallel processing capabilities make them efficient for training on large datasets.",
            "changed how we approach sequence modeling tasks. The attention mechanism allows the model to focus on relevant parts of the input sequence."
        ],
        "Knowledge transfer helps": [
            "smaller models learn from larger ones. This process, also known as distillation, allows compact models to benefit from the knowledge of more powerful models.",
            "in creating more efficient models. By distilling knowledge from teacher models, student models can achieve similar performance with fewer parameters.",
            "improve the performance of resource-constrained models. It enables the development of models that can run on edge devices while maintaining good accuracy."
        ],
        "Neural networks can": [
            "learn complex patterns in data through their hierarchical structure. Each layer extracts increasingly abstract features from the input.",
            "process information in ways inspired by biological brains. Their interconnected nodes work together to solve complex problems.",
            "adapt to various tasks through training. By adjusting their weights based on examples, they learn to make accurate predictions."
        ],
        "The Gated Linear Unit": [
            "improves transformer efficiency by combining linear transformations with a gating mechanism. This allows the model to selectively control information flow.",
            "provides better parameter efficiency compared to standard feed-forward networks. It achieves this by learning which features are most important for the task.",
            "acts as a learnable feature selector in neural networks. The gating component helps focus on relevant information while suppressing noise."
        ],
        "Rotary position embeddings": [
            "encode relative positions more effectively than absolute embeddings. They help the model understand the relationships between tokens in a sequence.",
            "provide a more robust way to represent token positions. Their mathematical properties make them particularly suitable for language models.",
            "improve the model's ability to handle varying sequence lengths. They naturally generalize to positions not seen during training."
        ],
        "Sliding window attention": [
            "reduces computational complexity by focusing on local context. This makes it possible to process longer sequences efficiently.",
            "limits the attention span to a fixed window size, making memory usage more predictable. It's particularly useful for processing long documents.",
            "balances efficiency and context length by using a moving window of attention. This approach works well for many natural language tasks."
        ]
    }
    
    # Find the best matching prompt
    best_match = None
    best_match_length = 0
    
    for key in continuations:
        if prompt.lower().startswith(key.lower()) and len(key) > best_match_length:
            best_match = key
            best_match_length = len(key)
    
    if best_match:
        # Get possible continuations for the matched prompt
        possible_continuations = continuations[best_match]
        
        # Select a continuation based on temperature
        # Higher temperature = more random selection
        if temperature < 0.5:
            # Low temperature = pick first (most likely) continuation
            continuation = possible_continuations[0]
        elif temperature < 0.8:
            # Medium temperature = pick from first two
            continuation = random.choice(possible_continuations[:2])
        else:
            # High temperature = pick any
            continuation = random.choice(possible_continuations)
        
        # Combine prompt with continuation
        result = prompt + " " + continuation[len(best_match):].lstrip()
        
        # Truncate to max_length if needed
        if len(result.split()) > max_length:
            result = " ".join(result.split()[:max_length])
            result += "..."
            
        return result
    else:
        # No match found, generate generic text
        generic_continuations = [
            "an important area of research in artificial intelligence. Researchers are constantly developing new techniques to improve their performance.",
            "used in various applications including text generation, translation, and summarization. They have transformed how we interact with technology.",
            "trained on large datasets to learn patterns in language. This allows them to generate coherent and contextually relevant text.",
            "designed to process and understand complex patterns. Through careful architecture choices and training, they achieve impressive results.",
            "optimized for efficiency and performance. Modern approaches focus on balancing model size with computational requirements.",
            "built using advanced neural architectures. These systems combine multiple techniques to achieve better results."
        ]
        
        continuation = random.choice(generic_continuations)
        result = prompt + " " + continuation
        
        # Truncate to max_length if needed
        if len(result.split()) > max_length:
            result = " ".join(result.split()[:max_length])
            result += "..."
            
        return result

def main():
    """Parse arguments and generate text."""
    parser = argparse.ArgumentParser(description="Generate text using MiniLLM (Mock Version)")
    
    parser.add_argument("--prompt", type=str, required=True,
                        help="Input prompt for text generation")
    parser.add_argument("--max_length", type=int, default=50,
                        help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of different samples to generate")
    
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("MOCK GENERATION DEMO")
    print("="*50)
    print(f"Prompt: '{args.prompt}'")
    print(f"Settings: max_length={args.max_length}, temperature={args.temperature}")
    print("-"*50)
    
    # Generate multiple samples if requested
    for i in range(args.num_samples):
        generated_text = mock_generate(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature
        )
        
        print(f"\nSample {i+1}:")
        print(f"{generated_text}")
    
    print("\n" + "="*50)
    print("NOTE: This is a mock implementation for demonstration purposes.")
    print("In a real implementation, the model would generate text based on learned patterns.")
    print("="*50)

if __name__ == "__main__":
    main() 