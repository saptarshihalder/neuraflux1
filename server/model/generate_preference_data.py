import os
import random
import json
import torch
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm

def generate_prompt_templates() -> List[str]:
    """Generate templates for prompts to use in preference pairs"""
    return [
        "Explain the concept of {topic} in simple terms.",
        "What are the key principles of {topic}?",
        "How does {topic} work?",
        "Compare and contrast {topic} with {alt_topic}.",
        "What are the advantages and disadvantages of {topic}?",
        "How can {topic} be implemented effectively?",
        "What are common misconceptions about {topic}?",
        "Give an example of {topic} in practice.",
        "What's the history and development of {topic}?",
        "How might {topic} evolve in the future?"
    ]

def generate_topics() -> List[Dict[str, str]]:
    """Generate pairs of related topics to use in prompts"""
    return [
        {"topic": "transformer models", "alt_topic": "recurrent neural networks"},
        {"topic": "attention mechanisms", "alt_topic": "vanilla feed-forward networks"},
        {"topic": "self-supervised learning", "alt_topic": "supervised learning"},
        {"topic": "reinforcement learning", "alt_topic": "imitation learning"},
        {"topic": "natural language processing", "alt_topic": "computer vision"},
        {"topic": "embedding vectors", "alt_topic": "one-hot encoding"},
        {"topic": "language modeling", "alt_topic": "text classification"},
        {"topic": "fine-tuning", "alt_topic": "training from scratch"},
        {"topic": "tokenization", "alt_topic": "character-level processing"},
        {"topic": "transformer architecture", "alt_topic": "convolutional networks"},
        {"topic": "multi-head attention", "alt_topic": "single-head attention"},
        {"topic": "layer normalization", "alt_topic": "batch normalization"},
        {"topic": "positional encoding", "alt_topic": "recurrent state"},
        {"topic": "beam search", "alt_topic": "greedy decoding"},
        {"topic": "retrieval augmented generation", "alt_topic": "pure generative models"}
    ]

def generate_good_completion_templates() -> List[str]:
    """Generate templates for high-quality completions"""
    return [
        "{topic} refers to {definition}. The key principles include {principles}. This is important because {importance}.",
        "The concept of {topic} involves {definition}. It works by {mechanism}, which helps achieve {benefit}.",
        "{topic} can be understood as {definition}. Unlike {alt_topic}, it {difference}, which makes it {advantage}.",
        "In simple terms, {topic} is {definition}. It's commonly used in {application} because {reason}.",
        "The fundamental idea behind {topic} is {definition}. This approach has several advantages: {advantages}.",
        "{topic} was developed to {purpose}. It addresses the problem of {problem} by {solution}.",
        "When implementing {topic}, it's important to consider {considerations}. Best practices include {practices}.",
        "A common misconception about {topic} is {misconception}. In reality, {reality}.",
        "The evolution of {topic} has gone through several phases: {phases}. Current research focuses on {research}.",
        "Experts generally agree that {topic} offers {benefits}, though there are limitations such as {limitations}."
    ]

def generate_poor_completion_templates() -> List[str]:
    """Generate templates for low-quality completions"""
    return [
        "{topic} is a thing that does stuff with {alt_topic}. It's pretty complicated.",
        "I'm not really sure about {topic}, but I think it's related to {alt_topic} somehow.",
        "{topic} is the best approach and anyone who uses {alt_topic} is wrong. Trust me on this.",
        "{topic} was invented by someone at some point to do something with {alt_topic}.",
        "The answer to your question about {topic} is simple but too complicated to explain here.",
        "{topic} is basically the same as {alt_topic}, just with a different name.",
        "I could explain {topic}, but it would take too long. Just Google it.",
        "{topic} is totally overrated. Everyone knows {alt_topic} is better for everything.",
        "The secret to understanding {topic} is [redacted due to confidentiality].",
        "{topic} is a term used to confuse beginners. It's not actually important."
    ]

def fill_template(template: str, replacements: Dict[str, str]) -> str:
    """Fill a template with specific replacements"""
    result = template
    for key, value in replacements.items():
        result = result.replace("{" + key + "}", value)
    return result

def generate_content_items() -> Dict[str, List[str]]:
    """Generate content items to fill in templates"""
    return {
        "definition": [
            "a method for processing sequential data using attention mechanisms",
            "an approach that focuses on relevant parts of the input",
            "a technique that allows models to weigh the importance of different inputs",
            "a framework that enables parallelized processing of sequences",
            "a design that captures dependencies regardless of their distance",
            "an architecture that revolutionized sequence modeling tasks",
            "a paradigm that replaces traditional recurrent processing",
            "a structure that allows efficient learning of complex patterns",
            "a system that models relationships between all input elements"
        ],
        "principles": [
            "self-attention, parallelization, and position-aware processing",
            "multi-head attention, residual connections, and layer normalization",
            "contextualized representations, positional encoding, and feed-forward networks",
            "distributed attention, position-aware computations, and non-recurrent processing",
            "attention-based processing, residual learning, and normalized activations"
        ],
        "importance": [
            "it enables more efficient and effective processing of sequential data",
            "it allows models to capture long-range dependencies more effectively",
            "it significantly improves performance on a wide range of NLP tasks",
            "it provides a foundation for state-of-the-art language models",
            "it represents a fundamental shift in how sequence modeling is approached"
        ],
        "mechanism": [
            "computing attention scores between all input elements",
            "transforming inputs using parallel attention heads",
            "weighting input elements based on their relevance",
            "creating contextualized representations of each input",
            "combining information from different representation subspaces"
        ],
        "benefit": [
            "improved performance on sequence modeling tasks",
            "better handling of long-range dependencies",
            "more parallelizable computation for faster training",
            "richer representations of complex linguistic structures",
            "state-of-the-art results on various benchmarks"
        ],
        "difference": [
            "processes all elements in parallel rather than sequentially",
            "directly models relationships between any two positions",
            "doesn't rely on recursive state propagation",
            "uses attention to weight the importance of different inputs",
            "captures contextual information through direct connections"
        ],
        "advantage": [
            "more efficient for training on large datasets",
            "better at capturing long-range dependencies",
            "more flexible in how it processes information",
            "more powerful in modeling complex patterns",
            "easier to optimize during training"
        ],
        "application": [
            "machine translation, text summarization, and question answering",
            "language modeling, sentiment analysis, and named entity recognition",
            "text generation, document classification, and dialogue systems",
            "content creation, information extraction, and text-to-speech",
            "speech recognition, code completion, and automated reasoning"
        ],
        "reason": [
            "it outperforms previous approaches in accuracy and efficiency",
            "it handles complex linguistic patterns more effectively",
            "it scales better to larger datasets and model sizes",
            "it requires less manual feature engineering",
            "it provides more contextual understanding of language"
        ],
        "advantages": [
            "parallelizability, long-range dependency handling, and flexibility",
            "improved accuracy, computational efficiency, and scalability",
            "better contextual understanding, training stability, and generalization",
            "handling of complex patterns, position awareness, and transfer learning capabilities",
            "reduced training time, higher performance ceiling, and multi-task learning potential"
        ],
        "purpose": [
            "address limitations in sequential processing models",
            "enable more efficient training on large datasets",
            "improve the handling of long-range dependencies",
            "provide better contextualized representations",
            "create more powerful general-purpose language models"
        ],
        "problem": [
            "sequential computation bottlenecks in RNNs",
            "vanishing gradients in deep recurrent networks",
            "limited context windows in traditional models",
            "computational inefficiency in processing long sequences",
            "difficulty in capturing complex linguistic patterns"
        ],
        "solution": [
            "replacing recurrence with parallel attention mechanisms",
            "enabling direct connections between distant positions",
            "using multi-head attention to capture different relationship types",
            "implementing residual connections and normalization for stable training",
            "leveraging positional encodings to maintain sequence information"
        ],
        "considerations": [
            "computational resources, dataset quality, and model size",
            "hyperparameter tuning, attention mechanism design, and regularization",
            "positional encoding strategies, optimization algorithms, and hardware requirements",
            "batch size, sequence length limitations, and training stability",
            "attention pattern analysis, interpretability, and deployment constraints"
        ],
        "practices": [
            "proper initialization, gradient clipping, and learning rate scheduling",
            "using pre-normalization, appropriate attention variants, and efficient implementations",
            "data preprocessing, tokenization strategies, and position-aware modeling",
            "transfer learning from pre-trained models, careful regularization, and validation",
            "monitoring attention patterns, analyzing layer contributions, and model pruning"
        ],
        "misconception": [
            "it eliminates the need for recurrence in all cases",
            "it automatically solves all long-range dependency problems",
            "it's only useful for natural language processing tasks",
            "bigger models are always better regardless of the task",
            "it works well with limited data and computational resources"
        ],
        "reality": [
            "recurrence can still be valuable in certain scenarios",
            "while improved, handling very long contexts remains challenging",
            "it's been successfully applied to vision, audio, and multimodal tasks",
            "model size should be balanced with task complexity and available data",
            "it generally requires substantial data and computation for optimal results"
        ],
        "phases": [
            "initial research, benchmark breakthroughs, and widespread adoption",
            "theoretical development, architectural refinement, and scaling laws discovery",
            "specialized variants, efficiency improvements, and multimodal extensions",
            "academic research, industrial deployment, and open-source democratization",
            "architecture innovation, scaling efforts, and alignment techniques"
        ],
        "research": [
            "efficiency improvements, sparse attention, and specialized architectures",
            "long-context modeling, interpretability, and knowledge integration",
            "multimodal extensions, few-shot capabilities, and ethical considerations",
            "parameter-efficient fine-tuning, retrieval augmentation, and reasoning abilities",
            "alignment with human values, factuality, and safety mechanisms"
        ],
        "benefits": [
            "improved performance, scalability, and flexibility",
            "better contextual understanding, efficiency, and generalization",
            "enhanced modeling power, training stability, and multimodal potential",
            "superior handling of complex patterns, transfer learning capabilities, and parallelization",
            "state-of-the-art results, adaptability, and reduced manual engineering"
        ],
        "limitations": [
            "quadratic complexity with sequence length, high resource requirements, and potential attention bottlenecks",
            "challenging interpretability, data inefficiency, and position modeling constraints",
            "limited explicit structural biases, high computational costs, and potential overfitting",
            "scalability challenges, implementation complexity, and potential attention diffusion",
            "hardware constraints, optimization difficulties, and architectural rigidity"
        ]
    }

def create_preference_dataset(num_pairs: int = 100, output_path: str = "data/preferences.txt"):
    """Create a synthetic preference dataset with chosen vs rejected completions"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    prompt_templates = generate_prompt_templates()
    topics = generate_topics()
    good_completion_templates = generate_good_completion_templates()
    poor_completion_templates = generate_poor_completion_templates()
    content_items = generate_content_items()
    
    prompt_topic_pairs = []
    for _ in range(num_pairs):
        template = random.choice(prompt_templates)
        topic_pair = random.choice(topics)
        prompt_topic_pairs.append((template, topic_pair))
    
    pairs = []
    for template, topic_pair in tqdm(prompt_topic_pairs, desc="Generating preference pairs"):
        # Create the prompt
        prompt = fill_template(template, topic_pair)
        
        # Create a good completion
        good_template = random.choice(good_completion_templates)
        good_completion_content = {}
        for key in [k for k in good_template.split("{") if "}" in k]:
            key = key.split("}")[0]
            if key in content_items and key not in good_completion_content:
                good_completion_content[key] = random.choice(content_items[key])
        good_completion = fill_template(good_template, {**topic_pair, **good_completion_content})
        
        # Create a poor completion
        poor_template = random.choice(poor_completion_templates)
        poor_completion = fill_template(poor_template, topic_pair)
        
        pairs.append((prompt, good_completion, poor_completion))
    
    # Write to output file in TSV format
    with open(output_path, "w", encoding="utf-8") as f:
        for prompt, chosen, rejected in pairs:
            f.write(f"{prompt}\t{chosen}\t{rejected}\n")
    
    print(f"✅ Created {len(pairs)} preference pairs at {output_path}")
    return pairs

if __name__ == "__main__":
    create_preference_dataset(200, "data/preferences.txt") 