#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to combine various grammar datasets into a single training file.
This prepares data for training the NeuraFlux model on English grammar.
"""

import os
import argparse
import random
from typing import List


def read_file(file_path: str) -> List[str]:
    """Read a text file and return its contents as a list of strings."""
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} does not exist. Skipping.")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def process_grammar_rules(file_path: str) -> List[str]:
    """Process the grammar rules file and convert it to training examples."""
    lines = read_file(file_path)
    examples = []
    current_section = ""
    
    for line in lines:
        if line.startswith("#"):
            # This is a heading - use as context
            current_section = line.lstrip("# ")
        elif line.startswith("EXAMPLE:"):
            # Extract the example
            example = line.replace("EXAMPLE:", "").strip()
            if current_section:
                examples.append(f"Grammar rule ({current_section}): {example}")
            else:
                examples.append(f"Grammar example: {example}")
    
    return examples


def process_corrections(file_path: str) -> List[str]:
    """Process the grammar corrections file and convert it to training examples."""
    lines = read_file(file_path)
    examples = []
    incorrect = ""
    
    for i, line in enumerate(lines):
        if line.startswith("INCORRECT:"):
            incorrect = line.replace("INCORRECT:", "").strip()
        elif line.startswith("CORRECT:") and incorrect:
            correct = line.replace("CORRECT:", "").strip()
            examples.append(f"Incorrect: {incorrect} -> Correct: {correct}")
            incorrect = ""
    
    return examples


def process_sentences(file_path: str) -> List[str]:
    """Process the example sentences file for training."""
    return read_file(file_path)


def combine_datasets(output_path: str, shuffle: bool = True) -> None:
    """Combine all grammar datasets into a single file."""
    # Define input files
    data_dir = os.path.dirname(os.path.abspath(output_path))
    grammar_rules_file = os.path.join(data_dir, "english_grammar.txt")
    corrections_file = os.path.join(data_dir, "grammar_corrections.txt")
    sentences_file = os.path.join(data_dir, "english_sentences.txt")
    advanced_grammar_file = os.path.join(data_dir, "advanced_grammar.txt")
    
    # Process each file
    examples = []
    
    # Add grammar rules
    examples.extend(process_grammar_rules(grammar_rules_file))
    
    # Add corrections
    examples.extend(process_corrections(corrections_file))
    
    # Add example sentences
    examples.extend(process_sentences(sentences_file))
    
    # Add advanced grammar
    examples.extend(process_grammar_rules(advanced_grammar_file))
    
    # Shuffle if requested
    if shuffle:
        random.shuffle(examples)
    
    # Write to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(f"{example}\n")
    
    print(f"Combined {len(examples)} examples into {output_path}")
    print(f"Dataset breakdown:")
    print(f"- Grammar rules: {len(process_grammar_rules(grammar_rules_file))}")
    print(f"- Grammar corrections: {len(process_corrections(corrections_file))}")
    print(f"- Example sentences: {len(process_sentences(sentences_file))}")
    print(f"- Advanced grammar: {len(process_grammar_rules(advanced_grammar_file))}")


def main():
    """Parse arguments and combine datasets."""
    parser = argparse.ArgumentParser(description="Combine grammar datasets for training")
    parser.add_argument("--output", type=str, default="combined_grammar_dataset.txt",
                        help="Path to output file")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Don't shuffle the examples")
    
    args = parser.parse_args()
    
    combine_datasets(args.output, not args.no_shuffle)


if __name__ == "__main__":
    main() 