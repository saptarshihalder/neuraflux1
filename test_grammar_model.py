#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple script to test and demonstrate the grammar capabilities of the trained model
using mock implementations for demonstration purposes.
"""

import os
import sys
import json
import random
from typing import List, Dict

# Add the correction examples data
CORRECTION_EXAMPLES = [
    {
        "incorrect": "She don't like chocolate cake.",
        "correct": "She doesn't like chocolate cake."
    },
    {
        "incorrect": "The children plays in the park every day.",
        "correct": "The children play in the park every day."
    },
    {
        "incorrect": "Me and my friend went to the movies.",
        "correct": "My friend and I went to the movies."
    },
    {
        "incorrect": "I seen the new movie yesterday.",
        "correct": "I saw the new movie yesterday."
    },
    {
        "incorrect": "The cat licked it's paws.",
        "correct": "The cat licked its paws."
    },
    {
        "incorrect": "Him and her are getting married next month.",
        "correct": "He and she are getting married next month."
    },
    {
        "incorrect": "There going to announce the results tomorrow.",
        "correct": "They're going to announce the results tomorrow."
    },
    {
        "incorrect": "The book is laying on the table.",
        "correct": "The book is lying on the table."
    },
    {
        "incorrect": "I should of studied harder for the test.",
        "correct": "I should have studied harder for the test."
    },
    {
        "incorrect": "Your going to love this restaurant.",
        "correct": "You're going to love this restaurant."
    }
]

class MockGrammarModel:
    """Mock implementation of a grammar correction model for demonstration."""
    
    def __init__(self, accuracy: float = 0.85):
        """
        Initialize the mock grammar model.
        
        Args:
            accuracy: The simulated accuracy of the model (0.0 to 1.0)
        """
        self.accuracy = accuracy
        self.loaded_data = self._load_grammar_corrections()
        print(f"Mock Grammar Model initialized with {len(self.loaded_data)} correction patterns")
        
    def _load_grammar_corrections(self) -> Dict[str, str]:
        """Load the grammar corrections from file."""
        corrections = {}
        
        try:
            with open('grammar_corrections.txt', 'r', encoding='utf-8') as f:
                current_incorrect = None
                
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    if line.startswith("INCORRECT:"):
                        current_incorrect = line.replace("INCORRECT:", "").strip()
                    elif line.startswith("CORRECT:") and current_incorrect:
                        correct = line.replace("CORRECT:", "").strip()
                        corrections[current_incorrect] = correct
                        current_incorrect = None
            
            print(f"Loaded {len(corrections)} corrections from file")
        except Exception as e:
            print(f"Error loading corrections from file: {e}")
            # Fall back to examples
            for item in CORRECTION_EXAMPLES:
                corrections[item["incorrect"]] = item["correct"]
            
        return corrections
    
    def correct_grammar(self, text: str) -> Dict:
        """
        Simulate grammar correction.
        
        Args:
            text: The input text to correct
            
        Returns:
            Dictionary with the original text, corrected text, and confidence
        """
        # Check if this is one of our known examples
        if text in self.loaded_data:
            correct_text = self.loaded_data[text]
            # Simulate confidence based on accuracy
            confidence = self.accuracy + (random.random() * 0.1)
            return {
                "original": text,
                "corrected": correct_text,
                "confidence": min(confidence, 0.99)
            }
        
        # For other text, simulate partial correction
        words = text.split()
        if random.random() < self.accuracy and len(words) > 3:
            # Randomly change a word to simulate correction
            idx = random.randint(0, len(words) - 1)
            if words[idx].lower() == "dont":
                words[idx] = "don't"
            elif words[idx].lower() == "its":
                words[idx] = "it's" if random.random() < 0.5 else "its"
            elif words[idx].lower() == "there":
                words[idx] = "they're" if random.random() < 0.7 else "there"
            elif words[idx].lower() == "your":
                words[idx] = "you're" if random.random() < 0.7 else "your"
                
            corrected = " ".join(words)
            confidence = 0.5 + (random.random() * 0.3)
            
            return {
                "original": text,
                "corrected": corrected,
                "confidence": confidence
            }
        
        # If no correction, return original with low confidence
        return {
            "original": text,
            "corrected": text,
            "confidence": 0.3 + (random.random() * 0.2)
        }
    
    def get_grammar_explanation(self, text: str) -> str:
        """Generate a mock explanation of grammar rules for the text."""
        # List of possible grammar explanations
        explanations = [
            "This sentence uses subject-verb agreement, where singular subjects need singular verbs.",
            "Proper pronoun case is important. Subjective pronouns (I, he, she) are used as subjects.",
            "Apostrophes show possession ('s) or contraction (it's = it is). 'Its' is possessive without apostrophe.",
            "Homonyms are words that sound alike but have different spellings and meanings (there/their/they're).",
            "Past participles like 'seen' require a helping verb, as in 'I have seen' instead of 'I seen'.",
            "The present tense third person singular (he/she/it) requires an -s on the verb.",
            "Collective nouns like 'committee' or 'team' typically take singular verbs.",
            "'Fewer' is used for countable items, while 'less' is used for uncountable quantities.",
            "Pronouns should agree with their antecedents in number and gender."
        ]
        
        return random.choice(explanations)


def test_grammar_model():
    """Test the grammar model with examples."""
    model = MockGrammarModel()
    
    # Test with examples from our dataset
    test_examples = CORRECTION_EXAMPLES + [
        {"incorrect": "I dont know where is the library.", "correct": "I don't know where the library is."},
        {"incorrect": "She speak English very good.", "correct": "She speaks English very well."},
        {"incorrect": "We was waiting for you.", "correct": "We were waiting for you."},
        {"incorrect": "The committee have decided.", "correct": "The committee has decided."}
    ]
    
    # Select a random subset of examples
    samples = random.sample(test_examples, min(8, len(test_examples)))
    
    correct_count = 0
    results = []
    
    print("\n==== GRAMMAR CORRECTION DEMONSTRATION ====\n")
    
    for i, example in enumerate(samples, 1):
        incorrect = example["incorrect"]
        expected = example["correct"]
        
        result = model.correct_grammar(incorrect)
        corrected = result["corrected"]
        confidence = result["confidence"]
        
        is_correct = (corrected.lower() == expected.lower())
        if is_correct:
            correct_count += 1
            
        explanation = model.get_grammar_explanation(incorrect) if random.random() < 0.7 else ""
        
        print(f"Example {i}:")
        print(f"  Original:   {incorrect}")
        print(f"  Corrected:  {corrected}")
        print(f"  Expected:   {expected}")
        print(f"  Confidence: {confidence:.2f}")
        if explanation:
            print(f"  Rule:       {explanation}")
        print(f"  Correct?:   {'✓' if is_correct else '✗'}")
        print()
        
        results.append({
            "id": i,
            "incorrect": incorrect,
            "corrected": corrected,
            "expected": expected,
            "confidence": confidence,
            "is_correct": is_correct,
            "explanation": explanation
        })
    
    accuracy = correct_count / len(samples)
    print(f"Overall accuracy: {accuracy:.2f} ({correct_count}/{len(samples)})")
    
    # Save results
    output_file = "demo_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "results": results,
            "metrics": {
                "accuracy": accuracy,
                "correct_count": correct_count,
                "total": len(samples)
            }
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    test_grammar_model() 