import sys
import os
import re
import random
from typing import List, Tuple, Optional, Dict, Any

# Add the server/model directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'server', 'model'))

# Import the GrammarProcessor class if nanorag.py is available
try:
    from nanorag import GrammarProcessor
    print("Successfully imported GrammarProcessor")
except ImportError:
    print("Could not import GrammarProcessor. Using a simplified version for testing.")
    
    # Define a simplified GrammarProcessor for testing
    class GrammarProcessor:
        def __init__(self):
            self.grammar_rules = self._load_grammar_rules()
            self.grammar_corrections = self._load_grammar_corrections()
            self.advanced_grammar = self._load_advanced_grammar()
            print("Grammar processor initialized with rules and corrections.")
        
        def _load_grammar_rules(self) -> Dict[str, str]:
            grammar_rules = {
                "subject verb agreement": "Subjects and verbs must agree in number. Singular subjects take singular verbs, plural subjects take plural verbs."
            }
            
            try:
                grammar_file = os.path.join(os.path.dirname(__file__), 'server', 'model', 'data', 'grammar', 'english_grammar.txt')
                if os.path.exists(grammar_file):
                    with open(grammar_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                key = line.split('.')[0].lower() if '.' in line else line.lower()
                                grammar_rules[key] = line
                    print(f"Loaded {len(grammar_rules)} grammar rules from file")
            except Exception as e:
                print(f"Error loading grammar rules: {e}")
            
            return grammar_rules
        
        def _load_grammar_corrections(self) -> Dict[str, str]:
            corrections = {
                "she don't like": "she doesn't like"
            }
            
            try:
                corrections_file = os.path.join(os.path.dirname(__file__), 'server', 'model', 'data', 'grammar', 'grammar_corrections.txt')
                if os.path.exists(corrections_file):
                    with open(corrections_file, 'r', encoding='utf-8') as f:
                        current_incorrect = None
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            
                            if line.startswith("INCORRECT:"):
                                current_incorrect = line.replace("INCORRECT:", "").strip().lower()
                            elif line.startswith("CORRECT:") and current_incorrect:
                                correct = line.replace("CORRECT:", "").strip()
                                corrections[current_incorrect] = correct
                                current_incorrect = None
                    print(f"Loaded {len(corrections)} grammar corrections from file")
            except Exception as e:
                print(f"Error loading grammar corrections: {e}")
            
            return corrections
        
        def _load_advanced_grammar(self) -> Dict[str, Dict[str, str]]:
            advanced_grammar = {
                "subjunctive mood": {
                    "example": "If I were president, I would lower taxes.",
                    "explanation": "'Were' is used instead of 'was' in the subjunctive mood, even with singular subjects."
                }
            }
            
            try:
                advanced_file = os.path.join(os.path.dirname(__file__), 'server', 'model', 'data', 'grammar', 'advanced_grammar.txt')
                if os.path.exists(advanced_file):
                    with open(advanced_file, 'r', encoding='utf-8') as f:
                        current_section = ""
                        examples = []
                        explanations = []
                        
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            
                            if line.startswith("##"):
                                if current_section and examples and explanations:
                                    advanced_grammar[current_section.lower()] = {
                                        "example": examples[0] if examples else "",
                                        "explanation": explanations[0] if explanations else ""
                                    }
                                current_section = line.lstrip("#").strip()
                                examples = []
                                explanations = []
                            elif line.startswith("EXAMPLE:"):
                                examples.append(line.replace("EXAMPLE:", "").strip())
                            elif line.startswith("EXPLANATION:"):
                                explanations.append(line.replace("EXPLANATION:", "").strip())
                    
                    # Add the last section
                    if current_section and examples and explanations:
                        advanced_grammar[current_section.lower()] = {
                            "example": examples[0] if examples else "",
                            "explanation": explanations[0] if explanations else ""
                        }
                    print(f"Loaded {len(advanced_grammar)} advanced grammar structures from file")
            except Exception as e:
                print(f"Error loading advanced grammar: {e}")
            
            return advanced_grammar
        
        def is_grammar_question(self, text: str) -> bool:
            grammar_keywords = [
                'grammar', 'grammatical', 'sentence', 'verb', 'noun', 'pronoun', 
                'adjective', 'adverb', 'preposition', 'conjunction', 'plural', 
                'singular', 'tense', 'past tense', 'present tense', 'future tense',
                'correct', 'incorrect', 'wrong', 'proper', 'improper', 'fix'
            ]
            
            text_lower = text.lower()
            
            # Check for grammar keywords
            if any(keyword in text_lower for keyword in grammar_keywords):
                return True
                
            # Check for grammar question patterns
            grammar_patterns = [
                r'is this (sentence|grammar|phrase) correct',
                r'(check|correct|fix) (my|this) (grammar|sentence)',
                r'(how|what) (do|should) I (say|write)',
                r'(is it|should it be) ["\'].*["\'] or ["\'].*["\']',
                r'what is the (correct|right) way to'
            ]
            
            for pattern in grammar_patterns:
                if re.search(pattern, text_lower):
                    return True
            
            return False
        
        def process(self, text: str) -> Optional[str]:
            if self.is_grammar_question(text):
                for concept, rule in self.grammar_rules.items():
                    if concept in text.lower():
                        return rule
                
                for incorrect, correct in self.grammar_corrections.items():
                    if incorrect in text.lower():
                        return f"The correct form is '{correct}' instead of '{incorrect}'."
            
            # Test for specific examples
            if "she don't like" in text.lower():
                return "Corrected: She doesn't like.\n\nGrammar rule: Subjects and verbs must agree in number. Singular subjects take singular verbs, plural subjects take plural verbs."
            
            if "subject verb agreement" in text.lower():
                return "Subjects and verbs must agree in number. Singular subjects take singular verbs, plural subjects take plural verbs."
            
            if "subjunctive mood" in text.lower():
                mood_info = self.advanced_grammar.get("subjunctive mood", {})
                return f"{mood_info.get('explanation', '')}\n\nExample: {mood_info.get('example', '')}"
            
            return None

# Sample questions to test
test_questions = [
    "What is subject-verb agreement?",
    "Can you explain the subjunctive mood?",
    "Is this sentence correct: She don't like apples",
    "Check this sentence: me and my friend went to the store",
    "Correct this: I should of done my homework",
    "What is a dangling modifier?",
    "Can you help me with grammar?",
    "How do I use apostrophes correctly?"
]

def test_grammar_processor():
    """Test the GrammarProcessor class with sample questions"""
    processor = GrammarProcessor()
    
    print("\n" + "="*80)
    print("Testing GrammarProcessor with sample questions:")
    print("="*80)
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        answer = processor.process(question)
        if answer:
            print(f"Answer: {answer}")
        else:
            print("No grammar-specific answer found")
        print("-"*80)

if __name__ == "__main__":
    test_grammar_processor() 