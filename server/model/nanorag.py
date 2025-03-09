import re
import sys
import json
import os
import random
from typing import List, Tuple, Optional, Dict, Any

class QASystem:
    def __init__(self):
        # Core metadata about the model
        self.metadata = {
            "name": "NeuraFlux",
            "creator": "Saptarshi Halder",
            "architecture": "Hybrid Transformer-CNN with RAG",
            "parameter_count": "1.45M",
            "training_data": ["SQuAD", "TinyStories", "Self-QA", "English Grammar Dataset"],
            "purpose": "Answer questions about itself, general knowledge, math calculations, and grammar corrections",
            "version": "v1.1 (2024-03)",
            "limitations": "Cannot access real-time data post-training"
        }

        # Self-referential QA pairs with variations
        self.self_qa = {
            # Greeting variations
            "hi": f"Hello! I'm {self.metadata['name']}. How can I help you today?",
            "hello": f"Hi! I'm {self.metadata['name']}. What can I help you with?",
            "hey": f"Hey there! I'm {self.metadata['name']}. How can I assist you?",
            "greetings": f"Greetings! I'm {self.metadata['name']}, ready to help you.",

            # Name variations
            "what is your name": self.metadata['name'],
            "what should i call you": self.metadata['name'],
            "do you have a name": f"Yes, my name is {self.metadata['name']}",
            "tell me your name": f"My name is {self.metadata['name']}",
            "whats your name": self.metadata['name'],
            "name please": f"I'm {self.metadata['name']}",

            # Identity variations
            "who are you": f"I am {self.metadata['name']}, a {self.metadata['architecture']} language model.",
            "what are you": f"I'm {self.metadata['name']}, an AI language model that uses {self.metadata['architecture']} technology.",
            "introduce yourself": f"Hello! I'm {self.metadata['name']}, a {self.metadata['architecture']} language model.",
            "tell me about yourself": f"I'm {self.metadata['name']}, an AI assistant built using {self.metadata['architecture']} technology. I can help with various tasks including answering questions, solving math problems, and correcting grammar.",
            "describe yourself": f"I'm {self.metadata['name']}, an AI model with {self.metadata['parameter_count']} parameters, trained to help with questions, calculations, and grammar corrections.",

            # Creator variations
            "who created you": f"I was created by {self.metadata['creator']}",
            "who made you": f"I was made by {self.metadata['creator']}",
            "who built you": f"I was built by {self.metadata['creator']}",
            "who developed you": f"I was developed by {self.metadata['creator']}",
            "your creator": f"{self.metadata['creator']} created me",

            # Capability variations
            "what can you do": f"I can {self.metadata['purpose']}",
            "what are your capabilities": f"My capabilities include {self.metadata['purpose']}",
            "tell me what you can do": f"I can {self.metadata['purpose']}",
            "how can you help me": f"I can help by {self.metadata['purpose']}",

            # Architecture variations
            "how do you work": f"I use a {self.metadata['architecture']} to process and respond to questions",
            "how were you built": f"I was built using {self.metadata['architecture']} architecture",
            "what is your architecture": self.metadata['architecture'],
            "what technology do you use": f"I'm built on {self.metadata['architecture']} technology",

            # Parameters variations
            "how many parameters do you have": f"I have {self.metadata['parameter_count']} parameters",
            "what is your parameter count": f"My parameter count is {self.metadata['parameter_count']}",
            "how big are you": f"I'm a {self.metadata['parameter_count']} parameter model",

            # Training variations
            "what were you trained on": "I was trained on " + ", ".join(self.metadata['training_data']),
            "what data were you trained with": "I was trained using " + ", ".join(self.metadata['training_data']),
            "what is your training data": "My training data includes " + ", ".join(self.metadata['training_data']),
            "have you been trained on grammar": "Yes, I've been trained on an English Grammar Dataset that includes grammar rules, corrections, and advanced structures.",

            # Limitations variations
            "what are your limitations": self.metadata['limitations'],
            "what can't you do": self.metadata['limitations'],
            "tell me your limitations": self.metadata['limitations'],

            # Version variations
            "what version are you": self.metadata['version'],
            "which version are you running": self.metadata['version'],
            "tell me your version": f"I'm running version {self.metadata['version']}",
            
            # New grammar capabilities
            "can you help with grammar": "Yes, I can help with grammar corrections, explain grammar rules, and provide examples of proper usage.",
            "can you correct grammar": "Yes, I can identify and correct grammatical errors in text. Just send me a sentence and I'll check it for you.",
            "can you check my grammar": "Yes, I can check your grammar and suggest corrections. Just send me the text you'd like me to review.",
            "what grammar can you help with": "I can help with various grammar topics including subject-verb agreement, pronoun usage, verb tenses, punctuation, and more complex structures like conditionals and subjunctive mood."
        }

        # Basic knowledge QA pairs
        self.qa_pairs = {
            "what is the capital of france": "Paris",
            "who wrote hamlet": "William Shakespeare",
            "what is the largest planet": "Jupiter",
            "who invented the telephone": "Alexander Graham Bell",
            "what is the longest river": "The Nile River",
            "what is the speed of light": "299,792 kilometers per second",
            "who painted the mona lisa": "Leonardo da Vinci",
            "what is the hardest natural substance": "Diamond",
            "what is the chemical symbol for gold": "Au",
            "what is the human body's largest organ": "The skin"
        }

    def normalize_question(self, question: str) -> str:
        """Normalize a question for better matching."""
        # Handle single-word greetings
        question = question.lower().strip()
        if question in ['hi', 'hello', 'hey', 'greetings']:
            return question

        # Remove punctuation and extra whitespace
        question = re.sub(r'[^\w\s]', '', question)

        # Remove common question starters
        question = re.sub(r'^(please |could you |can you |tell me |what |who |how |where |when |why |do |does |did |is |are |was |were )+', '', question)

        # Remove filler words
        filler_words = ['the', 'a', 'an', 'that', 'those', 'these', 'this', 'just', 'maybe', 'perhaps', 'well']
        words = question.split()
        words = [w for w in words if w not in filler_words]

        return ' '.join(words)

    def is_self_referential(self, question: str) -> bool:
        """Check if a question is about the model itself."""
        self_tokens = ['you', 'your', 'yourself', self.metadata['name'].lower()]
        clean_question = question.lower().strip('?!.,')
        return any(token in clean_question for token in self_tokens)

    def similarity_score(self, q1: str, q2: str) -> float:
        """Calculate similarity between two questions using word overlap and key term matching."""
        # Normalize both questions
        q1 = self.normalize_question(q1)
        q2 = self.normalize_question(q2)

        # Split into words
        words1 = set(q1.split())
        words2 = set(q2.split())

        # Calculate word overlap
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        overlap_score = len(intersection) / len(union) if union else 0

        # Boost score if key terms match
        key_terms = ['name', 'creator', 'built', 'parameters', 'version', 'trained', 'work', 'do', 'grammar', 'math']
        key_term_matches = sum(1 for term in key_terms if (term in words1) == (term in words2))
        key_term_score = key_term_matches / len(key_terms)

        # Combine scores with weights
        return (0.7 * overlap_score) + (0.3 * key_term_score)

    def get_answer(self, question: str) -> Optional[str]:
        """Get answer for a question, prioritizing self-reference checks."""
        question = question.lower().strip()
        normalized_question = self.normalize_question(question)

        # Check for self-referential questions first
        if self.is_self_referential(question):
            # Direct match in self-QA
            if question in self.self_qa:
                return self.self_qa[question]

            # Try normalized question
            if normalized_question in self.self_qa:
                return self.self_qa[normalized_question]

            # Try fuzzy match for self-questions
            best_match = None
            highest_score = 0
            for q in self.self_qa:
                score = self.similarity_score(normalized_question, q)
                if score > highest_score:
                    highest_score = score
                    best_match = q

            if highest_score > 0.6:
                return self.self_qa[best_match]

        # Check general knowledge QA
        if question in self.qa_pairs:
            return self.qa_pairs[question]

        if normalized_question in self.qa_pairs:
            return self.qa_pairs[normalized_question]

        # Try fuzzy match for general questions
        best_match = None
        highest_score = 0
        for q in self.qa_pairs:
            score = self.similarity_score(normalized_question, q)
            if score > highest_score:
                highest_score = score
                best_match = q

        if highest_score > 0.6:
            return self.qa_pairs[best_match]

        return None

class MathProcessor:
    def __init__(self):
        self.operators = {
            'plus': '+',
            'add': '+',
            '+': '+',
            'minus': '-',
            'subtract': '-',
            '-': '-',
            'times': '*',
            'multiply': '*',
            'multiplied by': '*',
            '*': '*',
            'x': '*',
            'divided by': '/',
            'divide': '/',
            '/': '/',
            'power': '**',
            '^': '**'
        }

    def extract_numbers_and_operators(self, text: str) -> Tuple[List[float], List[str]]:
        numbers = [float(x) for x in re.findall(r'-?\d*\.?\d+', text)]
        text = text.lower()
        operator = '+'

        for word, op in self.operators.items():
            if word in text:
                operator = op
                break

        if '+' in text or '-' in text or '*' in text or '/' in text:
            try:
                clean_expr = ''.join(text.split())
                result = eval(clean_expr)
                return [result], ['+']
            except:
                pass

        return numbers, [operator]

    def solve(self, text: str) -> Optional[str]:
        try:
            numbers, operators = self.extract_numbers_and_operators(text)

            if not numbers:
                return None

            if len(numbers) == 1 and operators[0] == '+':
                return f"The answer is {numbers[0]:,}"

            result = numbers[0]
            operator = operators[0]

            for num in numbers[1:]:
                if operator == '+':
                    result += num
                elif operator == '-':
                    result -= num
                elif operator == '*':
                    result *= num
                elif operator == '/':
                    if num == 0:
                        return "Cannot divide by zero"
                    result /= num
                elif operator == '**':
                    result = pow(result, num)

            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return f"The answer is {result:,}"

        except Exception as e:
            return None

class GrammarProcessor:
    def __init__(self):
        # Load grammar data from our training files
        self.grammar_rules = self._load_grammar_rules()
        self.grammar_corrections = self._load_grammar_corrections()
        self.advanced_grammar = self._load_advanced_grammar()
        print("Grammar processor initialized with rules and corrections.", file=sys.stderr)
    
    def _load_grammar_rules(self) -> Dict[str, str]:
        """Load basic grammar rules from our training data."""
        grammar_rules = {
            "subject verb agreement": "Subjects and verbs must agree in number. Singular subjects take singular verbs, plural subjects take plural verbs.",
            "pronoun case": "Use subject pronouns (I, he, she) as subjects and object pronouns (me, him, her) as objects.",
            "apostrophes": "Apostrophes show possession ('s) or contraction (it's = it is). 'Its' without apostrophe is possessive.",
            "verb tense": "Be consistent with verb tenses within a sentence or paragraph unless there's a logical reason to switch.",
            "comma splice": "Don't join independent clauses with just a comma; use a semicolon, conjunction, or separate them into two sentences.",
            "parallel structure": "Use the same grammatical form for elements in a series or comparison.",
            "dangling modifiers": "Make sure modifiers clearly refer to the intended subject.",
            "run-on sentences": "Avoid joining independent clauses without proper punctuation or conjunctions.",
            "prepositions": "End sentences with prepositions when it sounds natural, despite outdated rules against it.",
            "adjective order": "Multiple adjectives generally follow this order: opinion, size, age, shape, color, origin, material, purpose."
        }
        
        try:
            # Try to load from actual file if available
            grammar_file = os.path.join(os.path.dirname(__file__), "../data/grammar/english_grammar.txt")
            if os.path.exists(grammar_file):
                with open(grammar_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            key = line.split('.')[0].lower() if '.' in line else line.lower()
                            grammar_rules[key] = line
                print(f"Loaded {len(grammar_rules)} grammar rules from file", file=sys.stderr)
        except Exception as e:
            print(f"Error loading grammar rules: {e}", file=sys.stderr)
        
        return grammar_rules
    
    def _load_grammar_corrections(self) -> Dict[str, str]:
        """Load grammar correction examples."""
        corrections = {
            "she don't like": "she doesn't like",
            "me and my friend": "my friend and I",
            "i seen": "I saw",
            "they was": "they were",
            "it's paws": "its paws",
            "more taller": "taller",
            "him and her are": "he and she are",
            "there going to": "they're going to",
            "is laying": "is lying",
            "should of": "should have"
        }
        
        try:
            # Try to load from actual file if available
            corrections_file = os.path.join(os.path.dirname(__file__), "../data/grammar/grammar_corrections.txt")
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
                print(f"Loaded {len(corrections)} grammar corrections from file", file=sys.stderr)
        except Exception as e:
            print(f"Error loading grammar corrections: {e}", file=sys.stderr)
        
        return corrections
    
    def _load_advanced_grammar(self) -> Dict[str, Dict[str, str]]:
        """Load advanced grammar examples and explanations."""
        advanced_grammar = {
            "subjunctive mood": {
                "example": "If I were president, I would lower taxes.",
                "explanation": "'Were' is used instead of 'was' in the subjunctive mood, even with singular subjects."
            },
            "inversion": {
                "example": "Never have I seen such beauty.",
                "explanation": "The auxiliary verb 'have' comes before the subject 'I' due to the initial negative adverb."
            },
            "cleft sentences": {
                "example": "It was Jane who found the missing documents.",
                "explanation": "This structure emphasizes that Jane (not someone else) found the documents."
            },
            "participial phrases": {
                "example": "Walking along the beach, we saw a pod of dolphins.",
                "explanation": "The present participle phrase 'Walking along the beach' modifies 'we'."
            },
            "mixed conditionals": {
                "example": "If I had studied medicine, I would be a doctor now.",
                "explanation": "The 'if' clause refers to the past, while the main clause refers to the present."
            }
        }
        
        try:
            # Try to load from actual file if available
            advanced_file = os.path.join(os.path.dirname(__file__), "../data/grammar/advanced_grammar.txt")
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
                print(f"Loaded {len(advanced_grammar)} advanced grammar structures from file", file=sys.stderr)
        except Exception as e:
            print(f"Error loading advanced grammar: {e}", file=sys.stderr)
        
        return advanced_grammar
    
    def is_grammar_question(self, text: str) -> bool:
        """Determine if a question is about grammar."""
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
    
    def is_correction_query(self, text: str) -> bool:
        """Determine if the query is asking for a grammar correction."""
        correction_patterns = [
            r'(correct|fix|improve|check) (my|this) (grammar|sentence|paragraph)',
            r'is this (grammatically correct|proper grammar)',
            r'(how|what) (should|would|could) I (say|write) this',
            r'is there (a|any) (grammar|grammatical) (error|mistake|problem)',
            r'does this (sentence|phrase) sound (right|correct|proper)',
            r'(is|are) there (any|some) (grammar|grammatical) (issue|problem|error)'
        ]
        
        text_lower = text.lower()
        
        for pattern in correction_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def find_grammar_errors(self, text: str) -> List[Tuple[str, str]]:
        """Find grammar errors in the given text."""
        errors = []
        
        # Check the whole text first
        text_lower = text.lower()
        for incorrect, correct in self.grammar_corrections.items():
            if incorrect in text_lower:
                errors.append((incorrect, correct))
        
        # If no exact matches, try partial matches
        if not errors:
            # Split the text into words
            words = text_lower.split()
            word_pairs = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
            word_triplets = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            
            # Check each word pair and triplet against our corrections
            for phrase in word_pairs + word_triplets:
                for incorrect, correct in self.grammar_corrections.items():
                    if phrase in incorrect or incorrect in phrase:
                        # Calculate similarity
                        similarity = len(set(phrase.split()) & set(incorrect.split())) / len(set(phrase.split()) | set(incorrect.split()))
                        if similarity > 0.5:  # Threshold for similarity
                            errors.append((incorrect, correct))
        
        return errors
    
    def correct_grammar(self, text: str) -> str:
        """Generate a grammar correction response for the given text."""
        if self.is_correction_query(text):
            # Extract the actual text to check (after phrases like "correct this sentence:")
            match = re.search(r'(correct|fix|improve|check) (my|this) (grammar|sentence|paragraph)[:\s]+(.+)', text.lower())
            if match:
                text_to_check = match.group(4)
            else:
                text_to_check = text
        else:
            text_to_check = text
        
        errors = self.find_grammar_errors(text_to_check)
        
        if errors:
            # Apply corrections
            corrected_text = text_to_check
            for incorrect, correct in errors:
                corrected_text = corrected_text.lower().replace(incorrect, correct)
            
            # Capitalize first letter and make sure it ends with punctuation
            corrected_text = corrected_text[0].upper() + corrected_text[1:]
            if not corrected_text[-1] in '.!?':
                corrected_text += '.'
            
            # Choose a random grammar rule to explain
            rule_explanation = ""
            error_key = errors[0][0].split()[0] if len(errors[0][0].split()) > 0 else errors[0][0]
            for rule_key, rule_text in self.grammar_rules.items():
                if error_key in rule_key or rule_key in error_key:
                    rule_explanation = f"\n\nGrammar rule: {rule_text}"
                    break
            
            return f"Corrected: {corrected_text}{rule_explanation}"
        
        # If no errors found, check for other grammar issues
        common_errors = [
            "subject-verb agreement",
            "pronoun usage",
            "apostrophe misuse",
            "run-on sentences",
            "comma splices",
            "dangling modifiers"
        ]
        
        for error in common_errors:
            if error in text_to_check.lower():
                rule = next((rule for rule_key, rule in self.grammar_rules.items() if error in rule_key), None)
                if rule:
                    return f"The text seems correct, but I noticed you mentioned {error}. Here's a tip: {rule}"
        
        # If it's a correction query but no errors found
        if self.is_correction_query(text):
            return "The grammar looks correct! I don't see any errors to fix."
        
        # No errors, not a correction query - return None to let other processors handle it
        return None
    
    def answer_grammar_question(self, question: str) -> str:
        """Answer a grammar-related question."""
        question_lower = question.lower()
        
        # Check for questions about specific grammar concepts
        for concept, rule in self.grammar_rules.items():
            if concept in question_lower:
                return rule
        
        # Check for questions about advanced grammar
        for structure, details in self.advanced_grammar.items():
            if structure in question_lower:
                return f"{details['explanation']}\n\nExample: {details['example']}"
        
        # Check for specific correction questions
        for incorrect, correct in self.grammar_corrections.items():
            if incorrect in question_lower:
                return f"The correct form is '{correct}' instead of '{incorrect}'."
        
        # General grammar questions
        if "what is grammar" in question_lower:
            return "Grammar is the set of rules that explain how words are used in a language. It includes the structure of words, phrases, clauses, and sentences."
        
        if "grammar rules" in question_lower or "basic grammar" in question_lower:
            rules = list(self.grammar_rules.values())
            sampled_rules = random.sample(rules, min(3, len(rules)))
            return "Here are some important grammar rules:\n\n" + "\n\n".join(sampled_rules)
        
        if "advanced grammar" in question_lower:
            structures = random.sample(list(self.advanced_grammar.keys()), min(2, len(self.advanced_grammar)))
            response = "Here are some advanced grammar structures:\n\n"
            for structure in structures:
                details = self.advanced_grammar[structure]
                response += f"{structure.title()}: {details['explanation']}\nExample: {details['example']}\n\n"
            return response
        
        # Default response for grammar questions
        return "Grammar involves rules for how words are used in a language. I can help with specific grammar questions, check for errors, or explain grammar rules and concepts."
    
    def process(self, text: str) -> Optional[str]:
        """Process grammar-related queries."""
        if self.is_grammar_question(text):
            return self.answer_grammar_question(text)
        
        if self.is_correction_query(text) or any(incorrect in text.lower() for incorrect in self.grammar_corrections.keys()):
            return self.correct_grammar(text)
        
        return None

class NeuraFlex:
    def __init__(self):
        self.math_processor = MathProcessor()
        self.qa_system = QASystem()
        self.grammar_processor = GrammarProcessor()

    def process_query(self, text: str) -> str:
        """Process a query and return a response."""
        text = text.strip()

        # First try grammar processing for grammar-related questions or corrections
        grammar_answer = self.grammar_processor.process(text)
        if grammar_answer:
            return grammar_answer

        # Then try math processing
        math_answer = self.math_processor.solve(text)
        if math_answer:
            return math_answer

        # Then try QA system
        qa_answer = self.qa_system.get_answer(text)
        if qa_answer:
            return qa_answer

        return "I don't know the answer to that question. Could you try asking something else?"

def main():
    model = NeuraFlex()
    print("NeuraFlex model initialized and ready.", file=sys.stderr)

    for line in sys.stdin:
        input_text = line.strip()
        response = model.process_query(input_text)
        print(response, flush=True)

if __name__ == "__main__":
    main()