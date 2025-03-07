import re
import sys
import json
import os
from typing import List, Tuple, Optional

class QASystem:
    def __init__(self):
        # Core metadata about the model
        self.metadata = {
            "name": "NeuraFlex",
            "creator": "Saptarshi Halder",
            "architecture": "Hybrid Transformer-CNN with RAG",
            "parameter_count": "1.45M",
            "training_data": ["SQuAD", "TinyStories", "Self-QA"],
            "purpose": "Answer questions about itself and general knowledge",
            "version": "v1.0 (2024-03)",
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
            "tell me about yourself": f"I'm {self.metadata['name']}, an AI assistant built using {self.metadata['architecture']} technology. I can help with various tasks including answering questions and solving math problems.",
            "describe yourself": f"I'm {self.metadata['name']}, an AI model with {self.metadata['parameter_count']} parameters, trained to help with questions and calculations.",

            # Creator variations
            "who created you": f"I was created by {self.metadata['creator']}",
            "who made you": f"I was made by {self.metadata['creator']}",
            "who built you": f"I was built by {self.metadata['creator']}",
            "who developed you": f"I was developed by {self.metadata['creator']}",
            "your creator": f"{self.metadata['creator']} created me",

            # Capability variations
            "what can you do": self.metadata['purpose'],
            "what are your capabilities": self.metadata['purpose'],
            "tell me what you can do": self.metadata['purpose'],
            "how can you help me": self.metadata['purpose'],

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

            # Limitations variations
            "what are your limitations": self.metadata['limitations'],
            "what can't you do": self.metadata['limitations'],
            "tell me your limitations": self.metadata['limitations'],

            # Version variations
            "what version are you": self.metadata['version'],
            "which version are you running": self.metadata['version'],
            "tell me your version": f"I'm running version {self.metadata['version']}"
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
        key_terms = ['name', 'creator', 'built', 'parameters', 'version', 'trained', 'work', 'do']
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

class NeuraFlex:
    def __init__(self):
        self.math_processor = MathProcessor()
        self.qa_system = QASystem()

    def process_query(self, text: str) -> str:
        """Process a query and return a response."""
        text = text.strip()

        # First try math processing
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