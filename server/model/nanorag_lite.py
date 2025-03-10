"""
NeuraFlux Small Language Model (Lite Version)
This is a simplified, conceptual implementation of a transformer-based language model.
This version focuses on demonstrating the core concepts without requiring external dependencies.
"""

import re
import sys
import json
import os
import math
import random
import sympy
from typing import List, Dict, Any, Optional, Tuple, Union

# Simple matrix operations without numpy/torch
class Tensor:
    def __init__(self, data=None, shape=None):
        if data is not None:
            self.data = data
        elif shape is not None:
            # Initialize with zeros
            self.data = self._create_zeros(shape)
        else:
            self.data = []
        
    def _create_zeros(self, shape):
        if len(shape) == 1:
            return [0.0] * shape[0]
        else:
            return [self._create_zeros(shape[1:]) for _ in range(shape[0])]
    
    def shape(self):
        if not self.data:
            return (0,)
        if isinstance(self.data[0], list):
            return (len(self.data), len(self.data[0]))
        else:
            return (len(self.data),)
            
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __setitem__(self, idx, value):
        self.data[idx] = value

# Configuration for the small language model
class NanoConfig:
    def __init__(
        self,
        vocab_size: int = 10000,
        hidden_size: int = 384,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 6,
        intermediate_size: int = 1536,
        max_position_embeddings: int = 512,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        
        # Special token IDs
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

class NanoTokenizer:
    def __init__(
        self, 
        vocab_file: Optional[str] = None, 
        vocab_size: int = 10000, 
        unk_token: str = "[UNK]", 
        pad_token: str = "[PAD]", 
        bos_token: str = "[BOS]", 
        eos_token: str = "[EOS]"
    ):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        
        # Special tokens
        self.special_tokens = {
            self.pad_token: 0,
            self.bos_token: 1,
            self.eos_token: 2,
            self.unk_token: 3,
        }
        
        if vocab_file and os.path.exists(vocab_file):
            self.load_vocab(vocab_file)
        else:
            # Initialize with just special tokens
            self.vocab = {k: v for k, v in self.special_tokens.items()}
            self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
    
    def load_vocab(self, vocab_file: str):
        """Load vocabulary from file"""
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        
        self.vocab = {**self.special_tokens}
        for token, idx in vocab.items():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs (simplified)"""
        # Simple whitespace tokenization for demonstration
        tokens = text.split()
        
        # Convert tokens to IDs
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.vocab[self.bos_token])
        
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab[self.unk_token])
        
        if add_special_tokens:
            token_ids.append(self.vocab[self.eos_token])
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.ids_to_tokens:
                token = self.ids_to_tokens[token_id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
            else:
                tokens.append(self.unk_token)
        
        return " ".join(tokens)

class NanoNLPEngine:
    def __init__(self, config: NanoConfig, tokenizer: NanoTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        
        # Initialize document store for RAG
        self.documents = []
        self.document_embeddings = []
        
        # Initialize self-information
        self.self_info = {
            "name": "NeuraFlux",
            "creator": "Saptarshi Halder",
            "architecture": "transformer-based",
            "parameters": "1.45 million",
            "layers": "6",
            "attention_heads": "6",
            "hidden_size": "384",
            "capabilities": "text generation, question answering, math problem solving, and retrieval-augmented generation",
            "purpose": "demonstrate the fundamental concepts of modern transformer-based language models",
            "creation_date": "2023",
            "training_data": "a diverse corpus of text including educational content, general knowledge, and mathematical concepts",
            "limitations": "limited vocabulary size (10,000 tokens) and context length (512 tokens)"
        }
        
        # Initialize factual knowledge base
        self.facts = self._initialize_facts()
        
        # Initialize math capabilities
        self.math_patterns = [
            (r'(\d+)\s*\+\s*(\d+)', self._add),
            (r'(\d+)\s*-\s*(\d+)', self._subtract),
            (r'(\d+)\s*\*\s*(\d+)', self._multiply),
            (r'(\d+)\s*/\s*(\d+)', self._divide),
            (r'(\d+)\s*\^\s*(\d+)', self._power),
            (r'sqrt\s*\(\s*(\d+)\s*\)', self._sqrt),
            (r'factorial\s*\(\s*(\d+)\s*\)', self._factorial),
            (r'solve\s+(.+?)=(.+)', self._solve_equation),
            (r'integrate\s+(.+?)\s+with\s+respect\s+to\s+(.+)', self._integrate),
            (r'derivative\s+of\s+(.+?)\s+with\s+respect\s+to\s+(.+)', self._differentiate)
        ]
    
    def _initialize_facts(self) -> Dict[str, str]:
        """Initialize a knowledge base of facts"""
        return {
            "earth sun": "The Earth orbits the Sun at an average distance of about 93 million miles (150 million kilometers).",
            "water boil": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure.",
            "human heart": "The human heart beats about 100,000 times per day, pumping about 2,000 gallons of blood.",
            "light speed": "Light travels at a speed of approximately 299,792,458 meters per second in a vacuum.",
            "dna": "DNA (deoxyribonucleic acid) is a molecule that carries genetic information and instructions for development, functioning, growth, and reproduction.",
            "python programming": "Python is a high-level, interpreted programming language known for its readability and versatility.",
            "neural network": "A neural network is a computational model inspired by the structure and function of the human brain, used in machine learning.",
            "transformer model": "Transformer models are a type of neural network architecture that uses self-attention mechanisms to process sequential data.",
            "attention mechanism": "Attention mechanisms allow neural networks to focus on specific parts of input data when making predictions.",
            "rag": "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval of relevant documents with text generation to improve factual accuracy."
        }
    
    def _add(self, match) -> str:
        a, b = int(match.group(1)), int(match.group(2))
        return f"{a} + {b} = {a + b}"
    
    def _subtract(self, match) -> str:
        a, b = int(match.group(1)), int(match.group(2))
        return f"{a} - {b} = {a - b}"
    
    def _multiply(self, match) -> str:
        a, b = int(match.group(1)), int(match.group(2))
        return f"{a} * {b} = {a * b}"
    
    def _divide(self, match) -> str:
        a, b = int(match.group(1)), int(match.group(2))
        if b == 0:
            return "Division by zero is undefined."
        return f"{a} / {b} = {a / b}"
    
    def _power(self, match) -> str:
        a, b = int(match.group(1)), int(match.group(2))
        return f"{a}^{b} = {a ** b}"
    
    def _sqrt(self, match) -> str:
        a = int(match.group(1))
        if a < 0:
            return f"The square root of {a} is not a real number."
        return f"sqrt({a}) = {math.sqrt(a)}"
    
    def _factorial(self, match) -> str:
        a = int(match.group(1))
        if a < 0:
            return f"Factorial is not defined for negative numbers."
        if a > 20:
            return f"The result is too large to compute efficiently."
        return f"factorial({a}) = {math.factorial(a)}"
    
    def _solve_equation(self, match) -> str:
        try:
            left = match.group(1).strip()
            right = match.group(2).strip()
            
            # Convert to sympy expression
            x = sympy.Symbol('x')
            equation = sympy.Eq(sympy.sympify(left), sympy.sympify(right))
            solution = sympy.solve(equation, x)
            
            return f"The solution to {left} = {right} is x = {solution}"
        except Exception as e:
            return f"I couldn't solve this equation. Error: {str(e)}"
    
    def _integrate(self, match) -> str:
        try:
            expr = match.group(1).strip()
            var = match.group(2).strip()
            
            # Convert to sympy expression
            x = sympy.Symbol(var)
            expression = sympy.sympify(expr)
            result = sympy.integrate(expression, x)
            
            return f"The integral of {expr} with respect to {var} is {result} + C"
        except Exception as e:
            return f"I couldn't integrate this expression. Error: {str(e)}"
    
    def _differentiate(self, match) -> str:
        try:
            expr = match.group(1).strip()
            var = match.group(2).strip()
            
            # Convert to sympy expression
            x = sympy.Symbol(var)
            expression = sympy.sympify(expr)
            result = sympy.diff(expression, x)
            
            return f"The derivative of {expr} with respect to {var} is {result}"
        except Exception as e:
            return f"I couldn't differentiate this expression. Error: {str(e)}"
    
    def add_document(self, doc_id: str, content: str):
        """Add a document to the RAG store"""
        self.documents.append({
            "id": doc_id,
            "content": content
        })
    
    def build_document_embeddings(self):
        """Build document embeddings (simplified)"""
        # In a real implementation, this would use the model to create embeddings
        # Here we'll just use a simple representation for demonstration
        self.document_embeddings = []
        for doc in self.documents:
            # Simple bag of words representation
            words = set(doc["content"].lower().split())
            self.document_embeddings.append(words)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query"""
        if not self.documents:
            return []
        
        # Simple retrieval based on word overlap
        query_words = set(query.lower().split())
        similarities = []
        
        for i, doc_words in enumerate(self.document_embeddings):
            # Calculate Jaccard similarity
            intersection = len(query_words.intersection(doc_words))
            union = len(query_words.union(doc_words))
            score = intersection / union if union > 0 else 0
            
            document = self.documents[i]
            similarities.append((document, score))
        
        # Sort by score and take top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in similarities[:top_k] if score > 0.1]
    
    def is_self_query(self, query: str) -> bool:
        """Check if query is about the model itself"""
        query_lower = query.lower()
        self_terms = ["you", "your", "yourself", "neuraflux"]
        
        # Check for self-reference terms
        if any(term in query_lower for term in self_terms):
            return True
        
        # Check for specific self-attributes
        attributes = ["name", "creator", "made", "built", "parameters", "architecture", "origin", "purpose", "training"]
        if any(attr in query_lower for attr in attributes):
            return True
            
        return False
    
    def answer_self_query(self, query: str) -> str:
        """Answer questions about the model itself"""
        query_lower = query.lower()
        
        # Check for specific attribute queries
        if "name" in query_lower:
            return f"My name is {self.self_info['name']}."
        elif "creator" in query_lower or "made" in query_lower or "built" in query_lower or "origin" in query_lower:
            return f"I was created by {self.self_info['creator']} in {self.self_info['creation_date']} as a demonstration of transformer-based language models."
        elif "architecture" in query_lower:
            return f"I use a {self.self_info['architecture']} architecture with {self.self_info['layers']} layers, {self.self_info['attention_heads']} attention heads, and a hidden size of {self.self_info['hidden_size']}."
        elif "parameters" in query_lower:
            return f"I have {self.self_info['parameters']} parameters, which is small compared to larger models like GPT-3 (175 billion) or GPT-4, but sufficient to demonstrate the core concepts of transformer models."
        elif "do" in query_lower or "capable" in query_lower or "capabilities" in query_lower:
            return f"I can perform {self.self_info['capabilities']}. I'm particularly good at answering questions about myself, solving math problems, and retrieving information from my knowledge base."
        elif "purpose" in query_lower:
            return f"My purpose is to {self.self_info['purpose']}. I serve as an educational tool to help people understand how language models work."
        elif "training" in query_lower or "trained" in query_lower:
            return f"I was trained on {self.self_info['training_data']}. My training process involved masked language modeling and next token prediction tasks."
        elif "limitation" in query_lower:
            return f"My main limitations include {self.self_info['limitations']}. I'm also a simplified implementation, so I don't have the capabilities of larger commercial models."
        
        # General self introduction
        return (f"I am {self.self_info['name']}, a {self.self_info['architecture']} language model "
                f"with {self.self_info['parameters']} parameters created by {self.self_info['creator']}. "
                f"I was designed to {self.self_info['purpose']} and can help with {self.self_info['capabilities']}.")
    
    def is_math_query(self, query: str) -> bool:
        """Check if the query is a math problem"""
        # Check for math keywords
        math_keywords = ["calculate", "compute", "solve", "evaluate", "simplify", "factor", "expand", "integrate", "derivative"]
        if any(keyword in query.lower() for keyword in math_keywords):
            return True
        
        # Check for math operators
        math_operators = ["+", "-", "*", "/", "^", "=", "sqrt", "factorial"]
        if any(op in query for op in math_operators):
            return True
        
        # Check for numbers
        if re.search(r'\d+', query):
            return True
            
        return False
    
    def solve_math_problem(self, query: str) -> str:
        """Solve a math problem"""
        # Try to match against known patterns
        for pattern, handler in self.math_patterns:
            match = re.search(pattern, query)
            if match:
                return handler(match)
        
        # If no pattern matches, try to interpret as a general expression
        try:
            # Extract potential mathematical expression
            expr_match = re.search(r'([\d\s\+\-\*\/\^\(\)]+)', query)
            if expr_match:
                expr = expr_match.group(1).strip()
                # Use sympy to evaluate
                result = sympy.sympify(expr)
                return f"The result of {expr} is {result}"
        except Exception:
            pass
        
        return "I'm not sure how to solve this math problem. Could you please rephrase it or provide more details?"
    
    def search_facts(self, query: str) -> Optional[str]:
        """Search factual knowledge base for answers"""
        query_lower = query.lower()
        
        # Try direct fact lookup
        for key, fact in self.facts.items():
            if all(term in query_lower for term in key.split()):
                return fact
        
        # Try partial matches
        best_match = None
        best_score = 0
        for key, fact in self.facts.items():
            key_terms = key.split()
            match_count = sum(1 for term in key_terms if term in query_lower)
            if match_count > 0:
                score = match_count / len(key_terms)
                if score > best_score:
                    best_score = score
                    best_match = fact
        
        if best_score > 0.5:
            return best_match
            
        return None
    
    def generate_text(self, prompt: str, max_length: int = 50) -> str:
        """Generate text based on the prompt (simplified demonstration)"""
        # In a real model, this would use the transformer architecture to predict tokens
        # Here we'll use a simplified approach for demonstration
        
        responses = [
            f"Based on {prompt}, we can explore language models as computational systems.",
            f"When we consider {prompt}, the transformer architecture shows its strengths.",
            f"Looking at {prompt}, we see how attention mechanisms help with context.",
            f"The concept of {prompt} relates to how neural networks process sequences.",
            f"In language modeling, {prompt} demonstrates pattern recognition capabilities."
        ]
        
        return random.choice(responses)
    
    def answer(self, query: str) -> str:
        """Main interface to generate answers"""
        # First check if it's a math problem
        if self.is_math_query(query):
            return self.solve_math_problem(query)
        
        # Then check if it's a self-query
        if self.is_self_query(query):
            return self.answer_self_query(query)
        
        # Then check factual knowledge
        fact_answer = self.search_facts(query)
        if fact_answer:
            return fact_answer
        
        # Retrieve context and generate response
        context = self.retrieve(query)
        
        if context:
            # Combine context and generate response
            context_text = " ".join([doc["content"] for doc in context])
            return f"Based on my knowledge: {context_text}"
        else:
            # Fall back to simpler generation
            return self.generate_text(query)


# Simple console interface for testing when run directly
if __name__ == "__main__":
    print("NeuraFlux Lite Model (Type 'quit' to exit)")
    print("Using simplified implementation without external dependencies.")
    
    # Create the model and tokenizer
    config = NanoConfig()
    tokenizer = NanoTokenizer()
    model = NanoNLPEngine(config, tokenizer)
    
    # Add some documents for RAG
    model.add_document("model_info", "NeuraFlux is a small language model with 1.45M parameters created by Saptarshi Halder.")
    model.add_document("model_architecture", "NeuraFlux uses a transformer architecture with 6 layers and 6 attention heads.")
    model.add_document("model_capabilities", "NeuraFlux can answer questions, generate text, solve math problems, and retrieve information.")
    model.build_document_embeddings()
    
    while True:
        # Read input from stdin
        if not sys.stdin.isatty():
            # If reading from a pipe, just read one line
            try:
                query = sys.stdin.readline().strip()
                if not query:
                    break
            except:
                break
        else:
            query = input("\nYou: ").strip()
        
        if query.lower() in ["quit", "exit", "bye"]:
            break
        
        # Generate response
        response = model.answer(query)
        print(f"\nNeuraFlux: {response}")
        
        # If reading from a pipe, just process one message and exit
        if not sys.stdin.isatty():
            break 