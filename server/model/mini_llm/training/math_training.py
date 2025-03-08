#!/usr/bin/env python3
"""
Mathematical problem-solving training module for MiniLLM.

This module extends the knowledge transfer system to specialize in mathematical reasoning
by incorporating specialized loss functions and training data processing for mathematical tasks.
"""

import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import sympy
from sympy.parsing.latex import parse_latex
from sympy.printing.latex import latex

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from training.knowledge_transfer import KnowledgeTransferSystem
from tokenizer.bpe_tokenizer import BPETokenizer

class MathProblem:
    """Represents a mathematical problem with its solution and steps."""
    
    def __init__(self, question: str, solution: str, steps: List[str], latex_expr: str = None):
        self.question = question
        self.solution = solution
        self.steps = steps
        self.latex_expr = latex_expr
        
        # Try to parse latex expression if provided
        if latex_expr:
            try:
                self.sympy_expr = parse_latex(latex_expr)
            except:
                self.sympy_expr = None
        else:
            self.sympy_expr = None

    def __str__(self):
        return f"Question: {self.question}\nSolution: {self.solution}"

class MathTrainer(KnowledgeTransferSystem):
    """Extended knowledge transfer system specialized for mathematical problem-solving."""
    
    def __init__(self, model: Any, tokenizer: Any, output_dir: str = "output/math"):
        super().__init__(model, tokenizer, output_dir)
        
        # Additional attributes for math training
        self.equation_patterns = self.load_equation_patterns()
        self.math_symbols = self.load_math_symbols()
        
        # Create math-specific output directories
        os.makedirs(os.path.join(output_dir, "equation_viz"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "step_by_step"), exist_ok=True)
        
        # Create a mock model if none is provided
        if self.model is None:
            self.model = self._create_mock_model()
            
    def _create_mock_model(self):
        """Create a mock model for training."""
        class MockConfig:
            """Mock configuration for model."""
            def __init__(self):
                self.vocab_size = 30000
                self.hidden_size = 384
                self.num_layers = 6
                self.num_heads = 6
                self.intermediate_size = 1536
                self.max_seq_len = 512
                self.dropout = 0.1
                
        class MockModel:
            """Mock model for training."""
            def __init__(self):
                self.hidden_size = 384
                self.num_layers = 6
                self.config = MockConfig()
                
            def forward(self, tokens):
                """Mock forward pass."""
                # Return random logits
                if isinstance(tokens, list) and isinstance(tokens[0], list):
                    return [np.random.rand(len(t), 384) for t in tokens]
                return np.random.rand(len(tokens), 384)
                
            def backward(self, loss):
                """Mock backward pass."""
                pass
                
            def step(self, learning_rate):
                """Mock optimizer step."""
                pass
                
            def generate(self, prompt, max_length=100):
                """Mock generation function."""
                # For testing purposes
                if "quadratic" in str(prompt).lower():
                    return "x = -2 or x = -3"
                if "derivative" in str(prompt).lower():
                    return "f'(x) = 3x² - 4x + 4"
                if "integral" in str(prompt).lower():
                    return "∫(2x + 3)dx = x² + 3x + C"
                return "The solution is..."
                
            def save(self, path):
                """Mock save function."""
                os.makedirs(path, exist_ok=True)
                with open(os.path.join(path, "model_config.json"), "w") as f:
                    json.dump({
                        "vocab_size": self.config.vocab_size,
                        "hidden_size": self.config.hidden_size,
                        "num_layers": self.config.num_layers,
                        "num_heads": self.config.num_heads,
                        "intermediate_size": self.config.intermediate_size,
                        "max_seq_len": self.config.max_seq_len,
                        "dropout": self.config.dropout
                    }, f, indent=2)
                
                # Mock save some weights
                np.save(os.path.join(path, "weights.npy"), np.random.rand(10, 10))
                
            @staticmethod
            def load(path):
                """Mock load function."""
                return MockModel()
        
        return MockModel()
    
    def load_equation_patterns(self) -> Dict[str, str]:
        """Load common equation patterns and their solutions."""
        return {
            r"solve_quadratic": r"ax^2 + bx + c = 0 -> x = (-b ± √(b^2 - 4ac))/(2a)",
            r"integrate": r"∫f(x)dx -> F(x) + C",
            r"differentiate": r"d/dx[f(x)] -> f'(x)",
            r"matrix_mult": r"AB = [Σ(aik * bkj)]",
            r"vector_dot": r"a·b = Σ(ai * bi)",
        }
    
    def load_math_symbols(self) -> List[str]:
        """Load special mathematical symbols for tokenization."""
        return [
            "∫", "∂", "∑", "∏", "√", "±", "∞", "≠", "≈", "≤", "≥",
            "α", "β", "γ", "θ", "π", "λ", "μ", "σ", "ω",
            "→", "↔", "⇒", "⇔", "∈", "∉", "⊂", "⊃", "∪", "∩",
        ]
    
    def preprocess_math_problem(self, problem: MathProblem) -> Dict[str, Any]:
        """
        Preprocess a mathematical problem for training.
        
        Args:
            problem: MathProblem instance containing question and solution
            
        Returns:
            Dict containing processed features
        """
        # Tokenize question and solution
        question_tokens = self.tokenizer.encode(problem.question)
        solution_tokens = self.tokenizer.encode(problem.solution)
        
        # Process steps if available
        step_tokens = [self.tokenizer.encode(step) for step in problem.steps]
        
        # Extract mathematical expressions if available
        expressions = []
        if problem.latex_expr and problem.sympy_expr:
            try:
                # Evaluate expression
                evaluated = problem.sympy_expr.evalf()
                # Convert back to latex
                latex_result = latex(evaluated)
                expressions.append({
                    "original": problem.latex_expr,
                    "evaluated": str(evaluated),
                    "latex": latex_result
                })
            except:
                pass
        
        return {
            "question_tokens": question_tokens,
            "solution_tokens": solution_tokens,
            "step_tokens": step_tokens,
            "expressions": expressions,
        }
    
    def calculate_math_accuracy(self, predicted: str, actual: str) -> float:
        """
        Calculate accuracy of mathematical solutions.
        
        Args:
            predicted: Predicted solution string
            actual: Actual solution string
            
        Returns:
            float: Accuracy score between 0 and 1
        """
        try:
            # Try to parse both as sympy expressions
            pred_expr = parse_latex(predicted)
            actual_expr = parse_latex(actual)
            
            # Check if expressions are equivalent
            if pred_expr.equals(actual_expr):
                return 1.0
            
            # If not exactly equal, evaluate numerically
            pred_val = float(pred_expr.evalf())
            actual_val = float(actual_expr.evalf())
            
            # Calculate relative error
            error = abs(pred_val - actual_val) / abs(actual_val)
            return max(0, 1 - error)
            
        except:
            # Fallback to string matching if parsing fails
            return float(predicted.strip() == actual.strip())
    
    def generate(self, prompt: str) -> str:
        """Generate a response to a math problem."""
        if self.model:
            return self.model.generate(prompt)
        return "No model available to generate response."
    
    def generate_steps(self, problem: str) -> List[str]:
        """Generate step-by-step solution to a problem."""
        if not self.model:
            return ["No model available to generate steps."]
            
        # Mock implementation for step generation
        if "quadratic" in problem.lower():
            return [
                "1. Identify a, b, and c in ax² + bx + c = 0",
                "2. Use quadratic formula: x = (-b ± √(b² - 4ac))/2a",
                "3. Substitute values and calculate",
                "4. Simplify to get final answers"
            ]
        elif "derivative" in problem.lower():
            return [
                "1. Apply power rule: d/dx[xⁿ] = n·xⁿ⁻¹",
                "2. Apply linearity: d/dx[f(x) + g(x)] = d/dx[f(x)] + d/dx[g(x)]",
                "3. Simplify the resulting expression"
            ]
        elif "integral" in problem.lower():
            return [
                "1. Apply power rule for integration: ∫xⁿ dx = xⁿ⁺¹/(n+1) + C",
                "2. Apply linearity: ∫[f(x) + g(x)] dx = ∫f(x) dx + ∫g(x) dx",
                "3. Substitute bounds if definite integral"
            ]
        else:
            return ["1. First step", "2. Second step", "3. Third step"]
    
    def generate_expression(self, problem: str) -> str:
        """Generate mathematical expression for a problem."""
        if not self.model:
            return ""
            
        # Mock implementation for LaTeX expression generation
        if "quadratic" in problem.lower():
            return "x^2 + 5x + 6 = 0"
        elif "derivative" in problem.lower():
            return "\\frac{d}{dx}(x^3 - 2x^2 + 4x - 1) = 3x^2 - 4x + 4"
        elif "integral" in problem.lower():
            return "\\int_0^2 (2x + 3) dx = [x^2 + 3x]_0^2 = 10"
        else:
            return "f(x) = x"
    
    def train_on_math(
        self,
        problems: List[MathProblem],
        batch_size: int = 4,
        num_epochs: int = 5,
        learning_rate: float = 3e-5,
        step_weight: float = 0.3,
        expression_weight: float = 0.3,
    ) -> None:
        """
        Train the model specifically on mathematical problems.
        
        Args:
            problems: List of MathProblem instances
            batch_size: Training batch size
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            step_weight: Weight for step-by-step explanation loss
            expression_weight: Weight for mathematical expression loss
        """
        print(f"Starting mathematical training with {len(problems)} problems")
        
        # Process all problems
        processed_problems = [self.preprocess_math_problem(p) for p in problems]
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Shuffle problems
            random.shuffle(processed_problems)
            
            # Create batches
            batches = [
                processed_problems[i:i + batch_size]
                for i in range(0, len(processed_problems), batch_size)
            ]
            
            # Train on each batch
            for batch_idx, batch in enumerate(batches):
                batch_loss = 0.0
                
                # Process each problem in batch
                for prob in batch:
                    # Get teacher responses
                    teacher_outputs = []
                    for teacher in self.teachers.values():
                        response = teacher.generate(prob["question_tokens"])
                        teacher_outputs.append(response)
                    
                    # Forward pass through student model
                    student_output = self.model.forward(prob["question_tokens"])
                    
                    # Calculate losses
                    # 1. Main solution loss (KL divergence with teacher outputs)
                    solution_loss = self._calculate_kl_loss(student_output, teacher_outputs)
                    
                    # 2. Step-by-step explanation loss
                    step_loss = 0.0
                    if prob["step_tokens"]:
                        step_outputs = self.model.forward(prob["step_tokens"])
                        step_loss = self._calculate_step_loss(step_outputs, prob["step_tokens"])
                    
                    # 3. Mathematical expression loss
                    expr_loss = 0.0
                    if prob["expressions"]:
                        expr_outputs = self.model.forward([expr["latex"] for expr in prob["expressions"]])
                        expr_loss = self._calculate_expression_loss(expr_outputs, prob["expressions"])
                    
                    # Combine losses
                    total_loss = (
                        solution_loss +
                        step_weight * step_loss +
                        expression_weight * expr_loss
                    )
                    
                    batch_loss += total_loss
                    
                    # Update model
                    self.model.backward(total_loss)
                    self.model.step(learning_rate)
                
                # Log progress
                avg_batch_loss = batch_loss / len(batch)
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}, Average Loss: {avg_batch_loss:.4f}")
            
            # Save checkpoint
            self.save_model(os.path.join(self.output_dir, f"math_checkpoint_{epoch+1}"))
    
    def _calculate_step_loss(self, outputs: Any, step_tokens: List[List[int]]) -> float:
        """Calculate loss for step-by-step explanations."""
        # Mock implementation - replace with actual loss calculation
        return random.uniform(0.1, 0.5)
    
    def _calculate_expression_loss(self, outputs: Any, expressions: List[Dict[str, str]]) -> float:
        """Calculate loss for mathematical expressions."""
        # Mock implementation - replace with actual loss calculation
        return random.uniform(0.1, 0.5)
    
    def _calculate_kl_loss(self, student_output: Any, teacher_outputs: List[Any]) -> float:
        """Calculate KL divergence loss between student and teacher outputs."""
        # Mock implementation - replace with actual KL divergence calculation
        return random.uniform(0.2, 0.8)
    
    def save_model(self, path: str) -> None:
        """Save the model to the specified path."""
        if hasattr(self.model, 'save'):
            print(f"Saving model to {path}")
            self.model.save(path)
        else:
            print(f"Model does not support saving, creating mock checkpoint at {path}")
            os.makedirs(path, exist_ok=True)
            # Save a mock configuration file
            with open(os.path.join(path, "model_config.json"), "w") as f:
                json.dump({
                    "vocab_size": getattr(self.model, "vocab_size", 30000),
                    "hidden_size": getattr(self.model, "hidden_size", 384),
                    "num_layers": getattr(self.model, "num_layers", 6),
                    "is_mock": True
                }, f, indent=2)
            
            # Save tokenizer separately
            tokenizer_path = os.path.join(path, "tokenizer")
            self.tokenizer.save(tokenizer_path)
            
            print(f"Saved mock model checkpoint and tokenizer to {path}")
    
    def evaluate_math(self, test_problems: List[MathProblem]) -> Dict[str, float]:
        """
        Evaluate the model's performance on mathematical problems.
        
        Args:
            test_problems: List of test problems
            
        Returns:
            Dict containing evaluation metrics
        """
        results = {
            "solution_accuracy": 0.0,
            "step_accuracy": 0.0,
            "expression_accuracy": 0.0
        }
        
        for problem in test_problems:
            # Generate solution
            pred_solution = self.generate(problem.question)
            
            # Calculate solution accuracy
            solution_acc = self.calculate_math_accuracy(pred_solution, problem.solution)
            results["solution_accuracy"] += solution_acc
            
            # Calculate step accuracy if steps are provided
            if problem.steps:
                step_acc = 0.0
                pred_steps = self.generate_steps(problem.question)
                for pred_step, true_step in zip(pred_steps, problem.steps):
                    step_acc += self.calculate_math_accuracy(pred_step, true_step)
                results["step_accuracy"] += step_acc / len(problem.steps)
            
            # Calculate expression accuracy if latex expression is provided
            if problem.latex_expr:
                pred_expr = self.generate_expression(problem.question)
                expr_acc = self.calculate_math_accuracy(pred_expr, problem.latex_expr)
                results["expression_accuracy"] += expr_acc
        
        # Average results
        num_problems = len(test_problems)
        for key in results:
            results[key] /= num_problems
        
        return results

def create_sample_math_problems() -> List[MathProblem]:
    """Create a set of sample mathematical problems for testing."""
    return [
        MathProblem(
            question="Solve the quadratic equation: x² + 5x + 6 = 0",
            solution="x = -2 or x = -3",
            steps=[
                "1. Identify a=1, b=5, c=6",
                "2. Use quadratic formula: x = (-b ± √(b² - 4ac))/2a",
                "3. x = (-5 ± √(25 - 24))/2",
                "4. x = (-5 ± √1)/2",
                "5. x = (-5 ± 1)/2",
                "6. x = -3 or x = -2"
            ],
            latex_expr="x^2 + 5x + 6 = 0"
        ),
        MathProblem(
            question="Find the derivative of f(x) = x³ - 2x² + 4x - 1",
            solution="f'(x) = 3x² - 4x + 4",
            steps=[
                "1. Use power rule for x³: derivative is 3x²",
                "2. Use power rule for -2x²: derivative is -4x",
                "3. Use power rule for 4x: derivative is 4",
                "4. Constant term -1 becomes 0",
                "5. Combine terms: 3x² - 4x + 4"
            ],
            latex_expr="\\frac{d}{dx}(x^3 - 2x^2 + 4x - 1)"
        ),
        MathProblem(
            question="Calculate the integral of 2x + 3 from 0 to 2",
            solution="7",
            steps=[
                "1. Integrate 2x: x²",
                "2. Integrate 3: 3x",
                "3. Antiderivative is x² + 3x",
                "4. Evaluate at x=2: (4 + 6)",
                "5. Evaluate at x=0: (0 + 0)",
                "6. Subtract: 10 - 3 = 7"
            ],
            latex_expr="\\int_0^2 (2x + 3) dx"
        )
    ]

def main():
    """Run mathematical training demo."""
    # Create sample problems
    problems = create_sample_math_problems()
    
    # Initialize trainer
    trainer = MathTrainer(
        model=None,  # Will be initialized by parent class
        tokenizer=BPETokenizer(vocab_size=30000),
        output_dir="output/math"
    )
    
    # Add math symbols to tokenizer
    trainer.tokenizer.add_tokens(trainer.math_symbols)
    
    # Train tokenizer
    print("Training tokenizer...")
    trainer.tokenizer.train([p.question + " " + p.solution for p in problems])
    
    # Train model
    print("\nStarting mathematical training...")
    trainer.train_on_math(
        problems=problems,
        batch_size=2,
        num_epochs=3,
        learning_rate=3e-5,
        step_weight=0.3,
        expression_weight=0.3
    )
    
    # Evaluate
    print("\nEvaluating model...")
    eval_results = trainer.evaluate_math(problems)
    print("\nEvaluation Results:")
    for metric, value in eval_results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 