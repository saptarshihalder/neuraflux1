import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from training.math_training import MathTrainer
from tokenizer.bpe_tokenizer import BPETokenizer

app = Flask(__name__, static_folder="static", template_folder="templates")

# Initialize LLM
MODEL_PATH = os.environ.get("MODEL_PATH", "output/math_finetuned/final_model")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "output/math_finetuned/final_model/tokenizer")

# Initialize tokenizer
if os.path.exists(TOKENIZER_PATH):
    tokenizer = BPETokenizer.load(TOKENIZER_PATH)
else:
    tokenizer = BPETokenizer(vocab_size=30000)

# Initialize MathTrainer with the model and tokenizer
llm = MathTrainer(None, tokenizer, "output/temp")

@app.route('/')
def index():
    """Render the chat interface."""
    return render_template('index.html')

def is_self_referential(query):
    """Detect if a query is asking about NeuraFlex itself."""
    self_tokens = [
        "you", "your", "yourself", "neuraflex", "model", "language model", 
        "trained", "training", "parameter", "creator", "made", "built",
        "who are you", "what are you", "how do you work", "how were you",
        "saptarshi", "halder"
    ]
    
    query = query.lower()
    return any(token in query for token in self_tokens)

@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint for chat messages."""
    data = request.json
    
    if not data or 'message' not in data:
        return jsonify({
            'error': 'No message provided'
        }), 400
        
    user_message = data['message']
    show_steps = data.get('show_steps', True)
    
    try:
        # Check if it's a self-referential question
        if is_self_referential(user_message):
            # Get model information from the who-are-you endpoint
            model_info = who_are_you().json
            
            # Handle specific self-referential questions
            if any(token in user_message.lower() for token in ["who", "what are you", "tell me about"]):
                solution = f"I am {model_info['name']}, a {model_info['architecture']} language model created by {model_info['creator']}. I specialize in mathematical problem-solving and can answer questions about myself."
                steps = [
                    "1. Identified self-referential question about identity",
                    "2. Retrieved self-knowledge from my metadata",
                    "3. Generated response based on my architectural details and purpose"
                ]
            elif any(token in user_message.lower() for token in ["how do you work", "architecture", "transformer", "cnn"]):
                solution = f"I use a Hybrid Transformer-CNN architecture with {model_info['parameter_count']} parameters. My design combines transformer layers for sequence modeling with convolutional layers for efficient pattern recognition, plus RAG for fact retrieval."
                steps = [
                    "1. Identified question about my technical architecture",
                    "2. Referenced my architectural specifications",
                    "3. Described my hybrid transformer-CNN design and RAG components"
                ]
            elif any(token in user_message.lower() for token in ["trained", "training", "learn"]):
                solution = model_info['training']
                steps = [
                    "1. Identified question about my training process",
                    "2. Retrieved training information from my metadata",
                    "3. Described my multi-stage training including knowledge distillation and RLHF"
                ]
            elif any(token in user_message.lower() for token in ["created", "made", "built", "developer", "saptarshi"]):
                solution = f"I was created by {model_info['creator']} as part of a project to demonstrate efficient knowledge transfer from larger models to smaller, specialized architectures."
                steps = [
                    "1. Identified question about my creator",
                    "2. Retrieved creator information from my metadata",
                    "3. Provided information about my creator and development context"
                ]
            elif any(token in user_message.lower() for token in ["version", "update", "history"]):
                versions = "\n".join([f"• {v}" for v in model_info['version_history']])
                solution = f"My version history includes:\n{versions}"
                steps = [
                    "1. Identified question about my version history",
                    "2. Retrieved version information from my metadata",
                    "3. Listed my version history and updates"
                ]
            elif any(token in user_message.lower() for token in ["limit", "can't", "cannot", "unable"]):
                solution = model_info['limitations']
                steps = [
                    "1. Identified question about my limitations",
                    "2. Retrieved limitation information from my metadata",
                    "3. Explained my constraints and capabilities boundaries"
                ]
            else:
                # Generic self-referential response when no specific category matched
                solution = f"As {model_info['name']}, I'm a {model_info['architecture']} with {model_info['parameter_count']} parameters. My primary purpose is mathematical problem-solving, but I can also answer questions about myself and my capabilities."
                steps = [
                    "1. Identified general self-referential question",
                    "2. Retrieved relevant self-information",
                    "3. Generated a response covering my key attributes"
                ]
                
            # Return self-referential response
            return jsonify({
                'solution': solution,
                'steps': steps if show_steps else []
            })
        
        # For regular math problems, proceed with normal processing
        solution = llm.generate(user_message)
        
        # Get step-by-step explanation if requested
        steps = []
        if show_steps:
            steps = llm.generate_steps(user_message)
            
            # If steps are too generic, provide more detailed ones based on the problem type
            if len(steps) <= 3 and all(s.startswith(("First", "Second", "Third")) for s in steps):
                if "quadratic" in user_message.lower():
                    steps = [
                        "1. Identify the coefficients a, b, and c in the quadratic equation ax² + bx + c = 0",
                        "2. Apply the quadratic formula: x = (-b ± √(b² - 4ac))/2a",
                        "3. Calculate the discriminant: b² - 4ac",
                        "4. Calculate the two solutions by substituting the discriminant into the formula",
                        "5. Simplify the final answers"
                    ]
                elif "derivative" in user_message.lower():
                    steps = [
                        "1. Apply the power rule: d/dx[x^n] = n·x^(n-1)",
                        "2. Apply the sum rule: d/dx[f(x) + g(x)] = d/dx[f(x)] + d/dx[g(x)]",
                        "3. Apply the constant multiple rule: d/dx[c·f(x)] = c·d/dx[f(x)]",
                        "4. Combine all terms to get the final derivative"
                    ]
                elif "integral" in user_message.lower():
                    steps = [
                        "1. Apply the power rule for integration: ∫x^n dx = x^(n+1)/(n+1) + C",
                        "2. Apply the sum rule: ∫[f(x) + g(x)]dx = ∫f(x)dx + ∫g(x)dx",
                        "3. Apply the constant multiple rule: ∫c·f(x)dx = c·∫f(x)dx",
                        "4. If it's a definite integral, evaluate at the bounds and subtract"
                    ]
                elif "eigenvalue" in user_message.lower():
                    steps = [
                        "1. Set up the characteristic equation: det(A - λI) = 0",
                        "2. Calculate the determinant and expand the polynomial",
                        "3. Solve for the roots of the polynomial to find eigenvalues",
                        "4. For each eigenvalue λᵢ, solve the equation (A - λᵢI)v = 0 to find eigenvectors",
                        "5. Normalize eigenvectors if needed"
                    ]
        
        # Format solution if needed
        if "derivative" in user_message.lower() and "=" not in solution:
            solution = f"f'(x) = {solution}"
            
        # Prepare response
        response = {
            'solution': solution,
            'steps': steps
        }
        
        return jsonify(response)
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({
            'error': 'Failed to process message',
            'solution': 'I encountered an error while solving this problem.',
            'steps': []
        }), 500

@app.route('/api/examples')
def examples():
    """Get example math problems."""
    examples = [
        "Find the derivative of f(x) = x³ - 2x² + 4x - 1",
        "Solve the quadratic equation: x² + 5x + 6 = 0",
        "Calculate the integral of 2x + 3 from 0 to 2",
        "Find the eigenvalues and eigenvectors of the matrix A = [[2, 1], [1, 2]]",
        "Prove that for any positive integer n, the sum of the first n positive odd integers is n²"
    ]
    return jsonify(examples)

@app.route('/api/who-are-you', methods=['GET'])
def who_are_you():
    """Return information about the LLM for the 'who are you' question."""
    info = {
        "name": "NeuraFlex",
        "creator": "Saptarshi Halder",
        "description": "I am NeuraFlex, a specialized language model focusing on mathematical problem-solving. I was developed to demonstrate knowledge transfer from larger teacher models to a smaller, more efficient architecture.",
        "architecture": "Hybrid Transformer-CNN with RAG",
        "parameter_count": "1.45M",
        "training_data": ["SQuAD", "TinyStories", "Self-QA", "Mathematical datasets"],
        "version_history": [
            "v1.0 (2024-03): Initial release", 
            "v1.1 (2024-05): Added RLHF"
        ],
        "unique_features": [
            "Hybrid Transformer-CNN layers for efficient processing",
            "RAG with BM25 retrieval for factual accuracy",
            "Self-awareness module for answering questions about itself",
            "Specialized mathematical problem-solving capabilities",
            "RLHF fine-tuning for better alignment"
        ],
        "capabilities": [
            "Solving algebraic equations",
            "Calculating derivatives and integrals",
            "Finding eigenvalues and eigenvectors",
            "Providing step-by-step mathematical solutions",
            "Supporting various mathematical notations",
            "Answering questions about my own architecture and capabilities"
        ],
        "training": "I was trained using a multi-stage process including knowledge distillation from larger models, reinforcement learning from human feedback (RLHF), and specialized training on mathematical problem-solving.",
        "limitations": "As a specialized model, I perform best on mathematical problems and questions about myself. I cannot access real-time data beyond my training cutoff and may struggle with very complex problems beyond my parameter capacity."
    }
    return jsonify(info)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 