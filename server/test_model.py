import sys
import os
import traceback

# Add the model directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

from nanorag import NanoRAG, NanoConfig
from tokenizer import NanoTokenizer

def test_model():
    print("Initializing model...")
    
    # Initialize configuration - much smaller for testing
    config = NanoConfig(
        vocab_size=16000,
        hidden_size=128,  # Smaller size
        num_hidden_layers=2,  # Fewer layers
        num_attention_heads=2,  # Fewer heads
        intermediate_size=512
    )
    
    # Initialize tokenizer
    print("Creating tokenizer...")
    tokenizer = NanoTokenizer(vocab_size=config.vocab_size)
    tokenizer._init_byte_vocab()  # Ensure vocab is initialized
    tokenizer._compile_pattern()
    
    # Create model
    print("Creating model...")
    model = NanoRAG(config, tokenizer)
    
    # Add a few simple documents to help with responses
    model.add_document("identity", "NeuraFlux is a small language model designed to answer questions. It uses retrieval-augmented generation to enhance its knowledge.")
    model.add_document("capabilities", "NeuraFlux can answer questions about various topics including AI, machine learning, and general knowledge.")
    model.build_document_embeddings()
    model.enable_retrieval(retrieval_k=1)
    
    # Test questions
    test_questions = [
        "Hello",
        "Hi there",
        "What is your name?",
        "Tell me a joke",
        "What can you do?",
        "Who created you?"
    ]
    
    print("\nTesting NeuraFlux Model\n")
    print("Available test questions:")
    for i, q in enumerate(test_questions, 1):
        print(f"{i}. {q}")
    
    while True:
        try:
            user_input = input("\nEnter question number, custom question, or 'quit' to exit: ").strip()
            
            if user_input.lower() == 'quit':
                break
                
            if user_input.isdigit() and 1 <= int(user_input) <= len(test_questions):
                question = test_questions[int(user_input) - 1]
            else:
                question = user_input
            
            print("\nQuestion:", question)
            print("Generating response...")
            
            try:
                # Use answer_with_rag for all questions
                response = model.answer_with_rag(question)
                print("\nResponse:", response)
                
            except Exception as e:
                print(f"Error generating response: {str(e)}")
                traceback.print_exc()
                
            print("-" * 80)
            
        except Exception as e:
            print(f"Error: {str(e)}")
            traceback.print_exc()
            print("Try another question or type 'quit' to exit")

if __name__ == "__main__":
    try:
        test_model()
    except Exception as e:
        print(f"Critical error: {str(e)}")
        traceback.print_exc() 