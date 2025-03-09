# NeuraFlux - Advanced Language Model

NeuraFlux is a hybrid transformer language model with grammar and math capabilities. It combines a custom neural network architecture with efficient token handling to provide answers to a variety of questions.

## Features

- **Grammar Processing**: NeuraFlux can identify and correct grammatical errors, explain grammar rules, and provide examples of proper usage.
- **Mathematical Calculations**: Solve basic and complex mathematical operations with natural language.
- **General Knowledge QA**: Answer various factual and self-referential questions.

## Architecture

NeuraFlux uses a hybrid Transformer-CNN architecture with Retrieval-Augmented Generation (RAG) capabilities. It includes:

- 1.45M parameters
- Custom BPE tokenizer
- Grammar rule database
- Mathematical expression parser

## Web UI

The web interface allows you to:

1. Ask general knowledge questions
2. Request grammar corrections and explanations
3. Solve mathematical problems

### Question Types

When asking a question, you can specify the type of question:

- **General**: For factual or general knowledge questions
- **Math**: For mathematical calculations and problems
- **Grammar**: For grammar corrections, rules, and examples

## Getting Started

To run the application locally:

1. Install dependencies:
   ```
   npm install
   ```

2. Start the development server:
   ```
   node start-server.js
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:3001
   ```

## Examples

### Grammar Questions

- "What is subject-verb agreement?"
- "Is this sentence correct: She don't like apples"
- "What is the subjunctive mood?"

### Math Questions

- "What is 125 + 37?"
- "Calculate 15% of 230"
- "Divide 144 by 12"

### General Knowledge

- "What is the capital of France?"
- "Who are you?"
- "What can you help me with?"

## Training

NeuraFlux has been trained on:
- SQuAD dataset
- TinyStories
- Self-QA
- English Grammar Dataset

## Contributors

- Saptarshi Halder (Creator) 