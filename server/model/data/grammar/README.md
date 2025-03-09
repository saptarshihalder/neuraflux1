# NeuraFlux Grammar Processing

This directory contains data files used by NeuraFlux's grammar processing capabilities.

## Files and Their Purpose

### 1. english_grammar.txt
Contains definitions and examples of various grammar rules, from basic to advanced concepts. The model uses this information to answer questions about grammar rules.

### 2. grammar_corrections.txt
Contains pairs of incorrect and correct grammar examples. The model uses these to identify common grammar mistakes and provide corrections.

### 3. advanced_grammar.txt
Contains examples and explanations of advanced grammar structures like subjunctive mood, inversion, cleft sentences, etc. The model uses this to help with more complex grammar questions.

## How Grammar Processing Works

The NeuraFlux model uses a `GrammarProcessor` class that:

1. **Detects Grammar Questions**: Analyzes queries to determine if they're asking about grammar rules or concepts.

2. **Identifies Correction Requests**: Recognizes when users want to check if a sentence is grammatically correct.

3. **Finds Grammar Errors**: Compares input text against known incorrect patterns to identify errors.

4. **Provides Corrections**: Generates corrections for identified errors and explains the relevant grammar rule.

5. **Answers Grammar Questions**: Provides explanations of grammar concepts with examples.

## Examples of Grammar Processing

### Grammar Questions
- "What is subject-verb agreement?"
- "Can you explain the subjunctive mood?"
- "What is a dangling modifier?"

### Grammar Corrections
- "Is this sentence correct: She don't like apples"
- "Check this: me and my friend went to the store"
- "Correct this: I should of done my homework"

## Integration with NeuraFlux

The `GrammarProcessor` is integrated with the main `NeuraFlex` class in `nanorag.py`. The processing flow is:

1. User input is first processed by the `GrammarProcessor`
2. If no grammar-related answer is found, it is sent to the `MathProcessor`
3. If no math-related answer is found, it is sent to the `QASystem`

This ensures that the model can handle a wide range of queries efficiently. 