/**
 * teacher_interface.h
 * 
 * Interface for teacher models in the knowledge transfer framework.
 * This facilitates learning from larger models like LLaMA or Flux.
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>

namespace mini_llm {
namespace teachers {

// Forward declaration of Tensor from transformer
class Tensor;

/**
 * Abstract base class for teacher models.
 * This interface allows the student model to learn from various teacher models
 * without knowing the specific implementation details.
 */
class TeacherModel {
public:
    virtual ~TeacherModel() = default;
    
    /**
     * Generate a response for the given prompt.
     * 
     * @param prompt The input prompt text
     * @param max_length Maximum length of the generated response
     * @param temperature Sampling temperature (higher = more random, lower = more deterministic)
     * @return Generated text response
     */
    virtual std::string generate(const std::string& prompt, 
                               int max_length = 100,
                               float temperature = 0.7) = 0;
    
    /**
     * Get token logits (raw probabilistic outputs) for a given input.
     * This is used for knowledge distillation via KL divergence loss.
     * 
     * @param input_text Input text to get logits for
     * @param return_hidden_states Whether to return intermediate hidden states
     * @return Tensor of logits, shape [sequence_length, vocab_size]
     */
    virtual Tensor get_logits(const std::string& input_text,
                            bool return_hidden_states = false) = 0;
    
    /**
     * Get intermediate layer representations for a given input.
     * This allows for matching intermediate layers between teacher and student.
     * 
     * @param input_text Input text to get hidden states for
     * @param layers Vector of layer indices to extract (empty means all layers)
     * @return Map of layer index to tensor of hidden states
     */
    virtual std::map<int, Tensor> get_hidden_states(const std::string& input_text,
                                                 const std::vector<int>& layers = {}) = 0;
    
    /**
     * Tokenize a string into token IDs using the teacher's tokenizer.
     * 
     * @param text Input text to tokenize
     * @return Vector of token IDs
     */
    virtual std::vector<int> tokenize(const std::string& text) = 0;
    
    /**
     * Decode token IDs back to text using the teacher's tokenizer.
     * 
     * @param token_ids Vector of token IDs
     * @return Decoded text
     */
    virtual std::string decode(const std::vector<int>& token_ids) = 0;
    
    /**
     * Get the vocabulary size of the teacher model.
     * 
     * @return Vocabulary size
     */
    virtual int get_vocab_size() const = 0;
    
    /**
     * Get the hidden size of the teacher model.
     * 
     * @return Hidden size (embedding dimension)
     */
    virtual int get_hidden_size() const = 0;
    
    /**
     * Get the number of layers in the teacher model.
     * 
     * @return Number of layers
     */
    virtual int get_num_layers() const = 0;
};

/**
 * Factory function to create a teacher model by name.
 * 
 * @param model_name Name of the model (e.g., "llama-7b", "flux-7b")
 * @param model_path Path to the model weights
 * @return Unique pointer to a TeacherModel instance
 */
std::unique_ptr<TeacherModel> create_teacher_model(const std::string& model_name,
                                                const std::string& model_path);

} // namespace teachers
} // namespace mini_llm 