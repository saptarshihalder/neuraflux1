/**
 * glu_ffn.cpp
 * 
 * Implementation of Gated Linear Unit (GLU) Feed-Forward Network.
 * This uses SwiGLU activation as described in the PaLM paper for better parameter efficiency.
 */

#include "transformer.h"
#include <vector>
#include <stdexcept>
#include <random>
#include <cmath>

namespace mini_llm {

GLUFFNetwork::GLUFFNetwork(int hidden_size, int intermediate_size, float dropout_prob)
    : hidden_size_(hidden_size),
      intermediate_size_(intermediate_size),
      dropout_prob_(dropout_prob),
      gate_proj_(hidden_size, intermediate_size),
      up_proj_(hidden_size, intermediate_size),
      down_proj_(intermediate_size, hidden_size) {
    
    // Validate parameters
    if (hidden_size <= 0 || intermediate_size <= 0) {
        throw std::invalid_argument("Hidden size and intermediate size must be positive");
    }
}

Tensor GLUFFNetwork::forward(const Tensor& input) {
    // Input shape: [batch_size, seq_len, hidden_size]
    
    // Project to intermediate size
    Tensor gate = gate_proj_.forward(input);  // [batch_size, seq_len, intermediate_size]
    Tensor up = up_proj_.forward(input);      // [batch_size, seq_len, intermediate_size]
    
    // Apply SwiGLU activation: Swish(gate) * up
    // where Swish(x) = x * sigmoid(x)
    Tensor swish_gate = gate * gate.sigmoid();  // Element-wise multiplication
    
    // Apply gate
    Tensor intermediate = swish_gate * up;  // Element-wise multiplication
    
    // Dropout - simplified implementation for now
    if (dropout_prob_ > 0 && this->training) {
        const auto& shape = intermediate.shape();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution dist(1.0 - dropout_prob_);
        
        // Create dropout mask
        std::vector<float> mask_data(intermediate.numel());
        float scale = 1.0f / (1.0f - dropout_prob_);
        
        for (int i = 0; i < intermediate.numel(); ++i) {
            mask_data[i] = dist(gen) ? scale : 0.0f;
        }
        
        Tensor mask(shape, mask_data);
        intermediate = intermediate * mask;  // Element-wise multiplication
    }
    
    // Project back to hidden size
    return down_proj_.forward(intermediate);  // [batch_size, seq_len, hidden_size]
}

void GLUFFNetwork::zero_grad() {
    gate_proj_.zero_grad();
    up_proj_.zero_grad();
    down_proj_.zero_grad();
}

std::vector<Tensor*> GLUFFNetwork::parameters() {
    std::vector<Tensor*> params;
    
    // Collect parameters from submodules
    auto gate_params = gate_proj_.parameters();
    auto up_params = up_proj_.parameters();
    auto down_params = down_proj_.parameters();
    
    // Combine all parameters
    params.insert(params.end(), gate_params.begin(), gate_params.end());
    params.insert(params.end(), up_params.begin(), up_params.end());
    params.insert(params.end(), down_params.begin(), down_params.end());
    
    return params;
}

void GLUFFNetwork::save(const std::string& path) {
    gate_proj_.save(path + "_gate_proj");
    up_proj_.save(path + "_up_proj");
    down_proj_.save(path + "_down_proj");
}

void GLUFFNetwork::load(const std::string& path) {
    gate_proj_.load(path + "_gate_proj");
    up_proj_.load(path + "_up_proj");
    down_proj_.load(path + "_down_proj");
}

} // namespace mini_llm 