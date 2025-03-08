/**
 * rope_attention.cpp
 * 
 * Implementation of Rotary Positional Embedding (RoPE) attention with sliding window.
 * This is a key component of modern language models, offering efficient positional encoding.
 */

#include "transformer.h"
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>
#include <random>
#include <algorithm>

namespace mini_llm {

RoPEAttention::RoPEAttention(int hidden_size, int num_heads, int window_size, float dropout_prob)
    : hidden_size_(hidden_size),
      num_heads_(num_heads),
      head_dim_(hidden_size / num_heads),
      window_size_(window_size),
      dropout_prob_(dropout_prob),
      query_proj_(hidden_size, hidden_size),
      key_proj_(hidden_size, hidden_size),
      value_proj_(hidden_size, hidden_size),
      output_proj_(hidden_size, hidden_size) {
    
    // Validate parameters
    if (hidden_size % num_heads != 0) {
        throw std::invalid_argument("Hidden size must be divisible by number of heads");
    }
}

Tensor RoPEAttention::forward(const Tensor& input, const Tensor& attention_mask) {
    // input shape: [batch_size, seq_len, hidden_size]
    const auto& shape = input.shape();
    int batch_size = shape[0];
    int seq_len = shape[1];
    
    // 1. Project inputs to queries, keys, and values
    Tensor query = query_proj_.forward(input);  // [batch_size, seq_len, hidden_size]
    Tensor key = key_proj_.forward(input);      // [batch_size, seq_len, hidden_size]
    Tensor value = value_proj_.forward(input);  // [batch_size, seq_len, hidden_size]
    
    // 2. Reshape to [batch_size, seq_len, num_heads, head_dim]
    query = query.reshape({batch_size, seq_len, num_heads_, head_dim_});
    key = key.reshape({batch_size, seq_len, num_heads_, head_dim_});
    value = value.reshape({batch_size, seq_len, num_heads_, head_dim_});
    
    // 3. Transpose to [batch_size, num_heads, seq_len, head_dim]
    query = query.permute({0, 2, 1, 3});
    key = key.permute({0, 2, 1, 3});
    value = value.permute({0, 2, 1, 3});
    
    // 4. Apply rotary positional embeddings
    std::vector<int> positions(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        positions[i] = i;
    }
    Tensor pos_tensor({seq_len}, std::vector<float>(positions.begin(), positions.end()));
    
    query = apply_rotary_embeddings(query, pos_tensor);
    key = apply_rotary_embeddings(key, pos_tensor);
    
    // 5. Apply sliding window attention
    Tensor context = sliding_window_attention(query, key, value, attention_mask);
    
    // 6. Reshape and project output
    // [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, num_heads, head_dim]
    context = context.permute({0, 2, 1, 3});
    
    // [batch_size, seq_len, num_heads, head_dim] -> [batch_size, seq_len, hidden_size]
    context = context.reshape({batch_size, seq_len, hidden_size_});
    
    // Project to output dimension
    return output_proj_.forward(context);
}

Tensor RoPEAttention::apply_rotary_embeddings(const Tensor& x, const Tensor& positions) const {
    // x shape: [batch_size, num_heads, seq_len, head_dim]
    // positions shape: [seq_len]
    const auto& shape = x.shape();
    int batch_size = shape[0];
    int num_heads = shape[1];
    int seq_len = shape[2];
    int head_dim = shape[3];
    
    // Precompute freqs_cis: complex values for rotation
    auto [cos_vals, sin_vals] = precompute_freqs_cis(head_dim / 2, seq_len);
    
    // Initialize the output tensor
    Tensor output({batch_size, num_heads, seq_len, head_dim});
    
    // Apply rotations by complex multiplication
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                int pos = static_cast<int>(positions.at({s}));
                
                // Process pairs of dimensions (real and imaginary parts)
                for (int i = 0; i < head_dim / 2; ++i) {
                    float x_real = x.at({b, h, s, 2 * i});
                    float x_imag = x.at({b, h, s, 2 * i + 1});
                    
                    float cos_val = cos_vals.at({pos, i});
                    float sin_val = sin_vals.at({pos, i});
                    
                    // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                    output.at({b, h, s, 2 * i}) = x_real * cos_val - x_imag * sin_val;
                    output.at({b, h, s, 2 * i + 1}) = x_real * sin_val + x_imag * cos_val;
                }
            }
        }
    }
    
    return output;
}

std::pair<Tensor, Tensor> RoPEAttention::precompute_freqs_cis(int dim, int end) const {
    // Compute the complex exponentials (cos and sin) for the rotational embeddings
    // Implements RoPE: freqs_cis(pos, i) = exp(-i * pos * theta_i)
    // where theta_i = 10000^(-2i/d)
    
    float theta = 10000.0f;
    
    // Initialize tensors for cos and sin values
    Tensor cos_vals({end, dim});
    Tensor sin_vals({end, dim});
    
    for (int pos = 0; pos < end; ++pos) {
        for (int i = 0; i < dim; ++i) {
            float freq = static_cast<float>(pos) / std::pow(theta, static_cast<float>(2 * i) / static_cast<float>(dim * 2));
            cos_vals.at({pos, i}) = std::cos(freq);
            sin_vals.at({pos, i}) = std::sin(freq);
        }
    }
    
    return {cos_vals, sin_vals};
}

Tensor RoPEAttention::sliding_window_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                             const Tensor& mask) const {
    // q, k, v shapes: [batch_size, num_heads, seq_len, head_dim]
    const auto& shape = q.shape();
    int batch_size = shape[0];
    int num_heads = shape[1];
    int seq_len = shape[2];
    int head_dim = shape[3];
    
    // Initialize output tensor
    Tensor output({batch_size, num_heads, seq_len, head_dim});
    
    // For each position, attend only to a window of window_size tokens around it
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int qpos = 0; qpos < seq_len; ++qpos) {
                // Define the window boundaries
                int window_start = std::max(0, qpos - window_size_ / 2);
                int window_end = std::min(seq_len, qpos + window_size_ / 2 + 1);
                
                // Apply attention within the window
                Tensor attention_scores({window_end - window_start});
                
                // Compute attention scores for each key position in the window
                for (int kpos = window_start; kpos < window_end; ++kpos) {
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        score += q.at({b, h, qpos, d}) * k.at({b, h, kpos, d});
                    }
                    // Scale by sqrt(head_dim)
                    score /= std::sqrt(static_cast<float>(head_dim));
                    
                    // Apply mask if provided
                    if (mask.numel() > 0) {
                        score += mask.at({b, h, qpos, kpos});
                    }
                    
                    attention_scores.at({kpos - window_start}) = score;
                }
                
                // Apply softmax normalization
                attention_scores = attention_scores.softmax();
                
                // Apply attention to values
                std::vector<float> context_vector(head_dim, 0.0f);
                for (int kpos = window_start; kpos < window_end; ++kpos) {
                    float weight = attention_scores.at({kpos - window_start});
                    for (int d = 0; d < head_dim; ++d) {
                        context_vector[d] += weight * v.at({b, h, kpos, d});
                    }
                }
                
                // Store result
                for (int d = 0; d < head_dim; ++d) {
                    output.at({b, h, qpos, d}) = context_vector[d];
                }
            }
        }
    }
    
    return output;
}

void RoPEAttention::zero_grad() {
    query_proj_.zero_grad();
    key_proj_.zero_grad();
    value_proj_.zero_grad();
    output_proj_.zero_grad();
}

std::vector<Tensor*> RoPEAttention::parameters() {
    std::vector<Tensor*> params;
    
    // Collect parameters from submodules
    auto query_params = query_proj_.parameters();
    auto key_params = key_proj_.parameters();
    auto value_params = value_proj_.parameters();
    auto output_params = output_proj_.parameters();
    
    // Combine all parameters
    params.insert(params.end(), query_params.begin(), query_params.end());
    params.insert(params.end(), key_params.begin(), key_params.end());
    params.insert(params.end(), value_params.begin(), value_params.end());
    params.insert(params.end(), output_params.begin(), output_params.end());
    
    return params;
}

void RoPEAttention::save(const std::string& path) {
    query_proj_.save(path + "_query_proj");
    key_proj_.save(path + "_key_proj");
    value_proj_.save(path + "_value_proj");
    output_proj_.save(path + "_output_proj");
}

void RoPEAttention::load(const std::string& path) {
    query_proj_.load(path + "_query_proj");
    key_proj_.load(path + "_key_proj");
    value_proj_.load(path + "_value_proj");
    output_proj_.load(path + "_output_proj");
}

} // namespace mini_llm 