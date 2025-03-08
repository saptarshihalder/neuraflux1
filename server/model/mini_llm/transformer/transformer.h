/**
 * transformer.h
 * 
 * From-scratch implementation of a transformer architecture with Rotary 
 * Positional Embeddings (RoPE) and Sliding Window Attention.
 * 
 * This header defines the core structures and classes for the transformer model,
 * without dependencies on external machine learning frameworks.
 */

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <functional>

namespace mini_llm {

// Forward declarations
class Tensor;
class Module;
class Embedding;
class Linear;
class LayerNorm;
class RoPEAttention;
class GLUFFNetwork;
class TransformerLayer;
class Transformer;

/**
 * Simple tensor class for storing and manipulating multi-dimensional arrays.
 * This is a basic implementation that will be optimized later with SIMD/GPU support.
 */
class Tensor {
public:
    // Constructors
    Tensor();
    Tensor(const std::vector<int>& shape);
    Tensor(const std::vector<int>& shape, float fill_value);
    Tensor(const std::vector<int>& shape, const std::vector<float>& data);
    
    // Copy and move semantics
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    
    // Destructor
    ~Tensor() = default;
    
    // Basic operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const; // Element-wise multiplication
    Tensor operator/(const Tensor& other) const;
    
    // Matrix multiplication
    Tensor matmul(const Tensor& other) const;
    
    // Element-wise operations
    Tensor relu() const;
    Tensor gelu() const;
    Tensor tanh() const;
    Tensor sigmoid() const;
    Tensor softmax(int dim = -1) const;
    
    // Shape manipulation
    Tensor reshape(const std::vector<int>& new_shape) const;
    Tensor permute(const std::vector<int>& dims) const;
    Tensor view(const std::vector<int>& new_shape) const;
    Tensor transpose(int dim1, int dim2) const;
    Tensor squeeze(int dim = -1) const;
    Tensor unsqueeze(int dim) const;
    
    // Access and manipulation
    float& at(const std::vector<int>& indices);
    float at(const std::vector<int>& indices) const;
    Tensor slice(int dim, int start, int end) const;
    
    // Utility methods
    std::vector<int> shape() const;
    int ndim() const;
    int size(int dim) const;
    int numel() const;
    void fill(float value);
    void zero_();
    
    // For CUDA/optimization later
    float* data_ptr();
    const float* data_ptr() const;
    
private:
    std::vector<int> shape_;
    std::vector<float> data_;
    std::vector<int> strides_; // For fast indexing
    
    // Helper methods
    void compute_strides();
    int calculate_offset(const std::vector<int>& indices) const;
};

/**
 * Base class for all neural network modules.
 */
class Module {
public:
    virtual ~Module() = default;
    
    virtual Tensor forward(const Tensor& input) = 0;
    
    virtual void zero_grad() = 0;
    
    virtual std::vector<Tensor*> parameters() = 0;
    
    // Serialization
    virtual void save(const std::string& path) = 0;
    virtual void load(const std::string& path) = 0;
};

/**
 * Embedding layer that maps token IDs to vectors.
 */
class Embedding : public Module {
public:
    Embedding(int num_embeddings, int embedding_dim);
    
    Tensor forward(const Tensor& input) override;
    
    void zero_grad() override;
    
    std::vector<Tensor*> parameters() override;
    
    void save(const std::string& path) override;
    void load(const std::string& path) override;
    
private:
    int num_embeddings_;
    int embedding_dim_;
    Tensor weight_;
    Tensor grad_;
};

/**
 * Linear (fully connected) layer.
 */
class Linear : public Module {
public:
    Linear(int in_features, int out_features, bool bias = true);
    
    Tensor forward(const Tensor& input) override;
    
    void zero_grad() override;
    
    std::vector<Tensor*> parameters() override;
    
    void save(const std::string& path) override;
    void load(const std::string& path) override;
    
private:
    int in_features_;
    int out_features_;
    bool has_bias_;
    Tensor weight_;
    Tensor bias_;
    Tensor weight_grad_;
    Tensor bias_grad_;
};

/**
 * Layer normalization.
 */
class LayerNorm : public Module {
public:
    LayerNorm(int normalized_shape, float eps = 1e-5);
    
    Tensor forward(const Tensor& input) override;
    
    void zero_grad() override;
    
    std::vector<Tensor*> parameters() override;
    
    void save(const std::string& path) override;
    void load(const std::string& path) override;
    
private:
    int normalized_shape_;
    float eps_;
    Tensor weight_;
    Tensor bias_;
    Tensor weight_grad_;
    Tensor bias_grad_;
};

/**
 * Rotary Positional Embedding (RoPE) self-attention with sliding window.
 */
class RoPEAttention : public Module {
public:
    RoPEAttention(int hidden_size, int num_heads, int window_size = 512, 
                 float dropout_prob = 0.1);
    
    Tensor forward(const Tensor& input, const Tensor& attention_mask = Tensor()) override;
    
    void zero_grad() override;
    
    std::vector<Tensor*> parameters() override;
    
    void save(const std::string& path) override;
    void load(const std::string& path) override;
    
private:
    int hidden_size_;
    int num_heads_;
    int head_dim_;
    int window_size_;
    float dropout_prob_;
    
    // Projections for Q, K, V
    Linear query_proj_;
    Linear key_proj_;
    Linear value_proj_;
    Linear output_proj_;
    
    // Apply RoPE to queries and keys
    Tensor apply_rotary_embeddings(const Tensor& x, const Tensor& positions) const;
    
    // Compute cosine and sine for RoPE
    std::pair<Tensor, Tensor> precompute_freqs_cis(int dim, int end) const;
    
    // Apply sliding window attention
    Tensor sliding_window_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& mask) const;
};

/**
 * Gated Linear Unit (GLU) Feed-Forward Network.
 * Uses SwiGLU activation as described in the PaLM paper.
 */
class GLUFFNetwork : public Module {
public:
    GLUFFNetwork(int hidden_size, int intermediate_size, float dropout_prob = 0.1);
    
    Tensor forward(const Tensor& input) override;
    
    void zero_grad() override;
    
    std::vector<Tensor*> parameters() override;
    
    void save(const std::string& path) override;
    void load(const std::string& path) override;
    
private:
    int hidden_size_;
    int intermediate_size_;
    float dropout_prob_;
    
    Linear gate_proj_;
    Linear up_proj_;
    Linear down_proj_;
};

/**
 * Complete transformer layer with attention and feed-forward components.
 */
class TransformerLayer : public Module {
public:
    TransformerLayer(int hidden_size, int num_attention_heads, int intermediate_size,
                   int window_size = 512, float dropout_prob = 0.1);
    
    Tensor forward(const Tensor& input, const Tensor& attention_mask = Tensor()) override;
    
    void zero_grad() override;
    
    std::vector<Tensor*> parameters() override;
    
    void save(const std::string& path) override;
    void load(const std::string& path) override;
    
private:
    RoPEAttention self_attention_;
    LayerNorm attention_layernorm_;
    GLUFFNetwork ffn_;
    LayerNorm ffn_layernorm_;
    float dropout_prob_;
};

/**
 * Full transformer model.
 */
class Transformer : public Module {
public:
    Transformer(int vocab_size, int hidden_size, int num_hidden_layers,
              int num_attention_heads, int intermediate_size,
              int max_position_embeddings = 2048, int window_size = 512,
              float dropout_prob = 0.1);
    
    Tensor forward(const Tensor& input_ids, const Tensor& attention_mask = Tensor()) override;
    
    // Text generation method
    std::vector<int> generate(const std::vector<int>& prompt_ids, int max_length = 100,
                            float temperature = 1.0, int top_k = 40, float top_p = 0.9);
    
    void zero_grad() override;
    
    std::vector<Tensor*> parameters() override;
    
    void save(const std::string& path) override;
    void load(const std::string& path) override;
    
private:
    int vocab_size_;
    int hidden_size_;
    int num_hidden_layers_;
    int num_attention_heads_;
    int intermediate_size_;
    int max_position_embeddings_;
    int window_size_;
    float dropout_prob_;
    
    Embedding token_embeddings_;
    std::vector<std::unique_ptr<TransformerLayer>> layers_;
    LayerNorm final_layernorm_;
    Linear lm_head_;
    
    // Helper for generation
    std::vector<int> sample_top_p_top_k(const Tensor& logits, float temperature, int top_k, float top_p);
};

} // namespace mini_llm 