/**
 * tensor.cpp
 * 
 * Implementation of the Tensor class for the transformer architecture.
 */

#include "transformer.h"
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <sstream>

namespace mini_llm {

// Default constructor
Tensor::Tensor() : shape_(), data_(), strides_() {}

// Shape-only constructor, initializes to zeros
Tensor::Tensor(const std::vector<int>& shape) : shape_(shape) {
    int total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    data_.resize(total_size, 0.0f);
    compute_strides();
}

// Constructor with shape and fill value
Tensor::Tensor(const std::vector<int>& shape, float fill_value) : shape_(shape) {
    int total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    data_.resize(total_size, fill_value);
    compute_strides();
}

// Constructor with shape and data
Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& data) : shape_(shape), data_(data) {
    int total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    if (data.size() != total_size) {
        throw std::invalid_argument("Data size does not match shape");
    }
    compute_strides();
}

// Copy constructor
Tensor::Tensor(const Tensor& other) : shape_(other.shape_), data_(other.data_), strides_(other.strides_) {}

// Move constructor
Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), 
      data_(std::move(other.data_)), 
      strides_(std::move(other.strides_)) {}

// Copy assignment
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        shape_ = other.shape_;
        data_ = other.data_;
        strides_ = other.strides_;
    }
    return *this;
}

// Move assignment
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        data_ = std::move(other.data_);
        strides_ = std::move(other.strides_);
    }
    return *this;
}

// Element-wise addition
Tensor Tensor::operator+(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes must match for addition");
    }
    
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

// Element-wise subtraction
Tensor Tensor::operator-(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes must match for subtraction");
    }
    
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

// Element-wise multiplication (Hadamard product)
Tensor Tensor::operator*(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes must match for element-wise multiplication");
    }
    
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

// Element-wise division
Tensor Tensor::operator/(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor shapes must match for division");
    }
    
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        if (other.data_[i] == 0) {
            throw std::invalid_argument("Division by zero");
        }
        result.data_[i] = data_[i] / other.data_[i];
    }
    return result;
}

// Matrix multiplication
Tensor Tensor::matmul(const Tensor& other) const {
    // Basic 2D matrix multiplication for now
    // Will be optimized later with BLAS/cuBLAS
    if (ndim() != 2 || other.ndim() != 2) {
        throw std::invalid_argument("Matrix multiplication requires 2D tensors");
    }
    
    if (shape_[1] != other.shape_[0]) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }
    
    int m = shape_[0];
    int n = other.shape_[1];
    int k = shape_[1];
    
    Tensor result({m, n}, 0.0f);
    
    // Naive implementation for now
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                sum += at({i, p}) * other.at({p, j});
            }
            result.at({i, j}) = sum;
        }
    }
    
    return result;
}

// ReLU activation function
Tensor Tensor::relu() const {
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = std::max(0.0f, data_[i]);
    }
    return result;
}

// GELU activation function (Gaussian Error Linear Unit)
Tensor Tensor::gelu() const {
    Tensor result(shape_);
    constexpr float sqrt2 = 1.4142135623730951f; // sqrt(2)
    
    for (size_t i = 0; i < data_.size(); ++i) {
        // Approximation of GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        float x = data_[i];
        float x3 = x * x * x;
        result.data_[i] = x * 0.5f * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x3)));
    }
    return result;
}

// Tanh activation function
Tensor Tensor::tanh() const {
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = std::tanh(data_[i]);
    }
    return result;
}

// Sigmoid activation function
Tensor Tensor::sigmoid() const {
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = 1.0f / (1.0f + std::exp(-data_[i]));
    }
    return result;
}

// Softmax function
Tensor Tensor::softmax(int dim) const {
    if (dim < 0) {
        dim = ndim() + dim; // Convert negative dimension to positive
    }
    
    if (dim >= ndim() || dim < 0) {
        throw std::invalid_argument("Invalid dimension for softmax");
    }
    
    Tensor result = *this; // Start with a copy
    
    // Determine the shape of the dimension groups
    int outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= shape_[i];
    }
    
    int dim_size = shape_[dim];
    
    int inner_size = 1;
    for (int i = dim + 1; i < ndim(); ++i) {
        inner_size *= shape_[i];
    }
    
    // Apply softmax along the specified dimension
    for (int outer = 0; outer < outer_size; ++outer) {
        for (int inner = 0; inner < inner_size; ++inner) {
            // Find max value for numerical stability
            float max_val = -std::numeric_limits<float>::infinity();
            for (int d = 0; d < dim_size; ++d) {
                // Compute the flat index
                int idx = outer * dim_size * inner_size + d * inner_size + inner;
                max_val = std::max(max_val, data_[idx]);
            }
            
            // Compute exp(x - max) and the sum
            float sum = 0.0f;
            for (int d = 0; d < dim_size; ++d) {
                int idx = outer * dim_size * inner_size + d * inner_size + inner;
                result.data_[idx] = std::exp(data_[idx] - max_val);
                sum += result.data_[idx];
            }
            
            // Normalize
            for (int d = 0; d < dim_size; ++d) {
                int idx = outer * dim_size * inner_size + d * inner_size + inner;
                result.data_[idx] /= sum;
            }
        }
    }
    
    return result;
}

// Reshape the tensor to a new shape with the same number of elements
Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    int new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());
    if (new_size != numel()) {
        throw std::invalid_argument("Reshape: new shape must have the same number of elements");
    }
    
    Tensor result(new_shape);
    result.data_ = data_; // Copy the data
    return result;
}

// Permute dimensions
Tensor Tensor::permute(const std::vector<int>& dims) const {
    if (dims.size() != ndim()) {
        throw std::invalid_argument("Permute: dimensions must match tensor rank");
    }
    
    // Check if all dimensions are included
    std::vector<bool> dim_used(ndim(), false);
    for (int d : dims) {
        if (d < 0 || d >= ndim() || dim_used[d]) {
            throw std::invalid_argument("Permute: invalid or repeated dimension");
        }
        dim_used[d] = true;
    }
    
    // Create new shape and strides
    std::vector<int> new_shape(ndim());
    for (size_t i = 0; i < dims.size(); ++i) {
        new_shape[i] = shape_[dims[i]];
    }
    
    // Create result tensor with the new shape
    Tensor result(new_shape);
    
    // Implement permutation (naive approach for now)
    std::function<void(int, std::vector<int>&)> permute_recursive = 
        [&](int dim, std::vector<int>& indices) {
            if (dim == ndim()) {
                // Calculate source and target indices
                std::vector<int> src_indices(ndim());
                for (size_t i = 0; i < dims.size(); ++i) {
                    src_indices[dims[i]] = indices[i];
                }
                
                result.at(indices) = at(src_indices);
                return;
            }
            
            for (int i = 0; i < shape_[dims[dim]]; ++i) {
                indices[dim] = i;
                permute_recursive(dim + 1, indices);
            }
        };
    
    std::vector<int> indices(ndim(), 0);
    permute_recursive(0, indices);
    
    return result;
}

// Create a view with a different shape but the same data
Tensor Tensor::view(const std::vector<int>& new_shape) const {
    // For now, view is the same as reshape. In a more advanced implementation,
    // view would share data with the original tensor.
    return reshape(new_shape);
}

// Transpose two dimensions
Tensor Tensor::transpose(int dim1, int dim2) const {
    if (dim1 < 0) dim1 += ndim();
    if (dim2 < 0) dim2 += ndim();
    
    if (dim1 < 0 || dim1 >= ndim() || dim2 < 0 || dim2 >= ndim()) {
        throw std::invalid_argument("Transpose: invalid dimensions");
    }
    
    // Create permutation vector
    std::vector<int> perm(ndim());
    for (int i = 0; i < ndim(); ++i) {
        perm[i] = i;
    }
    std::swap(perm[dim1], perm[dim2]);
    
    return permute(perm);
}

// Remove a dimension of size 1
Tensor Tensor::squeeze(int dim) const {
    if (dim < 0) dim += ndim();
    
    if (dim < 0 || dim >= ndim()) {
        throw std::invalid_argument("Squeeze: invalid dimension");
    }
    
    if (shape_[dim] != 1) {
        throw std::invalid_argument("Squeeze: dimension must be of size 1");
    }
    
    std::vector<int> new_shape;
    for (int i = 0; i < ndim(); ++i) {
        if (i != dim) {
            new_shape.push_back(shape_[i]);
        }
    }
    
    return reshape(new_shape);
}

// Add a dimension of size 1
Tensor Tensor::unsqueeze(int dim) const {
    if (dim < 0) dim += ndim() + 1;
    
    if (dim < 0 || dim > ndim()) {
        throw std::invalid_argument("Unsqueeze: invalid dimension");
    }
    
    std::vector<int> new_shape = shape_;
    new_shape.insert(new_shape.begin() + dim, 1);
    
    return reshape(new_shape);
}

// Access element (non-const)
float& Tensor::at(const std::vector<int>& indices) {
    int offset = calculate_offset(indices);
    return data_[offset];
}

// Access element (const)
float Tensor::at(const std::vector<int>& indices) const {
    int offset = calculate_offset(indices);
    return data_[offset];
}

// Extract slice along a dimension
Tensor Tensor::slice(int dim, int start, int end) const {
    if (dim < 0) dim += ndim();
    
    if (dim < 0 || dim >= ndim()) {
        throw std::invalid_argument("Slice: invalid dimension");
    }
    
    if (start < 0) start += shape_[dim];
    if (end < 0) end += shape_[dim];
    
    if (start < 0 || start >= shape_[dim] || end <= start || end > shape_[dim]) {
        throw std::invalid_argument("Slice: invalid range");
    }
    
    // Create new shape with updated dimension
    std::vector<int> new_shape = shape_;
    new_shape[dim] = end - start;
    
    Tensor result(new_shape);
    
    // Implement slicing (naive approach)
    std::function<void(int, std::vector<int>&, std::vector<int>&)> slice_recursive = 
        [&](int d, std::vector<int>& src_indices, std::vector<int>& dst_indices) {
            if (d == ndim()) {
                result.at(dst_indices) = at(src_indices);
                return;
            }
            
            if (d == dim) {
                for (int i = start; i < end; ++i) {
                    src_indices[d] = i;
                    dst_indices[d] = i - start;
                    slice_recursive(d + 1, src_indices, dst_indices);
                }
            } else {
                for (int i = 0; i < shape_[d]; ++i) {
                    src_indices[d] = i;
                    dst_indices[d] = i;
                    slice_recursive(d + 1, src_indices, dst_indices);
                }
            }
        };
    
    std::vector<int> src_indices(ndim(), 0);
    std::vector<int> dst_indices(ndim(), 0);
    slice_recursive(0, src_indices, dst_indices);
    
    return result;
}

// Get tensor shape
std::vector<int> Tensor::shape() const {
    return shape_;
}

// Get number of dimensions
int Tensor::ndim() const {
    return static_cast<int>(shape_.size());
}

// Get size along a dimension
int Tensor::size(int dim) const {
    if (dim < 0) dim += ndim();
    
    if (dim < 0 || dim >= ndim()) {
        throw std::invalid_argument("Size: invalid dimension");
    }
    
    return shape_[dim];
}

// Get total number of elements
int Tensor::numel() const {
    return static_cast<int>(data_.size());
}

// Fill with a constant value
void Tensor::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

// Fill with zeros
void Tensor::zero_() {
    std::fill(data_.begin(), data_.end(), 0.0f);
}

// Get data pointer
float* Tensor::data_ptr() {
    return data_.data();
}

// Get const data pointer
const float* Tensor::data_ptr() const {
    return data_.data();
}

// Compute strides for fast indexing
void Tensor::compute_strides() {
    strides_.resize(shape_.size());
    int stride = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        strides_[i] = stride;
        stride *= shape_[i];
    }
}

// Calculate flat index from multi-dimensional indices
int Tensor::calculate_offset(const std::vector<int>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Index dimensions don't match tensor rank");
    }
    
    int offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        if (idx < 0) idx += shape_[i];
        
        if (idx < 0 || idx >= shape_[i]) {
            std::stringstream ss;
            ss << "Index " << idx << " out of bounds for dimension " << i << " with size " << shape_[i];
            throw std::out_of_range(ss.str());
        }
        
        offset += idx * strides_[i];
    }
    
    return offset;
}

} // namespace mini_llm 