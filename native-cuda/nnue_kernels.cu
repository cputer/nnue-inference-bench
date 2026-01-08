/**
 * NNUE Inference CUDA Kernels
 * 
 * Implements sparse accumulation and dense layers for NNUE evaluation.
 * Architecture: 40960 -> 256 -> 32 -> 32 -> 1 (+WDL)
 */

#include <cuda_runtime.h>
#include <stdint.h>

// Quantization scales (must match NKNN format)
#define QUANT_SCALE_W1 128.0f
#define QUANT_SCALE_DENSE 64.0f
#define QUANT_SCALE_BIAS 128.0f

// Network dimensions
#define INPUT_SIZE 40960
#define L1_SIZE 256
#define L2_SIZE 32
#define L3_SIZE 32

/**
 * SCReLU activation: clamp to [0,1] then square
 */
__device__ __forceinline__ float screlu(float x) {
    float clamped = fminf(fmaxf(x, 0.0f), 1.0f);
    return clamped * clamped;
}

/**
 * Sparse accumulation kernel for first layer.
 * Each thread block processes one position.
 * 
 * @param W1 Weight matrix [40960, 256] as int16
 * @param B1 Bias vector [256] as int16
 * @param feature_indices Active feature indices per position [batch, max_features]
 * @param feature_counts Number of active features per position [batch]
 * @param output Accumulated activations [batch, 256]
 * @param batch_size Number of positions
 * @param max_features Maximum features per position
 */
__global__ void kernel_sparse_accum(
    const int16_t* __restrict__ W1,
    const int16_t* __restrict__ B1,
    const int32_t* __restrict__ feature_indices,
    const int32_t* __restrict__ feature_counts,
    float* __restrict__ output,
    int batch_size,
    int max_features
) {
    int pos_idx = blockIdx.x;
    int neuron_idx = threadIdx.x;
    
    if (pos_idx >= batch_size || neuron_idx >= L1_SIZE) return;
    
    // Start with bias
    float acc = (float)B1[neuron_idx] / QUANT_SCALE_W1;
    
    // Accumulate active features
    int num_features = feature_counts[pos_idx];
    const int32_t* pos_features = feature_indices + pos_idx * max_features;
    
    for (int i = 0; i < num_features; i++) {
        int feat_idx = pos_features[i];
        if (feat_idx >= 0 && feat_idx < INPUT_SIZE) {
            acc += (float)W1[feat_idx * L1_SIZE + neuron_idx] / QUANT_SCALE_W1;
        }
    }
    
    // Apply SCReLU and store
    output[pos_idx * L1_SIZE + neuron_idx] = screlu(acc);
}

/**
 * Dense layer kernel (generic for L2, L3, L4).
 * Processes one output neuron per thread.
 * 
 * @param weights Weight matrix [in_size, out_size] as int8
 * @param bias Bias vector [out_size] as int16
 * @param input Input activations [batch, in_size]
 * @param output Output activations [batch, out_size]
 * @param batch_size Number of positions
 * @param in_size Input dimension
 * @param out_size Output dimension
 * @param apply_screlu Whether to apply SCReLU activation
 */
__global__ void kernel_dense_layer(
    const int8_t* __restrict__ weights,
    const int16_t* __restrict__ bias,
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int in_size,
    int out_size,
    int apply_screlu
) {
    int pos_idx = blockIdx.x;
    int neuron_idx = threadIdx.x;
    
    if (pos_idx >= batch_size || neuron_idx >= out_size) return;
    
    // Start with bias
    float acc = (float)bias[neuron_idx] / QUANT_SCALE_BIAS;
    
    // Matrix-vector multiplication
    const float* pos_input = input + pos_idx * in_size;
    for (int i = 0; i < in_size; i++) {
        acc += pos_input[i] * ((float)weights[i * out_size + neuron_idx] / QUANT_SCALE_DENSE);
    }
    
    // Apply activation if requested
    if (apply_screlu) {
        acc = screlu(acc);
    }
    
    output[pos_idx * out_size + neuron_idx] = acc;
}

/**
 * Concatenate white and black accumulators based on side to move.
 * Output: [stm_acc, other_acc] where stm_acc is the side-to-move accumulator.
 */
__global__ void kernel_concat_accum(
    const float* __restrict__ acc_white,
    const float* __restrict__ acc_black,
    const int32_t* __restrict__ stm,
    float* __restrict__ output,
    int batch_size
) {
    int pos_idx = blockIdx.x;
    int neuron_idx = threadIdx.x;
    
    if (pos_idx >= batch_size || neuron_idx >= L1_SIZE) return;
    
    float w = acc_white[pos_idx * L1_SIZE + neuron_idx];
    float b = acc_black[pos_idx * L1_SIZE + neuron_idx];
    
    if (stm[pos_idx] == 0) {
        // White to move: [white, black]
        output[pos_idx * 512 + neuron_idx] = w;
        output[pos_idx * 512 + L1_SIZE + neuron_idx] = b;
    } else {
        // Black to move: [black, white]
        output[pos_idx * 512 + neuron_idx] = b;
        output[pos_idx * 512 + L1_SIZE + neuron_idx] = w;
    }
}

// C interface for Python ctypes
extern "C" {
    
cudaError_t nnue_sparse_accum(
    const int16_t* W1, const int16_t* B1,
    const int32_t* features, const int32_t* feature_counts,
    float* output, int batch_size, int max_features
) {
    dim3 blocks(batch_size);
    dim3 threads(L1_SIZE);
    kernel_sparse_accum<<<blocks, threads>>>(
        W1, B1, features, feature_counts, output, batch_size, max_features
    );
    return cudaGetLastError();
}

cudaError_t nnue_dense_layer(
    const int8_t* weights, const int16_t* bias,
    const float* input, float* output,
    int batch_size, int in_size, int out_size, int apply_screlu
) {
    dim3 blocks(batch_size);
    dim3 threads(out_size);
    kernel_dense_layer<<<blocks, threads>>>(
        weights, bias, input, output, batch_size, in_size, out_size, apply_screlu
    );
    return cudaGetLastError();
}

cudaError_t nnue_concat_accum(
    const float* acc_white, const float* acc_black,
    const int32_t* stm, float* output, int batch_size
) {
    dim3 blocks(batch_size);
    dim3 threads(L1_SIZE);
    kernel_concat_accum<<<blocks, threads>>>(
        acc_white, acc_black, stm, output, batch_size
    );
    return cudaGetLastError();
}

} // extern "C"
