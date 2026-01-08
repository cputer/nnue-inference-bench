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
    float acc = (float)B1[neuron_idx] / QUANT_SCALE_W1;
    int num_features = feature_counts[pos_idx];
    const int32_t* pos_features = feature_indices + pos_idx * max_features;
    for (int i = 0; i < num_features; i++) {
        int feat_idx = pos_features[i];
        if (feat_idx >= 0 && feat_idx < INPUT_SIZE) {
            acc += (float)W1[feat_idx * L1_SIZE + neuron_idx] / QUANT_SCALE_W1;
        }
    }
    output[pos_idx * L1_SIZE + neuron_idx] = screlu(acc);
}

/**
 * Dense layer kernel (generic for L2, L3, L4).
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
    float acc = (float)bias[neuron_idx] / QUANT_SCALE_BIAS;
    const float* pos_input = input + pos_idx * in_size;
    for (int i = 0; i < in_size; i++) {
        acc += pos_input[i] * ((float)weights[i * out_size + neuron_idx] / QUANT_SCALE_DENSE);
    }
    if (apply_screlu) acc = screlu(acc);
    output[pos_idx * out_size + neuron_idx] = acc;
}

/**
 * Concatenate accumulators based on side to move.
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
        output[pos_idx * 512 + neuron_idx] = w;
        output[pos_idx * 512 + L1_SIZE + neuron_idx] = b;
    } else {
        output[pos_idx * 512 + neuron_idx] = b;
        output[pos_idx * 512 + L1_SIZE + neuron_idx] = w;
    }
}

// C interface for Python ctypes
extern "C" {

__declspec(dllexport) int nnue_get_version() { return 1; }
__declspec(dllexport) cudaError_t nnue_init(int device_id) { return cudaSetDevice(device_id); }
__declspec(dllexport) cudaError_t nnue_shutdown() { return cudaDeviceReset(); }
__declspec(dllexport) cudaError_t nnue_sync() { return cudaDeviceSynchronize(); }
__declspec(dllexport) cudaError_t nnue_malloc(void** ptr, size_t size) { return cudaMalloc(ptr, size); }
__declspec(dllexport) cudaError_t nnue_free(void* ptr) { return cudaFree(ptr); }
__declspec(dllexport) cudaError_t nnue_memcpy_h2d(void* dst, const void* src, size_t size) { return cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice); }
__declspec(dllexport) cudaError_t nnue_memcpy_d2h(void* dst, const void* src, size_t size) { return cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost); }

__declspec(dllexport) cudaError_t nnue_sparse_accum(
    const int16_t* W1, const int16_t* B1,
    const int32_t* features, const int32_t* feature_counts,
    float* output, int batch_size, int max_features
) {
    dim3 blocks(batch_size);
    dim3 threads(L1_SIZE);
    kernel_sparse_accum<<<blocks, threads>>>(W1, B1, features, feature_counts, output, batch_size, max_features);
    return cudaGetLastError();
}

__declspec(dllexport) cudaError_t nnue_dense_layer(
    const int8_t* weights, const int16_t* bias,
    const float* input, float* output,
    int batch_size, int in_size, int out_size, int apply_screlu
) {
    dim3 blocks(batch_size);
    dim3 threads(out_size);
    kernel_dense_layer<<<blocks, threads>>>(weights, bias, input, output, batch_size, in_size, out_size, apply_screlu);
    return cudaGetLastError();
}

__declspec(dllexport) cudaError_t nnue_concat_accum(
    const float* acc_white, const float* acc_black,
    const int32_t* stm, float* output, int batch_size
) {
    dim3 blocks(batch_size);
    dim3 threads(L1_SIZE);
    kernel_concat_accum<<<blocks, threads>>>(acc_white, acc_black, stm, output, batch_size);
    return cudaGetLastError();
}

} // extern "C"
