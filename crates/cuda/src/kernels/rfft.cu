#include "fft.h"

// Enhanced FFT kernel supporting up to 16K elements (handling up to 14 FFT layers)
// Uses 1024 threads (32 warps), with each thread handling multiple elements when needed
// Assumption: The FFT size is a power of 2 and at least 64.
extern "C" __global__ void fft(
    unsigned int *src, unsigned int *dst, unsigned int **twiddle_dbl, int n_layers, int log_extension, int twiddle_size)
{
    // Each block handles 2 * blockDim.x * (1<<log_extension) elements.
    // Each thread will then do (1<<log_extension) butterflies at each layer.
    const int elements_per_block = blockDim.x << (1 + log_extension);

    // Shared memory for all butterfly operations
    __shared__ unsigned int shared_data[1 << 14]; // 64K bytes

    // STEP 1: Load all data directly into shared memory from global memory
    for (int i = 0; i < (1 << (1 + log_extension)); i++)
    {
        shared_data[threadIdx.x + i * blockDim.x] =
            src[threadIdx.x + i * blockDim.x + blockIdx.x * elements_per_block];
    }
    __syncthreads();

    // STEP 2: First process butterfly operations for large masks using shared memory
    // Each layer operates on all elements at once before moving to the next layer
    // TODO: Read twiddles to shared memory as well.
    for (int layer = n_layers - 1; layer >= 0; layer--)
    {
        const int mask = 1 << layer;

        // Perform butterfly operations for all elements at this layer
        for (int i = 0; i < (1 << log_extension); i++)
        {
            int ethread_idx = threadIdx.x + i * blockDim.x;
            // Insert a 0 at bit `layer` to get the element index.
            int element_idx = ethread_idx + (~(ethread_idx & (mask - 1)));
            int paired_idx = element_idx ^ mask;

            int twiddle_idx = (ethread_idx & (twiddle_size - 1)) >> layer;
            unsigned int tw = twiddle_dbl[n_layers - 1 - layer][twiddle_idx];

            // Butterfly on shared memory.
            butterfly(shared_data[element_idx], shared_data[paired_idx], tw);
        }
        if (layer > 5)
        {
            __syncthreads();
        }
    }

    // STEP 3: Write back the results to global memory
    for (int i = 0; i < (1 << (1 + log_extension)); i++)
    {
        dst[threadIdx.x + i * blockDim.x + blockIdx.x * elements_per_block] =
            shared_data[threadIdx.x + i * blockDim.x];
    }
}

// Transpose.
extern "C" __global__ void transpose_vectors(unsigned int *data, int log_size)
{
    // block b0 c b1, thread x0 x1 transposes b0 x0 c b1 x1 <-> b1 x1 c b0 x0.
    // x0, x1 are 5 bits each (32 warps).
    // c is either 0 or 1, depending on the parity of log_size.
    // b0 is (log_size - c - 5)/2.
    // Assumption: log_size is least 10.
    const int x0 = threadIdx.x & 0x1F; // 5 bits
    const int x1 = threadIdx.x >> 5;   // 5 bits
    const int c_bits = log_size & 1;
    const int b_bits = (log_size - c_bits - 10) >> 1; // 5 bits
    const int b1 = blockIdx.x & ((1 << b_bits) - 1);  // b1 is the first 5 bits of blockIdx.x.
    const int c = (blockIdx.x >> b_bits) & 1;         // c is the next bit of blockIdx.x.
    const int b0 = blockIdx.x >> (b_bits + 1);        // b0 is the rest of blockIdx.x.

    if (b0 > b1)
    {
        return;
    }

    const int b0x0 = (b0 << 5) | x0;   // b0x0 is the last 10 bits of the index.
    const int b1x1 = (b1 << 5) | x1;   // b1x1 is the first 10 bits of the index.
    const int b0x1 = (b0 << 5) | x1;   // b0x1 is the last 10 bits of the index.
    const int b1x0 = (b1 << 5) | x0;   // b1x0 is the first 10 bits of the index.
    const int bx_bits = log_size >> 1; // Number of bits in b0 and b1.

    __shared__ unsigned int shared_data0[1 << 10]; // 4K bytes
    __shared__ unsigned int shared_data1[1 << 10]; // 4K bytes

    // Coalesced read data into shared memory.
    shared_data0[(x1 << 5) | x0] = data[(b1x1 << (bx_bits + c_bits)) | (c << bx_bits) | b0x0];
    if (b1 != b0)
    {
        shared_data1[(x1 << 5) | x0] = data[(b0x1 << (bx_bits + c_bits)) | (c << bx_bits) | b1x0];
    }
    // Sync thread in the block.
    __syncthreads();
    // Coalesced write back the transposed data to global memory.
    data[(b1x1 << (bx_bits + c_bits)) | (c << bx_bits) | b0x0] = shared_data0[(x0 << 5) | x1];
    if (b1 != b0)
    {
        // Avoid writing the same data twice.
        data[(b0x1 << (bx_bits + c_bits)) | (c << bx_bits) | b1x0] = shared_data1[(x0 << 5) | x1];
    }
}
