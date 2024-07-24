#include <stdint.h>
#include <stdio.h>
#include <cuda.h>
#include <cub/cub.cuh>
#include "scan.cuh"

#define __global___FUNC extern "C" __global__

/// Assert that a CUDA operation is correctly issued
#define cuda_check(err) cuda_check_impl(err, __FILE__, __LINE__)

__global__ void fill(uint32_t *target, uint32_t size) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
         i += blockDim.x * gridDim.x)
        target[i] = i;
}

void cuda_check_impl(cudaError_t errval, const char *file, const int line) {
    if (errval != cudaSuccess && errval != cudaErrorCudartUnloading)
        fprintf(stderr, "cuda_check(): runtime API error = %04d \"%s\" in "
                 "%s:%i.\n", (int) errval, cudaGetErrorName(errval), file, line);
}

void cuda_check_impl(CUresult errval, const char *file, const int line) {
    if (errval != CUDA_SUCCESS && errval != CUDA_ERROR_DEINITIALIZED) {
        const char *msg = nullptr;
        cuGetErrorString(errval, &msg);
        fprintf(stderr, "cuda_check(): API error = %04d (\"%s\") in "
                 "%s:%i.", (int) errval, msg, file, line);
    }
}

uint32_t round_pow2(uint32_t x) {
    x -= 1;
    x |= x >> 1;   x |= x >> 2;
    x |= x >> 4;   x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

/// Exclusive prefix sum (32 -> 32 bit)
void jit_scan_u32(const uint32_t *in, uint32_t *out, uint32_t size) {
    uint32_t items_per_thread = 16,
             thread_count     = 128,
             items_per_block  = items_per_thread * thread_count,
             block_count      = (size + items_per_block - 1) / items_per_block,
             shared_size      = items_per_block * sizeof(uint32_t);

    printf("jit_scan(%p -> %p , size=%u)\n", in, out, size);

    uint64_t *block_sums;
    cuda_check(cudaMalloc(&block_sums, (block_count + 32) * sizeof(uint64_t)));
    scan_large_u32_init<<<72, 1024>>>(block_sums, block_count + 32);
    scan_large_u32<<<block_count, thread_count, shared_size>>>(in, out, block_sums + 32);
    cudaFree(block_sums);
}

void jit_scan_u32(const uint32_t *in, uint32_t *out, uint32_t size) {
    uint32_t items_per_thread = 16,
             thread_count     = 128,
             items_per_block  = items_per_thread * thread_count,
             block_count      = (size + items_per_block - 1) / items_per_block,
             shared_size      = items_per_block * sizeof(uint32_t);

    printf("jit_scan(%p -> %p , size=%u)\n", in, out, size);

    scan_large_u32<<<block_count, thread_count, shared_size>>>(in, out, size);
}

int main(int argc, char **argv) {
    size_t size = 1024*1024*512;
    uint32_t *ptr, *ptr_out, *ptr_out_2;
    uint32_t *ptr_cpu, *ptr_cpu_2;

    cuda_check(cudaMalloc(&ptr, size * sizeof(uint32_t)));
    cuda_check(cudaMalloc(&ptr_out, size * sizeof(uint32_t)));
    cuda_check(cudaMalloc(&ptr_out_2, size * sizeof(uint32_t)));
    fill<<<72,512>>>(ptr, size);

    void *temp = nullptr;
    size_t temp_size;
    printf("Processing %zu elements.\n", size);
    cuda_check(cub::DeviceScan::ExclusiveSum(temp, temp_size, ptr, ptr_out, size));
    cuda_check(cudaMalloc(&temp, temp_size));
    cuda_check(cub::DeviceScan::ExclusiveSum(temp, temp_size, ptr, ptr_out, size));
    cuda_check(cub::DeviceScan::ExclusiveSum(temp, temp_size, ptr, ptr_out, size));
    jit_scan_u32_fast(ptr, ptr_out_2, size);
    jit_scan_u32_fast(ptr, ptr_out_2, size);

    cuda_check(cudaMallocManaged(&ptr_cpu, size * sizeof(uint32_t)));
    cuda_check(cudaMallocManaged(&ptr_cpu_2, size * sizeof(uint32_t)));
    cuda_check(cudaMemcpy(ptr_cpu, ptr_out, size * sizeof(uint32_t), cudaMemcpyDefault));
    cuda_check(cudaMemcpy(ptr_cpu_2, ptr_out_2, size * sizeof(uint32_t), cudaMemcpyDefault));
    cuda_check(cudaFree(ptr));
    cuda_check(cudaFree(ptr_out));
    cuda_check(cudaFree(ptr_out_2));
    int errors = 0;
    for (int i = 0; i < size; ++i) {
        if (ptr_cpu[i] != ptr_cpu_2[i]) {
            printf("%u %u %u\n", (uint32_t) i, ptr_cpu[i], ptr_cpu_2[i]);
            if (++errors > 10)
                break;
        }
    }
    printf("%i\n", memcmp(ptr_cpu, ptr_cpu_2, size * sizeof(uint32_t)));

    return 0;
}
