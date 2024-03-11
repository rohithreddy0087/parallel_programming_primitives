/*
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define BLOCK_DIM 1024


__global__ void SimpleReductionKernel(int* input, int* output, int size) {

    unsigned int tx = threadIdx.x;
    unsigned int i = 2*tx;

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        
        if (tx % stride == 0) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }

    if (tx == 0) {
        output[blockIdx.x] = input[0];
    }


__global__ void ReductionKernel(int* input, int* output, int size) {
    __shared__ int input_s[BLOCK_DIM];

    unsigned int tx = threadIdx.x;
    input_s[tx] = input[tx] + input[tx + BLOCK_DIM];
    
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tx < stride) {
            input_s[tx] += input_s[tx + stride];
        }
    }

    if (tx == 0) {
        output[blockIdx.x] = input_s[0];
    }
}

__global__ void SegmentationReductionKernel(int* input, int* output, int size) {
    __shared__ int input_s[BLOCK_DIM];

    unsigned int tx = threadIdx.x;
    unsigned int segment = 2*blockIdx.x*blockDim.x;
    input_s[tx] = input[segment + tx] + input[segment + tx + BLOCK_DIM];

    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tx < stride) {
            input_s[tx] += input_s[tx + stride];
        }
    }

    if (tx == 0) {
        atomicAdd(output, input_s[0]);
    }
}

int main() {
    const int gridSize = 16;
    const int arraySize = 2048* gridSize;
    
    int* h_input = new int[arraySize];
    for (int i = 0; i < arraySize; ++i) {
        h_input[i] = i + 1;
    }

    int* d_input, * d_output;
    cudaMalloc((void**)&d_input, sizeof(int) * arraySize);
    cudaMalloc((void**)&d_output, sizeof(int));

    cudaMemcpy(d_input, h_input, sizeof(int) * arraySize, cudaMemcpyHostToDevice);

    //SimpleReductionKernel << <gridSize, BLOCK_DIM >> > (d_input, d_output, arraySize);
    //ReductionKernel << <gridSize, BLOCK_DIM >> > (d_input, d_output, arraySize);
    SegmentationReductionKernel <<<gridSize, BLOCK_DIM >> > (d_input, d_output, arraySize);

    int* h_output = new int[1];
    cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sum: %d\n", h_output[0]);

    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
*/