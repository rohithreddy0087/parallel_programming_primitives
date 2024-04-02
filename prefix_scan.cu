#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define BLOCK_DIM 16

__global__ void koggeStoneKernel(float* input, float* output, int n) {
    __shared__ float temp[BLOCK_DIM];
    unsigned int tx = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tx;

    if (i < n) {
        temp[tx] = input[i];
    }
    else {
        temp[tx] = 0.0f;
    }
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float t;
        if (tx >= stride) {
            t = temp[tx] + temp[tx - stride];
        }
        __syncthreads();

        if (tx >= stride) {
            temp[tx] = t;
        }
    }
    if (i < n) {
        output[i] = temp[tx];
    }

}


int main() {
    
    const int arraySize = 16;
    const int gridSize = arraySize/BLOCK_DIM;

    float h_input[arraySize];
    for (int i = 0; i < arraySize; ++i) {
        h_input[i] = 1.0f;
    }

    float* d_input, * d_output;
    cudaMalloc((void**)&d_input, sizeof(float) * arraySize);
    cudaMalloc((void**)&d_output, sizeof(float)* arraySize);

    cudaMemcpy(d_input, h_input, sizeof(float) * arraySize, cudaMemcpyHostToDevice);

    koggeStoneKernel <<<gridSize, BLOCK_DIM >>> (d_input, d_output, arraySize);
    float h_output[arraySize];
    cudaMemcpy(h_output, d_output, sizeof(float)* arraySize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < arraySize; ++i) {
        printf("%f, ", h_output[i]);
    }

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
