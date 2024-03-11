#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define BLOCK_DIM 16


__global__ void koggeStoneKernel(unsigned int* input, int n) {
    __shared__ unsigned int temp[BLOCK_DIM];
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
        int t;
        if (tx >= stride) {
            t = temp[tx] + temp[tx - stride];
        }
        __syncthreads();

        if (tx >= stride) {
            temp[tx] = t;
        }
    }

    if (i < n) {
        input[i] = (tx > 0) ? temp[tx - 1] : 0.0f;
    }
}

__global__ void radixSortIterPt1(unsigned int* input, unsigned int* output, unsigned int* bits, unsigned int n, unsigned int iter) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int key, bit;
    if (i < n) {
        key = input[i];
        bit = (key >> iter) & 1;
        bits[i] = bit;
    }

    
}

__global__ void radixSortIterPt2(unsigned int* input, unsigned int* output, unsigned int* bits, unsigned int n, unsigned int iter) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int key, bit;
    if (i < n) {
        key = input[i];
        bit = (key >> iter) & 1;

        unsigned int numOnesBefore = bits[i];
        unsigned int numOnesTotal = bits[n];
        unsigned int dst = (bit == 0) ? (i - numOnesBefore) : (n - numOnesTotal - numOnesBefore);
        output[dst] = key;
    }
}


void radixSort(unsigned int* input, unsigned int* output, unsigned int n) {
    unsigned int numBits = 1;
    unsigned int* d_input;
    unsigned int* d_output;
    unsigned int* d_bits;

    cudaMalloc(&d_input, n * sizeof(unsigned int));
    cudaMalloc(&d_bits, n * sizeof(unsigned int));
    cudaMalloc(&d_output, n * sizeof(unsigned int));
    cudaMemcpy(d_input, input, n * sizeof(unsigned int), cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_DIM);
    dim3 gridSize(1);

    for (unsigned int iter = 0; iter < sizeof(unsigned int) * 4; iter += numBits) {
        radixSortIterPt1 << <gridSize, blockSize >> > (d_input, d_output, d_bits, n, iter);
        koggeStoneKernel<<< gridSize, blockSize >>>(d_bits, n);
        radixSortIterPt2 << <gridSize, blockSize >> > (d_input, d_output, d_bits, n, iter);
        unsigned int* temp = d_input;
        d_input = d_output;
        d_output = temp;
    }

    cudaMemcpy(output, d_output, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}


int main() {
    unsigned int input[16] = { 3,2,5,7,1,0,4,6,2,1,3,2,4,5,1,2};
    unsigned int output[16];
    unsigned int N = 16;

    radixSort(input, output, N);

    for (int i = 0; i < N; i++) {
        printf("%d, ", output[i]);
    }

    return 0;
}