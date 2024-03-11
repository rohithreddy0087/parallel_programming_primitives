
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.h"
#include <stdio.h>

#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 64
#define TILE_WIDTH 16
#define COARSE_FACTOR 4

void generateRandomFloatMatrix(int rows, int cols, float* matrix);
void printFloatMatrix(int rows, int cols, float* matrix);
void matmul();



__global__
void simpleMatMulkernel(float* A_d, float* B_d, float* C_d, int Width) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < Width && x < Width) {
        float P = 0.0f;
        for (int i = 0; i < Width; ++i) {
            P += A_d[y * Width + i] * B_d[i * Width + x];
        }
        C_d[y * Width + x] = P;
    }
}


__global__
void tiledMatMulkernel(float* A_d, float* B_d, float* C_d, int Width) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int y = blockIdx.y * blockDim.y + ty;
    int x = blockIdx.x * blockDim.x + tx;

    __shared__ float M[TILE_WIDTH][TILE_WIDTH], N[TILE_WIDTH][TILE_WIDTH];

    float Pvalue = 0.0f;
    for(int ph = 0; ph < Width/TILE_WIDTH; ++ph){
        
        if((y<Width) && (ph*TILE_WIDTH + tx < Width)){
            M[ty][tx] = A_d[y*Width + ph*TILE_WIDTH + tx];
        }
        else M[ty][tx] = 0;

        if((ph*TILE_WIDTH+ty) < Width && x<Width){
            N[ty][tx] = B_d[(ph*TILE_WIDTH+ty)*Width+x];
        }
        else N[ty][tx] = 0;

        __syncthreads();
        
        for(int k=0; k< TILE_WIDTH;++k){
            Pvalue += M[ty][k]*N[k][tx];
        }
        __syncthreads();
    }
    C_d[y*Width+x] = Pvalue;
}


__global__
void CoarsedtiledMatMulkernel(float *A_d, float *B_d, float *C_d, int Width){

    __shared__ float M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float N[TILE_WIDTH][TILE_WIDTH];
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int y = blockIdx.y*TILE_WIDTH + ty;
    int xStart = blockIdx.x*TILE_WIDTH* COARSE_FACTOR + tx;

    float Pvalue[COARSE_FACTOR];
    for(int c = 0; c < COARSE_FACTOR; ++c){
        Pvalue[c] = 0;
    }
    for(int ph = 0; ph < Width/TILE_WIDTH; ++ph){
        
        if((y<Width) && (ph*TILE_WIDTH + tx < Width)){
            M[ty][tx] = A_d[y*Width + ph*TILE_WIDTH + tx];
        } 
        else M[ty][tx] = 0;
        
        for(int c = 0; c < COARSE_FACTOR; ++c) {
            
            int x = xStart + c*TILE_WIDTH;
            
            if ((ph * TILE_WIDTH + ty) < Width && x < Width) {
                N[ty][tx] = B_d[(ph * TILE_WIDTH + ty) * Width + x];
            } 
            else N[ty][tx] = 0;

            __syncthreads();
            
            for (int k = 0; k < TILE_WIDTH; ++k) {
                Pvalue[c] += M[ty][k] * N[k][tx];
            }
            __syncthreads();
        }
    }
    for(int c = 0; c < COARSE_FACTOR; ++c) {
        int x = xStart + c*TILE_WIDTH;
        C_d[y*Width+x] = Pvalue[c];
    }
}

int main()
{
    
    matmul();
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
void matmul()
{
    srand(time(NULL));

    const int rows = 256;
    const int cols = 256;

    float A[rows][cols];
    float B[rows][cols];
    float C[rows][cols];

    generateRandomFloatMatrix(rows, cols, *A);
    generateRandomFloatMatrix(rows, cols, *B);

    float* A_d, * B_d, * C_d;
    int size = rows * cols * sizeof(float);

    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    //simpleMatMul(A_d, B_d, C_d, cols, rows);
    //tiledMatMul(A_d, B_d, C_d, cols, rows);
    CoarsedtiledMatMul(A_d, B_d, C_d, cols, rows);

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    /*printf("Random Float Matrix:\n");
    printFloatMatrix(rows, cols, *C);*/

}

void simpleMatMul(float* A_d, float* B_d, float* C_d, int cols, int rows) {
    clock_t start, stop;
    start = clock();
    dim3 dimGrid(ceil(cols / BLOCK_SIZE), ceil(rows / BLOCK_SIZE), 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    simpleMatMulkernel <<< dimGrid, dimBlock >>> (A_d, B_d, C_d, cols);
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Simple Matmul took %f seconds.\n", timer_seconds);
}

void tiledMatMul(float* A_d, float* B_d, float* C_d, int cols, int rows) {
    clock_t start, stop;
    start = clock();
    dim3 dimGrid(ceil(cols/TILE_WIDTH), ceil(rows/TILE_WIDTH), 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    tiledMatMulkernel <<< dimGrid, dimBlock >>> (A_d, B_d, C_d, cols);
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Tiled Matmul took %f seconds.\n", timer_seconds);
}

void CoarsedtiledMatMul(float* A_d, float* B_d, float* C_d, int cols, int rows) {
    clock_t start, stop;
    start = clock();
    dim3 dimGrid(ceil(cols / (TILE_WIDTH * COARSE_FACTOR)), ceil(rows / (TILE_WIDTH)), 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    CoarsedtiledMatMulkernel << < dimGrid, dimBlock >> > (A_d, B_d, C_d, cols);
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Thread Coarsened and Tiled Matmul took %f seconds.\n", timer_seconds);
}

float randFloat() {
    return (float)rand() / RAND_MAX;
}

void generateRandomFloatMatrix(int rows, int cols, float* matrix) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = 1.0; //randFloat();
        }
    }
}

void printFloatMatrix(int rows, int cols, float* matrix) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%0.4f\t", matrix[i * cols + j]);
        }
        printf("\n");
    }
}
