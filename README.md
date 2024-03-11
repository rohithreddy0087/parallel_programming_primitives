# Paralell Algorithms in CUDA

## Matrix Multiplication Algorithms: Simple, Tiled, and Coarse Tiled

CUDA implementations of three matrix multiplication algorithms: Simple, Tiled, and Coarse Tiled. 

### Algorithms

#### 1. Simple Matrix Multiplication

The `simpleMatMulkernel` kernel performs a simple matrix multiplication by using a basic element-wise approach. Each thread computes a single element of the resulting matrix.

#### 2. Tiled Matrix Multiplication

The `tiledMatMulkernel` kernel introduces a tiling strategy to improve memory access patterns and reduce global memory transactions. It divides the matrices into tiles and performs matrix multiplication on each tile.

#### 3. Coarse Tiled Matrix Multiplication

The `CoarsedtiledMatMulkernel` kernel further optimizes the tiled approach by incorporating thread coarsening. It processes multiple elements in parallel within each thread, reducing the total number of threads needed.


## Parallel Reduction Methods

Reduction is a common operation in parallel computing where a set of values is combined into a single result. In this case, the repository explores three different parallel reduction methods: Simple Reduction, Block-wise Reduction, and Segmented Reduction.

### Methods

#### 1. Simple Reduction

The `SimpleReductionKernel` kernel performs a basic parallel reduction by utilizing the full block of threads. Each thread in the block adds two elements from the input array and updates the result iteratively until a single value is obtained.

#### 2. Block-wise Reduction

The `ReductionKernel` kernel introduces shared memory to optimize memory access patterns. Threads within a block cooperate to reduce the partial sums, resulting in a block-wise reduction. This method reduces the global memory transactions and improves overall performance.

#### 3. Segmented Reduction

The `SegmentationReductionKernel` kernel extends the block-wise reduction to handle segmented input data. It calculates the partial sum for each segment independently and uses atomic operations to accumulate the final result.


## Parallel Sum using Kogge-Stone Algorithm

The Kogge-Stone algorithm is a parallel prefix sum (scan) algorithm that efficiently computes the sum of elements in an array in a parallel and scalable manner.

The `koggeStoneKernel` kernel performs the parallel sum operation using the Kogge-Stone algorithm. It divides the input array into blocks and computes partial sums within each block. The partial sums are then accumulated to obtain the final sum.

## Radix Sort Implementation with CUDA
 Radix Sort is a non-comparative sorting algorithm that sorts elements by processing individual digits of the numbers. The CUDA parallelization is achieved using the Kogge-Stone parallel prefix sum (scan) algorithm for efficient bit counting.

### Code Overview
1. Kogge-Stone Kernel
The koggeStoneKernel performs the parallel prefix sum (scan) using the Kogge-Stone algorithm. It computes the prefix sum of bits for each element in the array, indicating the number of set bits before each element.

2. Radix Sort Iteration - Part 1
The radixSortIterPt1 kernel is responsible for extracting the specified bit (based on the current iteration) for each element in the array and storing it in a separate array (bits).

3. Radix Sort Iteration - Part 2
The radixSortIterPt2 kernel uses the computed bits to rearrange the elements in the input array (d_input) based on the radix. It utilizes the parallel prefix sum result to determine the destination index for each element, ensuring a sorted order.

4. Radix Sort Function
The radixSort function orchestrates the entire radix sort process. It iterates through each bit position in the elements, using the Kogge-Stone kernel for parallel prefix sum and rearranging the elements accordingly.
