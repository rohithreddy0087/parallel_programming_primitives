# parallel_programming_primitives

## Matrix Multiplication Algorithms: Simple, Tiled, and Coarse Tiled

CUDA implementations of three matrix multiplication algorithms: Simple, Tiled, and Coarse Tiled. 

### Algorithms

#### 1. Simple Matrix Multiplication

The `simpleMatMulkernel` kernel performs a simple matrix multiplication by using a basic element-wise approach. Each thread computes a single element of the resulting matrix.

##### Pros:
- Simple and easy to understand.
- Straightforward parallelization.

##### Cons:
- Limited efficiency for large matrices due to lack of optimization.

#### 2. Tiled Matrix Multiplication

The `tiledMatMulkernel` kernel introduces a tiling strategy to improve memory access patterns and reduce global memory transactions. It divides the matrices into tiles and performs matrix multiplication on each tile.

##### Pros:
- Improved memory access patterns.
- Better performance for larger matrices compared to simple multiplication.

##### Cons:
- Requires additional shared memory for storing tile data.

#### 3. Coarse Tiled Matrix Multiplication

The `CoarsedtiledMatMulkernel` kernel further optimizes the tiled approach by incorporating thread coarsening. It processes multiple elements in parallel within each thread, reducing the total number of threads needed.

##### Pros:
- Improved memory access patterns.
- Reduced total number of threads, improving overall efficiency.

##### Cons:
- Increased complexity due to thread coarsening.
