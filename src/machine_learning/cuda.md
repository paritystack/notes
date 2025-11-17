# CUDA Programming

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and programming model that enables dramatic increases in computing performance by harnessing the power of Graphics Processing Units (GPUs).

## Table of Contents

1. [Introduction](#introduction)
2. [CUDA Architecture](#cuda-architecture)
3. [Programming Model](#programming-model)
4. [Memory Hierarchy](#memory-hierarchy)
5. [Common Patterns](#common-patterns)
6. [Optimization Techniques](#optimization-techniques)
7. [Advanced Topics](#advanced-topics)
8. [Libraries and Tools](#libraries-and-tools)
9. [Best Practices](#best-practices)

## Introduction

### What is CUDA?

CUDA enables developers to accelerate compute-intensive applications by offloading parallel computations to NVIDIA GPUs. Unlike traditional CPU programming, CUDA allows thousands of threads to execute simultaneously.

**Key Benefits:**
- Massive parallelism (thousands of cores)
- High memory bandwidth
- Specialized hardware for compute operations
- Rich ecosystem of libraries
- Integration with popular frameworks (PyTorch, TensorFlow)

### Setup and Installation

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Install CUDA Toolkit (Ubuntu)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda

# Set environment variables
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Basic Compilation

```bash
# Compile CUDA program
nvcc program.cu -o program

# With optimization
nvcc -O3 program.cu -o program

# Specify architecture
nvcc -arch=sm_80 program.cu -o program

# Debug mode
nvcc -g -G program.cu -o program

# Link with libraries
nvcc program.cu -o program -lcublas -lcudnn
```

## CUDA Architecture

### GPU Hardware Architecture

**Streaming Multiprocessors (SMs):**
- Multiple SMs per GPU (e.g., 68 SMs on A100)
- Each SM contains:
  - CUDA cores (FP32/FP64)
  - Tensor cores (matrix operations)
  - Special function units
  - Warp schedulers
  - Shared memory and L1 cache

**Memory System:**
```
┌─────────────────────────────────────┐
│         GPU Device Memory            │
│  (Global Memory: GB scale)          │
└─────────────────────────────────────┘
            ↑
            │
┌─────────────────────────────────────┐
│           L2 Cache                   │
│        (MB scale)                    │
└─────────────────────────────────────┘
            ↑
            │
┌─────────────────────────────────────┐
│  SM    SM    SM    SM    SM         │
│  ┌─┐  ┌─┐  ┌─┐  ┌─┐  ┌─┐          │
│  │L1│  │L1│  │L1│  │L1│  │L1│      │
│  │/S│  │/S│  │/S│  │/S│  │/S│      │ L1 Cache/Shared Memory
│  │M │  │M │  │M │  │M │  │M │      │ (KB scale per SM)
│  └─┘  └─┘  └─┘  └─┘  └─┘          │
└─────────────────────────────────────┘
```

### Compute Capability

Different GPU architectures have different capabilities:

| Architecture | Compute Capability | Key Features |
|--------------|-------------------|--------------|
| Volta | 7.0 | Tensor Cores, Independent Thread Scheduling |
| Turing | 7.5 | RT Cores, INT8 Tensor Cores |
| Ampere | 8.0, 8.6 | 3rd Gen Tensor Cores, Sparsity |
| Hopper | 9.0 | 4th Gen Tensor Cores, Thread Block Clusters |

```cpp
// Query device properties
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("Device: %s\n", prop.name);
printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
printf("Multiprocessors: %d\n", prop.multiProcessorCount);
printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
printf("Warp size: %d\n", prop.warpSize);
printf("Global memory: %.2f GB\n", prop.totalGlobalMem / 1e9);
printf("Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
```

## Programming Model

### Thread Hierarchy

CUDA organizes threads in a three-level hierarchy:

```
Grid
├── Block (0,0,0)
│   ├── Thread (0,0,0)
│   ├── Thread (1,0,0)
│   └── ...
├── Block (1,0,0)
│   └── ...
└── Block (gridDim-1)
    └── ...
```

**Key Concepts:**
- **Thread**: Basic execution unit
- **Warp**: Group of 32 threads executing together (SIMT)
- **Block**: Group of threads (up to 1024) sharing shared memory
- **Grid**: Collection of blocks

### Basic Kernel Structure

```cpp
// Kernel definition
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1000000;
    size_t size = n * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize host data
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // Copy result back
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```

### Thread Indexing

```cpp
// 1D indexing
__global__ void kernel1D() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
}

// 2D indexing
__global__ void kernel2D() {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;  // Row-major order
}

// 3D indexing
__global__ void kernel3D() {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = z * width * height + y * width + x;
}

// Launch examples
dim3 blockSize(16, 16);
dim3 gridSize((width + 15) / 16, (height + 15) / 16);
kernel2D<<<gridSize, blockSize>>>(...);
```

### Error Handling

```cpp
// Macro for checking CUDA errors
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Usage
CUDA_CHECK(cudaMalloc(&d_a, size));
CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));

// Check kernel launch errors
kernel<<<grid, block>>>(...);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

## Memory Hierarchy

### Memory Types and Characteristics

| Memory Type | Location | Cached | Access | Scope | Lifetime |
|-------------|----------|---------|--------|-------|----------|
| Register | On-chip | N/A | R/W | Thread | Thread |
| Local | Off-chip | L1/L2 | R/W | Thread | Thread |
| Shared | On-chip | N/A | R/W | Block | Block |
| Global | Off-chip | L1/L2 | R/W | Grid | Application |
| Constant | Off-chip | Yes | R | Grid | Application |
| Texture | Off-chip | Yes | R | Grid | Application |

### Global Memory

```cpp
// Basic allocation
float *d_data;
cudaMalloc(&d_data, size);

// Pitched allocation (for 2D arrays)
float *d_matrix;
size_t pitch;
cudaMallocPitch(&d_matrix, &pitch, width * sizeof(float), height);

// Access in kernel
__global__ void kernel(float *matrix, size_t pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    float *row = (float*)((char*)matrix + y * pitch);
    row[x] = ...;
}

// 3D allocation
cudaExtent extent = make_cudaExtent(width, height, depth);
cudaPitchedPtr devPitchedPtr;
cudaMalloc3D(&devPitchedPtr, extent);

// Zero initialization
cudaMemset(d_data, 0, size);
```

### Shared Memory

Shared memory is fast on-chip memory shared by threads in a block:

```cpp
// Static shared memory
__global__ void kernel() {
    __shared__ float s_data[256];

    int tid = threadIdx.x;
    s_data[tid] = ...;  // Each thread writes

    __syncthreads();  // Synchronize before reading

    float value = s_data[tid];  // Read
}

// Dynamic shared memory
__global__ void kernel(int n) {
    extern __shared__ float s_data[];  // Size specified at launch

    int tid = threadIdx.x;
    s_data[tid] = ...;
    __syncthreads();
}

// Launch with dynamic shared memory
int sharedMemSize = blockSize * sizeof(float);
kernel<<<gridSize, blockSize, sharedMemSize>>>(...);

// Multiple dynamic arrays
extern __shared__ char shared_mem[];
float *s_float = (float*)shared_mem;
int *s_int = (int*)&s_float[float_size];
```

### Memory Coalescing

Coalesced memory accesses are critical for performance:

```cpp
// GOOD: Coalesced access (sequential)
__global__ void coalescedAccess(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[idx];  // Each thread accesses consecutive elements
}

// BAD: Strided access
__global__ void stridedAccess(float *data, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    float value = data[idx];  // Large gaps between accesses
}

// GOOD: Structure of Arrays (SoA)
struct SoA {
    float *x;
    float *y;
    float *z;
};

__global__ void processCoalesced(SoA data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data.x[idx];  // Coalesced
        float y = data.y[idx];  // Coalesced
        float z = data.z[idx];  // Coalesced
    }
}

// BAD: Array of Structures (AoS)
struct Point { float x, y, z; };

__global__ void processUncoalesced(Point *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx].x;  // Not coalesced
    }
}
```

### Constant Memory

```cpp
// Constant memory declaration (64KB limit)
__constant__ float c_coefficients[1024];

// Copy to constant memory
float h_coefficients[1024];
cudaMemcpyToSymbol(c_coefficients, h_coefficients, sizeof(h_coefficients));

// Use in kernel (cached, broadcast)
__global__ void kernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float coeff = c_coefficients[idx % 1024];  // Fast cached access
}
```

### Unified Memory

```cpp
// Allocate unified memory (accessible from both CPU and GPU)
float *data;
cudaMallocManaged(&data, size);

// Can access from CPU
data[0] = 1.0f;

// Can access from GPU
kernel<<<grid, block>>>(data);
cudaDeviceSynchronize();

// Access from CPU again
float result = data[0];

// Free
cudaFree(data);

// Memory prefetching
cudaMemPrefetchAsync(data, size, deviceId);  // Prefetch to GPU
cudaMemPrefetchAsync(data, size, cudaCpuDeviceId);  // Prefetch to CPU

// Memory advise
cudaMemAdvise(data, size, cudaMemAdviseSetReadMostly, deviceId);
cudaMemAdvise(data, size, cudaMemAdviseSetPreferredLocation, deviceId);
```

## Common Patterns

### 1. Vector Addition

```cpp
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Launch
int blockSize = 256;
int gridSize = (n + blockSize - 1) / blockSize;
vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
```

### 2. Matrix Multiplication (Naive)

```cpp
// C = A * B
// A: M x K, B: K x N, C: M x N
__global__ void matmulNaive(const float *A, const float *B, float *C,
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Launch
dim3 blockSize(16, 16);
dim3 gridSize((N + 15) / 16, (M + 15) / 16);
matmulNaive<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
```

### 3. Matrix Multiplication (Tiled with Shared Memory)

```cpp
#define TILE_SIZE 16

__global__ void matmulTiled(const float *A, const float *B, float *C,
                            int M, int N, int K) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile into shared memory
        if (row < M && t * TILE_SIZE + tx < K)
            s_A[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        else
            s_A[ty][tx] = 0.0f;

        if (t * TILE_SIZE + ty < K && col < N)
            s_B[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            s_B[ty][tx] = 0.0f;

        __syncthreads();

        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += s_A[ty][k] * s_B[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### 4. Reduction (Sum)

```cpp
// Parallel reduction in shared memory
__global__ void reduce(const float *input, float *output, int n) {
    extern __shared__ float s_data[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    s_data[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = s_data[0];
    }
}

// Optimized reduction (avoiding bank conflicts)
__global__ void reduceOptimized(const float *input, float *output, int n) {
    extern __shared__ float s_data[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Load and perform first level of reduction during load
    s_data[tid] = 0.0f;
    if (idx < n) s_data[tid] += input[idx];
    if (idx + blockDim.x < n) s_data[tid] += input[idx + blockDim.x];
    __syncthreads();

    // Reduction with sequential addressing
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = s_data[0];
}
```

### 5. Scan (Prefix Sum)

```cpp
// Inclusive scan using Blelloch algorithm
__global__ void scanBlelloch(float *data, int n) {
    extern __shared__ float temp[];

    int tid = threadIdx.x;
    int offset = 1;

    // Load input into shared memory
    temp[2 * tid] = data[2 * tid];
    temp[2 * tid + 1] = data[2 * tid + 1];

    // Build sum tree
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Clear last element
    if (tid == 0) temp[n - 1] = 0;

    // Traverse down tree and build scan
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // Write results
    data[2 * tid] = temp[2 * tid];
    data[2 * tid + 1] = temp[2 * tid + 1];
}
```

### 6. Histogram

```cpp
// Atomic histogram
__global__ void histogram(const int *data, int *hist, int n, int numBins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int bin = data[idx] % numBins;
        atomicAdd(&hist[bin], 1);
    }
}

// Optimized with shared memory
__global__ void histogramShared(const int *data, int *hist, int n, int numBins) {
    extern __shared__ int s_hist[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared histogram
    for (int i = tid; i < numBins; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Accumulate in shared memory
    if (idx < n) {
        int bin = data[idx] % numBins;
        atomicAdd(&s_hist[bin], 1);
    }
    __syncthreads();

    // Write to global memory
    for (int i = tid; i < numBins; i += blockDim.x) {
        atomicAdd(&hist[i], s_hist[i]);
    }
}
```

### 7. Transpose

```cpp
// Naive transpose
__global__ void transposeNaive(const float *input, float *output,
                               int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        output[x * height + y] = input[y * width + x];
    }
}

// Optimized with shared memory (no bank conflicts)
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transposeCoalesced(const float *input, float *output,
                                   int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Coalesced read from global memory
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * width + x];
        }
    }

    __syncthreads();

    // Transpose block indices
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Coalesced write to global memory
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            output[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}
```

### 8. Convolution (1D)

```cpp
#define KERNEL_RADIUS 3
__constant__ float c_kernel[2 * KERNEL_RADIUS + 1];

__global__ void convolution1D(const float *input, float *output, int n) {
    extern __shared__ float s_data[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load data into shared memory with halo
    int halo_idx_left = (blockIdx.x - 1) * blockDim.x + tid;
    int halo_idx_right = (blockIdx.x + 1) * blockDim.x + tid;

    // Main data
    if (idx < n) {
        s_data[tid + KERNEL_RADIUS] = input[idx];
    }

    // Left halo
    if (tid < KERNEL_RADIUS) {
        s_data[tid] = (halo_idx_left >= 0) ? input[halo_idx_left] : 0.0f;
    }

    // Right halo
    if (tid >= blockDim.x - KERNEL_RADIUS) {
        int offset = tid - (blockDim.x - KERNEL_RADIUS);
        s_data[tid + 2 * KERNEL_RADIUS] =
            (halo_idx_right < n) ? input[halo_idx_right] : 0.0f;
    }

    __syncthreads();

    // Convolution
    if (idx < n) {
        float sum = 0.0f;
        for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++) {
            sum += s_data[tid + KERNEL_RADIUS + k] * c_kernel[k + KERNEL_RADIUS];
        }
        output[idx] = sum;
    }
}
```

## Optimization Techniques

### 1. Occupancy Optimization

```cpp
// Check theoretical occupancy
int blockSize = 256;
int minGridSize;
int maxBlockSize;

// Get optimal launch configuration
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &maxBlockSize,
                                   kernel, 0, 0);

// Calculate occupancy
int numBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, kernel,
                                              blockSize, 0);

// Launch with optimal configuration
int gridSize = (n + maxBlockSize - 1) / maxBlockSize;
kernel<<<gridSize, maxBlockSize>>>(args);
```

### 2. Warp-Level Primitives

```cpp
// Warp shuffle
__global__ void warpReduce(float *data) {
    int tid = threadIdx.x;
    float val = data[tid];

    // Warp-level reduction (no __syncthreads needed)
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if ((tid % 32) == 0) {
        data[tid / 32] = val;
    }
}

// Warp vote functions
__global__ void warpVote() {
    int tid = threadIdx.x;
    int value = tid % 2;

    // Check if all threads in warp have value == 1
    bool all_true = __all_sync(0xffffffff, value);

    // Check if any thread in warp has value == 1
    bool any_true = __any_sync(0xffffffff, value);

    // Count threads in warp with value == 1
    int count = __popc(__ballot_sync(0xffffffff, value));
}
```

### 3. Avoiding Bank Conflicts

```cpp
// BAD: Bank conflicts
__shared__ float s_data[32][32];
s_data[threadIdx.x][threadIdx.y] = ...;  // Conflicts when threadIdx.x varies

// GOOD: Add padding to avoid conflicts
__shared__ float s_data[32][33];  // Extra column eliminates conflicts
s_data[threadIdx.x][threadIdx.y] = ...;

// Access pattern analysis
// Each bank serves one 32-bit word per cycle
// Bank index = (address / 4) % 32
// Conflict occurs when multiple threads access same bank
```

### 4. Asynchronous Operations

```cpp
// Create CUDA streams
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Overlap computation and data transfer
for (int i = 0; i < nStreams; i++) {
    int offset = i * streamSize;

    // Async copy H2D
    cudaMemcpyAsync(&d_data[offset], &h_data[offset], streamBytes,
                    cudaMemcpyHostToDevice, stream[i]);

    // Launch kernel
    kernel<<<grid, block, 0, stream[i]>>>(&d_data[offset], ...);

    // Async copy D2H
    cudaMemcpyAsync(&h_result[offset], &d_result[offset], streamBytes,
                    cudaMemcpyDeviceToHost, stream[i]);
}

// Wait for all streams
cudaDeviceSynchronize();

// Cleanup
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

### 5. Memory Access Patterns

```cpp
// Benchmark different access patterns
__global__ void sequentialAccess(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];  // Coalesced: ~900 GB/s
    }
}

__global__ void stridedAccess(float *data, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (idx < n) {
        float val = data[idx];  // Non-coalesced: ~100 GB/s
    }
}

__global__ void randomAccess(float *data, int *indices, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[indices[idx]];  // Random: ~50 GB/s
    }
}
```

### 6. Loop Unrolling

```cpp
// Manual loop unrolling
__global__ void matmulUnrolled(const float *A, const float *B, float *C,
                               int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        // Unroll by 4
        int k;
        for (k = 0; k < K - 3; k += 4) {
            sum += A[row * K + k] * B[k * N + col];
            sum += A[row * K + k + 1] * B[(k + 1) * N + col];
            sum += A[row * K + k + 2] * B[(k + 2) * N + col];
            sum += A[row * K + k + 3] * B[(k + 3) * N + col];
        }

        // Handle remainder
        for (; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

// Pragma unroll
__global__ void kernel() {
    #pragma unroll 8
    for (int i = 0; i < ITERATIONS; i++) {
        // Loop body
    }
}
```

## Advanced Topics

### 1. Dynamic Parallelism

```cpp
// Parent kernel launches child kernels
__global__ void childKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

__global__ void parentKernel(float *data, int n, int depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0 && depth > 0) {
        // Launch child kernel from GPU
        int childBlocks = (n + 255) / 256;
        childKernel<<<childBlocks, 256>>>(data, n);

        // Synchronize child kernel
        cudaDeviceSynchronize();

        // Recursive launch
        parentKernel<<<1, 1>>>(data, n, depth - 1);
    }
}

// Compile with: nvcc -arch=sm_35 -rdc=true -lcudadevrt
```

### 2. Cooperative Groups

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Thread block group
__global__ void kernelWithCG() {
    cg::thread_block block = cg::this_thread_block();

    // Synchronize block
    block.sync();

    // Get block info
    int rank = block.thread_rank();
    int size = block.size();
}

// Tiled partition (warp-level)
__global__ void warpLevelCG() {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int value = threadIdx.x;

    // Warp-level reduction
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        value += warp.shfl_down(value, offset);
    }

    if (warp.thread_rank() == 0) {
        // First thread in warp has the sum
    }
}

// Grid-wide synchronization
__global__ void gridSync(int *data) {
    cg::grid_group grid = cg::this_grid();

    // All threads in grid must reach this point
    grid.sync();
}

// Launch with cooperative groups
void *kernelArgs[] = {&d_data};
int numBlocks = 100;
int blockSize = 256;
cudaLaunchCooperativeKernel((void*)gridSync, numBlocks, blockSize,
                            kernelArgs, 0, 0);
```

### 3. Tensor Cores

```cpp
#include <mma.h>
using namespace nvcuda;

// Matrix multiplication with Tensor Cores (WMMA API)
__global__ void wmma_matmul(half *a, half *b, float *c, int M, int N, int K) {
    // Tile dimensions (16x16x16 for half precision)
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    // Warp and lane IDs
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

    // Initialize accumulator
    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over K
    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;

        // Bounds checking
        if (aRow < M && bCol < N) {
            // Load matrices
            wmma::load_matrix_sync(a_frag, a + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, b + bRow * N + bCol, N);

            // Perform matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Store result
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(c + cRow * N + cCol, acc_frag, N, wmma::mem_row_major);
    }
}
```

### 4. CUDA Graphs

```cpp
// Create and execute CUDA graph
cudaGraph_t graph;
cudaGraphExec_t graphExec;

// Begin graph capture
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// Operations to capture
kernel1<<<grid1, block1, 0, stream>>>(args1);
cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
kernel2<<<grid2, block2, 0, stream>>>(args2);

// End capture
cudaStreamEndCapture(stream, &graph);

// Instantiate graph
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

// Execute graph (can be launched multiple times)
for (int i = 0; i < iterations; i++) {
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);
}

// Cleanup
cudaGraphExecDestroy(graphExec);
cudaGraphDestroy(graph);

// Manual graph construction
cudaGraphNode_t kernel1Node, kernel2Node, memcpyNode;
cudaKernelNodeParams kernel1Params = {};
kernel1Params.func = (void*)kernel1;
kernel1Params.gridDim = grid1;
kernel1Params.blockDim = block1;
kernel1Params.kernelParams = args1;

cudaGraphAddKernelNode(&kernel1Node, graph, NULL, 0, &kernel1Params);
```

## Libraries and Tools

### cuBLAS (Linear Algebra)

```cpp
#include <cublas_v2.h>

// Initialize cuBLAS
cublasHandle_t handle;
cublasCreate(&handle);

// Matrix multiplication: C = α*A*B + β*C
const float alpha = 1.0f;
const float beta = 0.0f;
int m = 1024, n = 1024, k = 1024;

cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            d_A, m,
            d_B, k,
            &beta,
            d_C, m);

// Vector operations
cublasSaxpy(handle, n, &alpha, d_x, 1, d_y, 1);  // y = α*x + y
cublasSdot(handle, n, d_x, 1, d_y, 1, &result);  // dot product

// Cleanup
cublasDestroy(handle);
```

### cuDNN (Deep Learning)

```cpp
#include <cudnn.h>

// Initialize cuDNN
cudnnHandle_t cudnn;
cudnnCreate(&cudnn);

// Convolution forward
cudnnTensorDescriptor_t input_desc, output_desc;
cudnnFilterDescriptor_t kernel_desc;
cudnnConvolutionDescriptor_t conv_desc;

cudnnCreateTensorDescriptor(&input_desc);
cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                           batch_size, channels, height, width);

cudnnCreateFilterDescriptor(&kernel_desc);
cudnnSetFilter4dDescriptor(kernel_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                          num_filters, channels, kernel_h, kernel_w);

cudnnCreateConvolutionDescriptor(&conv_desc);
cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, stride_h, stride_w,
                                dilation_h, dilation_w,
                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

// Find best algorithm
cudnnConvolutionFwdAlgoPerf_t perfResults;
int returnedAlgoCount;
cudnnFindConvolutionForwardAlgorithm(cudnn, input_desc, kernel_desc, conv_desc,
                                     output_desc, 1, &returnedAlgoCount, &perfResults);

// Execute convolution
const float alpha = 1.0f, beta = 0.0f;
cudnnConvolutionForward(cudnn, &alpha, input_desc, d_input,
                       kernel_desc, d_kernel, conv_desc,
                       perfResults.algo, workspace, workspace_size,
                       &beta, output_desc, d_output);

cudnnDestroy(cudnn);
```

### Thrust (C++ Template Library)

```cpp
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

// Device vectors (automatic memory management)
thrust::device_vector<float> d_vec(1000000);
thrust::fill(d_vec.begin(), d_vec.end(), 1.0f);

// Sorting
thrust::sort(d_vec.begin(), d_vec.end());

// Reduction
float sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, thrust::plus<float>());

// Transform
thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(),
                 thrust::negate<float>());

// Custom functor
struct square {
    __host__ __device__
    float operator()(const float &x) const {
        return x * x;
    }
};
thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(), square());

// Scan (prefix sum)
thrust::inclusive_scan(d_vec.begin(), d_vec.end(), d_vec.begin());

// Copy to host
thrust::host_vector<float> h_vec = d_vec;
```

### Profiling Tools

```bash
# nvprof (legacy)
nvprof ./program
nvprof --print-gpu-trace ./program
nvprof --metrics achieved_occupancy ./program

# Nsight Compute (detailed kernel profiling)
ncu --set full --export profile ./program
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed ./program

# Nsight Systems (timeline analysis)
nsys profile --stats=true ./program
nsys profile --trace=cuda,nvtx --output=report ./program

# cuda-memcheck
cuda-memcheck ./program
cuda-memcheck --tool memcheck ./program
cuda-memcheck --tool racecheck ./program
```

## Best Practices

### Performance Optimization Checklist

1. **Memory Access**
   - [ ] Use coalesced memory accesses
   - [ ] Minimize global memory accesses
   - [ ] Use shared memory for frequently accessed data
   - [ ] Avoid bank conflicts in shared memory
   - [ ] Use appropriate memory types (constant, texture)

2. **Execution Configuration**
   - [ ] Maximize occupancy
   - [ ] Use block sizes that are multiples of warp size (32)
   - [ ] Balance register usage and occupancy
   - [ ] Minimize warp divergence

3. **Compute Optimization**
   - [ ] Minimize thread divergence
   - [ ] Use fast math functions when appropriate (-use_fast_math)
   - [ ] Unroll loops when beneficial
   - [ ] Fuse kernels to reduce memory traffic

4. **Data Transfer**
   - [ ] Minimize host-device transfers
   - [ ] Use pinned memory for faster transfers
   - [ ] Overlap computation and communication with streams
   - [ ] Batch small transfers

### Common Pitfalls

```cpp
// 1. Race conditions
__global__ void badKernel(int *data) {
    int idx = threadIdx.x;
    data[0] += idx;  // WRONG: Race condition
}

__global__ void goodKernel(int *data) {
    int idx = threadIdx.x;
    atomicAdd(&data[0], idx);  // CORRECT: Atomic operation
}

// 2. Missing synchronization
__global__ void badSync() {
    __shared__ int s_data[256];
    int tid = threadIdx.x;
    s_data[tid] = tid;
    // WRONG: No synchronization
    int val = s_data[(tid + 1) % 256];  // Undefined behavior
}

__global__ void goodSync() {
    __shared__ int s_data[256];
    int tid = threadIdx.x;
    s_data[tid] = tid;
    __syncthreads();  // CORRECT: Synchronize before reading
    int val = s_data[(tid + 1) % 256];
}

// 3. Ignoring error checking
cudaMalloc(&d_ptr, size);  // BAD: No error check
kernel<<<grid, block>>>();  // BAD: No error check

// GOOD:
CUDA_CHECK(cudaMalloc(&d_ptr, size));
kernel<<<grid, block>>>();
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());

// 4. Unaligned memory access
struct BadStruct {
    char c;
    float f;  // Unaligned on device
};

struct GoodStruct {
    float f;
    char c;
    char padding[3];  // Explicit padding
};

// 5. Oversubscribing shared memory
__global__ void badShared() {
    __shared__ float s_data[10000];  // Too large!
    // Kernel may not launch or have low occupancy
}
```

### Debugging Tips

```cpp
// Use printf in kernels
__global__ void debugKernel() {
    int idx = threadIdx.x;
    printf("Thread %d: value = %d\n", idx, value);
}

// Kernel with boundary checks
__global__ void safeKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Always check bounds
    if (idx >= n) return;

    // Assertions (only in debug builds)
    assert(data[idx] >= 0.0f && "Negative value detected");

    data[idx] = sqrt(data[idx]);
}

// Compile with debug info and run with cuda-memcheck
// nvcc -g -G program.cu -o program
// cuda-memcheck ./program
```

### Memory Management Patterns

```cpp
// RAII wrapper for CUDA memory
template<typename T>
class CudaArray {
private:
    T *d_ptr;
    size_t n;

public:
    CudaArray(size_t size) : n(size) {
        CUDA_CHECK(cudaMalloc(&d_ptr, n * sizeof(T)));
    }

    ~CudaArray() {
        cudaFree(d_ptr);
    }

    void copyToDevice(const T *h_ptr) {
        CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, n * sizeof(T),
                             cudaMemcpyHostToDevice));
    }

    void copyToHost(T *h_ptr) {
        CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, n * sizeof(T),
                             cudaMemcpyDeviceToHost));
    }

    T* get() { return d_ptr; }
    size_t size() const { return n; }
};

// Usage
CudaArray<float> d_data(1000000);
d_data.copyToDevice(h_data);
kernel<<<grid, block>>>(d_data.get(), d_data.size());
d_data.copyToHost(h_result);
// Automatic cleanup when out of scope
```

## Quick Reference

### Memory Bandwidth Hierarchy

| Memory Type | Bandwidth | Latency |
|-------------|-----------|---------|
| Registers | ~20 TB/s | 1 cycle |
| Shared Memory | ~15 TB/s | ~20 cycles |
| L1 Cache | ~15 TB/s | ~30 cycles |
| L2 Cache | ~5 TB/s | ~200 cycles |
| Global Memory | ~1.5 TB/s | ~400 cycles |
| Host Memory | ~50 GB/s | ~100,000 cycles |

### Atomic Operations

```cpp
// Integer atomics
atomicAdd(&addr, val);
atomicSub(&addr, val);
atomicMin(&addr, val);
atomicMax(&addr, val);
atomicExch(&addr, val);
atomicCAS(&addr, compare, val);  // Compare and swap
atomicAnd(&addr, val);
atomicOr(&addr, val);
atomicXor(&addr, val);

// Floating-point atomics (newer GPUs)
atomicAdd(&float_addr, float_val);  // sm_20+
atomicAdd(&double_addr, double_val);  // sm_60+
```

### Grid and Block Limits

| Parameter | Limit |
|-----------|-------|
| Max threads per block | 1024 |
| Max x-dimension of block | 1024 |
| Max y/z-dimension of block | 1024 |
| Max x-dimension of grid | 2^31-1 |
| Max y/z-dimension of grid | 65535 |
| Warp size | 32 |
| Max shared memory per block | 48-163 KB (arch dependent) |

### Resources

- **Documentation**: [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- **Programming Guide**: [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- **Best Practices**: [CUDA C Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- **Samples**: CUDA SDK Samples
- **Books**:
  - "Programming Massively Parallel Processors" by Kirk & Hwu
  - "CUDA by Example" by Sanders & Kandrot
- **Online Courses**:
  - Udacity: Intro to Parallel Programming
  - Coursera: GPU Programming Specialization

This guide covers the essential aspects of CUDA programming. For specific applications in machine learning, refer to the [PyTorch](./pytorch.md), [Deep Learning](./deep_learning.md), and [Quantization](./quantization.md) documentation.
