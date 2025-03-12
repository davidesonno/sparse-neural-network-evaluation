# CUDA Program for Feedforward Neural Network

## Overview

This program implements a feedforward neural network using CUDA to accelerate computations on the GPU. The network processes inputs through multiple layers using various kernel strategies to optimize performance.

## Kernels

The program includes several CUDA kernels, each implementing a different strategy for processing the network:

- **STENCIL**: Basic stencil sum kernel.
- **GLBL_MEM** *(poor performances)*: Weighted sum with global memory access.
- **GLBL_MEM_TRAN**: Transposed weights for better memory coalescing.
- **SHRD_X_TRAN**: Shared memory for input values.
- **READ_ONLY_W_TRAN**: Shared input and read-only cache for weights.
- **SHRD_XW_TRAN** *(poor performances)*: Shared memory for both input and weights.
- **SHRD_XW_PAD_TRAN** *(poor performances)*: Shared memory for both input and padded weights.
- **GLB_MEM_STRUCT**: Structured weights for coalesced access.
- **SHRD_X_STRUCT**: Shared memory for input with structured weights.

## Input

The program accepts the following command-line arguments:

1. **N**: A comma-separated list of input sizes.
2. **K**: A comma-separated list of layer counts.
3. **NREPS**: (Optional) Number of repetitions for each configuration (default is 5).

Example:

```
./cuda-program 1000000,5000000 10000,20000 3
```

## Output

The program prints a `csv` containing:

* **Network and kernel info**
* **Input size and number of layers**
* **Kernel Timings**
* **Kernel Throughputs**
* **Kernel Speedups**

The outputs can be customized inside the `main.cu` file.

Example:

```
BLKDIM,R,N,K,t1-GLBL_MEM,t2-GLBL_MEM,t3-GLBL_MEM,t1-SHRD_X_TRAN,t2-SHRD_X_TRAN,t3-SHRD_X_TRAN,Average-GLBL_MEM,Average-SHRD_X_TRAN,Speedup-GLBL_MEM,Speedup-SHRD_X_TRAN 1024,29,1000000,100,0.173844,0.168244,0.168308,0.022867,0.022882,0.022873,0.170132,0.022874,1.000000,7.437754
```

## Execution Instructions

Compile and run the program with:

```bash
   bash cuda-program.sh [N1,N2,...,Nn] [K1,K2,...,Kj] [NREPS; Default 5]
```

The same script could be also submitted as a job using `sbatch`. In this case, check the *SLURM* options inside the bash script or overwrite them from the command line command.
