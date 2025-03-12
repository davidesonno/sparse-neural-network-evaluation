# OpenMP Program for Feedforward Neural Network

## Overview

This program implements a feedforward neural network using OpenMP to parallelize computations on the CPU. The network processes inputs through multiple layers.

The program uses a macro defined in `network.h` to control the unroll factor used by the function. A value of `4` or `8` will unroll that many operaitons, any other value will execute the full loop.

## Input

The program accepts the following command-line arguments:

1. **N**: A comma-separated list of input sizes.
2. **K**: A comma-separated list of layer counts.
3. **T**: (Optional) A comma-separated list of thread counts to use (default is 4).
4. **-s**: (Optional) Flag for strong scaling evaluation.
5. **-w**: (Optional) Flag for weak scaling evaluation.
6. **NREPS**: (Optional) Number of repetitions for each configuration (default is 5).

If `-s` or `-w` are not selected, N and K must be singular integers.

Example:

```
./omp-program 1024,2048 10,20,30 1,2,3,4 -s 10
```

## Output

The program prints a `csv` containing:

* **Input size and number of layers**
* **Timings**
* **Efficencies**

Example:

```
N,K,p,t1,t2,t3,t4,t5,Average,S(p),E(p)
150000,1000,1,13.946900,13.946070,13.941201,13.922192,14.083744,13.968021,1.000000,1.000000
150000,1000,2,7.101936,7.013766,6.998404,7.006580,7.148958,7.053929,1.980176,0.990088
150000,1000,3,4.765833,4.741761,4.674997,4.685953,4.649044,4.703518,2.969697,0.989899
150000,1000,4,3.503188,3.516616,3.514071,3.512862,3.517590,3.512866,3.976247,0.994062
150000,1000,5,2.819326,2.843022,2.851836,2.853299,2.815417,2.836580,4.924247,0.984849
150000,1000,6,2.354147,2.380099,2.365484,2.345358,2.345761,2.358170,5.923246,0.987208
150000,1000,7,2.031657,2.056367,2.056397,2.049894,2.047286,2.048320,6.819257,0.974180
150000,1000,8,1.806650,1.788765,1.786044,1.784638,1.771154,1.787450,7.814496,0.976812
```

## Execution Instructions

1. **Compile the Program**:

   ```bash
   gcc -o omp-program -std=c99 -Wall -Wpedantic -fopenmp main.c src/*.c -lm
   ```
2. **Run the Program**:

   ```bash
   ./omp-program [N1,N2,...,Nn] [K1,K2,...,Kj] [T1,T2,...,Tk; Default 4] [[[-s | -w] [NREPS; Default 5]] | empty; Default empty]
   ```
   or by executing the job with the bash script `omp-program.sh`

   ```bash
   bash ./omp-program.sh [N1,N2,...,Nn] [K1,K2,...,Kj] [T1,T2,...,Tk; Default 4] [[[-s | -w] [NREPS; Default 5]] | empty; Default empty]
   ```
   The same script could be also submitted as a job using `sbatch`. In this case, check the SLURM options inside the bash script.
