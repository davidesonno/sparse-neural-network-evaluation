# Sparse Feed Forward Neural Network

This project contains two bash scripts, `run-omp.sh` and `run-cuda.sh`, which can be used to execute OpenMP and CUDA implementations for the computation of the neural networks.

## General informations

In the `./cfg` folder the user can find the configurations file used to run evaluations. They follow a simple structure detailed inside the files themself.

## Usage

The scripts provided submit the job using `sbatch`. To instead run the scripts locally the user can change the command to `bash`, such that the program script are not submitted but executed.

Alternatively, the user can check the individual `README` for more informations.

In both approaches, launching the executions is very similar. There are two alternatives:

1. Launch the scripts with a configuration file
2. Launch the scripts with arbitrary command line arguments.

Another script that can be used to compare the two approeaches cn be used: `run-comparison.sh`. Input sizes are set inside the file.

### Executing OpenMP: run-omp.sh

Evaluates one or more Neural Networks of input sizes `N` and `K` layers, using `T` threads. Computes the strong or weak scaling efficencies.

To run this script, you have 2 options:

```
bash run-omp.sh [-s | -w] [Config file; Default "omp-strong.cfg" if -s, "omp-weak.cfg" if -w]
```

or

```
bash run-omp.sh [N1,N2,...] [K1,K2,...] [T1,T2,...; Default 4] [[[-s | -w] [NREPS; Default 5]] | empty]
```

If -s or -w, performs the Strong or Weak scaling efficiency. Else, only one value for `N` and `K` can be used, the program will displays the elapsed time.

#### Example:

```bash
./run-omp.sh 1000 10 5
./run-omp.sh 1000,2000 10,20 3 -w
./run-omp.sh omp-strong.cfg -s
./run-omp.sh -s
./run-omp.sh -w
```

### Executing CUDA: run-cuda.sh

Evaluates one or more Neural Networks of input sizes `N` and `K` layers.

To run this script, you have 2 options:

```
bash run-omp.sh [Config file]
```

or

```
bash run-omp.sh [N1,N2,...] [K1,K2,...] [NREPS; Default 5]
```

#### Example:

```bash
./run-omp.sh 1000 10 5
./run-omp.sh 1000,2000 10,20 3
./run-omp.sh cuda.cfg
```

## Requirements

* CUDA Toolkit for running CUDA programs

- GCC with OpenMP support for running OpenMP programs
- The timing routines used are defined in `hpc.h` for *POSIX-based* systems. If you are running the programs in a different system, remove the line `#include "../hpc.h"` at the start of the two `main` files.
