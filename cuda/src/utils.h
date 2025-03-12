#ifndef UTILS_H
#define UTILS_H

#include <stdbool.h>
#include <math.h>
#include <stdint.h>


// if hpc.h is not available/not used by the main.cu
#ifndef HPC_H
	typedef clock_t TIME_SPLIT;
	#define gettime clock
    #define time_difference(tstart, tstop) (((double)(tstop - tstart)) / CLOCKS_PER_SEC)
#endif

#define DEBUG false // All the inputs and weights are set to 1. Bias is set to 0. Sigmoid returns x. Outputs are printed after each computation. Used to check the correctness of the kernels
#define DEFAULT_NREPS 5
#define RANDOM_NUMBER 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f
#define ARGS_MAX_LEN 10 // Maximum length of the command line arguments for each parameter. Can be increased if needed. 

/* === Command Line Args === */
typedef struct {
    int N[ARGS_MAX_LEN], K[ARGS_MAX_LEN];
    int N_len, K_len;
    int N_max;
    int NREPS;
} ParsedArgs;


/* === Utility Functions, mostly used by the main.cu program === */

int next_power_of_2(int x);

/* === Array filling === */
void fillInput(float* x, int N);
void fillWeights(float* w, int N, int M);
void fillWeightsStruct(weights* w, int N);
__device__ __inline__ float sigmoid(const float x) {
    if(DEBUG) return x;
    return (float)1.0 / ((float)1.0 + expf(-x));
}

/* === Memory allocations === */
void malloc_weights(weights *w, size_t size);
void cudaMallocMemcpy_weights(weights *h_w, weights **d_w, const size_t size);
void free_weights(weights *w);
void cudaFree_weights(weights *w);

/* === Input parsing === */
int parse_list(const char *arg, int *arr, int max_size);
void parse_arguments(int argc, char *argv[], ParsedArgs *args);
void inspect_arguments(ParsedArgs *args);

/* === Output parsing === */
void printCsvHeader(bool kernels[], int reps, bool kernel_info, bool times, bool averages, bool throughputs, bool speedups, KernelMode baseline);
void print_averages(bool kernels[], double times[]);
void log_tp(uint64_t num, double avg);
void print_throughputs(bool kernels[], double times[],const int in_size, const int layers, const int stencil_size );    
void print_speedups(bool kernels[], double times[], KernelMode baseline_kernel);

#endif 
// UTILS_H
