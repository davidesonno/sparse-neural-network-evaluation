#include "network.h"

#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// === misc ===
int next_power_of_2(int x){
    if (x <= 0) return 1;

    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;

    return x;
}

// === array filling ===

int seed = 42;

void set_seed(){
    srand(seed);
}

void fillInput( float* x, int N ){
    set_seed();
    int i;
    for (i=0; i<N; i++){
        if(DEBUG){
            x[i] = 1;
            continue;
        }
        x[i] = RANDOM_NUMBER;
    }
}

void fillWeights( float* weights, int N, int M ){
    fillInput( weights, N * M );
}

void fillWeightsStruct(weights* weights, int N) {
    for (int i=0; i<R; i++){
        fillInput( weights->w[i], N);
    }
}

// === weights memory allocation ===

void malloc_weights(weights *weights, size_t size){
    for (int i=0; i<R; i++){
        weights->w[i] = (float*)malloc(size);
        if(!weights->w[i]){
            printf("Error allocating memory for weights!\n");
            exit(EXIT_FAILURE);
        }
    }
}

// Used to copy a struct to the gPU. We have to first allocate the struct, the arrays, copy the pointers and the values
void cudaMallocMemcpy_weights(weights *h_w, weights **d_w, const size_t size) {
    // Allocate device struct
    cudaMalloc(d_w, sizeof(weights));
    
   // Allocate device memory for each weight in the struct
    float *d_w_arr[R];
    for (int i = 0; i < R; i++) {
        cudaMalloc(&d_w_arr[i], size);
        cudaMemcpy(d_w_arr[i], h_w->w[i], size, cudaMemcpyHostToDevice);
        cudaMemcpy(&((*d_w)->w[i]), &d_w_arr[i], sizeof(float*), cudaMemcpyHostToDevice);
    }

    cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA Error in cudaMallocMemcpy_weights: %s", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}
}

void free_weights(weights *weights){
    for (int i=0; i<R; i++){
        free(weights->w[i]);
    }
}

// similarly to allocation, we need to clean pointers, struct and weights
void cudaFree_weights(weights *d_w){
    // Copy the device struct back to host to get the array pointers
    weights h_w;
    cudaMemcpy(&h_w, d_w, sizeof(weights), cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA Error in cudaFree_weights-memcpy: %s", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}
    
    // Free the device arrays inside the struct
    for (int i=0; i<R; i++){
        cudaFree(h_w.w[i]);
    }

    err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA Error in cudaFree_weights-cudafree of arrays: %s", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}

    // Finally, free the device struct itself
    cudaFree(d_w);

    err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA Error in cudaFree_weights-cudafree struct: %s", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}
}

// === parameter parsing ===

// Parses a comma separated list of integers into an array
int parse_list( const char *arg, int *arr, int max_size ){
    int count = 0;
    char *token, *copy = strdup(arg);
    token = strtok(copy, ",");

    while (token != NULL && count < max_size){
        arr[count++] = atoi(token);
        token = strtok(NULL, ",");
    }
    
    free(copy);
    return count;
}

// parses the command line arguments into the struct ParsedArgs
void parse_arguments( int argc, char *argv[], ParsedArgs *args ){
    if (argc < 3){
        fprintf(stderr, "Usage: %s [N1,N2,...] [K1,K2,...] [NREPS; Default 5]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    args->N_len = parse_list(argv[1], args->N, ARGS_MAX_LEN);
    args->K_len = parse_list(argv[2], args->K, ARGS_MAX_LEN);
    int max = 0;
    for (int i=0; i<args->N_len; i++){
        int element = args->N[i];
        max = element > max ? element : max;
    }
    args->N_max = max;
    if (argc > 3) {
        args->NREPS = atoi(argv[3]);
    } else {
        args->NREPS = DEFAULT_NREPS;
    }
}

void inspect_arguments(ParsedArgs *args){
    printf("N: ");
    for (int i = 0; i < args->N_len; i++) printf("%d ", args->N[i]);
    printf("\nK: ");
    for (int i = 0; i < args->K_len; i++) printf("%d ", args->K[i]);
    printf("\nN_max: %d\n", args->N_max);
    printf("\nNREPS: %d\n", args->NREPS);
}

// === CSV ===

void printCsvHeader(bool kernels[], int reps, bool kernel_info, bool times, bool averages, bool throughputs, bool speedups, KernelMode baseline){
    if (kernel_info) {
        printf("BLKDIM,R,");
    }
    printf("N,K");
    if(times) for (int k=0; k<NUM_KERNELS; k++){
        if (kernels[k])for (int i=0; i<reps; i++){
            printf(",t%d-%s",i+1,KernelModeNames[k]);
        }
    }
    if(averages) for (int k=0; k<NUM_KERNELS; k++){
        if (kernels[k])
            printf(",Average-%s",KernelModeNames[k]);
    }
    if(throughputs) for (int k=0; k<NUM_KERNELS; k++){
        if (kernels[k])
            printf(",Throughput-%s",KernelModeNames[k]);
    }
    if(speedups) for (int k=0; k<NUM_KERNELS; k++){
        if (kernels[k]
            //  && k != baseline
            )
            printf(",Speedup-%s",KernelModeNames[k]);
    }
}

void print_averages(bool kernels[], double times[]){
    for (int k=0; k<NUM_KERNELS; k++){
        if (kernels[k]){
            printf(",%lf",times[k]);
        }
    }
}

void log_tp(uint64_t num, double avg){
    if(avg == -1){ printf(",0"); return;}
    printf(",%.0lf", num / avg);
}

// troughput is computed as total number of outputs per second
void print_throughputs(bool kernels[], double times[],const int in_size, const int layers, const int stencil_size ){
    // num_outputs = sum_{i=1}^{K-1} [N - i(R-1)] = (...) = (K-1)[N - K(R-1)/2]
    uint64_t num_outputs = (uint64_t)(layers - 1) * (uint64_t)(in_size - (layers * (stencil_size - 1)) / 2);
    for (int k=0; k<NUM_KERNELS; k++){
        if (kernels[k]){
            log_tp(num_outputs, times[k]);
        }
    }
}

void print_speedups(bool kernels[], double times[], KernelMode baseline_kernel){
    double baseline_time = times[baseline_kernel];
    for (int k=0; k<NUM_KERNELS; k++){
        if (kernels[k]
            // // comment next line to also display the speedup of the baseline (1.000x)
            //  && k != baseline_kernel
            ){
            printf(",%lf", baseline_time/times[k]);
        }
    }
}
