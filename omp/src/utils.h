#ifndef UTILS_H
#define UTILS_H

#include <stdbool.h>

// if hpc.h is not available/not used by the main.cu
#ifndef HPC_H
	typedef clock_t TIME_SPLIT;
	#define gettime clock
    #define time_difference(tstart, tstop) (((double)(tstop - tstart)) / CLOCKS_PER_SEC)
#endif

#define DEFAULT_T 4 // default number of threads
#define DEFAULT_NREPS 5 // default number of repetitions

typedef struct {
    int N[100], K[100], T[100];
    int N_len, K_len, T_len;
    bool strong, weak;
    int NREPS;
} ParsedArgs;

/* === Utility Functions === */
void fillInput(float* x, int N);
void fillWeights(float* x, int N, int M);
float sigmoid(float x);
int parse_list(const char *arg, int *arr, int max_size);
int parse_arguments(int argc, char *argv[], ParsedArgs *args);
int inspect_arguments(ParsedArgs *args);

#endif
