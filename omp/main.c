/**
 * - Compile the program with:
 *      gcc -o [program] -std=c99 -Wall -Wpedantic -fopenmp main.c src/*.c -lm
 * - Run the program with:
 *      [program] [N1,N2,...,Nn] [K1,K2,...,Kj] [T1,T2,...,Tk; Default 4] [[[-s | -w] [NREPS; Default 5]] | empty; Default empty]
 * if not -s nor -w: N and K must be a single integer.
 */

#include "../hpc.h" // comment if not available
#include "src/network.h"

#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

/**
 * Runs a single evaluation based on the provided network.
 * 
 * The output is stored in x or y, depending on K being odd or even.
 */
double run_single_evaluation(int N, int K, int T){

    if (N - (K-1)*(R-1) < 1){ // output size
        return -1.0;
    }
    /* === Initialize the network variables === */
    TIME_SPLIT tstart, tstop;
    float *x, *w, *y;
    float b = 0.005f;

    /* === Allocate memory === */
    x = (float*)malloc(N * sizeof(float));
    w = (float*)malloc(N * R * sizeof(float));
    
    /* === Populate the arrays === */
    fillInput(x,N);
    fillWeights(w,N,R);

    /* === Set the number of threads === */
    omp_set_num_threads(T);

    /* === Start the network evaluation === */
    tstart = gettime();
    y = computeOutput(N, K, x, w, b);
    tstop = gettime();

    /* === Free memory === */
    free(x);
    free(w);
    if (K%2 != 0){  // since that we are using x to store the output, if the
        free(y);    // number of layers is odd y and x are pointing to the same 
    }               // memory location
    
    return time_difference(tstart,tstop);
}

/**
 * Runs the strong scaling evaluation based on the provided arguments.
 * 
 * Each input-layers pair is evaluated for each thread in T.
 */
void run_strong_scaling_evaluation(ParsedArgs *args){

    printf("N,K,p,");
    for (int i=0; i<args->NREPS; i++){
        printf("t%d,",i+1);
    }
    printf("Average,S(p),E(p)\n");
    for (int n=0; n<args->N_len; n++){
        for (int k=0; k<args->K_len; k++){
            // for each thread, compute the speedup and strong scaling factor
            // also initialize the variables to store T1 (the time with 1 thread)
            double T1 = 0;
            bool get_T1 = true;

            for (int t=0; t<args->T_len; t++){
                printf("%d,%d,%d,", args->N[n], args->K[k], args->T[t]);

                // for each repetition, store the time and then average
                double average = 0;
                for (int rep=0; rep<args->NREPS; rep++){
                    double rep_time = run_single_evaluation(args->N[n], args->K[k], args->T[t]);
                    average += rep_time;
                    printf("%f,",rep_time);
                }
                average/=args->NREPS;
                if (get_T1){
                    T1 = average;
                    get_T1 = false;
                }
                double speedup = T1/average; // speedup of the computatin with p threads w.r.t. 1 thread
                double sse = speedup/args->T[t]; // strong scaling efficiency E(p) = S(p)/p
                printf("%lf,%lf,%lf\n",average,speedup,sse);
            }
        }
    }
}

/**
 * Runs the weak scaling evaluation based on the provided arguments.
 * 
 * Each input size multiplied by each thread in T and evaluated, for all the layers size.
 */
void run_weak_scaling_evaluation(ParsedArgs *args){
    
    printf("N,K,p,");
    for (int i=0; i<args->NREPS; i++){
        printf("t%d,",i+1);
    }
    printf("Average,W(p)\n");
    for (int n=0; n<args->N_len; n++){
        for (int k=0; k<args->K_len; k++){
            // for each thread, compute the weak scaling factor
            // also initialize the variables to store T1 (the time with 1 thread)
            double T1 = 0;
            bool get_T1 = true;

            for (int t=0; t<args->T_len; t++){
                printf("%d,%d,%d,", args->N[n] * args->T[t], args->K[k], args->T[t]);

                // for each repetition, store the time and then average
                double average = 0;
                for (int rep=0; rep<args->NREPS; rep++){
                    double rep_time = run_single_evaluation(args->N[n] * args->T[t], args->K[k], args->T[t]);
                    average += rep_time;
                    printf("%f,",rep_time);
                }
                average/=args->NREPS;
                if (get_T1){
                    T1 = average;
                    get_T1 = false;
                }
                double wse = T1/average; // weak scaling efficiency W(p) = T1/T(p)
                printf("%lf,%lf\n",average,wse);
            }
        }
    }
}


int main( int argc, char *argv[] ) {
	/* === Program initialization === */
	// Command line arguments
    ParsedArgs args;
    if (parse_arguments(argc, argv, &args) != 0) {
        return 1;  // Arguments error
    }

    if (args.strong) { 
        run_strong_scaling_evaluation(&args);
    } else if (args.weak) {
        run_weak_scaling_evaluation(&args);
    } else {
        printf("N,K,p,Time\n");
        double time = run_single_evaluation(args.N[0], args.K[0], args.T[0]);
        printf("%d,%d,%d,%lf\n",args.N[0], args.K[0], args.T[0],time);
    }

    return 0;
}