#include "network.h"

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/**
 * Stores the output values in the `y` array.
 * The workload is split accross the threads such that each one computes adjacent
 * output values. 
 * 
 * The unrolling factor can be set to 4,8 or else in the header file.
 * Suggested value is UNROLL = 4.
 */
void computeLayer(int N, float *x, float *w, float b, float *y) {
    #pragma omp parallel for schedule(static) default(none) shared(N, x, w, y, b)
    for (int i = 0; i < N - (R - 1); i++) {
        float sum = b;
        int r;
        
        #if UNROLL == 4
            for (r = 0; r <= R - 4; r += 4) {
                sum += x[i + r + 0] * w[i * R + r + 0] +
                       x[i + r + 1] * w[i * R + r + 1] +
                       x[i + r + 2] * w[i * R + r + 2] +
                       x[i + r + 3] * w[i * R + r + 3];
            }
        #elif UNROLL == 8
            for (r = 0; r <= R - 8; r += 8) {
                sum += x[i + r + 0] * w[i * R + r + 0] +
                       x[i + r + 1] * w[i * R + r + 1] +
                       x[i + r + 2] * w[i * R + r + 2] +
                       x[i + r + 3] * w[i * R + r + 3] +
                       x[i + r + 4] * w[i * R + r + 4] +
                       x[i + r + 5] * w[i * R + r + 5] +
                       x[i + r + 6] * w[i * R + r + 6] +
                       x[i + r + 7] * w[i * R + r + 7];
            }
        #else
            for (r=0; r<R; r++){
                sum += x[i+r] * w[i*R + r];
            }
        #endif
            
        for (; r < R; r++) {
            sum += x[i + r] * w[i * R + r];
        }

        y[i] = sigmoid(sum);
    }
}
    

// returns a pointer to the output
float* computeOutput(int N, int K, float *x, float *w, float b){
    float *out = (float*)malloc((N - (R-1)) * sizeof(float));
    
    int curr_N = N;
    for (int i=1; i<K; i++){
        computeLayer(curr_N, x, w, b, out);
        curr_N -= (R-1);
        // to avoid copying the array
        // out will be the input and x the output
        float *tmp = x;
        x = out;
        out = tmp;
    }

    if (K%2 == 0){ // this now holds the original 'out'
        free(x);
    }
    return out;
}