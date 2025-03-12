#ifndef NETWORK_H
#define NETWORK_H

#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* === Network configuration === */
#define UNROLL 4
#define R 29


float* computeOutput(int N, int K, float *x, float *w, float b);

#endif // NETWORK_H
