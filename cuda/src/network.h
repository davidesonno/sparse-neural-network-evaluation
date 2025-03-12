#include <stdio.h>
#include <stdlib.h>

#ifndef NETWORK_H
#define NETWORK_H

/* === Network configuration === */
#define BLKDIM 1024
#define R 29

/* === Kernels definitions === */
typedef enum {
    STENCIL = 0,
    GLBL_MEM = 1,
    GLBL_MEM_TRAN = 2,
    SHRD_X_TRAN = 3,
    READ_ONLY_W_TRAN = 4,
    SHRD_XW_TRAN = 5,
    SHRD_XW_PAD_TRAN = 6,
    GLB_MEM_STRUCT = 7,  
    SHRD_X_STRUCT = 8,
    EXPERIMENTAL = 9,
    
    NUM_KERNELS = 10
} KernelMode;

static const char* KernelModeNames[] = {
    "STENCIL",
    "GLBL_MEM",
    "GLBL_MEM_TRAN",
    "SHRD_X_TRAN",
    "READ_ONLY_W_TRAN",
    "SHRD_XW_TRAN",
    "SHRD_XW_PAD_TRAN",
    "GLB_MEM_STRUCT",
    "SHRD_X_STRUCT",
    "EXPERIMENTAL"
};
// kernels that needs the strucured weights
static const bool KernelModeNeedsStructuredWeights[] = {
    false,
    false,
    false,
    false,
    false,
    false,
    false,
    true,
    true,
    false
};

// structured weights struct. I R was smaller we could declared the R arrays manually
typedef struct {
    float *w[R];
} weights;

#include "utils.h"

void update_network_global_variables();
void print_global_variables();
void computeOutput( KernelMode kernel_mode, const int N, const int K, float **x, float *w, weights *w_struct, const float b, float **y);

// NETWORK_H
#endif
