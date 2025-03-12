#include "network.h"


//  =============================
// ||     GLOBAL VARIABLES      ||
//  =============================

/* === IMPORTANT NOTE === */
//   | The strategies using the following variables are not efficient because loading wights into shared memory is not useful.
//   | Still, I keep the code used for testing them.

// Input and weights in shared memory
int BLKDIM_SHRD_W = 0;
int SHARED_MEMORY_SIZE_SHRD_W = 0;

// Input and padded weights in shared memory
int padded_R = next_power_of_2(R);
int BLKDIM_PAD_W = 0;
int SHARED_MEMORY_SIZE_PAD_W = 0;

/** === Compute the BLKNUM, BLKDIM and shared memory size for certain kernels ===
 * 
 * Using the BLKDIM and R, compute the launch parameters for certain kernels
 * that require much shared memory, by eventually decreasing the blocksize 
 */
void update_network_global_variables(){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int max_shared_mem = prop.sharedMemPerBlock; // usually 49152 bytes
    int max_floats = max_shared_mem / sizeof(float);

    // Max block size with the current R
    int max_block_dim;
    int max_k;

    /* === Input and weights in shared memory === */
    // the bigger dividend of 32 (bank dim) that could fit into shared wit w being R x N
    max_k = (max_floats - R + 1) / (32 * (R + 1));
    max_block_dim = max_k * 32;
    BLKDIM_SHRD_W = max_block_dim < BLKDIM ? max_block_dim : BLKDIM;
    SHARED_MEMORY_SIZE_SHRD_W = (BLKDIM_SHRD_W * (R + 1) + R - 1) * sizeof(float);
    
    /* === Input and padded weights in shared memory === */
    // the bigger blkdim that could fit into shared with w being R_pad x N
    max_block_dim = (max_floats - R + 1) / (padded_R + 1);
    BLKDIM_PAD_W = max_block_dim < BLKDIM ? max_block_dim : BLKDIM;
    SHARED_MEMORY_SIZE_PAD_W = (BLKDIM_PAD_W * (R + 1) + R - 1) * sizeof(float); // assign space to the padding
}

void check_global_variables(){
    if (BLKDIM_SHRD_W == 0 || SHARED_MEMORY_SIZE_SHRD_W == 0 || BLKDIM_PAD_W == 0 || SHARED_MEMORY_SIZE_PAD_W == 0){
        printf("Error: global variables not initialized\nBefore launching any kernel launch update_network_global_variables()\n");
        exit(EXIT_FAILURE);
    }
}

void print_global_variables() {
    printf("BLKDIM_SHRD_W: %d\n", BLKDIM_SHRD_W);
    printf("SHARED_MEMORY_SIZE_SHRD_W: %d\n", SHARED_MEMORY_SIZE_SHRD_W);
    printf("padded_R: %d\n", padded_R);
    printf("BLKDIM_PAD_W: %d\n", BLKDIM_PAD_W);
    printf("SHARED_MEMORY_SIZE_PAD_W: %d\n", SHARED_MEMORY_SIZE_PAD_W);
}

//  =============================
// ||         KERNELS           ||
//  =============================
// In this section we present the tested kernels, used to address the assignment.

/** === STENCIL ===
 * We start by simply creating a stencil-sum kernel, used to check the impact
 * caused by accessing the weights.
 */
__global__ void __stencil(
    const int input_size,
    float *x,
    float bias,
    float *y
) { 
    // assign global indexe to the thread in a standard manner
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;

    int output_size = input_size - R + 1;
    
    // compute the result if the thread is inside the output
    if (global_index < output_size) {
        float sum = bias;
        
        // perform the sum
        for (int r = 0; r < R; r++) {
            sum += x[global_index + r];
        }
        
        y[global_index] = sigmoid(sum);
    }
}

/** === GLBL_MEM ===
 * Extend the stencil kernel with a weighted sum.
 * Accessing the weights cause an huge speedloss.
 */
__global__ void __global_memory(
    const int input_size,
    float *x,
    float *w,
    float bias,
    float *y
) {    
    // assign global indexe to the thread in a standard manner
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;

    int output_size = input_size - R + 1;
    
    // compute the result if the thread is inside the output
    if (global_index < output_size) {
        float sum = bias;
        
        // perform the weighted sum
        for (int r = 0; r < R; r++) {
            sum += x[global_index + r] * w[global_index * R + r];
        }
        
        y[global_index] = sigmoid(sum);
    }
}

/** === GLBL_MEM_TRAN ===
 * 
 * Instead of considering weights as a N x R matrix, access it as as R x N matrix.
 * In this way, memory accesses to global memory can be coalesced way better.
 * 
 * NOTE: the speedup can also be achieved by lowering the BLKDIM, but rearranging
 * the matrix on order to have coalesced accesses is a way better solution.
 * Also, upcoming kernels may suffer from this, so we stick to BLKDIM = 1024.
 */
__global__ void __global_memory_transpose(
    const int input_size,
    float *x,
    float *w,
    float bias,
    float *y,
    const int starting_size
) {    
    // assign global index to the thread in a standard manner
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;

    int output_size = input_size - R + 1;
    
    // compute the result if the thread is inside the output
    if (global_index < output_size) {
        float sum = bias;
        
        // perform the weighted sum
        for (int r = 0; r < R; r++) {
            // Access the transposed weights.
            // We need starting size, otherwise only using the input size will result in accessing
            // the wrong weights since that this value decreases, but the array are initialized
            // using the starting input size of the netowrk.
            sum += x[global_index + r] * w[r * starting_size + global_index];
        }
        
        y[global_index] = sigmoid(sum);
    }
}

/** === SHRD_X_TRAN ===
 * 
 * Store x in the shared memory of the blocks. W is considered as R x N matrix
 * to benefit from coalesced accesses.
 * 
 * Each input value is read R-1 times, so storing those in the shared memory
 * should benefit when increasing R. In our case R is big enough to motivate
 * the use of shared memory.
 */
__global__ void __shared_x(
    const int input_size, 
    const float* x, 
    const float* w, 
    const float bias, 
    float* y,
    const int starting_size
) {
    // If each block computes BLKDIM outputs, we need BLKDIM + R - 1
    // input values to compute them.
    __shared__ float shared_x[BLKDIM + R - 1];
    
    // assign indexes to the thread in a standard manner
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    int local_index = threadIdx.x;
    
    const int output_size = input_size - (R-1);
    
    // Load from global memory to shared if possible
    shared_x[local_index] = global_index < input_size ? x[global_index] : 0.0f;
    
    // First R-1 threads loads the remaining input values. Those are the first R-1 elements exceeding the indexes of the block
    if (local_index < R - 1){
        int out_of_block_index = global_index + blockDim.x;
        shared_x[local_index + blockDim.x] = out_of_block_index < input_size ? x[out_of_block_index] : 0.0f;
    }
    
    // synchronize so that all the inputs have been copied in the shared memory
    __syncthreads();
    
    // compute the result if the thread is inside the output
    if (global_index < output_size){ 
        float sum = bias;
        
        // perform the weighted sum
        for (int r = 0; r < R; r++) {
            sum += shared_x[local_index + r] * w[r * starting_size + global_index];
        }

        y[global_index] = sigmoid(sum);
    }
}

/** === READ_ONLY_W_TRAN ===
 * Inform the compiler that w is read only, so it can optimize the accesses.
 */
__global__ void __read_only_w(
    const int input_size, 
    const float* x, 
    const float* __restrict__ w, 
    const float bias, 
    float* y,
    const int starting_size
) {
    // If each block computes BLKDIM outputs, we need BLKDIM + R - 1
    // input values to compute them.
    __shared__ float shared_x[BLKDIM + R - 1];

    const int output_size = input_size - (R-1);

    // assign indexes to the thread in a standard manner
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    int local_index = threadIdx.x;

    // Load from global memory to shared if possible
    shared_x[local_index] = global_index < input_size ? x[global_index] : 0.0f;

    // First R-1 threads loads the remaining input values. Those are the first R-1 elements exceeding the indexes of the block
    if (local_index < R - 1){
        int out_of_block_index = global_index + blockDim.x;
        shared_x[local_index + blockDim.x] = out_of_block_index < input_size ? x[out_of_block_index] : 0.0f;
    }

    // synchronize so that all the inputs have been copied in the shared memory
    __syncthreads();

    // compute the result if the thread is inside the output
    if (global_index < output_size){ 
        float sum = bias;

        // perform the weighted sum
        for (int r = 0; r < R; r++) {
            // Use the read only cache
            sum += shared_x[local_index + r] * __ldg(&w[r * starting_size + global_index]);
        }

        y[global_index] = sigmoid(sum);
    }
}

/** === SHRD_XW_TRAN ===
 * 
 * Store x and w in the shared memory of the blocks.
 * W is considered as R x N matrix.
 * 
 * We try to also store w in global memory, even if it shouldn't be really
 * helpful since that weights are only read once either way. So reading them 
 * from global memory, storing them into shared emory and reading them again,
 * should be a waste.
 * 
 * NOTE: When launching the kernel, we must ensure that we don't exceed the 
 * maximum shared memory of each block and decrease the blockdim accordingly.
 */
__global__ void __shared_x_w(
    const int input_size, 
    const float* x,
    const float* w, 
    const float bias, 
    float* y,
    const int starting_size
) {
    // The size of this array has to be determined at runtime,
    // according to R and the BLKDIM.
    extern __shared__ float shared_memory[];
    
    // The total array is used in the first part for x, with length blockDim.x + R - 1.
    // After that, the weights starts.
    float* shared_x = shared_memory;
    float* shared_w = shared_memory + blockDim.x + R - 1;
    
    const int output_size = input_size - (R-1); 
    
    // assign indexes to the thread in a standard manner
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    int local_index = threadIdx.x;
    
    // Load from global memory to shared if possible
    shared_x[local_index] = global_index < input_size ? x[global_index] : 0.0f;
    for (int r = 0; r < R; r++) {
        shared_w[r * blockDim.x + local_index] = global_index < output_size ? w[r * starting_size + global_index] : 0.0f;
    }
    
    // First R-1 threads loads the remaining input values. Those are the first R-1 elements exceeding the indexes of the block
    if (local_index < R - 1){
        int out_of_block_index = global_index + blockDim.x;
        shared_x[local_index + blockDim.x] = out_of_block_index < input_size ? x[out_of_block_index] : 0.0f;
    }
    
    // synchronize so that all the inputs have been copied in the shared memory
    __syncthreads();
    
    // compute the result if the thread is inside the output
    if (global_index < output_size){ 
        float sum = bias;
        
        // perform the weighted sum
        for (int r = 0; r < R; r++) {
            sum += shared_x[local_index + r] * shared_w[r * blockDim.x + local_index];
        }
        
        y[global_index] = sigmoid(sum);
    }
}

/** === SHRD_XW_PAD_TRAN ===
 * 
 * Store x and w in the shared memory of the blocks.
 * 
 * The w array (RxN) in shared memory is padded to the next power of 2 after R,
 * becoming a (r+pad)xN array. So, the blockdim available will be smaller.
 * In practice it doesn't change anything, but the shared memory used is 
 * bigger. The padding elements are at the end of the array so they are useless.
 * Still, this results in less store bank conflicts, verified with Nsight profiler.
 * Overall, this kernel is not effective due to the same reasoning over 
 * having the weights in shared memory.
 * 
 * Even if it should make more sense to pad them to have a Rx(N+pad), to maybe
 * reduce the bank conflicts by making the rows a multiple of 32 (bank size),
 * in practice it resulted to be slower. NOTE: this consideration was valid
 * during the development process, running on my local rtx1660 and somehow having
 * better performances than the other kernels. On the hpc this kernel performs bad,
 * as expected.
 * 
 * NOTE: the kernel is exactly the same as the previous kernel,
 * the only difference is the number of blocks and their dimension,
 * aswell as the size of the shared memory and how the
 * data is stored in it.
 */
__global__ void __shared_x_w_padded(
    const int input_size,
    const float* x,
    const float* w, 
    const float bias, 
    float* y,
    const int starting_size
) {
    // The size of this array has to be determined at runtime,
    // according to R and the BLKDIM.
    extern __shared__ float shared_memory[];

    // The total array is used in the first part for x, with length blockDim.x + R - 1.
    // After that, the weights starts.
    float* shared_x = shared_memory;
    float* shared_w = shared_memory + blockDim.x + R - 1;
    
    const int output_size = input_size - (R-1); 

    // assign indexes to the thread in a standard manner
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    int local_index = threadIdx.x;

    // Load from global memory to shared if possible
    shared_x[local_index] = global_index < input_size ? x[global_index] : 0.0f;    
    for (int r = 0; r < R; r++) {
        shared_w[r * blockDim.x + local_index] = global_index < output_size ? w[r * starting_size + global_index] : 0.0f;
    }

    // First R-1 threads loads the remaining input values. Those are the first R-1 elements exceeding the indexes of the block
    if (local_index < R - 1){
        int out_of_block_index = global_index + blockDim.x;
        shared_x[local_index + blockDim.x] = out_of_block_index < input_size ? x[out_of_block_index] : 0.0f;
    }
    
    // synchronize so that all the inputs have been copied in the shared memory
    __syncthreads();

    // compute the result if the thread is inside the output
    if (global_index < output_size){ 
        float sum = bias;

        // perform the weighted sum
        for (int r = 0; r < R; r++) {
            sum += shared_x[local_index + r] * shared_w[r * blockDim.x + local_index];
        }
        
        y[global_index] = sigmoid(sum);
    }
}

/** === GLB_MEM_STRUCT ===
 * 
 * W is an array containing pointers to the R set of weights.
 * 
 * In this way, we try to coalesce the weights accesses, modifing the
 * weights to be a structure of arrays.
 */
__global__ void __global_memory_structured_w(
    int input_size,
    float *x,
    weights *w,
    float bias,
    float *y
) {    
    // assign global index to the thread in a standard manner
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;

    int output_size = input_size - R + 1;
    
    // compute the result if the thread is inside the output
    if (global_index < output_size) {
        float sum = bias;

        // perform the weighted sum
        for( int r=0; r<R; r++){
            // access each of the r-th set of weights on the output position
            sum += x[global_index + r] * w->w[r][global_index];
        }
        
        y[global_index] = sigmoid(sum);
    }
}

/** === SHRD_X_STRUCT ===
 * 
 * Store x in the shared memory of the blocks.
 * W is a struct containing an array of pointers to the R sets of weights.
 * 
 * Once again, we try to keep part of the input in the shared memory,
 * because each element is used R times.
 */
__global__ void __shared_x_structured_w(
    const int input_size, 
    const float* x, 
    const weights* w, 
    const float bias, 
    float* y
) {
    // If each block computes BLKDIM outputs, we need BLKDIM + R - 1
    // input values to compute them.
    __shared__ float shared_x[BLKDIM + R - 1];

    const int output_size = input_size - (R-1);

    // assign indexes to the thread in a standard manner
    const int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    const int local_index = threadIdx.x;

    // Load from global memory to shared if possible
    shared_x[local_index] = global_index < input_size ? x[global_index] : 0.0f;

    // First R-1 threads loads the remaining input values. Those are the first R-1 elements exceeding the indexes of the block
    const int out_of_block_index = global_index + blockDim.x;
    if (threadIdx.x < R - 1){
        shared_x[local_index + blockDim.x] = out_of_block_index < input_size ? x[out_of_block_index] : 0.0f;
    }

    // synchronize so that all the inputs have been copied in the shared memory
    __syncthreads();

    // compute the result if the thread is inside the output
    if (global_index < output_size){ 
        float sum = bias;

        // perform the weighted sum
        for( int r=0; r<R; r++){
            // access each of the r-th set of weights on the output position
            sum += shared_x[local_index + r] * w->w[r][global_index];
        }

        y[global_index] = sigmoid(sum);
    }
}

/**
 * NOTE: other ideas included trying to applying a reduction but the
 * complexity introduced by accessing the weights seems to be way bigger
 * of any improvement performed by the kernel.
 * 
 * Even the shared memory versions of the kernels do not gain much speedup
 * compared to the global versions. Using the best possible approach to 
 * access those weights can be exploited by any kernel, hence no kernel
 * seems to gain any relevant speedup.
 * 
 * Another idea:
 *   Store x and w in the shared memory.
 *   Read w from the global memory as an R x N matrix for better coalescing,
 *   but store it in the shared memory as a BLKDIM x R matrix, 
 *   to avoid bank conflicts.
 *
 * Not working as expected,results in the same downsides as the GBL_MEM kernel.
 */

/** === EXPERIMENTAL ===
 * 
 */
__global__ void __exp(
    const int input_size, 
    const float* x,
    const float* w, 
    const float bias, 
    float* y,
    const int starting_size
) {
    // This kernel has been used to test new approaches without having to modify the 
    // whole program structure.
}


//  =============================
// ||         LAUNCHER          ||
//  =============================

/**
 * Function used to launch launch the network evaluation, using one of the proposed kernels.
 * The result is being stored in the output array `y`.
 *
 * It only uses the variables needed by the specific kernel, other variables are ignored.
 */
void computeOutput(
    KernelMode kernel_mode,
    const int N, 
    const int K, 
    float **x, 
    float *w, 
    weights *w_struct, 
    const float b, 
    float **y
) {   
    check_global_variables(); 
    int curr_N = N; // current input size of the layer
    for (int l=1; l<K; l++){
        /* === Compute the number of blocks needed for the layer computation === */
        // Each block processes BLKDIM outputs,
        // the number of outputs are N - (R-1),
        // so we need BLKNUM = ceil((N - (R-1)) / BLKDIM) blocks
        int BLKNUM = (curr_N - (R-1) + BLKDIM - 1) / BLKDIM;
        
        /* === for certain kernels we need different launch parameters === */
        // Input and weights in shared memory
        int BLKNUM_SHRD_W = (curr_N - (R-1) + BLKDIM_SHRD_W - 1) / BLKDIM_SHRD_W;
        
        // Input and padded weights in shared memory
        int BLKNUM_PAD_W = (curr_N - (R-1) + BLKDIM_PAD_W - 1) / BLKDIM_PAD_W;

        // Launch the kernel with the right launch parameters
        switch (kernel_mode){
            case STENCIL:
                __stencil <<< BLKNUM,BLKDIM >>>(curr_N, *x, b, *y);
                break;

            case GLBL_MEM:
                __global_memory <<< BLKNUM, BLKDIM >>>(curr_N, *x, w, b, *y);
                break;

            case GLBL_MEM_TRAN:
                __global_memory_transpose <<< BLKNUM, BLKDIM >>>(curr_N, *x, w, b, *y, N);
                break;

            case SHRD_X_TRAN:
                __shared_x <<< BLKNUM, BLKDIM >>>(curr_N, *x, w, b, *y, N);
                break;
                
            case READ_ONLY_W_TRAN:
                __read_only_w <<< BLKNUM, BLKDIM >>>(curr_N, *x, w, b, *y, N);
                break;
                
            case SHRD_XW_TRAN:
                // NOTE: different BLKDIM and BLKNUM. Also, size of the shared memory is being passed.
                __shared_x_w <<< BLKNUM_SHRD_W, BLKDIM_SHRD_W, SHARED_MEMORY_SIZE_SHRD_W >>>(curr_N, *x, w, b, *y, N);
                break;
                
            case SHRD_XW_PAD_TRAN:
                // NOTE: different BLKDIM and BLKNUM. Also, size of the shared memory is being passed.
                __shared_x_w_padded <<< BLKNUM_PAD_W, BLKDIM_PAD_W, SHARED_MEMORY_SIZE_PAD_W >>>(curr_N, *x, w, b, *y, N);
                break;
                
            case GLB_MEM_STRUCT:
                __global_memory_structured_w <<< BLKNUM, BLKDIM >>>(curr_N, *x, w_struct, b, *y);
                break;
                
            case SHRD_X_STRUCT:
                __shared_x_structured_w <<< BLKNUM,BLKDIM >>>(curr_N, *x, w_struct, b, *y);
                break;
                
            case EXPERIMENTAL:
                __exp <<< BLKNUM_SHRD_W, BLKDIM_SHRD_W, SHARED_MEMORY_SIZE_SHRD_W >>>(curr_N, *x, w, b, *y, N);
                break;

            default:
                printf("Error: unkown kernel\n");
                exit(EXIT_FAILURE);
        }

        /* === Update the input size ===*/
        curr_N -= (R-1);
        
        if (l == K-1){ // don't swap the pointers if this is the last layer
            break;
        }

        /* === Use the last output as the new input ===*/
        float *tmp = *x;
        *x = *y;
        *y = tmp;
    }
}