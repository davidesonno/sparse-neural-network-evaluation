/**
 * - Compile the program with
 * 		nvcc -o [program] main.cu src/network.cu src/utils.cu -Isrc
 * - Run the program with:
 * 		[program] [N1,N2,...,Nn] [K1,K2,...,Kj] [NREPS; Default 5]
 */

#include "../hpc.h" // comment if not available
#include "src/network.h"

#include <stdio.h>
#include <cstdlib>
#include <stdbool.h>
#include <stdint.h>

/*
 * Before each repetition, load the starting x into the cuda memory.
 * Only the needed amount of input values are going to be copied.
 */
void launchNetEvaluations(
	KernelMode Kernel_mode,
	const int N,
	const int K,
	float *d_x,
	float *original_d_x,
	float *h_x,
	float *d_w,
	weights *d_wStruct,
	float *d_y,
	float *h_y,
	const float bias,
	const int NREPS,
	double *averageTime,
	bool log_times
) {
	if (N - (K-1)*(R-1) < 1) {
		if (log_times) for (int rep = 0; rep < NREPS; rep++) {
			printf(",%lf",-1.0f);
		}
		*averageTime = -1;
		return;
	}
	double rep_time;
	for (int rep = 0; rep < NREPS; rep++) {
		/* === Copy x to CUDA === */
		d_x = original_d_x;
		cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

		/* === Start the kernel === */
		TIME_SPLIT tstart = gettime();
		computeOutput(Kernel_mode, N, K, &d_x, d_w, d_wStruct, bias, &d_y);
		cudaDeviceSynchronize();
		TIME_SPLIT tstop = gettime();
		
		/* === Check for errors === */
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess){
			printf("\nKernel %s: %s\n", KernelModeNames[Kernel_mode], cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		/* === Get the elapsed time === */
		rep_time = time_difference(tstart, tstop);
        *averageTime += rep_time;
        if (log_times) printf(",%lf", rep_time);
    }
	/* === Compute the average === */
	*averageTime /= NREPS;

	if (DEBUG){ // this would print the output arrays
		const size_t h_y_size = (N - (K-1)*(R-1)) * sizeof(float);
		cudaMemcpy(h_y, d_y, h_y_size, cudaMemcpyDeviceToHost);
		printf("\nY: ");
		for (int i = 0; i < (N - (K-1)*(R-1)); i++) {
			printf("%f ", h_y[i]);
		}
		printf("\n");
	}
}

int main(int argc, char *argv[]) {
	/* === Program initialization === */
	// Command line arguments
    ParsedArgs args;
    parse_arguments(argc, argv, &args);

	// Check the GPU properties and update some global variable used in the kernels
    update_network_global_variables();
	// print_global_variables();
	
	/* === Network variables === */
	float *h_x, *h_w, *h_y;
	float *d_x, *d_w = nullptr, *d_y;
	float *original_d_x, *original_d_y;
	weights h_wStruct, *d_wStruct = nullptr;
	const float bias = DEBUG ? 0.0f : 0.005f;
	int N,K;
	const int NREPS = args.NREPS;

	/*
	 * Initialize x and y just once, with enough size to work with all the arguments.
	 * All the executions will process the same input.
	 * Weights will be initialized for each N. 
	 * h_x is copied into d_x before each repetition.
	*/
	const int host_x_len = args.N_max;
	if (host_x_len < R) {
		printf("The inputs are too small!");
		exit(EXIT_FAILURE);
	}
	const size_t x_size = host_x_len * sizeof(float);
	const size_t y_size = (host_x_len - R + 1) * sizeof(float);
	size_t w_size;
	
	h_x = (float*)malloc(x_size); cudaMalloc((void**)&d_x, x_size);
	h_y = (float*)malloc(y_size); cudaMalloc((void**)&d_y, y_size);
	original_d_x = d_x; original_d_y = d_y;

	fillInput(h_x, host_x_len);
	
	/* === Kernels to be executed === */
	bool selectedKernels[NUM_KERNELS] = { // the values currently set are the once actively reported in the report
		false,	// STENCIL // does NOT use the weights
		true,	// GLBL_MEM
		true,	// GLBL_MEM_TRAN
		true,	// SHRD_X_TRAN
		true,	// READ_ONLY_W_TRAN
		false,	// SHRD_XW_TRAN
		false,	// SHRD_XW_PAD_TRAN
		true,	// GLB_MEM_STRUCT
		true,	// SHRD_X_STRUCT
		false	// EXPERIMENTAL // empty kernel
	};

	/* === Check if a particular type of weights is needed === */
	bool kernels_use_matrix_w = false;
	bool kernels_use_struct_w = false;

	for (int kernel = 0; kernel < NUM_KERNELS; kernel++) {
		if (KernelModeNeedsStructuredWeights[kernel] && selectedKernels[kernel]) {
			kernels_use_struct_w = true;
		} else {
			kernels_use_matrix_w = true;
		}
	}

	/** With the strutured versions, we can initialize them just once,
	 * since that they are multiple arrays.
	 */
	if (kernels_use_struct_w) {
		w_size = host_x_len * sizeof(float);
		malloc_weights(&h_wStruct, w_size); fillWeightsStruct(&h_wStruct, host_x_len);
		cudaMallocMemcpy_weights(&h_wStruct, &d_wStruct, w_size); // cudaMalloc + cudaMemcpy
		free_weights(&h_wStruct);
	}		

	/* === Decide what to log into the csv === */
	bool log_kernel_info = true;
	bool log_times = false;
	bool log_averages = true;
	bool log_throughputs = true;
	bool log_speedups = true;
	
	/* === Decide wich kernel to use as a baseline for the speedup === */
	KernelMode baseline_kernel = GLBL_MEM_TRAN;
	
	// if the baseline kernel is not among the ones to be executed use the first selected
	if (!selectedKernels[baseline_kernel]){
		for (int kernel=0; kernel<NUM_KERNELS; kernel++){
			if (selectedKernels[kernel]){
				baseline_kernel = (KernelMode)kernel;
                break;
            }
        }
    }

	/* === Print the columns of the csv === */
	printCsvHeader(selectedKernels, NREPS, log_kernel_info, log_times, log_averages, log_throughputs, log_speedups, baseline_kernel);

	/* === Compute the layers with the input arguments === */
	for(int n=0; n<args.N_len; n++){
		N = args.N[n];

		/* === For each new N value, initialize the weights === */
		if (kernels_use_matrix_w) {
			w_size = N * R * sizeof(float); // the matrix has N rows of R elements
			h_w = (float*)malloc(w_size); fillWeights(h_w, N, R); // initialize host weights
			cudaMalloc((void**)&d_w, w_size); cudaMemcpy(d_w, h_w, w_size, cudaMemcpyHostToDevice); // initialize device weigths
			free(h_w); // we can immediately free the host original weights
		}
		
		for(int k=0; k<args.K_len; k++){
			K = args.K[k];

			/* === Timing Variables === */
			double kernel_average_times[NUM_KERNELS] = {0}; 
			
			/* === Start printing the CSV row === */
			printf("\n");
			if (log_kernel_info){
				printf("%d,%d,",BLKDIM, R);
			}
			printf("%d,%d",N , K);

			/* === Compute the network using the kernels selected === */
			for (int kernel=0; kernel<NUM_KERNELS; kernel++){
				if ( selectedKernels[kernel] ){
					launchNetEvaluations( (KernelMode)kernel, N, K, d_x, original_d_x, h_x, d_w, d_wStruct, d_y, h_y, bias, NREPS, &kernel_average_times[kernel], log_times);
				}
			}

			/* === Print average times === */
			if (log_averages) print_averages(selectedKernels, kernel_average_times);
			
			/* === Compute and print throughputs === */
			if (log_throughputs) print_throughputs(selectedKernels, kernel_average_times, N, K, R);

			/* === Compute and print throughputs === */
			if (log_speedups) print_speedups(selectedKernels, kernel_average_times, baseline_kernel);

		}
		
		/* === Clean the weights because they will be instantiated for the next N ===*/	
		if (kernels_use_matrix_w) {
			cudaFree(d_w);
			d_w = nullptr;
		}
		
	}
	
	/* === Clean x, y and eventually structured weights ===*/
	free(h_x); cudaFree(original_d_x); 
	free(h_y); cudaFree(original_d_y); 
	if (kernels_use_struct_w) {
		cudaFree_weights(d_wStruct);
	}	

	return EXIT_SUCCESS;
}