#include "utils.h"

#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>


// === array filling ===
int seed = 42;

void set_seed(){
    srand(seed);
}

void fillInput( float* x, int N ){
    set_seed();
    int i;
    for (i=0; i<N; i++){
        x[i] = 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
    }
}

void fillWeights( float* x, int N, int M ){
    set_seed();
    int i, j;
    for (i=0; i<N; i++){
        for (j=0; j<M; j++){
            x[i*M + j] = 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
        }
    }
}

// === misc ===
float sigmoid( float x ){
    return 1 / (1 + exp(-x));
}

// === parameter parsing ===
int parse_list(const char *arg, int *arr, int max_size) {
    int count = 0;
    char *token;
    
    // Allocate memory for a copy of arg
    size_t len = strlen(arg) + 1;
    char *copy = (char *)malloc(len);
    if (copy == NULL) {
        return 0; // Allocation failed
    }
    strcpy(copy, arg);

    token = strtok(copy, ",");

    while (token != NULL && count < max_size) {
        arr[count++] = atoi(token);
        token = strtok(NULL, ",");
    }

    free(copy);
    return count;
}

// parses the command line arguments into the struct ParsedArgs
int parse_arguments( int argc, char *argv[], ParsedArgs *args ){
    if (argc < 3){
        fprintf(stderr, "Usage: %s [N1,N2,...] [K1,K2,...] [T1,T2,...; Default 4] [[[-s | -w] [NREPS; Default 5]] | empty]\n", argv[0]);
        return 1;
    }

    args->N_len = parse_list(argv[1], args->N, 100);
    args->K_len = parse_list(argv[2], args->K, 100);
    args->T_len = 1;
    args->strong = false;
    args->weak = false;
    args->NREPS = DEFAULT_NREPS;

    int arg_index = 3;
    if (arg_index < argc && argv[arg_index][0] != '-') {
        args->T_len = parse_list(argv[arg_index], args->T, 100);
        arg_index++;
    } else {
        args->T[0] = DEFAULT_T;
    }

    while (arg_index < argc){
        if (strcmp(argv[arg_index], "-s") == 0){
            if (args->weak){
                fprintf(stderr, "Error: Options -s and -w cannot be used together.\n");
                return 1;
            }
            args->strong = true;
        } else if (strcmp(argv[arg_index], "-w") == 0){
            if (args->strong){
                fprintf(stderr, "Error: Options -s and -w cannot be used together.\n");
                return 1;
            }
            args->weak = true;
        } else {
            if (args->strong || args->weak){
                char *endptr;
                int parsed_reps = strtol(argv[arg_index], &endptr, 10);
                if (*endptr == '\0'){  
                    args->NREPS = parsed_reps;
                } else {
                    fprintf(stderr, "Error: Invalid NREPS value '%s'.\n", argv[arg_index]);
                    return 1;
                }
            } else {
                fprintf(stderr, "Error: Unexpected argument '%s'.\n", argv[arg_index]);
                return 1;
            }
        }
        arg_index++;
    }

    if (!args->strong && !args->weak && arg_index < argc) {
        fprintf(stderr, "Error: Unexpected argument '%s'.\n", argv[arg_index]);
        return 1;
    }

    if (!(args->strong || args->weak) && (args->N_len > 1 || args->K_len > 1 || args->T_len > 1)){
        fprintf(stderr, "Error: If neither -s nor -w is given, N, K and T must be single integers and not lists.\n");
        return 1;
    }

    return 0;
}

int inspect_arguments(ParsedArgs *args){
    printf("N: ");
    for (int i = 0; i < args->N_len; i++) printf("%d ", args->N[i]);
    printf("\nK: ");
    for (int i = 0; i < args->K_len; i++) printf("%d ", args->K[i]);
    printf("\nT: ");
    for (int i = 0; i < args->T_len; i++) printf("%d ", args->T[i]);
    printf("\n");

    if (args->strong || args->weak) {
        printf("Flag: %s\n", args->strong ? "-s" : "-w");
        printf("NREPS: %d\n", args->NREPS);
    }

    return 0;
}
