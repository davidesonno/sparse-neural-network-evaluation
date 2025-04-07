#!/bin/bash
# This script runs tests for OpenMP and CUDA programs with fixed parameters
# and calculates the CUDA speedup

N="50000,100000,300000"
K="1000"
CORES="4" 
NREPS="3"
PARTITION="rtx2080"
OUT_FOLDER="comparison-$PARTITION"

# Create output folder if it doesn't exist
if [ ! -d "$OUT_FOLDER" ]; then
    mkdir -p "$OUT_FOLDER"
fi

# Run OpenMP
EXE="omp/omp-program"
DEPENDENCIES="omp/src/*.c"
if [ -f "$EXE" ]; then # Remove old compiled file
    rm "$EXE"
fi
gcc -std=c99 -Wall -Wpedantic -fopenmp omp/main.c $DEPENDENCIES  -o $EXE -lm
sbatch --job-name="omp-comp" -p "$PARTITION" -c "4" -o $OUT_FOLDER/omp.out $EXE.sh $N $K $CORES -s $NREPS

# Run CUDA
sbatch --job-name="cuda-comp" -p "$PARTITION" -o "$OUT_FOLDER/cuda.out" "cuda/cuda-program.sh" $N $K $NREPS

# Comparison
OMP_COL="Average"
CUDA_COL="Average-SHRD_X_STRUCT"
OMP_FILE="$OUT_FOLDER/omp.out"
CUDA_FILE="$OUT_FOLDER/cuda.out"
# Wait for the jobs to complete
echo "Waiting for jobs to complete..."
while squeue -u "$USER" | grep -q "omp-comp\|cuda-comp"; do
    sleep 3
done

if [ -f "$OMP_FILE" ] && [ -f "$CUDA_FILE" ]; then
    # Print header for the table
    printf "%-15s %-15s %-15s %-15s %-15s\n" "N" "K" "OMP Times" "CUDA Times" "CUDA Speedup"

    # Print the selected columns from omp.out and cuda.out
    paste <(awk -F',' -v col="N" 'NR==1 {for (i=1; i<=NF; i++) if ($i == col) col_num=i} NR>1 {print $col_num}' "$OMP_FILE") \
          <(awk -F',' -v col="K" 'NR==1 {for (i=1; i<=NF; i++) if ($i == col) col_num=i} NR>1 {print $col_num}' "$OMP_FILE") \
          <(awk -F',' -v col="$OMP_COL" 'NR==1 {for (i=1; i<=NF; i++) if ($i == col) col_num=i} NR>1 {print $col_num}' "$OMP_FILE") \
          <(awk -F',' -v col="$CUDA_COL" 'NR==1 {for (i=1; i<=NF; i++) if ($i == col) col_num=i} NR>1 {print $col_num}' "$CUDA_FILE") \
    | while IFS=$'\t' read -r omp_n omp_k omp_times cuda_times; do
        # Calculate the speedup
        speedup=$(awk "BEGIN {print $omp_times / $cuda_times}")
        # Align each value for better display in the table
        printf "%-15s %-15s %-15s %-15s %-15s\n" "$omp_n" "$omp_k" "$omp_times" "$cuda_times" "$speedup"
    done
fi