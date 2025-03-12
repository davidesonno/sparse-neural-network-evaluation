#!/bin/bash
# omp-program.sh
#SBATCH --job-name=FFNN-omp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 8
#SBATCH --time=1-00:00:00
#SBATCH --output omp-out.out
#SBATCH --partition=l40

PROGRAM="omp/omp-program"
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    ./$PROGRAM "$@" # run directly 
else
    ./$PROGRAM $ARGS # run from sbatch
fi