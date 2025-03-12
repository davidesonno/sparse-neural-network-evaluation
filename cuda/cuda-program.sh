#!/bin/bash
# cuda-program.sh
#SBATCH --job-name=FFNN-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 1
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output cuda-out.out
#SBATCH --partition=l40

PROGRAM="main.cu"
PROGRAM_DIR="cuda"
DEPENDENCIES="$PROGRAM_DIR/src/network.cu $PROGRAM_DIR/src/utils.cu"
INCLUDE_DIRS="-I$PROGRAM_DIR/src" # headers
EXE="$PROGRAM_DIR/cuda-program"
nvcc $PROGRAM_DIR/$PROGRAM $DEPENDENCIES $INCLUDE_DIRS -o $EXE && ./$EXE "$@"