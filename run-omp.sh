#!/bin/bash
# To run this script, you have 2 options:
#  - bash run-omp.sh [N1,N2,...] [K1,K2,...] [T1,T2,...; Default 4] [[[-s | -w] [NREPS; Default 5]] | empty]
#  - bash run-omp.sh [-s | -w] [Config file; Default "omp-strong.cfg" if -s, "omp-weak.cfg" if -w]
# If -s or -w, performs the Strong or Weak scaling efficiency, else it simply displays the elapsed time.

# == Argument parsing ==
# automatically link -s to the strong configuration and -w to the weak configuration
if [[ "$1" == "-s" ]] ||  [[ "$1" == "-w" ]]; then
    if [[ "$1" == "-s" ]]; then
        TYPE="strong"
    else
        TYPE="weak"
    fi
    if [ -z "$2" ]; then
        if [[ "$1" == "-s" ]]; then
            source cfg/omp-strong.cfg
        else
            source cfg/omp-weak.cfg
        fi
    else 
        source "$2"
    fi
    ARGS="$(IFS=,; echo "${N[*]}") $(IFS=,; echo "${K[*]}") $(IFS=,; echo "${T[*]}") $1 $NREPS"
else
    ARGS="$@" # Default usage. Let the program check the arugments
    TYPE="single"
fi
export ARGS

PROGRAM="omp/main"
EXE="omp/omp-program"
DEPENDENCIES="omp/src/*.c"
JOB_NAME="omp-$TYPE"
PARTITION="l40"
if [ "$PARTITION" == "l40" ]; then
    CPUS=8
elif [ "$PARTITION" == "rtx2080" ]; then
    CPUS=4
else 
    CPUS=1
fi
OUT_FOLDER="out-$PARTITION"
OUT="$OUT_FOLDER/omp-$TYPE-unroll.out"

# == Compiling ==
echo "== Compiling $PROGRAM =="
if [ -f "$EXE" ]; then # Remove old compiled file
    rm "$EXE"
fi
gcc -std=c99 -Wall -Wpedantic -fopenmp $PROGRAM.c $DEPENDENCIES  -o $EXE -lm
echo "== End of compiling =="
if [ -f "$OUT" ]; then # Remove old output file
    rm "$OUT"
fi
# == Create output folder if it doesn't exist ==
if [ ! -d "$OUT_FOLDER" ]; then
    mkdir -p "$OUT_FOLDER"
fi

# == Submit job ==
echo "Args: $ARGS"
sbatch -J $JOB_NAME -p $PARTITION -c $CPUS -o $OUT $EXE.sh $ARGS