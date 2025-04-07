#!/bin/bash
# To run this script, you have 2 options:
#  - bash run-omp.sh [N1,N2,...] [K1,K2,...] [NREPS; Default 5]
#  - bash run-omp.sh [Config file]

OUT_NAME="cuda-single"
# == Argument parsing ==
if [ $# -eq 0 ]; then
    echo "   To run this script, you have 2 options:"
    echo "   - bash run-omp.sh [N1,N2,...] [K1,K2,...] [NREPS; Default 5]"
    echo "   - bash run-omp.sh [Config file]"
    exit 1
elif [ $# -eq 1 ]; then
    CONFIG_FILE="$1"
    OUT_NAME="$(basename $CONFIG_FILE .cfg)"
    source "$CONFIG_FILE"
elif [ $# -ge 2 ]; then
    N=("$1")
    K=("$2")
    NREPS="${3:-5}"
else
    echo "Invalid arguments."
    exit 1
fi

# == Convert cfg arrays to comma-separated lists ==
ARGS="$(IFS=,; echo "${N[*]}") $(IFS=,; echo "${K[*]}") $NREPS"
export ARGS

PROGRAM="cuda/cuda-program"
PARTITION="l40"
OUT_FOLDER="out-$PARTITION"
OUT="$OUT_FOLDER/$OUT_NAME.out"

# == Remove old output file ==
if [ -f "$OUT" ]; then 
    rm "$OUT"
fi

# == Create output folder if it doesn't exist ==
if [ ! -d "$OUT_FOLDER" ]; then
    mkdir -p "$OUT_FOLDER"
fi

# == Submit job ==
echo "Args: $ARGS"
sbatch --job-name=$OUT_NAME -p "$PARTITION" -o "$OUT" "$PROGRAM.sh" $ARGS
