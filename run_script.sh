#!/bin/bash
# set -x  # Comment out or remove this line to reduce verbosity

# Initialize variables with default values
model=""
data=""
seq_len=512
use_sam_flag=""
add_results_flag=""

# Parse named command line arguments
while getopts "m:d:s:ua" opt; do
  case ${opt} in
    m ) model=$OPTARG ;;
    d ) data=$OPTARG ;;
    s ) seq_len=$OPTARG ;;
    u ) use_sam_flag="--use_sam" ;;  # Activate SAM option
    a ) add_results_flag="--add_results" ;;  # Activate add_results option
    \? ) echo "Usage: cmd [-m model] [-d data] [-s seq_len] [-u] [-a]"
         exit 1 ;;
  esac
done

shift $((OPTIND -1))

# Validation to ensure required parameters are provided
if [ -z "$model" ] || [ -z "$data" ]; then
    echo "Model and data must be specified."
    echo "Usage: cmd [-m model] [-d data] [-s seq_len] [-u (use SAM)] [-a (add results)]"
    exit 1
fi

# Define a list of 'pred_len' (prediction lengths) you want to execute.
pred_lengths=(96 192 336 720) # Extend or modify based on your experiment's needs.
# Use a default rho value. For optimal rho values per model/dataset/horizon, please refer to Appendix Table 3.
rhos=(0.7) # Placeholder. Adjust based on empirical data or specific requirements.

# Loop over each 'pred_len'.
for pred_len in "${pred_lengths[@]}"
do
    num_runs=1 # Define the number of runs for each prediction length.
    for rho in "${rhos[@]}"
    do
        for (( run=1; run<=num_runs; run++ ))
        do
            # Execute the Python script with the specified parameters.
            # Adjust parameters like learning_rate, n_block, dropout, ff_dim, num_heads, and d_model based on the model and dataset.
            # Default parameters are used in https://github.com/google-research/google-research/tree/32d7e53a1bfedb36d659bc44cb03d93f2aef2c9b/tsmixer
            # Use the '--use_sam' flag conditionally based on the 'use_sam' variable.
            if [[ "$data" =~ ^ETT ]]; then
                learning_rate=0.001
                n_block=2
                dropout=0.9
                ff_dim=64
            elif [ "$data" = "weather" ]; then
                learning_rate=0.0001
                n_block=4
                dropout=0.3
                ff_dim=32
            elif [ "$data" = "electricity" ]; then
                learning_rate=0.0001
                n_block=4
                dropout=0.7
                ff_dim=64
            elif [ "$data" = "traffic" ]; then
                learning_rate=0.0001
                n_block=8
                dropout=0.7
                ff_dim=64
            elif [ "$data" = "exchange_rate" ]; then
                learning_rate=0.001
                n_block=8
                dropout=0.7
                ff_dim=64
            elif [ "$data" = "toy" ]; then
                learning_rate=0.001
                n_block=2
                dropout=0.9
                ff_dim=64
            else
                echo "Unknown dataset: $data"
                continue
            fi
            
            command="python run.py --model $model --data $data --seq_len $seq_len --pred_len $pred_len --learning_rate $learning_rate --n_block $n_block --dropout $dropout --ff_dim $ff_dim --num_heads 1 --d_model 16 --rho $rho ${use_sam_flag} ${add_results_flag}"
            
            echo "Executing: $command"
            eval $command
        done
    done
done
