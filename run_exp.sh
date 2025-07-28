#!/bin/bash

# Create results and bestweights folder
[ -d results ] || mkdir results
[ -d bestweights ] || mkdir bestweights

# Define the models and reduction methods
MODELS=("lite" "inception" "resnet" "fcn")
REDUCE_METHODS=("gap mpvH" "gap ppvHste" "gap mpvH ppvHste" "ppvHste mpvH mipvH" "ppvH" "ppvS" "ppvHste" "mpvH" "mpvS" "mipvH" "mipvS" "gap" "gmp" "gap gmp" "map2" "map4" "map8" "gap map2 map4 map8" "gmp mmp2 mmp4 mmp8" "gru128")
DATASETS=("tsc_ucr") # "tsc_ucr" "tsc_ucr_s" "tsc_ucr_m" "mtsc_uea"

# Number of times to run each case
NUM_RUNS=11

# Loop to run the command NUM_RUNS times
for ((i=1; i<=NUM_RUNS; i++))
do
    # Loop over each dataset
    for DATASET in "${DATASETS[@]}"
    do
        # Loop over each model
        for MODEL in "${MODELS[@]}"
        do
            # Loop over each reduction method
            for REDUCE in "${REDUCE_METHODS[@]}"
            do
                REDUCE_UNDERSCORE=${REDUCE// /_}
                FILE_PATTERN="results/EXP_${MODEL}_${REDUCE_UNDERSCORE}_[0-9]*.csv"
                FILE_COUNT=$(ls $FILE_PATTERN 2>/dev/null | wc -l)

                if [ $FILE_COUNT -ge $i ]; then
                    echo "Skipping already completed run: $FILE_PATTERN ($i / $FILE_COUNT)"
                    continue
                fi

                echo "Running case: model=$MODEL, reduce=$REDUCE, run=$i/$NUM_RUNS"
                echo "python run.py --model $MODEL --reduce $REDUCE --dataset $DATASET --results results"
                python run.py --model $MODEL --reduce $REDUCE --dataset $DATASET --results results
                echo "Finished run $i for case: model=$MODEL, reduce=$REDUCE"
            done
        done
    done
done
