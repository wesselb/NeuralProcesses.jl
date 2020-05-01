#!/usr/bin/env bash

set -e

DATA_SETS="eq matern52 weakly-periodic sawtooth"
EPOCHS_INC=20
EPOCHS_TOTAL=100

EPOCHS_INC_PLUS_ONE=$(echo $EPOCHS_INC + 1 | bc)

JULIA=${1:-julia}
BIN="$JULIA train.jl --epochs $EPOCHS_INC --model convcnp"

echo Binary: $BIN
echo Data sets: $DATA_SETS

# First train all models.
for data in $DATA_SETS
do
    echo Data set: $data
    $BIN --data $data --starting-epoch 1
    for starting_epoch in $(seq $EPOCHS_INC_PLUS_ONE $EPOCHS_INC $EPOCHS_TOTAL)
    do
        echo Resuming from epoch $starting_epoch
        $BIN --data $data --starting-epoch $starting_epoch
    done
done

# Finally evaluate all models.
for data in $DATA_SETS
do
    echo Data set: $data
    $BIN --data $data --evaluate
done
