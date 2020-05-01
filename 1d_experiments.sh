#!/usr/bin/env bash

set -e

MODELS="convcnp"
DATA_SETS="eq matern52 weakly-periodic sawtooth"
EPOCHS_INC=20
EPOCHS_TOTAL=100

JULIA=${1:-julia}

EPOCHS_INC_PLUS_ONE=$(echo $EPOCHS_INC + 1 | bc)
BIN="$JULIA train.jl --epochs $EPOCHS_INC"

echo "---------------------------"
echo "Binary:    $BIN            "
echo "Models:    $MODELS         "
echo "Data sets: $DATA_SETS      "
echo "---------------------------"

for model in $MODELS
do
    echo Model: $model
    for data in $DATA_SETS
    do
        echo Data set: $data
        $BIN --data $data --model $model --starting-epoch 1
        for starting_epoch in $(seq $EPOCHS_INC_PLUS_ONE $EPOCHS_INC $EPOCHS_TOTAL)
        do
            echo Resuming from epoch $starting_epoch
            $BIN --data $data --model $model --starting-epoch $starting_epoch
        done
    done
done

for model in $MODELS
do
    echo Model: $model
    for data in $DATA_SETS
    do
        echo Data set: $data
        $BIN --data $data --model $model --evaluate
    done
done
