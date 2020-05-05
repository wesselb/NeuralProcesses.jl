#!/usr/bin/env bash

set -e

MODELS="convnp"
LOSSES="loglik"
DATA_SETS="eq matern52 weakly-periodic sawtooth"

EPOCHS_INC=5
EPOCHS_TOTAL=40

JULIA=${1:-julia}

EPOCHS_INC_PLUS_ONE=$(echo $EPOCHS_INC + 1 | bc)
BIN="$JULIA train.jl --epochs $EPOCHS_INC"

line () {
    echo "------------------------------------------------------"
}
section () {
    line
    echo ">>> $1"
}

section SETTINGS
echo "Binary:    $BIN      "
echo "Models:    $MODELS   "
echo "Losses:    $LOSSES"
echo "Data sets: $DATA_SETS"

section TRAINING
for model in $MODELS
do
    section "Model: $model"
    for loss in $LOSSES
    do
        section "Loss: $loss"
        for data in $DATA_SETS
        do
            section "Data set: $data"
            $BIN --data $data --model $model --loss $loss --starting-epoch 1
            for starting_epoch in $(seq $EPOCHS_INC_PLUS_ONE $EPOCHS_INC $EPOCHS_TOTAL)
            do
                echo Resuming from epoch $starting_epoch
                $BIN --data $data --model $model --loss $loss --starting-epoch $starting_epoch
            done
        done
    done
done

section EVALUATING
for model in $MODELS
do
    section "Model: $model"
    for loss in $LOSSES
    do
        section "Loss: $loss"
        for data in $DATA_SETS
        do
            section "Data set: $data"
            $BIN --data $data --model $model --loss $loss --evaluate
        done
    done
done
