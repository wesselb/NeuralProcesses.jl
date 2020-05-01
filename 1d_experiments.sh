#!/usr/bin/env bash

set -e

MODELS=(eq matern52 weakly-periodic sawtooth)
EPOCHS_INC=20
EPOCHS_TOTAL=100

EPOCHS_INC_PLUS_ONE=$(echo $EPOCHS_INC + 1 | bc)

JULIA=${1:-julia}
BIN="$JULIA train.jl --epochs $EPOCHS_INC"

echo Binary: $BIN
echo Models: $MODELS

# First train all models.
for model in $MODELS
do
    echo Training model: $model
    $BIN --model $model --starting-epoch 1
    for starting_epoch in $(seq $EPOCHS_INC_PLUS_ONE $EPOCHS_INC $EPOCHS_TOTAL)
    do
        echo Resuming from epoch $starting_epoch
        $BIN --model $model --starting-epoch $starting_epoch
    done
done

# Finally evaluate all models.
for model in $MODELS
do
    echo Evaluating model: $model
    $BIN --model $model --evaluate
done
