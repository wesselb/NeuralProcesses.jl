#!/usr/bin/env bash

set -e

JULIA=${1:-julia}
SETTINGS="--epochs 20"

BIN="$JULIA train.jl $SETTINGS"
echo Binary: $BIN

# First train all models.
for model in eq matern52 weakly-periodic sawtooth
do
    echo Model: $model
    
    echo Training model
    $BIN --model $model --starting-epoch 1

    for starting_epoch in $(seq 21 20 100)
    do
        echo Resuming from epoch $starting_epoch
        $BIN --model $model --starting-epoch $starting_epoch
    done
done

# Finally evaluate all models.
for model in eq matern52 weakly-periodic sawtooth
do
    echo Model: $model

    echo Evaluating model
    $BIN --model $model --evaluate
done
