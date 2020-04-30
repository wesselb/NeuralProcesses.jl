#!/usr/bin/env bash

set -e

JULIA=${1:-julia}
SETTINGS="--epochs 20"

BIN="$JULIA train.jl $SETTINGS"
echo Binary: $BIN

for model in eq matern52 weakly-periodic sawtooth
do
    echo Model: $model
    echo Initial training iteration
    $BIN --model $model
    for training_iteration in $(seq 2 5)
    do
        echo Training iteration: $training_iteration
        $BIN --model $model --continue
    done
    echo Evaluate model
    $BIN --model $model --evaluate
done

