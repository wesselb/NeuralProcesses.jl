#!/usr/bin/env bash

set -e

MODEL_LOSSES="
    convcnp,loglik
    convnp,loglik
    convnp,elbo
    anp,loglik
    anp,elbo
    np,loglik
    np,elbo"
DATA_SETS="
   eq
   matern52
   noisy-mixture
   weakly-periodic
   sawtooth"

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
echo "Binary: $BIN"
echo "Models and losses:"
for model_loss in $MODEL_LOSSES
do
    IFS="," read model loss <<< "$model_loss"
    echo "    $model + $loss"
done
echo "Data sets: $DATA_SETS"

section TRAINING
for model_loss in $MODEL_LOSSES
do
    IFS="," read model loss <<< "$model_loss"
    echo "Model: $model"
    echo "Loss:  $loss"
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

section EVALUATING
for model_loss in $MODEL_LOSSES
do
    IFS="," read model loss <<< "$model_loss"
    echo "Model: $model"
    echo "Loss:  $loss"
    for data in $DATA_SETS
    do
        section "Data set: $data"
        $BIN --data $data --model $model --loss $loss --evaluate
    done
done
