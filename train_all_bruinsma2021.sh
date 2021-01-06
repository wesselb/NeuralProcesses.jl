#!/usr/bin/env bash

mkdir -p logs2

for loss in loglik; do
    for model in convcnp corconvcnp anp-het convnp-het; do
        for data in eq-noisy matern52-noisy noisy-mixture weakly-periodic-noisy sawtooth-noisy mixture-noisy; do
            ./train.sh --model $model --loss $loss --data $data \
                2>&1 | tee logs2/train_${model}_${loss}_${data}.log
        done
    done
done
