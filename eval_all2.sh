#!/usr/bin/env bash

mkdir -p logs2

for loss in loglik; do
    for model in convcnp corconvcnp np-het anp-het convnp-het; do
        for data in eq matern52 noisy-mixture weakly-periodic sawtooth mixture; do
            ./eval.sh --model $model --loss $loss --data $data \
                2>&1 | tee logs2/eval_${model}_${loss}_${data}.log
        done
    done
done
