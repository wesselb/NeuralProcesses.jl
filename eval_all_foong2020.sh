#!/usr/bin/env bash

mkdir -p logs

for loss in loglik elbo loglik-iw; do
    for model in cnp acnp convcnp np-het anp-het convnp-het; do
        for data in eq matern52 noisy-mixture weakly-periodic sawtooth mixture; do
            ./eval.sh --model $model --loss $loss --data $data \
                2>&1 | tee logs/eval_${model}_${loss}_${data}.log
        done
    done
done
