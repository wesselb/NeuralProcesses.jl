#!/usr/bin/env bash

for loss in loglik elbo loglik-iw; do
    for model in cnp acnp convcnp np-het anp-het convnp-het; do
        for data in matern52 eq noisy-mixture weakly-periodic sawtooth mixture; do
            ./train.sh --model $model --loss $loss --data $data \
                2>&1 | tee logs/${model}_${loss}_${data}.log
        done
    done
done
