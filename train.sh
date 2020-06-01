#!/usr/bin/env bash

set -e

JULIA="${JULIA:-julia}"
EPOCHS_INC="${EPOCHS_INC:-10}"
EPOCHS_TOTAL="${EPOCHS_TOTAL:-100}"

echo "Epochs total:     $EPOCHS_TOTAL"
echo "Epochs increment: $EPOCHS_INC"

$JULIA --project=. train.jl --starting-epoch 1 --epochs $EPOCHS_INC $@
for starting_epoch in $(seq $(echo $EPOCHS_INC + 1 | bc) $EPOCHS_INC $EPOCHS_TOTAL)
do
    echo Resuming from epoch $starting_epoch
    $JULIA --project=. train.jl --starting-epoch $starting_epoch --epochs $EPOCHS_INC $@
done
