#!/usr/bin/env bash

set -e

JULIA="${JULIA:-julia}"

EPOCHS_TOTAL="${EPOCHS_TOTAL:-100}"
EPOCHS_INC="${EPOCHS_INC:-${EPOCHS_TOTAL}}"

echo "Epochs total:     $EPOCHS_TOTAL"
if (( $EPOCHS_TOTAL > $EPOCHS_INC ));
then
    echo "Epochs increment: $EPOCHS_INC"
fi

$JULIA --project=. train.jl --starting-epoch 1 --epochs $EPOCHS_INC $@
if (( $EPOCHS_TOTAL > $EPOCHS_INC ));
then
    for starting_epoch in $(seq $(echo $EPOCHS_INC + 1 | bc) $EPOCHS_INC $EPOCHS_TOTAL)
    do
        echo Resuming from epoch $starting_epoch
        $JULIA --project=. train.jl --starting-epoch $starting_epoch --epochs $EPOCHS_INC $@
    done
fi
