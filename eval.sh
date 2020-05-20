#!/usr/bin/env bash

set -e

JULIA="${JULIA:-julia}"

$JULIA --project=. train.jl --evaluate $@
