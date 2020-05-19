#!/usr/bin/env bash

set -e

JULIA="${JULIA:-julia}"

$JULIA train.jl --evaluate $@
