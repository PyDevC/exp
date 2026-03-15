#!/usr/bin/env bash

CONFIG=(
    TORCH_COMPILE_DEBUG=1
)

export $CONFIG
python3 $1
tlparse torch_compile_debug/tlparse/*.log --overwrite
rm torch_compile_debug/tlparse/*.log
