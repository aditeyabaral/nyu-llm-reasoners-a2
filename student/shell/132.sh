#!/bin/bash

export CPATH=/home/ab12057/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/include/python3.12:$CPATH

source .venv/bin/activate

uv run student/bench_flash_attention.py \
    > results/flash_attention.txt 2>&1