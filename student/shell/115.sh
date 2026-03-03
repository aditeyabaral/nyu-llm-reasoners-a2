#!/bin/bash

mkdir -p results/bf16

for size in small medium large xl 2.7B; do
    echo "--- $size ---"
    uv run student/benchmark.py \
        --model-size $size \
        --context-length 512 \
        --mixed-precision bf16 \
        --profile-mode all \
        --warmup-steps 5 \
        --measurement-steps 10 \
        > results/bf16/${size}_bf16.txt 2>&1
done