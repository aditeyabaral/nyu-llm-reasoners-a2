#!/bin/bash

mkdir -p results/nsys

source .venv/bin/activate

for size in small medium large xl 2.7B; do
    for ctx in 128 256 512 1024; do
        echo "--- Profiling: $size, context=$ctx ---"
        nsys profile \
            --trace nvtx,cuda \
            --stats=true \
            --output=results/nsys/${size}_ctx${ctx} \
            --force-overwrite=true \
            uv run student/benchmark.py \
            --model-size $size \
            --context-length $ctx \
            --profile-mode all \
            --warmup-steps 5 \
            --measurement-steps 3 \
            --profile-attention
    done
done