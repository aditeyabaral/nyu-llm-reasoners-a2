#!/bin/bash

mkdir -p results/memory

source .venv/bin/activate

# 2.7B forward only (fits for inference)
for ctx in 128 256 512; do
    echo "--- 2.7B ctx${ctx} forward ---"
    uv run student/benchmark.py --model-size 2.7B --context-length $ctx \
        --profile-memory --profile-mode forward \
        --memory-snapshot-path results/memory/2.7B_ctx${ctx}_forward.pickle \
        --warmup-steps 2 \
        > results/memory/2.7B_ctx${ctx}_forward.txt 2>&1
done

# large model for training (fits in memory)
for ctx in 128 256 512; do
    echo "--- large ctx${ctx} training ---"
    uv run student/benchmark.py --model-size large --context-length $ctx \
        --profile-memory --profile-mode training \
        --memory-snapshot-path results/memory/large_ctx${ctx}_training.pickle \
        --warmup-steps 2 \
        > results/memory/large_ctx${ctx}_training.txt 2>&1
done

# BF16
echo "--- 2.7B ctx512 BF16 forward ---"
uv run student/benchmark.py --model-size 2.7B --context-length 512 \
    --profile-memory --profile-mode forward \
    --mixed-precision bf16 \
    --memory-snapshot-path results/memory/2.7B_ctx512_bf16_forward.pickle \
    --warmup-steps 2 \
    > results/memory/2.7B_ctx512_bf16_forward.txt 2>&1

echo "--- large ctx512 BF16 training ---"
uv run student/benchmark.py --model-size large --context-length 512 \
    --profile-memory --profile-mode training \
    --mixed-precision bf16 \
    --memory-snapshot-path results/memory/large_ctx512_bf16_training.pickle \
    --warmup-steps 2 \
    > results/memory/large_ctx512_bf16_training.txt 2>&1