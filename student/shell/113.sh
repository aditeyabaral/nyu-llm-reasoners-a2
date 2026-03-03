#!/bin/bash

mkdir -p results

source .venv/bin/activate

echo "=========================================="
echo "1.1.3(b): All model sizes, 5 warmup steps"
echo "=========================================="

for size in small medium large xl 2.7B; do
    echo ""
    echo "--- Model: $size ---"
    uv run student/benchmark.py --model-size $size --context-length 512 --profile-mode all --warmup-steps 5 --measurement-steps 10 \
        2>&1 | tee results/113b_${size}_warmup5.txt
done

echo ""
echo "=========================================="
echo "1.1.3(c): All model sizes, varying warmup"
echo "=========================================="

for size in small medium large xl 2.7B; do
    echo ""
    echo "  Model: $size, warmup: 0"
    uv run student/benchmark.py --model-size $size --context-length 512 --profile-mode all --no-warmup --measurement-steps 10 \
        2>&1 | tee results/113c_${size}_warmup0.txt

    for w in 1 2; do
        echo ""
        echo "  Model: $size, warmup: $w"
        uv run student/benchmark.py --model-size $size --context-length 512 --profile-mode all --warmup-steps $w --measurement-steps 10 \
            2>&1 | tee results/113c_${size}_warmup${w}.txt
    done
done