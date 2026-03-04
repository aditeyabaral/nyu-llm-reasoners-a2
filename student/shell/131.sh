export CPATH=/home/ab12057/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/include/python3.12:$CPATH

mkdir -p results/compile

source .venv/bin/activate

for size in small medium large xl 2.7B; do
    echo "--- $size ---"
    uv run student/benchmark.py \
        --model-size $size \
        --context-length 512 \
        --compile \
        --profile-mode all \
        --warmup-steps 5 \
        --measurement-steps 10 \
        > results/compile/compiled_${size}.txt 2>&1
done

uv run student/bench_attention.py --compile \
    > results/compile/compiled_attention.txt 2>&1