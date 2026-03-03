#!/bin/bash

for size in small medium large xl 2.7B; do
    for ctx in 128 256 512 1024; do
        echo "--- $size ctx$ctx ---"
        nsys stats results/nsys/${size}_ctx${ctx}.nsys-rep --report cuda_gpu_kern_sum --force-export=true
        nsys stats results/nsys/${size}_ctx${ctx}.nsys-rep --report nvtx_sum --force-export=true
    done
done > results/nsys/all_stats.txt 2>&1