#!/bin/bash

# Default values
LEVELS="1 2 3"
NAME_SUFFIX="dsr1_0528"
GPU_DEVICES=1
CPU_WORKERS=8
BUILD_CACHE="True"
TIME_OUT=300

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -l|--levels) LEVELS="${2//,/ }"; shift ;;
        -n|--name) NAME_SUFFIX="$2"; shift ;;
        -g|--gpu) GPU_DEVICES="$2"; shift ;;
        -c|--cpu) CPU_WORKERS="$2"; shift ;;
        -t|--timeout) TIME_OUT="$2"; shift ;;
        --no-cache) BUILD_CACHE="False" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

for LVL in $LEVELS; do
    RUN_NAME="test_hf_level_${LVL}_${NAME_SUFFIX}"
    echo "[$(date)] Starting Level ${LVL} Evaluation ($RUN_NAME)..."
    
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/eval_from_generations.py \
        run_name="${RUN_NAME}" \
        dataset_src=local \
        level="${LVL}" \
        num_gpu_devices="${GPU_DEVICES}" \
        build_cache="${BUILD_CACHE}" \
        gpu_arch='["Ampere"]' \
        verbose=True \
        num_cpu_workers="${CPU_WORKERS}" \
        timeout="${TIME_OUT}"
done
