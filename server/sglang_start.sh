#!/usr/bin/env bash
# sglang_start.sh — Launch SGLang with TP sized by GPU_TYPE + EAGLE speculative decoding.
# GPU_TYPE=MI355X → TP=1, GPU_TYPE=4xH100 → TP=4

set -euo pipefail

GPU_TYPE="${GPU_TYPE:?Set GPU_TYPE env var (MI355X | 4xH100)}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
EAGLE_MODEL="${EAGLE_MODEL:-}"

case "$GPU_TYPE" in
    MI355X)  TP=1 ;;
    4xH100)  TP=4 ;;
    *)       echo "Unknown GPU_TYPE: $GPU_TYPE"; exit 1 ;;
esac

EAGLE_FLAGS=""
if [ -n "$EAGLE_MODEL" ]; then
    EAGLE_FLAGS="--speculative-algorithm EAGLE --speculative-draft-model-path $EAGLE_MODEL --speculative-num-steps 3 --speculative-eagle-topk 4"
fi

echo "Starting SGLang  model=$MODEL_NAME  tp=$TP  gpu=$GPU_TYPE  eagle=${EAGLE_MODEL:-none}"

exec python -m sglang.launch_server \
    --model-path "$MODEL_NAME" \
    --tp "$TP" \
    --host 0.0.0.0 \
    --port 8000 \
    $EAGLE_FLAGS
