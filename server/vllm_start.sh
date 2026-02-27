#!/usr/bin/env bash
# vllm_start.sh — Launch vLLM with Expert Parallelism on 4xH100.
#
# Strategy: EP across 4 GPUs, DP=1. Each GPU holds (total_experts / 4)
# whole experts rather than sharding every expert across all GPUs via TP.
# This switches MoE communication from AllReduce to AllToAll.
#
# Requires vLLM >= 0.9.0 (V1 engine) with --enable-expert-parallel support.

set -euo pipefail

GPU_TYPE="${GPU_TYPE:-4xH100}"
MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-V3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
EPLB_ENABLED="${EPLB_ENABLED:-1}"
EPLB_NUM_REDUNDANT="${EPLB_NUM_REDUNDANT:-32}"

# ── Parallelism layout ──────────────────────────────────────────────
# 4xH100: TP=4, DP=1  →  EP_SIZE = TP * DP = 4
# All 4 GPUs participate in expert parallelism; experts are distributed
# across GPUs so each holds (num_routed_experts / 4) full expert copies.
TP=4
DP=1

# ── Expert-parallel load balancer (EPLB) ─────────────────────────────
# Redistributes expert mappings to even out load across EP ranks.
# num_redundant_experts: extra expert replicas to absorb hot-spot traffic.
EPLB_FLAGS=""
if [ "$EPLB_ENABLED" = "1" ]; then
    EPLB_FLAGS="--enable-eplb --eplb-config {\"num_redundant_experts\":${EPLB_NUM_REDUNDANT}}"
fi

# ── AllToAll backend ─────────────────────────────────────────────────
# Options: deepep (nvshmem), pplx (Perplexity), allgather_reducescatter (NCCL fallback)
# NCCL fallback is the safest default; switch to deepep/pplx for production perf.
export VLLM_ALL2ALL_BACKEND="${VLLM_ALL2ALL_BACKEND:-allgather_reducescatter}"

echo "════════════════════════════════════════════════════════════════"
echo " vLLM Expert-Parallel Launch"
echo "  model         = $MODEL_NAME"
echo "  gpu_type      = $GPU_TYPE"
echo "  TP=$TP  DP=$DP  → EP_SIZE=$((TP * DP))"
echo "  max_model_len = $MAX_MODEL_LEN"
echo "  all2all       = $VLLM_ALL2ALL_BACKEND"
echo "  eplb          = $EPLB_ENABLED (redundant=$EPLB_NUM_REDUNDANT)"
echo "════════════════════════════════════════════════════════════════"

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --tensor-parallel-size "$TP" \
    --data-parallel-size "$DP" \
    --enable-expert-parallel \
    $EPLB_FLAGS \
    --max-model-len "$MAX_MODEL_LEN" \
    --host 0.0.0.0 \
    --port 8000
