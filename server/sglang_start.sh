#!/usr/bin/env bash
# sglang_start.sh — Disaggregated serving: GLM-4.5-Air-FP8 via SGLang + Mooncake
#
# Based on official docs:
# https://kvcache-ai.github.io/Mooncake/getting_started/examples/sglang-integration-v1.html
#
# GPU layout for 4× H100 80GB with GLM-4.5-Air-FP8 (106B, FP8 = ~106 GB weights):
#
#   TP=2 is required per worker: 106 GB / 2 GPUs = ~53 GB each (fits in 80 GB)
#   TP=1 would require 106 GB on one GPU → OOM on H100 80 GB
#
#   Topology: 1P1D  (1 Prefill + 1 Decode — each spans 2 GPUs via TP=2)
#   ┌────────────────────┐   KV-cache   ┌────────────────────┐
#   │  Prefill Worker    │ ──────────▶  │  Decode Worker     │
#   │  TP=2  (GPU 0+1)   │  Mooncake    │  TP=2  (GPU 2+3)   │
#   │  port: 30000       │              │  port: 30001        │
#   └────────────────────┘              └────────────────────┘
#             ▲                                   ▲
#             └───────── Router (:8000) ──────────┘
#
# Note: Multiple prefill workers on the same node are NOT supported by SGLang
# disaggregation (bootstrap port conflict — official limitation).
#
# Usage:
#   bash server/sglang_start.sh           # serve (default)
#   bash server/sglang_start.sh install   # only sync deps

set -euo pipefail

# ── Load .env if present ─────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../.env" ]]; then
    set -a; source "${SCRIPT_DIR}/../.env"; set +a
elif [[ -f ".env" ]]; then
    set -a; source ".env"; set +a
fi

# ── Config (override via env) ─────────────────────────────────────────
MODEL="${MODEL_PATH:-zai-org/GLM-4.5-Air-FP8}"
HOST="${HOST_IP:-0.0.0.0}"
MAX_LEN="${MAX_MODEL_LEN:-32768}"

# TP=2 required for both workers: 106B FP8 model needs 2× H100 80GB per worker
TP="${TP_SIZE:-2}"

P1_PORT="${PREFILL_PORT_1:-30000}"
D1_PORT="${DECODE_PORT_1:-30001}"
R_PORT="${ROUTER_PORT:-8000}"

# Mooncake: just set the protocol — no separate master process needed
# (mooncake.json config file requirement removed in SGLang PR 5460)
export MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-tcp}"

UV="uv run --extra sglang"

# ── Helpers ───────────────────────────────────────────────────────────
cleanup() { kill $(jobs -p) 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT INT TERM

install() {
    echo "▸ Syncing uv deps (sglang + mooncake)…"
    uv sync --extra sglang
    echo "✅ Done."
}

serve() {
    echo "════════════════════════════════════════════════════════════"
    echo " 1P1D Disaggregated · ${MODEL} · TP=${TP} · ${MOONCAKE_PROTOCOL}"
    echo " Memory: ~$(( 106 / TP )) GB weights per GPU (fits in H100 80GB)"
    echo " Prefill: GPUs 0+1 (base-gpu-id 0) → :${P1_PORT}"
    echo " Decode:  GPUs 2+3 (base-gpu-id 2) → :${D1_PORT}"
    echo " Router:  :${R_PORT}"
    echo "════════════════════════════════════════════════════════════"

    # ── Step 1: Prefill worker — GPUs 0+1 (TP=2) ─────────────────────
    echo "▸ Starting prefill worker (GPUs 0+1, TP=${TP})..."
    $UV -m sglang.launch_server \
        --model-path "$MODEL" \
        --tp-size "$TP" \
        --disaggregation-mode prefill \
        --base-gpu-id 0 \
        --host "$HOST" \
        --port "$P1_PORT" \
        --max-total-tokens "$MAX_LEN" \
        --trust-remote-code \
        --enable-metrics &
    sleep 5

    # ── Step 2: Decode worker — GPUs 2+3 (TP=2) ──────────────────────
    echo "▸ Starting decode worker (GPUs 2+3, TP=${TP})..."
    $UV -m sglang.launch_server \
        --model-path "$MODEL" \
        --tp-size "$TP" \
        --disaggregation-mode decode \
        --base-gpu-id 2 \
        --host "$HOST" \
        --port "$D1_PORT" \
        --max-total-tokens "$MAX_LEN" \
        --trust-remote-code \
        --enable-metrics &
    sleep 5

    # ── Step 3: Router ────────────────────────────────────────────────
    echo "▸ Starting router on port ${R_PORT}..."
    $UV -m sglang_router.launch_router \
        --pd-disaggregation \
        --prefill "http://127.0.0.1:${P1_PORT}" \
        --decode  "http://127.0.0.1:${D1_PORT}" \
        --host "$HOST" \
        --port "$R_PORT" \
        --policy cache_aware &

    echo ""
    echo "✅ All processes started!"
    echo "   Workers take 60–120s to load the 106B model..."
    echo "   Monitor: uv run python server/health_check.py"
    wait
}

# ── Entrypoint ────────────────────────────────────────────────────────
case "${1:-serve}" in
    install) install ;;
    serve)   serve ;;
    *)       echo "Usage: $0 {install|serve}"; exit 1 ;;
esac
