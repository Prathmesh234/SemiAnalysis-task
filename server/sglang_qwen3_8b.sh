#!/usr/bin/env bash
# sglang_qwen3_8b.sh — Simple single-GPU serving: Qwen3-8B via SGLang
#
# Lightweight setup for testing the Claude Code → proxy → SGLang pipeline
# without burning multiple GPUs on the large 235B model.
#
# GPU layout: 1× H100 80 GB (TP=1)
#
#   ┌─────────────────────────────┐
#   │  SGLang Server              │
#   │  Qwen3-8B  (TP=1, GPU 0)   │
#   │  port: 8000                 │
#   │  OpenAI-compat /v1/...      │
#   └─────────────────────────────┘
#          ▲
#          │  Anthropic→OpenAI translation
#   ┌──────┴──────────┐
#   │  Metrics Proxy   │
#   │  (:8001)         │
#   └─────────────────┘
#          ▲
#          │  Anthropic Messages API
#   ┌──────┴──────────┐
#   │  Claude Code     │
#   │  CLI / harness   │
#   └─────────────────┘
#
# Usage:
#   bash server/sglang_qwen3_8b.sh           # serve (default)
#   bash server/sglang_qwen3_8b.sh install   # only sync deps

set -euo pipefail

# ── Load .env if present ─────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../.env" ]]; then
    set -a; source "${SCRIPT_DIR}/../.env"; set +a
elif [[ -f ".env" ]]; then
    set -a; source ".env"; set +a
fi

# ── Config (override via env) ─────────────────────────────────────────
# Default to the instruct model; use Qwen/Qwen3-8B-FP8 for quantised variant
MODEL="${QWEN3_MODEL_PATH:-Qwen/Qwen3-8B}"
HOST="${HOST_IP:-0.0.0.0}"
PORT="${QWEN3_PORT:-8000}"
MAX_LEN="${QWEN3_MAX_MODEL_LEN:-32768}"

# TP=1: Qwen3-8B is ~16 GB in BF16 (or ~8 GB in FP8) — fits on a single H100
TP="${QWEN3_TP_SIZE:-1}"

# Which GPU to use (0-indexed). Set to e.g. 3 to use the last GPU and leave
# GPUs 0-2 free for the larger model.
BASE_GPU="${QWEN3_BASE_GPU_ID:-0}"

# Reasoning parser for Qwen3 thinking mode (produces <think>...</think> blocks)
REASONING_PARSER="${QWEN3_REASONING_PARSER:-qwen3}"

# Memory fraction SGLang can use for KV-cache on the GPU (0.0–1.0)
MEM_FRACTION="${QWEN3_MEM_FRACTION:-0.90}"

UV="uv run --extra sglang"

# ── Helpers ───────────────────────────────────────────────────────────
cleanup() { kill $(jobs -p) 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT INT TERM

install() {
    echo "▸ Syncing uv deps (sglang)…"
    uv sync --extra sglang
    echo "✅ Done."
}

serve() {
    echo "════════════════════════════════════════════════════════════"
    echo " Qwen3-8B · Single-GPU · TP=${TP} · GPU ${BASE_GPU}"
    echo " Model:  ${MODEL}"
    echo " Server: http://${HOST}:${PORT}"
    echo " Max context: ${MAX_LEN} tokens"
    echo " Reasoning parser: ${REASONING_PARSER}"
    echo " Mem fraction: ${MEM_FRACTION}"
    echo "════════════════════════════════════════════════════════════"

    exec $UV -m sglang.launch_server \
        --model-path "$MODEL" \
        --tp-size "$TP" \
        --base-gpu-id "$BASE_GPU" \
        --host "$HOST" \
        --port "$PORT" \
        --max-total-tokens "$MAX_LEN" \
        --mem-fraction-static "$MEM_FRACTION" \
        --reasoning-parser "$REASONING_PARSER" \
        --trust-remote-code \
        --enable-metrics
}

# ── Entrypoint ────────────────────────────────────────────────────────
case "${1:-serve}" in
    install) install ;;
    serve)   serve ;;
    *)       echo "Usage: $0 {install|serve}"; exit 1 ;;
esac
