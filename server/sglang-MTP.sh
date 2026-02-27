#!/usr/bin/env bash
# sglang-MTP.sh — 2P2D disaggregated serving with MTP speculative decoding
# Same as sglang_start.sh but enables the built-in MTP head on GLM-4.5-Air-FP8
# for speculative token generation (EAGLE algorithm, no separate draft model).

set -euo pipefail

# ── Config (override via env) ───────────────────────────────────────
MODEL="${MODEL_PATH:-zai-org/GLM-4.5-Air-FP8}"
HOST="${HOST_IP:-0.0.0.0}"
MAX_LEN="${MAX_MODEL_LEN:-32768}"
TP="${TP_SIZE:-1}"

P1_PORT="${PREFILL_PORT_1:-30000}"
P2_PORT="${PREFILL_PORT_2:-30002}"
D1_PORT="${DECODE_PORT_1:-30001}"
D2_PORT="${DECODE_PORT_2:-30003}"
R_PORT="${ROUTER_PORT:-8000}"
MC_PORT="${MOONCAKE_MASTER_PORT:-50051}"

# MTP / speculative decoding settings
SPEC_ALGO="${SPEC_ALGO:-EAGLE}"
SPEC_NUM_STEPS="${SPEC_NUM_STEPS:-5}"
SPEC_EAGLE_TOPK="${SPEC_EAGLE_TOPK:-4}"
SPEC_NUM_DRAFT="${SPEC_NUM_DRAFT:-8}"

export MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-tcp}"
export MOONCAKE_DEVICE="${MOONCAKE_DEVICE:-}"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="${MOONCAKE_GLOBAL_SEG:-4gb}"
export MOONCAKE_MASTER="127.0.0.1:${MC_PORT}"

UV="uv run --extra sglang"

# ── MTP flags (appended to every worker) ────────────────────────────
MTP_FLAGS="--speculative-algorithm ${SPEC_ALGO} \
--speculative-num-steps ${SPEC_NUM_STEPS} \
--speculative-eagle-topk ${SPEC_EAGLE_TOPK} \
--speculative-num-draft-tokens ${SPEC_NUM_DRAFT}"

# ── Helpers ─────────────────────────────────────────────────────────
cleanup() { kill $(jobs -p) 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT INT TERM

serve() {
    echo "════════════════════════════════════════════════════════════"
    echo " 2P2D + MTP · ${MODEL} · TP=${TP} · ${MOONCAKE_PROTOCOL}"
    echo " Spec: algo=${SPEC_ALGO}  steps=${SPEC_NUM_STEPS}  topk=${SPEC_EAGLE_TOPK}  draft=${SPEC_NUM_DRAFT}"
    echo "════════════════════════════════════════════════════════════"

    # 1) Mooncake master
    $UV -m mooncake.master_service \
        --enable_http_metadata_server=true \
        --http_metadata_server_port=8080 &
    sleep 3

    # 2) Prefill workers
    $UV -m sglang.launch_server --model-path "$MODEL" --tp-size "$TP" \
        --disaggregation-mode prefill --base-gpu-id 0 \
        --host "$HOST" --port "$P1_PORT" --max-total-tokens "$MAX_LEN" \
        --trust-remote-code $MTP_FLAGS &
    sleep 2

    $UV -m sglang.launch_server --model-path "$MODEL" --tp-size "$TP" \
        --disaggregation-mode prefill --base-gpu-id 1 \
        --host "$HOST" --port "$P2_PORT" --max-total-tokens "$MAX_LEN" \
        --trust-remote-code $MTP_FLAGS &
    sleep 2

    # 3) Decode workers
    $UV -m sglang.launch_server --model-path "$MODEL" --tp-size "$TP" \
        --disaggregation-mode decode --base-gpu-id 2 \
        --host "$HOST" --port "$D1_PORT" --max-total-tokens "$MAX_LEN" \
        --trust-remote-code $MTP_FLAGS &
    sleep 2

    $UV -m sglang.launch_server --model-path "$MODEL" --tp-size "$TP" \
        --disaggregation-mode decode --base-gpu-id 3 \
        --host "$HOST" --port "$D2_PORT" --max-total-tokens "$MAX_LEN" \
        --trust-remote-code $MTP_FLAGS &
    sleep 2

    # 4) Router
    $UV -m sglang.srt.disaggregation.mini_lb \
        --prefill "http://127.0.0.1:${P1_PORT},http://127.0.0.1:${P2_PORT}" \
        --decode  "http://127.0.0.1:${D1_PORT},http://127.0.0.1:${D2_PORT}" \
        --host "$HOST" --port "$R_PORT" &

    echo ""
    echo "✅ Router ready → http://${HOST}:${R_PORT}  (MTP speculative decoding ON)"
    wait
}

case "${1:-serve}" in
    serve) serve ;;
    *)     echo "Usage: $0 {serve}  (run 'bash server/sglang_start.sh install' first)"; exit 1 ;;
esac
