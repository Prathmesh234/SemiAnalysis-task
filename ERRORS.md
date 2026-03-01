# SGLang + Mooncake Disaggregated Serving Integration Logs

This document tracks all the roadblocks and corresponding fixes encountered while setting up a 1P1D SGLang disaggregated serving environment with the 106B parameter `GLM-4.5-Air-FP8` model on 4x H100s, targeting the SWE-bench evaluation via the `claude` CLI.

## Resolved Errors

1. **Mooncake Master Service Module Not Found**
   - **Error**: `No module named mooncake.master_service`
   - **Fix**: SGLang PR 5460 removed the requirement for a separate master JSON. Used `--port 8080 -Dprotocol="tcp"` on `mooncake.mooncake_store_service` instead.

2. **RDMA Dependencies Missing (`libibverbs.so.1`)**
   - **Error**: `ImportError: libibverbs.so.1` upon initializing the Mooncake Transfer Engine.
   - **Fix**: Installed Ubuntu system dependencies (`libibverbs-dev` and associated RDMA/infiniband utilities).

3. **Topology / GPU Memory Constraints (OOM)**
   - **Error**: Launching a 2P2D (2 prefill, 2 decode) with `TP=1` fails on H100 80GB VRAM because the ~106GB model weights exceed single GPU capacity.
   - **Fix**: Switched to `TP=2`. The 4 GPUs are split into `1P1D` (GPUs 0+1 = Prefill Worker, GPUs 2+3 = Decode Worker).

4. **SGLang Router Missing Module**
   - **Error**: `No module named sglang.srt.disaggregation.mini_lb`
   - **Fix**: SGLang v0.5.9 migrated the router module to `sglang_router.launch_router`.

5. **Prometheus Metrics Not Available**
   - **Error**: SGLang `/metrics` returned `404 Not Found`.
   - **Fix**: Added the `--enable-metrics` flag to both prefill and decode worker launch commands.

6. **Double Endpoint Path (`/v1/v1/messages`)**
   - **Error**: SGLang router threw 404s for Claude CLI queries.
   - **Fix**: Removed `/v1` from the `ANTHROPIC_BASE_URL` in `.env` as the `claude` CLI automatically appends `/v1/messages`.

7. **Decode Worker Healthcheck Failure (Deadlock)**
   - **Error**: `Failed to select PD pair error=No available decode workers (all circuits open or unhealthy)` and HTTP health checks on `:30001` timed out.
   - **Fix**: The SWE-bench prompts were too large (~41k tokens total) for the default `MAX_MODEL_LEN=32768`. Increased this to `MAX_MODEL_LEN=65536`.

---

## Resolved: Format Translation / SSE Mismatch

8. **SSE Schema Mismatch — Empty Responses from Claude CLI**
   - **Error**: Proxy returned `200 OK` with `"content": []` and `"usage": {"input_tokens": 0, "output_tokens": 0}`. The `claude` CLI showed 0 tokens.
   - **Root Cause**: Five bugs in the OpenAI → Anthropic SSE translation layer:
     1. **Missing `ping` event** — Anthropic SSE requires `event: ping` after `content_block_start`. The Claude SDK's httpx-sse parser uses this as a stream-alive signal.
     2. **Missing `stream_options`** — Without `{"include_usage": true}` in the OpenAI request, SGLang never sends token counts in streaming chunks, so `output_tokens` was always 0.
     3. **`content: null` not handled** — SGLang can return `"content": null` instead of `""`. `msg.get("content", "")` returns `None` (not `""`), and `if None:` is falsy → empty content array.
     4. **Fragile first-chunk detection** — Used a monotonic `chunk_index` counter that always incremented, even for role-only chunks with no content. Replaced with a `sent_header` boolean that only flips on the first chunk with actual choices.
     5. **No graceful stream termination** — If SGLang closed the stream without a `finish_reason`, the client never received `content_block_stop` / `message_delta` / `message_stop`, leaving the Claude CLI hanging.
   - **Fix**: Rewrote the streaming generator in `proxy.py` to emit the full Anthropic SSE event sequence (`message_start` → `content_block_start` → `ping` → `content_block_delta`* → `content_block_stop` → `message_delta` → `message_stop`) with proper null-safety and graceful termination.
