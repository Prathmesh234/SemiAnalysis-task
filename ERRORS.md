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

## Current Issue: Format Translation / SSE Mismatch

**State:**
The server stack is fully up, the proxy processes the requests and returns a `200 OK`. However, the Claude CLI receives an empty response and `0` tokens. 

**Symptoms:**
When hitting the Anthropic mock endpoint directly via `curl` with `stream: false`, the proxy spits back:
```json
{
    "id": "msg_1772397840",
    "type": "message",
    "role": "assistant",
    "content": [],
    "model": "zai-org/GLM-4.5-Air-FP8",
    "stop_reason": "end_turn",
    "stop_sequence": null,
    "usage": {
        "input_tokens": 0,
        "output_tokens": 0
    }
}
```
**Core Problem Theory:**
The root cause lies in the translation layer between the SGLang Server (which speaks the **OpenAI Chat Completions** and SSE stream format) and the `claude` CLI (which strictly speaks the **Anthropic Messages** and Anthropic SSE stream format). 

1. **Streaming Issues (SSE Formats):** SGLang yields `data: {"id": "...", "choices": [{"delta": {"content": "..."}}]}`. The Anthropic CLI expects fine-grained `message_start`, `content_block_start`, `content_block_delta` (containing `text_delta`), `content_block_stop`, and `message_stop` events.
2. **Missing Content Mapping:** For non-stream requests, `openai_data.get("choices")[0].get("message", {}).get("content", "")` might be entirely empty, missing, or structured differently by SGLang compared to what our proxy implementation expects. 

**Next Actionable Step:**
We have added raw verbose logging to `proxy.py` to capture exactly what SGLang is outputting to stdout (`[Proxy] Raw SGLang Response: {...}`). We will analyze the shape of the SGLang OpenAI JSON payload (both static and streamed) to correctly map its tokens to the Anthropic SSE format.
