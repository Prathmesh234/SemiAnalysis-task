# AGENTS.md — Complete Guide to the GLM-4.5-Air-FP8 H100 Benchmark Pipeline

This document is the single source of truth for understanding, running, and interpreting the full benchmark pipeline. It covers every component from spinning up the inference server to reading the metrics that tell you how the H100s are performing under a multi-turn agentic workload.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Layout](#2-repository-layout)
3. [Prerequisites & Environment Setup](#3-prerequisites--environment-setup)
4. [Step 1 — Install Dependencies](#4-step-1--install-dependencies)
5. [Step 2 — Start the SGLang Inference Server (2P2D)](#5-step-2--start-the-sglang-inference-server-2p2d)
6. [Step 3 — Start the Metrics Proxy](#6-step-3--start-the-metrics-proxy)
7. [Step 4 — Run the Benchmark](#7-step-4--run-the-benchmark)
8. [Step 5 — Read the Metrics](#8-step-5--read-the-metrics)
9. [MTP Variant — Speculative Decoding](#9-mtp-variant--speculative-decoding)
10. [Component Deep Dives](#10-component-deep-dives)
    - [10a. sglang_start.sh](#10a-sglang_startsh)
    - [10b. proxy.py](#10b-proxypy)
    - [10c. shell_executor.py](#10c-shell_executorpy)
    - [10d. turn_manager.py](#10d-turn_managerpy)
    - [10e. task_queue.py](#10e-task_queuepy)
    - [10f. load_swebench.py](#10f-load_swebenchpy)
    - [10g. collector.py](#10g-collectorpy)
    - [10h. report.py](#10h-reportpy)
    - [10i. health_check.py](#10i-health_checkpy)
11. [Environment Variable Reference](#11-environment-variable-reference)
12. [Interpreting H100 Performance Metrics](#12-interpreting-h100-performance-metrics)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. System Overview

The full pipeline has four distinct layers that run concurrently:

```
┌──────────────────────────────────────────────────────────────────────┐
│  LAYER 4 — Benchmark Client  (run_benchmark.py)                      │
│  Loads SWE-bench tasks → multi-turn claude CLI loop → collects data  │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │ HTTP requests to :8001
┌────────────────────────────────▼─────────────────────────────────────┐
│  LAYER 3 — Metrics Proxy  (proxy/proxy.py  →  :8001)                │
│  Transparent FastAPI reverse-proxy; logs every req/resp to           │
│  metrics.jsonl with latency, path, bytes, token counts              │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │ HTTP forward to :8000
┌────────────────────────────────▼─────────────────────────────────────┐
│  LAYER 2 — SGLang Router  (mini_lb  →  :8000)                       │
│  Load-balances across the 2 prefill & 2 decode workers               │
└────────┬──────────────────────────────────────┬──────────────────────┘
         │                                      │
┌────────▼────────┐                   ┌─────────▼────────┐
│  Prefill Pool   │                   │   Decode Pool    │
│  P-1  :30000    │ ←── Mooncake ──→  │   D-1  :30001   │
│  P-2  :30002    │   KV transfer     │   D-2  :30003   │
└─────────────────┘                   └──────────────────┘
         ↑
┌────────┴────────┐
│ Mooncake Master │  :50051  +  HTTP metadata :8080
│ (KV pool mgr)   │
└─────────────────┘

  LAYER 1 — Inference Server
```

**Data flow in a single request:**

```
run_benchmark.py
  → claude --print "<prompt>"          # shell_executor.py
      → ANTHROPIC_BASE_URL (:8001)     # the proxy
          → :8000 (SGLang router)      # load-balanced
              → Prefill worker         # fills KV cache
              → Mooncake transfer      # moves KV to decode
              → Decode worker          # streams tokens back
          ← response + usage{} 
      ← metrics.jsonl updated          # proxy logs the entry
  ← stdout captured by turn_manager
```

---

## 2. Repository Layout

```
semianalysis-task/
├── AGENTS.md                    ← you are here
├── README.md                    ← high-level project README
├── .env.example                 ← copy to .env and edit
├── pyproject.toml               ← all Python deps, managed by uv
│
├── server/
│   ├── sglang_start.sh          ← launches 2P2D stack (no speculative decoding)
│   ├── sglang-MTP.sh            ← same + MTP speculative decoding enabled
│   ├── vllm_start.sh            ← alternative: vLLM expert-parallel launch
│   └── health_check.py          ← polls :8000/health until ready
│
├── proxy/
│   └── proxy.py                 ← FastAPI reverse-proxy (:8001 → :8000)
│
├── client/
│   ├── shell_executor.py        ← subprocess wrapper: `claude --print`
│   ├── turn_manager.py          ← multi-turn loop, up to 5 turns per task
│   └── task_queue.py            ← async task queue with bounded concurrency
│
├── datasets/
│   └── load_swebench.py         ← loads princeton-nlp/SWE-bench_Verified
│
├── metrics/
│   ├── collector.py             ← parses metrics.jsonl → AggregatedMetrics
│   └── report.py                ← markdown table + Pareto scatter plot
│
└── run_benchmark.py             ← single CLI entrypoint for the whole run
```

**Runtime-generated files** (not committed):

| File | Created by | Contains |
|------|-----------|---------|
| `metrics.jsonl` | `proxy/proxy.py` | One JSON line per LLM request |
| `pareto.png` | `metrics/report.py` | Latency vs throughput scatter plot |
| `.env` | You (copy from `.env.example`) | Runtime config |

---

## 3. Prerequisites & Environment Setup

### Hardware

| Item | Minimum |
|------|---------|
| GPUs | 4× NVIDIA H100 80GB SXM (one per P/D worker) |
| CUDA | 12.1+ |
| System RAM | 128 GB+ |
| Network | RDMA NICs (Mellanox ConnectX-6/7) or TCP fallback |
| OS | Ubuntu 22.04+ |

### Software

- **Python 3.12+** — required by `pyproject.toml`
- **uv** — all dep management and Python execution goes through `uv`
- **claude CLI** — the benchmark client calls `claude --print` as a subprocess; it must be on `$PATH`

### Install `uv` (if not present)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone and configure

```bash
git clone https://github.com/Prathmesh234/MultiTurnKernel-SFT.git
cd MultiTurnKernel-SFT

# Create your local env config
cp .env.example .env
# Edit .env:
#   MOONCAKE_PROTOCOL=rdma     # if you have RDMA NICs
#   MOONCAKE_DEVICE=mlx5_0     # your NIC device name  (ibv_devices to list)
#   MAX_MODEL_LEN=65536        # increase for long-context
```

---

## 4. Step 1 — Install Dependencies

Everything is managed by `uv`. Run this **once** on the GPU server:

```bash
bash server/sglang_start.sh install
```

This runs `uv sync --extra sglang` which installs:

| Package | Why |
|---------|-----|
| `sglang[all]>=0.4.0` | The inference engine + router |
| `mooncake-transfer-engine` | KV-cache RDMA transfer between P & D workers |
| `fastapi`, `uvicorn` | The metrics proxy |
| `httpx` | Async HTTP client used by proxy and health check |
| `datasets` | HuggingFace Datasets — to load SWE-bench |
| `matplotlib` | Pareto plot generation |
| `python-dotenv` | Load `.env` file |

> **Note:** `vllm` and `sglang` have conflicting transitive deps and cannot coexist. The `pyproject.toml` enforces this via `[tool.uv] conflicts`. Only install one at a time.

---

## 5. Step 2 — Start the SGLang Inference Server (2P2D)

In a **dedicated terminal** (keep it running for the whole experiment):

```bash
bash server/sglang_start.sh serve
```

This launches **6 background processes** in order:

### Process startup sequence

```
t=0s   Mooncake Master (:50051, HTTP meta :8080)
         ↳ orchestrates the distributed KV-cache memory pool
t=3s   Prefill Worker 1 — GPU 0, port 30000
         ↳ sglang.launch_server --disaggregation-mode prefill --base-gpu-id 0
t=5s   Prefill Worker 2 — GPU 1, port 30002
         ↳ sglang.launch_server --disaggregation-mode prefill --base-gpu-id 1
t=7s   Decode Worker 1  — GPU 2, port 30001
         ↳ sglang.launch_server --disaggregation-mode decode  --base-gpu-id 2
t=9s   Decode Worker 2  — GPU 3, port 30003
         ↳ sglang.launch_server --disaggregation-mode decode  --base-gpu-id 3
t=11s  Router — port 8000 (CPU only)
         ↳ sglang.srt.disaggregation.mini_lb
             --prefill http://127.0.0.1:30000,http://127.0.0.1:30002
             --decode  http://127.0.0.1:30001,http://127.0.0.1:30003
```

All processes are managed by the shell's job table. `Ctrl+C` sends `SIGINT` and the `trap cleanup` cleanly kills all children.

### Why 2P2D on 4× H100?

| Approach | Problem |
|----------|---------|
| Standard (co-located) | Prefill (compute-bound) and decode (memory-bound) fight for the same GPU resources, hurting both TTFT and TPOT |
| **2P2D disaggregated** | Prefill GPUs (0,1) are dedicated to heavy prompt computation; decode GPUs (2,3) focus on streaming tokens. Mooncake moves the KV cache between them via RDMA |

With GLM-4.5-Air's MoE architecture (106B total / 12B active per token), the KV cache is relatively small per active expert — making the RDMA transfer cost lower than the benefit of eliminating resource contention.

### Wait for healthy

In another terminal, run this to block until all workers are up (up to 5 minutes):

```bash
uv run python server/health_check.py
```

It polls `http://localhost:8000/health` every 5 seconds and exits 0 when 200 OK is returned. Model loading typically takes 60–120 seconds.

### Quick sanity check

```bash
curl http://127.0.0.1:8000/health
# → {"status":"ok"}

curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"zai-org/GLM-4.5-Air-FP8","messages":[{"role":"user","content":"Say hello."}]}'
```

---

## 6. Step 3 — Start the Metrics Proxy

In a **second dedicated terminal**:

```bash
uv run uvicorn proxy.proxy:app --host 0.0.0.0 --port 8001
```

The proxy is a FastAPI app that:
1. Receives every request from the benchmark client on **`:8001`**
2. Forwards it unchanged to the SGLang router on **`:8000`**
3. Measures wall-clock latency with `time.perf_counter()`
4. Parses the OpenAI-compatible response body for `usage.prompt_tokens` and `usage.completion_tokens`
5. Appends one JSON line to **`metrics.jsonl`** in the current working directory

### What each JSONL record looks like

```json
{
  "ts": 1740682800.123,
  "path": "/v1/chat/completions",
  "latency_s": 2.3841,
  "req_bytes": 512,
  "resp_bytes": 1204,
  "usage": {
    "prompt_tokens": 128,
    "completion_tokens": 64
  }
}
```

| Field | Source | Meaning |
|-------|--------|---------|
| `ts` | `time.time()` | Unix timestamp of response |
| `path` | Request URL | Which endpoint was hit |
| `latency_s` | `perf_counter` diff | Full round-trip time (client → proxy → sglang → proxy → client) |
| `req_bytes` | `len(body)` | Request body size in bytes |
| `resp_bytes` | `len(content)` | Response body size in bytes |
| `usage.prompt_tokens` | Parsed from response | Tokens in the prompt (set by SGLang) |
| `usage.completion_tokens` | Parsed from response | Tokens generated |

> **`ANTHROPIC_BASE_URL`** in your `.env` must point to the proxy: `http://localhost:8001/v1`. The claude CLI reads this env var to know where to send requests.

---

## 7. Step 4 — Run the Benchmark

In a **third terminal**, with the server and proxy already running:

```bash
# Minimal smoke test — 5 tasks, 2 concurrent
uv run python run_benchmark.py --n-tasks 5 --concurrency 2

# Full run — all SWE-bench Verified (500 tasks), 4 concurrent
uv run python run_benchmark.py --concurrency 4

# Custom
uv run python run_benchmark.py --n-tasks 50 --concurrency 8
```

Or via the installed script alias:

```bash
uv run benchmark --n-tasks 50 --concurrency 4
```

### What happens inside `run_benchmark.py`

```
main()
  ├─ load_swebench_tasks(n_tasks=N)       # datasets/load_swebench.py
  │    → downloads princeton-nlp/SWE-bench_Verified (test split)
  │    → converts each row to TaskItem(instance_id, prompt, repo, base_commit, metadata)
  │
  ├─ run_queue(tasks, _worker, concurrency=C)   # client/task_queue.py
  │    → creates asyncio.Queue with all (idx, task) pairs
  │    → spins up C asyncio workers, each guarded by a Semaphore(C)
  │    → each worker calls _worker(task) and stores result by index
  │    → returns results list in original task order
  │
  │    _worker(task)
  │      └─ run_multi_turn(task.prompt)         # client/turn_manager.py
  │           → Turn 1: run_claude_print(initial_prompt)
  │           → Turn 2: run_claude_print(initial_prompt + "\n\n# Previous output:\n" + stdout_1)
  │           → Turn N: ... (up to max_turns=5, stops early on any returncode != 0)
  │           → returns TurnResult(turns=[...], final_output=stdout_last_turn)
  │
  │           run_claude_print(prompt)           # client/shell_executor.py
  │             → asyncio.create_subprocess_exec("claude", "--print", prompt)
  │             → captures stdout + stderr, timeout=600s
  │             → returns ExecutionResult(stdout, stderr, returncode)
  │
  ├─ collect()                                  # metrics/collector.py
  │    → reads metrics.jsonl
  │    → returns AggregatedMetrics
  │
  └─ print(markdown_summary(metrics))           # metrics/report.py
```

### CLI arguments

| Flag | Default | Meaning |
|------|---------|---------|
| `--domain` | `swebench` | Task domain (only `swebench` is wired up) |
| `--concurrency` | `4` | Max simultaneous multi-turn tasks |
| `--n-tasks` | `None` (all) | Truncate dataset to first N tasks |

### Concurrency guidance for H100s

| `--concurrency` | Effect |
|----------------|--------|
| `1` | Sequential — good for latency profiling per task |
| `4` | Default — moderate throughput, GPU well utilised |
| `8–16` | Higher throughput — watch for latency P99 spike |
| `>16` | Risk of backpressure / OOM depending on model length |

Each concurrent task maps to one `claude --print` subprocess talking to GLM via the proxy. The SGLang router handles request batching internally across the P and D workers.

---

## 8. Step 5 — Read the Metrics

### Live tail (while benchmark runs)

```bash
tail -f metrics.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    r = json.loads(line)
    u = r.get('usage', {})
    print(f\"{r['latency_s']:6.2f}s  {u.get('prompt_tokens',0):5}p  {u.get('completion_tokens',0):5}c  {r['path']}\")
"
```

### Aggregate summary (after run)

```bash
uv run python -c "
from metrics.collector import collect
from metrics.report import markdown_summary, pareto_plot
m = collect()
print(markdown_summary(m))
pareto_plot('pareto.png')
print('Plot saved to pareto.png')
"
```

### What the metrics mean

| Metric | Field | What it tells you about the H100s |
|--------|-------|-----------------------------------|
| **Total requests** | `total_requests` | How many LLM API calls fired across all turns |
| **Avg latency** | `avg_latency_s` | Mean time per completion — dominated by decode speed |
| **P50 latency** | `p50_latency_s` | Median; good representation of typical request |
| **P99 latency** | `p99_latency_s` | Tail latency; reveals queue pressure or memory issues |
| **Prompt tokens** | `total_prompt_tokens` | Total prefill work done across all requests |
| **Completion tokens** | `total_completion_tokens` | Total decode work; divide by total wall time for tok/s |

### Derived throughput numbers you care about

```python
# After collecting metrics:
m = collect()

# Overall token throughput (completion tokens per second of wall-clock time)
toks_per_sec = m.total_completion_tokens / m.total_latency_s

# Mean tokens per request
mean_tokens = m.total_completion_tokens / m.total_requests

# Time-to-first-token proxy (latency on single-token completions, if any)
# Not directly measurable from this proxy — would need streaming extension
```

### Pareto plot

`pareto.png` shows a scatter of per-request **latency (x)** vs **token throughput (y)** with a red star for the aggregate average. A cluster in the bottom-left (low latency, low throughput) is typical for short prompts; a spread toward top-right shows the GPU ramping up for long generations.

---

## 9. MTP Variant — Speculative Decoding

GLM-4.5-Air has a **built-in MTP (Multi-Token Prediction) head** — 1 layer, 0.3 scaling factor. SGLang activates it via the EAGLE algorithm (SGLang treats MTP/EAGLE/NEXTN as equivalent for models with native MTP heads — no separate draft model is needed).

### Start with MTP

Replace step 2 with:

```bash
bash server/sglang-MTP.sh serve
```

Everything else is identical. The MTP flags added per worker are:

| Flag | Value | Effect |
|------|-------|--------|
| `--speculative-algorithm` | `EAGLE` | Activates the MTP head as the draft model |
| `--speculative-num-steps` | `5` | Depth of speculative drafting per step |
| `--speculative-eagle-topk` | `4` | Branching factor — candidate diversity |
| `--speculative-num-draft-tokens` | `8` | Max tokens verified in parallel |

Tune via env vars:

```bash
SPEC_NUM_STEPS=3 SPEC_EAGLE_TOPK=2 bash server/sglang-MTP.sh serve
```

### Expected impact on H100 metrics

MTP speculative decoding improves **TPOT (time per output token)** only if the draft acceptance rate is high. For GLM-4.5-Air with its 1 MTP layer, expect:
- **~1.3–1.8× decode speedup** on short to medium completions
- **Acceptance rate drops** on highly stochastic outputs (creative, diverse) — tune `SPEC_NUM_STEPS` down if P99 latency increases
- **No change to TTFT** — prefill is unaffected by speculative decoding

---

## 10. Component Deep Dives

### 10a. `server/sglang_start.sh`

```bash
UV="uv run --extra sglang"   # all Python calls go through uv
```

- `install` subcommand: `uv sync --extra sglang` — pulls `sglang[all]` + `mooncake-transfer-engine`
- `serve` subcommand: background-launches all 6 processes, `trap cleanup EXIT INT TERM` kills them all on exit
- `--base-gpu-id N` maps each worker to a specific GPU without needing `CUDA_VISIBLE_DEVICES`
- `--disaggregation-mode prefill|decode` sets which phase the worker handles
- Mooncake env vars (`MOONCAKE_MASTER`, `MOONCAKE_PROTOCOL`, `MOONCAKE_GLOBAL_SEGMENT_SIZE`) are exported before any worker starts

### 10b. `proxy/proxy.py`

```python
BACKEND_URL = "http://localhost:8000"   # SGLang router
METRICS_PATH = Path("metrics.jsonl")    # appended line-by-line
```

- Single FastAPI app with a `/{path:path}` catch-all route — transparently proxies `GET`, `POST`, `PUT`, `DELETE`
- Uses `httpx.AsyncClient` with a 300-second timeout (safe for long completions)
- Timing uses `time.perf_counter()` — high-resolution, monotonic, no NTP drift
- Token counts are best-effort: parsed from `response.usage` only if the body is valid JSON. Streaming responses will not have token counts — only final aggregated responses do

### 10c. `client/shell_executor.py`

```python
proc = await asyncio.create_subprocess_exec(
    "claude", "--print", prompt,
    stdout=PIPE, stderr=PIPE, env=env,
)
```

- Calls the `claude` CLI binary — this must be installed and on `$PATH`
- `env_overrides` dict is merged on top of `os.environ`, letting you inject `ANTHROPIC_BASE_URL` per-task if needed
- Hard timeout of 600 seconds — any task taking longer is killed and returns `returncode=-1`, `stderr="TIMEOUT"`
- The proxy routes the claude CLI's outbound HTTPS call to the local SGLang server via `ANTHROPIC_BASE_URL=http://localhost:8001/v1`

### 10d. `client/turn_manager.py`

```python
for _ in range(max_turns):           # default max_turns=5
    exec_result = await run_claude_print(prompt)
    if exec_result.returncode != 0:
        break                         # stop on failure/timeout
    prompt = f"{initial_prompt}\n\n# Previous output:\n{exec_result.stdout}"
```

- Each turn appends the previous turn's **full stdout** to the original prompt, giving the agent context to refine its answer
- Terminates early on any non-zero return code (error or timeout)
- `TurnResult.turns` accumulates all `ExecutionResult` objects — you can inspect intermediate outputs
- `TurnResult.final_output` is always the last successful stdout

### 10e. `client/task_queue.py`

```python
sem = asyncio.Semaphore(concurrency)
workers = [asyncio.create_task(_worker()) for _ in range(concurrency)]
```

- Uses an `asyncio.Queue` pre-populated with `(idx, task)` tuples
- `concurrency` parallel workers each grab tasks via `queue.get_nowait()` inside a `Semaphore`
- Results are stored in a dict keyed by original index, then sorted back to original order before returning — **order is preserved regardless of completion order**
- The `Semaphore` and `Queue` cooperate to prevent over-dispatch when tasks complete at different speeds

### 10f. `datasets/load_swebench.py`

```python
DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
ds = load_dataset(DATASET_NAME, split="test")
```

- Streams from HuggingFace Hub on first run, cached locally on subsequent runs (`~/.cache/huggingface/datasets/`)
- Each row becomes a `TaskItem` with:
  - `instance_id` — unique SWE-bench instance identifier
  - `prompt` — the raw `problem_statement` field (the bug description / issue text)
  - `repo` — GitHub repo (e.g. `django/django`)
  - `base_commit` — git SHA at the time of the issue
  - `metadata.hints` — optional hints text
- `n_tasks` truncates by breaking early from the for-loop — no shuffle, always the first N items

### 10g. `metrics/collector.py`

```python
@dataclass
class AggregatedMetrics:
    total_requests: int
    total_latency_s: float
    avg_latency_s: float
    p50_latency_s: float          # latencies[n // 2]
    p99_latency_s: float          # latencies[min(int(n * 0.99), n-1)]
    total_prompt_tokens: int
    total_completion_tokens: int
```

- Reads `metrics.jsonl` line-by-line; gracefully returns empty metrics if the file doesn't exist
- All latencies are sorted before percentile computation — P50 is median, P99 uses `min()` guard against index out-of-range on small datasets
- Token counts are summed from `usage.prompt_tokens` / `usage.completion_tokens` — records without `usage` (e.g. `/health` pings) contribute 0

### 10h. `metrics/report.py`

**`markdown_summary(metrics)`** — returns a formatted markdown table string, printed to stdout after the benchmark finishes.

**`pareto_plot(output_path)`** — saved to `pareto.png`:
- X-axis: per-request latency in seconds
- Y-axis: completion tokens / latency (token throughput per request)  
- Grey dots: individual requests
- Red star: aggregate average point

### 10i. `server/health_check.py`

```python
HEALTH_URL = "http://localhost:8000/health"
POLL_INTERVAL_S = 5
TIMEOUT_S = 300          # give up after 5 minutes
```

- Polls with `httpx.AsyncClient` — returns `True` as soon as a 200 response is received
- Exits with code 0 (healthy) or 1 (timeout) — suitable for use in CI wait loops: `python server/health_check.py || exit 1`

---

## 11. Environment Variable Reference

Copy `.env.example` → `.env` and edit before running anything.

| Variable | Default | Set in | Used by |
|----------|---------|--------|---------|
| `MODEL_PATH` | `zai-org/GLM-4.5-Air-FP8` | `.env` | `sglang_start.sh` |
| `ANTHROPIC_BASE_URL` | `http://localhost:8001/v1` | `.env` | `claude` CLI → hits the proxy |
| `ANTHROPIC_API_KEY` | `dummy` | `.env` | `claude` CLI (SGLang doesn't check it) |
| `HOST_IP` | `0.0.0.0` | `.env` | All workers + router |
| `MAX_MODEL_LEN` | `32768` | `.env` | `--max-total-tokens` on workers |
| `TP_SIZE` | `1` | `.env` | `--tp-size` per worker |
| `PREFILL_PORT_1` | `30000` | `.env` | Prefill worker 1 |
| `PREFILL_PORT_2` | `30002` | `.env` | Prefill worker 2 |
| `DECODE_PORT_1` | `30001` | `.env` | Decode worker 1 |
| `DECODE_PORT_2` | `30003` | `.env` | Decode worker 2 |
| `ROUTER_PORT` | `8000` | `.env` | Router (mini_lb) |
| `MOONCAKE_MASTER_PORT` | `50051` | `.env` | Mooncake master gRPC |
| `MOONCAKE_PROTOCOL` | `tcp` | `.env` | `rdma` for production |
| `MOONCAKE_DEVICE` | *(auto)* | `.env` | e.g. `mlx5_0,mlx5_1` |
| `MOONCAKE_GLOBAL_SEG` | `4gb` | `.env` | KV pool per worker |
| `SPEC_ALGO` | `EAGLE` | shell | `sglang-MTP.sh` MTP algorithm |
| `SPEC_NUM_STEPS` | `5` | shell | Speculative draft depth |
| `SPEC_EAGLE_TOPK` | `4` | shell | Draft branching factor |
| `SPEC_NUM_DRAFT` | `8` | shell | Max draft tokens verified |

---

## 12. Interpreting H100 Performance Metrics

Here is what you should look for in the numbers to understand how the H100s are performing:

### Latency breakdown

```
Total latency ≈  TTFT  +  (output_tokens × TPOT)
```

- **TTFT (Time to First Token)** — driven by the **prefill workers** (GPUs 0 & 1). Long prompts = high TTFT. With 2P2D, prefill is isolated so TTFT is not penalised by decode backpressure.
- **TPOT (Time Per Output Token)** — driven by the **decode workers** (GPUs 2 & 3). This is where the H100's HBM3 bandwidth matters. GLM-4.5-Air's MoE means only 12B params are active per token, so TPOT should be significantly lower than a full 106B dense model.

### What good numbers look like on H100s

| Metric | Good | Concerning |
|--------|------|-----------|
| Avg latency | < 5s for 256-token output | > 15s — check GPU utilisation |
| P99 / P50 ratio | < 3× | > 5× — indicates queue backlog or OOM pressure |
| Token throughput | > 50 tok/s per decode GPU | < 20 tok/s — may be TPC-bound or batching too small |
| Prompt tokens / total | < 80% | > 90% — confirm completions aren't being truncated |

### Reading the Pareto plot

- **Tight cluster, low latency, moderate throughput** → the system is underutilised; increase `--concurrency`
- **Wide horizontal spread at the same throughput** → latency is variable; check P99 — could be routing or Mooncake transfer jitter
- **Top-right cluster with a few bottom-left outliers** → the system is mostly efficient; outliers could be long-prompt prefills

### GPU utilisation (external tools)

Run these alongside the benchmark:

```bash
# Per-GPU utilisation every second
watch -n 1 nvidia-smi

# More detail: SM utilisation, memory bandwidth, tensor core %
dcgmi dmon -e 1002,1003,1004,1005,1009

# Check which PIDs own each GPU
nvidia-smi pmon -s u
```

Expected pattern:
- GPUs 0,1 (prefill): **high SM utilisation** during prompt ingestion, then idle
- GPUs 2,3 (decode): **moderate SM utilisation**, **high memory bandwidth** continuously

---

## 13. Troubleshooting

### The proxy writes to `metrics.jsonl` but token counts are all 0

The proxy parses `usage` from the raw response JSON. If claude CLI is using streaming (`stream: true`), the final chunk may not contain `usage`. Check if the claude CLI sends `stream: false` by default, or explicitly add `"stream": false` to requests.

### Workers start but the router returns 502

The prefill/decode workers take 60–120 seconds to load the 106B model. The router may bind before the workers are ready. Run `python server/health_check.py` (which polls `:8000/health`) and wait for it to return success before running the benchmark.

### `mooncake_master: command not found`

The `mooncake-transfer-engine` pip package installs the Python module but may not install the `mooncake_master` binary. If missing:

```bash
# Use the Python module directly (equivalent)
uv run --extra sglang python -m mooncake.master_service \
    --enable_http_metadata_server=true --http_metadata_server_port=8080
```

### `claude: command not found`

The benchmark client calls `claude --print` as a subprocess. Install and authenticate the Claude CLI:

```bash
npm install -g @anthropic-ai/claude-code
claude auth login
```

### H100 Out-of-Memory

GLM-4.5-Air-FP8 in FP8 requires ~20–25 GB per GPU for weights + KV cache at `MAX_MODEL_LEN=32768`. If OOM:

```bash
MAX_MODEL_LEN=16384 bash server/sglang_start.sh serve
```

### RDMA errors / Mooncake transfer failures

If RDMA is unavailable, fall back to TCP:

```bash
MOONCAKE_PROTOCOL=tcp bash server/sglang_start.sh serve
```

Note: TCP transfer adds ~5–20% latency overhead for the KV handoff between prefill and decode. For correctness testing this is fine; for production benchmarking use RDMA.

### `HuggingFace ConnectionError` when loading the model

```bash
# Use ModelScope mirror
export SGLANG_USE_MODELSCOPE=true

# Or pre-download the model
huggingface-cli download zai-org/GLM-4.5-Air-FP8 --local-dir ./models/glm-4.5-air-fp8
MODEL_PATH=./models/glm-4.5-air-fp8 bash server/sglang_start.sh serve
```
