# SemiAnalysis Task — GLM-4.5-Air-FP8 Disaggregated Serving Benchmark

> **Goal:** Serve [GLM-4.5-Air-FP8](https://huggingface.co/zai-org/GLM-4.5-Air-FP8) (106 B total / 12 B active MoE) via **SGLang** with **2P2D disaggregated prefill-decode** architecture using the **Mooncake Transfer Engine** for high-throughput KV-cache transfer, then benchmark it on SWE-bench tasks through a metrics-logging proxy.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Model Details](#model-details)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Serving — 2P2D Disaggregated Setup](#serving--2p2d-disaggregated-setup)
- [Metrics Proxy](#metrics-proxy)
- [Running the Benchmark](#running-the-benchmark)
- [Metrics & Reporting](#metrics--reporting)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
                        ┌──────────────────────────────────────────────────┐
                        │            SGLang Router (:8000)                 │
                        │       (sglang.srt.disaggregation.mini_lb)       │
                        └───────────┬──────────────┬───────────────────────┘
                                    │              │
                     ┌──────────────▼──┐  ┌────────▼──────────────┐
                     │   Prefill Pool  │  │    Decode Pool        │
                     │                 │  │                       │
                     │  P-1 (GPU 0)    │  │  D-1 (GPU 2)         │
                     │  :30000         │  │  :30001               │
                     │                 │  │                       │
                     │  P-2 (GPU 1)    │  │  D-2 (GPU 3)         │
                     │  :30002         │  │  :30003               │
                     └────────┬────────┘  └────────┬──────────────┘
                              │   KV-cache xfer    │
                              └────────┬───────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │   Mooncake Transfer Engine   │
                        │   (RDMA / TCP KV transfer)   │
                        │   Master :50051              │
                        └──────────────────────────────┘

          ┌──────────────┐          ┌────────────────────┐
          │ Metrics Proxy │ ◄──────►│   Benchmark Client │
          │    (:8001)    │          │   (run_benchmark)  │
          └──────┬───────┘          └────────────────────┘
                 │
          metrics.jsonl
```

**2P2D** = 2 Prefill workers + 2 Decode workers — an SGLang disaggregated serving configuration where:

| Phase | Role | Why separate? |
|-------|------|---------------|
| **Prefill** | Process the input prompt, populate KV-cache | Compute-bound (FLOPS-heavy) |
| **Decode** | Generate output tokens autoregressively | Memory-bound (KV-cache reads) |

The **Mooncake Transfer Engine** shuttles KV-cache tensors from prefill workers to decode workers via RDMA (or TCP fallback), enabling zero-copy, low-latency handoff.

---

## Model Details

| Property | Value |
|----------|-------|
| **HuggingFace ID** | [`zai-org/GLM-4.5-Air-FP8`](https://huggingface.co/zai-org/GLM-4.5-Air-FP8) |
| **Architecture** | Mixture-of-Experts (MoE) |
| **Total Parameters** | 106 B |
| **Active Parameters** | 12 B |
| **Precision** | FP8 (8-bit floating point) |
| **Developer** | Zhipu AI (Z.ai / THUDM) |
| **License** | MIT |
| **Modes** | Thinking (deep reasoning) / Non-thinking (instant response) |

---

## Prerequisites

| Requirement | Minimum |
|-------------|---------|
| GPUs | 4× NVIDIA H100 80 GB (or comparable) |
| CUDA | 12.1+ |
| Python | 3.10+ |
| Network | RDMA NICs recommended (Mellanox ConnectX–6/7), TCP fallback available |
| OS | Linux (Ubuntu 22.04+) |
| RAM | 128 GB+ system memory |

---

## Project Structure

```
semianalysis-task/
├── .env.example              # Environment variable template
├── pyproject.toml             # Project metadata & dependencies
├── run_benchmark.py           # CLI entrypoint for the benchmark harness
│
├── server/                    # Inference server launch scripts
│   ├── sglang_start.sh        # ★ 2P2D disaggregated SGLang + Mooncake
│   ├── vllm_start.sh          # Alternative: vLLM expert-parallel launch
│   └── health_check.py        # Poll /health until server is ready
│
├── proxy/                     # Reverse-proxy with metrics logging
│   └── proxy.py               # FastAPI proxy :8001 → :8000, writes metrics.jsonl
│
├── client/                    # Benchmark client logic
│   ├── shell_executor.py      # Run `claude --print` as subprocess
│   ├── turn_manager.py        # Multi-turn loop: prompt → output → re-prompt
│   └── task_queue.py          # Async task queue with semaphore concurrency
│
├── datasets/                  # Dataset loaders
│   └── load_swebench.py       # Load SWE-bench Verified → TaskItem list
│
└── metrics/                   # Metrics collection & reporting
    ├── collector.py            # Parse metrics.jsonl → AggregatedMetrics
    └── report.py               # Markdown summary + Pareto plot
```

---

## Installation

### Quick Start (on the GPU server)

```bash
# 1. Clone the repo
git clone https://github.com/Prathmesh234/MultiTurnKernel-SFT.git
cd MultiTurnKernel-SFT

# 2. Create a venv (recommended)
python3.10 -m venv .venv && source .venv/bin/activate

# 3. Copy and edit your env config
cp .env.example .env
# → Edit .env to set MOONCAKE_PROTOCOL=rdma if RDMA NICs are available

# 4. Install everything (SGLang from source + Mooncake + Python deps)
bash server/sglang_start.sh install
```

### Manual Installation

If you prefer to install components individually:

```bash
# SGLang (from source, latest)
git clone https://github.com/sgl-project/sglang.git /tmp/sglang
pip install -e "/tmp/sglang/python[all]" \
    --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python

# Mooncake Transfer Engine
pip install mooncake-transfer-engine
# Or from source:
# git clone https://github.com/kvcache-ai/Mooncake.git --recursive
# cd Mooncake && bash dependencies.sh
# mkdir build && cd build && cmake .. && make -j && sudo make install

# Project Python dependencies
pip install fastapi uvicorn httpx datasets matplotlib python-dotenv
```

---

## Serving — 2P2D Disaggregated Setup

### Launch All Services

```bash
# Install + serve in one command
bash server/sglang_start.sh all

# Or just serve (if already installed)
bash server/sglang_start.sh serve
```

This starts **5 processes**:

| # | Service | Port | GPU | Description |
|---|---------|------|-----|-------------|
| 1 | Mooncake Master | 50051 | — | KV-cache pool orchestrator |
| 2 | Prefill Worker 1 | 30000 | GPU 0 | Prompt processing |
| 3 | Prefill Worker 2 | 30002 | GPU 1 | Prompt processing |
| 4 | Decode Worker 1 | 30001 | GPU 2 | Token generation |
| 5 | Decode Worker 2 | 30003 | GPU 3 | Token generation |
| 6 | Router | 8000 | — | Load-balance across P & D |

### Verify

```bash
# Health check
python server/health_check.py

# Quick test
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello GLM-4.5-Air!","sampling_params":{"temperature":0}}'

# OpenAI-compatible endpoint
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"zai-org/GLM-4.5-Air-FP8","messages":[{"role":"user","content":"Explain MoE."}]}'
```

### Configuration Overrides

All settings can be overridden via environment variables:

```bash
# Example: Use RDMA, custom ports, larger context
MODEL_PATH=zai-org/GLM-4.5-Air-FP8 \
MOONCAKE_PROTOCOL=rdma \
MOONCAKE_DEVICE="mlx5_0,mlx5_1" \
MAX_MODEL_LEN=65536 \
ROUTER_PORT=9000 \
bash server/sglang_start.sh serve
```

---

## Metrics Proxy

The **FastAPI reverse proxy** sits between the benchmark client and the SGLang router, transparently logging every request/response to `metrics.jsonl`.

```bash
# Start the proxy on port 8001 (forwards to :8000)
uvicorn proxy.proxy:app --host 0.0.0.0 --port 8001
```

Each logged JSONL record contains:

| Field | Type | Description |
|-------|------|-------------|
| `ts` | float | Unix timestamp |
| `path` | string | API endpoint path |
| `latency_s` | float | End-to-end latency in seconds |
| `req_bytes` | int | Request body size |
| `resp_bytes` | int | Response body size |
| `usage.prompt_tokens` | int | Prompt tokens (from OpenAI-format response) |
| `usage.completion_tokens` | int | Completion tokens |

---

## Running the Benchmark

```bash
# Run the full benchmark
python run_benchmark.py --domain swebench --concurrency 4 --n-tasks 50

# Or use the script alias
python -m run_benchmark --n-tasks 10
```

**What happens:**

1. **`datasets/load_swebench.py`** loads tasks from `princeton-nlp/SWE-bench_Verified`
2. **`client/task_queue.py`** feeds tasks through an `asyncio.Queue` with bounded concurrency
3. **`client/turn_manager.py`** runs each task through a multi-turn `claude --print` loop (up to 5 turns)
4. **`proxy/proxy.py`** logs every LLM request to `metrics.jsonl`
5. **`metrics/collector.py`** aggregates latency, throughput, and token counts
6. **`metrics/report.py`** prints a markdown summary table and generates a Pareto plot

---

## Metrics & Reporting

### Collected Metrics

| Metric | Description |
|--------|-------------|
| `total_requests` | Total number of LLM API calls |
| `avg_latency_s` | Mean end-to-end latency |
| `p50_latency_s` | Median latency |
| `p99_latency_s` | 99th percentile latency |
| `total_prompt_tokens` | Sum of all prompt tokens |
| `total_completion_tokens` | Sum of all completion tokens |

### Generate Reports

```python
from metrics.collector import collect
from metrics.report import markdown_summary, pareto_plot

# Aggregate metrics from the JSONL log
metrics = collect()
print(markdown_summary(metrics))

# Generate a Pareto frontier plot
pareto_plot("pareto.png")
```

Example output:

```
| Metric | Value |
|--------|-------|
| Total requests | 200 |
| Avg latency (s) | 2.341 |
| P50 latency (s) | 1.890 |
| P99 latency (s) | 8.120 |
| Prompt tokens | 145200 |
| Completion tokens | 52340 |
```

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `zai-org/GLM-4.5-Air-FP8` | HuggingFace model ID |
| `HOST_IP` | `0.0.0.0` | Bind address for all services |
| `MAX_MODEL_LEN` | `32768` | Maximum sequence length |
| `TP_SIZE` | `1` | Tensor parallelism per worker |
| `PREFILL_PORT_1` | `30000` | Port for prefill worker 1 |
| `PREFILL_PORT_2` | `30002` | Port for prefill worker 2 |
| `DECODE_PORT_1` | `30001` | Port for decode worker 1 |
| `DECODE_PORT_2` | `30003` | Port for decode worker 2 |
| `ROUTER_PORT` | `8000` | Port for the SGLang router |
| `MOONCAKE_PROTOCOL` | `tcp` | Transfer protocol (`rdma` or `tcp`) |
| `MOONCAKE_DEVICE` | *(auto)* | RDMA device names (e.g., `mlx5_0,mlx5_1`) |
| `MOONCAKE_GLOBAL_SEG` | `4gb` | Memory contributed to KV-cache pool |
| `MOONCAKE_MASTER_PORT` | `50051` | Mooncake master service port |

### pyproject.toml Extras

```bash
# Install with SGLang extra
uv sync --extra sglang

# Install with vLLM extra (conflicts with SGLang)
uv sync --extra vllm
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `mooncake_master: command not found` | Run `pip install mooncake-transfer-engine` or build from source |
| `CUDA out of memory` | Reduce `MAX_MODEL_LEN` or use fewer workers; the FP8 model needs ~4×20 GB |
| `Connection refused on :30000` | Workers need 30–120 seconds to load the 106 B model; check logs |
| `RDMA device not found` | Set `MOONCAKE_PROTOCOL=tcp` for development, or install Mellanox OFED drivers |
| `Multiple prefill on same node fails` | Known SGLang limitation — bootstrap port conflict. Use different nodes for each prefill worker in production |
| `HuggingFace timeout` | Run `export SGLANG_USE_MODELSCOPE=true` or pre-download the model with `huggingface-cli download zai-org/GLM-4.5-Air-FP8` |
| Workers crash silently | Check individual worker logs; set `SGLANG_LOG_LEVEL=debug` for verbose output |

### Verifying the Stack

```bash
# 1. Check all processes are running
ps aux | grep sglang

# 2. Test prefill worker directly
curl http://127.0.0.1:30000/health

# 3. Test decode worker directly
curl http://127.0.0.1:30001/health

# 4. Test through router
curl http://127.0.0.1:8000/health

# 5. Run health check script
python server/health_check.py
```

---

## References

- [SGLang Documentation](https://docs.sglang.ai/)
- [SGLang Disaggregated Serving with Mooncake](https://kvcache-ai.github.io/Mooncake/getting_started/examples/sglang-integration-v1.html)
- [SGLang HiCache with Mooncake Backend](https://kvcache-ai.github.io/Mooncake/getting_started/examples/sglang-integration/hicache-integration-v1.html)
- [Mooncake Transfer Engine (GitHub)](https://github.com/kvcache-ai/Mooncake)
- [GLM-4.5-Air-FP8 on HuggingFace](https://huggingface.co/zai-org/GLM-4.5-Air-FP8)
- [SWE-bench Verified Dataset](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)
