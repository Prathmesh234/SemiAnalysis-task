# 📊 SGLang Benchmark Metrics Report

## 1. Request Summary (Proxy)

| Metric | Value |
|--------|-------|
| Total requests | 10 |
| Avg latency | 6.792s |
| P50 latency | 6.717s |
| P90 latency | 9.256s |
| P95 latency | 9.256s |
| P99 latency | 9.256s |
| Min latency | 5.710s |
| Max latency | 9.256s |
| Prompt tokens | 49,899 |
| Completion tokens | 9,342 |
| Cached tokens | 0 |
| Avg tokens/request | 5924.1 |
| Total request bytes | 276,126 |
| Total response bytes | 221 |
| **Overall throughput** | **137.5 tok/s** |

## 2. Latency Breakdown (SGLang Prometheus)

| Metric | Avg | P50 | P95 | P99 |
|--------|-----|-----|-----|-----|
| **TTFT** (Time to First Token) | 184.9ms | 68.8ms | 1450.0ms | 1890.0ms |
| **ITL** (Inter-Token Latency) | 7.1ms | 7.0ms | 7.9ms | 8.0ms |
| **E2E** (End-to-End) | 6.247s | 6.714s | 8.900s | 9.780s |
| **Queue time** | 0.6ms | — | — | 99.0ms |

> **TPOT** (Time Per Output Token) ≈ **7.1ms**

## 3. Throughput (SGLang Prometheus)

| Metric | Value |
|--------|-------|
| Generation throughput | 139.5 tok/s |
| Total prompt tokens processed | 49,905 |
| Total generation tokens processed | 9,350 |
| Total requests (server-side) | 11 |
| Aborted requests | 0 |

## 4. GPU Hardware (nvidia-smi)

| GPU | Name | Util% | Mem Used | Mem Total | Mem BW% | Temp°C | Power W | SM MHz | Mem MHz |
|-----|------|-------|----------|-----------|---------|--------|---------|--------|---------|
| 0 | NVIDIA H100 80GB HBM3 | 0% | 69680 MiB | 81559 MiB | 0% | 41°C | 231/700 | 1980 | 2619 |

## 5. KV Cache & Prefix Cache

| Metric | Value |
|--------|-------|
| Cache hit rate | 0.0% |
| Tokens served from cache | 37,870 |
| Tokens evicted (GPU → CPU) | 0 |
| Tokens loaded back (CPU → GPU) | 0 |
| Eviction duration (avg) | 0.0ms |
| Load-back duration (avg) | 0.0ms |
| KV cache utilization | 1.6% |
| Tokens in KV cache | 5,776 |
| Max KV cache capacity | 365,979 |

## 6. Scheduler State

| Metric | Value |
|--------|-------|
| Running requests | 1 |
| Queued requests | 0 |
| Retracted (preempted) | 0 |
| Paused requests | 0 |
| GPU execution time (total) | 0.00s |

## 7. Speculative Decoding (MTP/EAGLE)

*Not active — no speculative decoding metrics detected.*

## 8. Disaggregated KV Transfer (2P2D / Mooncake)

| Metric | Value |
|--------|-------|
| KV transfer speed | 0.00 GB/s |
| KV transfer latency | 0.00ms |
| Prefill pre-alloc queue | 0 |

---
*Collected in 0.069s*