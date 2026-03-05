# Project TODOs

## Urgent Tasks

- [ ] **Replace `claude` CLI harness with an Open-Source Agent Harness**
  - **Reason**: The current `claude --print` setup strongly expects the reasoning capabilities and exact Anthropic JSON tool-calling format of Claude 3.5 Sonnet. When we plug local, open-source 8B models (like Qwen3-8B or GLM 4.5 Air) into it, they often fail to perfectly mimic the complex XML/JSON formats Claude uses, causing the CLI to error out on Turn 1. This prevents us from testing the multi-turn, long-context latency scaling (TTFT at high context sizes) which is the actual objective of our H100 hardware benchmark.
  - **Proposed Solution**: Remove the Anthropic reverse-proxy and use an agent harness that natively targets the OpenAI `/v1/chat/completions` API format with standard tool calling. This lets the models use the API natively and is much more fault-tolerant to open-source model syntax.
  - **Recommended Harnesses**:
    1. **[SWE-agent](https://github.com/princeton-nlp/SWE-agent) / [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent)**: The industry standard created by the SWE-bench authors. `mini-SWE-agent` is specifically designed for evaluating any local model via LiteLLM without requiring complex agent scaffoldings, making it an excellent benchmark driver for our GPU testing.
    2. **[Aider](https://aider.chat/)**: Extremely robust command-line AI coding assistant. It's famous for handling local models gracefully and has built-in benchmarking suites for SWE-bench. It relies more on system prompts than complex JSON schemas, making it very accessible for 8B/9B class models.
    3. **[OpenHands](https://github.com/All-Hands-AI/OpenHands)**: A larger web-based Devin alternative, which might be overkill but has strong official support for SWE-bench evaluations against any OpenAI-compatible endpoint.
