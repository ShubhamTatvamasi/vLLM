# vLLM Flags Cheat Sheet
*Practical reference for `vllm serve` — checked against the official vLLM CLI docs on 12 Oct 2026.*

> **Version note:** vLLM flags evolve quickly and some are model-, backend-, GPU-, or version-specific. Always verify on the installed version with `vllm serve --help=all` or `vllm serve --help=<flag-name>`. This is a practical shortlist, not an exhaustive dump of every flag.

## 1. Start, inspect, and configure

| Flag / command | What it does |
|---|---|
| `vllm serve MODEL` | Start the OpenAI-compatible API server |
| `--help` / `--help=all` | Show help / all options |
| `--help=ModelConfig` | Show a configuration group |
| `--help=max-num-seqs` | Help for a specific option |
| `--config config.yaml` | Load options from YAML |
| `--host 0.0.0.0` | Bind server to all interfaces; secure access at the network layer |
| `--port 8000` | HTTP server port |
| `--served-model-name my-model` | Public API model name |
| `--api-key "$VLLM_API_KEY"` | Require a key on supported API path prefixes; **not a complete security boundary** |

CLI arguments override config-file values; config values override defaults.

## 2. Model and tokenizer

| Flag | Typical use / meaning |
|---|---|
| `--model MODEL_ID_OR_PATH` | Hugging Face model ID or local model directory (often positional in `vllm serve MODEL`) |
| `--tokenizer TOKENIZER_ID_OR_PATH` | Override tokenizer source |
| `--revision REV` | Select model revision, where supported |
| `--dtype auto` | Select model weight dtype automatically; other supported values depend on build/model |
| `--quantization METHOD` | Select a supported weight quantization method, e.g. AWQ/GPTQ/FP8 when compatible |
| `--load-format auto` | Choose supported checkpoint loading format automatically |
| `--trust-remote-code` | Allow custom model code from the model repository; use only with trusted code |
| `--generation-config vllm` | Use vLLM generation defaults rather than loading the model's generation config |
| `--override-generation-config '{"temperature":0.7}'` | Override generation-config values when supported |
| `--max-model-len 8192` | Set maximum sequence/context length; may increase KV-cache demand |

**Tip:** Do not assume a quantization method is compatible with every architecture or GPU. Check the model card and vLLM support matrix.

## 3. GPU memory and KV cache

| Flag | What it controls |
|---|---|
| `--gpu-memory-utilization 0.90` | Fraction of GPU memory vLLM targets for its use; not a hard guarantee of total process memory |
| `--kv-cache-dtype auto` | KV-cache storage dtype; supported choices vary by platform/model |
| `--max-num-batched-tokens 8192` | Token budget for a scheduler iteration; higher can improve throughput but may affect latency/memory |
| `--max-num-seqs 128` | Maximum sequences processed in a batch/iteration; tune to workload and memory |
| `--block-size N` | KV-cache block size; available values depend on backend |
| `--enable-prefix-caching` | Reuse KV cache for matching prompt prefixes when supported |
| `--disable-sliding-window` | Disable sliding-window behavior for compatible models; can raise memory needs |
| `--enforce-eager` | Force eager execution; disables CUDA graphs and `torch.compile` in current documented behavior, often useful for debugging but can reduce performance |

**Memory intuition:** weights + KV cache + activations/workspaces + CUDA/runtime overhead must fit. Longer context and more concurrent sequences generally increase KV-cache requirements.

## 4. Scheduling and throughput

| Flag | What it controls |
|---|---|
| `--max-num-seqs N` | Concurrency per scheduler iteration |
| `--max-num-batched-tokens N` | Total scheduled tokens per iteration |
| `--scheduling-policy fcfs` | Request scheduling policy; verify supported values in your version |
| `--enable-chunked-prefill` | Split long prompt prefill into chunks where supported; can improve mixed prefill/decode scheduling |
| `--max-num-partial-prefills N` | Limit concurrent partial prefills, when supported |
| `--max-long-partial-prefills N` | Limit concurrent long partial prefills, when supported |
| `--disable-log-stats` | Disable periodic engine statistics logging |

**Tuning rule:** Change one variable at a time and measure TTFT, inter-token latency, throughput, queue time, GPU memory, and error rate. More concurrency is not always lower latency.

## 5. Multi-GPU and distributed serving

| Flag | Meaning |
|---|---|
| `--tensor-parallel-size N` | Split model tensors across N GPUs |
| `--pipeline-parallel-size N` | Split model layers into pipeline stages |
| `--data-parallel-size N` | Run data-parallel engine replicas, where supported |
| `--distributed-executor-backend mp` | Select distributed execution backend; supported choices depend on version/platform |
| `--enable-expert-parallel` | Enable expert parallelism for supported MoE models |
| `--worker-use-ray` | Legacy/version-specific option; check current help before use |

**Rule of thumb:** Tensor parallelism is commonly used when one model replica needs multiple GPUs. Data parallelism increases replica capacity. Pipeline parallelism can help with very large models but introduces pipeline trade-offs. Confirm topology, interconnect, and backend requirements.

## 6. Quantization and speculative decoding

| Flag | Meaning |
|---|---|
| `--quantization METHOD` | Load a compatible quantized model/weight format |
| `--kv-cache-dtype DTYPE` | Reduce KV-cache footprint if a supported lower-precision cache is appropriate |
| `--speculative-config JSON` | Configure speculative decoding on versions/models that support it |
| `--num-speculative-tokens N` | Draft-token count in versions that expose this option directly; otherwise set via speculative config |

**Caution:** Speculative decoding requires a compatible configuration and may help some workloads while hurting others. Benchmark quality, acceptance rate, latency, and throughput.

## 7. API, tools, and operational flags

| Flag | Meaning |
|---|---|
| `--enable-auto-tool-choice` | Enable automatic tool choice for supported models |
| `--tool-call-parser PARSER` | Choose parser compatible with the model's tool-call format |
| `--enable-log-requests` | Log request information; prompt details may be logged at DEBUG |
| `--enable-log-outputs` | Log generated outputs; requires request logging and may expose sensitive data |
| `--enable-request-id-headers` | Version-specific request-ID header behavior; check installed help |
| `--disable-frontend-multiprocessing` | Version-specific frontend behavior; verify before using |
| `--uvicorn-log-level info` | Set HTTP server log verbosity where supported |
| `--ssl-keyfile PATH` / `--ssl-certfile PATH` | Configure TLS when supported; often TLS is terminated at an ingress/proxy |
| `--allowed-origins ...` | Configure CORS origins; do not treat CORS as authentication |

**Security:** Keep the server behind an authenticated gateway or private network, protect metrics/admin endpoints, avoid logging prompts and outputs unless necessary, and use trusted model code only.

## 8. Useful launch recipes

### A. Simple single-GPU server
```bash
vllm serve Qwen/Qwen3-8B \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```

### B. Higher-throughput starting point
```bash
vllm serve MODEL_ID \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching
```
Tune the sequence/token budgets to the model, GPU memory, prompt lengths, and latency target.

### C. Model split across 4 GPUs
```bash
vllm serve MODEL_ID \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90
```
Ensure the container sees all four GPUs and the distributed backend/interconnect are configured correctly.

### D. Local checkpoint with explicit API model name
```bash
vllm serve /models/my-model \
  --served-model-name my-model \
  --host 0.0.0.0 \
  --port 8000
```

### E. YAML config
```yaml
# config.yaml — verify each key against your installed vLLM version
model: Qwen/Qwen3-8B
host: 0.0.0.0
port: 8000
gpu-memory-utilization: 0.90
max-model-len: 8192
max-num-seqs: 64
```
```bash
vllm serve --config config.yaml
```

## 9. Quick troubleshooting map

| Symptom | First things to inspect |
|---|---|
| CUDA OOM during startup | Model dtype/quantization, `--gpu-memory-utilization`, `--max-model-len`, GPU visibility, parallelism |
| OOM under load | Concurrency, KV-cache dtype, context lengths, batch token budget, workload mix |
| High TTFT | Queue time, long prompts/prefill, batching, chunked prefill support, CPU/tokenizer bottlenecks |
| Slow output tokens | GPU saturation, model size/quantization, batch contention, interconnect, kernel compatibility |
| Low GPU utilization | Request concurrency, CPU preprocessing, tokenizer/rendering, batch size, workload rate |
| Prefix-heavy prompts are slow | Check prefix caching compatibility and whether prompts actually share identical prefixes |
| Multi-GPU startup failure | Visible GPU count, TP/PP sizes, NCCL/network configuration, container runtime, version support |
| Unexpected model behavior | Chat template, tokenizer/model revisions, generation config, model-specific flags |

## 10. Handy diagnostics

```bash
# List installed version
vllm --version

# Full CLI reference for this installed build
vllm serve --help=all

# Search help for memory-related options
vllm serve --help=memory

# Confirm GPUs visible to the container
nvidia-smi

# Query OpenAI-compatible models endpoint
curl http://localhost:8000/v1/models

# Basic chat completion smoke test
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Explain KV cache in one sentence."}],
    "max_tokens": 64
  }'
```

## Official references

- CLI guide: https://docs.vllm.ai/en/latest/cli/
- `vllm serve` reference: https://docs.vllm.ai/en/latest/cli/serve/
- Engine arguments: https://docs.vllm.ai/en/stable/configuration/engine_args/
- Server arguments: https://docs.vllm.ai/en/latest/configuration/serve_args/

*Flags shown are common examples. Some flags change names, defaults, or availability between vLLM releases; treat the installed `--help` output as authoritative.*
