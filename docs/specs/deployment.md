# Deployment and Quantization Specification

> Cross-references: [Architecture](architecture.md) · [Schema Reference](schema-reference.md) · [CLI Reference](cli-reference.md)

## Deploy Pipeline

```
Trained checkpoint → Weight packing (ternary / low-bit) → Activation scaling (INT8) → Deployable artifact
```

The codebase treats quantization as a deploy-stage transformation rather than a separate model family. The pipeline is accessed via the `deploy` and `quantize` CLI subcommands.

## Quantization Methods

### Ternary Weight Packing

Given weight tensor `W`, the practical scaling approximates BitNet-style low-bit updates:

```
s = mean(|W|)
W̃ = clip(round(W / s), −1, 1)
```

The packed mapping uses **two bits per weight symbol** for storage efficiency. This approximates 1.58-bit storage (log₂(3) ≈ 1.58 bits per ternary symbol).

### INT8 Activation Quantization

For activations `x`:

```
q = round(x · 127 / (max(|x|) + ε)),   q ∈ [−128, 127]
```

Dequantization: `x ≈ q / α` where `α = 127 / (max(|x|) + ε)`.

## Size Estimates

For `N` parameters:

| Format | Approximate Size | Formula |
|---|---|---|
| FP32 | 4N bytes | `4 × N` |
| FP16 | 2N bytes | `2 × N` |
| 1.58-bit (ternary packed) | ~0.1975N bytes | `(1.58/8) × N` |

Before metadata and packing overhead. A 110M parameter model:
- FP32: ~440 MB
- FP16: ~220 MB
- 1.58-bit: ~22 MB

## Deployment Artifact Contents

A deployment artifact directory contains:

| File | Description |
|---|---|
| `config.json` | Model architecture configuration (JSON) |
| `model_quantized.pt` | Quantized model weights (PyTorch format) |
| `deployment_info.json` | Metadata: quantization params, original checkpoint hash, timestamp |

For `--format standard`, weights are stored in FP16 without ternary packing.

## Inference Modes

The `infer` subcommand supports three modes:

| Mode | Flag | Description |
|---|---|---|
| Single text | `--text "input"` | Run inference on one text string, print result |
| File batch | `--input file.txt --output results.json` | Process a file (one text per line), write JSON results |
| Benchmark | `--benchmark` | Measure throughput and latency over multiple runs |

### Inference Options

| Flag | Effect |
|---|---|
| `--fp16` | Load model in FP16 precision (reduces memory ~50%) |
| `--batch-size N` | Process N texts simultaneously in file mode |
| `--device` | Select compute device (auto/cpu/cuda/mps) |

## Quantization CLI

The `quantize` subcommand is a convenience wrapper that calls `deploy` with `--format quantized`:

```bash
# These are equivalent:
frankestein-transformer quantize --checkpoint model.pt --output ./out
frankestein-transformer deploy --checkpoint model.pt --output ./out --format quantized
```

## Validation

The `--validate` flag on `deploy`/`quantize` runs a forward pass with the quantized model and checks:
- Output tensor shapes match expectations
- No NaN/Inf in activations
- Numerical consistency within tolerance vs FP32 reference

## HuggingFace Transformers Export

The `transformers-export` subcommand (and `--transformers-export` flag on `train`/`deploy`/`quantize`) converts a checkpoint + YAML config into a HuggingFace Transformers-compatible folder with:
- `config.json` (HF format)
- `pytorch_model.bin` or `model.safetensors`
- Tokenizer files (if available)

Compatibility is validated before export — only standard-compatible architectures (no custom mixers) can be exported.
