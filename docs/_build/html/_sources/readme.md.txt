# Frankestein Transformer

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://img.shields.io/badge/CI-passing-brightgreen.svg)](https://github.com/erickfmm/frankestein-transformer/actions)

Config-driven transformer experimentation toolkit with 17+ mixer architectures and 23 optimizer families.

## Quick Start

| Method | Command |
|--------|---------|
| **uv** (recommended) | `git clone https://github.com/erickfmm/frankestein-transformer.git && cd frankestein-transformer && uv venv && source .venv/bin/activate && uv pip install -e ".[train]"` |
| **pip** | `python -m venv .venv && source .venv/bin/activate && pip install -e ".[train]"` |
| **conda** | `conda create -n frankestein python=3.9 && conda activate frankestein && pip install -e ".[train]"` |

Verify: `frankestein-transformer --help`

## Feature Matrix

| Feature | Scale |
|---------|-------|
| Sequence mixer architectures | 19 across 4 categories (Dense, Recurrent, Sparse, Gated) |
| Optimizer families | 23 across 6 categories |
| Model classes | `frankenstein`, `mini`, `frankesteindecoder` |
| Training modes | Encoder (MLM) / Decoder (autoregressive) |
| Normalization types | `layer_norm`, `dynamic_tanh`, `derf` |
| CLI subcommands | 8 |
| Web configuration UI | Streamlit schema-driven YAML builder |
| Quantized deployment | BitNet + checkpoint export pipeline |
| SBERT workflows | Training + inference (similarity, search, cluster, encode) |

## Architecture Decision Table

| Model Class | Mode | Use Case |
|-------------|------|----------|
| `frankenstein` | Encoder | Full-featured MLM pre-training with mixed attention, MoE, and all 19 mixer types |
| `mini` | Encoder | Lightweight encoder for constrained resources; reduced hidden size and layers |
| `frankesteindecoder` | Decoder | Autoregressive causal decoder for LLM-style generation; forces `mode: decoder` |

See [configs/README.md](configs/README.md) for preset details and [docs/specs/](docs/specs/) for architecture deep-dives.

## CLI Command Reference

| Subcommand | Purpose | Example |
|------------|---------|---------|
| `train` | Run schema-validated training | `frankestein-transformer train --config-name mini --device auto` |
| `deploy` | Export checkpoint to deployment artifacts | `frankestein-transformer deploy --checkpoint ckpt.pt --output deployed/ --format quantized` |
| `quantize` | Shortcut for quantized deployment | `frankestein-transformer quantize --checkpoint ckpt.pt --output deployed_q/ --validate` |
| `infer` | Batch/interactive/benchmark inference | `frankestein-transformer infer --model deployed/ --text "hello" --device auto` |
| `sbert-train` | Train sentence embedding model | `frankestein-transformer sbert-train --output_dir ./sbert_out --batch_size 16 --epochs 4` |
| `sbert-infer` | SBERT similarity/search/cluster/encode | `frankestein-transformer sbert-infer --model_path ./sbert_out --mode similarity --sentence1 "a" --sentence2 "b"` |
| `transformers-export` | Export to HuggingFace Transformers format | `frankestein-transformer transformers-export --config-name mini --output ./hf_export/` |
| `web-server` | Launch Streamlit config builder UI | `frankestein-transformer web-server` |

All model-executing commands accept `--device auto|cpu|cuda|mps`.

## Mixer Categories

| Category | Code Names | Description |
|----------|------------|-------------|
| **Dense** | `standard_attn`, `sigmoid_attn`, `gated_softmax_attn`, `titan_attn` | Full quadratic attention variants with positional encoding support |
| **Recurrent** | `retnet`, `retnet_attn`, `mamba`, `ode` | Retention networks, state-space models, and continuous-depth ODE layers |
| **Sparse** | `sparse_transformer_attn`, `longformer_attn`, `bigbird_attn`, `sparsek_attn`, `nsa_attn`, `sparge_attn` ⚠️, `fasa_attn` ⚠️ | Factorized, sliding-window, and token-selection patterns |
| **Gated** | `gla_attn`, `deltanet_attn`, `gated_deltanet_attn`, `hgrn2_attn`, `fox_attn`, `engram_attn` | Linear attention with multiplicative gates, delta rules, and n-gram memory |

⚠️ `sparge_attn` and `fasa_attn` are **eval-only** — training raises a runtime error.

Configure via `layer_pattern` in YAML. See [configs/schema.yaml](src/schema.yaml) for the full mixer reference table.

## Optimizer Categories

| Category | Optimizers | Count |
|----------|------------|-------|
| **Classical** | `sgd_momentum`, `adamw`, `radam`, `adan`, `adopt`, `ademamix`, `lamb` | 7 |
| **Variance Reduction** | `mars_adamw`, `cautious_adamw` | 2 |
| **Memory-Efficient** | `adafactor`, `galore_adamw`, `lion`, `apollo`, `apollo_mini`, `q_apollo` | 6 |
| **Schedule-Free** | `schedulefree_adamw`, `prodigy` | 2 |
| **Second-Order** | `sophia`, `shampoo`, `soap` | 3 |
| **Geometry-Oriented** | `muon`, `turbo_muon`, `anon` | 3 |

Parameters use prefixed keys: `<optimizer_class>-<group>_<param>` (e.g. `adamw-lr_embeddings`, `muon-ns_steps`). See [configs/README.md](configs/README.md) for the full parameter reference.

## Documentation Map

| Resource | Content |
|----------|---------|
| [configs/README.md](configs/README.md) | Schema walkthrough, preset details, optimizer parameter reference |
| [configs/schema.yaml](src/schema.yaml) | Authoritative training config schema (source of truth) |
| [docs/README.md](docs/README.md) | CLI reference and workflow guide |
| [docs/paper.pdf](docs/paper.pdf) | Technical report (English) |
| [docs/paper-es.pdf](docs/paper-es.pdf) | Technical report (Spanish) |
| [docs/specs/](docs/specs/) | Architecture and feature specifications |
| [docs/pdoc/](docs/pdoc/) | API documentation |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Development roadmap |
| [docs/transformers_compatibility.md](docs/transformers_compatibility.md) | HuggingFace export compatibility guide |

## Installation

### uv (recommended)

```bash
git clone https://github.com/erickfmm/frankestein-transformer.git
cd frankestein-transformer
uv venv
source .venv/bin/activate
uv pip install -e ".[train]"
```

### pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[train]"
```

### conda

```bash
conda create -n frankestein python=3.9
conda activate frankestein
pip install -e ".[train]"
```

Verify installation:

```bash
frankestein-transformer --help
```

## Quick Training Example

Minimal YAML config (`my_config.yaml`):

```yaml
model_class: mini
model:
  vocab_size: 30522
  hidden_size: 256
  num_layers: 4
  num_loops: 1
  num_heads: 8
  retention_heads: 8
  num_experts: 1
  top_k_experts: 1
  dropout: 0.1
  layer_pattern: [standard_attn, standard_attn, standard_attn, standard_attn]
  ode_solver: rk4
  ode_steps: 4
  use_bitnet: false
  norm_type: layer_norm
  use_factorized_embedding: false
  factorized_embedding_dim: 128
  use_embedding_conv: false
  embedding_conv_kernel: 3
  use_hope: false
  use_moe: false
  ffn_hidden_size: 1024
  ffn_activation: gelu
training:
  task: mlm
  batch_size: 8
  dataloader_workers: 2
  max_length: 128
  mlm_probability: 0.15
  max_samples: 100000
  dataset_batch_size: 10000
  num_workers: 4
  cache_dir: "./temp_data/cache"
  use_amp: false
  gradient_accumulation_steps: 1
  optimizer:
    optimizer_class: adamw
    parameters:
      adamw-lr_embeddings: 1e-4
      adamw-lr_norms: 1e-4
      adamw-lr_ode: 1e-4
      adamw-lr_retnet: 1e-4
      adamw-lr_mamba: 1e-4
      adamw-lr_attention: 1e-4
      adamw-lr_other: 1e-4
      adamw-wd_embeddings: 0.01
      adamw-wd_norms: 0.01
      adamw-wd_ode: 0.01
      adamw-wd_retnet: 0.01
      adamw-wd_mamba: 0.01
      adamw-wd_attention: 0.01
      adamw-wd_other: 0.01
      adamw-betas_embeddings: [0.9, 0.95]
      adamw-betas_norms: [0.9, 0.95]
      adamw-betas_ode: [0.9, 0.95]
      adamw-betas_retnet: [0.9, 0.95]
      adamw-betas_mamba: [0.9, 0.95]
      adamw-betas_attention: [0.9, 0.95]
      adamw-betas_other: [0.9, 0.95]
      adamw-eps_embeddings: 1e-8
      adamw-eps_norms: 1e-8
      adamw-eps_ode: 1e-8
      adamw-eps_retnet: 1e-8
      adamw-eps_mamba: 1e-8
      adamw-eps_attention: 1e-8
      adamw-eps_other: 1e-8
  scheduler_total_steps: 1000
  scheduler_warmup_ratio: 0.1
  scheduler_type: cosine
  grad_clip_max_norm: 5.0
  inf_post_clip_threshold: 100.0
  max_nan_retries: 3
  checkpoint_every_n_steps: 500
  max_rolling_checkpoints: 3
  num_best_checkpoints: 2
  nan_check_interval: 10
  log_gradient_stats: true
  gradient_log_interval: 10
  csv_log_path: "training_metrics.csv"
  csv_rotate_on_schema_change: true
  gpu_metrics_backend: nvml
  nvml_device_index: 0
  enable_block_grad_norms: true
  telemetry_log_interval: 1
  gpu_temp_guard_enabled: false
  gpu_temp_pause_threshold_c: 90.0
  gpu_temp_resume_threshold_c: 80.0
  gpu_temp_critical_threshold_c: null
  gpu_temp_poll_interval_seconds: 30.0
  use_galore: false
  galore_rank: 64
  galore_update_interval: 1
  galore_scale: 1.0
  galore_max_dim: 4096
```

Run:

```bash
frankestein-transformer train --config my_config.yaml --device auto
```

List available named presets:

```bash
frankestein-transformer train --list-configs
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) for full text.
