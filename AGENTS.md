# AGENTS.md

## Environment & Commands

- **Conda environment**: `frankestein` (Python 3.9). Prefix all commands with `conda run -n frankestein`.
- **Install**: `conda run -n frankestein pip install -e ".[train]"`
- **Run tests**: `conda run -n frankestein python -m pytest tests/ --continue-on-collection-errors -v --tb=short -p no:warnings`
- **CLI**: `conda run -n frankestein frankestein-transformer <subcommand> ...`
- **Web server**: `conda run -n frankestein frankestein-transformer web-server` (Streamlit app)

## Architecture

- **Entrypoint**: `src/cli.py` → `frankestein-transformer` (registered in `pyproject.toml`)
- **Package root**: `src/` — contains `model/`, `training/`, `deploy/`, `sbert/`, `tokenizer/`, `streamlit_gui/`, `utils/`
- **Schema**: `configs/schema.yaml` is the single source of truth for all training config. Strict `additionalProperties: false`.
- **Training presets**: `configs/*.yaml` (named: mini, frankenstein, frankesteindecoder, standard, etc.) and `configs/examples/*.yaml` (optimizer + architecture combos).

## Where to Edit by Feature Type

Based on commit history patterns, here is where to find and modify code for each kind of change. Before implementing, read recent commits touching those files (`git log --oneline -20 -- <path>`).

| Feature / Fix | Primary files to read & edit |
|---|---|
| Add a new **attention mixer** (layer_pattern block) | `src/model/attention/<family>/new_attn.py`, `src/model/attention/__init__.py`, `src/model/tormented_bert_frankestein.py`, `src/schema.yaml`, `configs/examples/` (add example), `tests/test_schema_attention_layers.py`, `tests/test_attention_refactor.py` |
| Add a new **optimizer** | `src/model/optimizer/<new_opt>.py`, `src/model/optimizer/factory.py`, `src/model/optimizer/__init__.py`, `src/schema.yaml` (enum + by_optimizer entry), `configs/examples/` (add example), `tests/test_optimizer_factory.py`, `tests/test_yaml_examples.py` |
| Modify **training config / schema** | `configs/schema.yaml` (source of truth), `src/training/config_loader.py`, `configs/README.md`, `src/streamlit_gui/app.py` (web interface), `tests/test_config_loader.py`, `tests/test_yaml_examples.py` |
| Modify **model architecture / UltraConfig** | `src/model/tormented_bert_frankestein.py`, `configs/schema.yaml`, `tests/test_model_config.py`, `tests/test_model_variants.py` |
| Modify **training loop / trainer** | `src/training/trainer.py`, `src/training/main.py`, `tests/test_mlm_masking.py`, `tests/test_model_training_fake_data.py` |
| Modify **SBERT training/inference** | `src/sbert/train_sbert.py`, `src/sbert/inference_sbert.py`, `configs/schema.yaml` (sbert block), `configs/modernbert_sbert*.yaml`, `tests/test_sbert_thermal_offload.py` |
| Modify **deployment / quantization** | `src/deploy/deploy.py`, `src/deploy/inference.py`, `src/model/tormented_bert_frankestein.py` (BitNet path) |
| Modify **CLI / entrypoint** | `src/cli.py`, `pyproject.toml` (scripts entry), `tests/test_cli_parser.py`, `tests/test_cli_gpu_temp_flags.py` |
| Add/modify **GPU thermal guard** | `src/utils/gpu_temp_guard.py`, `src/training/trainer.py`, `src/cli.py`, `tests/test_gpu_temp_guard.py`, `tests/test_cli_gpu_temp_flags.py` |
| Modify **positional encodings** | `src/model/attention/rope.py`, `src/model/attention/hope.py`, `src/model/attention/common.py`, `configs/schema.yaml`, `tests/test_positional_encodings.py` |
| Modify **streaming dataset** | `src/training/streaming_mlm_dataset.py`, `src/utils/storage_manager.py` |
| Modify **Streamlit web interface** | `src/streamlit_gui/app.py` |
| Update **paper / docs** | `docs/paper.tex`, `docs/paper-es.tex`, `docs/bibliography/*.bib`, `docs/bibliography/*.md` |

## Critical Gotchas

- **`norm_type`** only accepts `layer_norm`, `dynamic_tanh`, `derf`. `rms_norm` is NOT valid.
- **`fasa_attn` and `sparge_attn`** are eval-only blocks. Training with either raises a runtime error.
- **Optimizer parameters** use prefixed keys: `<optimizer_class>-<group>_<param>` (e.g. `adamw-lr_embeddings`, `sgd_momentum-momentum`).
- **`training.task`** is required (`mlm` or `sbert`). Legacy top-level optimizer keys are no longer accepted.
- **`model_class: frankesteindecoder`** forces `mode: decoder` at runtime.
- **`hidden_size`** must be divisible by **`num_heads`**. Agents should not change one without checking the other.
- **CUDA is not needed for tests** — CI uses CPU torch; GPU code paths are skipped with `TORCH_AVAILABLE` guards.
- **`configs/examples/`** YAML files are validated by `tests/test_yaml_examples.py` — adding a new optimizer family or mixer requires an example YAML or the test breaks.

## Documentation

- **Docstring style**: Google-style (Args, Returns, Raises, Attributes sections). pdoc renders them at `docs/pdoc/`.
- **Generate API docs**: `pdoc src/ -o docs/pdoc/ --docformat google` (requires `torch` + deps installed in the pdoc venv).
- **Specs**: `docs/specs/*.md` — architecture, attention-mixers, optimizers, schema-reference, cli-reference, deployment, sbert-workflows, training-safety.
- **Academic references**: `docs/paper.tex`, `docs/bibliography/*.md` contain full mathematical formulations and citations for all mixers/optimizers.
- When adding a new mixer/optimizer, also update the relevant `docs/specs/*.md` file.

## Code Style

- All Python modules use `from __future__ import annotations`.
- Imports use absolute paths from `src/` (e.g. `from src.training.config_loader import ...`).
- No project-level linter or formatter config was found — match existing style in files you edit.
