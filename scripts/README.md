# Script runners for notebooks

This folder provides CLI scripts to run the notebook pipelines without opening Jupyter.

- `run_pipeline_01_to_06.py`: runs notebooks **01 → 06** (preprocessing to downstream evaluation), excluding `06c`.
- `run_timevae_07.py`: runs notebook **07** (TimeVAE train + generate).

## Requirements

Run from repository root (`/home/runner/work/synthetic-data-generation/synthetic-data-generation`):

```bash
python scripts/run_pipeline_01_to_06.py --help
python scripts/run_timevae_07.py --help
```

Use the same Python environment required by your notebooks (PyTorch, NumPy, pandas, sklearn, etc.).

## 01 → 06 pipeline script

### Basic run

```bash
python scripts/run_pipeline_01_to_06.py \
  --project-root /home/runner/work/synthetic-data-generation/synthetic-data-generation
```

### Run only selected steps

```bash
python scripts/run_pipeline_01_to_06.py --steps 1,2,3
```

### Main parameters

- `--project-root`: project root injected into notebook configs.
- `--steps`: comma-separated subset of `1,2,3,4,5,6`.
- `--selected-model-family`: notebook 06 `SELECTED_MODEL_FAMILY` (`kovae` or `timevae`).
- `--selected-synthetic-method`: notebook 06 synthetic method (`rollout_v1`, `posterior_bank_v2`, `prior_v1`, etc. depending on family).
- `--run-real-to-real-only`: notebook 06 `RUN_REAL_TO_REAL_ONLY=True`.
- `--disable-use-real-models`: notebook 06 `USE_REAL_MODELS=False`.
- `--modalities`: comma-separated modalities for notebook 06 (`acc,bvp,eda,temp,fused`).
- `--summary-path`: optional JSON output summary path.

### Notebook config overrides (JSON)

Each notebook remains configurable via override arguments. You can pass either:
- JSON string inline, or
- path to a JSON file.

Available override flags:

- `--preprocessing-overrides` → notebook 01 `CONFIG`
- `--train-overrides` → notebook 02 `CONFIG`
- `--generation-overrides` → notebook 03 `GEN_CONFIG`
- `--comparison-overrides` → notebook 04 `COMPARISON_CONFIG`
- `--realism-overrides` → notebook 05 `EVAL_CONFIG`
- `--downstream-overrides` → notebook 06 top-level assignment overrides (example: `EPOCHS_CNNBILSTM`, `BATCH_SIZE`)

Example:

```bash
python scripts/run_pipeline_01_to_06.py \
  --steps 1,2,3,6 \
  --selected-model-family kovae \
  --selected-synthetic-method rollout_v1 \
  --train-overrides '{"num_epochs": 20, "batch_size": 128}' \
  --downstream-overrides '{"EPOCHS_CNNBILSTM": 20, "BATCH_SIZE": 128}'
```

## TimeVAE script (07)

### Basic run

```bash
python scripts/run_timevae_07.py \
  --project-root /home/runner/work/synthetic-data-generation/synthetic-data-generation
```

### TimeVAE overrides

- `--timevae-overrides`: JSON string or JSON file path applied to `TIMEVAE_CONFIG`.
- `--summary-path`: optional path to write output JSON.

Example:

```bash
python scripts/run_timevae_07.py \
  --timevae-overrides '{"epochs": 50, "num_synthetic_subjects": 10, "generation_method": "prior_v1"}' \
  --summary-path results/timevae_run_summary.json
```
