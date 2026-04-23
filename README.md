# Student-Scale Persistent 3D World Model

This repository implements the first two milestones of the project roadmap:

1. MOVi data ingestion and standardized `ClipSample` export.
2. Oracle persistent 3D memory with ground-truth RGB-D + camera poses.

The code is intentionally lightweight. It does **not** claim to reproduce the paper's full training stack or backbone. It implements the paper's core mechanism in a student-feasible form:

- write RGB-D observations into persistent 3D memory
- accumulate memory over time
- render / query that memory from a target view
- compare against a non-persistent baseline

## Current layout

```text
src/world_model/
  data/
  geometry/
  memory/
scripts/
  prepare_movi.py
  build_oracle_memory.py
  render_memory_view.py
  debug_oracle_memory.py
tests/
```

## Quick start

Run the synthetic oracle-memory demo:

```bash
PYTHONPATH=src python3 scripts/debug_oracle_memory.py \
  --source synthetic \
  --output-dir outputs/synthetic_oracle_demo
```

Run the unit tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## MOVi preparation

`prepare_movi.py` is wired for the official Kubric MOVi schema. It expects `tensorflow-datasets` and `tensorflow`.

Example:

```bash
PYTHONPATH=src python3 scripts/prepare_movi.py \
  --dataset movi_a \
  --resolution 128x128 \
  --split train \
  --limit 10 \
  --output-dir data/processed/movi_a_128
```

The script converts raw examples into compressed `.npz` clips with:

- `video`
- `depth`
- `poses`
- `intrinsics`
- `segmentations`
- `visibility`
- `metadata`

### Real MOVi-A export

On macOS arm64, create a Python 3.11 env for the TFDS export path:

```bash
conda create -y -p ./.conda-movi python=3.11 pip numpy matplotlib pillow
conda run -p ./.conda-movi pip install '.[data]'
```

Export a small real MOVi-A subset:

```bash
conda run --no-capture-output -p ./.conda-movi \
  env PYTHONPATH=src \
  python scripts/prepare_movi.py \
  --dataset movi_a \
  --resolution 128x128 \
  --split train \
  --limit 50 \
  --data-dir gs://kubric-public/tfds \
  --output-dir data/processed/movi_a_128_subset50
```

## Oracle memory demo

`debug_oracle_memory.py` builds persistent memory from context frames and renders a held-out target view. It also renders a `last-frame-only` comparison to make persistence visible quickly.

Outputs:

- `summary.png`
- `target_render.png`
- `last_frame_render.png`
- `context_strip.png`
- `metrics.json`

Run the real-data oracle-memory demo across the exported subset:

```bash
conda run --no-capture-output -p ./.conda-movi \
  env PYTHONPATH=src \
  python scripts/eval_oracle_memory_subset.py \
  --manifest data/processed/movi_a_128_subset50/manifest.json \
  --limit 50 \
  --output-dir outputs/movi_a_real_oracle_subset50
```

Each clip output directory contains:

- `context_strip.png`
- `target_gt.png`
- `target_render.png`
- `last_frame_render.png`
- `summary.png`
- `metrics.json`

## Compare runs

Summarize the current learned runs in one place:

```bash
PYTHONPATH=src python3 scripts/compare_run_metrics.py \
  --output-dir outputs/model_comparison_current
```

This writes:

- `outputs/model_comparison_current/comparison.md`
- `outputs/model_comparison_current/comparison.json`

By default the script looks for:

- `no_memory`
- `memory_baseline`
- `memory_strengthened`
- `uncertainty_writes`

The uncertainty row is allowed to be missing until that phase is implemented.

## Phase 5 evaluation

Run the benchmark harness over the validation split:

```bash
PYTHONPATH=src python3 scripts/eval_all.py \
  --checkpoint-kind last \
  --output-dir outputs/eval_phase5_current
```

This writes:

- `outputs/eval_phase5_current/evaluation.md`
- `outputs/eval_phase5_current/evaluation.json`

The current harness evaluates:

- overall validation metrics
- `high_motion`
- `occlusion_recovery`
- `high_memory_coverage`
- `depth_edge_heavy`

It also includes a `high_camera_motion` slice, which may be empty on `MOVi-A` if the camera is effectively static in the exported subset.
