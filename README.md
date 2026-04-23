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

## Oracle memory demo

`debug_oracle_memory.py` builds persistent memory from context frames and renders a held-out target view. It also renders a `last-frame-only` comparison to make persistence visible quickly.

Outputs:

- `summary.png`
- `target_render.png`
- `last_frame_render.png`
- `context_strip.png`
- `metrics.json`
