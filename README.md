# CIRFUZZ

Research code for **fuzzing V2X (vehicle‑to‑everything) cooperative perception**.

The repository bundles three sub‑projects that together build a pipeline:
prepare a semantic road dataset → run a cooperative perception model on it →
mutate cooperative road objects to expose ego/V2X disagreements.

```
CIRFUZZ/
├── CIRFUZZING/        # the fuzzing tool: v2x_fuzz.py + helpers   (main entry point)
├── road_prediction/   # builds the semantic road dataset with SalsaNext
└── V2V4Real/          # OpenCOOD-based cooperative perception stack (code only)
```

> **Not tracked in this repo** (populated at runtime, several tens of GB — see
> [.gitignore](.gitignore)): `V2V4Real/model/` (checkpoints), `*/logs/`
> (training/inference logs), `CIRFUZZING/_assets/` (ShapeNet meshes).

---

## 1. `CIRFUZZING/v2x_fuzz.py` — the fuzzer

### What it does

`v2x_fuzz.py` fuzzes a cooperative perception model by **editing the objects a
cooperative agent (CAV) reports on the road**, then measuring how much the
ego‑vs‑V2X perception *disagrees* afterwards. Mutations happen in a configurable
road region so that inserted/deleted objects look plausible to the model.

Two mutation primitives are used:

| Operation | Effect |
|-----------|--------|
| **insert** | Adds a synthetic vehicle (ShapeNet mesh) onto a road region of the cooperative agent's point cloud + YAML annotation. |
| **delete** | Removes an existing cooperative vehicle from the point cloud + YAML, optionally filling the resulting lidar shadow with sampled road points. |

Each mutated scene is written out as a self‑contained OpenCOOD case
(`velodyne/*.pcd` + `*.yaml` per agent), so it can be fed straight to
`V2V4Real/opencood/tools/inference.py`.

### Pipeline

```
                        ┌─────────────────────────────┐
  dataset frame ──────► │  load ego + coop PC / YAML  │
                        │  segment road & non-road    │
                        └──────────────┬──────────────┘
                                       ▼
                        ┌─────────────────────────────┐
                        │  candidate road regions in  │
                        │  the chosen mutation region │
                        └──────────────┬──────────────┘
                                       ▼
        ┌──────────────────────────────────────────────────┐
        │  beam search over `--ops-per-case` mutation steps │
        │  each step: generate `--candidates-per-step`      │
        │  insert/delete candidates → save all → optionally │
        │  run inference → keep best (fuzz) / random        │
        └──────────────────────┬───────────────────────────┘
                               ▼
        ┌──────────────────────────────────────────────────┐
        │  fitness = object_missing + localization_failure  │
        │  (Δ V2X − ego), from inference.py --compare_ego_v2x│
        └──────────────────────┬───────────────────────────┘
                               ▼
        valid_cases/ · baseline_cases/ · interesting_cases/ · fuzz_manifest.yaml
```

**Search processes** (`--process`):

- `fuzz` — greedy/beam search: at every step keep the candidate with the
  highest fitness as the seed for the next step.
- `random` — same candidate generation, but sample a random candidate instead.
- `both` — every selected frame is processed once by each strategy
  (a controlled comparison between guided and random search).

### Requirements

`v2x_fuzz.py` itself needs:

```
python >= 3.7
numpy, pyyaml, open3d, matplotlib
```

Inference (`--run-inference`) additionally shells out to
`V2V4Real/opencood/tools/inference.py`, which needs the full OpenCOOD/torch
stack — see [V2V4Real/requirements.txt](V2V4Real/requirements.txt) and run it in
the project's `v2v4real` conda env.

> **Layout assumption:** the script treats its own parent directory as the repo
> root and expects `V2V4Real/` to be a sibling of `CIRFUZZING/` (it is, in this
> repo). Override with `--inference-script` if you move things around.

### Quick start

```bash
cd CIRFUZZING

# 1) generate fuzzing cases only (no model inference)
python v2x_fuzz.py \
    --model-dir /path/to/opencood/logs/point_pillar_coalign_v2v4real \
    --iterations 50 \
    --process both \
    --ops-per-case 2 \
    --candidates-per-step 5 \
    --mutation-region coop_only_outside \
    --output-dir ./fuzz_out/run1

# 2) same, but run the model on every candidate and keep only cases that
#    increase the ego/V2X error over the unmutated baseline
python v2x_fuzz.py \
    --model-dir /path/to/opencood/logs/point_pillar_coalign_v2v4real \
    --iterations 50 \
    --process fuzz \
    --run-inference \
    --fusion-method early \
    --output-dir ./fuzz_out/run1_infer
```

`--model-dir` is an OpenCOOD log directory (it must contain `config.yaml` and at
least one `*epoch*.pth` checkpoint when `--run-inference` is used).

### CLI reference

**Required**

| Flag | Description |
|------|-------------|
| `--model-dir` | OpenCOOD model/log directory under test. |
| `--iterations` | Number of fuzzing iterations (frames sampled). |

**Search / mutation**

| Flag | Default | Description |
|------|---------|-------------|
| `--process` | `fuzz` | `fuzz`, `random`, or `both`. |
| `--ops-per-case` | `1` | Mutation steps per selected scene (beam‑search depth). |
| `--candidates-per-step` | `5` | Distinct offspring generated per step. |
| `--operations` | `both` | `insert`, `delete`, or `both` (chosen 50/50 at each step). |
| `--insert-prob` | `0.5` | Kept for compatibility; `both` always uses 50/50. |
| `--mutation-region` | `coop_only_outside` | Where candidates may be placed — see below. |
| `--max-attempts-per-iteration` | `20` | Rejection‑sampling attempts per candidate. |
| `--fill-delete-shadow` | off | Fill the lidar shadow left by a deleted vehicle. |
| `--check-non-road-overlap` | off | Reject insertions overlapping coop non‑road points (strict). |
| `--skip-existing-collision-check` | off | Allow insertions colliding with / occluded by existing YAML vehicles. |
| `--mesh-path` | auto | Vehicle mesh for insertion (defaults to a ShapeNet mesh under `_assets/`). |
| `--seed` | `0` | RNG seed. |
| `--verbose-failures` | off | Print why each mutation attempt was rejected. |

**Data / regions**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset-dir` | `CIRFUZZING/dataset/v2x_dataset` | Input dataset directory. |
| `--dataset-format` | `auto` | `auto`, `v2xgen_flat` (`0/yaml`,`1/yaml`), or `opencood` (`scenario/0`,`scenario/1`). |
| `--frames` | all | Comma‑separated frame filter (`scenario/frame` or `scenario:frame` for OpenCOOD). |
| `--ego-id` / `--coop-id` | `0` / `1` | Agent IDs. |
| `--road-source` | `auto` | Use semantic labels when present, else a ground heuristic. |
| `--road-label` | `7` | Road label id. |
| `--ego-range` / `--coop-range` | `[-100,-40,-5,100,40,3]` | 6‑tuple lidar extent `[xmin,ymin,zmin,xmax,ymax,zmax]`. |
| `--voxel-size` | `2.5` | Voxel size used for region bookkeeping. |
| `--dilation` | `2` | Dilation radius (voxels) for road regions. |

**Inference / fitness**

| Flag | Default | Description |
|------|---------|-------------|
| `--run-inference` | off | Run `inference.py` on generated candidates. |
| `--inference-script` | `V2V4Real/opencood/tools/inference.py` | Path to the inference entry point. |
| `--fusion-method` | `early` | `late`, `early`, `intermediate`, `nofusion`. |
| `--fitness-threshold` | `0.0` | Cases with fitness ≥ threshold are `interesting` (when no baseline). |
| `--is-sim` | off | Pass `--isSim` to `inference.py`. |
| `--log-every` | `1` | Inference log frequency. |

**Output**

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `CIRFUZZING/fuzz_out` | Where cases and the manifest are written. |
| `--output-timestamp` | `000000` | Timestamp name used inside each generated case. |
| `--save-visualizations` | off | Save a 3D matplotlib image per valid candidate. |
| `--show-visualization` | off | Open an interactive Open3D window (needs X11/DISPLAY). |

### Mutation regions

`--mutation-region` selects which road region candidates are drawn from:

- `coop_only_outside` *(default)* — the part of the road that only the
  cooperative lidar can see; mutations are invisible to the ego sensor, which
  is where fusion errors are most likely.
- `overlap` — the ego/coop overlap region (both insert and delete here).
- `ego_only` — only the ego‑side files are mutated, in the region only the ego
  lidar sees; the cooperative side is left untouched.

### Fitness

When inference is enabled, `inference.py --compare_ego_v2x` reports per‑frame
error counts; fitness is

```
fitness = object_missing + localization_failure
```

taken from the `delta_v2x_minus_ego` counts — i.e. how much worse V2X fusion is
than ego‑only perception on the mutated frame. A case is **interesting** when
its fitness exceeds the unmutated baseline (or `--fitness-threshold` when no
baseline is available).

### Outputs

```
<output-dir>/
├── baseline_cases/       # unmutated copy of every selected frame (for comparison)
├── valid_cases/          # every generated candidate: pcd + yaml + metadata.yaml + vis
├── interesting_cases/    # copies of cases whose fitness beat the baseline/threshold
├── inference_model/      # model dir staged for inference (config + symlinked checkpoints)
├── inference_staging/    # cases copied into a single dir for one inference call
├── inference_logs/       # stdout.txt / stderr.txt of the inference run
└── fuzz_manifest.yaml    # run config + per‑case records (operation, step, fitness, …)
```

### Helper modules

| File | Role |
|------|------|
| `v2x_ops.py` | Point‑cloud primitives: `insert_object`, `delete_object`, `load_point_cloud`, mesh scaling, collision/occlusion checks, road segmentation. |
| `v2x_outside_ego.py` | Outside‑ego region logic: 3‑way point classification, pose transforms, and a standalone CLI (`python v2x_outside_ego.py`) to prepare/visualise outside‑ego assets. |
| `config/` | Simulator config (`lidar_config.yml`) and dataset path configs used by the helpers. |
| `core/`, `utils/` | lidar simulation and shared file/point‑cloud utilities. |
| `dataset/v2x_dataset/` | Small sample dataset (2 agents × 3 frames) for smoke‑testing. |

---

## 2. `road_prediction/` — dataset builder

Standalone project that turns a raw V2V4Real‑style testset into a semantic
road‑segmented dataset using the bundled **SalsaNext** network
(`dataset_init.py` → `predictions/*.label`). See
[road_prediction/README.md](road_prediction/README.md) for details.

## 3. `V2V4Real/` — cooperative perception stack

OpenCOOD‑based perception stack (`opencood/`) providing the models, the
`inference.py --compare_ego_v2x` entry point used by the fuzzer, and the
`hypes_yaml/` model configs. Code only — checkpoints and logs are excluded.

---

## Notes

- All `_assets/` ShapeNet meshes and model checkpoints must be present locally
  before running insertion fuzzing or inference; they are not part of this repo.
- Paths inside the shipped configs are absolute and environment‑specific —
  adjust `--dataset-dir`, `--model-dir`, `config/dataset_config.yml` and the
  `V2V4Real/opencood/hypes_yaml/*.yaml` `root_dir`/`validate_dir` entries to
  match your machine.
