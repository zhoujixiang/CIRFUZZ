# road_prediction

Standalone project for **road prediction dataset initialization** — extracted from
`V2XGen` so the dataset pipeline can run independently of the fuzzing codebase.

## What it does

[dataset_init.py](dataset_init.py) takes a raw V2V4Real-style test dataset directory
(`-d`) and builds a semantic road-segmented dataset:

1. Copies each agent's PCD point clouds and YAML annotations into a `v2x_dataset/` tree.
2. Converts PCD → binary velodyne format (SemanticKITTI layout).
3. Runs the bundled **FRNet** LiDAR semantic segmentation network
   (`third/FRNet`, via [tools/frnet_infer_labels.py](tools/frnet_infer_labels.py))
   to produce per-point road predictions (`predictions/*.label`).
4. Records the dataset root in `config/dataset_config.yml` for downstream scripts.

## Layout

```
road_prediction/
├── dataset_init.py             # entry point
├── tools/
│   └── frnet_infer_labels.py   # FRNet per-point inference → .label writer
├── third/FRNet/                # FRNet (Apache-2.0), vendored
│   └── checkpoints/            # frnet-semantickitti_seg.pth (Git LFS, ~40 MB)
├── config/
│   └── dataset_config.yml      # dataset root, written by dataset_init.py
├── requirements.txt
└── .gitattributes              # FRNet weights tracked with Git LFS
```

## Usage

```bash
# Run from this project root (paths inside dataset_init.py are relative to CWD).
# --frnet-python must point at an interpreter that has the FRNet stack.
python dataset_init.py -d /path/to/your/v2v4real/testset \
    --frnet-python /path/to/frnet/env/bin/python

# or export it once:
export FRNET_PYTHON=/path/to/frnet/env/bin/python
python dataset_init.py -d /path/to/your/v2v4real/testset
```

The input `-d <test_dir>` is expected to contain two subdirs `0/` and `1/` holding
`*.pcd` and `*.yaml` files per agent.

**Multi-scene test data.** V2V4Real `test` data comes as several `testoutput_*`
scene dirs (each with its own `0/`, `1/`, frame numbers restarting at 0). Use
[merge_test_scenes.py](merge_test_scenes.py) to merge them into one dir that
satisfies `dataset_init.py`'s layout (frames re-numbered globally, `0/` and `1/`
share the same frame number per frame):

```bash
python merge_test_scenes.py -s /path/to/Data/test -o /path/to/merged_test
python dataset_init.py -d /path/to/merged_test --frnet-python /path/to/frnet/env/bin/python
```

Outputs are written next to the input, in the **parent** directory:

- `<parent>/v2x_dataset/` — pcd / yaml / velodyne / predictions per agent (`0/`, `1/`)
- `<parent>/semantic/semanticKitti/sequences/{11,12}/velodyne` — bin files fed to FRNet
- `<parent>/semantic/result/sequences/{11,12}/predictions/*.label` — FRNet predictions
- `config/dataset_config.yml` — records `<parent>` as `dataset_path`

## Label contract

The `.label` files hold **one `int32` label per point**, in the same order as the
matching `velodyne/<frame>.bin`, using Cityscapes-style ids:

```
0 unlabeled  7 road   8 sidewalk  9 parking  11 building  13 fence
17 pole  20 traffic sign  21 vegetation  22 terrain  24 person
25 rider  26 car  27 truck  28 bus  32 motorcycle  33 bicycle
```

**`road == 7`** is load-bearing: the fuzzer (`CIRFUZZING/cps_outside_ego.py`,
`cps_ops.py`) extracts the drivable region as id 7 and resolves the
`road` / `sidewalk` / `parking` / `terrain` / `ground` names against
`CIRFUZZING/third/SalsaNext/train/tasks/semantic/pretrained/data_cfg.yaml`.

FRNet predicts the SemanticKITTI taxonomy (`road` is index 8, 20 logits); the driver
remaps it to the ids above via the `FRNET_TO_CITYSCAPES` LUT in
[tools/frnet_infer_labels.py](tools/frnet_infer_labels.py). If you change FRNet
weights, check that LUT — and note `parking` (→ 9) is deliberately **not** mapped
to 7, so it stays out of the drivable region, matching the previous behaviour.

## Notes

- **[Environment]** FRNet needs mmengine 0.9.0, mmcv 2.1.0, mmdet 3.2.0, mmdet3d 1.3.0
  and `torch-scatter`. Install them in a **dedicated** env (see
  [requirements.txt](requirements.txt)) — in particular do **not** install mmcv 2.x
  into an env that has mmcv 1.x (such as the OpenCOOD `v2v4real` env), or
  `mmcv.ops` users break. The env must have a CUDA-enabled torch; the reference
  setup is Python 3.7 + torch 1.12.0+cu113.
- **[Weights]** `third/FRNet/checkpoints/frnet-semantickitti_seg.pth` (~40 MB) is
  Git-LFS tracked. Fetch it with `git lfs pull`, or download the official
  checkpoint from the FRNet repo:
  <https://github.com/Xiangxu-0103/FRNet> (Google Drive folder linked in its README).
  Provenance: FRNet, Xiang Xu et al., IEEE TIP 2025.
- **[Accuracy caveat]** The official checkpoint is trained on **SemanticKITTI**
  (HDL-64E, FOV +3°/−25°). V2V4Real uses a different lidar, so roughly 9% of points
  fall outside FRNet's vertical FOV and are clamped to the two edge rows of the
  range image. Expect road IoU below the SemanticKITTI number; if accuracy matters,
  fine-tune FRNet on V2V4Real or adjust the `fov_up`/`fov_down` used by
  `RangeInterpolation` and `FrustumRangePreprocessor`.
- The driver bypasses `SemanticKittiDataset` and its `*.pkl` infos on purpose — the
  frustum projection lives entirely in the transforms and the data preprocessor, so
  no dataset/annotation files are needed for inference.
- No GPU is required to *build* the dataset layout, but FRNet inference is run on CUDA.
