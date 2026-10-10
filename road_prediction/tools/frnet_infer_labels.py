#!/usr/bin/env python3
"""Per-point semantic labels from an FRNet checkpoint (SemanticKITTI 20-class).

FRNet ships no inference entry point (its ``test.py`` only runs evaluation and
writes no ``.label`` files), so this driver reproduces the official *test*
path by hand:

    LoadPointsFromFile -> RangeInterpolation -> Pack3DDetInputs
        -> model.test_step -> per-point argmax
        -> FRNet ids remapped to Cityscapes-style ids (road == 7)
        -> int32 ``.label`` written next to the input frame

It deliberately does **not** use ``SemanticKittiDataset`` or the
``semantickitti_infos_*.pkl`` files: the frustum projection lives entirely in
``FrustumRangePreprocessor`` / ``RangeInterpolation``, neither of which needs a
dataset. It also drops the ``LoadAnnotations3D`` / ``PointSegClassMapping``
steps of FRNet's own ``test_pipeline``, because those require ground-truth
labels which the unlabelled V2X frames do not have.

Output convention (see ``dataset_init.py`` and the downstream fuzzer):
``<out-root>/sequences/<seq>/predictions/<frame>.label``, int32, one label per
point of ``<data-root>/sequences/<seq>/velodyne/<frame>.bin``, in the same
order, using Cityscapes-style ids with **road = 7**.
"""

import argparse
import glob
import os
import sys

import numpy as np

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TOOLS_DIR)
_DEFAULT_FRNET_REPO = os.path.join(_REPO_ROOT, "third", "FRNet")
_DEFAULT_CONFIG = os.path.join(
    _DEFAULT_FRNET_REPO, "configs", "frnet", "frnet-semantickitti_seg.py")
_DEFAULT_CHECKPOINT = os.path.join(
    _DEFAULT_FRNET_REPO, "checkpoints", "frnet-semantickitti_seg.pth")

_LOAD_DIM = 4  # x, y, z, intensity

# FRNet / SemanticKITTI class order -> Cityscapes-style ids used downstream.
# ``class_names`` (configs/_base_/datasets/semantickitti_seg.py) is:
#   car bicycle motorcycle truck bus person bicyclist motorcyclist road
#   parking sidewalk other-ground building fence vegetation trunck terrian
#   pole traffic-sign  (+ index 19 == ignore / unlabeled)
# The downstream fuzzer resolves these names against
# CIRFUZZING/third/SalsaNext/.../pretrained/data_cfg.yaml, where road == 7.
FRNET_TO_CITYSCAPES = np.array([
    26,  #  0 car            -> car
    33,  #  1 bicycle        -> bicycle
    32,  #  2 motorcycle     -> motorcycle
    27,  #  3 truck          -> truck
    28,  #  4 bus/other-veh  -> bus
    24,  #  5 person         -> person
    25,  #  6 bicyclist      -> rider
    25,  #  7 motorcyclist   -> rider
    7,   #  8 road           -> road            <-- load-bearing
    9,   #  9 parking        -> parking         (non-drivable downstream)
    8,   # 10 sidewalk       -> sidewalk
    6,   # 11 other-ground   -> ground
    11,  # 12 building       -> building
    13,  # 13 fence          -> fence
    21,  # 14 vegetation     -> vegetation
    17,  # 15 trunk          -> pole
    22,  # 16 terrain        -> terrain
    17,  # 17 pole           -> pole
    20,  # 18 traffic-sign   -> traffic sign
    0,   # 19 ignore         -> unlabeled
], dtype=np.int64)

_ROAD_ID = 7


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FRNet per-point inference and write int32 .label files "
                    "in the Cityscapes-style id space used by dataset_init.py.")
    parser.add_argument("--data-root", required=True,
                        help="Directory containing sequences/<seq>/velodyne/*.bin")
    parser.add_argument("--out-root", required=True,
                        help="Output root; sequences/<seq>/predictions/ is created under it.")
    parser.add_argument("--sequences", default="11,12",
                        help="Comma-separated sequence names to process. (default: 11,12)")
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=_DEFAULT_CHECKPOINT)
    parser.add_argument("--frnet-repo", default=_DEFAULT_FRNET_REPO,
                        help="Path to the vendored FRNet repository.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N frames per sequence (0 = all). "
                             "Useful for smoke tests.")
    parser.add_argument("--report", action="store_true",
                        help="Print per-frame point counts and road fraction.")
    return parser.parse_args()


def build_model(config_path, checkpoint_path, device):
    """Load the FRNet config, build the model and load the checkpoint."""
    import torch
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint

    import mmdet       # noqa: F401  (registers 'mmdet.CrossEntropyLoss' etc.)
    import mmdet3d     # noqa: F401
    from mmdet3d.registry import MODELS

    import frnet.models  # noqa: F401  (registers FRNet, FrustumRangePreprocessor, FRHead)

    init_default_scope("mmdet3d")

    cfg = Config.fromfile(config_path)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint_path, map_location="cpu", strict=False)
    model.to(device)
    model.eval()
    return model


def build_pipeline():
    """Official FRNet test_pipeline minus the label-loading steps."""
    from mmcv.transforms import Compose
    from mmdet3d.registry import TRANSFORMS

    import frnet.datasets.transforms  # noqa: F401  (registers RangeInterpolation)

    return Compose([
        TRANSFORMS.build(dict(type="LoadPointsFromFile", coord_type="LIDAR",
                              load_dim=_LOAD_DIM, use_dim=_LOAD_DIM,
                              backend_args=None)),
        TRANSFORMS.build(dict(type="RangeInterpolation", H=64, W=2048,
                              fov_up=3.0, fov_down=-25.0, ignore_index=19)),
        # meta_keys MUST include num_points: FRHead.predict slices the
        # interpolated points off with it, otherwise the output is longer
        # than the input cloud.
        TRANSFORMS.build(dict(type="Pack3DDetInputs", keys=["points"],
                              meta_keys=["num_points"])),
    ])


def infer_frame(model, pipeline, bin_path, device):
    """Return FRNet class ids (0..19) for every point of ``bin_path``, in file order."""
    import torch
    from mmengine.dataset import pseudo_collate

    result = pipeline(dict(lidar_points=dict(lidar_path=bin_path,
                                             num_pts_feats=_LOAD_DIM)))
    data = pseudo_collate([result])
    with torch.no_grad():
        outputs = model.test_step(data)

    sample = outputs[0]
    pred = getattr(sample, "pred_pts_seg", None)
    if pred is not None and "pts_semantic_mask" in pred:
        return pred.pts_semantic_mask.cpu().numpy().astype(np.int64)

    # Fallback: argmax the raw per-point logits.
    logits = sample.pts_seg_logits.pts_seg_logits
    return logits.cpu().numpy().argmax(axis=-1).astype(np.int64)


def main():
    args = parse_args()

    if not os.path.isdir(args.frnet_repo):
        raise SystemExit(f"FRNet repo not found: {args.frnet_repo}")
    if not os.path.isfile(args.checkpoint):
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    sys.path.insert(0, args.frnet_repo)

    model = build_model(args.config, args.checkpoint, args.device)
    pipeline = build_pipeline()

    total = 0
    for seq in [s.strip() for s in args.sequences.split(",") if s.strip()]:
        velodyne_dir = os.path.join(args.data_root, "sequences", seq, "velodyne")
        bins = sorted(glob.glob(os.path.join(velodyne_dir, "*.bin")))
        if not bins:
            raise SystemExit(f"no .bin files under {velodyne_dir}")
        if args.limit > 0:
            bins = bins[:args.limit]

        pred_dir = os.path.join(args.out_root, "sequences", seq, "predictions")
        os.makedirs(pred_dir, exist_ok=True)

        for bin_path in bins:
            expected = os.path.getsize(bin_path) // (4 * _LOAD_DIM)
            frnet_ids = infer_frame(model, pipeline, bin_path, args.device)
            if frnet_ids.shape[0] != expected:
                raise RuntimeError(
                    f"{bin_path}: got {frnet_ids.shape[0]} labels for "
                    f"{expected} points (num_points meta key not applied?)")

            city_ids = FRNET_TO_CITYSCAPES[frnet_ids]
            out_path = os.path.join(
                pred_dir, os.path.basename(bin_path)[:-len(".bin")] + ".label")
            city_ids.astype(np.int32).tofile(out_path)
            total += 1

            if args.report:
                road = float((city_ids == _ROAD_ID).mean())
                print(f"seq {seq} {os.path.basename(bin_path)}: "
                      f"{expected} pts, road={road:.3f}, "
                      f"classes={sorted(np.unique(city_ids).tolist())}")

    print(f"done: wrote {total} .label files under {args.out_root}")


if __name__ == "__main__":
    main()
