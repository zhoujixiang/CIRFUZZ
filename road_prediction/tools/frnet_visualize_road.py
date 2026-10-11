#!/usr/bin/env python3
"""Run FRNet road prediction and save a BEV (top-down) visualisation per frame.

Road points are drawn in red, everything else in grey, on equal-aspect axes in
metres. One PNG per input frame:

    <out-dir>/<seq>/<frame>_bev.png

Inference reuses the exact path of ``frnet_infer_labels.py`` (same model build,
same pipeline, same FRNet -> Cityscapes remap), so the road mask shown here is
the same one written to ``predictions/*.label``.
"""

import argparse
import glob
import os
import sys

import numpy as np

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TOOLS_DIR)
_DEFAULT_FRNET_REPO = os.path.join(_REPO_ROOT, "third", "FRNet")

# The driver lives next to this file and needs third/FRNet importable, so set
# up sys.path before importing it (it imports frnet.* at module import time).
sys.path.insert(0, _TOOLS_DIR)
sys.path.insert(0, _DEFAULT_FRNET_REPO)

from frnet_infer_labels import (  # noqa: E402
    FRNET_TO_CITYSCAPES,
    _ROAD_ID,
    build_model,
    build_pipeline,
    infer_frame,
)

_LOAD_DIM = 4
_ROAD_COLOUR = "#e02020"
_OTHER_COLOUR = "#8a8a8a"


def parse_args():
    parser = argparse.ArgumentParser(
        description="FRNet road prediction -> BEV PNG (red = road, grey = other).")
    parser.add_argument("--data-root", required=True,
                        help="Directory containing sequences/<seq>/velodyne/*.bin")
    parser.add_argument("--out-dir", default=os.path.join(_REPO_ROOT, "road_viz"),
                        help="Where to write <seq>/<frame>_bev.png")
    parser.add_argument("--sequences", default="11,12",
                        help="Comma-separated sequence names. (default: 11,12)")
    parser.add_argument("--limit", type=int, default=5,
                        help="Max frames per sequence (0 = all). (default: 5)")
    parser.add_argument("--extent", type=float, default=60.0,
                        help="Half-size of the BEV window in metres. (default: 60)")
    parser.add_argument("--point-size", type=float, default=1.0,
                        help="Scatter marker size. (default: 1.0)")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--config",
                        default=os.path.join(_DEFAULT_FRNET_REPO,
                                             "configs/frnet/frnet-semantickitti_seg.py"))
    parser.add_argument("--checkpoint",
                        default=os.path.join(_DEFAULT_FRNET_REPO, "checkpoints",
                                             "frnet-semantickitti_seg.pth"))
    parser.add_argument("--frnet-repo", default=_DEFAULT_FRNET_REPO)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-labels", action="store_true",
                        help="Also write predictions/<frame>.label next to the PNGs.")
    return parser.parse_args()


def render_bev(points, labels, out_path, title, extent, point_size, dpi):
    """Top-down scatter: road in red over grey non-road."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz = np.asarray(points, dtype=np.float64)[:, :3]
    road = np.asarray(labels) == _ROAD_ID

    inside = (np.abs(xyz[:, 0]) <= extent) & (np.abs(xyz[:, 1]) <= extent)
    other = inside & ~road
    on_road = inside & road

    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    ax.scatter(xyz[other, 0], xyz[other, 1], s=point_size,
               c=_OTHER_COLOUR, linewidths=0, label="non-road")
    ax.scatter(xyz[on_road, 0], xyz[on_road, 1], s=point_size,
               c=_ROAD_COLOUR, linewidths=0, label="road")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.grid(True, linewidth=0.3, alpha=0.3)
    ax.legend(loc="upper right", markerscale=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    args = parse_args()

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

        png_dir = os.path.join(args.out_dir, seq)
        os.makedirs(png_dir, exist_ok=True)
        label_dir = os.path.join(args.out_dir, seq, "predictions")
        if args.save_labels:
            os.makedirs(label_dir, exist_ok=True)

        for bin_path in bins:
            frame = os.path.basename(bin_path)[:-len(".bin")]
            expected = os.path.getsize(bin_path) // (4 * _LOAD_DIM)
            points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, _LOAD_DIM)

            frnet_ids = infer_frame(model, pipeline, bin_path, args.device)
            if frnet_ids.shape[0] != expected:
                raise RuntimeError(
                    f"{bin_path}: got {frnet_ids.shape[0]} labels for {expected} points")
            # infer_frame returns raw FRNet ids (road == 8); remap to the
            # Cityscapes-style ids used everywhere downstream (road == 7).
            labels = FRNET_TO_CITYSCAPES[frnet_ids]

            road_visible = float((labels == _ROAD_ID).mean())
            out_png = os.path.join(png_dir, f"{frame}_bev.png")
            render_bev(
                points, labels, out_png,
                title=f"FRNet road prediction — seq {seq} frame {frame} "
                      f"(road {road_visible:.1%}, {expected} pts)",
                extent=args.extent, point_size=args.point_size, dpi=args.dpi)

            if args.save_labels:
                labels.astype(np.int32).tofile(
                    os.path.join(label_dir, f"{frame}.label"))

            total += 1
            print(f"seq {seq} {frame}: {expected} pts, road={road_visible:.3f} -> {out_png}")

    print(f"done: wrote {total} BEV images under {args.out_dir}")


if __name__ == "__main__":
    main()
