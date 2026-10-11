#!/usr/bin/env python3
"""Side-by-side BEV comparison of two road-segmentation label sets on the SAME frames.

Renders a 3-panel PNG per frame:

    [ FRNet ]  [ SalsaNext ]  [ agreement ]

Panels 1-2 use the same convention as ``frnet_visualize_road.py`` (road red,
other grey). Panel 3 shows where the two road masks agree and disagree.
Both label sets are read from disk — no model is loaded, so this runs without
the FRNet stack.

Example::

    python tools/compare_road_predictions.py \
        --data-root /bdata3/zjx/v2x/Data/predicted_data/semantic/semanticKitti \
        --frnet-labels /tmp/frnet_smoke/sequences \
        --salsa-root  /bdata3/zjx/v2x/Data/predicted_data/v2x_dataset \
        --sequences 11,12 --limit 3 --out-dir ./road_viz/compare
"""

import argparse
import os
import sys

import numpy as np

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TOOLS_DIR)

sys.path.insert(0, _TOOLS_DIR)

from frnet_visualize_road import _OTHER_COLOUR, _ROAD_COLOUR, _ROAD_ID  # noqa: E402

_LOAD_DIM = 4
_BOTH_COLOUR = "#4d4d4d"
_ONLY_FRNET_COLOUR = "#e02020"
_ONLY_SALSA_COLOUR = "#1f6feb"
_NEITHER_COLOUR = "#e4e4e4"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare FRNet vs SalsaNext road predictions on identical frames.")
    parser.add_argument("--data-root", required=True,
                        help="semanticKitti dir: sequences/<seq>/velodyne/*.bin")
    parser.add_argument("--frnet-labels", required=True,
                        help="Dir holding <seq>/predictions/<frame>.label (FRNet).")
    parser.add_argument("--salsa-root", required=True,
                        help="v2x_dataset dir holding <agent>/predictions/<frame>.label.")
    parser.add_argument("--seq-agent", default="11:0,12:1",
                        help="Mapping from sequence to SalsaNext agent dir. "
                             "(default: 11:0,12:1)")
    parser.add_argument("--sequences", default="11,12")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--out-dir", default=os.path.join(_REPO_ROOT, "road_viz", "compare"))
    parser.add_argument("--extent", type=float, default=60.0)
    parser.add_argument("--point-size", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=130)
    return parser.parse_args()


def load_labels(path):
    return np.fromfile(path, dtype=np.int32).astype(np.int64)


def _scatter_panel(ax, xyz, road, extent, point_size, title):
    inside = (np.abs(xyz[:, 0]) <= extent) & (np.abs(xyz[:, 1]) <= extent)
    other = inside & ~road
    on = inside & road
    ax.scatter(xyz[other, 0], xyz[other, 1], s=point_size, c=_OTHER_COLOUR,
               linewidths=0, label="non-road")
    ax.scatter(xyz[on, 0], xyz[on, 1], s=point_size, c=_ROAD_COLOUR,
               linewidths=0, label="road")
    _finish(ax, extent, title)
    ax.legend(loc="upper right", markerscale=8, framealpha=0.9)


def _diff_panel(ax, xyz, a_road, b_road, extent, point_size, title):
    inside = (np.abs(xyz[:, 0]) <= extent) & (np.abs(xyz[:, 1]) <= extent)
    a, b = a_road & inside, b_road & inside
    for mask, colour, label in (
        (~a & ~b, _NEITHER_COLOUR, "neither"),
        (b & ~a, _ONLY_SALSA_COLOUR, "only SalsaNext"),
        (a & ~b, _ONLY_FRNET_COLOUR, "only FRNet"),
        (a & b, _BOTH_COLOUR, "both"),
    ):
        ax.scatter(xyz[mask, 0], xyz[mask, 1], s=point_size, c=colour,
                   linewidths=0, label=label)
    _finish(ax, extent, title)
    ax.legend(loc="upper right", markerscale=8, framealpha=0.9)


def _finish(ax, extent, title):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title, fontsize=10)
    ax.grid(True, linewidth=0.3, alpha=0.3)


def main():
    args = parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seq_to_agent = {}
    for item in args.seq_agent.split(","):
        seq, agent = item.split(":")
        seq_to_agent[seq.strip()] = agent.strip()

    total = 0
    for seq in [s.strip() for s in args.sequences.split(",") if s.strip()]:
        agent = seq_to_agent.get(seq)
        if agent is None:
            raise SystemExit(f"no --seq-agent mapping for sequence {seq}")

        velodyne_dir = os.path.join(args.data_root, "sequences", seq, "velodyne")
        bins = sorted(f for f in os.listdir(velodyne_dir) if f.endswith(".bin"))
        if args.limit > 0:
            bins = bins[:args.limit]

        out_dir = os.path.join(args.out_dir, seq)
        os.makedirs(out_dir, exist_ok=True)

        for name in bins:
            frame = name[:-len(".bin")]
            points = np.fromfile(os.path.join(velodyne_dir, name),
                                 dtype=np.float32).reshape(-1, _LOAD_DIM)
            xyz = points[:, :3].astype(np.float64)

            frnet = load_labels(os.path.join(
                args.frnet_labels, seq, "predictions", f"{frame}.label"))
            salsa = load_labels(os.path.join(
                args.salsa_root, agent, "predictions", f"{frame}.label"))

            n = min(len(xyz), len(frnet), len(salsa))
            if not (len(xyz) == len(frnet) == len(salsa)):
                print(f"warn: seq {seq} {frame} length mismatch "
                      f"(pts={len(xyz)} frnet={len(frnet)} salsa={len(salsa)}), "
                      f"truncating to {n}")
            xyz, fa, sa = xyz[:n], frnet[:n] == _ROAD_ID, salsa[:n] == _ROAD_ID

            inter = int((fa & sa).sum())
            union = int((fa | sa).sum())
            iou = inter / union if union else float("nan")

            fig, axes = plt.subplots(1, 3, figsize=(21, 7.4), dpi=args.dpi)
            _scatter_panel(axes[0], xyz, fa, args.extent, args.point_size,
                           f"FRNet — road {fa.mean():.1%}")
            _scatter_panel(axes[1], xyz, sa, args.extent, args.point_size,
                           f"SalsaNext — road {sa.mean():.1%}")
            _diff_panel(axes[2], xyz, fa, sa, args.extent, args.point_size,
                        f"agreement — road IoU {iou:.3f}")
            fig.suptitle(f"Road prediction comparison — seq {seq} frame {frame} "
                         f"({n} pts)", fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, 0.96))

            out_png = os.path.join(out_dir, f"{frame}_compare.png")
            fig.savefig(out_png)
            plt.close(fig)

            total += 1
            print(f"seq {seq} {frame}: road frnet={fa.mean():.3f} "
                  f"salsa={sa.mean():.3f} IoU={iou:.3f} -> {out_png}")

    print(f"done: wrote {total} comparison images under {args.out_dir}")


if __name__ == "__main__":
    main()
