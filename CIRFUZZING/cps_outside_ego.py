
import argparse
import os
import re
import subprocess
import sys
from typing import Optional

import numpy as np
import yaml

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

_DEFAULT_DATASET_DIR = os.path.join(
    os.path.dirname(_ROOT), "V2V4Real", "v2v4real", "v2x_dataset",
)
_ROAD_LABEL_ID = 7
_VOXEL_SIZE = 2.5
_DILATION_RADIUS = 2
_VIS_COLOURS = {
    "ego_only":          (0.2, 0.2, 0.8),
    "overlap":           (0.2, 0.8, 0.2),
    "coop_only_inside":  (0.8, 0.2, 0.2),
    "coop_only_outside": (0.8, 0.8, 0.2),
}

_EGO_CAV_LIDAR_RANGE = [-100, -40, -5, 100, 40, 3]

_OPEN3D_OK = None
_O3D_MOD = None



def _probe_open3d(timeout: float = 8.0) -> bool:
    global _OPEN3D_OK
    if _OPEN3D_OK is not None:
        return _OPEN3D_OK
    try:
        r = subprocess.run(
            [sys.executable, "-c", "import open3d; print('ok')"],
            capture_output=True, text=True, timeout=timeout,
        )
        _OPEN3D_OK = (r.returncode == 0 and r.stdout.strip() == "ok")
    except subprocess.TimeoutExpired:
        _OPEN3D_OK = False
    except Exception:
        _OPEN3D_OK = False
    return _OPEN3D_OK


def _lazy_o3d():
    global _O3D_MOD
    if _O3D_MOD is not None:
        return _O3D_MOD
    if not _probe_open3d():
        return None
    try:
        import open3d as m
        _O3D_MOD = m
    except Exception:
        return None
    return _O3D_MOD



def _load_bin(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def _load_label(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint32).ravel()


def _load_yaml(path: str) -> dict:
    loader = yaml.Loader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    with open(path, "r") as f:
        return yaml.load(f, Loader=loader)


def load_cav_data(dataset_dir: str, cam_id: str, frame: str):
    velodyne_file = os.path.join(dataset_dir, cam_id, "velodyne", f"{frame}.bin")
    label_file = os.path.join(dataset_dir, cam_id, "predictions", f"{frame}.label")
    yaml_file = os.path.join(dataset_dir, cam_id, "yaml", f"{frame}.yaml")

    pc = _load_bin(velodyne_file)

    labels = None
    if os.path.exists(label_file):
        labels = _load_label(label_file)
        if labels.shape[0] != pc.shape[0]:
            print(f"  ⚠ Label count ({labels.shape[0]}) != point "
                  f"count ({pc.shape[0]}) for {cam_id}/{frame}")
            labels = None

    params = _load_yaml(yaml_file)
    lidar_pose = np.asarray(params["lidar_pose"], dtype=np.float64)
    assert lidar_pose.shape == (4, 4), f"lidar_pose shape {lidar_pose.shape}"

    return pc, labels, lidar_pose


def compute_transform(source_pose, target_pose):
    return np.linalg.inv(target_pose) @ source_pose


def transform_points(xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    xyz_h = np.column_stack([xyz, np.ones(xyz.shape[0], dtype=xyz.dtype)])
    return (xyz_h @ T.T)[:, :3]



def extract_road(pc: np.ndarray, labels: np.ndarray,
                 road_label: int = _ROAD_LABEL_ID) -> Optional[np.ndarray]:
    if labels is None:
        return None
    mask = labels == road_label
    if mask.sum() == 0:
        return None
    return pc[mask, :3].copy()


def load_vehicle_corners_from_yaml(
    yaml_path: str,
    source_pose: Optional[np.ndarray] = None,
    target_pose: Optional[np.ndarray] = None,
):
    if not yaml_path or not os.path.exists(yaml_path):
        return []

    params = _load_yaml(yaml_path)
    if not params.get("vehicles"):
        return []

    from utils.common_utils import get_corners_positions

    corners_lidar, _, _, _ = get_corners_positions(params)
    corners_lidar = np.asarray(corners_lidar, dtype=np.float32)

    if source_pose is not None and target_pose is not None and corners_lidar.size > 0:
        T = compute_transform(source_pose, target_pose)
        corners_lidar = np.asarray(
            [transform_points(corner, T) for corner in corners_lidar],
            dtype=np.float32,
        )

    return [corners_lidar[i] for i in range(corners_lidar.shape[0])]



def _voxel_set(xyz: np.ndarray, voxel_size: float, dilation: int):
    raw = set()
    for p in xyz:
        raw.add(tuple(np.floor(p / voxel_size).astype(np.int64)))

    if dilation <= 0:
        return raw

    offsets = [(dx, dy, dz)
               for dx in range(-dilation, dilation + 1)
               for dy in range(-dilation, dilation + 1)
               for dz in range(-dilation, dilation + 1)]
    dilated = set()
    for vx in raw:
        for off in offsets:
            dilated.add(tuple(vx[i] + off[i] for i in range(3)))
    dilated.update(raw)
    return dilated


def classify_three_way(
    ego_road: np.ndarray,
    coop_road: np.ndarray,
    voxel_size: float = _VOXEL_SIZE,
    dilation: int = _DILATION_RADIUS,
    ego_range: Optional[list] = None,
):
    ego_voxels = _voxel_set(ego_road, voxel_size, dilation) if ego_road is not None else set()
    coop_voxels = _voxel_set(coop_road, voxel_size, dilation) if coop_road is not None else set()

    overlap_voxels = ego_voxels & coop_voxels

    if ego_range is None:
        ego_range = [-np.inf, -np.inf, -np.inf, np.inf, np.inf, np.inf]

    def _within_ego_range(p):
        return (ego_range[0] <= p[0] <= ego_range[3] and
                ego_range[1] <= p[1] <= ego_range[4] and
                ego_range[2] <= p[2] <= ego_range[5])

    def _classify(pts):
        if pts is None:
            return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)
        n = pts.shape[0]
        ov = np.zeros(n, dtype=bool)
        for i, p in enumerate(pts):
            vx = tuple(np.floor(p / voxel_size).astype(np.int64))
            if vx in overlap_voxels:
                ov[i] = True
        return ov, ~ov

    ego_overlap, ego_only = _classify(ego_road)
    coop_overlap, coop_only = _classify(coop_road)

    out = {}

    if ego_road is not None and ego_only.sum() > 0:
        out["ego_only"] = ego_road[ego_only]
    else:
        out["ego_only"] = np.zeros((0, 3), dtype=np.float32)

    overlap_parts = []
    if ego_road is not None and ego_overlap.sum() > 0:
        overlap_parts.append(ego_road[ego_overlap])
    if coop_road is not None and coop_overlap.sum() > 0:
        overlap_parts.append(coop_road[coop_overlap])
    out["overlap"] = np.vstack(overlap_parts) if overlap_parts else np.zeros((0, 3), dtype=np.float32)

    if coop_road is not None and coop_only.sum() > 0:
        coop_pts = coop_road[coop_only]
        inside_mask = np.array([_within_ego_range(p) for p in coop_pts])
        out["coop_only_inside"] = coop_pts[inside_mask] if inside_mask.sum() > 0 else np.zeros((0, 3), dtype=np.float32)
        out["coop_only_outside"] = coop_pts[~inside_mask] if (~inside_mask).sum() > 0 else np.zeros((0, 3), dtype=np.float32)
    else:
        out["coop_only_inside"] = np.zeros((0, 3), dtype=np.float32)
        out["coop_only_outside"] = np.zeros((0, 3), dtype=np.float32)

    return out


def prepare_outside_ego_assets(
    dataset_dir: str,
    frame: str,
    road_label: int = _ROAD_LABEL_ID,
    ego_range: Optional[list] = None,
    voxel_size: float = _VOXEL_SIZE,
    dilation: int = _DILATION_RADIUS,
):
    if ego_range is None:
        ego_range = _EGO_CAV_LIDAR_RANGE

    ego_pc, ego_labels, ego_pose = load_cav_data(dataset_dir, "0", frame)
    coop_pc, coop_labels, coop_pose = load_cav_data(dataset_dir, "1", frame)

    ego_road = extract_road(ego_pc, ego_labels, road_label)
    coop_road = extract_road(coop_pc, coop_labels, road_label)

    T = compute_transform(coop_pose, ego_pose)
    coop_pc_xyz = transform_points(coop_pc[:, :3], T)
    if coop_road is not None:
        coop_road = transform_points(coop_road, T)

    categories = classify_three_way(
        ego_road, coop_road, voxel_size=voxel_size, dilation=dilation, ego_range=ego_range
    )
    coop_only_outside = categories.get("coop_only_outside", np.zeros((0, 3), dtype=np.float32))

    ego_non_road = None
    if ego_labels is not None:
        ego_non_road = ego_pc[ego_labels != road_label, :3].copy()

    coop_yaml = os.path.join(dataset_dir, "1", "yaml", f"{frame}.yaml")
    coop_vehicle_corners = load_vehicle_corners_from_yaml(
        coop_yaml,
        source_pose=coop_pose,
        target_pose=ego_pose,
    )

    if ego_pc.shape[1] > 3:
        intensity = float(np.mean(ego_pc[:, 3]))
    else:
        intensity = 0.0
    road_patch_xyzi = np.hstack([
        coop_only_outside.astype(np.float32),
        np.full((coop_only_outside.shape[0], 1), intensity, dtype=np.float32),
    ]) if coop_only_outside.size > 0 else np.zeros((0, 4), dtype=np.float32)
    scene_pc = np.vstack([ego_pc, road_patch_xyzi]) if road_patch_xyzi.shape[0] > 0 else ego_pc.copy()

    return {
        "ego_pc": ego_pc,
        "ego_labels": ego_labels,
        "ego_pose": ego_pose,
        "coop_pc_xyz": coop_pc_xyz,
        "coop_pose": coop_pose,
        "ego_road": ego_road,
        "coop_road": coop_road,
        "categories": categories,
        "coop_only_outside": coop_only_outside,
        "ego_non_road": ego_non_road,
        "coop_vehicle_corners": coop_vehicle_corners,
        "vehicle_corners": coop_vehicle_corners,
        "scene_pc": scene_pc,
        "ego_total": ego_road.shape[0] if ego_road is not None else 0,
        "coop_total": coop_road.shape[0] if coop_road is not None else 0,
    }



def _print_summary(categories: dict, ego_total: int, coop_total: int,
                   ego_range: Optional[list] = None):
    if ego_range is None:
        ego_range = _EGO_CAV_LIDAR_RANGE

    print(f"\n  ┌──────────────────────────────────────────────────┐")
    print(f"  │  Three-way road perception breakdown (4-colour)  │")
    print(f"  ├──────────────────────────────────────────────────┤")
    print(f"  │  Ego road total:       {ego_total:>7,} points                 │")
    print(f"  │  Coop road total:      {coop_total:>7,} points                 │")
    print(f"  ├──────────────────────────────────────────────────┤")

    print(f"  │  Ego perception range (X Y Z):                         │")
    print(f"  │    X [{ego_range[0]}, {ego_range[3]}] m   "
          f"Y [{ego_range[1]}, {ego_range[4]}] m   "
          f"Z [{ego_range[2]}, {ego_range[5]}] m  │")
    print(f"  ├──────────────────────────────────────────────────┤")

    rows = [
        ("① Ego-only",       "ego_only",           ego_total),
        ("② Overlap",        "overlap",            ego_total + coop_total),
        ("③ Coop-only inside","coop_only_inside",  coop_total),
        ("④ Coop-only outside","coop_only_outside", coop_total),
    ]
    for label, key, denom in rows:
        arr = categories.get(key)
        count = arr.shape[0] if arr is not None else 0
        pct = count / denom * 100 if denom > 0 else 0.0
        print(f"  │  {label:20s}  {count:>7,} points  ({pct:5.1f}%)  │")

    outside = categories.get("coop_only_outside")
    outside_count = outside.shape[0] if outside is not None else 0
    total_coop_only = outside_count + (categories.get("coop_only_inside").shape[0]
                                        if categories.get("coop_only_inside") is not None else 0)
    if total_coop_only > 0:
        print(f"  │  ──────────────────────────────────────────────── │")
        print(f"  │  Coop-only outside ego range:  {outside_count:>7,} / "
              f"{total_coop_only:<7,}  ({outside_count/total_coop_only*100:5.1f}%)  │")

    print(f"  └──────────────────────────────────────────────────┘")



def _make_pcd_xyz(xyz, color):
    o3d = _lazy_o3d()
    if o3d is None:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    n = xyz.shape[0]
    colors = np.tile(color, (n, 1))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def _visualize(title: str, geometries: list, point_size: float = 2.0,
               window_size: tuple = (1024, 768),
               save_path: Optional[str] = None):
    o3d = _lazy_o3d()
    if o3d is None:
        return

    if save_path is None:
        display = os.environ.get("DISPLAY")
        if not display or display.strip() == "":
            print(f"  [skipped — no display]  {title}")
            return

    print(f"\n  🖼  {title}")
    if save_path is None:
        print("  └─ Drag to rotate | Scroll to zoom | Ctrl+C to exit\n")

    vis = o3d.visualization.Visualizer()
    kwargs = dict(window_name=title, width=window_size[0],
                  height=window_size[1])
    if save_path:
        kwargs["visible"] = False
    vis.create_window(**kwargs)

    opt = vis.get_render_option()
    if opt:
        opt.point_size = point_size
        opt.background_color = np.array([0.1, 0.1, 0.1])

    for g in geometries:
        if g is not None:
            vis.add_geometry(g)

    ctr = vis.get_view_control()
    ctr.set_front([0.5, -0.3, -0.8])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.7)

    if save_path:
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path, do_render=True)
        print(f"  → Saved: {save_path}")
    else:
        vis.run()
    vis.destroy_window()



def process_frame(
    dataset_dir: str,
    frame: str,
    voxel_size: float,
    dilation: int,
    enable_gui: bool,
    save_dir: Optional[str],
    road_label: int = _ROAD_LABEL_ID,
    ego_range: Optional[list] = None,
):
    print(f"\n{'=' * 60}")
    print(f"  Frame: {frame}")
    print(f"{'=' * 60}")

    if ego_range is None:
        ego_range = _EGO_CAV_LIDAR_RANGE

    print("\n  Loading data …")
    ego_pc, ego_labels, ego_pose = load_cav_data(dataset_dir, "0", frame)
    coop_pc, coop_labels, coop_pose = load_cav_data(dataset_dir, "1", frame)

    ego_has_labels = (ego_labels is not None and len(ego_labels) == ego_pc.shape[0])
    coop_has_labels = (coop_labels is not None and len(coop_labels) == coop_pc.shape[0])

    print(f"  Ego:   {ego_pc.shape[0]:,} points  "
          f"labels={'✓' if ego_has_labels else '—'}")
    print(f"  Coop:  {coop_pc.shape[0]:,} points  "
          f"labels={'✓' if coop_has_labels else '—'}")

    if not ego_has_labels and not coop_has_labels:
        print("  ❌ Neither vehicle has semantic labels — nothing to compare.")
        return

    print("\n  Extracting road from semantic labels …")
    ego_road = extract_road(ego_pc, ego_labels, road_label)
    coop_road = extract_road(coop_pc, coop_labels, road_label)

    ego_total = ego_road.shape[0] if ego_road is not None else 0
    coop_total = coop_road.shape[0] if coop_road is not None else 0
    print(f"  Ego road:   {ego_total:>7,} points" if ego_road is not None else "  Ego road:   None (no road in labels)")
    print(f"  Coop road:  {coop_total:>7,} points" if coop_road is not None else "  Coop road:  None (no road in labels)")

    if coop_road is None and ego_road is None:
        print("  ❌ No road points from either vehicle — nothing to compare.")
        return

    T = compute_transform(coop_pose, ego_pose)
    if coop_road is not None:
        coop_road = transform_points(coop_road, T)
    coop_pc_xyz = transform_points(coop_pc[:, :3], T)


    print(f"\n  Classifying (voxel={voxel_size}m, dilation={dilation}) …")
    print(f"  Ego perception range: X [{ego_range[0]}, {ego_range[3]}] m, "
          f"Y [{ego_range[1]}, {ego_range[4]}] m, "
          f"Z [{ego_range[2]}, {ego_range[5]}] m")
    categories = classify_three_way(
        ego_road, coop_road, voxel_size, dilation,
        ego_range=ego_range,
    )

    _print_summary(categories, ego_total, coop_total, ego_range=ego_range)

    if enable_gui:
        ego_ctx = _make_pcd_xyz(ego_pc[:, :3], color=(0.4, 0.4, 0.4))
        coop_ctx = _make_pcd_xyz(coop_pc_xyz, color=(0.4, 0.4, 0.4))
        merge_ctx_parts = []
        if ego_ctx is not None:
            merge_ctx_parts.append(ego_ctx)
        if coop_ctx is not None:
            merge_ctx_parts.append(coop_ctx)
        merge_ctx = merge_ctx_parts if merge_ctx_parts else None

        merged_parts = []
        if ego_road is not None:
            merged_parts.append(ego_road)
        if coop_road is not None:
            merged_parts.append(coop_road)
        merged_road = np.vstack(merged_parts) if merged_parts else None

        views = [
            ("① Ego road",       _VIS_COLOURS["ego_only"],  ego_road,  [g for g in [ego_ctx] if g is not None]),
            ("② Coop road",      (0.8, 0.2, 0.2),             coop_road, [g for g in [coop_ctx] if g is not None]),
            ("③ Merged road",    (1.0, 0.8, 0.0),           merged_road, merge_ctx or []),
        ]

        for label, color, pts, bg in views:
            geom = list(bg)
            if pts is not None and pts.shape[0] > 0:
                g = _make_pcd_xyz(pts, color=color)
                if g is not None:
                    geom.append(g)
            sp = None
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                sp = os.path.join(save_dir, f"{frame}_{label[0]}.png")
            _visualize(f"Frame {frame} — {label}", geom, save_path=sp)

        geom = [g for g in [ego_ctx] if g is not None]
        for key, rgb in _VIS_COLOURS.items():
            arr = categories.get(key)
            if arr is not None and arr.shape[0] > 0:
                g = _make_pcd_xyz(arr, color=rgb)
                if g is not None:
                    geom.append(g)
        sp = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            sp = os.path.join(save_dir, f"{frame}_④_three_way.png")
        _visualize(f"Frame {frame} — ④ Three-way partition", geom, save_path=sp)



def main():
    parser = argparse.ArgumentParser(
        description="Three-way road perception partition for V2V scenarios.")
    parser.add_argument("--dataset_dir", type=str,
                        default=_DEFAULT_DATASET_DIR)
    parser.add_argument("--frame", type=str, default=None,
                        help="Single frame (e.g. 000001). Default: all.")
    parser.add_argument("--gui", action="store_true",
                        help="Enable Open3D visualisation")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Screenshot dir (implies --gui)")
    parser.add_argument("--voxel_size", type=float, default=_VOXEL_SIZE,
                        help="Voxel size in meters for overlap detection "
                             f"(default: {_VOXEL_SIZE})")
    parser.add_argument("--dilation", type=int, default=_DILATION_RADIUS,
                        help=f"Voxel dilation radius for overlap detection "
                             f"(default: {_DILATION_RADIUS})")
    parser.add_argument("--road_label", type=int, default=_ROAD_LABEL_ID,
                        help="Semantic label id for road "
                             f"(default: {_ROAD_LABEL_ID} = SemanticKITTI road)")
    parser.add_argument("--ego_range", type=float, nargs=6,
                        default=_EGO_CAV_LIDAR_RANGE,
                        help="Ego perception range [min_x, min_y, min_z, "
                             "max_x, max_y, max_z] in meters "
                             f"(default: {_EGO_CAV_LIDAR_RANGE})")
    args = parser.parse_args()

    enable_gui = args.gui or args.save_dir is not None
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    ego_range = args.ego_range
    print(f"Dataset: {args.dataset_dir}")
    print(f"Overlap: voxel={args.voxel_size}m  dilation={args.dilation}")
    print(f"Ego range: {ego_range}")

    for cam in ["0", "1"]:
        d = os.path.join(args.dataset_dir, cam, "velodyne")
        if not os.path.isdir(d):
            print(f"❌ Missing {d}")
            sys.exit(1)

    velo_0 = os.path.join(args.dataset_dir, "0", "velodyne")
    available = sorted(f.replace(".bin", "")
                       for f in os.listdir(velo_0) if f.endswith(".bin"))
    if not available:
        print("❌ No .bin files found")
        sys.exit(1)

    if args.frame is not None:
        if args.frame not in available:
            print(f"❌ Frame {args.frame} not found. "
                  f"Available: {available[:5]}…")
            sys.exit(1)
        frames = [args.frame]
    else:
        frames = available

    print(f"Frames to process: {len(frames)}")

    for frame in frames:
        process_frame(
            dataset_dir=args.dataset_dir,
            frame=frame,
            voxel_size=args.voxel_size,
            dilation=args.dilation,
            enable_gui=enable_gui,
            save_dir=args.save_dir,
            road_label=args.road_label,
            ego_range=ego_range,
        )

    print(f"\n{'=' * 60}")
    print("  Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
