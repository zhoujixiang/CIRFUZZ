
import argparse
import copy
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _ROOT.parent
_V2V4REAL_ROOT = _REPO_ROOT / "V2V4Real"

sys.path.insert(0, str(_ROOT))

from v2x_ops import delete_object, insert_object, load_point_cloud 
from v2x_outside_ego import (  
    _DILATION_RADIUS,
    _EGO_CAV_LIDAR_RANGE,
    _ROAD_LABEL_ID,
    _VIS_COLOURS,
    _VOXEL_SIZE,
    classify_three_way,
    compute_transform,
    transform_points,
)


DEFAULT_DATASET_DIR = _ROOT / "dataset" / "v2x_dataset"
DEFAULT_OUTPUT_DIR = _ROOT / "fuzz_out"
DEFAULT_INFERENCE_SCRIPT = (
    _V2V4REAL_ROOT / "opencood" / "tools" / "inference.py"
)


@dataclass(frozen=True)
class FrameRef:
    dataset_format: str
    root: Path
    scenario: str
    frame: str


@dataclass(frozen=True)
class FramePaths:
    ego_pc: Path
    coop_pc: Path
    ego_yaml: Path
    coop_yaml: Path
    ego_label: Optional[Path]
    coop_label: Optional[Path]


def _yaml_loader():
    loader = yaml.Loader
    loader.add_implicit_resolver(
        u"tag:yaml.org,2002:float",
        re.compile(u"""^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""", re.X),
        list(u"-+0123456789."),
    )
    return loader


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.load(f, Loader=_yaml_loader())


def save_yaml(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, sort_keys=False)


def load_labels(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None or not path.exists():
        return None
    return np.fromfile(path, dtype=np.uint32).reshape(-1)


def normalize_pc(pc: np.ndarray) -> np.ndarray:
    pc = np.asarray(pc, dtype=np.float32)
    if pc.ndim != 2 or pc.shape[1] < 3:
        raise ValueError(f"point cloud must be Nx3 or Nx4, got {pc.shape}")
    if pc.shape[1] == 3:
        pc = np.hstack([pc, np.zeros((pc.shape[0], 1), dtype=np.float32)])
    return pc[:, :4].astype(np.float32, copy=False)


def write_pcd_xyzi(path: Path, pc: np.ndarray):
    import open3d as o3d

    pc = normalize_pc(pc)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
    intensity = np.clip(pc[:, 3:4], 0.0, 1.0)
    pcd.colors = o3d.utility.Vector3dVector(
        np.repeat(intensity, 3, axis=1)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), pcd):
        raise IOError(f"failed to write point cloud: {path}")


@contextmanager
def working_directory(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def detect_dataset_format(dataset_dir: Path) -> str:
    if (dataset_dir / "0" / "yaml").is_dir() and (
        dataset_dir / "1" / "yaml"
    ).is_dir():
        return "v2xgen_flat"

    scenario_dirs = [
        p for p in dataset_dir.iterdir() if p.is_dir()
    ] if dataset_dir.exists() else []
    for scenario in scenario_dirs:
        if (scenario / "0").is_dir() and (scenario / "1").is_dir():
            return "opencood"

    raise ValueError(
        f"cannot detect dataset format for {dataset_dir}; expected either "
        "0/yaml + 1/yaml or scenario/0 + scenario/1"
    )


def _pc_path_for_flat(root: Path, cav_id: str, frame: str) -> Path:
    bin_path = root / cav_id / "velodyne" / f"{frame}.bin"
    if bin_path.exists():
        return bin_path
    pcd_path = root / cav_id / "pcd" / f"{frame}.pcd"
    if pcd_path.exists():
        return pcd_path
    raise FileNotFoundError(f"missing point cloud for CAV {cav_id}/{frame}")


def _label_path_for_flat(root: Path, cav_id: str, frame: str) -> Optional[Path]:
    path = root / cav_id / "predictions" / f"{frame}.label"
    return path if path.exists() else None


def _label_path_for_opencood(
    scenario_dir: Path, cav_id: str, frame: str
) -> Optional[Path]:
    candidates = [
        scenario_dir / cav_id / "predictions" / f"{frame}.label",
        scenario_dir / cav_id / f"{frame}.label",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def list_frames(
    dataset_dir: Path,
    dataset_format: str,
    ego_id: str,
    coop_id: str,
) -> List[FrameRef]:
    frames: List[FrameRef] = []
    if dataset_format == "v2xgen_flat":
        ego_yaml_dir = dataset_dir / ego_id / "yaml"
        coop_yaml_dir = dataset_dir / coop_id / "yaml"
        ego_frames = {p.stem for p in ego_yaml_dir.glob("*.yaml")}
        coop_frames = {p.stem for p in coop_yaml_dir.glob("*.yaml")}
        for frame in sorted(ego_frames & coop_frames):
            frames.append(FrameRef(dataset_format, dataset_dir, "", frame))
        return frames

    for scenario_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        ego_dir = scenario_dir / ego_id
        coop_dir = scenario_dir / coop_id
        if not ego_dir.is_dir() or not coop_dir.is_dir():
            continue
        ego_frames = {p.stem for p in ego_dir.glob("*.yaml")}
        coop_frames = {p.stem for p in coop_dir.glob("*.yaml")}
        for frame in sorted(ego_frames & coop_frames):
            frames.append(
                FrameRef(dataset_format, dataset_dir, scenario_dir.name, frame)
            )
    return frames


def filter_frames(frames: List[FrameRef], selected: str) -> List[FrameRef]:
    if not selected:
        return frames

    wanted = {item.strip() for item in selected.split(",") if item.strip()}
    out = []
    for ref in frames:
        aliases = {
            ref.frame,
            f"{ref.scenario}/{ref.frame}" if ref.scenario else ref.frame,
            f"{ref.scenario}:{ref.frame}" if ref.scenario else ref.frame,
        }
        if wanted & aliases:
            out.append(ref)
    return out


def frame_paths(ref: FrameRef, ego_id: str, coop_id: str) -> FramePaths:
    if ref.dataset_format == "v2xgen_flat":
        return FramePaths(
            ego_pc=_pc_path_for_flat(ref.root, ego_id, ref.frame),
            coop_pc=_pc_path_for_flat(ref.root, coop_id, ref.frame),
            ego_yaml=ref.root / ego_id / "yaml" / f"{ref.frame}.yaml",
            coop_yaml=ref.root / coop_id / "yaml" / f"{ref.frame}.yaml",
            ego_label=_label_path_for_flat(ref.root, ego_id, ref.frame),
            coop_label=_label_path_for_flat(ref.root, coop_id, ref.frame),
        )

    scenario_dir = ref.root / ref.scenario
    return FramePaths(
        ego_pc=scenario_dir / ego_id / f"{ref.frame}.pcd",
        coop_pc=scenario_dir / coop_id / f"{ref.frame}.pcd",
        ego_yaml=scenario_dir / ego_id / f"{ref.frame}.yaml",
        coop_yaml=scenario_dir / coop_id / f"{ref.frame}.yaml",
        ego_label=_label_path_for_opencood(scenario_dir, ego_id, ref.frame),
        coop_label=_label_path_for_opencood(scenario_dir, coop_id, ref.frame),
    )


def extract_road_points(
    pc: np.ndarray,
    labels: Optional[np.ndarray],
    road_label: int,
    road_source: str,
) -> Optional[np.ndarray]:
    if labels is not None and labels.shape[0] == pc.shape[0]:
        semantic = labels.reshape(-1) & np.uint32(0xFFFF)
        mask = semantic == np.uint32(road_label)
        if np.any(mask):
            return pc[mask, :3].astype(np.float32, copy=True)

    if road_source == "labels":
        return None

    z = pc[:, 2]
    low = np.percentile(z, 15)
    high = np.percentile(z, 45)
    mask = (z >= low - 0.35) & (z <= high + 0.15)
    ground = pc[mask, :3]
    if ground.shape[0] < 32:
        return None
    return ground.astype(np.float32, copy=True)


def non_road_points(
    pc: np.ndarray,
    labels: Optional[np.ndarray],
    road_label: int,
) -> Optional[np.ndarray]:
    if labels is None or labels.shape[0] != pc.shape[0]:
        return None
    semantic = labels.reshape(-1) & np.uint32(0xFFFF)
    return pc[semantic != np.uint32(road_label), :3].astype(np.float32)


def point_within_range(point: Sequence[float], range6: Sequence[float]) -> bool:
    return (
        range6[0] <= point[0] <= range6[3]
        and range6[1] <= point[1] <= range6[4]
        and range6[2] <= point[2] <= range6[5]
    )


def vehicle_items(vehicles: Optional[dict]) -> Iterable[Tuple[object, dict]]:
    if not vehicles:
        return []
    return list(vehicles.items())


def vehicle_center(vehicle: dict) -> np.ndarray:
    location = np.asarray(vehicle.get("location", [0.0, 0.0, 0.0]),
                          dtype=np.float64)
    center = np.asarray(vehicle.get("center", [0.0, 0.0, 0.0]),
                        dtype=np.float64)
    return location + center


def vehicle_extent(vehicle: dict) -> np.ndarray:
    return np.asarray(vehicle.get("extent", [2.25, 0.9, 0.75]),
                      dtype=np.float64)


def vehicle_yaw(vehicle: dict) -> float:
    angle = vehicle.get("angle", [0.0, 0.0, 0.0])
    if len(angle) >= 2:
        return float(angle[1])
    return 0.0


def vehicle_to_corners(vehicle: dict) -> np.ndarray:
    center = vehicle_center(vehicle)
    half = vehicle_extent(vehicle)
    local = np.array([
        [-half[0], -half[1], -half[2]],
        [ half[0], -half[1], -half[2]],
        [ half[0],  half[1], -half[2]],
        [-half[0],  half[1], -half[2]],
        [-half[0], -half[1],  half[2]],
        [ half[0], -half[1],  half[2]],
        [ half[0],  half[1],  half[2]],
        [-half[0],  half[1],  half[2]],
    ], dtype=np.float64)
    yaw = np.radians(vehicle_yaw(vehicle))
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return (local @ rot.T + center).astype(np.float32)


def vehicle_corners_from_yaml(params: dict) -> List[np.ndarray]:
    return [vehicle_to_corners(v) for _, v in vehicle_items(params.get("vehicles"))]


def max_vehicle_key(vehicles: Optional[dict]) -> int:
    max_key = 0
    for key, _ in vehicle_items(vehicles):
        try:
            max_key = max(max_key, int(key))
        except (TypeError, ValueError):
            continue
    return max_key


def max_ass_id(*params_list: dict) -> int:
    max_id = 0
    for params in params_list:
        for _, vehicle in vehicle_items(params.get("vehicles")):
            try:
                max_id = max(max_id, int(vehicle.get("ass_id", -1)))
            except (TypeError, ValueError):
                continue
    return max_id


def make_inserted_vehicle(result: dict, ass_id: int) -> dict:
    center = np.asarray(result["center"], dtype=np.float64)
    extent = np.asarray(result["extent"], dtype=np.float64)
    return {
        "angle": [0.0, float(result.get("rz_degree", 0.0)), 0.0],
        "ass_id": int(ass_id),
        "center": [0, 0, 0],
        "extent": [float(v) for v in extent.tolist()],
        "location": [float(v) for v in center.tolist()],
        "obj_type": "Car",
    }


def transform_yaw_between_frames(
    yaw_deg: float,
    source_pose: np.ndarray,
    target_pose: np.ndarray,
) -> float:
    theta = np.radians(yaw_deg)
    direction = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float64)
    t_source_to_target = compute_transform(source_pose, target_pose)
    transformed = t_source_to_target[:3, :3] @ direction
    return float(np.degrees(np.arctan2(transformed[1], transformed[0])))


def delete_vehicle_from_yaml(params: dict, vehicle_key: object):
    vehicles = params.get("vehicles") or {}
    if vehicle_key in vehicles:
        del vehicles[vehicle_key]
        params["vehicles"] = vehicles
        return

    key_str = str(vehicle_key)
    for key in list(vehicles.keys()):
        if str(key) == key_str:
            del vehicles[key]
            break
    params["vehicles"] = vehicles


def find_matching_vehicle_key(
    vehicles: Optional[dict],
    target_center: Sequence[float],
    preferred_ass_id: Optional[int] = None,
    max_distance: float = 6.0,
) -> Optional[object]:
    if not vehicles:
        return None

    if preferred_ass_id is not None:
        try:
            preferred_ass_id = int(preferred_ass_id)
        except (TypeError, ValueError):
            preferred_ass_id = None
    if preferred_ass_id is not None and preferred_ass_id != -1:
        for key, vehicle in vehicle_items(vehicles):
            try:
                if int(vehicle.get("ass_id", -1)) == preferred_ass_id:
                    return key
            except (TypeError, ValueError):
                continue

    target = np.asarray(target_center, dtype=np.float64)
    best_key = None
    best_dist = float("inf")
    for key, vehicle in vehicle_items(vehicles):
        dist = float(np.linalg.norm(vehicle_center(vehicle) - target))
        if dist < best_dist:
            best_dist = dist
            best_key = key
    return best_key if best_dist <= max_distance else None


def find_mesh(mesh_path: Optional[str]) -> Optional[Path]:
    if mesh_path:
        path = Path(mesh_path).expanduser().resolve()
        return path if path.exists() else None

    assets_dir = _ROOT / "_assets"
    found = sorted(assets_dir.glob("**/*.gltf")) if assets_dir.exists() else []
    if found:
        return found[0]
    return None


def road_region_candidates(
    ego_road: np.ndarray,
    coop_road: np.ndarray,
    ego_pose: np.ndarray,
    coop_pose: np.ndarray,
    ego_range: Sequence[float],
    coop_range: Sequence[float],
    voxel_size: float,
    dilation: int,
    region: str = "coop_only_outside",
) -> Tuple[np.ndarray, np.ndarray]:
    if ego_road is None or coop_road is None or len(coop_road) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    t_coop_to_ego = compute_transform(coop_pose, ego_pose)
    t_ego_to_coop = np.linalg.inv(t_coop_to_ego)
    coop_road_ego = transform_points(coop_road, t_coop_to_ego)
    categories = classify_three_way(
        ego_road,
        coop_road_ego,
        voxel_size=voxel_size,
        dilation=dilation,
        ego_range=list(ego_range),
    )
    if region == "overlap":
        candidates_ego = categories.get("overlap", np.zeros((0, 3), dtype=np.float32))
    elif region == "ego_only":
        candidates_ego = categories.get(
            "ego_only", np.zeros((0, 3), dtype=np.float32)
        )
    elif region == "coop_only_outside":
        candidates_ego = categories.get(
            "coop_only_outside", np.zeros((0, 3), dtype=np.float32)
        )
    else:
        raise ValueError(f"unsupported mutation region: {region}")
    if candidates_ego.shape[0] == 0:
        return candidates_ego, np.zeros((0, 3), dtype=np.float32)

    candidates_coop = transform_points(candidates_ego, t_ego_to_coop)
    coop_mask = np.array(
        [point_within_range(p, coop_range) for p in candidates_coop],
        dtype=bool,
    )
    return candidates_ego[coop_mask], candidates_coop[coop_mask]


def outside_ego_road_candidates(
    ego_road: np.ndarray,
    coop_road: np.ndarray,
    ego_pose: np.ndarray,
    coop_pose: np.ndarray,
    ego_range: Sequence[float],
    coop_range: Sequence[float],
    voxel_size: float,
    dilation: int,
) -> Tuple[np.ndarray, np.ndarray]:
    return road_region_candidates(
        ego_road,
        coop_road,
        ego_pose,
        coop_pose,
        ego_range,
        coop_range,
        voxel_size,
        dilation,
        region="coop_only_outside",
    )


def deletion_candidates(
    coop_params: dict,
    ego_pose: np.ndarray,
    coop_pose: np.ndarray,
    ego_range: Sequence[float],
    coop_range: Sequence[float],
    region: str = "coop_only_outside",
    ego_params: Optional[dict] = None,
    ego_road: Optional[np.ndarray] = None,
    coop_road: Optional[np.ndarray] = None,
    voxel_size: float = _VOXEL_SIZE,
    dilation: int = _DILATION_RADIUS,
) -> List[dict]:
    ego_side_only = region == "ego_only"
    if ego_side_only:
        vehicles = ((ego_params or {}).get("vehicles") or {}) if ego_params is not None else {}
    else:
        vehicles = coop_params.get("vehicles") or {}
    if not vehicles:
        return []

    t_coop_to_ego = compute_transform(coop_pose, ego_pose)
    t_ego_to_coop = np.linalg.inv(t_coop_to_ego)
    ego_vehicles = (ego_params or {}).get("vehicles") if ego_params is not None else None
    overlap_voxels = None
    if region in ("overlap", "ego_only"):
        if ego_road is None or coop_road is None:
            return []
        if region == "overlap" and not ego_vehicles:
            return []
        coop_road_ego = transform_points(coop_road, t_coop_to_ego)
        categories = classify_three_way(
            ego_road,
            coop_road_ego,
            voxel_size=voxel_size,
            dilation=dilation,
            ego_range=list(ego_range),
        )
        overlap_voxels = {
            tuple(np.floor(p[:3] / voxel_size).astype(np.int64))
            for p in np.asarray(categories.get("overlap", np.zeros((0, 3))))
        }
        if region == "overlap" and not overlap_voxels:
            return []

    out = []
    for key, vehicle in vehicle_items(vehicles):
        if ego_side_only:
            center_ego = vehicle_center(vehicle)
            if not point_within_range(center_ego, ego_range):
                continue
            center_coop = transform_points(center_ego.reshape(1, 3), t_ego_to_coop)[0]
            center_voxel = tuple(
                np.floor(center_ego[:3] / voxel_size).astype(np.int64)
            )
            if center_voxel in overlap_voxels:
                continue
            out.append({
                "vehicle_key": key,
                "vehicle": vehicle,
                "center_coop": center_coop.astype(float).tolist(),
                "center_ego": center_ego.astype(float).tolist(),
                "corner": vehicle_to_corners(vehicle),
                "ego_side_only": True,
            })
            continue

        center_coop = vehicle_center(vehicle)
        if not point_within_range(center_coop, coop_range):
            continue
        center_ego = transform_points(center_coop.reshape(1, 3), t_coop_to_ego)[0]
        in_ego = point_within_range(center_ego, ego_range)
        if region == "coop_only_outside":
            if in_ego:
                continue
        elif region == "overlap":
            center_voxel = tuple(
                np.floor(center_ego[:3] / voxel_size).astype(np.int64)
            )
            if center_voxel not in overlap_voxels:
                continue
            ego_key = find_matching_vehicle_key(
                ego_vehicles,
                center_ego,
                preferred_ass_id=vehicle.get("ass_id"),
            )
            if ego_key is None:
                continue
        else:
            raise ValueError(f"unsupported mutation region: {region}")
        entry = {
            "vehicle_key": key,
            "vehicle": vehicle,
            "center_coop": center_coop.astype(float).tolist(),
            "center_ego": center_ego.astype(float).tolist(),
            "corner": vehicle_to_corners(vehicle),
        }
        if region == "overlap":
            ego_vehicle = ego_vehicles[ego_key]
            entry.update({
                "ego_vehicle_key": ego_key,
                "ego_vehicle": ego_vehicle,
                "ego_corner": vehicle_to_corners(ego_vehicle),
            })
        out.append(entry)
    return out


def _cluster_road_regions(
    points_xy: np.ndarray, voxel_size: float = 4.0
) -> np.ndarray:
    n = points_xy.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)

    gx = np.floor(points_xy[:, 0] / voxel_size).astype(np.int64)
    gy = np.floor(points_xy[:, 1] / voxel_size).astype(np.int64)

    cell_to_idx: dict = {}
    for i in range(n):
        key = (int(gx[i]), int(gy[i]))
        if key not in cell_to_idx:
            cell_to_idx[key] = i

    cells = list(cell_to_idx.keys())
    cell_set = set(cells)

    labels = np.full(n, -1, dtype=np.int64)
    label_id = 0
    visited: set = set()

    for start in cells:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        comp_cells = []
        while stack:
            cell = stack.pop()
            comp_cells.append(cell)
            cx, cy = cell
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nb = (cx + dx, cy + dy)
                    if nb in cell_set and nb not in visited:
                        visited.add(nb)
                        stack.append(nb)

        for cell in comp_cells:
            mask = (gx == cell[0]) & (gy == cell[1])
            labels[mask] = label_id
        label_id += 1

    return labels


def _prioritized_order(
    points: np.ndarray, rng: np.random.Generator, voxel_size: float = 1.0
) -> np.ndarray:
    n = points.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)

    labels = _cluster_road_regions(points[:, :2], voxel_size)
    non_noise = labels >= 0

    if not np.any(non_noise):
        return rng.permutation(n)

    unique, counts = np.unique(labels[non_noise], return_counts=True)
    cluster_order = unique[np.argsort(-counts)]

    parts = []
    for cid in cluster_order:
        idx = np.where(labels == cid)[0]
        parts.append(idx[rng.permutation(len(idx))])

    noise_idx = np.where(~non_noise)[0]
    if len(noise_idx) > 0:
        parts.append(noise_idx[rng.permutation(len(noise_idx))])

    return np.concatenate(parts)


def load_frame_data(paths: FramePaths, road_label: int, road_source: str) -> dict:
    ego_pc = normalize_pc(load_point_cloud(str(paths.ego_pc)))
    coop_pc = normalize_pc(load_point_cloud(str(paths.coop_pc)))
    ego_labels = load_labels(paths.ego_label)
    coop_labels = load_labels(paths.coop_label)

    ego_params = load_yaml(paths.ego_yaml)
    coop_params = load_yaml(paths.coop_yaml)
    ego_pose = np.asarray(ego_params["lidar_pose"], dtype=np.float64)
    coop_pose = np.asarray(coop_params["lidar_pose"], dtype=np.float64)

    ego_road = extract_road_points(ego_pc, ego_labels, road_label, road_source)
    coop_road = extract_road_points(coop_pc, coop_labels, road_label, road_source)
    return {
        "ego_pc": ego_pc,
        "coop_pc": coop_pc,
        "ego_labels": ego_labels,
        "coop_labels": coop_labels,
        "ego_params": ego_params,
        "coop_params": coop_params,
        "ego_pose": ego_pose,
        "coop_pose": coop_pose,
        "ego_road": ego_road,
        "coop_road": coop_road,
        "ego_non_road": non_road_points(ego_pc, ego_labels, road_label),
        "coop_non_road": non_road_points(coop_pc, coop_labels, road_label),
    }


def save_case(
    case_dir: Path,
    timestamp: str,
    ego_id: str,
    coop_id: str,
    ego_pc: np.ndarray,
    coop_pc: np.ndarray,
    ego_params: dict,
    coop_params: dict,
):
    write_pcd_xyzi(case_dir / ego_id / f"{timestamp}.pcd", ego_pc)
    write_pcd_xyzi(case_dir / coop_id / f"{timestamp}.pcd", coop_pc)
    save_yaml(ego_params, case_dir / ego_id / f"{timestamp}.yaml")
    save_yaml(coop_params, case_dir / coop_id / f"{timestamp}.yaml")


def corner_lines(corner, color=(1, 0, 0)):
    import open3d as o3d

    corner = np.asarray(corner, dtype=np.float64)
    if corner.shape != (8, 3):
        raise ValueError(f"corner must be (8, 3), got {corner.shape}")

    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corner)
    ls.lines = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (12, 1)))
    return ls


def _pc_to_o3d(pc: np.ndarray, color=(0.75, 0.75, 0.75)):
    import open3d as o3d

    pc = normalize_pc(pc)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
    colors = np.tile(np.asarray(color, dtype=np.float64), (pc.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def save_case_visualization(
    save_path: Path,
    coop_pc: np.ndarray,
    boxes: Optional[Sequence[np.ndarray]] = None,
    yellow_points: Optional[np.ndarray] = None,
    title: str = "Case",
):
    _save_case_visualization_matplotlib(
        save_path, coop_pc, boxes, yellow_points, title
    )


def visualize(title, geometries, point_size=2.0, window_size=(1024, 768)):
    import open3d as o3d

    display = os.environ.get("DISPLAY")
    if not display or display.strip() == "":
        print(f"  [skipped — no display]  {title}")
        return

    print(f"\n  🖼  {title}")
    print("  └─ Drag to rotate | Scroll to zoom | Ctrl+C in terminal to exit\n")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=window_size[0], height=window_size[1])

    opt = vis.get_render_option()
    if opt is None:
        vis.destroy_window()
        return
    opt.point_size = point_size
    opt.background_color = np.array([0.1, 0.1, 0.1])

    for g in geometries:
        if g is not None:
            vis.add_geometry(g)

    vis.run()
    vis.destroy_window()


def _set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    plot_radius = 0.5 * max([x_range, y_range, z_range, 1e-6])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def _save_case_visualization_matplotlib(
    save_path: Path,
    coop_pc: np.ndarray,
    boxes: Optional[Sequence[np.ndarray]],
    yellow_points: Optional[np.ndarray],
    title: str,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    pc = normalize_pc(coop_pc)
    if pc.shape[0] > 40000:
        rng = np.random.default_rng(0)
        idx = rng.choice(pc.shape[0], 40000, replace=False)
        pc = pc[idx]

    fig = plt.figure(figsize=(10, 8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=0.15, c="#7d7d7d", alpha=0.9)

    if yellow_points is not None:
        yellow_points = np.asarray(yellow_points, dtype=np.float64)
        if yellow_points.ndim == 2 and yellow_points.shape[0] > 0:
            if yellow_points.shape[0] > 40000:
                rng = np.random.default_rng(0)
                idx = rng.choice(yellow_points.shape[0], 40000, replace=False)
                yellow_points = yellow_points[idx]
            ax.scatter(
                yellow_points[:, 0],
                yellow_points[:, 1],
                yellow_points[:, 2],
                s=0.35,
                c="#d9d12a",
                alpha=0.95,
            )

    if boxes:
        edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ])
        all_box_corners = []
        for box_corners in boxes:
            box_corners = np.asarray(box_corners, dtype=np.float64)
            all_box_corners.append(box_corners)
            segments = [(box_corners[i], box_corners[j]) for i, j in edges]
            ax.add_collection3d(
                Line3DCollection(segments, colors="red", linewidths=2.4)
            )
            center = box_corners.mean(axis=0)
            ghost = center + (box_corners - center) * 1.6
            ghost_segments = [(ghost[i], ghost[j]) for i, j in edges]
            ax.add_collection3d(
                Line3DCollection(
                    ghost_segments,
                    colors="magenta",
                    linewidths=1.2,
                    linestyles=(0, (4, 4)),
                    alpha=0.8,
                )
            )
        if len(all_box_corners) == 1:
            box_corners = all_box_corners[0]
            cmin = box_corners.min(axis=0) - 4.0
            cmax = box_corners.max(axis=0) + 4.0
            inset = fig.add_axes([0.66, 0.60, 0.30, 0.30], projection="3d")
            in_pc = pc[
                (pc[:, 0] >= cmin[0]) & (pc[:, 0] <= cmax[0])
                & (pc[:, 1] >= cmin[1]) & (pc[:, 1] <= cmax[1])
                & (pc[:, 2] >= cmin[2]) & (pc[:, 2] <= cmax[2])
            ]
            if in_pc.shape[0] > 8000:
                rng = np.random.default_rng(0)
                idx = rng.choice(in_pc.shape[0], 8000, replace=False)
                in_pc = in_pc[idx]
            inset.scatter(in_pc[:, 0], in_pc[:, 1], in_pc[:, 2],
                          s=0.5, c="#7d7d7d", alpha=0.9)
            segments = [(box_corners[i], box_corners[j]) for i, j in edges]
            inset.add_collection3d(
                Line3DCollection(segments, colors="red", linewidths=2.6)
            )
            inset.set_xlim(cmin[0], cmax[0])
            inset.set_ylim(cmin[1], cmax[1])
            inset.set_zlim(cmin[2], cmax[2])
            inset.set_box_aspect((cmax - cmin).tolist())
            inset.set_title("mutated vehicle", fontsize=8)
            inset.view_init(elev=30, azim=-60)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _set_axes_equal(ax)
    if boxes:
        fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.06)
    else:
        fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def build_insert_cases(
    ref: FrameRef,
    data: dict,
    args,
    rng: np.random.Generator,
    mesh_path: Path,
    count: int,
    state: Optional[dict] = None,
    candidate_points: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> List[dict]:
    del ref
    state = state or {}
    region = args.mutation_region
    ego_side_only = region == "ego_only"
    coop_pc = state.get("coop_pc", data["coop_pc"])
    ego_pc = state.get("ego_pc", data["ego_pc"])
    coop_params = copy.deepcopy(state.get("coop_params", data["coop_params"]))
    ego_params = copy.deepcopy(state.get("ego_params", data["ego_params"]))
    coop_vehicles = coop_params.setdefault("vehicles", {})
    ego_vehicles = ego_params.setdefault("vehicles", {})
    base_ass_id = max_ass_id(
        data["ego_params"], data["coop_params"], ego_params, coop_params
    )

    if candidate_points is None:
        cand_ego, cand_coop = road_region_candidates(
            data["ego_road"],
            data["coop_road"],
            data["ego_pose"],
            data["coop_pose"],
            args.ego_range,
            args.coop_range,
            args.voxel_size,
            args.dilation,
            region=region,
        )
    else:
        cand_ego, cand_coop = candidate_points
    if cand_coop.shape[0] == 0:
        return []

    coop_existing_corners = (
        vehicle_corners_from_yaml(coop_params)
        if not args.skip_existing_collision_check else None
    )
    ego_existing_corners = (
        vehicle_corners_from_yaml(ego_params)
        if not args.skip_existing_collision_check else None
    )

    order = _prioritized_order(cand_coop, rng, voxel_size=2.0)
    cases: List[dict] = []
    pointer = 0
    while len(cases) < count and pointer < args.max_attempts_per_iteration:
        if pointer >= len(order):
            break
        idx = int(order[pointer])
        pointer += 1
        pos_coop = cand_coop[idx].astype(float)
        pos_ego = cand_ego[idx].astype(float)

        this_coop_pc = coop_pc
        this_ego_pc = ego_pc
        this_coop_params = copy.deepcopy(coop_params)
        this_ego_params = copy.deepcopy(ego_params)
        this_coop_vehicles = this_coop_params.setdefault("vehicles", {})
        this_ego_vehicles = this_ego_params.setdefault("vehicles", {})

        if ego_side_only:
            with working_directory(_ROOT):
                ego_result = insert_object(
                    point_cloud=this_ego_pc,
                    road_pc=data["ego_road"],
                    position=pos_ego.tolist(),
                    mesh_path=str(mesh_path),
                    non_road_pc=data["ego_non_road"]
                    if args.check_non_road_overlap else None,
                    existing_vehicle_corners=ego_existing_corners,
                    rz_degree=None,
                )
            if not ego_result.get("success"):
                if args.verbose_failures:
                    print(
                        "insert attempt failed on ego side: "
                        f"{ego_result.get('reason', 'unknown')}"
                    )
                continue
            coop_result = None

            ego_vehicle_key = max_vehicle_key(this_ego_vehicles) + 1
            ass_id = base_ass_id + len(cases) + 1
            this_ego_vehicles[ego_vehicle_key] = make_inserted_vehicle(
                ego_result, ass_id=ass_id
            )
            this_ego_params["vehicles"] = this_ego_vehicles
            coop_vehicle_key = None

            cases.append({
                "operation": "insert",
                "coop_pc": this_coop_pc,
                "ego_pc": ego_result["point_cloud"],
                "ego_params": this_ego_params,
                "coop_params": this_coop_params,
                "viz_box": np.asarray(
                    ego_result.get(
                        "bbox_corners",
                        vehicle_to_corners(this_ego_vehicles[ego_vehicle_key]),
                    ),
                    dtype=np.float64,
                ),
                "viz_coords": "ego",
                "mutation": {
                    "vehicle_key": int(ego_vehicle_key)
                    if str(ego_vehicle_key).isdigit()
                    else ego_vehicle_key,
                    "ass_id": int(ass_id),
                    "candidate_center_ego": pos_ego.tolist(),
                    "candidate_center_coop": pos_coop.tolist(),
                    "inserted_center_ego": [
                        float(v) for v in np.asarray(ego_result["center"]).tolist()
                    ],
                    "inserted_extent": [
                        float(v) for v in np.asarray(ego_result["extent"]).tolist()
                    ],
                    "inserted_vehicle": copy.deepcopy(
                        this_ego_vehicles[ego_vehicle_key]
                    ),
                    "ego_inserted_vehicle": copy.deepcopy(
                        this_ego_vehicles[ego_vehicle_key]
                    ),
                    "ego_vehicle_key": int(ego_vehicle_key)
                    if str(ego_vehicle_key).isdigit()
                    else ego_vehicle_key,
                    "coop_vehicle_key": None,
                    "mutation_region": region,
                    "coop_rz_degree": None,
                    "ego_rz_degree": float(ego_result.get("rz_degree", 0.0)),
                    "ego_side_only": True,
                    "fallback": False,
                },
            })
            continue

        with working_directory(_ROOT):
            coop_result = insert_object(
                point_cloud=this_coop_pc,
                road_pc=data["coop_road"],
                position=pos_coop.tolist(),
                mesh_path=str(mesh_path),
                non_road_pc=data["coop_non_road"]
                if args.check_non_road_overlap else None,
                existing_vehicle_corners=coop_existing_corners,
                rz_degree=None,
            )
        if not coop_result.get("success"):
            if args.verbose_failures:
                print(
                    "insert attempt failed: "
                    f"{coop_result.get('reason', 'unknown')}"
                )
            continue

        ego_result = None
        if region == "overlap":
            ego_rz = transform_yaw_between_frames(
                float(coop_result.get("rz_degree", 0.0)),
                data["coop_pose"],
                data["ego_pose"],
            )
            ego_target_extent = (
                np.asarray(coop_result["extent"], dtype=np.float64) * 1.0
            ).tolist()
            with working_directory(_ROOT):
                ego_result = insert_object(
                    point_cloud=this_ego_pc,
                    road_pc=data["ego_road"],
                    position=pos_ego.tolist(),
                    mesh_path=str(mesh_path),
                    non_road_pc=data["ego_non_road"]
                    if args.check_non_road_overlap else None,
                    existing_vehicle_corners=ego_existing_corners,
                    rz_degree=ego_rz,
                    target_extent=ego_target_extent,
                )
            if not ego_result.get("success"):
                if args.verbose_failures:
                    print(
                        "insert attempt failed on ego side: "
                        f"{ego_result.get('reason', 'unknown')}"
                    )
                continue

        coop_vehicle_key = max_vehicle_key(this_coop_vehicles) + 1
        ass_id = base_ass_id + len(cases) + 1
        this_coop_vehicles[coop_vehicle_key] = make_inserted_vehicle(
            coop_result, ass_id=ass_id
        )
        this_coop_params["vehicles"] = this_coop_vehicles

        if region == "overlap" and ego_result is not None:
            ego_vehicle_key = max_vehicle_key(this_ego_vehicles) + 1
            this_ego_vehicles[ego_vehicle_key] = make_inserted_vehicle(
                ego_result, ass_id=ass_id
            )
            this_ego_params["vehicles"] = this_ego_vehicles
        else:
            ego_vehicle_key = None

        inserted_vehicle = copy.deepcopy(this_coop_vehicles[coop_vehicle_key])
        ego_inserted_vehicle = (
            copy.deepcopy(this_ego_vehicles[ego_vehicle_key])
            if region == "overlap" and ego_vehicle_key is not None
            else None
        )
        cases.append({
            "operation": "insert",
            "coop_pc": coop_result["point_cloud"],
            "ego_pc": ego_result["point_cloud"] if ego_result is not None else this_ego_pc,
            "ego_params": this_ego_params,
            "coop_params": this_coop_params,
            "viz_box": np.asarray(
                coop_result.get(
                    "bbox_corners",
                    vehicle_to_corners(this_coop_vehicles[coop_vehicle_key]),
                ),
                dtype=np.float64,
            ),
            "mutation": {
                "vehicle_key": int(coop_vehicle_key),
                "ass_id": int(ass_id),
                "candidate_center_ego": pos_ego.tolist(),
                "candidate_center_coop": pos_coop.tolist(),
                "inserted_center_coop": [
                    float(v) for v in np.asarray(coop_result["center"]).tolist()
                ],
                "inserted_extent": [
                    float(v) for v in np.asarray(coop_result["extent"]).tolist()
                ],
                "inserted_vehicle": inserted_vehicle,
                "ego_inserted_vehicle": ego_inserted_vehicle,
                "ego_vehicle_key": int(ego_vehicle_key)
                if ego_vehicle_key is not None and str(ego_vehicle_key).isdigit()
                else ego_vehicle_key,
                "coop_vehicle_key": int(coop_vehicle_key)
                if str(coop_vehicle_key).isdigit()
                else coop_vehicle_key,
                "mutation_region": region,
                "coop_rz_degree": float(coop_result.get("rz_degree", 0.0)),
                "ego_rz_degree": float(ego_result.get("rz_degree", 0.0))
                if ego_result is not None else None,
                "fallback": False,
            },
        })
    return cases


def build_delete_cases(
    ref: FrameRef,
    data: dict,
    args,
    rng: np.random.Generator,
    count: int,
    state: Optional[dict] = None,
) -> List[dict]:
    del ref
    state = state or {}
    region = args.mutation_region
    ego_side_only = region == "ego_only"
    sync_ego = region == "overlap"
    coop_pc = state.get("coop_pc", data["coop_pc"])
    ego_pc = state.get("ego_pc", data["ego_pc"])
    coop_params = copy.deepcopy(state.get("coop_params", data["coop_params"]))
    ego_params = copy.deepcopy(state.get("ego_params", data["ego_params"]))

    candidates = deletion_candidates(
        coop_params,
        data["ego_pose"],
        data["coop_pose"],
        args.ego_range,
        args.coop_range,
        region=region,
        ego_params=ego_params if sync_ego or ego_side_only else None,
        ego_road=data["ego_road"],
        coop_road=data["coop_road"],
        voxel_size=args.voxel_size,
        dilation=args.dilation,
    )

    if not candidates:
        return []

    order = rng.permutation(len(candidates))
    cases: List[dict] = []
    pointer = 0
    while len(cases) < count and pointer < args.max_attempts_per_iteration:
        if pointer >= len(order):
            break
        candidate = candidates[int(order[pointer])]
        pointer += 1
        current_yaml = ego_params if ego_side_only else coop_params
        if candidate["vehicle_key"] not in current_yaml.get("vehicles", {}):
            continue
        if sync_ego:
            if candidate.get("ass_id") == state.get("last_inserted_ass_id"):
                continue
        elif candidate["vehicle_key"] == state.get("last_inserted_key"):
            continue

        if ego_side_only:
            ego_result = delete_object(
                ego_pc,
                candidate["corner"],
                road_pc=data["ego_road"],
                fill_shadow=args.fill_delete_shadow,
            )
            if not ego_result.get("success"):
                if args.verbose_failures:
                    print(
                        "delete attempt failed on ego side: "
                        f"vehicle={candidate['vehicle_key']}"
                    )
                continue

            this_ego_params = copy.deepcopy(ego_params)
            delete_vehicle_from_yaml(this_ego_params, candidate["vehicle_key"])
            cases.append({
                "operation": "delete",
                "coop_pc": coop_pc,
                "ego_pc": ego_result["point_cloud"],
                "ego_params": this_ego_params,
                "coop_params": coop_params,
                "viz_box": np.asarray(candidate["corner"], dtype=np.float64),
                "viz_coords": "ego",
                "mutation": {
                    "vehicle_key": int(candidate["vehicle_key"])
                    if str(candidate["vehicle_key"]).isdigit()
                    else str(candidate["vehicle_key"]),
                    "ass_id": int(candidate["vehicle"].get("ass_id", -1)),
                    "deleted_center_ego": candidate["center_ego"],
                    "deleted_center_coop": candidate["center_coop"],
                    "deleted_vehicle": copy.deepcopy(candidate["vehicle"]),
                    "deleted_vehicle_corners": np.asarray(
                        candidate["corner"], dtype=np.float64
                    ).tolist(),
                    "ego_vehicle_key": int(candidate["vehicle_key"])
                    if str(candidate["vehicle_key"]).isdigit()
                    else candidate["vehicle_key"],
                    "ego_deleted_vehicle": copy.deepcopy(candidate["vehicle"]),
                    "ego_deleted_vehicle_corners": np.asarray(
                        candidate["corner"], dtype=np.float64
                    ).tolist(),
                    "mutation_region": region,
                    "ego_side_only": True,
                    "fallback": False,
                },
            })
            continue

        coop_result = delete_object(
            coop_pc,
            candidate["corner"],
            road_pc=data["coop_road"],
            fill_shadow=args.fill_delete_shadow,
        )
        if not coop_result.get("success"):
            if args.verbose_failures:
                print(
                    "delete attempt failed: "
                    f"vehicle={candidate['vehicle_key']}"
                )
            continue

        ego_result = None
        this_ego_params = copy.deepcopy(ego_params)
        if sync_ego:
            ego_result = delete_object(
                ego_pc,
                candidate["ego_corner"],
                road_pc=data["ego_road"],
                fill_shadow=args.fill_delete_shadow,
            )
            if not ego_result.get("success"):
                if args.verbose_failures:
                    print(
                        "delete attempt failed on ego side: "
                        f"vehicle={candidate['vehicle_key']}"
                    )
                continue

        this_coop_params = copy.deepcopy(coop_params)
        delete_vehicle_from_yaml(this_coop_params, candidate["vehicle_key"])
        if sync_ego:
            delete_vehicle_from_yaml(
                this_ego_params, candidate["ego_vehicle_key"]
            )
        cases.append({
            "operation": "delete",
            "coop_pc": coop_result["point_cloud"],
            "ego_pc": ego_result["point_cloud"] if ego_result is not None else ego_pc,
            "ego_params": this_ego_params,
            "coop_params": this_coop_params,
            "viz_box": np.asarray(candidate["corner"], dtype=np.float64),
            "mutation": {
                "vehicle_key": int(candidate["vehicle_key"])
                if str(candidate["vehicle_key"]).isdigit()
                else str(candidate["vehicle_key"]),
                "ass_id": int(candidate["vehicle"].get("ass_id", -1)),
                "deleted_center_ego": candidate["center_ego"],
                "deleted_center_coop": candidate["center_coop"],
                "deleted_vehicle": copy.deepcopy(candidate["vehicle"]),
                "deleted_vehicle_corners": np.asarray(
                    candidate["corner"], dtype=np.float64
                ).tolist(),
                "ego_vehicle_key": int(candidate["ego_vehicle_key"])
                if sync_ego and candidate.get("ego_vehicle_key") is not None
                and str(candidate["ego_vehicle_key"]).isdigit()
                else candidate.get("ego_vehicle_key"),
                "ego_deleted_vehicle": (
                    copy.deepcopy(candidate.get("ego_vehicle"))
                    if sync_ego
                    else None
                ),
                "ego_deleted_vehicle_corners": (
                    np.asarray(candidate.get("ego_corner"), dtype=np.float64).tolist()
                    if sync_ego and candidate.get("ego_corner") is not None
                    else None
                ),
                "mutation_region": region,
                "fallback": False,
            },
        })
    return cases


def choose_operation(args, rng: np.random.Generator, can_delete: bool = True) -> str:
    if args.operations == "insert":
        return "insert"
    if args.operations == "delete":
        return "delete"
    if not can_delete:
        return "insert"
    return "insert" if rng.random() < 0.5 else "delete"


def case_source_payload(ref: FrameRef, paths: FramePaths) -> dict:
    return {
        "dataset_format": ref.dataset_format,
        "scenario": ref.scenario,
        "frame": ref.frame,
        "ego_pc": str(paths.ego_pc),
        "coop_pc": str(paths.coop_pc),
        "ego_yaml": str(paths.ego_yaml),
        "coop_yaml": str(paths.coop_yaml),
        "ego_label": str(paths.ego_label) if paths.ego_label else "",
        "coop_label": str(paths.coop_label) if paths.coop_label else "",
    }


def enabled_processes(args) -> List[str]:
    if args.process == "both":
        return ["fuzz", "random"]
    return [args.process]


def _baseline_case_dir(
    output_dir: Path,
    iteration: int,
    ref: FrameRef,
    process: str = "fuzz",
) -> Path:
    if process == "fuzz":
        name = f"case_{iteration:06d}_baseline"
    else:
        name = f"case_{iteration:06d}_{process}_baseline"
    if ref.scenario:
        name += "_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", ref.scenario)
    name += f"_{ref.frame}"
    return output_dir / "baseline_cases" / name


def run_source_inference(
    iteration: int,
    ref: FrameRef,
    paths: FramePaths,
    data: dict,
    args,
    run_inference_fn,
    process: str = "fuzz",
) -> Optional[dict]:
    baseline_dir = _baseline_case_dir(
        Path(args.output_dir), iteration, ref, process=process
    )
    baseline_dir.mkdir(parents=True, exist_ok=True)
    save_case(
        baseline_dir,
        args.output_timestamp,
        args.ego_id,
        args.coop_id,
        data["ego_pc"],
        data["coop_pc"],
        data["ego_params"],
        data["coop_params"],
    )

    record = _case_metadata(
        copy.deepcopy({
            "operation": "baseline",
            "coop_pc": data["coop_pc"],
            "ego_params": data["ego_params"],
            "coop_params": data["coop_params"],
            "mutation": {},
        }),
        baseline_dir.name,
        baseline_dir,
        args.output_timestamp,
        paths,
        ref,
        iteration,
        0,
        args,
        0,
        None,
        None,
        process,
    )
    record["candidate_index"] = 0
    save_yaml(record, baseline_dir / "metadata.yaml")

    try:
        inference = run_inference_fn(args, [baseline_dir])
    except Exception as exc:
        print(
            f"  baseline inference failed "
            f"({type(exc).__name__}: {exc}); mutations will score without it"
        )
        return None

    annotated = annotate_cases_with_inference(
        [record], inference, args.fitness_threshold
    )[0]
    save_yaml(annotated, baseline_dir / "metadata.yaml")
    return annotated


def _baseline_payload_for_cases(baseline: Optional[dict]) -> Optional[dict]:
    if baseline is None:
        return None
    return {
        "fitness": baseline.get("fitness"),
        "interesting": bool(baseline.get("interesting", False)),
        "delta_v2x_minus_ego": (
            (baseline.get("inference") or {})
            .get("error_counts") or {}
        ).get("delta_v2x_minus_ego") or {},
    }


def _make_case_dir(
    output_dir: Path,
    iteration: int,
    step: int,
    operation: str,
    seed_index: int,
    ref: FrameRef,
    process: str = "fuzz",
) -> Path:
    if process == "fuzz":
        name = f"case_{iteration:06d}_step{step:02d}_{operation}_s{seed_index:02d}"
    else:
        name = (
            f"case_{iteration:06d}_{process}_step{step:02d}_"
            f"{operation}_s{seed_index:02d}"
        )
    if ref.scenario:
        name += "_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", ref.scenario)
    name += f"_{ref.frame}"
    return output_dir / "valid_cases" / name


def _case_metadata(
    case: dict,
    case_name: str,
    case_dir: Path,
    timestamp: str,
    paths: FramePaths,
    ref: FrameRef,
    iteration: int,
    step: int,
    args,
    seed_index: int,
    seed_fitness: Optional[float],
    candidate_yellow_coop: Optional[np.ndarray],
    process: str = "fuzz",
) -> dict:
    saved_visualizations: List[str] = []
    return {
        "case_index": iteration,
        "case_name": case_name,
        "case_dir": str(case_dir),
        "timestamp": timestamp,
        "process": process,
        "selection_strategy": "fitness" if process == "fuzz" else "random",
        "mutation_region": args.mutation_region,
        "operation": case["operation"],
        "step": int(step),
        "seed_index": int(seed_index),
        "seed_fitness": seed_fitness,
        "ops_per_case": int(args.ops_per_case),
        "candidates_per_step": int(args.candidates_per_step),
        "operations": case["operation"],
        "steps": [
            {
                "step": int(step),
                "operation": case["operation"],
                "mutation": case["mutation"],
            }
        ],
        "source": case_source_payload(ref, paths),
        "mutation": case["mutation"],
        "visualization": saved_visualizations[-1] if saved_visualizations else "",
        "visualizations": saved_visualizations,
        "fitness": None,
        "interesting": None,
        "_yellow_coop_count": int(
            np.asarray(candidate_yellow_coop).shape[0]
            if candidate_yellow_coop is not None
            else 0
        ),
    }


def evolve_one_frame(
    iteration: int,
    ref: FrameRef,
    args,
    rng: np.random.Generator,
    mesh_path: Optional[Path],
    run_inference_fn=None,
    process: str = "fuzz",
) -> Optional[dict]:
    paths = frame_paths(ref, args.ego_id, args.coop_id)
    data = load_frame_data(paths, args.road_label, args.road_source)
    setup_yellow_ego = None
    setup_yellow_coop = None
    need_visualization = args.save_visualizations or args.show_visualization
    if need_visualization:
        setup_yellow_ego, setup_yellow_coop = road_region_candidates(
            data["ego_road"],
            data["coop_road"],
            data["ego_pose"],
            data["coop_pose"],
            args.ego_range,
            args.coop_range,
            args.voxel_size,
            args.dilation,
            region=args.mutation_region,
        )
    candidate_points = (
        (setup_yellow_ego, setup_yellow_coop)
        if setup_yellow_ego is not None and setup_yellow_coop is not None
        else None
    )

    baseline = None
    if run_inference_fn is not None:
        baseline = run_source_inference(
            iteration, ref, paths, data, args, run_inference_fn, process=process
        )

    state = {
        "coop_pc": data["coop_pc"],
        "coop_params": copy.deepcopy(data["coop_params"]),
        "ego_pc": data["ego_pc"],
        "ego_params": copy.deepcopy(data["ego_params"]),
        "last_inserted_key": None,
        "last_inserted_ass_id": None,
        "last_viz_box": None,
        "last_viz_coords": None,
    }
    baseline_fitness = baseline.get("fitness") if baseline else None
    seed_fitness = baseline_fitness

    step = 0
    operation = None
    final_meta: Optional[dict] = None
    while step < args.ops_per_case:
        step += 1
        can_delete = len(
            deletion_candidates(
                state["coop_params"],
                data["ego_pose"],
                data["coop_pose"],
                args.ego_range,
                args.coop_range,
                region=args.mutation_region,
                ego_params=(
                    state["ego_params"]
                    if args.mutation_region in ("overlap", "ego_only")
                    else None
                ),
                ego_road=data["ego_road"],
                coop_road=data["coop_road"],
                voxel_size=args.voxel_size,
                dilation=args.dilation,
            )
        ) > 0
        operation = choose_operation(args, rng, can_delete=can_delete)
        cases: List[dict] = []
        if operation == "insert":
            if mesh_path is None or data["coop_road"] is None or data["ego_road"] is None:
                operation = "delete"
                cases = build_delete_cases(
                    ref, data, args, rng, args.candidates_per_step, state
                )
            else:
                cases = build_insert_cases(
                    ref,
                    data,
                    args,
                    rng,
                    mesh_path,
                    args.candidates_per_step,
                    state=state,
                    candidate_points=candidate_points,
                )
                if not cases and can_delete:
                    operation = "delete"
                    cases = build_delete_cases(
                        ref, data, args, rng, args.candidates_per_step, state
                    )
        else:
            cases = build_delete_cases(
                ref, data, args, rng, args.candidates_per_step, state
            )
            if not cases and mesh_path is not None \
                    and data["coop_road"] is not None and data["ego_road"] is not None:
                operation = "insert"
                cases = build_insert_cases(
                    ref,
                    data,
                    args,
                    rng,
                    mesh_path,
                    args.candidates_per_step,
                    state=state,
                    candidate_points=candidate_points,
                )

        if not cases:
            print(f"  step {step}: no feasible {operation} candidate — stopping")
            break

        step_records: List[dict] = []
        for seed_index, case in enumerate(cases):
            case["step"] = step
            case_name = _make_case_dir(
                Path(args.output_dir),
                iteration,
                step,
                case["operation"],
                seed_index,
                ref,
                process=process,
            ).name
            case_dir = Path(args.output_dir) / "valid_cases" / case_name
            case_dir.mkdir(parents=True, exist_ok=True)
            save_case(
                case_dir,
                args.output_timestamp,
                args.ego_id,
                args.coop_id,
                normalize_pc(case.get("ego_pc", data["ego_pc"])),
                normalize_pc(case["coop_pc"]),
                case["ego_params"],
                case["coop_params"],
            )
            saved_vis = []
            if need_visualization:
                vis_path = case_dir / (
                    f"{args.output_timestamp}_"
                    f"{case['operation']}_step{step}_vis.png"
                )
                viz_ego = case.get("viz_coords") == "ego"
                save_case_visualization(
                    vis_path,
                    case["ego_pc"] if viz_ego else case["coop_pc"],
                    boxes=[case["viz_box"]] if case.get("viz_box") is not None else None,
                    yellow_points=setup_yellow_ego if viz_ego else setup_yellow_coop,
                    title=f"{case['operation'].capitalize()} step {step}",
                )
                saved_vis.append(str(vis_path))
            meta = _case_metadata(
                case,
                case_name,
                case_dir,
                args.output_timestamp,
                paths,
                ref,
                iteration,
                step,
                args,
                seed_index,
                seed_fitness,
                setup_yellow_coop,
                process,
            )
            meta["candidate_index"] = seed_index
            if baseline is not None:
                meta["baseline"] = _baseline_payload_for_cases(baseline)
            if saved_vis:
                meta["visualization"] = saved_vis[-1]
                meta["visualizations"] = saved_vis
            save_yaml(meta, case_dir / "metadata.yaml")
            step_records.append(meta)

        step_inference = None
        if run_inference_fn is not None:
            try:
                step_inference = run_inference_fn(
                    args,
                    [Path(record["case_dir"]) for record in step_records],
                )
            except Exception as exc:
                print(
                    f"  step {step}: inference failed "
                    f"({type(exc).__name__}: {exc}); scoring skipped"
                )
            step_records = annotate_cases_with_inference(
                step_records,
                step_inference,
                args.fitness_threshold,
                baseline_fitness=baseline_fitness,
            )
            for record in step_records:
                save_yaml(record, Path(record["case_dir"]) / "metadata.yaml")

        if run_inference_fn is not None:
            copy_interesting_cases(step_records, Path(args.output_dir))

        if process == "random":
            best_idx = int(rng.integers(len(step_records)))
            best = step_records[best_idx]
            best["random_selected"] = True
        else:
            scored = [
                (i, record) for i, record in enumerate(step_records)
                if record.get("fitness") is not None
            ]
            if scored:
                scored.sort(key=lambda pair: float(pair[1]["fitness"]), reverse=True)
                best_idx, best = scored[0]
            else:
                best = step_records[0]
                best_idx = step_records.index(best)
            best["fitness_selected"] = True
            if (
                best.get("fitness") is not None
                and seed_fitness is not None
                and float(best["fitness"]) <= float(seed_fitness)
            ):
                best["fitness_selected"] = False
                best["rejected_non_improving"] = True
                save_yaml(best, Path(best["case_dir"]) / "metadata.yaml")
                print(
                    f"  step {step}: {len(step_records)} candidates -> "
                    f"best={best['case_name']} fitness={best['fitness']} "
                    f"did not improve seed_fitness={seed_fitness}; "
                    "keeping previous seed"
                )
                continue
        best_case = cases[best_idx]

        case_dir = Path(best["case_dir"])
        save_yaml(best, case_dir / "metadata.yaml")
        final_meta = best

        state = {
            "coop_pc": best_case["coop_pc"],
            "coop_params": copy.deepcopy(best_case["coop_params"]),
            "ego_pc": best_case.get("ego_pc", data["ego_pc"]),
            "ego_params": copy.deepcopy(best_case.get("ego_params", data["ego_params"])),
            "last_inserted_key": (
                best_case["mutation"].get("coop_vehicle_key", best_case["mutation"].get("vehicle_key"))
                if best_case["operation"] == "insert"
                else None
            ),
            "last_inserted_ass_id": (
                int(best_case["mutation"].get("ass_id", -1))
                if best_case["operation"] == "insert"
                else None
            ),
            "last_viz_box": (
                np.asarray(best_case.get("viz_box"), dtype=np.float64)
                if best_case.get("viz_box") is not None
                else None
            ),
            "last_viz_coords": (
                best_case.get("viz_coords", "coop")
                if best_case.get("viz_box") is not None
                else None
            ),
        }
        seed_fitness = best["fitness"]
        selection_value = (
            "random"
            if process == "random"
            else f"fitness={best['fitness']}"
        )
        print(
            f"  step {step}: {len(step_records)} candidates → "
            f"selected={best['case_name']} {selection_value}"
        )

        if args.show_visualization:
            _visualize_evolution_step(
                data, setup_yellow_ego, candidate_points, state, step, operation
            )

    if final_meta is None:
        return None
    final_meta["selected_final_case"] = True
    final_meta["operations"] = final_meta["operation"]
    save_yaml(final_meta, Path(final_meta["case_dir"]) / "metadata.yaml")
    return final_meta


def _visualize_evolution_step(
    data: dict,
    setup_yellow_ego: Optional[np.ndarray],
    candidate_points,
    state: dict,
    step: int,
    operation: str,
):
    t_coop_to_ego = compute_transform(data["coop_pose"], data["ego_pose"])

    def _to_ego(xyz_or_pc):
        if xyz_or_pc is None or xyz_or_pc.size == 0:
            return None
        return transform_points(xyz_or_pc[:, :3], t_coop_to_ego)

    geom = [
        _pc_to_o3d(state["ego_pc"], color=_VIS_COLOURS["ego_only"]),
        _pc_to_o3d(_to_ego(state["coop_pc"]),
                   color=_VIS_COLOURS["coop_only_inside"]),
    ]
    if candidate_points is not None and np.asarray(setup_yellow_ego).size > 0:
        geom.append(_pc_to_o3d(setup_yellow_ego,
                    color=_VIS_COLOURS["coop_only_outside"]))
    viz_box = state.get("last_viz_box")
    if viz_box is not None:
        viz_coords = state.get("last_viz_coords", "coop")
        geom.append(corner_lines(
            viz_box if viz_coords == "ego" else _to_ego(viz_box),
            color=(0.0, 1.0, 0.0),
        ))
    visualize(
        f"Step {step} seed ({operation}) — blue=ego red=coop yellow=candidates "
        f"green=bbox({'insert' if operation == 'insert' else 'delete'})",
        geom,
    )


def prepare_eval_model_dir(model_dir: Path, eval_dir: Path, validate_dir: Path):
    model_dir = model_dir.resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)

    config = load_yaml(model_dir / "config.yaml")
    config["validate_dir"] = str(validate_dir.resolve())
    if "test_dir" in config:
        config["test_dir"] = str(validate_dir.resolve())
    save_yaml(config, eval_dir / "config.yaml")

    for src in sorted(model_dir.glob("*epoch*.pth")):
        dst = eval_dir / src.name
        if dst.exists():
            continue
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def run_inference(args, case_dirs: Sequence[Path]) -> dict:
    eval_model_dir = Path(args.output_dir) / "inference_model"
    staging_dir = Path(args.output_dir) / "inference_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    for case_dir in sorted(case_dirs):
        dst = staging_dir / case_dir.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(case_dir, dst)
    prepare_eval_model_dir(
        Path(args.model_dir), eval_model_dir, staging_dir
    )

    cmd = [
        sys.executable,
        str(Path(args.inference_script).resolve()),
        "--model_dir",
        str(eval_model_dir.resolve()),
        "--fusion_method",
        args.fusion_method,
        "--compare_ego_v2x",
        "--log_every",
        str(args.log_every),
    ]
    if args.is_sim:
        cmd.append("--isSim")

    env = os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    completed = subprocess.run(
        cmd,
        cwd=str(_V2V4REAL_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    log_dir = Path(args.output_dir) / "inference_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "stdout.txt").write_text(completed.stdout)
    (log_dir / "stderr.txt").write_text(completed.stderr)

    output_dir = eval_model_dir / "compare_ego_v2x"
    results = {
        "returncode": int(completed.returncode),
        "eval_model_dir": str(eval_model_dir),
        "compare_dir": str(output_dir),
        "summary": None,
        "different_cases": [],
        "frame_errors": [],
    }
    summary_path = output_dir / "compare_error_diff.yaml"
    cases_path = output_dir / "different_cases.yaml"
    frame_errors_path = output_dir / "frame_errors.yaml"
    if summary_path.exists():
        results["summary"] = load_yaml(summary_path)
    if cases_path.exists():
        cases = load_yaml(cases_path)
        results["different_cases"] = cases if cases is not None else []
    if frame_errors_path.exists():
        fe = load_yaml(frame_errors_path)
        results["frame_errors"] = fe if fe is not None else []
    return results


def compute_fitness(case_meta: dict, inference_results: Optional[dict]) -> Optional[float]:
    inference = case_meta.get("inference") or {}
    if not inference:
        return None
    error_counts = (inference.get("error_counts") or {}).get("delta_v2x_minus_ego") or {}
    return float(
        int(error_counts.get("object_missing", 0))
        + int(error_counts.get("localization_failure", 0))
    )


def annotate_cases_with_inference(
    case_records: List[dict],
    inference_results: Optional[dict],
    fitness_threshold: float,
    baseline_fitness: Optional[float] = None,
) -> List[dict]:
    if not inference_results:
        for record in case_records:
            record["fitness"] = compute_fitness(record, None)
            record["interesting"] = False
        return case_records

    diff_by_frame = {
        int(case.get("frame_index", -1)): case
        for case in inference_results.get("different_cases", [])
    }
    error_by_frame = {}
    for frame_payload in inference_results.get("frame_errors", []):
        fidx = int(frame_payload.get("frame_index", -1))
        error_by_frame[fidx] = frame_payload

    for record in case_records:
        fidx = int(record.get("candidate_index", -1))
        frame_errors = error_by_frame.get(fidx)
        record["inference"] = {
            "frame_index": fidx,
            "has_ego_v2x_difference": fidx in diff_by_frame,
            "difference_case": diff_by_frame.get(fidx),
            "error_counts": frame_errors,
        }
        fitness = compute_fitness(record, inference_results)
        record["fitness"] = fitness
        if fitness is None:
            record["interesting"] = fidx in diff_by_frame
        elif baseline_fitness is not None:
            record["interesting"] = float(fitness) > float(baseline_fitness)
        else:
            record["interesting"] = float(fitness) >= fitness_threshold
    return case_records


def copy_interesting_cases(case_records: List[dict], output_dir: Path):
    interesting_root = output_dir / "interesting_cases"
    interesting_root.mkdir(parents=True, exist_ok=True)
    for record in case_records:
        if not record.get("interesting"):
            continue
        src = Path(record["case_dir"])
        dst = interesting_root / src.name
        if dst.exists():
            record["interesting_case_dir"] = str(dst)
            continue
        shutil.copytree(src, dst)
        record["interesting_case_dir"] = str(dst)


def rewrite_case_metadata(case_records: List[dict]):
    for record in case_records:
        save_yaml(record, Path(record["case_dir"]) / "metadata.yaml")
        interesting_dir = record.get("interesting_case_dir")
        if interesting_dir:
            save_yaml(record, Path(interesting_dir) / "metadata.yaml")


def save_manifest(
    output_dir: Path,
    args,
    case_records: List[dict],
    inference_results: Optional[dict],
):
    manifest = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "dataset_format": args.dataset_format,
        "process": args.process,
        "enabled_processes": enabled_processes(args),
        "mutation_region": args.mutation_region,
        "iterations": int(args.iterations),
        "ops_per_case": int(args.ops_per_case),
        "candidates_per_step": int(args.candidates_per_step),
        "valid_case_count": int(len(case_records)),
        "interesting_case_count": int(
            sum(1 for record in case_records if record.get("interesting"))
        ),
        "valid_cases_dir": str((output_dir / "valid_cases").resolve()),
        "baseline_cases_dir": str((output_dir / "baseline_cases").resolve()),
        "interesting_cases_dir": str((output_dir / "interesting_cases").resolve()),
        "run_inference": bool(args.run_inference),
        "inference_results": inference_results,
        "cases": case_records,
    }
    save_yaml(manifest, output_dir / "fuzz_manifest.yaml")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuzz V2X perception by mutating coop road objects in a selected region."
    )
    parser.add_argument("--model-dir", required=True,
                        help="OpenCOOD model/log directory under test.")
    parser.add_argument("--iterations", type=int, required=True,
                        help="Number of fuzzing iterations.")
    parser.add_argument("--process", choices=["fuzz", "random", "both"],
                        default="fuzz",
                        help="Process to run: fitness-guided fuzzing, random "
                             "candidate selection, or both. In both mode each "
                             "selected source frame is processed once by each "
                             "strategy. (default: fuzz)")
    parser.add_argument("--ops-per-case", type=int, default=1,
                        help="Number of beam-search mutation steps applied to "
                             "each selected scene. Every step expands the "
                             "current scene into --candidates-per-step distinct "
                             "offspring, optionally runs the model on them, and "
                             "keeps either the best candidate (fuzz) or a random "
                             "candidate (random) as the next step's seed. Each step "
                             "picks insert or delete with probability 0.5; if "
                             "the chosen operation is infeasible the other is "
                             "tried. (default: 1, i.e. single step per case)")
    parser.add_argument("--candidates-per-step", type=int, default=5,
                        help="Number of DISTINCT offspring generated at every "
                             "mutation step of a selected scene. For an insert "
                             "step this is the number of different vehicle "
                             "placement positions; for a delete step it is the "
                             "number of different vehicles removed. All "
                             "candidates of a step are saved and optionally "
                             "inferred; fuzz mode keeps the highest-fitness "
                             "candidate, random mode samples a candidate. "
                             "(default: 5)")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR),
                        help="Input V2X dataset directory.")
    parser.add_argument("--dataset-format", default="auto",
                        choices=["auto", "v2xgen_flat", "opencood"],
                        help="Input dataset layout.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory for generated fuzzing cases.")
    parser.add_argument("--frames", default="",
                        help="Comma-separated frame filter. For OpenCOOD use "
                             "scenario/frame or scenario:frame.")
    parser.add_argument("--ego-id", default="0")
    parser.add_argument("--coop-id", default="1")
    parser.add_argument("--operations", choices=["insert", "delete", "both"],
                        default="both")
    parser.add_argument("--insert-prob", type=float, default=0.5,
                        help="Kept for compatibility; --operations both uses 0.5/0.5.")
    parser.add_argument("--mesh-path", default=None,
                        help="Vehicle mesh path for insertion.")
    parser.add_argument("--road-source", choices=["auto", "labels"],
                        default="auto",
                        help="Use semantic labels when available; auto falls "
                             "back to a ground heuristic.")
    parser.add_argument("--road-label", type=int, default=_ROAD_LABEL_ID)
    parser.add_argument("--ego-range", type=float, nargs=6,
                        default=list(_EGO_CAV_LIDAR_RANGE))
    parser.add_argument("--coop-range", type=float, nargs=6,
                        default=list(_EGO_CAV_LIDAR_RANGE))
    parser.add_argument("--voxel-size", type=float, default=_VOXEL_SIZE)
    parser.add_argument("--dilation", type=int, default=_DILATION_RADIUS)
    parser.add_argument("--mutation-region",
                        choices=["coop_only_outside", "overlap", "ego_only"],
                        default="coop_only_outside",
                        help="Road region used to generate insertion/deletion "
                             "candidates. Use 'overlap' to place both insertions "
                             "and deletions in the ego/coop overlap region, or "
                             "'ego_only' to mutate ONLY the ego-side files in the "
                             "region only the ego lidar can see (the coop side is "
                             "left untouched).")
    parser.add_argument("--max-attempts-per-iteration", type=int, default=20)
    parser.add_argument("--fill-delete-shadow", action="store_true",
                        help="Fill deleted vehicle shadow with sampled road points.")
    parser.add_argument("--check-non-road-overlap", action="store_true",
                        help="For insertion, reject candidates whose inserted "
                             "box overlaps coop non-road points. This is strict "
                             "and is off by default for outside-ego fuzzing.")
    parser.add_argument("--skip-existing-collision-check", action="store_true",
                        help="For insertion, do not reject candidates that "
                             "collide with or are occluded by existing YAML "
                             "vehicles. Use only for aggressive fuzzing.")
    parser.add_argument("--output-timestamp", default="000000",
                        help="Timestamp name used inside each generated case.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose-failures", action="store_true",
                        help="Print mutation failure reasons during search.")
    parser.add_argument("--save-visualizations", action="store_true",
                        help="Save a 3D matplotlib visualization image for each valid "
                             "candidate.  Also enables pre-computation of candidate "
                             "points passed to the insert builder.")
    parser.add_argument("--show-visualization", action="store_true",
                        help="Pop up an interactive Open3D window during fuzzing "
                             "(requires DISPLAY / X11 forwarding).  Independent of "
                             "--save-visualizations.")

    parser.add_argument("--run-inference", action="store_true",
                        help="Run OpenCOOD inference.py on generated cases.")
    parser.add_argument("--inference-script", default=str(DEFAULT_INFERENCE_SCRIPT))
    parser.add_argument("--fusion-method", default="early",
                        choices=["late", "early", "intermediate", "nofusion"])
    parser.add_argument("--fitness-threshold", type=float, default=0.0)
    parser.add_argument("--is-sim", action="store_true",
                        help="Pass --isSim to inference.py.")
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    args.dataset_dir = str(Path(args.dataset_dir).expanduser().resolve())
    args.output_dir = str(Path(args.output_dir).expanduser().resolve())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(args.dataset_dir)
    if args.dataset_format == "auto":
        args.dataset_format = detect_dataset_format(dataset_dir)

    all_frames = list_frames(dataset_dir, args.dataset_format,
                             args.ego_id, args.coop_id)
    frames = filter_frames(all_frames, args.frames)
    if not frames:
        raise RuntimeError("no frames available for fuzzing")

    mesh_path = find_mesh(args.mesh_path)
    if args.operations in ("insert", "both") and mesh_path is None:
        print("warning: no mesh found; insertion iterations will be skipped")

    rng = np.random.default_rng(args.seed)
    run_inference_fn = run_inference if args.run_inference else None

    processes = enabled_processes(args)
    case_records: List[dict] = []
    for iteration in range(args.iterations):
        ref = frames[int(rng.integers(len(frames)))]
        for process in processes:
            print(
                f"[{iteration + 1}/{args.iterations}] {process} "
                f"{ref.scenario or ''}/{ref.frame}"
            )
            try:
                case = evolve_one_frame(
                    iteration,
                    ref,
                    args,
                    rng,
                    mesh_path,
                    run_inference_fn=run_inference_fn,
                    process=process,
                )
            except Exception as exc:
                print(
                    f"[{iteration + 1}/{args.iterations}] {process} skipped "
                    f"{ref.scenario}/{ref.frame}: {type(exc).__name__}: {exc}"
                )
                continue

            if case is None:
                print(
                    f"[{iteration + 1}/{args.iterations}] "
                    f"{process} no valid mutation"
                )
                continue

            case_records.append(case)
            print(
                f"[{iteration + 1}/{args.iterations}] {process} final case: "
                f"{case['case_name']} fitness={case.get('fitness')}"
            )

    copy_interesting_cases(case_records, output_dir)
    rewrite_case_metadata(case_records)
    save_manifest(output_dir, args, case_records, None)
    print(
        f"done: {len(case_records)} selected cases, "
        f"{sum(1 for c in case_records if c.get('interesting'))} interesting cases"
    )
    print(f"manifest: {output_dir / 'fuzz_manifest.yaml'}")


if __name__ == "__main__":
    main()
