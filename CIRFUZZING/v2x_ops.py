
import copy
import math
import os
import sys
from typing import Dict, List, Optional, Union

import numpy as np
import open3d as o3d

from utils.common_utils import pc_numpy_2_o3d, pcd_to_np, get_obj_pcd_from_corner
from core.lidar_simulation import lidar_simulation, lidar_intensity_convert


_DEFAULT_LABEL_CFG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "third", "SalsaNext", "train", "tasks", "semantic", "pretrained", "data_cfg.yaml",
)


def _load_label_config(config_path: Optional[str] = None):
    import yaml

    path = config_path or _DEFAULT_LABEL_CFG
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _label_ids_by_name(cfg, target_names: List[str]) -> List[int]:
    wanted = {name.lower() for name in target_names}
    out = []
    for key, value in cfg.get("labels", {}).items():
        if str(value).lower() in wanted:
            out.append(int(key))
    return out


def get_label_groups(config_path: Optional[str] = None):
    cfg = _load_label_config(config_path)
    return {
        "road": _label_ids_by_name(cfg, ["road"]),
        "other_road": _label_ids_by_name(cfg, ["parking", "sidewalk", "rail track"]),
        "other_ground": _label_ids_by_name(cfg, ["terrain", "ground", "other-ground"]),
    }


def _split_indices_by_label_groups(
    labels: np.ndarray,
    label_groups: Optional[Dict[str, List[int]]] = None,
):
    groups = label_groups or get_label_groups()
    flat = labels.reshape(-1)

    road_ids = set(groups.get("road", []))
    other_road_ids = set(groups.get("other_road", []))
    other_ground_ids = set(groups.get("other_ground", []))

    inx_road = np.flatnonzero(np.isin(flat, list(road_ids))).tolist()
    inx_other_road = np.flatnonzero(np.isin(flat, list(other_road_ids))).tolist()
    inx_other_ground = np.flatnonzero(np.isin(flat, list(other_ground_ids))).tolist()

    known_ids = road_ids | other_road_ids | other_ground_ids
    if known_ids:
        inx_no_road = np.flatnonzero(~np.isin(flat, list(known_ids))).tolist()
    else:
        inx_no_road = list(range(flat.shape[0]))

    return inx_road, inx_other_road, inx_other_ground, inx_no_road



def load_point_cloud(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".bin":
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    elif ext == ".pcd":
        pcd = o3d.io.read_point_cloud(path)
        xyz = np.asarray(pcd.points, dtype=np.float32)
        intensity = np.full((xyz.shape[0], 1), 0.0, dtype=np.float32)
        if pcd.has_colors():
            intensity = np.asarray(pcd.colors)[:, :1].astype(np.float32)
        return np.concatenate([xyz, intensity], axis=1)
    else:
        raise ValueError(f"Unsupported format: {ext}  (use .bin or .pcd)")


def load_labels(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint32).reshape(-1, 1)



def _load_mesh(path: str) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty():
        raise ValueError(f"Failed to load mesh from {path}")
    return mesh


def _mesh_half_height(mesh: o3d.geometry.TriangleMesh) -> float:
    mn, mx = mesh.get_min_bound(), mesh.get_max_bound()
    return (mx[2] - mn[2]) / 2.0


def _mesh_half_diagonal_xy(mesh: o3d.geometry.TriangleMesh) -> float:
    mn, mx = mesh.get_min_bound(), mesh.get_max_bound()
    return math.sqrt((mx[0] - mn[0]) ** 2 + (mx[1] - mn[1]) ** 2) / 2.0


def _transform_mesh(
    mesh: o3d.geometry.TriangleMesh,
    position: List[float],
    rz_degree: float,
) -> o3d.geometry.TriangleMesh:
    m = copy.deepcopy(mesh)
    theta = np.radians(rz_degree)
    R = m.get_rotation_matrix_from_xyz((0.0, 0.0, theta))
    m.rotate(R, center=m.get_center())
    m.translate(np.array(position, dtype=np.float64))
    return m


def _corners_half_diags_centers(
    corners_list: List[np.ndarray],
) -> (np.ndarray, np.ndarray):
    n = len(corners_list)
    diags = np.empty(n, dtype=np.float64)
    cents = np.empty((n, 2), dtype=np.float64)
    for i, c in enumerate(corners_list):
        lo, hi = np.min(c, axis=0), np.max(c, axis=0)
        diags[i] = math.sqrt((hi[0] - lo[0]) ** 2 + (hi[1] - lo[1]) ** 2) / 2.0
        cents[i] = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0]
    return diags, cents


def _corners_extents(corners_list: List[np.ndarray]) -> np.ndarray:
    extents = []
    for c in corners_list:
        lo, hi = np.min(c, axis=0), np.max(c, axis=0)
        extents.append(hi - lo)
    return np.asarray(extents, dtype=np.float64)


def _normalize_box_yaw(yaw_deg: float) -> float:
    if yaw_deg > 90.0:
        yaw_deg -= 180.0
    elif yaw_deg <= -90.0:
        yaw_deg += 180.0
    return float(yaw_deg)


def _corner_yaw_deg(corner: np.ndarray) -> float:
    corner = np.asarray(corner, dtype=np.float64)
    if corner.ndim != 2 or corner.shape[0] < 2 or corner.shape[1] < 2:
        return 0.0

    xy = corner[:, :2]
    centered = xy - np.mean(xy, axis=0, keepdims=True)
    cov = centered.T @ centered / max(xy.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, int(np.argmax(eigvals))]
    yaw_deg = float(np.degrees(np.arctan2(principal[1], principal[0])))
    return _normalize_box_yaw(yaw_deg)


def _nearest_vehicle_yaw_at(xy, corners_list):
    if not corners_list:
        return 0.0, -1, None

    center_xy = []
    for c in corners_list:
        c = np.asarray(c, dtype=np.float64)
        center_xy.append(np.mean(c[:, :2], axis=0))
    center_xy = np.asarray(center_xy, dtype=np.float64)

    xy = np.asarray(xy[:2], dtype=np.float64)
    d2 = np.sum((center_xy - xy) ** 2, axis=1)
    idx = int(np.argmin(d2))
    yaw_deg = _corner_yaw_deg(corners_list[idx])
    return yaw_deg, idx, center_xy[idx]


def _scale_mesh_to_extent(
    mesh: o3d.geometry.TriangleMesh,
    target_extent: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    m = copy.deepcopy(mesh)
    obb = m.get_minimal_oriented_bounding_box()
    src_extent = np.asarray(obb.extent, dtype=np.float64)
    if np.any(src_extent <= 1e-6):
        return m

    scale = np.asarray(target_extent, dtype=np.float64) / src_extent
    center = np.asarray(obb.center, dtype=np.float64)
    vertices = np.asarray(m.vertices, dtype=np.float64)
    vertices = (vertices - center) * scale + center
    m.vertices = o3d.utility.Vector3dVector(vertices)
    return m


def _ordered_bbox_corners(obb: o3d.geometry.OrientedBoundingBox) -> np.ndarray:
    center = np.asarray(obb.center, dtype=np.float64)
    rot = np.asarray(obb.R, dtype=np.float64)
    half = np.asarray(obb.extent, dtype=np.float64) / 2.0

    local_corners = np.array([
        [-half[0], -half[1], -half[2]],
        [ half[0], -half[1], -half[2]],
        [ half[0],  half[1], -half[2]],
        [-half[0],  half[1], -half[2]],
        [-half[0], -half[1],  half[2]],
        [ half[0], -half[1],  half[2]],
        [ half[0],  half[1],  half[2]],
        [-half[0],  half[1],  half[2]],
    ], dtype=np.float64)

    return local_corners @ rot.T + center



_center_cam = np.array([0.0, 0.0, 0.0], dtype=np.float64)


def _lidar_shadow_mesh(mesh: o3d.geometry.TriangleMesh):
    hull = mesh.compute_convex_hull()[0]
    verts = np.asarray(hull.vertices)
    shadow_pts = []
    for p in verts:
        ba = _center_cam - p
        bl = math.sqrt(np.dot(ba, ba))
        if bl < 1e-12:
            continue
        shadow_pts.append(_center_cam + (-200.0) * (ba / bl))
    if not shadow_pts:
        return hull
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack([verts, shadow_pts]))
    return pcd.compute_convex_hull()[0]


def _points_in_mesh_mask(pts: np.ndarray, mesh) -> np.ndarray:
    tri = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tri)
    occ = scene.compute_occupancy(
        o3d.core.Tensor(pts, dtype=o3d.core.Dtype.Float32)
    ).numpy()
    return occ.ravel() == 1.0


def get_shadow_mask(
    obj: Union[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud, np.ndarray],
    bg: Union[o3d.geometry.PointCloud, np.ndarray],
    high: float = 0,
):
    global _center_cam
    _center_cam[2] = high

    if isinstance(obj, np.ndarray):
        mesh = pc_numpy_2_o3d(obj).compute_convex_hull()[0]
    elif isinstance(obj, o3d.geometry.PointCloud):
        mesh = obj.compute_convex_hull()[0]
    elif isinstance(obj, o3d.geometry.TriangleMesh):
        mesh = obj
    else:
        raise TypeError(f"unsupported obj type: {type(obj)}")

    shadow = _lidar_shadow_mesh(mesh)

    if isinstance(bg, o3d.geometry.PointCloud):
        pts = np.asarray(bg.points)
    else:
        pts = np.asarray(bg)

    return _points_in_mesh_mask(pts, shadow), shadow



def segment_road(
    point_cloud: np.ndarray,
    semantic_labels: np.ndarray,
    denoise: bool = False,
    sample_points: int = 50000,
    min_road_points: int = 10,
    near_radius: float = 4.0,
    config_path: Optional[str] = None,
) -> Dict[str, Union[np.ndarray, None]]:
    if point_cloud.ndim < 2 or point_cloud.shape[1] < 3:
        raise ValueError(f"point_cloud shape {point_cloud.shape} — need (N, 3+)")
    xyz = point_cloud[:, :3]

    if semantic_labels.ndim == 1:
        labels_2d = semantic_labels.reshape(-1, 1)
    elif semantic_labels.shape[1] == 1:
        labels_2d = semantic_labels
    else:
        labels_2d = semantic_labels.reshape(-1, 1)
    labels_out = semantic_labels.ravel()

    label_groups = get_label_groups(config_path)
    inx_road, inx_other_road, inx_other_ground, inx_no_road = _split_indices_by_label_groups(
        labels_2d, label_groups
    )
    inx_non_road = inx_other_road + inx_other_ground + inx_no_road

    if len(inx_road) <= min_road_points:
        return {"road_pc": None, "non_road_pc": xyz[inx_non_road], "labels": labels_out}

    _pc_road = xyz[inx_road]

    if denoise:
        pcd = pc_numpy_2_o3d(_pc_road)
        _, ind = pcd.remove_radius_outlier(nb_points=7, radius=1.0)
        pcd = pcd.select_by_index(ind)

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=10)
        dense = mesh.sample_points_uniformly(number_of_points=sample_points)
        road_xyz = np.asarray(dense.points, dtype=np.float32)
        road_xyz = road_xyz[np.linalg.norm(road_xyz, axis=1) > near_radius]
    else:
        road_xyz = _pc_road.astype(np.float32)

    return {
        "road_pc": road_xyz,
        "non_road_pc": xyz[inx_non_road],
        "labels": labels_out,
    }



def _road_z_at(xy, road_pc):
    d2 = (road_pc[:, 0] - xy[0]) ** 2 + (road_pc[:, 1] - xy[1]) ** 2
    return float(road_pc[np.argmin(d2), 2])


def _road_yaw_at(
    xy,
    road_pc,
    neighborhood_radius: float = 6.0,
    min_points: int = 20,
):
    if road_pc is None or len(road_pc) == 0:
        return 0.0

    road_xy = road_pc[:, :2]
    center = np.asarray(xy[:2], dtype=np.float64)
    d2 = np.sum((road_xy - center) ** 2, axis=1)

    mask = d2 <= neighborhood_radius ** 2
    local_xy = road_xy[mask]

    if local_xy.shape[0] < min_points:
        k = min(max(min_points, 1), road_xy.shape[0])
        nearest_idx = np.argpartition(d2, k - 1)[:k]
        local_xy = road_xy[nearest_idx]

    if local_xy.shape[0] < 2:
        return 0.0

    centered = local_xy - np.mean(local_xy, axis=0, keepdims=True)
    cov = centered.T @ centered / max(local_xy.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, int(np.argmax(eigvals))]

    yaw_deg = float(np.degrees(np.arctan2(principal[1], principal[0])))
    return _normalize_box_yaw(yaw_deg)


def _on_road(mesh_obj, road_pc, non_road_pc, position, threshold=0.1):
    box = mesh_obj.get_oriented_bounding_box()
    inside = pc_numpy_2_o3d(non_road_pc).crop(box)
    if len(inside.points) >= 3:
        return False
    d2 = (road_pc[:, 0] - position[0]) ** 2 + (road_pc[:, 1] - position[1]) ** 2
    return bool(np.min(d2) <= threshold ** 2)


def _collision_free(xy, half_diag, diags, cents, n_initial):
    for k in range(n_initial):
        yd = abs(cents[k][1] - xy[1])
        d0 = math.sqrt(cents[k][0] ** 2 + cents[k][1] ** 2)
        d1 = math.sqrt(xy[0] ** 2 + xy[1] ** 2)
        if d1 > d0 and yd < diags[k] + half_diag - 2:
            return False
    for k in range(len(diags)):
        xd = cents[k][0] - xy[0]
        yd = cents[k][1] - xy[1]
        if math.sqrt(xd * xd + yd * yd) <= diags[k] + half_diag:
            return False
    return True


def _occlusion_rates(obj_pcd, point_cloud, vehicle_corners=None):
    if not vehicle_corners:
        return {}
    bg = pc_numpy_2_o3d(point_cloud)
    rates = {}
    for idx, corner in enumerate(vehicle_corners):
        vpcd = get_obj_pcd_from_corner(corner, bg)
        try:
            tetra, pm = o3d.geometry.TetraMesh.create_from_point_cloud(vpcd)
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                vpcd, alpha=10, tetra_mesh=tetra, pt_map=pm,
            )
        except Exception:
            rates[idx] = 0.0
            continue
        mask, _ = get_shadow_mask(mesh, obj_pcd)
        rates[idx] = float(np.sum(mask)) / max(mask.size, 1)
    return rates


def _merge_clouds(bg_pc, obj_pcd, mesh, high=0):
    bg = pc_numpy_2_o3d(bg_pc)
    mask, _ = get_shadow_mask(mesh, bg, high)
    bg.points = o3d.utility.Vector3dVector(np.asarray(bg.points)[~mask])
    bg.colors = o3d.utility.Vector3dVector(np.asarray(bg.colors)[~mask])
    return pcd_to_np(bg + obj_pcd)


def insert_object(
    point_cloud: np.ndarray,
    road_pc: np.ndarray,
    position: List[float],
    mesh_path: str,
    non_road_pc: Optional[np.ndarray] = None,
    existing_vehicle_corners: Optional[List[np.ndarray]] = None,
    rz_degree: Optional[float] = None,
    gt_box: Optional[np.ndarray] = None,
    target_extent: Optional[List[float]] = None,
    occlusion_threshold: float = 0.9,
) -> Dict:
    pos = list(position)
    if len(pos) == 2:
        pos.append(_road_z_at(pos, road_pc))
    pos = list(pos)

    mesh_init = _load_mesh(mesh_path)
    if mesh_init.is_empty():
        return {
            "success": False,
            "point_cloud": point_cloud,
            "reason": "mesh_empty",
        }
    if len(np.asarray(mesh_init.vertices)) == 0 or len(np.asarray(mesh_init.triangles)) == 0:
        return {
            "success": False,
            "point_cloud": point_cloud,
            "reason": "mesh_has_no_geometry",
        }

    corners = existing_vehicle_corners
    diags = np.empty((0,))
    cents = np.empty((0, 2))
    if corners and len(corners) > 0:
        diags, cents = _corners_half_diags_centers(corners)

    if gt_box is not None:
        lo, hi = np.min(gt_box, axis=0), np.max(gt_box, axis=0)
        mesh_init = _scale_mesh_to_extent(mesh_init, hi - lo)
    else:
        if target_extent is None:
            if corners and len(corners) > 0:
                target_extent = np.median(_corners_extents(corners), axis=0).tolist()
            else:
                target_extent = [4.5, 1.8, 1.5]
        mesh_init = _scale_mesh_to_extent(mesh_init, np.asarray(target_extent, dtype=np.float64))

    if rz_degree is None:
        if corners and len(corners) > 0:
            rz_degree, ref_idx, ref_center = _nearest_vehicle_yaw_at(pos, corners)
            print(
                f"Inferred yaw {rz_degree:.1f}° at position {pos[:2]} "
                f"based on nearest existing vehicle #{ref_idx} at "
                f"({ref_center[0]:.2f}, {ref_center[1]:.2f})."
            )
        else:
            rz_degree = _road_yaw_at(pos, road_pc)
            print(f"Inferred yaw {rz_degree:.1f}° at position {pos[:2]} based on local road heading.")
    half_h = _mesh_half_height(mesh_init)
    pos[2] += half_h
    mesh = _transform_mesh(mesh_init, pos, rz_degree)

    if non_road_pc is not None and not _on_road(mesh, road_pc, non_road_pc, pos):
        return {
            "success": False,
            "point_cloud": point_cloud,
            "reason": "off_road_or_overlaps_non_road",
        }

    if corners and len(corners) > 0:
        xy = mesh.get_center()[:2]
        half_diag = _mesh_half_diagonal_xy(mesh)
        if not _collision_free(xy, half_diag, diags, cents, len(corners)):
            return {
                "success": False,
                "point_cloud": point_cloud,
                "reason": "collision_or_occlusion_with_existing_vehicle",
            }

    try:
        pcd_obj = lidar_simulation(mesh)
    except ValueError:
        return {
            "success": False,
            "point_cloud": point_cloud,
            "reason": "lidar_simulation_failed_no_points",
        }
    except Exception as exc:
        return {
            "success": False,
            "point_cloud": point_cloud,
            "reason": f"lidar_simulation_exception:{type(exc).__name__}",
        }
    if pcd_obj.is_empty() or len(np.asarray(pcd_obj.points)) == 0:
        return {
            "success": False,
            "point_cloud": point_cloud,
            "reason": "lidar_simulation_returned_empty_point_cloud",
        }

    bg_pcd = pc_numpy_2_o3d(point_cloud)
    pcd_obj = lidar_intensity_convert(bg_pcd, pcd_obj)

    obb = mesh.get_minimal_oriented_bounding_box()
    visible_obb = pcd_obj.get_oriented_bounding_box()
    extent = np.asarray(obb.extent) / 2.0
    ctr = np.asarray(obb.center)

    occ = _occlusion_rates(pcd_obj, point_cloud, corners)
    if any(v >= occlusion_threshold for v in occ.values()):
        return {
            "success": False,
            "point_cloud": point_cloud,
            "reason": "occlusion_threshold_exceeded",
            "occ_rate": occ,
        }

    merged = _merge_clouds(point_cloud, pcd_obj, mesh, pos[2])

    return {
        "success": True,
        "point_cloud": merged,
        "center": ctr,
        "extent": extent,
        "rz_degree": rz_degree,
        "occ_rate": occ,
        "bbox_obb": obb,
        "visible_bbox_obb": visible_obb,
        "bbox_corners": _ordered_bbox_corners(obb).astype(np.float32),
    }



def delete_object(
    point_cloud: np.ndarray,
    vehicle_corner: np.ndarray,
    road_pc: Optional[np.ndarray] = None,
    fill_shadow: bool = True,
    shadow_sample_points: int = 50000,
) -> Dict:
    bg_pcd = pc_numpy_2_o3d(point_cloud)
    vehicle_pcd = get_obj_pcd_from_corner(vehicle_corner, bg_pcd)

    if vehicle_pcd.is_empty():
        return {"success": False, "point_cloud": point_cloud}

    try:
        tetra, pm = o3d.geometry.TetraMesh.create_from_point_cloud(vehicle_pcd)
        mesh_obj = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            vehicle_pcd, alpha=10, tetra_mesh=tetra, pt_map=pm,
        )
    except Exception:
        return _delete_simple(point_cloud, vehicle_corner)

    lo = np.min(vehicle_corner, axis=0)
    hi = np.max(vehicle_corner, axis=0)
    keep = ~(
        (point_cloud[:, 0] >= lo[0]) & (point_cloud[:, 0] <= hi[0]) &
        (point_cloud[:, 1] >= lo[1]) & (point_cloud[:, 1] <= hi[1]) &
        (point_cloud[:, 2] >= lo[2]) & (point_cloud[:, 2] <= hi[2])
    )
    cropped_xyzi = point_cloud[keep]

    if road_pc is not None and fill_shadow:
        try:
            road_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pc_numpy_2_o3d(road_pc), alpha=10,
            )
            samples = road_mesh.sample_points_uniformly(number_of_points=shadow_sample_points)
            s_mask, _ = get_shadow_mask(mesh_obj, samples)
            s_pts = np.asarray(samples.points)[s_mask]
            if len(s_pts) > 5:
                spcd = o3d.geometry.PointCloud()
                spcd.points = o3d.utility.Vector3dVector(s_pts)
                t2, pm2 = o3d.geometry.TetraMesh.create_from_point_cloud(spcd)
                if t2 is not None:
                    s_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                        spcd, alpha=5000, tetra_mesh=t2, pt_map=pm2,
                    )
                    lidar_pcd = lidar_simulation(s_mesh)
                    cropped_pcd = pc_numpy_2_o3d(cropped_xyzi)
                    lidar_pcd = lidar_intensity_convert(cropped_pcd, lidar_pcd)
                    merged = cropped_pcd + lidar_pcd
                    return {"success": True, "point_cloud": pcd_to_np(merged)}
        except Exception:
            pass

    return {"success": True, "point_cloud": cropped_xyzi.astype(np.float32)}


def _delete_simple(point_cloud, vehicle_corner):
    lo = np.min(vehicle_corner, axis=0)
    hi = np.max(vehicle_corner, axis=0)
    keep = ~(
        (point_cloud[:, 0] >= lo[0]) & (point_cloud[:, 0] <= hi[0]) &
        (point_cloud[:, 1] >= lo[1]) & (point_cloud[:, 1] <= hi[1]) &
        (point_cloud[:, 2] >= lo[2]) & (point_cloud[:, 2] <= hi[2])
    )
    return {"success": True, "point_cloud": point_cloud[keep].astype(np.float32)}



def process_point_cloud(
    point_cloud: np.ndarray,
    semantic_labels: np.ndarray,
    operation: str = "segmentation",
    **kwargs,
) -> Dict:
    seg = segment_road(point_cloud, semantic_labels)
    out = {**seg, "operation": operation}

    if operation == "segmentation":
        return out

    road = seg.get("road_pc")
    if road is None:
        return {**out, "success": False, "error": "No road surface detected"}

    if operation == "insert":
        result = insert_object(
            point_cloud=point_cloud,
            road_pc=road,
            non_road_pc=seg.get("non_road_pc"),
            position=kwargs.pop("position"),
            mesh_path=kwargs.pop("mesh_path"),
            **kwargs,
        )
        out.update(result)

    elif operation == "delete":
        result = delete_object(
            point_cloud=point_cloud,
            vehicle_corner=kwargs.pop("vehicle_corner"),
            road_pc=road,
            fill_shadow=kwargs.pop("fill_shadow", True),
        )
        out.update(result)

    return out
