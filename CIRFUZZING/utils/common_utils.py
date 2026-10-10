import config
import math
import copy
import open3d as o3d
import numpy as np
from shapely.geometry import Polygon
from build.mtest.utils import box_utils
from scipy.spatial.transform import Rotation as RR


def pc_numpy_2_o3d(pc):
    if isinstance(pc, np.ndarray) and pc.shape[1] == 4:
        xyz = pc[:, :3]
        intensity = pc[:, 3:4]
        colors = np.hstack((intensity, intensity, intensity))
        pcd_bg = o3d.geometry.PointCloud()
        pcd_bg.points = o3d.utility.Vector3dVector(xyz)
        pcd_bg.colors = o3d.utility.Vector3dVector(colors)
        return pcd_bg
    pcd_bg = o3d.geometry.PointCloud()
    pcd_bg.points = o3d.utility.Vector3dVector(pc)
    return pcd_bg


def pcd_to_np(pcd):
    xyz = np.asarray(pcd.points)
    intensity = np.expand_dims(np.asarray(pcd.colors)[:, 0], -1)

    pcd_np = np.hstack((xyz, intensity))

    return np.asarray(pcd_np, dtype=np.float32)


def get_obj_pcd_from_corner(corner, bg_pcd):
    min_bound = np.min(corner, axis=0)
    max_bound = np.max(corner, axis=0)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    obj_pcd = bg_pcd.crop(bbox)
    return obj_pcd


def get_corners(param):
    objs_ry = []
    objs_l = []
    objs_w = []
    objs_h = []
    objs_loc = []

    for i, vehicle in param["vehicles"].items():
        objs_ry.insert(i, vehicle['angle'][1] * np.pi / 180)
        objs_l.insert(i, vehicle['extent'][0] * 2)
        objs_w.insert(i, vehicle['extent'][1] * 2)
        objs_h.insert(i, vehicle['extent'][2] * 2)
        arr_loc = np.array(vehicle['location']) + np.array(vehicle['center'])
        objs_loc.insert(i, arr_loc)
    rots = np.array(objs_ry)

    l, h, w = np.array(objs_l).reshape(-1, 1), np.array(objs_h).reshape(-1, 1), np.array(objs_w).reshape(-1, 1)
    loc_lidar = np.array(objs_loc)

    rots = rots[..., np.newaxis]

    gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, rots], axis=1)

    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)

    return corners_lidar


def get_corners_positions(param):
    objs_ry = []
    objs_ry_degree = []
    objs_l = []
    objs_w = []
    objs_h = []
    objs_loc = []

    for i, vehicle in param["vehicles"].items():
        objs_ry.insert(i, vehicle['angle'][1] * np.pi / 180)
        objs_ry_degree.insert(i, vehicle['angle'][1])
        objs_l.insert(i, vehicle['extent'][0] * 2)
        objs_w.insert(i, vehicle['extent'][1] * 2)
        objs_h.insert(i, vehicle['extent'][2] * 2)
        arr_loc = np.array(vehicle['location']) + np.array(vehicle['center'])
        objs_loc.insert(i, arr_loc)
    rots = np.array(objs_ry)

    l, h, w = np.array(objs_l).reshape(-1, 1), np.array(objs_h).reshape(-1, 1), np.array(objs_w).reshape(-1, 1)
    loc_lidar = np.array(objs_loc)

    rots = rots[..., np.newaxis]
    gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, rots], axis=1)

    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)

    return corners_lidar, objs_ry_degree, objs_loc, objs_h


def find_cp_vehicle_id(cp_info, ego_id):
    for cp_id, cp_vehicle in cp_info.param['vehicles'].items():
        if cp_vehicle['ass_id'] == ego_id:
            return cp_id, True

    return -1, False


def get_pc_index_in_corner(pc, corner):
    min_bound = np.min(corner, axis=0)
    max_bound = np.max(corner, axis=0)
    indices = np.where(
        (min_bound[0] <= pc[:, 0]) & (pc[:, 0] <= max_bound[0]) &
        (min_bound[1] <= pc[:, 1]) & (pc[:, 1] <= max_bound[1]) &
        (min_bound[2] <= pc[:, 2]) & (pc[:, 2] <= max_bound[2])
    )[0]

    return indices


def center_system_transform(center, cur_lidar_pose, tar_lidar_pose):
    center_pos = np.append(center, 1)
    T_cur_tar = np.linalg.inv(tar_lidar_pose) @ cur_lidar_pose
    target_pos = T_cur_tar @ center_pos
    return target_pos[:3]


def points_system_transform(points, cur_lidar_pose, tar_lidar_pose):
    tar_points = []
    for pt in points:
        tar_pt = center_system_transform(pt, cur_lidar_pose, tar_lidar_pose)
        tar_points.append(tar_pt)
    return tar_points


def rz_degree_system_transform(rz_degree, cur_lidar_pose, tar_lidar_pose):
    theta = np.radians(rz_degree)

    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    T = tar_lidar_pose @ np.linalg.inv(cur_lidar_pose)
    T_inv = np.linalg.inv(T)

    Rz_tar = T @ Rz @ T_inv

    theta_tar = np.arctan2(Rz_tar[1, 0], Rz_tar[0, 0])

    rz_degree_tar = np.degrees(theta_tar)

    return rz_degree_tar


def is_box_containing_points(box, points):
    min_bound = np.min(box, axis=0)
    max_bound = np.max(box, axis=0)

    for pt in points:
        if (min_bound[0] <= pt[0] <= max_bound[0] and
                min_bound[1] <= pt[1] <= max_bound[1] and
                min_bound[2] <= pt[2] <= max_bound[2]):
            return True
        else:
            continue

    return False


def is_box_containing_position(box, position):
    min_bound = np.min(box, axis=0)
    max_bound = np.max(box, axis=0)

    if (min_bound[0] <= position[0] <= max_bound[0] and
            min_bound[1] <= position[1] <= max_bound[1] and
            min_bound[2] <= position[2] <= max_bound[2]):
        return True

    return False


def is_boxes_overlap(box1, box2):
    box1_2d = box1[:, :2]
    box2_2d = box2[:, :2]

    poly1 = Polygon(box1_2d)
    poly2 = Polygon(box2_2d)

    return poly1.intersects(poly2)


def param_vehicle_check(param):
    for key, value in param['vehicles'].items():
        print(key)


def get_geometric_info(obj):
    min_xyz = obj.get_min_bound()
    max_xyz = obj.get_max_bound()
    x_min, x_max = min_xyz[0], max_xyz[0]
    y_min, y_max = min_xyz[1], max_xyz[1]
    z_min, z_max = min_xyz[2], max_xyz[2]
    half_diagonal = math.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2) / 2
    center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
    half_height = (max_xyz[2] - min_xyz[2]) / 2

    return half_diagonal, center, half_height


def corner_to_line_set_box(corner, line_color=[1, 0, 0]):
    lines_box = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 5], [2, 6], [3, 7],
                          [4, 5], [5, 6], [6, 7], [7, 4]])

    colors = np.array([line_color for _ in range(len(lines_box))])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corner)
    line_set.lines = o3d.utility.Vector2iVector(lines_box)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def create_cylinder_between_points(point1, point2, radius=0.02, resolution=20):
    point1 = np.array(point1)
    point2 = np.array(point2)

    height = np.linalg.norm(point2 - point1)

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height, resolution=resolution)
    cylinder.compute_vertex_normals()

    midpoint = (point2 + point1) / 2
    direction = (point2 - point1) / height

    cylinder.translate(midpoint)

    z_axis = np.array([0, 0, 1])
    axis = np.cross(z_axis, direction)
    angle = np.arccos(np.dot(z_axis, direction))
    if np.linalg.norm(axis) > 1e-6:
        axis = axis / np.linalg.norm(axis)
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        cylinder.rotate(rotation_matrix, center=midpoint)

    return cylinder


def get_box_corners(center, extent):
    x, y, z = center
    l, w, h = extent

    corners = [
        [x + l, y + w, z + h],
        [x + l, y + w, z - h],
        [x + l, y - w, z + h],
        [x + l, y + w, z - h],
        [x - l, y + w, z + h],
        [x - l, y + w, z - h],
        [x - l, y - w, z + h],
        [x - l, y - w, z - h],
    ]

    return np.array(corners)


def get_euler_from_matrix(R):
    euler_type = "XYZ"

    sciangle_0, sciangle_1, sciangle_2 = RR.from_matrix(R).as_euler(euler_type)

    return sciangle_0, sciangle_1, sciangle_2


def get_box3d_R(box3d):
    return copy.copy(box3d.R)


def crop_point_cloud(pc, center, size):
    x, y, z = center

    x_min = x - size / 2
    x_max = x + size / 2
    y_min = y - size / 2
    y_max = y + size / 2

    cropped_pc = pc[
        (pc[:, 0] >= x_min) & (pc[:, 0] <= x_max) &
        (pc[:, 1] >= y_min) & (pc[:, 1] <= y_max)
        ]

    return cropped_pc

