
import numpy as np
import torch
from shapely.geometry import Polygon


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def check_contain_nan(x):
    if isinstance(x, dict):
        return any(check_contain_nan(v) for k, v in x.items())
    if isinstance(x, list):
        return any(check_contain_nan(itm) for itm in x)
    if isinstance(x, int) or isinstance(x, float):
        return False
    if isinstance(x, np.ndarray):
        return np.any(np.isnan(x))
    return torch.any(x.isnan()).detach().cpu().item()


def rotate_points_along_z(points, angle):
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3].float(), rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def rotate_points_along_z_2d(points, angle):
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    rot_matrix = torch.stack((cosa, sina, -sina, cosa), dim=1).view(-1, 2,
                                                                    2).float()
    points_rot = torch.einsum("ik, ikj->ij", points.float(), rot_matrix)
    return points_rot.numpy() if is_numpy else points_rot


def remove_ego_from_objects(objects, ego_id):
    if ego_id in objects:
        del objects[ego_id]


def retrieve_ego_id(base_data_dict):
    ego_id = None

    for cav_id, cav_content in base_data_dict.items():
        if cav_content['ego']:
            ego_id = cav_id
            break
    return ego_id


def compute_iou(box, boxes):
    iou = [box.intersection(b).area / box.union(b).area for b in boxes]

    return np.array(iou, dtype=np.float32)


def convert_format(boxes_array):
    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in
                boxes_array]
    return np.array(polygons)


def torch_tensor_to_numpy(torch_tensor):
    return torch_tensor.numpy() if not torch_tensor.is_cuda else \
        torch_tensor.cpu().detach().numpy()
