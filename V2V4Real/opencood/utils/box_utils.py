import sys

import numpy as np

import torch
import torch.nn.functional as F
import opencood.utils.common_utils as common_utils
from opencood.utils.transformation_utils import x1_to_x2


def corner_to_center(corner3d, order='lwh'):
    assert corner3d.ndim == 3
    batch_size = corner3d.shape[0]

    xyz = np.mean(corner3d[:, [0, 3, 5, 6], :], axis=1)
    h = abs(np.mean(corner3d[:, 4:, 2] - corner3d[:, :4, 2], axis=1,
                    keepdims=True))
    l = (np.sqrt(np.sum((corner3d[:, 0, [0, 1]] - corner3d[:, 3, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 2, [0, 1]] - corner3d[:, 1, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 4, [0, 1]] - corner3d[:, 7, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 5, [0, 1]] - corner3d[:, 6, [0, 1]]) ** 2,
                        axis=1, keepdims=True))) / 4

    w = (np.sqrt(
        np.sum((corner3d[:, 0, [0, 1]] - corner3d[:, 1, [0, 1]]) ** 2, axis=1,
               keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 2, [0, 1]] - corner3d[:, 3, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 4, [0, 1]] - corner3d[:, 5, [0, 1]]) ** 2,
                        axis=1, keepdims=True)) +
         np.sqrt(np.sum((corner3d[:, 6, [0, 1]] - corner3d[:, 7, [0, 1]]) ** 2,
                        axis=1, keepdims=True))) / 4

    theta = (np.arctan2(corner3d[:, 1, 1] - corner3d[:, 2, 1],
                        corner3d[:, 1, 0] - corner3d[:, 2, 0]) +
             np.arctan2(corner3d[:, 0, 1] - corner3d[:, 3, 1],
                        corner3d[:, 0, 0] - corner3d[:, 3, 0]) +
             np.arctan2(corner3d[:, 5, 1] - corner3d[:, 6, 1],
                        corner3d[:, 5, 0] - corner3d[:, 6, 0]) +
             np.arctan2(corner3d[:, 4, 1] - corner3d[:, 7, 1],
                        corner3d[:, 4, 0] - corner3d[:, 7, 0]))[:,
            np.newaxis] / 4

    if order == 'lwh':
        return np.concatenate([xyz, l, w, h, theta], axis=1).reshape(
            batch_size, 7)
    elif order == 'hwl':
        return np.concatenate([xyz, h, w, l, theta], axis=1).reshape(
            batch_size, 7)
    else:
        sys.exit('Unknown order')


def boxes_to_corners2d(boxes3d, order):
    corners3d = boxes_to_corners_3d(boxes3d, order)
    corners2d = corners3d[:, :4, :]
    return corners2d


def boxes2d_to_corners2d(boxes2d, order="lwh"):
    assert order == "lwh", "boxes2d_to_corners_2d only supports lwh order for now."
    boxes2d, is_numpy = common_utils.check_numpy_to_torch(boxes2d)
    template = boxes2d.new_tensor((
        [1, -1], [1, 1], [-1, 1], [-1, -1]
    )) / 2
    input_shape = boxes2d.shape
    boxes2d = boxes2d.view(-1, 5)
    corners2d = boxes2d[:, None, 2:4].repeat(1, 4, 1) * template[None, :, :]
    corners2d = common_utils.rotate_points_along_z_2d(corners2d.view(-1, 2),
                                                      boxes2d[:,
                                                      4].repeat_interleave(
                                                          4)).view(-1, 4,
                                                                   2)
    corners2d += boxes2d[:, None, 0:2]
    corners2d = corners2d.view(*(input_shape[:-1]), 4, 2)
    return corners2d


def boxes_to_corners_3d(boxes3d, order):

    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)

    if order == 'hwl':
        boxes3d[:, 3:6] = boxes3d[:, [5, 4, 3]]

    template = boxes3d.new_tensor((
        [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1],
        [1, -1, 1], [1, 1, 1], [-1, 1, 1], [-1, -1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = common_utils.rotate_points_along_z(corners3d.view(-1, 8, 3),
                                                   boxes3d[:, 6]).view(-1, 8,
                                                                       3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def box3d_to_2d(box3d):
    box2d = box3d[:, :4, :2]
    return box2d


def corner2d_to_standup_box(box2d):
    N = box2d.shape[0]
    standup_boxes2d = np.zeros((N, 4))

    standup_boxes2d[:, 0] = np.min(box2d[:, :, 0], axis=1)
    standup_boxes2d[:, 1] = np.min(box2d[:, :, 1], axis=1)
    standup_boxes2d[:, 2] = np.max(box2d[:, :, 0], axis=1)
    standup_boxes2d[:, 3] = np.max(box2d[:, :, 1], axis=1)

    return standup_boxes2d


def corner_to_standup_box_torch(box_corner):
    N = box_corner.shape[0]
    standup_boxes2d = torch.zeros((N, 4))

    standup_boxes2d = standup_boxes2d.to(box_corner.device)

    standup_boxes2d[:, 0] = torch.min(box_corner[:, :, 0], dim=1).values
    standup_boxes2d[:, 1] = torch.min(box_corner[:, :, 1], dim=1).values
    standup_boxes2d[:, 2] = torch.max(box_corner[:, :, 0], dim=1).values
    standup_boxes2d[:, 3] = torch.max(box_corner[:, :, 1], dim=1).values

    return standup_boxes2d


def project_box3d(box3d, transformation_matrix):
    assert transformation_matrix.shape == (4, 4)
    box3d, is_numpy = \
        common_utils.check_numpy_to_torch(box3d)
    transformation_matrix, _ = \
        common_utils.check_numpy_to_torch(transformation_matrix)

    box3d_corner = box3d.transpose(1, 2)
    torch_ones = torch.ones((box3d_corner.shape[0], 1, 8))
    torch_ones = torch_ones.to(box3d_corner.device)
    box3d_corner = torch.cat((box3d_corner, torch_ones),
                             dim=1)
    projected_box3d = torch.matmul(transformation_matrix,
                                   box3d_corner)
    projected_box3d = projected_box3d[:, :3, :].transpose(1, 2)

    return projected_box3d if not is_numpy else projected_box3d.numpy()


def project_points_by_matrix_torch(points, transformation_matrix):
    points_homogeneous = F.pad(points, (0, 1), mode="constant", value=1)
    projected_points = torch.einsum("ik, jk->ij", points_homogeneous,
                                    transformation_matrix)
    return projected_points[:, :3]


def get_mask_for_boxes_within_range_torch(boxes):
    from opencood.data_utils.datasets import GT_RANGE

    device = boxes.device
    boundary_lower_range = \
        torch.Tensor(GT_RANGE[:2]).reshape(1, 1, -1).to(device)
    boundary_higher_range = \
        torch.Tensor(GT_RANGE[3:5]).reshape(1, 1, -1).to(device)

    mask = torch.all(
        torch.all(boxes[:, :, :2] >= boundary_lower_range,
                  dim=-1) & \
        torch.all(boxes[:, :, :2] <= boundary_higher_range,
                  dim=-1), dim=-1)

    return mask


def mask_boxes_outside_range_numpy(boxes, limit_range, order,
                                   min_num_corners=2):
    assert boxes.shape[1] == 8 or boxes.shape[1] == 7

    new_boxes = boxes.copy()
    if boxes.shape[1] == 7:
        new_boxes = boxes_to_corners_3d(new_boxes, order)

    mask = ((new_boxes >= limit_range[0:3]) &
            (new_boxes <= limit_range[3:6])).all(axis=2)
    mask = mask.sum(axis=1) >= min_num_corners

    return boxes[mask], mask


def create_bbx(extent):

    bbx = np.array([[extent[0], -extent[1], -extent[2]],
                    [extent[0], extent[1], -extent[2]],
                    [-extent[0], extent[1], -extent[2]],
                    [-extent[0], -extent[1], -extent[2]],
                    [extent[0], -extent[1], extent[2]],
                    [extent[0], extent[1], extent[2]],
                    [-extent[0], extent[1], extent[2]],
                    [-extent[0], -extent[1], extent[2]]])

    return bbx


def project_world_objects(object_dict,
                          output_dict,
                          transformation_matrix,
                          lidar_range,
                          order):
    for object_id, object_content in object_dict.items():
        location = object_content['location']
        rotation = object_content['angle']
        center = object_content['center']
        extent = object_content['extent']
        if 'ass_id' not in object_content:
            ass_id = object_id
        else:
            ass_id = object_content['ass_id']
        if 'obj_type' not in object_content:
            obj_type = 'Car'
        else:
            obj_type = object_content['obj_type']

        if obj_type == 'Pedestrian':
            continue

        object_pose = [location[0] + center[0],
                       location[1] + center[1],
                       location[2] + center[2],
                       rotation[0], rotation[1], rotation[2]]
        object2lidar = x1_to_x2(object_pose, transformation_matrix)

        bbx = create_bbx(extent).T
        bbx = np.r_[bbx, [np.ones(bbx.shape[1])]]

        bbx_lidar = np.dot(object2lidar, bbx).T
        bbx_lidar = np.expand_dims(bbx_lidar[:, :3], 0)
        bbx_lidar = corner_to_center(bbx_lidar, order=order)
        bbx_lidar, _ = mask_boxes_outside_range_numpy(bbx_lidar,
                                                      lidar_range,
                                                      order,
                                                      2)

        if bbx_lidar.shape[0] > 0:
            output_dict.update({object_id: {'coord': bbx_lidar,
                                            'ass_id': ass_id}})


def get_points_in_rotated_box(p, box_corner):
    edge1 = box_corner[1, :] - box_corner[0, :]
    edge2 = box_corner[3, :] - box_corner[0, :]
    if _has_degenerate_box_edges(edge1, edge2) or \
            not np.all(np.isfinite(box_corner)):
        return np.zeros((0, p.shape[-1]), dtype=p.dtype)
    p_rel = p - box_corner[0, :].reshape(1, -1)

    l1 = get_projection_length_for_vector_projection(p_rel, edge1)
    l2 = get_projection_length_for_vector_projection(p_rel, edge2)
    mask = np.logical_and(l1 >= 0, l1 <= 1)
    mask = np.logical_and(mask, l2 >= 0)
    mask = np.logical_and(mask, l2 <= 1)
    p_in_box = p[mask, :]
    return p_in_box


def get_points_in_rotated_box_3d(p, box_corner):
    edge1 = box_corner[1, :] - box_corner[0, :]
    edge2 = box_corner[3, :] - box_corner[0, :]
    edge3 = box_corner[4, :] - box_corner[0, :]
    if _has_degenerate_box_edges(edge1, edge2, edge3) or \
            not np.all(np.isfinite(box_corner)):
        return np.zeros((0, p.shape[-1]), dtype=p.dtype)

    p_rel = p - box_corner[0, :].reshape(1, -1)

    l1 = get_projection_length_for_vector_projection(p_rel, edge1)
    l2 = get_projection_length_for_vector_projection(p_rel, edge2)
    l3 = get_projection_length_for_vector_projection(p_rel, edge3)
    mask1 = np.logical_and(l1 >= 0, l1 <= 1)
    mask2 = np.logical_and(l2 >= 0, l2 <= 1)
    mask3 = np.logical_and(l3 >= 0, l3 <= 1)

    mask = np.logical_and(mask1, mask2)
    mask = np.logical_and(mask, mask3)
    p_in_box = p[mask, :]

    return p_in_box


def get_projection_length_for_vector_projection(a, b):
    assert np.sum(b ** 2, axis=-1) > 1e-6
    length = a.dot(b) / np.sum(b ** 2, axis=-1)
    return length


def _has_degenerate_box_edges(*edges, eps=1e-6):
    for edge in edges:
        if not np.all(np.isfinite(edge)) or np.sum(edge ** 2, axis=-1) <= eps:
            return True
    return False


def nms_rotated(boxes, scores, threshold):
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int32)
    boxes = boxes.cpu().detach().numpy()
    scores = scores.cpu().detach().numpy()

    polygons = common_utils.convert_format(boxes)

    top = 1000
    ixs = scores.argsort()[::-1][:top]

    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = common_utils.compute_iou(polygons[i], polygons[ixs[1:]])
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)

    return np.array(pick, dtype=np.int32)


def nms_pytorch(boxes: torch.tensor, thresh_iou: float):

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    scores = boxes[:, 4]

    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()

    keep = []

    while len(order) > 0:

        idx = order[-1]

        keep.append(idx.numpy().item()
                    if not idx.is_cuda else idx.cpu().detach().numpy().item())

        order = order[:-1]

        if len(order) == 0:
            break

        xx1 = torch.index_select(x1, dim=0, index=order)
        xx2 = torch.index_select(x2, dim=0, index=order)
        yy1 = torch.index_select(y1, dim=0, index=order)
        yy2 = torch.index_select(y2, dim=0, index=order)

        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])

        w = xx2 - xx1
        h = yy2 - yy1

        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        inter = w * h

        rem_areas = torch.index_select(areas, dim=0, index=order)

        union = (rem_areas - inter) + areas[idx]

        IoU = inter / union

        mask = IoU < thresh_iou
        order = order[mask]

    return keep


def remove_large_pred_bbx(bbx_3d):
    bbx_x_max = torch.max(bbx_3d[:, :, 0], dim=1)[0]
    bbx_x_min = torch.min(bbx_3d[:, :, 0], dim=1)[0]
    x_len = bbx_x_max - bbx_x_min

    bbx_y_max = torch.max(bbx_3d[:, :, 1], dim=1)[0]
    bbx_y_min = torch.min(bbx_3d[:, :, 1], dim=1)[0]
    y_len = bbx_y_max - bbx_y_min

    bbx_z_max = torch.max(bbx_3d[:, :, 1], dim=1)[0]
    bbx_z_min = torch.min(bbx_3d[:, :, 1], dim=1)[0]
    z_len = bbx_z_max - bbx_z_min

    index = torch.logical_and(x_len <= 6, y_len <= 6)
    index = torch.logical_and(index, z_len)

    return index


def remove_bbx_abnormal_z(bbx_3d):
    bbx_z_min = torch.min(bbx_3d[:, :, 2], dim=1)[0]
    bbx_z_max = torch.max(bbx_3d[:, :, 2], dim=1)[0]
    index = torch.logical_and(bbx_z_min >= -3, bbx_z_max <= 1)

    return index


def project_points_by_matrix_torch(points, transformation_matrix):
    points, is_numpy = \
        common_utils.check_numpy_to_torch(points)
    transformation_matrix, _ = \
        common_utils.check_numpy_to_torch(transformation_matrix)

    points_homogeneous = F.pad(points, (0, 1), mode="constant", value=1)
    projected_points = torch.einsum("ik, jk->ij", points_homogeneous,
                                    transformation_matrix)

    return projected_points[:, :3] if not is_numpy \
        else projected_points[:, :3].numpy()


if __name__ == "__main__":
    x = np.arange(-5, 5, 0.1)
    y = np.arange(-5, 5, 0.1)
    xx, yy = np.meshgrid(x, y)
    points = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=-1)
    box_corners = np.array([
        [2, -2], [2, 2], [-2, 2], [-2, -2]
    ])
    temp = get_points_in_rotated_box(points, box_corners)
    assert np.all(np.logical_and(temp[:, 0] >= -2, temp[:, 0] <= 2))
    assert np.all(np.logical_and(temp[:, 1] >= -2, temp[:, 1] <= 2))
