
import torch
import numpy as np
from opencood.utils.common_utils import check_numpy_to_torch


def normalize_pairwise_tfm(pairwise_t_matrix, H, W, discrete_ratio, downsample_rate=1):
    pairwise_t_matrix = pairwise_t_matrix[:, :, :, [0, 1], :][:, :, :, :, [0, 1, 3]]
    pairwise_t_matrix[..., 0, 1] = pairwise_t_matrix[..., 0, 1] * H / W
    pairwise_t_matrix[..., 1, 0] = pairwise_t_matrix[..., 1, 0] * W / H
    pairwise_t_matrix[..., 0, 2] = pairwise_t_matrix[..., 0, 2] / (downsample_rate * discrete_ratio * W) * 2
    pairwise_t_matrix[..., 1, 2] = pairwise_t_matrix[..., 1, 2] / (downsample_rate * discrete_ratio * H) * 2

    return pairwise_t_matrix


def warp_affine_simple(src, M, dsize, mode='bilinear', padding_mode='zeros', align_corners=False):
    import torch.nn.functional as F
    B, C, H, W = src.size()
    grid = F.affine_grid(M, [B, C, dsize[0], dsize[1]], align_corners=align_corners).to(src.dtype).to(src.device)
    return F.grid_sample(src, grid, align_corners=align_corners, mode=mode, padding_mode=padding_mode)


def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x


def pose_to_tfm(pose):
    pose_tensor, is_np = check_numpy_to_torch(pose)

    if pose_tensor.shape[1] == 3:
        N = pose_tensor.shape[0]
        x = pose_tensor[:, 0]
        y = pose_tensor[:, 1]
        yaw = pose_tensor[:, 2]
        tfm = torch.eye(4, device=pose_tensor.device).view(1, 4, 4).repeat(N, 1, 1)
        tfm[:, 0, 0] = torch.cos(torch.deg2rad(yaw))
        tfm[:, 0, 1] = -torch.sin(torch.deg2rad(yaw))
        tfm[:, 1, 0] = torch.sin(torch.deg2rad(yaw))
        tfm[:, 1, 1] = torch.cos(torch.deg2rad(yaw))
        tfm[:, 0, 3] = x
        tfm[:, 1, 3] = y
    elif pose_tensor.shape[1] == 6:
        N = pose_tensor.shape[0]
        x, y, z, roll, yaw, pitch = pose_tensor[:, 0], pose_tensor[:, 1], pose_tensor[:, 2], \
                                     pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5]
        c_y = torch.cos(torch.deg2rad(yaw))
        s_y = torch.sin(torch.deg2rad(yaw))
        c_r = torch.cos(torch.deg2rad(roll))
        s_r = torch.sin(torch.deg2rad(roll))
        c_p = torch.cos(torch.deg2rad(pitch))
        s_p = torch.sin(torch.deg2rad(pitch))
        tfm = torch.eye(4, device=pose_tensor.device).view(1, 4, 4).repeat(N, 1, 1)
        tfm[:, 0, 3] = x
        tfm[:, 1, 3] = y
        tfm[:, 2, 3] = z
        tfm[:, 0, 0] = c_p * c_y
        tfm[:, 0, 1] = c_y * s_p * s_r - s_y * c_r
        tfm[:, 0, 2] = -c_y * s_p * c_r - s_y * s_r
        tfm[:, 1, 0] = s_y * c_p
        tfm[:, 1, 1] = s_y * s_p * s_r + c_y * c_r
        tfm[:, 1, 2] = -s_y * s_p * c_r + c_y * s_r
        tfm[:, 2, 0] = s_p
        tfm[:, 2, 1] = -c_p * s_r
        tfm[:, 2, 2] = c_p * c_r
    else:
        raise ValueError(f"Unsupported pose DOF: {pose_tensor.shape[1]}")

    if is_np:
        tfm = tfm.numpy()
    return tfm


def x_to_world(pose):
    x, y, z, roll, yaw, pitch = pose[:]
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))
    matrix = np.identity(4)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix


def x1_to_x2(x1, x2):
    x1_to_world_tfm = x_to_world(x1)
    x2_to_world_tfm = x_to_world(x2)
    world_to_x2 = np.linalg.inv(x2_to_world_tfm)
    transformation_matrix = np.dot(world_to_x2, x1_to_world_tfm)
    return transformation_matrix


def get_pairwise_transformation(base_data_dict, max_cav, proj_first=False):
    pairwise_t_matrix = np.tile(np.eye(4), (max_cav, max_cav, 1, 1))

    if proj_first:
        return pairwise_t_matrix
    else:
        t_list = []
        for cav_id, cav_content in base_data_dict.items():
            lidar_pose = cav_content['params']['lidar_pose']
            t_list.append(x_to_world(lidar_pose))

        for i in range(len(t_list)):
            for j in range(len(t_list)):
                if i != j:
                    t_matrix = np.linalg.solve(t_list[j], t_list[i])
                    pairwise_t_matrix[i, j] = t_matrix

    return pairwise_t_matrix
