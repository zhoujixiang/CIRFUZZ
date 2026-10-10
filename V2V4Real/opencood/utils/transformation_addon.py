import torch
import torch.nn.functional as F


def align_features(features, ego_poses, _d=1.6, target_pos=-1):
    assert target_pos in [-1, 0]

    B, T, C, H, W = features.shape

    pairwise_t_matrix = get_pairwise_transformation_torch(ego_poses, max_cav=T)

    pairwise_t_matrix = pairwise_t_matrix[:, :, :, [0, 1], :][:, :, :, :, [0, 1, 3]]
    pairwise_t_matrix[..., 0, 1] = pairwise_t_matrix[..., 0, 1] * H / W
    pairwise_t_matrix[..., 1, 0] = pairwise_t_matrix[..., 1, 0] * W / H
    pairwise_t_matrix[..., 0, 2] = pairwise_t_matrix[..., 0, 2] / (_d * W) * 2
    pairwise_t_matrix[..., 1, 2] = pairwise_t_matrix[..., 1, 2] / (_d * H) * 2

    x_fuse = []
    for b in range(B):
        t_matrix = pairwise_t_matrix[b][:T, :T, :, :]
        node_features = features[b]
        neighbor_feature = warp_affine_simple(
            node_features, t_matrix[target_pos, :, :, :], (H, W))
        x_fuse.append(neighbor_feature)

    aligned_features = torch.stack(x_fuse)
    return aligned_features


def pose_to_tfm(pose):
    import math
    from opencood.utils.common_utils import check_numpy_to_torch

    pose_tensor, is_np = check_numpy_to_torch(pose)
    pose = pose_tensor

    if pose.shape[1] == 3:
        N = pose.shape[0]
        x = pose[:, 0]
        y = pose[:, 1]
        yaw = pose[:, 2]

        tfm = torch.eye(4, device=pose.device).view(1, 4, 4).repeat(N, 1, 1)
        tfm[:, 0, 0] = torch.cos(torch.deg2rad(yaw))
        tfm[:, 0, 1] = -torch.sin(torch.deg2rad(yaw))
        tfm[:, 1, 0] = torch.sin(torch.deg2rad(yaw))
        tfm[:, 1, 1] = torch.cos(torch.deg2rad(yaw))
        tfm[:, 0, 3] = x
        tfm[:, 1, 3] = y

    elif pose.shape[1] == 6:
        N = pose.shape[0]
        x = pose[:, 0]
        y = pose[:, 1]
        z = pose[:, 2]
        roll = pose[:, 3]
        yaw = pose[:, 4]
        pitch = pose[:, 5]

        c_y = torch.cos(torch.deg2rad(yaw))
        s_y = torch.sin(torch.deg2rad(yaw))
        c_r = torch.cos(torch.deg2rad(roll))
        s_r = torch.sin(torch.deg2rad(roll))
        c_p = torch.cos(torch.deg2rad(pitch))
        s_p = torch.sin(torch.deg2rad(pitch))

        tfm = torch.eye(4, device=pose.device).view(1, 4, 4).repeat(N, 1, 1)

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

    if is_np:
        tfm = tfm.numpy()

    return tfm


def warp_affine_simple(src, M, dsize, mode='bilinear',
                       padding_mode='zeros', align_corners=False):
    B, C, H, W = src.size()
    grid = F.affine_grid(M, [B, C, dsize[0], dsize[1]],
                         align_corners=align_corners).to(src)
    return F.grid_sample(src, grid, align_corners=align_corners)


def get_pairwise_transformation_torch(lidar_poses_list, max_cav=3):
    B = lidar_poses_list.shape[0]

    pairwise_t_matrix = torch.eye(4, device=lidar_poses_list.device).view(
        1, 1, 1, 4, 4).repeat(B, max_cav, max_cav, 1, 1)

    for b in range(B):
        t_list = lidar_poses_list[b]
        if t_list.shape[-1] != 4:
            t_list = pose_to_tfm(t_list)

        for i in range(len(t_list)):
            for j in range(len(t_list)):
                if i != j:
                    t_matrix = torch.linalg.solve(t_list[j], t_list[i])
                    pairwise_t_matrix[b][i, j] = t_matrix

    return pairwise_t_matrix
