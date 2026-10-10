
import torch
from torch import nn

from opencood.models.fuse_modules.fuse_utils import regroup_v1 as regroup
from opencood.utils.transformation_utils_ext import warp_affine_simple
from opencood.models.fuse_modules.att_fuse import ScaledDotProductAttention


class MaxFusion(nn.Module):
    def __init__(self):
        super(MaxFusion, self).__init__()

    def forward(self, x, record_len, pairwise_t_matrix):
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(x, record_len)
        out = []
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            i = 0
            neighbor_feature = warp_affine_simple(
                split_x[b], t_matrix[i, :, :, :], (H, W))
            out.append(torch.max(neighbor_feature, dim=0)[0])
        return torch.stack(out)


class AttFusion(nn.Module):
    def __init__(self, feature_dims):
        super(AttFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dims)

    def forward(self, xx, record_len, normalized_affine_matrix):
        _, C, H, W = xx.shape
        B, L = normalized_affine_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        out = []
        for b in range(B):
            N = record_len[b]
            t_matrix = normalized_affine_matrix[b][:N, :N, :, :]
            i = 0
            x = warp_affine_simple(split_x[b], t_matrix[i, :, :, :], (H, W))
            cav_num = x.shape[0]
            x = x.view(cav_num, C, -1).permute(2, 0, 1)
            h = self.att(x, x, x)
            h = h.permute(1, 2, 0).view(cav_num, C, H, W)[0, ...]
            out.append(h)
        return torch.stack(out)
