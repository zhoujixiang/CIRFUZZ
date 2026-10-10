
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.utils.transformation_utils_ext import warp_affine_simple


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context


class AttFusion(nn.Module):
    def __init__(self, args):
        super(AttFusion, self).__init__()
        self.discrete_ratio = args['voxel_size'][0]
        self.downsample_rate = args.get('downsample_rate', 1)
        in_channels = args.get('in_channels', args.get('feat_dim'))
        self.att = ScaledDotProductAttention(in_channels)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        return torch.tensor_split(x, cum_sum_len[:-1].cpu())

    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = self.regroup(xx, record_len)

        pairwise_t_matrix = pairwise_t_matrix[:, :, :, [0, 1], :][:, :, :, :, [0, 1, 3]]
        pairwise_t_matrix[..., 0, 1] = pairwise_t_matrix[..., 0, 1] * H / W
        pairwise_t_matrix[..., 1, 0] = pairwise_t_matrix[..., 1, 0] * W / H
        pairwise_t_matrix[..., 0, 2] = pairwise_t_matrix[..., 0, 2] / (
                    self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[..., 1, 2] = pairwise_t_matrix[..., 1, 2] / (
                    self.downsample_rate * self.discrete_ratio * H) * 2

        out = []
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            i = 0
            x = warp_affine_simple(split_x[b], t_matrix[i, :, :, :], (H, W))
            cav_num = x.shape[0]
            x = x.view(cav_num, C, -1).permute(2, 0, 1)
            h = self.att(x, x, x)
            h = h.permute(1, 2, 0).view(cav_num, C, H, W)[0, ...]
            out.append(h)

        return torch.stack(out)
