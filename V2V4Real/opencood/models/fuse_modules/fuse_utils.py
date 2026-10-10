
import torch
import numpy as np
from opencood.utils.common_utils import torch_tensor_to_numpy


def regroup(dense_feature, record_len, max_len):
    cum_sum_len = list(np.cumsum(torch_tensor_to_numpy(record_len)))
    split_features = torch.tensor_split(dense_feature, cum_sum_len[:-1])
    regroup_features = []
    mask = []

    for split_feature in split_features:
        feature_shape = split_feature.shape
        padding_len = max_len - feature_shape[0]
        mask.append([1] * feature_shape[0] + [0] * padding_len)

        padding_tensor = torch.zeros(
            padding_len, feature_shape[1], feature_shape[2], feature_shape[3],
            device=split_feature.device, dtype=split_feature.dtype
        )
        split_feature = torch.cat([split_feature, padding_tensor], dim=0)
        split_feature = split_feature.view(-1, feature_shape[2], feature_shape[3]).unsqueeze(0)
        regroup_features.append(split_feature)

    regroup_features = torch.cat(regroup_features, dim=0)
    regroup_features = regroup_features.view(
        -1, max_len, feature_shape[1], feature_shape[2], feature_shape[3]
    )
    mask = torch.from_numpy(np.array(mask)).to(regroup_features.device)
    return regroup_features, mask


def regroup_v1(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x
