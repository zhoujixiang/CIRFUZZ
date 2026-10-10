
import json
import numpy as np
import torch
from collections import OrderedDict


def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def limit_period(val, offset=0.5, period=2 * np.pi):
    val, is_numpy = check_numpy_to_torch(val)
    ans = val - torch.floor(val / period + offset) * period
    return ans.numpy() if is_numpy else ans


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def merge_features_to_dict(processed_feature_list, merge=None):
    merged_feature_dict = OrderedDict()
    for i in range(len(processed_feature_list)):
        for feature_name, feature in processed_feature_list[i].items():
            if feature_name not in merged_feature_dict:
                merged_feature_dict[feature_name] = []
            if isinstance(feature, list):
                merged_feature_dict[feature_name] += feature
            else:
                merged_feature_dict[feature_name].append(feature)
    if merge == 'stack':
        for feature_name, features in merged_feature_dict.items():
            merged_feature_dict[feature_name] = torch.stack(features, dim=0)
    elif merge == 'cat':
        for feature_name, features in merged_feature_dict.items():
            merged_feature_dict[feature_name] = torch.cat(features, dim=0)
    return merged_feature_dict


def torch_tensor_to_numpy(torch_tensor):
    return (torch_tensor.numpy() if not torch_tensor.is_cuda
            else torch_tensor.cpu().detach().numpy())
