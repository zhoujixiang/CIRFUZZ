import re
import yaml
import os
import math

import numpy as np


def load_yaml(file, opt=None):
    if opt and opt.model_dir:
        file = os.path.join(opt.model_dir, 'config.yaml')

    stream = open(file, 'r')
    loader = yaml.Loader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    param = yaml.load(stream, Loader=loader)
    if "yaml_parser" in param:
        param = eval(param["yaml_parser"])(param)

    return param


def load_voxel_params(param):
    anchor_args = param['postprocess']['anchor_args']
    cav_lidar_range = anchor_args['cav_lidar_range']
    voxel_size = param['preprocess']['args']['voxel_size']

    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    anchor_args['vw'] = vw
    anchor_args['vh'] = vh
    anchor_args['vd'] = vd

    anchor_args['W'] = int((cav_lidar_range[3] - cav_lidar_range[0]) / vw)
    anchor_args['H'] = int((cav_lidar_range[4] - cav_lidar_range[1]) / vh)
    anchor_args['D'] = int((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    param['postprocess'].update({'anchor_args': anchor_args})
    if 'model' in param:
        param['model']['args']['W'] = anchor_args['W']
        param['model']['args']['H'] = anchor_args['H']
        param['model']['args']['D'] = anchor_args['D']
    return param


def load_point_pillar_params(param):
    cav_lidar_range = param['preprocess']['cav_lidar_range']
    voxel_size = param['preprocess']['args']['voxel_size']

    grid_size = (np.array(cav_lidar_range[3:6]) - np.array(
        cav_lidar_range[0:3])) / \
                np.array(voxel_size)
    grid_size = np.round(grid_size).astype(np.int64)
    param['model']['args']['point_pillar_scatter']['grid_size'] = grid_size

    ego_encoder = param['model']['args'].get('ego_encoder')
    if ego_encoder and 'point_pillar_scatter' in ego_encoder:
        ego_encoder['point_pillar_scatter']['grid_size'] = grid_size

    anchor_args = param['postprocess']['anchor_args']

    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    anchor_args['vw'] = vw
    anchor_args['vh'] = vh
    anchor_args['vd'] = vd

    anchor_args['W'] = math.ceil((cav_lidar_range[3] - cav_lidar_range[0]) / vw)
    anchor_args['H'] = math.ceil((cav_lidar_range[4] - cav_lidar_range[1]) / vh)
    anchor_args['D'] = math.ceil((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    param['postprocess'].update({'anchor_args': anchor_args})

    return param

def load_second_params(param):
    cav_lidar_range = param['preprocess']['cav_lidar_range']
    voxel_size = param['preprocess']['args']['voxel_size']

    grid_size = (np.array(cav_lidar_range[3:6]) - np.array(
        cav_lidar_range[0:3])) / \
                np.array(voxel_size)
    grid_size = np.round(grid_size).astype(np.int64)
    param['model']['args']['grid_size'] = grid_size

    anchor_args = param['postprocess']['anchor_args']

    vw = voxel_size[0]
    vh = voxel_size[1]
    vd = voxel_size[2]

    anchor_args['vw'] = vw
    anchor_args['vh'] = vh
    anchor_args['vd'] = vd

    anchor_args['W'] = math.ceil((cav_lidar_range[3] - cav_lidar_range[0]) / vw)
    anchor_args['H'] = math.ceil((cav_lidar_range[4] - cav_lidar_range[1]) / vh)
    anchor_args['D'] = math.ceil((cav_lidar_range[5] - cav_lidar_range[2]) / vd)

    param['postprocess'].update({'anchor_args': anchor_args})

    return param

def load_bev_params(param):
    res = param["preprocess"]["args"]["res"]
    L1, W1, H1, L2, W2, H2 = param["preprocess"]["cav_lidar_range"]
    downsample_rate = param["preprocess"]["args"]["downsample_rate"]

    def f(low, high, r):
        return int((high - low) / r)

    input_shape = (
        int((f(L1, L2, res))),
        int((f(W1, W2, res))),
        int((f(H1, H2, res)) + 1)
    )
    label_shape = (
        int(input_shape[0] / downsample_rate),
        int(input_shape[1] / downsample_rate),
        7
    )
    geometry_param = {
        'L1': L1,
        'L2': L2,
        'W1': W1,
        'W2': W2,
        'H1': H1,
        'H2': H2,
        "downsample_rate": downsample_rate,
        "input_shape": input_shape,
        "label_shape": label_shape,
        "res": res
    }
    param["preprocess"]["geometry_param"] = geometry_param
    param["postprocess"]["geometry_param"] = geometry_param
    param["model"]["args"]["geometry_param"] = geometry_param
    return param


def save_yaml(data, save_name):

    save_dir = os.path.dirname(save_name)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with open(save_name, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, sort_keys=False)
