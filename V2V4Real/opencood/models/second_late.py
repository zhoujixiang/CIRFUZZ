import numpy as np
import torch.nn as nn

from opencood.models.sub_modules.cia_ssd_utils import SSFA
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d_second import \
    VoxelBackBone8x


class SecondLate(nn.Module):
    def __init__(self, args):
        super(SecondLate, self).__init__()

        grid_size = args.get('grid_size')
        if grid_size is None:
            lidar_range = np.array(args['lidar_range'])
            voxel_size = np.array(args['voxel_size'])
            grid_size = np.round(
                (lidar_range[3:6] - lidar_range[:3]) / voxel_size
            ).astype(np.int64)

        self.vfe = MeanVFE(args['mean_vfe'],
                           args['mean_vfe']['num_point_features'])
        self.spconv_block = VoxelBackBone8x(
            args['spconv'],
            input_channels=args['spconv']['num_features_in'],
            grid_size=grid_size,
        )
        self.map_to_bev = HeightCompression(args['map2bev'])
        self.ssfa = SSFA(args['ssfa'])

        self.shrink_flag = False
        head_dim = args['head']['num_input']
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])
            head_dim = args['shrink_header']['dim'][-1]

        self.cls_head = nn.Conv2d(head_dim, args['anchor_num'], kernel_size=1)
        self.reg_head = nn.Conv2d(head_dim,
                                  7 * args['anchor_num'],
                                  kernel_size=1)

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        batch_size = int(voxel_coords[:, 0].max().item()) + 1

        batch_dict = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
            'batch_size': batch_size
        }

        batch_dict = self.vfe(batch_dict)
        batch_dict = self.spconv_block(batch_dict)
        batch_dict = self.map_to_bev(batch_dict)

        spatial_features_2d = self.ssfa(batch_dict['spatial_features'])
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)

        return {
            'psm': self.cls_head(spatial_features_2d),
            'rm': self.reg_head(spatial_features_2d),
        }
