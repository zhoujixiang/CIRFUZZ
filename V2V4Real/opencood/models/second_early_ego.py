import numpy as np
import torch
import torch.nn as nn

from opencood.models.sub_modules.cia_ssd_utils import SSFA
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.height_compression import HeightCompression
from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.sparse_backbone_3d_second import \
    VoxelBackBone8x


class EgoPreservingHead(nn.Module):

    def __init__(self, feature_dim, anchor_number, gate_bias_init=0.0,
                 gate_floor=0.3):
        super(EgoPreservingHead, self).__init__()

        self.gate_floor = gate_floor

        self.gate_net = nn.Sequential(
            nn.Conv2d(feature_dim * 3, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 1, kernel_size=1, bias=True)
        )
        if gate_bias_init is not None and self.gate_net[-1].bias is not None:
            nn.init.constant_(self.gate_net[-1].bias, gate_bias_init)

        self.cls_head = nn.Conv2d(feature_dim,
                                  anchor_number,
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(feature_dim,
                                  7 * anchor_number,
                                  kernel_size=1)

        self.ego_cls_head = nn.Conv2d(feature_dim,
                                      anchor_number,
                                      kernel_size=1)
        self.ego_reg_head = nn.Conv2d(feature_dim,
                                      7 * anchor_number,
                                      kernel_size=1)

    def forward(self, ego_feature, cooperative_feature):
        cooperative_delta = cooperative_feature - ego_feature
        gate_input = torch.cat([ego_feature,
                                cooperative_feature,
                                cooperative_delta], dim=1)
        gate = torch.sigmoid(self.gate_net(gate_input))
        fusion_gate = self.gate_floor + (1.0 - self.gate_floor) * gate
        protected_feature = ego_feature + fusion_gate * cooperative_delta

        return {
            'psm': self.cls_head(protected_feature),
            'rm': self.reg_head(protected_feature),
            'ego_psm': self.ego_cls_head(ego_feature),
            'ego_rm': self.ego_reg_head(ego_feature),
            'fusion_gate': fusion_gate
        }


class SecondEarlyEgo(nn.Module):

    def __init__(self, args):
        super(SecondEarlyEgo, self).__init__()

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
        self.shrink_conv = DownsampleConv(args['shrink_header'])

        self.ego_encoder = None
        if 'ego_encoder' in args and args['ego_encoder']:
            ego_cfg = args['ego_encoder']
            self.ego_vfe = MeanVFE(ego_cfg['mean_vfe'],
                                   ego_cfg['mean_vfe']['num_point_features'])
            self.ego_spconv_block = VoxelBackBone8x(
                ego_cfg['spconv'],
                input_channels=ego_cfg['spconv']['num_features_in'],
                grid_size=grid_size,
            )
            self.ego_map_to_bev = HeightCompression(ego_cfg['map2bev'])
            self.ego_encoder = True

        ego_args = args.get('ego_preserving', {})
        feature_dim = ego_args.get('feature_dim', 128)
        gate_bias_init = ego_args.get('gate_bias_init', 0.0)
        gate_floor = ego_args.get('gate_floor', 0.3)
        self.protection_head = EgoPreservingHead(feature_dim,
                                                 args['anchor_num'],
                                                 gate_bias_init,
                                                 gate_floor)

    def _encode(self, lidar_dict, use_ego=False):
        voxel_features = lidar_dict['voxel_features']
        voxel_coords = lidar_dict['voxel_coords']
        voxel_num_points = lidar_dict['voxel_num_points']
        batch_size = int(voxel_coords[:, 0].max().item()) + 1

        batch_dict = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
            'batch_size': batch_size
        }

        if use_ego:
            batch_dict = self.ego_vfe(batch_dict)
            batch_dict = self.ego_spconv_block(batch_dict)
            batch_dict = self.ego_map_to_bev(batch_dict)
        else:
            batch_dict = self.vfe(batch_dict)
            batch_dict = self.spconv_block(batch_dict)
            batch_dict = self.map_to_bev(batch_dict)

        spatial_features_2d = self.ssfa(batch_dict['spatial_features'])
        spatial_features_2d = self.shrink_conv(spatial_features_2d)
        return spatial_features_2d

    def forward(self, data_dict):
        fused_lidar = data_dict['processed_lidar']
        ego_lidar = data_dict.get('ego_processed_lidar', fused_lidar)

        fused_feature = self._encode(fused_lidar, use_ego=False)
        ego_feature = self._encode(ego_lidar,
                                   use_ego=self.ego_encoder is not None)

        return self.protection_head(ego_feature, fused_feature)
