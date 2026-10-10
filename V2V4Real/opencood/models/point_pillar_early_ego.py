






















import torch
import torch.nn as nn

from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter


class EgoPreservingHead(nn.Module):

    def __init__(self, feature_dim, anchor_number, gate_bias_init=-2.0,
                 gate_floor=0.0, ego_residual=False):
        super(EgoPreservingHead, self).__init__()

        self.gate_floor = gate_floor
        self.ego_residual = ego_residual

        self.gate_net = nn.Sequential(
            nn.Conv2d(feature_dim * 3, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 1, kernel_size=1, bias=True)
        )
        if ego_residual:
            self.ego_proj = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True),
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
        if self.ego_residual:
            cooperative_feature = cooperative_feature + \
                self.ego_proj(ego_feature)
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


class PointPillarEarlyEgo(nn.Module):

    def __init__(self, args):
        super(PointPillarEarlyEgo, self).__init__()

        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        self.compression = False
        if args.get('compression', 0) > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        ego_args = args.get('ego_preserving', {})
        feature_dim = ego_args.get('feature_dim', args.get('cls_head_dim', 256))
        gate_bias_init = ego_args.get('gate_bias_init', -2.0)
        gate_floor = ego_args.get('gate_floor', 0.0)
        ego_residual = ego_args.get('ego_residual', False)
        self.protection_head = EgoPreservingHead(feature_dim,
                                                 args['anchor_number'],
                                                 gate_bias_init,
                                                 gate_floor,
                                                 ego_residual)

        self.ego_encoder = None
        if 'ego_encoder' in args and args['ego_encoder']:
            ego_cfg = args['ego_encoder']
            self.ego_pillar_vfe = PillarVFE(
                ego_cfg.get('pillar_vfe', args['pillar_vfe']),
                num_point_features=4,
                voxel_size=args['voxel_size'],
                point_cloud_range=args['lidar_range'])
            self.ego_scatter = PointPillarScatter(
                ego_cfg.get('point_pillar_scatter', args['point_pillar_scatter']))
            self.ego_backbone = BaseBEVBackbone(
                ego_cfg['base_bev_backbone'], 64)
            ego_shrink_dim = ego_cfg.get('shrink_dim', feature_dim)
            ego_backbone_in_ch = int(
                sum(ego_cfg['base_bev_backbone']['num_upsample_filter']))
            self.ego_shrink = nn.Conv2d(
                ego_backbone_in_ch, ego_shrink_dim,
                kernel_size=3, stride=2, padding=1, bias=False)
            self.ego_encoder = True

    def _encode(self, lidar_dict):
        batch_dict = {
            'voxel_features': lidar_dict['voxel_features'],
            'voxel_coords': lidar_dict['voxel_coords'],
            'voxel_num_points': lidar_dict['voxel_num_points']
        }
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)
        return spatial_features_2d

    def _encode_ego(self, lidar_dict):
        batch_dict = {
            'voxel_features': lidar_dict['voxel_features'],
            'voxel_coords': lidar_dict['voxel_coords'],
            'voxel_num_points': lidar_dict['voxel_num_points']
        }
        batch_dict = self.ego_pillar_vfe(batch_dict)
        batch_dict = self.ego_scatter(batch_dict)
        batch_dict = self.ego_backbone(batch_dict)
        spatial_features_2d = batch_dict['spatial_features_2d']
        spatial_features_2d = self.ego_shrink(spatial_features_2d)
        return spatial_features_2d

    def forward(self, data_dict):
        fused_lidar = data_dict['processed_lidar']
        ego_lidar = data_dict.get('ego_processed_lidar', fused_lidar)

        fused_feature = self._encode(fused_lidar)
        if self.ego_encoder is not None:
            ego_feature = self._encode_ego(ego_lidar)
        else:
            ego_feature = self._encode(ego_lidar)

        return self.protection_head(ego_feature, fused_feature)
