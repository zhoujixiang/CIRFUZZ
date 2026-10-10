import torch
import torch.nn as nn

from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.ego_preserving_modules import \
    CooperativeFusionAdapter, EgoPreservingHead, infer_feature_dim
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter


class PointPillarCoopEgo(nn.Module):

    def __init__(self, args):
        super(PointPillarCoopEgo, self).__init__()

        self.max_cav = args['max_cav']
        self.feature_dim = infer_feature_dim(args)

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
        if args['compression'] > 0:
            self.compression = True
            self.naive_compressor = NaiveCompressor(self.feature_dim,
                                                    args['compression'])

        self.fusion_adapter = CooperativeFusionAdapter(args, self.feature_dim)
        self.protection_head = EgoPreservingHead(self.feature_dim,
                                                 args['anchor_number'])

        ego_args = args.get('ego_preserving', {})
        gate_bias_init = ego_args.get('gate_bias_init', None)
        if gate_bias_init is not None and \
                self.protection_head.gate_net[-1].bias is not None:
            nn.init.constant_(self.protection_head.gate_net[-1].bias,
                              gate_bias_init)

        if args['backbone_fix']:
            self.backbone_fix()

    def backbone_fix(self):
        for p in self.pillar_vfe.parameters():
            p.requires_grad = False

        for p in self.scatter.parameters():
            p.requires_grad = False

        for p in self.backbone.parameters():
            p.requires_grad = False

        if self.compression:
            for p in self.naive_compressor.parameters():
                p.requires_grad = False

        if self.shrink_flag:
            for p in self.shrink_conv.parameters():
                p.requires_grad = False

        self.protection_head.freeze_detection_heads()

    def _extract_spatial_features(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict['spatial_features_2d']
        if self.shrink_flag:
            spatial_features_2d = self.shrink_conv(spatial_features_2d)
        if self.compression:
            spatial_features_2d = self.naive_compressor(spatial_features_2d)

        return spatial_features_2d

    def forward(self, data_dict):
        spatial_features_2d = self._extract_spatial_features(data_dict)
        ego_feature, cooperative_feature = self.fusion_adapter(
            spatial_features_2d,
            data_dict
        )
        return self.protection_head(ego_feature, cooperative_feature)
