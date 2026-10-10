import torch
import torch.nn as nn

from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.fuse_utils import regroup
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.v2v_fuse import V2VNetFusion


class PointPillarV2VNetEgo(nn.Module):

    def __init__(self, args):
        super(PointPillarV2VNetEgo, self).__init__()

        self.max_cav = args['max_cav']
        self.feature_dim = args['v2vfusion']['in_channels']

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
            self.naive_compressor = NaiveCompressor(256, args['compression'])

        self.fusion_net = V2VNetFusion(args['v2vfusion'])
        self.gate_net = nn.Sequential(
            nn.Conv2d(self.feature_dim * 3,
                      self.feature_dim,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_dim, 1, kernel_size=1, bias=True)
        )
        ego_args = args.get('ego_preserving', {})
        gate_bias_init = ego_args.get('gate_bias_init', None)
        if gate_bias_init is not None and self.gate_net[-1].bias is not None:
            nn.init.constant_(self.gate_net[-1].bias, gate_bias_init)

        self.cls_head = nn.Conv2d(self.feature_dim,
                                  args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(self.feature_dim,
                                  7 * args['anchor_number'],
                                  kernel_size=1)

        self.ego_cls_head = nn.Conv2d(self.feature_dim,
                                      args['anchor_number'],
                                      kernel_size=1)
        self.ego_reg_head = nn.Conv2d(self.feature_dim,
                                      7 * args['anchor_number'],
                                      kernel_size=1)

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

        for p in self.cls_head.parameters():
            p.requires_grad = False

        for p in self.reg_head.parameters():
            p.requires_grad = False

        for p in self.ego_cls_head.parameters():
            p.requires_grad = False

        for p in self.ego_reg_head.parameters():
            p.requires_grad = False

    @staticmethod
    def unpad_prior_encoding(x, record_len):
        out = []
        for i in range(x.shape[0]):
            out.append(x[i, :record_len[i], :])
        return torch.cat(out, dim=0)

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        pairwise_t_matrix = data_dict['pairwise_t_matrix']
        prior_encoding = self.unpad_prior_encoding(data_dict['prior_encoding'],
                                                   record_len)

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

        regroup_feature, _ = regroup(spatial_features_2d,
                                     record_len,
                                     self.max_cav)
        ego_feature = regroup_feature[:, 0, ...]

        cooperative_feature = self.fusion_net(spatial_features_2d,
                                              record_len,
                                              pairwise_t_matrix,
                                              prior_encoding)

        cooperative_delta = cooperative_feature - ego_feature
        gate_input = torch.cat([ego_feature,
                                cooperative_feature,
                                cooperative_delta], dim=1)
        fusion_gate = torch.sigmoid(self.gate_net(gate_input))
        protected_feature = ego_feature + fusion_gate * cooperative_delta

        output_dict = {
            'psm': self.cls_head(protected_feature),
            'rm': self.reg_head(protected_feature),
            'ego_psm': self.ego_cls_head(ego_feature),
            'ego_rm': self.ego_reg_head(ego_feature),
            'fusion_gate': fusion_gate
        }
        return output_dict
