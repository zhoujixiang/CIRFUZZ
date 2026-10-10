
import torch
import torch.nn as nn

from opencood.models.point_pillar_baseline_multiscale import \
    PointPillarBaselineMultiscale
from opencood.models.sub_modules.ego_preserving_modules import \
    EgoPreservingHead
from opencood.utils.transformation_utils_ext import normalize_pairwise_tfm


class PointPillarCoalignEgo(PointPillarBaselineMultiscale):
    def __init__(self, args):
        args = dict(args)
        if 'point_pillar_scatter' in args:
            scatter_cfg = dict(args['point_pillar_scatter'])
            if 'grid_size' not in scatter_cfg:
                voxel_size = args['voxel_size']
                lidar_range = args['lidar_range']
                scatter_cfg['grid_size'] = [
                    int(round((lidar_range[3] - lidar_range[0]) / voxel_size[0])),
                    int(round((lidar_range[4] - lidar_range[1]) / voxel_size[1])),
                    int(round((lidar_range[5] - lidar_range[2]) / voxel_size[2])),
                ]
            args['point_pillar_scatter'] = scatter_cfg

        super(PointPillarCoalignEgo, self).__init__(args)

        ego_args = args.get('ego_preserving', {})
        feature_dim = args['shrink_header']['dim'][-1] \
            if self.shrink_flag and 'shrink_header' in args else self.out_channel
        configured_dim = ego_args.get('feature_dim', feature_dim)
        if configured_dim != feature_dim:
            print('[CoAlignEgo] override ego_preserving.feature_dim=%s -> %s'
                  % (configured_dim, feature_dim))
        self.protection_head = EgoPreservingHead(feature_dim,
                                                 args['anchor_number'])

        gate_bias_init = ego_args.get('gate_bias_init', None)
        if gate_bias_init is not None and \
                self.protection_head.gate_net[-1].bias is not None:
            nn.init.constant_(self.protection_head.gate_net[-1].bias,
                              gate_bias_init)

        if args.get('backbone_fix', False):
            self.backbone_fix()

    def backbone_fix(self):
        super(PointPillarCoalignEgo, self).backbone_fix()

        if hasattr(self, 'dir_head'):
            for p in self.dir_head.parameters():
                p.requires_grad = False

        if hasattr(self, 'protection_head'):
            self.protection_head.freeze_detection_heads()

    @staticmethod
    def _select_ego_level_feature(feature_map, record_len):
        split_idx = torch.cumsum(record_len, dim=0)[:-1].cpu()
        split_feature = torch.tensor_split(feature_map, split_idx)
        ego_feature = torch.cat([item[0].unsqueeze(0) for item in
                                 split_feature], dim=0)
        return ego_feature

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        if torch.is_tensor(record_len):
            record_len = record_len.to(device=voxel_features.device,
                                       dtype=torch.long)
        else:
            record_len = torch.tensor(record_len,
                                      device=voxel_features.device,
                                      dtype=torch.long)

        spatial_features = {
            'voxel_features': voxel_features,
            'voxel_coords': voxel_coords,
            'voxel_num_points': voxel_num_points,
            'record_len': record_len
        }

        batch_dict = self.pillar_vfe(spatial_features)
        if batch_dict['pillar_features'].dim() == 1:
            batch_dict['pillar_features'] = batch_dict['pillar_features'].unsqueeze(0)
        if batch_dict['voxel_coords'].dim() == 1:
            batch_dict['voxel_coords'] = batch_dict['voxel_coords'].unsqueeze(0)
        if batch_dict['voxel_num_points'].dim() == 0:
            batch_dict['voxel_num_points'] = batch_dict['voxel_num_points'].unsqueeze(0)
        batch_dict = self.scatter(batch_dict)

        _, _, H0, W0 = batch_dict['spatial_features'].shape
        normalized_affine_matrix = normalize_pairwise_tfm(
            data_dict['pairwise_t_matrix'], H0, W0, self.voxel_size[0])

        spatial_features = batch_dict['spatial_features']
        if self.compression:
            spatial_features = self.naive_compressor(spatial_features)

        feature_list = self.backbone.get_multiscale_feature(spatial_features)

        fused_feature_list = []
        ego_feature_list = []
        for i, fuse_module in enumerate(self.fusion_net):
            fused_feature_list.append(
                fuse_module(feature_list[i], record_len, normalized_affine_matrix)
            )
            ego_feature_list.append(
                self._select_ego_level_feature(feature_list[i], record_len)
            )

        fused_feature = self.backbone.decode_multiscale_feature(
            fused_feature_list)
        ego_feature = self.backbone.decode_multiscale_feature(ego_feature_list)

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)
            ego_feature = self.shrink_conv(ego_feature)

        cooperative_delta = fused_feature - ego_feature
        gate_input = torch.cat([ego_feature,
                                fused_feature,
                                cooperative_delta], dim=1)
        fusion_gate = torch.sigmoid(self.protection_head.gate_net(gate_input))
        protected_feature = ego_feature + fusion_gate * cooperative_delta

        output_dict = {
            'cls_preds': self.protection_head.cls_head(protected_feature),
            'reg_preds': self.protection_head.reg_head(protected_feature),
            'psm': self.protection_head.cls_head(protected_feature),
            'rm': self.protection_head.reg_head(protected_feature),
            'ego_psm': self.protection_head.ego_cls_head(ego_feature),
            'ego_rm': self.protection_head.ego_reg_head(ego_feature),
            'fusion_gate': fusion_gate
        }

        if self.use_dir:
            output_dict.update({
                'dir_preds': self.dir_head(protected_feature),
                'dm': self.dir_head(protected_feature)
            })

        return output_dict
