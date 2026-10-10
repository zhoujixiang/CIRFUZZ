import torch
import torch.nn as nn
import numpy as np

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.fuse_utils import regroup
from opencood.models.sub_modules.deformable_detr import DeformableDETR
from opencood.utils.transformation_addon import align_features
from opencood.models.sub_modules.communication import CompressFuse


class PointPillarCost(nn.Module):
    def __init__(self, args):
        super(PointPillarCost, self).__init__()

        self.max_cav = args['max_cav']
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

        self.temporal_compression = True if 'compression_module' in args else False
        if 'compression_module' in args:
            self.temporal_compressor = CompressFuse(args['compression_module'], args['max_cav'])

        self.fusion_net = DeformableDETR(args['deformable_transformer'])

        self.cls_head = nn.Conv2d(128 * 2, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                  kernel_size=1)

        self.solo_head = True if 'solo_head' in args else False
        if self.solo_head:
            self.cls_head1 = nn.Conv2d(128 * 2, args['anchor_number'],
                                       kernel_size=1)
            self.reg_head1 = nn.Conv2d(128 * 2, 7 * args['anchor_number'],
                                       kernel_size=1)

        self.num_levels = len(args['base_bev_backbone']['layer_nums'])

        if args['backbone_fix']:
            self.backbone_fix()

        self.prev_backbone, self.prev_map = None, None
        self.pre_fused_feature, self.pre_ego_pose = None, None
        self.prev_backbone2, self.prev_map2 = None, None
        self.pre_fused_feature2, self.pre_ego_pose2 = None, None
        self.communication_rates = []
        self.memory_frames = 0
        if 'align_ratio' in args:
            self.align_ratio = args['align_ratio']
        else:
            self.align_ratio = args['feature_stride'] * args['voxel_size'][0]
        self.use_temporal = args['deformable_transformer']['multi_time']

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

    def forward_ori(self, data_dict, pre_fused_feature=None, prev_pose=None,
                    prev_backbone=None, prev_map=None, inference=False):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        prior_encoding = data_dict['prior_encoding']

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

        regroup_feature, mask = regroup(spatial_features_2d,
                                        record_len,
                                        self.max_cav)

        if self.solo_head:
            pre_psm = self.cls_head1(regroup_feature.flatten(0, 1))
            pre_rm = self.reg_head1(regroup_feature.flatten(0, 1))
        else:
            pre_psm = self.cls_head(regroup_feature.flatten(0, 1))
            pre_rm = self.reg_head(regroup_feature.flatten(0, 1))
        B, L, C, H, W = regroup_feature.shape

        has_temporal_data = (
            self.temporal_compression and
            'prev_exist' in data_dict and
            'ego_pose' in data_dict and
            data_dict.get('prev_exist') is not None
        )

        if self.temporal_compression:
            if not has_temporal_data:
                pass
            else:
                backbone_feat = regroup_feature
                confidence_map = pre_psm.sigmoid().max(dim=-3)[0].reshape(B, L, 1, H, W)
                confidence_map = confidence_map * mask.reshape(B, L, 1, 1, 1).expand(B, L, 1, H, W)

                if prev_pose is None or not data_dict['prev_exist'][0].item():
                    self.memory_frames = 0
                    backbone_feat, communication_rate = self.temporal_compressor(
                        backbone_feat, confidence_map,
                        None, None, reset=True
                    )
                else:
                    ego_poses = torch.stack([prev_pose, data_dict['ego_pose']], dim=1)

                    if len(ego_poses.shape) == 3:
                        ego_poses = ego_poses.unsqueeze(1).repeat(1, L, 1, 1).flatten(0, 1)
                    elif len(ego_poses.shape) == 4:
                        ego_poses = ego_poses.unsqueeze(1).repeat(1, L, 1, 1, 1).flatten(0, 1)

                    align_backbone_feats = torch.stack(
                        [prev_backbone, backbone_feat], dim=2).flatten(0, 1)
                    align_confidence = torch.stack(
                        [prev_map, confidence_map], dim=2).flatten(0, 1)

                    align_backbone_feats = align_features(
                        align_backbone_feats, ego_poses, self.align_ratio).reshape(
                        B, L, 2, -1, H, W)
                    align_confidence = align_features(
                        align_confidence, ego_poses, self.align_ratio).reshape(
                        B, L, 2, 1, H, W)

                    backbone_feat, communication_rate = self.temporal_compressor(
                        align_backbone_feats[:, :, 1], align_confidence[:, :, 1],
                        align_backbone_feats[:, :, 0], align_confidence[:, :, 0],
                        reset=False
                    )

                prev_backbone, prev_map = backbone_feat, confidence_map
                self.memory_frames += 1
                self.communication_rates.append(communication_rate)
                regroup_feature = backbone_feat

        if self.use_temporal and 'ego_pose' in data_dict:
            B = regroup_feature.shape[0]
            if pre_fused_feature is None:
                add_mask = torch.zeros_like(mask[:, :1])
                add_feat = torch.zeros_like(regroup_feature[:, :1])
                add_prior = torch.tensor([0, 0, 0]).to(prior_encoding).repeat(B, 1, 1)
            else:
                add_mask = torch.ones_like(mask[:, :1])
                stack_feat = torch.stack([pre_fused_feature, regroup_feature[:, 0]], dim=1)
                align_poses = torch.stack([prev_pose, data_dict['ego_pose']], dim=1)
                add_feat = align_features(stack_feat, align_poses, self.align_ratio)[:, 0:1]
                add_prior = torch.tensor([0, 100, 2]).to(prior_encoding).repeat(B, 1, 1)

            regroup_feature = torch.cat([regroup_feature, add_feat], dim=1)
            mask = torch.cat([mask, add_mask], dim=1)
            prior_encoding = torch.cat([prior_encoding, add_prior], dim=1)

        fused_feature = self.fusion_net(regroup_feature, mask, prior_encoding)

        if has_temporal_data:
            return fused_feature, pre_psm, pre_rm, prev_backbone, prev_map
        else:
            return fused_feature, pre_psm, pre_rm, None, None

    def forward(self, data_dicts, inference=False):
        if isinstance(data_dicts, dict):
            data_dicts = [data_dicts]

        if inference:
            pre_fused_feature, pre_psm, pre_rm, prev_backbone, prev_map = self.forward_ori(
                data_dicts[-1],
                self.pre_fused_feature,
                self.pre_ego_pose,
                self.prev_backbone,
                self.prev_map,
            )
            pre_ego_pose = data_dicts[-1]['ego_pose']

            self.pre_fused_feature = pre_fused_feature
            self.pre_ego_pose = pre_ego_pose
            self.prev_backbone = prev_backbone
            self.prev_map = prev_map
            psm = self.cls_head(pre_fused_feature)
            rm = self.reg_head(pre_fused_feature)

            output_dict = {
                'psm': psm,
                'rm': rm,
                'psm_single': pre_psm,
                'rm_single': pre_rm,
            }
            return output_dict

        pre_fused_feature = None
        pre_ego_pose = None
        prev_backbone = None
        prev_map = None
        psms, rms = [], []

        start_idx = 0
        if self.use_temporal:
            T = len(data_dicts)
            with torch.no_grad():
                for i in range(start_idx, T - 1):
                    pre_fused_feature, pre_psm, pre_rm, prev_backbone, prev_map = self.forward_ori(
                        data_dicts[i],
                        pre_fused_feature,
                        pre_ego_pose,
                        prev_backbone,
                        prev_map
                    )
                    pre_ego_pose = data_dicts[i]['ego_pose']

        pre_fused_feature, pre_psm, pre_rm, prev_backbone, prev_map = self.forward_ori(
            data_dicts[-1],
            pre_fused_feature,
            pre_ego_pose,
            prev_backbone,
            prev_map,
        )

        psm = self.cls_head(pre_fused_feature)
        rm = self.reg_head(pre_fused_feature)

        output_dict = {
            'psm': psm,
            'rm': rm,
            'psm_single': pre_psm,
            'rm_single': pre_rm,
        }

        return output_dict
