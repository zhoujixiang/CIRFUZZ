import torch.nn as nn

from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone
from opencood.models.sub_modules.ego_preserving_modules import \
    EgoPreservingHead
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter


class PointPillarIntermediateEgo(nn.Module):
    def __init__(self, args):
        super(PointPillarIntermediateEgo, self).__init__()

        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = AttBEVBackbone(args['base_bev_backbone'], 64)

        ego_args = args.get('ego_preserving', {})
        feature_dim = ego_args.get('feature_dim', 128 * 3)
        self.protection_head = EgoPreservingHead(feature_dim,
                                                 args['anchor_number'])

    def forward(self, data_dict):
        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len,
                      'return_ego_features': True}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        fused_feature = batch_dict['spatial_features_2d']
        ego_feature = batch_dict['ego_spatial_features_2d']

        return self.protection_head(ego_feature, fused_feature)
